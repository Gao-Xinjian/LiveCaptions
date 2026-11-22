"""Endpointer, energy VAD and LocalAgreement with a scripted engine (no model needed)."""
import numpy as np

from livecaptions import SAMPLE_RATE
from livecaptions.config.schema import Config
from livecaptions.engines.base import ASREngine, EngineCaps, Piece, Segment, TranscribeResult
from livecaptions.segmentation import create_policy
from livecaptions.segmentation.endpointer import Endpointer
from livecaptions.segmentation.local_agreement import LocalAgreementPolicy
from livecaptions.segmentation.sliding_window import SlidingWindowPolicy
from livecaptions.segmentation.vad_chunk import VadChunkPolicy
from livecaptions.vad.base import NoVAD, VadResult
from livecaptions.vad.energy import EnergyVAD


def make_vres(start, probs, hop=320, rms=0.1):
    probs = np.asarray(probs, np.float32)
    return VadResult(start, hop, probs, np.full(len(probs), rms, np.float32))


def test_endpointer_states():
    cfg = Config()
    ep = Endpointer(cfg.vad, cfg.stream)
    ep.update(make_vres(0, [0.9] * 5))           # 100 ms: not yet speech (min 150 ms)
    assert not ep.utterance
    ep.update(make_vres(1600, [0.9] * 5))        # 200 ms in a row
    assert ep.utterance and ep.speech_since_run
    ep.update(make_vres(3200, [0.4] * 10))       # between the thresholds: still speaking
    assert ep.get_silence_s() == 0.0
    ep.update(make_vres(6400, [0.1] * 50))       # 1 s of silence
    assert abs(ep.get_silence_s() - 1.0) < 1e-6


def test_adaptive_pause():
    cfg = Config()
    cfg.stream.adaptive_pause = True
    ep = Endpointer(cfg.vad, cfg.stream)
    assert ep.get_required_pause(0) == cfg.stream.pause_s
    assert abs(ep.get_required_pause(100) - cfg.stream.pause_min_s) < 1e-9


def test_quietest_point():
    cfg = Config()
    ep = Endpointer(cfg.vad, cfg.stream)
    rms = np.full(50, 0.2, np.float32)
    rms[30] = 0.001
    ep.update(VadResult(0, 320, np.ones(50, np.float32), rms))
    q = ep.find_quietest(0, 50 * 320)
    assert 29 * 320 <= q <= 32 * 320


def test_energy_vad_tracks_noise_floor():
    cfg = Config()
    vad = EnergyVAD(cfg.vad)
    rng = np.random.default_rng(0)
    noise = (rng.standard_normal(SAMPLE_RATE * 5) * 0.001).astype(np.float32)
    t = np.arange(SAMPLE_RATE) / SAMPLE_RATE
    tone = (0.1 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    quiet = vad.process(noise)
    loud = vad.process(tone)
    assert quiet.probs[-50:].mean() < 0.2
    assert loud.probs[10:].mean() > 0.8


class ScriptedEngine(ASREngine):
    """Returns the words spoken so far: a stand-in for Whisper with perfect hearing."""
    name = 'scripted'

    def __init__(self, words, words_per_s=2.0):
        self.words = words
        self.rate = words_per_s
        self.caps = EngineCaps()
        self.policy = None
        self.calls = 0
        self.lengths = []  # seconds of audio per run

    def tokenize(self, text):
        return list(range(len(text.split())))

    def transcribe(self, audio, language, task, prompt_tokens, recog_cfg, hotwords=''):
        self.calls += 1
        self.lengths.append(len(audio) / SAMPLE_RATE)
        start = self.policy.buf_start / SAMPLE_RATE
        end = start + len(audio) / SAMPLE_RATE
        pieces = []
        for i, w in enumerate(self.words):
            t0, t1 = i / self.rate, (i + 1) / self.rate
            if start <= (t0 + t1) / 2 < end:  # a word is heard if most of it is in the buffer
                pieces.append(Piece((' ' + w).encode(), max(0.0, t0 - start), min(end, t1) - start))
        segs = [Segment(pieces[0].t0, pieces[-1].t1, pieces)] if pieces else []
        return TranscribeResult(segs, 'en', 0.01)


def feed(policy, seconds, speech, chunk_s=0.1, tentative=None):
    out_text = ''
    vad = policy._test_vad
    n = int(chunk_s * SAMPLE_RATE)
    for _ in range(int(round(seconds / chunk_s))):
        chunk = np.full(n, 0.1 if speech else 0.0, np.float32)
        vres = vad.process(chunk)
        if not speech:
            vres.probs[:] = 0.0
        out = policy.process(chunk, vres, 0.0)
        out_text += out.committed
        if tentative is not None and out.tentative:
            tentative.append(out.tentative)
    return out_text


def make_policy(words, policy='local_agreement', **stream):
    cfg = Config()
    cfg.vad.method = 'off'
    cfg.stream.policy = policy
    for k, v in stream.items():
        setattr(cfg.stream, k, v)
    eng = ScriptedEngine(words)
    pol = create_policy(eng, cfg)
    pol._test_vad = NoVAD()  # one VAD per policy, so hop positions stay absolute
    eng.policy = pol
    return pol, eng


def test_local_agreement_commits_and_pause_finalises():
    words = [f'w{i}' for i in range(10)]
    pol, eng = make_policy(words)
    text = feed(pol, 5.0, speech=True)
    assert text.split() == words[:len(text.split())] and len(text.split()) >= 6  # agreed words become final
    text += feed(pol, 1.5, speech=False)                                          # T2 pause: the rest is final
    assert text.split() == words
    assert text.endswith('\n') and not pol.kept and not pol.history
    assert pol.buf_len <= int((pol.cfg.stream.preroll_s + 0.2) * SAMPLE_RATE)  # only the pre-roll is kept


def test_local_agreement_no_runs_in_silence():
    pol, eng = make_policy(['a'])
    feed(pol, 5.0, speech=False)
    assert eng.calls == 0
    assert pol.buf_len <= int((pol.cfg.stream.preroll_s + 0.2) * SAMPLE_RATE)  # only the pre-roll is kept


def test_local_agreement_max_buffer_cut():
    words = [f'w{i}' for i in range(40)]
    pol, eng = make_policy(words, max_buffer_s=6.0, trim_after_s=28.0)
    text = feed(pol, 15.0, speech=True)
    assert pol.get_buffer_s() <= 6.5
    assert text.split() == words[:len(text.split())]   # nothing lost or repeated across cuts


def test_flush_commits_tentative():
    pol, eng = make_policy([f'w{i}' for i in range(4)])
    text = feed(pol, 2.0, speech=True)
    text += pol.flush(commit=True).committed
    assert text.split() == ['w0', 'w1', 'w2', 'w3']


def test_create_policy_by_name():
    for name, cls in (('local_agreement', LocalAgreementPolicy), ('vad_chunk', VadChunkPolicy),
                      ('sliding_window', SlidingWindowPolicy)):
        assert isinstance(make_policy(['a'], policy=name)[0], cls)


def test_vad_chunk_runs_once_per_sentence():
    words = [f'w{i}' for i in range(8)]
    pol, eng = make_policy(words, policy='vad_chunk')
    text = feed(pol, 4.0, speech=True)
    assert text == '' and eng.calls == 0          # nothing runs while the sentence goes on
    text += feed(pol, 1.5, speech=False)
    assert eng.calls == 1 and text.split() == words and text.endswith('\n')
    # The run leaves out the trailing silence beyond the pad (0.3 s).
    assert eng.lengths[0] <= 4.0 + pol.cfg.stream.preroll_s + 0.15


def test_vad_chunk_preview_is_grey_until_the_pause():
    words = [f'w{i}' for i in range(9)]
    pol, eng = make_policy(words, policy='vad_chunk', preview_on=True, preview_s=1.0)
    grey = []
    text = feed(pol, 4.5, speech=True, tentative=grey)
    assert text == '' and eng.calls == 4 and grey and grey[-1].split()[0] == 'w0'
    text += feed(pol, 1.5, speech=False)
    assert text.split() == words and eng.calls == 5   # speech after the last preview: one final run


def test_vad_chunk_reuses_a_complete_preview():
    words = [f'w{i}' for i in range(8)]
    pol, eng = make_policy(words, policy='vad_chunk', preview_on=True, preview_s=1.0)
    feed(pol, 4.0, speech=True)
    calls = eng.calls
    text = feed(pol, 1.5, speech=False)
    assert text.split() == words and eng.calls == calls  # the last preview heard everything: no extra run


def test_vad_chunk_long_sentence_is_cut():
    words = [f'w{i}' for i in range(30)]  # 15 s at 2 words/s
    pol, eng = make_policy(words, policy='vad_chunk', max_buffer_s=6.0)
    text = feed(pol, 15.0, speech=True)
    assert pol.get_buffer_s() <= 6.5 and text
    text += feed(pol, 1.5, speech=False)
    assert text.split() == words                  # nothing lost or repeated across the cuts


def test_sliding_window_commits_full_windows():
    words = [f'w{i}' for i in range(30)]
    pol, eng = make_policy(words, policy='sliding_window', window_s=4.0, keep_s=1.0, step_s=1.0)
    text = feed(pol, 15.0, speech=True)
    assert pol.get_buffer_s() <= 4.5
    n = len(text.split())
    assert n >= 20 and text.split() == words[:n]   # overlap text is not repeated
    text += feed(pol, 1.5, speech=False)
    assert text.split() == words
