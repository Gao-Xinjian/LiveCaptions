"""
Shared machinery of the streaming policies: the audio buffer, engine runs with the filters, final text and lines,
cuts, and the endpoint rules (PLAN.md §6.1 C). Subclasses decide when to run and what becomes final.
"""
import threading
from collections import Counter

import numpy as np

from livecaptions import SAMPLE_RATE
from livecaptions.filters import hallucination as hf
from livecaptions.filters.postprocess import to_simplified
from livecaptions.segmentation.base import Line, PolicyOutput, StreamingPolicy
from livecaptions.segmentation.endpointer import Endpointer
from livecaptions.segmentation.prompt import PromptBuilder
from livecaptions.segmentation.units import SENTENCE_END, is_cjk, kept_prefix, segments2units, units2text


class BufferedPolicy(StreamingPolicy):
    def __init__(self, engine, cfg, terms_text=''):
        self.engine = engine
        self.cfg = cfg  # read on every call, so live settings apply at once
        self.ep = Endpointer(cfg.vad, cfg.stream)
        self.prompt = PromptBuilder(engine)
        self._terms_lock = threading.Lock()
        self._terms_text = terms_text
        self._terms_dirty = True
        # Audio buffer: the audio from buf_start (absolute sample) on.
        self.buf = np.zeros(int((cfg.stream.max_buffer_s + 10) * SAMPLE_RATE), np.float32)
        self.buf_len = 0
        self.buf_start = 0
        self.total = 0            # samples received in this session
        self.last_run_end = 0     # absolute sample the last run reached
        # Run scheduling: the recent run time and the step in use (longer than set when adaptation kicked in).
        self.run_s_avg = None
        self.step_s = cfg.stream.step_s
        self.step_base_s = cfg.stream.step_s
        # Text state.
        self.kept = []            # final units whose audio is still in the buffer
        self.history = []         # unfinished text of recent runs; the last one is shown in grey
        self.context = ''         # final text whose audio was dropped: the prompt's recent text
        self.line = ''            # final text of the current caption line
        self.line_t0 = 0.0
        self.line_t1 = 0.0
        self.shown_tentative = None
        # Language lock.
        self.lang_votes = Counter()
        self.locked_lang = None

    def set_terms(self, text):
        with self._terms_lock:
            self._terms_text = text
            self._terms_dirty = True

    def get_task(self):
        rc = self.cfg.recog
        return 'translate' if rc.translate and self.engine.caps.can_translate else 'transcribe'

    def get_language(self):
        lang = self.cfg.recog.language
        if lang and lang != 'auto':
            return lang
        return self.locked_lang

    def get_buffer_s(self):
        return self.buf_len / SAMPLE_RATE

    def get_new_s(self):
        """Return the seconds of audio the last run has not seen."""
        return (self.buf_start + self.buf_len - self.last_run_end) / SAMPLE_RATE

    def get_max_buffer_s(self):
        """Return T3; a fixed encoder window (50 frames per second) also caps it, or audio would be cut off."""
        rc = self.cfg.recog
        limit = self.cfg.stream.max_buffer_s
        if rc.audio_ctx_mode == 'fixed' and self.engine.name == 'whispercpp':
            limit = min(limit, rc.audio_ctx / 50 - 1.0)
        return max(2.0, limit)

    def get_step_s(self, base):
        """
        Return how much new audio to wait for before the next run: base, or longer with adaptation on when recent runs
        took more than the allowed share of it; then each run covers more audio and inference keeps up (PLAN.md §11.6).
        """
        sc = self.cfg.stream
        step = base
        if sc.adapt_on and self.run_s_avg is not None:
            need = self.run_s_avg / (max(30, sc.adapt_load) / 100)
            step = min(max(base, need), max(base, sc.adapt_max_s))
        self.step_s, self.step_base_s = step, base
        return step

    def has_text(self):
        """Return whether the current utterance has recognised text (final or unfinished)."""
        return bool(self.kept or (self.history and self.history[-1]) or self.line)

    # Buffer handling.

    def _append(self, chunk):
        if self.buf_len + len(chunk) > len(self.buf):
            self.buf = np.resize(self.buf, self.buf_len + len(chunk) + 10 * SAMPLE_RATE)
        self.buf[self.buf_len:self.buf_len + len(chunk)] = chunk
        self.buf_len += len(chunk)
        self.total += len(chunk)

    def _trim_to(self, t_abs, n_units, out):
        """Drop the audio before absolute sample t_abs; the first n_units kept units were in it and become context."""
        n = int(np.clip(t_abs - self.buf_start, 0, self.buf_len))
        if n:
            self.buf[:self.buf_len - n] = self.buf[n:self.buf_len]
            self.buf_len -= n
            self.buf_start += n
        n_units = min(n_units, len(self.kept))
        self.context += ''.join(u.text for u in self.kept[:n_units])
        del self.kept[:n_units]
        limit = max(self.cfg.terms.context_chars, 1)
        if len(self.context) > 2 * limit:
            self.context = self.context[-limit:]

    def _keep_preroll(self, out):
        """Only silence so far: keep a little, so the first word isn't cut off."""
        keep = int((self.cfg.stream.preroll_s + self.cfg.vad.min_speech_ms / 1000) * SAMPLE_RATE)  # + unconfirmed speech
        self._trim_to(self.buf_start + self.buf_len - keep, 0, out)
        self.last_run_end = max(self.last_run_end, self.buf_start)

    def _drop_noise(self, units, had_text, out):
        """T1: sound in which nothing is recognised is dropped once it gets long."""
        sc = self.cfg.stream
        if not units and not had_text and self.get_buffer_s() >= sc.noise_reset_s:
            keep = int((sc.preroll_s + sc.step_s) * SAMPLE_RATE)
            self._trim_to(self.buf_start + self.buf_len - keep, 0, out)

    def _check_endpoint(self, idle_s):
        """Return (open, endpoint): an utterance is open, and a pause (T2) or idle device ends it."""
        open_ = self.ep.utterance or self.has_text()
        pause = self.ep.get_required_pause(self.get_buffer_s())
        endpoint = open_ and (self.ep.get_silence_s() >= pause or idle_s >= self.cfg.stream.idle_end_s)
        return open_, endpoint

    def _get_speech_end(self):
        """Return how many buffer samples to transcribe at an endpoint: trailing silence beyond the pad is left out."""
        pad = int(self.cfg.stream.preroll_s * SAMPLE_RATE)
        return max(1, self.buf_len - max(0, self.ep.silence - pad))

    # Text handling.

    def _end_line(self, out):
        if not self.line:
            return
        out.lines.append(Line(self.line, self.line_t0, self.line_t1))
        out.committed += '\n'
        self.line = ''

    def _commit(self, units, out):
        """Make units final: add them to the line, which ends at a sentence end or when it gets long."""
        sc = self.cfg.stream
        now_s = self.total / SAMPLE_RATE
        for u in units:
            text = u.text.strip() if not self.line else u.text
            if not text:
                continue
            if not self.line:
                self.line_t0 = u.t0
            self.line += text
            self.line_t1 = max(self.line_t1, u.t1) if self.line != text else u.t1
            out.committed += text
            out.latencies.append(max(0.0, now_s - u.t1))
            self.kept.append(u)
            if (sc.break_on_sentence and text[-1] in SENTENCE_END) or len(self.line) >= sc.max_line_chars:
                self._end_line(out)

    def _finish_all(self, out, commit=True):
        """Endpoint: the unfinished text becomes final (or is dropped), the line ends, the buffer is cleared."""
        if self.history and commit:
            self._commit(self.history[-1], out)
        self.history = []
        self._end_line(out)
        self._trim_to(self.buf_start + self.buf_len, len(self.kept), out)
        self.ep.reset_utterance()

    def _drop_terms_only(self, units, skip):
        """A new line made only of course terms is how Whisper misreads noise when they're in its prompt."""
        if not self.line and self.cfg.filters.terms_only and hf.is_terms_only(units[skip:], self.prompt.term_keys):
            return units[:skip]
        return units

    def _run_new(self, out, upto=None):
        """Run the engine and return (all units, units after the kept text, segment ends, skip)."""
        units, seg_ends = self._run(out, upto)
        skip = kept_prefix(units, self.kept)
        units = self._drop_terms_only(units, skip)
        return units, units[skip:], seg_ends, skip

    def _cut_at_quietest(self, out):
        """T3: the buffer is too long; cut it at its quietest recent point and make the text before it final."""
        end = self.buf_start + self.buf_len
        q = self.ep.find_quietest(max(self.buf_start, end - int(self.cfg.stream.smart_cut_s * SAMPLE_RATE)), end)
        units, _ = self._run(out, upto=q - self.buf_start)
        skip = kept_prefix(units, self.kept)
        if skip >= len(self.kept):  # all final text is before q: what else was heard there is final too
            self._commit(self._drop_terms_only(units, skip)[skip:], out)
            self.history = []
            self._trim_to(q, len(self.kept), out)
        else:  # final text reaches past q: it stays with its audio, the next run skips it
            self._trim_to(q, skip, out)
        self.last_run_end = self.buf_start  # the rest is new to the next run
        self.ep.speech_since_run = True

    # Running the engine.

    def _run(self, out, upto=None):
        """Transcribe buf[:upto] and return (units after filters, absolute end time of each segment)."""
        with self._terms_lock:
            if self._terms_dirty:
                self.prompt.set_terms(self._terms_text, self.cfg.terms, self.get_task() == 'translate')
                self._terms_dirty = False
        n = self.buf_len if upto is None else int(np.clip(upto, 0, self.buf_len))
        audio = self.buf[:n]
        offset = self.buf_start / SAMPLE_RATE
        task = self.get_task()
        res = self.engine.transcribe(audio, self.get_language(), task, self.prompt.build(self.context, self.cfg.terms),
                                     self.cfg.recog, self._get_hotwords())
        out.runs += 1
        out.busy_s += res.seconds
        out.audio_s += n / SAMPLE_RATE
        self.run_s_avg = res.seconds if self.run_s_avg is None else 0.7 * self.run_s_avg + 0.3 * res.seconds
        out.language = res.language
        self.ep.mark_run()
        self.last_run_end = self.buf_start + self.buf_len  # audio after a shortened run is silence or cut off
        fc = self.cfg.filters
        segs = [s for s in res.segments if not hf.is_unlikely_speech(s, fc)]
        units = segments2units(segs, offset)
        seg_ends = [offset + s.t1 for s in segs]
        if hf.is_blacklisted(units2text(units), fc):
            units = []
        lang = res.language or self.get_language() or ''
        if lang == 'zh' and task == 'transcribe':
            if fc.kana_in_zh:
                units = hf.drop_kana(units)
            if self.cfg.recog.to_simplified:
                for u in units:
                    if any(is_cjk(c) for c in u.text):
                        u.text, u.key = to_simplified(u.text), to_simplified(u.key)
        if fc.repetition_on:
            units = hf.cut_repetitions(units, fc.repetition_max)
        if hf.is_too_dense(units, n / SAMPLE_RATE, fc):
            units = []
        self._vote_language(res.language, bool(units))
        return units, seg_ends

    def _get_hotwords(self):
        if not (self.cfg.recog.hotwords and self.cfg.terms.use_terms and self.engine.caps.hotwords):
            return ''
        with self._terms_lock:
            return ' '.join(self._terms_text.split())

    def _vote_language(self, lang, has_text):
        rc = self.cfg.recog
        if rc.language != 'auto' or rc.lang_lock_runs <= 0 or self.locked_lang or not lang or not has_text:
            return
        self.lang_votes[lang] += 1
        if sum(self.lang_votes.values()) >= rc.lang_lock_runs:
            self.locked_lang = self.lang_votes.most_common(1)[0][0]

    def _update_tentative(self, out):
        grey = ''.join(u.text for u in self.history[-1]) if self.history else ''
        if not self.line:
            grey = grey.strip()
        if grey != self.shown_tentative or out.committed:
            out.tentative = grey
            self.shown_tentative = grey

    # Entry points shared by the policies.

    def _feed(self, chunk, vres):
        if len(chunk):
            self._append(chunk)
        self.ep.update(vres)

    def _endpoint(self, out, upto=None):
        """T2: transcribe what was not heard yet, then make everything final and clear the buffer."""
        if self.ep.speech_since_run and self.buf_len:
            units, new, seg_ends, skip = self._run_new(out, upto)
            self._on_endpoint_run(units, new, seg_ends, skip, out)
        self._finish_all(out)

    def _on_endpoint_run(self, units, new, seg_ends, skip, out):
        """What the last run before an endpoint does with its text (default: it all becomes the unfinished tail)."""
        self.history = [new]

    def flush(self, commit=True):
        out = PolicyOutput()
        if commit and self.ep.speech_since_run and self.buf_len:
            units, new, seg_ends, skip = self._run_new(out)
            self._on_endpoint_run(units, new, seg_ends, skip, out)
        self._finish_all(out, commit=commit)
        self._update_tentative(out)
        return out
