"""End-to-end: a WAV through the real pipeline and model. Runs when LC_TEST_MODEL points to a model file."""
import os
import wave

import numpy as np
import pytest

from livecaptions import SAMPLE_RATE, native
from livecaptions.audio.capture import WavFileSource, read_wav
from livecaptions.config.schema import Config
from livecaptions.pipeline import EngineHolder, Pipeline

MODEL = os.environ.get('LC_TEST_MODEL', '')
WAV = os.environ.get('LC_TEST_WAV', '')


def run_pipeline(cfg, samples):
    events = []
    pipe = Pipeline(cfg, EngineHolder(), lambda k, p: events.append((k, p)), source=WavFileSource(samples, speed=0),
                    save=False)
    pipe.start()
    pipe.join(600)
    return events, pipe


@pytest.mark.skipif(not (MODEL and native.is_available()), reason='set LC_TEST_MODEL to a ggml model')
def test_noise_gives_no_text():
    cfg = Config()
    cfg.model.path = MODEL
    cfg.vad.method = 'energy'
    rng = np.random.default_rng(1)
    noise = (rng.standard_normal(SAMPLE_RATE * 20) * 0.02).astype(np.float32)
    events, _ = run_pipeline(cfg, noise)
    text = ''.join(p for k, p in events if k == 'commit').strip()
    assert not any(k == 'error' for k, _ in events)
    assert len(text) < 10, text


@pytest.mark.skipif(not (MODEL and WAV and native.is_available()), reason='set LC_TEST_MODEL and LC_TEST_WAV')
def test_speech_gives_text():
    cfg = Config()
    cfg.model.path = MODEL
    cfg.vad.method = 'energy'
    cfg.recog.device = os.environ.get('LC_TEST_DEVICE', 'gpu')
    events, pipe = run_pipeline(cfg, read_wav(WAV))
    text = ''.join(p for k, p in events if k == 'commit')
    assert not any(k == 'error' for k, _ in events), events
    assert len(text.split()) >= 3 and pipe.runs > 0


def test_read_wav_resamples(tmp_path):
    path = str(tmp_path / 'a.wav')
    with wave.open(path, 'wb') as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(48000)
        w.writeframes((np.ones((48000, 2)) * 1000).astype('<i2').tobytes())
    x = read_wav(path)
    assert len(x) == SAMPLE_RATE and abs(x.mean() - 1000 / 32768) < 1e-3


FW_MODEL = os.environ.get('LC_TEST_FW_MODEL', '')


@pytest.mark.parametrize('policy', ['vad_chunk', 'sliding_window'])
@pytest.mark.skipif(not (MODEL and WAV and native.is_available()), reason='set LC_TEST_MODEL and LC_TEST_WAV')
def test_other_policies_give_text(policy):
    cfg = Config()
    cfg.model.path = MODEL
    cfg.vad.method = 'energy'
    cfg.stream.policy = policy
    cfg.recog.device = os.environ.get('LC_TEST_DEVICE', 'gpu')
    events, pipe = run_pipeline(cfg, read_wav(WAV))
    text = ''.join(p for k, p in events if k == 'commit')
    assert not any(k == 'error' for k, _ in events), events
    assert len(text.split()) >= 3


@pytest.mark.skipif(not (FW_MODEL and WAV), reason='set LC_TEST_FW_MODEL (a CTranslate2 model folder) and LC_TEST_WAV')
def test_faster_whisper_gives_text():
    cfg = Config()
    cfg.model.backend = 'fasterwhisper'
    cfg.model.path = FW_MODEL
    cfg.vad.method = 'energy'
    cfg.recog.device = 'cpu'
    events, pipe = run_pipeline(cfg, read_wav(WAV))
    text = ''.join(p for k, p in events if k == 'commit')
    assert not any(k == 'error' for k, _ in events), events
    assert len(text.split()) >= 3
