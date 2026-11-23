"""Automatic gain, TEN VAD, the GitHub download check, and capture smoke tests (macOS tap, microphone)."""
import os
import subprocess
import sys
import tempfile
import time
import wave

import numpy as np
import pytest

from livecaptions import SAMPLE_RATE, native
from livecaptions.config.schema import Config
from livecaptions.models import manager
from livecaptions.models.registry import find_entry
from livecaptions.vad.agc import AutoGain
from livecaptions.vad.ten import find_ten_library, get_ten_lib_name

TEN_LIB = os.environ.get('LC_TEST_TEN_VAD', '')  # path to a TEN VAD library; or LC_TEST_TEN_DOWNLOAD=1 to fetch it


def test_auto_gain_lifts_quiet_audio_only():
    cfg = Config().vad
    t = np.arange(SAMPLE_RATE * 4) / SAMPLE_RATE
    quiet = (0.003 * np.sin(2 * np.pi * 200 * t)).astype(np.float32).reshape(-1, 512)  # about -53 dBFS
    loud = (0.5 * np.sin(2 * np.pi * 200 * t)).astype(np.float32).reshape(-1, 512)
    out = AutoGain(cfg, 512).apply(quiet)
    gain_db = 20 * np.log10(np.abs(out[-20:]).max() / 0.003)
    assert abs(gain_db - cfg.agc_max_db) < 1.0          # capped at the most gain
    assert np.allclose(AutoGain(cfg, 512).apply(loud), loud)
    cfg.agc_on = False
    assert np.array_equal(AutoGain(cfg, 512).apply(quiet), quiet)


def test_git_blob_hash():
    assert manager.calculate_git_blob_sha1(b'hello\n') == 'ce013625030ba8dba906f756967f9e9ca394464a'


def test_ten_registry_entry():
    e = find_entry('ten-vad')
    assert e is not None and e.source == 'github' and e.files == [get_ten_lib_name()]
    assert len(e.commit) == 40 and len(e.github['blob']) == 40


def _get_ten_library():
    if TEN_LIB:
        return TEN_LIB
    if os.environ.get('LC_TEST_TEN_DOWNLOAD') == '1':
        return manager.download(find_entry('ten-vad'), tempfile.mkdtemp())
    return ''


@pytest.fixture(scope='module')
def ten_library():
    path = _get_ten_library()
    if not path:
        pytest.skip('set LC_TEST_TEN_VAD or LC_TEST_TEN_DOWNLOAD=1')
    return path


def test_ten_vad_scores_tone_and_silence(ten_library):
    from livecaptions.vad.ten import TenVAD
    cfg = Config().vad
    cfg.agc_on = False
    vad = TenVAD(ten_library, 256, cfg)
    assert vad.get_version()
    silence = vad.process(np.zeros(SAMPLE_RATE, np.float32))
    assert len(silence.probs) == SAMPLE_RATE // 256 and silence.probs.max() < 0.5
    vad.reset()
    assert vad.process(np.zeros(SAMPLE_RATE, np.float32)).start == 0
    vad.close()


def test_ten_vad_found_in_models_folder(tmp_path):
    cfg = Config().vad
    assert find_ten_library(cfg, str(tmp_path)) == ''
    (tmp_path / get_ten_lib_name()).write_bytes(b'x')
    assert find_ten_library(cfg, str(tmp_path)) == str(tmp_path / get_ten_lib_name())
    assert manager.detect_format(str(tmp_path / get_ten_lib_name())) == 'vad'


def play_tone(seconds):
    """Play a 440 Hz tone with afplay (macOS) in the background and return the process."""
    path = os.path.join(tempfile.mkdtemp(), 'tone.wav')
    t = np.arange(int(48000 * seconds)) / 48000
    data = (0.3 * np.sin(2 * np.pi * 440 * t) * 32767).astype('<i2')
    with wave.open(path, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(48000)
        w.writeframes(data.tobytes())
    return subprocess.Popen(['afplay', path])


def measure_capture(audio_cfg, seconds, on_started=None):
    """
    Open, start, read and close a capture, timing each step and printing the result; skip if it cannot open.

    Args:
        audio_cfg: AudioCfg with the source to open
        seconds: how long to read
        on_started: called right after the capture started, e.g. to play a sound

    Returns:
        (times, n_samples, peak): seconds per step, samples read and their peak
    """
    from livecaptions.audio.capture import NativeCapture
    native.set_log_level(2)  # info: lc_tap_macos.mm logs its Core Audio calls
    times = {}
    t0 = time.monotonic()
    try:
        cap = NativeCapture(audio_cfg)
    except native.NativeError as e:
        native.set_log_level(3)
        pytest.skip(f'cannot open the {audio_cfg.source} capture here: {e}')
    times['open'] = time.monotonic() - t0
    n_samples, peak = 0, 0.0
    try:
        t0 = time.monotonic()
        cap.start()
        times['start'] = time.monotonic() - t0
        if on_started:
            on_started()
        t0 = time.monotonic()
        while time.monotonic() - t0 < seconds:
            x = cap.read(100)
            n_samples += len(x)
            peak = max(peak, float(np.abs(x).max())) if len(x) else peak
    finally:
        t0 = time.monotonic()
        cap.stop()
        times['stop'] = time.monotonic() - t0
        t0 = time.monotonic()
        cap.close()
        times['close'] = time.monotonic() - t0
        native.set_log_level(3)
    print(f'{audio_cfg.source}:', ', '.join(f'{k} {v:.2f} s' for k, v in times.items()),
          f'| read {n_samples} samples in {seconds:.0f} s, peak {peak:.3f}')
    return times, n_samples, peak


@pytest.mark.skipif(not (sys.platform == 'darwin' and os.environ.get('LC_TEST_TAP') == '1'),
                    reason='macOS with LC_TEST_TAP=1 (plays a tone and records system audio)')
def test_macos_tap_smoke():
    """
    Open the process tap and read while afplay plays a tone; skip if it cannot open.

    For a real Mac: macOS asks for the system audio recording permission first (until it is answered the IO calls
    wait, up to minutes on the CI VM). Timings, samples and the peak are printed (pytest -rP); with the permission
    the peak should be near 0.3, the tone's level.
    """
    from livecaptions.audio.devices import supports_system_audio
    if not supports_system_audio():
        pytest.skip('needs macOS 14.2+')
    cfg = Config().audio
    cfg.source = 'system'
    players = []
    measure_capture(cfg, 2.0, on_started=lambda: players.append(play_tone(1.5)))
    for p in players:
        try:
            p.wait(10)
        except subprocess.TimeoutExpired:
            p.kill()


@pytest.mark.skipif(os.environ.get('LC_TEST_MIC') != '1', reason='set LC_TEST_MIC=1 to open the default microphone')
def test_mic_smoke():
    """Open the default microphone through miniaudio and read 1 s; timings, samples and the peak are printed."""
    cfg = Config().audio
    cfg.source = 'mic'
    measure_capture(cfg, 1.0)
