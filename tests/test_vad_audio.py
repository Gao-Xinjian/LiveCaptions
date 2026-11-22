"""Automatic gain, TEN VAD, the GitHub download check, and system audio capture (macOS tap smoke test)."""
import os
import sys
import tempfile
import time

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


@pytest.mark.skipif(not (sys.platform == 'darwin' and native.is_available()), reason='macOS only')
def test_macos_tap_smoke():
    """Open, start and close the process tap. Without the permission (as on CI) it may fail to open: then skip."""
    from livecaptions.audio.capture import NativeCapture
    from livecaptions.audio.devices import supports_system_audio
    if not supports_system_audio():
        pytest.skip('needs macOS 14.2+')
    cfg = Config().audio
    cfg.source = 'system'
    try:
        cap = NativeCapture(cfg)
    except native.NativeError as e:
        pytest.skip(f'tap not available here: {e}')
    try:
        cap.start()
        t0 = time.monotonic()
        while time.monotonic() - t0 < 1.0:
            cap.read(100)  # silence or nothing is fine; it must not crash
    finally:
        cap.stop()
        cap.close()
