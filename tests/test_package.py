"""The self-check and the packaging helpers; the PyInstaller build itself runs in CI (scripts/build_app.py)."""
import os
import sys

import pytest

from livecaptions import native
from livecaptions.selfcheck import run_self_check

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.environ.get('LC_TEST_MODEL', '')
WAV = os.environ.get('LC_TEST_WAV', '')


@pytest.mark.skipif(not native.is_available(), reason='needs lc_native')
def test_self_check_passes(tmp_path):
    report = tmp_path / 'report.txt'
    args = ['--self-check', '--cpu', '--report', str(report)]
    if MODEL and WAV:
        args += ['--model', MODEL, '--wav', WAV]
    code = run_self_check(args)
    text = report.read_text(encoding='utf-8')
    assert code == 0, text
    assert ('ok   transcribe' in text) == bool(MODEL and WAV)


def test_license_folder(tmp_path):
    sys.path.insert(0, os.path.join(ROOT, 'scripts'))
    try:
        import build_app
    finally:
        sys.path.pop(0)
    out = tmp_path / 'licenses'
    build_app.write_licenses(str(out), ['chinese'])
    index = (out / 'README.txt').read_text(encoding='utf-8')
    for name in ('numpy', 'PySide6_Essentials', 'platformdirs', 'Python '):
        assert name in index
    assert (out / 'LiveCaptions-LICENSE').exists()
    assert any((out / 'numpy').iterdir())
