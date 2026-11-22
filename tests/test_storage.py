"""Settings, session files, model detection and the native struct layout."""
import json
import os

import numpy as np
import pytest

from livecaptions import native
from livecaptions.config import store
from livecaptions.config.schema import Config
from livecaptions.models import manager
from livecaptions.models.registry import load_registry
from livecaptions.output.session import SessionWriter, make_session_name
from livecaptions.output.writers import sec2clock
from livecaptions.segmentation.base import Line


def test_config_roundtrip(tmp_path):
    cfg = Config()
    cfg.audio.gain_db = 6.0
    cfg.save.formats = ['txt', 'srt']
    cfg.filters.blacklist = ['x']
    path = str(tmp_path / 'c.toml')
    store.save_config(cfg, path)
    back = store.load_config(path)
    assert back == cfg


def test_config_tolerates_bad_values():
    cfg = store.dict2cfg({'audio': {'gain_db': 'loud', 'unknown': 1}, 'nosection': {}, 'stream': {'agree_n': '3'}})
    assert cfg.audio.gain_db == Config().audio.gain_db
    assert cfg.stream.agree_n == 3


def test_session_writer(tmp_path):
    cfg = Config()
    cfg.save.dir = str(tmp_path)
    cfg.save.formats = ['txt', 'srt', 'vtt', 'jsonl', 'md']
    s = SessionWriter(cfg)
    s.write_audio(np.zeros(16000, np.float32))
    s.write_lines([Line('Hello world.', 1.0, 2.5), Line('第二行', 3.0, 4.0)])
    out = s.close(1.0)
    names = sorted(os.listdir(out))
    assert names == ['audio.wav', 'session.json', 'transcript.jsonl', 'transcript.md', 'transcript.srt',
                     'transcript.txt', 'transcript.vtt']
    assert os.path.getsize(os.path.join(out, 'audio.wav')) == 44 + 32000
    srt = open(os.path.join(out, 'transcript.srt'), encoding='utf-8').read()
    assert '1\n00:00:01,000 --> 00:00:02,500\nHello world.' in srt
    lines = [json.loads(x) for x in open(os.path.join(out, 'transcript.jsonl'), encoding='utf-8')]
    assert lines[1]['text'] == '第二行'


def test_empty_session_is_deleted(tmp_path):
    cfg = Config()
    cfg.save.dir = str(tmp_path)
    s = SessionWriter(cfg)
    assert s.close(0.0) == '' and not os.listdir(tmp_path)


def test_session_name():
    from datetime import datetime
    now = datetime(2026, 11, 21, 9, 5, 7)
    assert make_session_name('%Y-%m-%d %H-%M-%S {profile}', 'Neuro', now) == '2026-11-21 09-05-07 Neuro'
    assert make_session_name('%Y-%m-%d {profile}', 'default', now) == '2026-11-21'
    assert make_session_name('a/b:{profile}', 'x', now) == 'a_b_x'


def test_sec2clock():
    assert sec2clock(3723.5) == '01:02:03,500'
    assert sec2clock(1.25, '.') == '00:00:01.250'


def test_detect_format(tmp_path):
    ggml = tmp_path / 'ggml-tiny.bin'
    ggml.write_bytes(b'lmgg' + b'\0' * 16)
    vad = tmp_path / 'ggml-silero-v6.2.0.bin'
    vad.write_bytes(b'lmgg' + b'\0' * 16)
    ct2 = tmp_path / 'fw'
    ct2.mkdir()
    for n in ('model.bin', 'config.json', 'tokenizer.json'):
        (ct2 / n).write_bytes(b'{}')
    other = tmp_path / 'notes.bin'
    other.write_bytes(b'abcd')
    assert manager.detect_format(str(ggml)) == 'whispercpp'
    assert manager.detect_format(str(vad)) == 'vad'
    assert manager.detect_format(str(ct2)) == 'fasterwhisper'
    assert manager.detect_format(str(other)) is None
    assert {m.backend for m in manager.list_installed(str(tmp_path))} == {'whispercpp', 'vad', 'fasterwhisper'}


def test_registry():
    entries = load_registry()
    assert any(e.backend == 'vad' for e in entries)
    turbo = next(e for e in entries if e.id == 'large-v3-turbo-q5_0')
    assert not turbo.can_translate and turbo.files == ['ggml-large-v3-turbo-q5_0.bin']


@pytest.mark.skipif(not native.is_available(), reason='lc_native not built')
def test_native_structs_match():
    native.check_struct_sizes(native.load_lib())  # raises if a ctypes mirror is out of date
