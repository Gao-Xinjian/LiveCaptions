"""One recording session: a folder with the audio, the transcripts and a settings snapshot."""
import json
import os
import shutil
import time
from datetime import datetime

from livecaptions import SAMPLE_RATE
from livecaptions.config.store import cfg2dict, get_save_dir
from livecaptions.output.audio_writer import create_audio_writer
from livecaptions.output.writers import WRITERS


def make_session_name(pattern, profile, now=None):
    """Return the session folder name from a strftime pattern with {profile}."""
    now = now or datetime.now()
    mark = 'LCPROFILEMARK'  # stands in for {profile} while strftime runs
    name = now.strftime(pattern.replace('{profile}', mark)).replace(mark, profile if profile != 'default' else '')
    name = ''.join('_' if c in '\\/:*?"<>|' else c for c in name).strip(' .') or now.strftime('%Y-%m-%d %H-%M-%S')
    return name


class SessionWriter:
    def __init__(self, cfg, model_desc=''):
        sc = cfg.save
        root = get_save_dir(cfg)
        name = make_session_name(sc.name_pattern, cfg.terms.profile)
        self.dir = os.path.join(root, name)
        k = 2
        while os.path.exists(self.dir):  # two sessions in the same second
            self.dir = os.path.join(root, f'{name} ({k})')
            k += 1
        os.makedirs(self.dir)
        base = os.path.join(self.dir, 'transcript')
        self.audio = create_audio_writer(os.path.join(self.dir, 'audio'), sc.audio_format) if sc.save_audio else None
        self.texts = [WRITERS[f](base) for f in sc.formats if f in WRITERS] if sc.save_text else []
        self.flush_s = sc.flush_s
        self._last_flush = time.monotonic()
        self.n_lines = 0
        self.meta = {'started': datetime.now().isoformat(timespec='seconds'), 'model': model_desc,
                     'settings': cfg2dict(cfg)}
        self._write_meta()

    def _write_meta(self):
        with open(os.path.join(self.dir, 'session.json'), 'w', encoding='utf-8') as f:
            json.dump(self.meta, f, ensure_ascii=False, indent=1)

    def write_audio(self, x):
        if self.audio is not None and len(x):
            self.audio.write(x)

    def write_lines(self, lines):
        for line in lines:
            for w in self.texts:
                w.write_line(line)
            self.n_lines += 1

    def maybe_flush(self):
        """Flush to disk if the flush interval has passed."""
        if time.monotonic() - self._last_flush >= self.flush_s:
            self.flush()

    def flush(self):
        self._last_flush = time.monotonic()
        if self.audio is not None:
            self.audio.flush()
        for w in self.texts:
            w.flush()

    def get_audio_seconds(self):
        return self.audio.n_samples / SAMPLE_RATE if self.audio is not None else 0.0

    def close(self, captured_s):
        """Close all files; an empty session (nothing captured) is deleted. Returns the folder or ''."""
        if self.audio is not None:
            self.audio.close()
        for w in self.texts:
            w.close()
        if captured_s <= 0 and self.n_lines == 0:
            shutil.rmtree(self.dir, ignore_errors=True)
            return ''
        self.meta['ended'] = datetime.now().isoformat(timespec='seconds')
        self.meta['audio_s'] = round(captured_s, 2)
        self.meta['lines'] = self.n_lines
        self._write_meta()
        return self.dir
