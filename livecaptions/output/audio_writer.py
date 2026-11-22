"""Audio recording: WAV written by hand (header kept valid after every flush), FLAC and Opus through soundfile."""
import os
import struct

import numpy as np

from livecaptions import SAMPLE_RATE


class WavWriter:
    """16 kHz mono 16-bit WAV; the header is rewritten on flush, so the file stays playable after a crash."""

    def __init__(self, path):
        self.path = path
        self.f = open(path, 'wb')
        self.n_samples = 0
        self._write_header()

    def _write_header(self):
        data_bytes = self.n_samples * 2
        hdr = struct.pack('<4sI4s4sIHHIIHH4sI', b'RIFF', 36 + data_bytes, b'WAVE', b'fmt ', 16, 1, 1,
                          SAMPLE_RATE, SAMPLE_RATE * 2, 2, 16, b'data', data_bytes)
        self.f.seek(0)
        self.f.write(hdr)
        self.f.seek(0, os.SEEK_END)

    def write(self, x):
        pcm = np.clip(np.round(x * 32767.0), -32768, 32767).astype('<i2')
        self.f.write(pcm.tobytes())
        self.n_samples += len(pcm)

    def flush(self):
        self._write_header()
        self.f.flush()

    def close(self):
        if self.f:
            self.flush()
            self.f.close()
            self.f = None


class SoundFileWriter:
    """FLAC or Opus (Ogg) through the soundfile package."""

    def __init__(self, path, fmt):
        import soundfile as sf
        self.path = path
        if fmt == 'flac':
            self.f = sf.SoundFile(path, 'w', SAMPLE_RATE, 1, format='FLAC', subtype='PCM_16')
        else:
            self.f = sf.SoundFile(path, 'w', SAMPLE_RATE, 1, format='OGG', subtype='OPUS')
        self.n_samples = 0

    def write(self, x):
        self.f.write(x.astype(np.float32))
        self.n_samples += len(x)

    def flush(self):
        self.f.flush()

    def close(self):
        if self.f:
            self.f.close()
            self.f = None


def create_audio_writer(base_path, fmt):
    """Create a writer for base_path + extension; falls back to WAV if soundfile is missing."""
    if fmt in ('flac', 'opus'):
        try:
            return SoundFileWriter(base_path + ('.flac' if fmt == 'flac' else '.opus'), fmt)
        except (ImportError, OSError, RuntimeError):
            pass
    return WavWriter(base_path + '.wav')
