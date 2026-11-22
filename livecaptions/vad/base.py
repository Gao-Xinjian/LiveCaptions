"""VAD interface: audio in, one speech probability and one RMS level per hop out."""
from dataclasses import dataclass

import numpy as np


@dataclass
class VadResult:
    start: int          # absolute sample index of the first hop
    hop: int            # samples per hop
    probs: np.ndarray   # speech probability per hop, 0..1
    rms: np.ndarray     # RMS level per hop (linear, full scale = 1)


class VAD:
    """Base class: buffers partial hops; subclasses score whole hops in _score."""
    hop = 320  # samples (20 ms at 16 kHz)

    def __init__(self):
        self._rest = np.zeros(0, np.float32)
        self._pos = 0  # absolute sample index of the first sample in _rest

    def process(self, x):
        """Return a VadResult for the hops completed by this chunk (possibly none)."""
        buf = np.concatenate([self._rest, x]) if len(self._rest) else x
        n_hops = len(buf) // self.hop
        start = self._pos
        frames = buf[:n_hops * self.hop].reshape(n_hops, self.hop)
        self._rest = buf[n_hops * self.hop:].copy()
        self._pos += n_hops * self.hop
        if n_hops == 0:
            return VadResult(start, self.hop, np.zeros(0, np.float32), np.zeros(0, np.float32))
        rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1)).astype(np.float32)
        return VadResult(start, self.hop, self._score(frames, rms), rms)

    def _score(self, frames, rms):
        raise NotImplementedError

    def reset(self):
        """Forget the state between sessions."""
        self._rest = np.zeros(0, np.float32)
        self._pos = 0

    def close(self):
        pass


class NoVAD(VAD):
    """Treat everything as speech."""

    def _score(self, frames, rms):
        return np.ones(len(frames), np.float32)
