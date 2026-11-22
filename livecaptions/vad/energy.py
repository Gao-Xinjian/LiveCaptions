"""Adaptive energy VAD: level against a tracked noise floor, nearly free to run."""
from collections import deque

import numpy as np

from livecaptions import SAMPLE_RATE
from livecaptions.vad.base import VAD


class EnergyVAD(VAD):
    """Speech probability from how far the (band-limited) level is above the noise floor."""

    def __init__(self, vad_cfg, hop=320):
        super().__init__()
        self.hop = hop
        self.cfg = vad_cfg
        n_floor = max(10, int(vad_cfg.floor_window_s * SAMPLE_RATE / hop))
        self._hist = deque(maxlen=n_floor)  # recent hop levels, dB
        self._floor = -70.0
        self._since_floor = 0
        freqs = np.fft.rfftfreq(hop, 1.0 / SAMPLE_RATE)
        self._band = (freqs >= 100) & (freqs <= 4000)
        self._win = np.hanning(hop).astype(np.float32)

    def get_levels_db(self, frames, rms):
        """Return the level of each hop in dBFS, limited to speech frequencies if configured."""
        if self.cfg.band_limit:
            spec = np.abs(np.fft.rfft(frames * self._win, axis=1)) ** 2
            band = spec[:, self._band].sum(axis=1) * 2.0 / (np.sum(self._win ** 2) * self.hop)
            level = np.sqrt(band)
        else:
            level = rms
        return 20.0 * np.log10(np.maximum(level, 1e-7))

    def _score(self, frames, rms):
        db = self.get_levels_db(frames, rms)
        probs = np.empty(len(db), np.float32)
        for i, d in enumerate(db):
            # Track the noise floor.
            self._hist.append(d)
            self._since_floor += 1
            if self._since_floor >= 10 or len(self._hist) < 10:
                self._floor = float(np.percentile(self._hist, self.cfg.floor_percentile))
                self._since_floor = 0
            # Score the hop.
            thr = self._floor + self.cfg.margin_db
            p = 1.0 / (1.0 + np.exp(-(d - thr) / 2.0))  # 2 dB per logit unit
            probs[i] = p if d >= self.cfg.min_level_db else 0.0
        return probs

    def reset(self):
        super().reset()
        self._hist.clear()
        self._floor = -70.0
