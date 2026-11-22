"""Automatic gain for the input of the neural VADs (Silero, TEN); the recorded and transcribed audio is not changed."""
from collections import deque

import numpy as np

from livecaptions import SAMPLE_RATE


class AutoGain:
    """
    Slow automatic gain: Silero misses speech below about -30 dBFS (a lecturer far from a laptop microphone).
    The level is a high percentile of recent window RMS, so pauses do not pump the gain up. Targets were compared
    on lecture recordings: -10 dBFS found as much speech as a fixed +30 dB, without triggering on hiss or loud noise.
    """
    TARGET = 0.3  # window RMS the gain aims for (-10 dBFS)

    def __init__(self, vad_cfg, hop, window_s=3.0):
        self.cfg = vad_cfg
        self.levels = deque(maxlen=max(4, int(window_s * SAMPLE_RATE / hop)))
        self.gain = 1.0

    def apply(self, frames):
        """Return frames (n_windows, hop) scaled by the gain of each window, or unchanged if the gain is off."""
        if not self.cfg.agc_on:
            return frames
        max_gain = 10 ** (self.cfg.agc_max_db / 20)
        gains = np.empty(len(frames), np.float32)
        for i, rms in enumerate(np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))):
            self.levels.append(float(rms))
            level = np.percentile(self.levels, 80)
            target = float(np.clip(self.TARGET / max(level, 1e-6), 1.0, max_gain))
            self.gain += 0.2 * (target - self.gain)  # smooth over a few windows
            gains[i] = self.gain
        return np.clip(frames * gains[:, None], -1.0, 1.0)

    def reset(self):
        self.levels.clear()
        self.gain = 1.0
