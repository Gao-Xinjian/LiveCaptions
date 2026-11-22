"""VAD state machine for endpointing (PLAN.md §6.1 C): speech start, trailing silence, quietest cut points."""
from collections import deque

import numpy as np

from livecaptions import SAMPLE_RATE


class Endpointer:
    def __init__(self, vad_cfg, stream_cfg, history_s=40.0):
        self.vad_cfg = vad_cfg
        self.stream_cfg = stream_cfg
        self.speaking = False       # hop state, with hysteresis between the two thresholds
        self.speech_run = 0         # samples of speech in a row
        self.utterance = False      # speech confirmed (min speech reached) since the last reset
        self.silence = 0            # samples since the last speech hop
        self.speech_since_run = False  # confirmed speech arrived since the last engine run
        self.last_prob = 0.0
        self._levels = deque()      # (start sample, hop, rms) for quietest-point search
        self._history = int(history_s * SAMPLE_RATE)

    def update(self, vres):
        """Advance the state by the hops of a VadResult."""
        thr, neg = self.vad_cfg.threshold, min(self.vad_cfg.neg_threshold, self.vad_cfg.threshold)
        min_speech = self.vad_cfg.min_speech_ms * SAMPLE_RATE // 1000
        for i, p in enumerate(vres.probs):
            if p >= thr:
                self.speaking = True
            elif p < neg:
                self.speaking = False
            if self.speaking:
                self.speech_run += vres.hop
                self.silence = 0
                if self.speech_run >= min_speech:
                    self.utterance = True
                    self.speech_since_run = True
            else:
                self.speech_run = 0
                self.silence += vres.hop
            self._levels.append((vres.start + i * vres.hop, vres.hop, float(vres.rms[i])))
        if len(vres.probs):
            self.last_prob = float(vres.probs[-1])
        while self._levels and self._levels[-1][0] - self._levels[0][0] > self._history:
            self._levels.popleft()

    def mark_run(self):
        """Note that the engine just ran over everything received so far."""
        self.speech_since_run = False

    def reset_utterance(self):
        """Start waiting for the next utterance (after an endpoint or a cut)."""
        self.utterance = False
        self.speech_run = 0
        self.speech_since_run = False

    def get_silence_s(self):
        return self.silence / SAMPLE_RATE

    def get_required_pause(self, utterance_s):
        """Return the pause (T2) that ends an utterance of this length; shrinks if adaptive pauses are on."""
        sc = self.stream_cfg
        if not sc.adaptive_pause or sc.adaptive_to_s <= sc.adaptive_from_s:
            return sc.pause_s
        frac = np.clip((utterance_s - sc.adaptive_from_s) / (sc.adaptive_to_s - sc.adaptive_from_s), 0.0, 1.0)
        return sc.pause_s + frac * (min(sc.pause_min_s, sc.pause_s) - sc.pause_s)

    def find_quietest(self, start, end):
        """Return the absolute sample in [start, end) at the centre of the quietest stretch (3 hops), or end."""
        cand = [(s, h, r) for s, h, r in self._levels if start <= s and s + h <= end]
        if len(cand) < 3:
            return end
        rms = np.array([r for _, _, r in cand])
        smooth = np.convolve(rms, np.ones(3) / 3, mode='same')
        smooth[0], smooth[-1] = np.inf, np.inf  # edges have only half a window
        k = int(np.argmin(smooth))
        return cand[k][0] + cand[k][1] // 2
