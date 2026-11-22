"""Silero VAD through whisper.cpp's ggml port in lc_native; the LSTM state carries over between chunks."""
import ctypes as C
import os

import numpy as np

from livecaptions import native
from livecaptions.vad.agc import AutoGain
from livecaptions.vad.base import VAD
from livecaptions.vad.energy import EnergyVAD

SILERO_FILE = 'ggml-silero-v6.2.0.bin'


def find_silero_model(vad_cfg, models_dir):
    """Return the Silero model path from the settings, or the default file in the models folder ('' if none)."""
    if vad_cfg.silero_model:
        return vad_cfg.silero_model
    cand = os.path.join(models_dir, SILERO_FILE)
    return cand if os.path.exists(cand) else ''


class SileroVAD(VAD):
    def __init__(self, model_path, n_threads, vad_cfg):
        super().__init__()
        self.lib = native.load_lib()
        err = native.make_err_buf()
        self.handle = self.lib.lc_vad_load(model_path.encode('utf-8'), n_threads, err, len(err))
        if not self.handle:
            raise native.NativeError(native.err2str(err))
        self.hop = self.lib.lc_vad_window(self.handle)
        self._out = (C.c_float * 4096)()
        self.agc = AutoGain(vad_cfg, self.hop)

    def _score(self, frames, rms):
        frames = self.agc.apply(frames)
        flat = np.ascontiguousarray(frames.reshape(-1), dtype=np.float32)
        probs = np.zeros(len(frames), np.float32)
        for i in range(0, len(frames), 4096):  # at most 4096 windows per call
            part = flat[i * self.hop:(i + 4096) * self.hop]
            n = self.lib.lc_vad_process(self.handle, part.ctypes.data_as(C.POINTER(C.c_float)), len(part), self._out, 4096)
            probs[i:i + n] = np.ctypeslib.as_array(self._out)[:n]
        return probs

    def reset_state(self):
        """Reset the LSTM, e.g. after a stretch that was not analysed."""
        self.lib.lc_vad_reset(self.handle)

    def reset(self):
        super().reset()
        self.reset_state()
        self.agc.reset()

    def close(self):
        if self.handle:
            self.lib.lc_vad_free(self.handle)
            self.handle = None


class CascadeVAD(VAD):
    """Energy gate first; the neural VAD (Silero or TEN) only scores chunks in which the energy VAD hears something."""

    def __init__(self, neural, vad_cfg, gate=0.2):
        super().__init__()
        self.neural = neural
        self.hop = neural.hop
        self.energy = EnergyVAD(vad_cfg, hop=neural.hop)
        self.gate = gate
        self._gated = False

    def _score(self, frames, rms):
        e_probs = self.energy._score(frames, rms)
        if e_probs.max(initial=0.0) < self.gate:
            self._gated = True
            return np.zeros(len(frames), np.float32)
        if self._gated:  # the neural VAD skipped some audio: start it fresh
            self.neural.reset_state()
            self._gated = False
        return self.neural._score(frames, rms)

    def reset(self):
        super().reset()
        self.energy.reset()
        self.neural.reset()

    def close(self):
        self.neural.close()
