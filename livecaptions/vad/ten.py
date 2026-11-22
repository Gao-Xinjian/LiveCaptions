"""
TEN VAD (TEN framework / Agora) through its prebuilt library and ctypes.
Licence: Apache 2.0 with Agora's additional conditions (see the TEN VAD repository); downloaded only on request.
"""
import ctypes as C
import os
import sys

import numpy as np

from livecaptions import native
from livecaptions.vad.agc import AutoGain
from livecaptions.vad.base import VAD

TEN_LIB_FILES = {'win32': 'ten_vad.dll', 'darwin': 'libten_vad.dylib', 'linux': 'libten_vad.so'}
_libs = {}  # path -> loaded library


def get_ten_lib_name():
    """Return the file name of the TEN VAD library on this OS."""
    return TEN_LIB_FILES.get(sys.platform, 'libten_vad.so')


def find_ten_library(vad_cfg, models_dir):
    """Return the TEN VAD library path from the settings, or the default file in the models folder ('' if none)."""
    if vad_cfg.ten_library:
        return vad_cfg.ten_library
    cand = os.path.join(models_dir, get_ten_lib_name())
    return cand if os.path.exists(cand) else ''


def load_ten_lib(path):
    """Load the TEN VAD library once and declare its four functions."""
    path = os.path.abspath(path)
    if path in _libs:
        return _libs[path]
    if sys.platform == 'win32':
        os.add_dll_directory(os.path.dirname(path))
    try:
        lib = C.CDLL(path)
    except OSError as e:
        hint = ' (it needs the Visual C++ Redistributable)' if sys.platform == 'win32' else ''
        raise native.NativeError(f'could not load TEN VAD from {path}{hint}: {e}') from e
    lib.ten_vad_create.argtypes = [C.POINTER(C.c_void_p), C.c_size_t, C.c_float]
    lib.ten_vad_create.restype = C.c_int
    lib.ten_vad_process.argtypes = [C.c_void_p, C.POINTER(C.c_int16), C.c_size_t, C.POINTER(C.c_float), C.POINTER(C.c_int)]
    lib.ten_vad_process.restype = C.c_int
    lib.ten_vad_destroy.argtypes = [C.POINTER(C.c_void_p)]
    lib.ten_vad_destroy.restype = C.c_int
    lib.ten_vad_get_version.restype = C.c_char_p
    _libs[path] = lib
    return lib


class TenVAD(VAD):
    def __init__(self, lib_path, hop, vad_cfg):
        """
        Args:
            lib_path: TEN VAD library file
            hop: samples per frame, 160 (10 ms) or 256 (16 ms)
            vad_cfg: VadCfg (automatic gain options)
        """
        super().__init__()
        self.lib = load_ten_lib(lib_path)
        self.hop = int(hop)
        self.handle = C.c_void_p()
        self._create()
        self.agc = AutoGain(vad_cfg, self.hop)
        self._prob = C.c_float()
        self._flag = C.c_int()

    def _create(self):
        if self.lib.ten_vad_create(C.byref(self.handle), self.hop, C.c_float(0.5)) != 0:
            raise native.NativeError(f'TEN VAD could not start with frames of {self.hop} samples')

    def get_version(self):
        v = self.lib.ten_vad_get_version()
        return v.decode('ascii', errors='replace') if v else ''

    def _score(self, frames, rms):
        frames = self.agc.apply(frames)
        pcm = np.ascontiguousarray(np.clip(np.round(frames * 32767.0), -32768, 32767), dtype=np.int16)
        probs = np.empty(len(pcm), np.float32)
        base = pcm.ctypes.data
        p_prob, p_flag = C.byref(self._prob), C.byref(self._flag)
        for i in range(len(pcm)):  # one call per frame, as the API takes exactly one hop
            ptr = C.cast(base + i * self.hop * 2, C.POINTER(C.c_int16))
            if self.lib.ten_vad_process(self.handle, ptr, self.hop, p_prob, p_flag) != 0:
                probs[i] = 0.0
            else:
                probs[i] = self._prob.value
        return probs

    def reset_state(self):
        """The API has no reset: start a new instance."""
        self.close()
        self._create()

    def reset(self):
        super().reset()
        self.reset_state()
        self.agc.reset()

    def close(self):
        if self.handle:
            self.lib.ten_vad_destroy(C.byref(self.handle))
            self.handle = C.c_void_p()
