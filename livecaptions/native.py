"""ctypes bindings for lc_native (native/lc_native): audio capture, whisper.cpp and Silero VAD."""
import ctypes as C
import os
import sys
import threading

DEVICE_ID_BYTES = 512
NAME_BYTES = 256

_lib = None
_lib_lock = threading.Lock()


class NativeError(RuntimeError):
    """The native library is missing or a native call failed."""


class DeviceInfo(C.Structure):
    _fields_ = [('name', C.c_char * NAME_BYTES), ('id', C.c_ubyte * DEVICE_ID_BYTES), ('is_default', C.c_int32)]


class AudioParams(C.Structure):
    _fields_ = [('kind', C.c_int32), ('device_id', C.POINTER(C.c_ubyte)), ('channel_mode', C.c_int32),
                ('period_ms', C.c_int32), ('ring_ms', C.c_int32), ('lpf_order', C.c_int32), ('tap_apps', C.c_char_p)]


class DecodeParams(C.Structure):
    _fields_ = [
        ('n_threads', C.c_int32), ('translate', C.c_int32), ('no_context', C.c_int32), ('single_segment', C.c_int32),
        ('token_timestamps', C.c_int32), ('max_len', C.c_int32), ('split_on_word', C.c_int32), ('max_tokens', C.c_int32),
        ('audio_ctx', C.c_int32), ('beam_size', C.c_int32), ('best_of', C.c_int32), ('suppress_blank', C.c_int32),
        ('suppress_nst', C.c_int32), ('temperature', C.c_float), ('temperature_inc', C.c_float),
        ('entropy_thold', C.c_float), ('logprob_thold', C.c_float), ('no_speech_thold', C.c_float),
        ('length_penalty', C.c_float), ('max_initial_ts', C.c_float), ('thold_pt', C.c_float),
        ('thold_ptsum', C.c_float), ('language', C.c_char_p), ('prompt_tokens', C.POINTER(C.c_int32)),
        ('prompt_n_tokens', C.c_int32), ('suppress_regex', C.c_char_p),
    ]


class TokenInfo(C.Structure):
    _fields_ = [('id', C.c_int32), ('p', C.c_float), ('plog', C.c_float), ('t0', C.c_int64), ('t1', C.c_int64)]


class GpuInfo(C.Structure):
    _fields_ = [('name', C.c_char * NAME_BYTES), ('description', C.c_char * NAME_BYTES), ('type', C.c_int32),
                ('memory_free', C.c_int64), ('memory_total', C.c_int64)]


_STRUCTS = [DeviceInfo, AudioParams, DecodeParams, TokenInfo, GpuInfo]  # order of lc_struct_sizes

LIB_VERSION = 3   # lc_version() of the library this module expects
_P = C.c_void_p
_SIGNATURES = {
    'lc_version': (C.c_int32, []),
    'lc_struct_sizes': (None, [C.POINTER(C.c_int32), C.c_int32]),
    'lc_set_log_level': (None, [C.c_int32]),
    'lc_system_info': (C.c_char_p, []),
    'lc_gpu_count': (C.c_int32, []),
    'lc_gpu_get': (C.c_int32, [C.c_int32, C.POINTER(GpuInfo)]),
    'lc_system_audio_available': (C.c_int32, []),
    'lc_audio_list': (C.c_int32, [C.c_int32, C.POINTER(DeviceInfo), C.c_int32]),
    'lc_audio_open': (_P, [C.POINTER(AudioParams), C.c_char_p, C.c_int32]),
    'lc_audio_start': (C.c_int32, [_P]),
    'lc_audio_stop': (C.c_int32, [_P]),
    'lc_audio_read': (C.c_int32, [_P, C.POINTER(C.c_float), C.c_int32, C.c_int32]),
    'lc_audio_idle_seconds': (C.c_double, [_P]),
    'lc_audio_dropped': (C.c_int64, [_P]),
    'lc_audio_levels': (C.c_int32, [_P, C.c_uint64, C.POINTER(C.c_float), C.c_int32, C.POINTER(C.c_uint64)]),
    'lc_audio_position': (C.c_uint64, [_P]),
    'lc_audio_flags': (C.c_int32, [_P]),
    'lc_audio_close': (None, [_P]),
    'lc_whisper_load': (_P, [C.c_char_p, C.c_int32, C.c_int32, C.c_int32, C.c_char_p, C.c_int32]),
    'lc_whisper_free': (None, [_P]),
    'lc_whisper_full': (C.c_int32, [_P, C.POINTER(DecodeParams), C.POINTER(C.c_float), C.c_int32]),
    'lc_whisper_abort': (None, [_P, C.c_int32]),
    'lc_whisper_tokenize': (C.c_int32, [_P, C.c_char_p, C.POINTER(C.c_int32), C.c_int32]),
    'lc_whisper_token_eot': (C.c_int32, [_P]),
    'lc_whisper_is_multilingual': (C.c_int32, [_P]),
    'lc_whisper_n_text_ctx': (C.c_int32, [_P]),
    'lc_whisper_model_type': (C.c_char_p, [_P]),
    'lc_whisper_lang_id': (C.c_int32, [_P]),
    'lc_lang_str': (C.c_char_p, [C.c_int32]),
    'lc_whisper_n_segments': (C.c_int32, [_P]),
    'lc_whisper_segment_t0': (C.c_int64, [_P, C.c_int32]),
    'lc_whisper_segment_t1': (C.c_int64, [_P, C.c_int32]),
    'lc_whisper_segment_no_speech_prob': (C.c_float, [_P, C.c_int32]),
    'lc_whisper_n_tokens': (C.c_int32, [_P, C.c_int32]),
    'lc_whisper_token': (C.c_int32, [_P, C.c_int32, C.c_int32, C.POINTER(TokenInfo)]),
    'lc_whisper_token_text': (C.c_char_p, [_P, C.c_int32, C.c_int32]),
    'lc_vad_load': (_P, [C.c_char_p, C.c_int32, C.c_char_p, C.c_int32]),
    'lc_vad_window': (C.c_int32, [_P]),
    'lc_vad_process': (C.c_int32, [_P, C.POINTER(C.c_float), C.c_int32, C.POINTER(C.c_float), C.c_int32]),
    'lc_vad_reset': (None, [_P]),
    'lc_vad_free': (None, [_P]),
}


def get_lib_candidates():
    """Return the paths where lc_native is looked for, in order."""
    name = {'win32': 'lc_native.dll', 'darwin': 'lc_native.dylib'}.get(sys.platform, 'lc_native.so')
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    cands = []
    if os.environ.get('LC_NATIVE_PATH'):
        cands.append(os.environ['LC_NATIVE_PATH'])
    if getattr(sys, 'frozen', False):  # PyInstaller bundle
        cands.append(os.path.join(getattr(sys, '_MEIPASS', ''), 'livecaptions', 'bin', name))
    cands.append(os.path.join(pkg_dir, 'bin', name))
    return cands


def load_lib():
    """Load lc_native once and check that its structs match the ctypes mirrors."""
    global _lib
    with _lib_lock:
        if _lib is not None:
            return _lib
        path = next((p for p in get_lib_candidates() if os.path.exists(p)), None)
        if path is None:
            raise NativeError('lc_native not found; build it with: python scripts/build_native.py')
        if sys.platform == 'win32':
            os.add_dll_directory(os.path.dirname(path))
        lib = C.CDLL(path)
        lib.lc_version.restype = C.c_int32
        if lib.lc_version() < LIB_VERSION:   # an older build lacks functions this app calls
            raise NativeError(f'lc_native at {path} is version {lib.lc_version()}, this app needs {LIB_VERSION}: '
                              'rebuild it with python scripts/build_native.py')
        for fn_name, (restype, argtypes) in _SIGNATURES.items():
            fn = getattr(lib, fn_name)
            fn.restype = restype
            fn.argtypes = argtypes
        check_struct_sizes(lib)
        _lib = lib
        return lib


def check_struct_sizes(lib):
    """Raise NativeError if a ctypes struct does not have the size of its C counterpart."""
    sizes = (C.c_int32 * len(_STRUCTS))()
    lib.lc_struct_sizes(sizes, len(_STRUCTS))
    for struct, size in zip(_STRUCTS, sizes):
        if C.sizeof(struct) != size:
            raise NativeError(f'{struct.__name__}: ctypes size {C.sizeof(struct)} != native size {size}; '
                              'rebuild lc_native or update native.py')


def is_available():
    """Return whether lc_native can be loaded."""
    try:
        load_lib()
        return True
    except (NativeError, OSError):
        return False


def make_err_buf():
    """Return a buffer for native error messages."""
    return C.create_string_buffer(512)


def err2str(buf):
    """Convert a native error buffer to str."""
    return buf.value.decode('utf-8', errors='replace')


def list_gpus():
    """Return [(index, description, type)] of GPUs whisper.cpp can use; type is 'discrete' or 'integrated'."""
    lib = load_lib()
    out = []
    info = GpuInfo()
    for i in range(lib.lc_gpu_count()):
        if lib.lc_gpu_get(i, C.byref(info)):
            desc = info.description.decode('utf-8', errors='replace') or info.name.decode('utf-8', errors='replace')
            out.append((i, desc, 'discrete' if info.type == 1 else 'integrated'))
    return out


def set_log_level(level):
    """Set the lowest whisper.cpp/ggml log level printed to stderr (1 debug .. 4 error, 5 silent)."""
    load_lib().lc_set_log_level(level)
