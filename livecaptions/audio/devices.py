"""List the audio devices that can be captured."""
import ctypes as C
import sys
from dataclasses import dataclass

from livecaptions import native


@dataclass
class Device:
    id_hex: str   # raw ma_device_id as hex; '' = the system default
    name: str
    is_default: bool


def supports_system_audio():
    """Return whether system audio can be captured: Windows (loopback) or macOS 14.2+ (process tap)."""
    try:
        return bool(native.load_lib().lc_system_audio_available())
    except (native.NativeError, OSError, AttributeError):
        return False


def list_devices(source):
    """
    List the devices for an audio source.

    Args:
        source: 'mic' (capture devices) or 'system' (playback devices, captured by loopback)

    Returns:
        list of Device; empty if the native library is missing
    """
    if not native.is_available():
        return []
    lib = native.load_lib()
    kind = 0 if source == 'mic' else 1
    if kind == 1 and (sys.platform == 'darwin' or not supports_system_audio()):
        return []  # the macOS tap follows the default output device; there is nothing to choose
    infos = (native.DeviceInfo * 64)()
    n = lib.lc_audio_list(kind, infos, 64)
    out = []
    for info in infos[:n]:
        out.append(Device(bytes(info.id).hex(), info.name.decode('utf-8', errors='replace'), bool(info.is_default)))
    return out


def hex2id(id_hex):
    """Convert a hex device id to a ctypes byte array, or None for the default device."""
    if not id_hex:
        return None
    raw = bytes.fromhex(id_hex)[:native.DEVICE_ID_BYTES]
    return (C.c_ubyte * native.DEVICE_ID_BYTES).from_buffer_copy(raw.ljust(native.DEVICE_ID_BYTES, b'\0'))
