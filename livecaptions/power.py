"""Process priority and Windows efficiency mode (EcoQoS)."""
import ctypes as C
import sys

import psutil

PROCESS_POWER_THROTTLING = 4          # PROCESS_INFORMATION_CLASS ProcessPowerThrottling
POWER_THROTTLING_EXECUTION_SPEED = 0x1


class _PowerThrottlingState(C.Structure):
    _fields_ = [('Version', C.c_ulong), ('ControlMask', C.c_ulong), ('StateMask', C.c_ulong)]


def set_eco_qos(on):
    """Turn Windows 11 EcoQoS (efficient cores, low clock) on or off for this process; returns success."""
    if sys.platform != 'win32':
        return False
    state = _PowerThrottlingState(1, POWER_THROTTLING_EXECUTION_SPEED, POWER_THROTTLING_EXECUTION_SPEED if on else 0)
    k32 = C.windll.kernel32
    k32.GetCurrentProcess.restype = C.c_void_p
    k32.SetProcessInformation.argtypes = [C.c_void_p, C.c_int, C.c_void_p, C.c_ulong]
    return bool(k32.SetProcessInformation(k32.GetCurrentProcess(), PROCESS_POWER_THROTTLING,
                                          C.byref(state), C.sizeof(state)))


def set_low_priority(on):
    """Lower (or restore) the priority of this process; on macOS a lowered priority stays until restart."""
    p = psutil.Process()
    try:
        if sys.platform == 'win32':
            p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if on else psutil.NORMAL_PRIORITY_CLASS)
        elif on:
            p.nice(10)
        return True
    except (psutil.Error, OSError):
        return False


def apply_power_settings(general_cfg):
    set_low_priority(general_cfg.low_priority)
    if sys.platform == 'win32':
        set_eco_qos(general_cfg.eco_qos)
