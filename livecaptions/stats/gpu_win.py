"""Per-process GPU utilisation on Windows from the PDH "GPU Engine" counters (what Task Manager shows)."""
import ctypes as C
from ctypes import wintypes as W

PDH_FMT_DOUBLE = 0x00000200
PDH_MORE_DATA = 0x800007D2
PDH_CSTATUS_NEW_DATA = 1   # 0 = valid data, 1 = new data; higher = no usable value
ERROR_SUCCESS = 0


class _FmtValue(C.Structure):
    _fields_ = [('CStatus', W.DWORD), ('doubleValue', C.c_double)]  # the union's double member, 8-byte aligned


class _FmtItem(C.Structure):
    _fields_ = [('szName', W.LPWSTR), ('FmtValue', _FmtValue)]


class GpuUsage:
    """Samples the utilisation of this process's GPU engines; sample() returns the busiest engine type in %."""

    def __init__(self, pid):
        self.pdh = C.WinDLL('pdh')
        for fn in ('PdhOpenQueryW', 'PdhAddEnglishCounterW', 'PdhCollectQueryData', 'PdhGetFormattedCounterArrayW',
                   'PdhCloseQuery'):   # PDH status codes such as PDH_MORE_DATA have the high bit set: read them unsigned
            getattr(self.pdh, fn).restype = W.DWORD
        self.query = C.c_void_p()
        self.counter = C.c_void_p()
        self.ok = False
        if self.pdh.PdhOpenQueryW(None, None, C.byref(self.query)) != ERROR_SUCCESS:
            return
        path = f'\\GPU Engine(pid_{pid}_*)\\Utilization Percentage'
        if self.pdh.PdhAddEnglishCounterW(self.query, path, None, C.byref(self.counter)) != ERROR_SUCCESS:
            return
        self.pdh.PdhCollectQueryData(self.query)  # rates need two samples
        self.ok = True

    def sample(self):
        """Return the GPU utilisation since the last call (0..100), or None if unavailable."""
        if not self.ok or self.pdh.PdhCollectQueryData(self.query) != ERROR_SUCCESS:
            return None
        size, count = W.DWORD(0), W.DWORD(0)
        ret = self.pdh.PdhGetFormattedCounterArrayW(self.counter, PDH_FMT_DOUBLE, C.byref(size), C.byref(count), None)
        if ret != PDH_MORE_DATA or size.value == 0:
            return 0.0
        buf = (C.c_byte * size.value)()
        ret = self.pdh.PdhGetFormattedCounterArrayW(self.counter, PDH_FMT_DOUBLE, C.byref(size), C.byref(count), buf)
        if ret != ERROR_SUCCESS:
            return None
        items = C.cast(buf, C.POINTER(_FmtItem))
        per_type = {}  # engine type -> summed utilisation
        for i in range(count.value):
            if items[i].FmtValue.CStatus > PDH_CSTATUS_NEW_DATA:   # e.g. an engine that appeared since the last sample
                continue
            name = items[i].szName or ''
            eng = name.split('engtype_')[-1] if 'engtype_' in name else 'other'
            per_type[eng] = per_type.get(eng, 0.0) + max(0.0, items[i].FmtValue.doubleValue)
        return min(100.0, max(per_type.values(), default=0.0))

    def close(self):
        if self.ok:
            self.pdh.PdhCloseQuery(self.query)
            self.ok = False
