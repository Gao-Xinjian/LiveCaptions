"""Resource usage of this process: memory, CPU, inference load, GPU (Windows), battery."""
import os
import sys
import time
from dataclasses import dataclass

import psutil


@dataclass
class Sample:
    rss: int = 0                 # bytes; Windows working set, macOS physical footprint
    cpu_pct: float = 0.0         # of all cores, or of one core if per_core
    busy_pct: float = 0.0        # share of wall time spent in inference
    rtf: float = 0.0             # inference seconds per second of audio captured (real-time factor)
    gpu_pct: float | None = None
    battery_pct: float | None = None
    plugged: bool | None = None


def _get_footprint_mac(pid):
    """Return the physical footprint on macOS (what Activity Monitor shows), or None."""
    import ctypes as C

    class RusageInfoV2(C.Structure):
        _fields_ = [('ri_uuid', C.c_uint8 * 16), ('ri_user_time', C.c_uint64), ('ri_system_time', C.c_uint64),
                    ('ri_pkg_idle_wkups', C.c_uint64), ('ri_interrupt_wkups', C.c_uint64), ('ri_pageins', C.c_uint64),
                    ('ri_wired_size', C.c_uint64), ('ri_resident_size', C.c_uint64), ('ri_phys_footprint', C.c_uint64),
                    ('ri_proc_start_abstime', C.c_uint64), ('ri_proc_exit_abstime', C.c_uint64),
                    ('ri_child_user_time', C.c_uint64), ('ri_child_system_time', C.c_uint64),
                    ('ri_child_pkg_idle_wkups', C.c_uint64), ('ri_child_interrupt_wkups', C.c_uint64),
                    ('ri_child_pageins', C.c_uint64), ('ri_child_elapsed_abstime', C.c_uint64),
                    ('ri_diskio_bytesread', C.c_uint64), ('ri_diskio_byteswritten', C.c_uint64)]
    try:
        libc = C.CDLL('/usr/lib/libSystem.dylib')
        info = RusageInfoV2()
        if libc.proc_pid_rusage(pid, 2, C.byref(info)) == 0:  # RUSAGE_INFO_V2
            return int(info.ri_phys_footprint)
    except OSError:
        pass
    return None


class ResourceSampler:
    def __init__(self):
        self.proc = psutil.Process()
        self.proc.cpu_percent(None)
        self.n_cpu = psutil.cpu_count() or 1
        self.gpu = None
        if sys.platform == 'win32':
            try:
                from livecaptions.stats.gpu_win import GpuUsage
                self.gpu = GpuUsage(os.getpid())
            except OSError:
                self.gpu = None
        self._t = time.monotonic()
        self._busy = 0.0
        self._audio = 0.0

    def sample(self, pipeline=None, per_core=False, with_gpu=True):
        """Return a Sample; busy/RTF come from the running pipeline's counters (since the previous call)."""
        now = time.monotonic()
        dt = max(1e-6, now - self._t)
        s = Sample()
        s.rss = self.proc.memory_info().rss
        if sys.platform == 'darwin':
            s.rss = _get_footprint_mac(self.proc.pid) or s.rss
        cpu = self.proc.cpu_percent(None)
        s.cpu_pct = cpu if per_core else cpu / self.n_cpu
        if pipeline is not None:
            d_busy = pipeline.busy_s - self._busy
            d_audio = pipeline.audio_s - self._audio
            self._busy, self._audio = pipeline.busy_s, pipeline.audio_s
            s.busy_pct = 100.0 * max(0.0, d_busy) / dt
            s.rtf = d_busy / d_audio if d_audio > 0 else 0.0
        else:
            self._busy = self._audio = 0.0
        if with_gpu and self.gpu is not None:
            s.gpu_pct = self.gpu.sample()
        try:
            bat = psutil.sensors_battery()
        except (AttributeError, NotImplementedError, OSError):
            bat = None
        if bat is not None:
            s.battery_pct, s.plugged = bat.percent, bat.power_plugged
        self._t = now
        return s

    def reset_pipeline(self):
        self._busy = self._audio = 0.0
