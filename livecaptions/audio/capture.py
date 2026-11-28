"""Audio sources for the pipeline: live capture through lc_native, or a WAV file replayed in real time."""
import ctypes as C
import sys
import threading
import time
import wave

import numpy as np

from livecaptions import SAMPLE_RATE, native
from livecaptions.audio.devices import hex2id, supports_system_audio

FLAG_STOPPED = 1   # the device stopped by itself (unplugged, driver error)
FLAG_REROUTED = 2  # the default device changed and capture followed it
LEVEL_SAMPLES = 160       # samples per waveform column: 10 ms (lc_native kLcLevelSamples)
LEVEL_COLS_MAX = 6000     # columns lc_native keeps: 60 s


class NativeCapture:
    """Microphone or loopback capture at 16 kHz mono float32; the device thread only fills a ring buffer."""

    def __init__(self, audio_cfg, max_read=SAMPLE_RATE):
        self.lib = native.load_lib()
        source = audio_cfg.source
        if source == 'system' and not supports_system_audio():
            raise native.NativeError('System audio capture needs Windows, or macOS 14.2 or later')
        self._dev_id = hex2id(audio_cfg.device_id)  # kept alive while the device is open
        self._apps = audio_cfg.tap_apps.encode('utf-8')
        params = native.AudioParams(
            kind=1 if source == 'system' else 0,
            device_id=C.cast(self._dev_id, C.POINTER(C.c_ubyte)) if self._dev_id is not None else None,
            channel_mode={'mix': 0, 'left': 1, 'right': 2}.get(audio_cfg.channel_mode, 0),
            period_ms=audio_cfg.period_ms, ring_ms=audio_cfg.ring_s * 1000, lpf_order=audio_cfg.lpf_order,
            tap_apps=self._apps if sys.platform == 'darwin' else None)
        err = native.make_err_buf()
        self.handle = self.lib.lc_audio_open(C.byref(params), err, len(err))
        if not self.handle:
            raise native.NativeError(native.err2str(err))
        self._buf = (C.c_float * max_read)()
        self._max = max_read
        self._levels = (C.c_float * LEVEL_COLS_MAX)()
        self._close_lock = threading.Lock()   # the UI reads levels while the pipeline thread may close
        self.finished = False  # a live device never runs out

    def start(self):
        if not self.lib.lc_audio_start(self.handle):
            raise native.NativeError('could not start the audio device')

    def stop(self):
        self.lib.lc_audio_stop(self.handle)

    def read(self, timeout_ms=100):
        """Return the samples captured so far (float32, may be empty after the timeout)."""
        n = self.lib.lc_audio_read(self.handle, self._buf, self._max, timeout_ms)
        return np.ctypeslib.as_array(self._buf)[:n].copy()

    def idle_seconds(self):
        return self.lib.lc_audio_idle_seconds(self.handle)

    def flags(self):
        return self.lib.lc_audio_flags(self.handle)

    def dropped(self):
        return self.lib.lc_audio_dropped(self.handle)

    def position(self):
        """Return the capture position (samples since the device was opened) of the next sample read() returns."""
        return self.lib.lc_audio_position(self.handle)

    def levels(self, since):
        """
        Return the peak of each 10 ms column captured since column `since`, as it arrives (any thread).

        Returns:
            (peaks, first): float32 array and the column number of its first value; empty once closed
        """
        with self._close_lock:
            if not self.handle:
                return np.zeros(0, np.float32), since
            total = C.c_uint64()
            n = self.lib.lc_audio_levels(self.handle, since, self._levels, LEVEL_COLS_MAX, C.byref(total))
            return np.ctypeslib.as_array(self._levels)[:n].copy(), total.value - n

    def close(self):
        with self._close_lock:
            if self.handle:
                self.lib.lc_audio_close(self.handle)
                self.handle = None


def read_wav(path):
    """Read a PCM WAV file as 16 kHz mono float32 (other rates are resampled linearly)."""
    with wave.open(path, 'rb') as w:
        n_ch, width, rate, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        raw = w.readframes(n)
    if width == 2:
        x = np.frombuffer(raw, dtype='<i2').astype(np.float32) / 32768.0
    elif width == 4:
        x = np.frombuffer(raw, dtype='<i4').astype(np.float32) / 2147483648.0
    elif width == 1:
        x = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f'unsupported sample width {width} in {path}')
    x = x.reshape(-1, n_ch).mean(axis=1)
    if rate != SAMPLE_RATE:
        t_new = np.arange(int(len(x) * SAMPLE_RATE / rate)) / SAMPLE_RATE
        x = np.interp(t_new, np.arange(len(x)) / rate, x).astype(np.float32)
    return x.astype(np.float32)


class WavFileSource:
    """Replays samples as if they were captured live, for tests and tools/replay.py."""

    def __init__(self, samples, speed=1.0, chunk_ms=50, tail_s=2.0):
        """
        Args:
            samples: float32 16 kHz mono audio
            speed: 1 = real time, 2 = twice as fast, 0 = as fast as the pipeline reads
            chunk_ms: size of each read
            tail_s: silence appended at the end, so the last sentence ends by a pause
        """
        self.samples = np.concatenate([samples.astype(np.float32), np.zeros(int(tail_s * SAMPLE_RATE), np.float32)])
        self.speed = speed
        self.chunk = int(chunk_ms * SAMPLE_RATE / 1000)
        self.pos = 0
        self.t_start = None
        self.finished = False

    def start(self):
        self.t_start = time.perf_counter()

    def stop(self):
        pass

    def read(self, timeout_ms=100):
        if self.pos >= len(self.samples):
            self.finished = True
            time.sleep(timeout_ms / 1000 if self.speed > 0 else 0)
            return np.zeros(0, np.float32)
        end = min(self.pos + self.chunk, len(self.samples))
        if self.speed > 0:  # wait until this chunk would have been captured
            due = self.t_start + end / SAMPLE_RATE / self.speed
            delay = due - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
        out = self.samples[self.pos:end]
        self.pos = end
        return out

    def idle_seconds(self):
        return 0.0

    def flags(self):
        return 0

    def dropped(self):
        return 0

    def position(self):
        return self.pos

    def levels(self, since):
        """Like NativeCapture.levels: the columns 'captured' so far by the clock (all that was read at speed 0)."""
        live = self.pos
        if self.speed > 0 and self.t_start is not None:
            live = min(len(self.samples), int((time.perf_counter() - self.t_start) * self.speed * SAMPLE_RATE))
        first = max(since, live // LEVEL_SAMPLES - LEVEL_COLS_MAX)
        n = max(0, live // LEVEL_SAMPLES - first)
        cols = self.samples[first * LEVEL_SAMPLES:(first + n) * LEVEL_SAMPLES].reshape(n, LEVEL_SAMPLES)
        return np.abs(cols).max(axis=1) if n else np.zeros(0, np.float32), first

    def close(self):
        pass
