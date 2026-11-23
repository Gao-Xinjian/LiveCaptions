"""
Sliding window, in the style of whisper.cpp's `stream` example (PLAN.md §6.2).

Every step the current window (audio since the window started) is transcribed again and shown in grey. When the
window reaches its length, its text becomes final and a new window starts, keeping a short overlap of audio for
context; final text whose audio is in that overlap is skipped by the next runs. Pauses (T2) also end a window.
"""
from livecaptions import SAMPLE_RATE
from livecaptions.segmentation.base import PolicyOutput
from livecaptions.segmentation.buffered import BufferedPolicy


class SlidingWindowPolicy(BufferedPolicy):
    def get_window_s(self):
        return max(1.0, min(self.cfg.stream.window_s, self.get_max_buffer_s()))

    def _close_window(self, out):
        """The window is full: its text becomes final, and only the overlap stays in the buffer."""
        if self.ep.speech_since_run:
            _, new, _, _ = self._run_new(out)
            self.history = [new]
        if self.history:
            self._commit(self.history[-1], out)
        self.history = []
        keep_from = self.buf_start + self.buf_len - int(self.cfg.stream.keep_s * SAMPLE_RATE)
        n_before = sum(1 for u in self.kept if u.t1 * SAMPLE_RATE <= keep_from)  # the rest is in the overlap
        self._trim_to(keep_from, n_before, out)
        self.last_run_end = self.buf_start + self.buf_len

    def process(self, chunk, vres, idle_s):
        sc = self.cfg.stream
        out = PolicyOutput()
        self._feed(chunk, vres)
        had_text = self.has_text()
        open_, endpoint = self._check_endpoint(idle_s)
        if endpoint:
            self._endpoint(out)
        elif not open_:
            self._keep_preroll(out)
        elif self.get_buffer_s() >= self.get_window_s():
            self._close_window(out)
        elif self.ep.speech_since_run and self.get_new_s() >= self.get_step_s(sc.step_s):
            units, new, _, _ = self._run_new(out)
            self.history = [new]
            self._drop_noise(units, had_text, out)
        self._update_tentative(out)
        return out
