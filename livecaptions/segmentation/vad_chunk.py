"""
VAD-chunked (PLAN.md §6.2): one run per utterance, at its endpoint; the cheapest policy.

An utterance runs from the start of speech (plus pre-roll) to a pause (T2), the device going idle, or the longest
buffer (T3, cut at the quietest point). Only then is it transcribed, once, without the trailing silence, and all of
its text becomes final. Optional previews transcribe the utterance so far every few seconds and show it in grey;
if no speech arrives after the last preview, the endpoint reuses it instead of running again.
"""
from livecaptions.segmentation.base import PolicyOutput
from livecaptions.segmentation.buffered import BufferedPolicy


class VadChunkPolicy(BufferedPolicy):
    def process(self, chunk, vres, idle_s):
        sc = self.cfg.stream
        out = PolicyOutput()
        self._feed(chunk, vres)
        had_text = self.has_text()
        open_, endpoint = self._check_endpoint(idle_s)
        if endpoint:
            self._endpoint(out, upto=self._get_speech_end())
        elif self.get_buffer_s() >= self.get_max_buffer_s():
            self._cut_at_quietest(out)
        elif not open_:
            self._keep_preroll(out)
        elif sc.preview_on and self.ep.speech_since_run and self.get_new_s() >= self.get_step_s(sc.preview_s):
            units, new, _, _ = self._run_new(out)
            self.history = [new]  # shown in grey; final only at the endpoint
            self._drop_noise(units, had_text, out)
        self._update_tentative(out)
        return out
