"""
LocalAgreement-n with recognition-driven buffer cuts (PLAN.md §6.3).

Every step the buffer (audio since the last cut) is transcribed again, with the terms and earlier final text
as prompt. Text that n runs in a row agree on becomes final. The buffer is cut at the first of: the end of a
fully final Whisper segment (or sentence) once the buffer is long, an endpoint (pause after recognised speech),
or the longest buffer (cut at its quietest point).
"""
from livecaptions import SAMPLE_RATE
from livecaptions.segmentation.base import PolicyOutput
from livecaptions.segmentation.buffered import BufferedPolicy
from livecaptions.segmentation.units import SENTENCE_END


def count_common_prefix(a, b):
    """Return how many units a and b share at their start (by key)."""
    n = 0
    while n < len(a) and n < len(b) and a[n].key == b[n].key:
        n += 1
    return n


class LocalAgreementPolicy(BufferedPolicy):
    def _agree(self, units, new, seg_ends, skip, out):
        """Commit what the last n runs agree on; return (cut sample, units before it) for trimming."""
        n_agree = max(2, self.cfg.stream.agree_n)
        if len(self.history) >= n_agree - 1:
            agreed = min(count_common_prefix(new, tail) for tail in self.history[-(n_agree - 1):])
        else:
            agreed = 0
        kept_offset = len(self.kept) - skip
        self._commit(new[:agreed], out)
        self.history = [tail[agreed:] for tail in self.history[-(n_agree - 2):]] if n_agree > 2 else []
        self.history.append(new[agreed:])
        # Whole Whisper segments whose text is all final can leave the buffer.
        cut, cut_units = -1, 0
        sentence_only = self.cfg.stream.cut_at == 'sentence'
        i = 0
        while i < len(units):
            j = i
            while j < len(units) and units[j].segment == units[i].segment:
                j += 1
            if j > skip + agreed:
                break
            if not sentence_only or units[j - 1].text.rstrip()[-1:] in SENTENCE_END:
                cut = int(seg_ends[units[i].segment] * SAMPLE_RATE)
                cut_units = max(0, j + kept_offset)
            i = j
        return cut, cut_units

    def process(self, chunk, vres, idle_s):
        sc = self.cfg.stream
        out = PolicyOutput()
        self._feed(chunk, vres)
        had_text = self.has_text()
        open_, endpoint = self._check_endpoint(idle_s)
        if endpoint:
            self._endpoint(out)
        elif self.get_buffer_s() >= self.get_max_buffer_s():
            self._cut_at_quietest(out)
        elif not open_:
            self._keep_preroll(out)
        elif self.ep.speech_since_run and self.get_new_s() >= self.get_step_s(sc.step_s):
            units, new, seg_ends, skip = self._run_new(out)
            cut, cut_units = self._agree(units, new, seg_ends, skip, out)
            if self.get_buffer_s() > min(sc.trim_after_s, self.get_max_buffer_s() - 2.0) and cut > self.buf_start:
                self._trim_to(cut, cut_units, out)
            self._drop_noise(units, had_text, out)
        self._update_tentative(out)
        return out
