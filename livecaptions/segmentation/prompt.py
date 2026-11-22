"""Whisper's prompt: course terms as one line, then recent final text (PLAN.md §8)."""
import re

from livecaptions.segmentation.units import is_cjk

TERM_SEPARATORS = r'[\s,，、;；]+'


def split_terms(text):
    """Split a terms text into words (separated by spaces, commas, 、 or semicolons)."""
    return [w for w in re.split(TERM_SEPARATORS, text) if w]


def terms2line(words):
    """Write terms as one short line, e.g. "突触、递质、抑制。", which reads to Whisper like earlier text."""
    if not words:
        return ''
    cjk = any(is_cjk(c) for w in words for c in w)
    return ('、' if cjk else ', ').join(words) + ('。' if cjk else '.')


def get_term_key(term):
    """Return a term as units compare it: lower case, without punctuation."""
    return ''.join(c.lower() for c in term if is_cjk(c) or c.isalnum())


class PromptBuilder:
    """Builds prompt tokens; the terms are tokenised only when they change."""

    def __init__(self, engine):
        self.engine = engine
        self.term_tokens = []
        self.term_keys = []
        self.terms_cut = False  # the terms did not all fit
        self._terms_text = None
        self._term_cfg = None

    def set_terms(self, text, terms_cfg, translating):
        """Tokenise the terms; terms beyond the token budget are left out, whole words at a time."""
        key = (text, terms_cfg.use_terms, terms_cfg.term_tokens, translating)
        if key == self._term_cfg:
            return
        self._term_cfg = key
        words = split_terms(text) if terms_cfg.use_terms and not translating else []
        self.term_keys = [k for k in (get_term_key(w) for w in words) if k]
        self.terms_cut = False
        self.term_tokens = []
        while words:
            tokens = self.engine.tokenize(terms2line(words))
            if len(tokens) <= terms_cfg.term_tokens:
                self.term_tokens = tokens
                break
            words = words[:-1]
            self.terms_cut = True

    def build(self, context, terms_cfg):
        """Return terms tokens + as much recent text as fits in the prompt."""
        tokens = list(self.term_tokens)
        if terms_cfg.use_context and terms_cfg.context_chars > 0 and context:
            recent = self.engine.tokenize(context[-terms_cfg.context_chars:])
            room = max(0, terms_cfg.prompt_tokens - len(tokens))
            tokens += recent[-room:] if room else []
        return tokens[:terms_cfg.prompt_tokens] if terms_cfg.prompt_tokens > 0 else []
