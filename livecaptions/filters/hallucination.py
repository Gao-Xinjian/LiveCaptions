"""Filters against Whisper hallucinations: known phrases, unlikely segments, repetitions, terms read from noise."""
import re

from livecaptions.segmentation.units import is_kana


def is_unlikely_speech(seg, filter_cfg):
    """Return whether a segment is probably not speech (high no-speech probability and low confidence)."""
    return (filter_cfg.no_speech_on and seg.no_speech_prob > filter_cfg.no_speech_prob
            and seg.avg_logprob < filter_cfg.avg_logprob)


def is_blacklisted(text, filter_cfg):
    """Return whether a run's text matches a known hallucination."""
    if not filter_cfg.blacklist_on:
        return False
    text = text.strip()
    if text in filter_cfg.exact:
        return True
    for phrase in filter_cfg.blacklist:
        phrase = phrase.strip()
        if not phrase:
            continue
        if filter_cfg.blacklist_regex:
            try:
                if re.search(phrase, text):
                    return True
            except re.error:
                continue
        elif phrase.lower() in text.lower():
            return True
    return False


def cut_repetitions(units, max_repeats, max_ngram=8):
    """Cut units where an n-gram repeats back to back more than max_repeats times (a decoding loop)."""
    keys = [u.key for u in units]
    cut = len(units)
    for n in range(1, max_ngram + 1):
        i = 0
        while i + n <= len(keys) and i < cut:
            reps = 1
            while i + (reps + 1) * n <= len(keys) and keys[i + reps * n:i + (reps + 1) * n] == keys[i:i + n]:
                reps += 1
            if reps > max_repeats:
                cut = min(cut, i + n * max_repeats)
                break
            i += 1
    return units[:cut]


def drop_kana(units):
    """Remove units made only of Japanese kana (Whisper misreading noise in Chinese)."""
    return [u for u in units if not (u.key and all(is_kana(c) for c in u.key))]


def is_too_dense(units, seconds, filter_cfg):
    """Return whether a run has implausibly many units for its audio length."""
    return filter_cfg.rate_on and seconds > 0.5 and len(units) / seconds > filter_cfg.max_units_per_s


def is_terms_only(units, term_keys):
    """Return whether units spell nothing but course terms, e.g. "抑制。"."""
    text = ''.join(u.key for u in units)
    if not text or not term_keys:
        return False
    pos = 0
    while pos < len(text):
        best = max((len(k) for k in term_keys if text.startswith(k, pos)), default=0)
        if best == 0:
            return False
        pos += best
    return True
