"""Split recognised text into units (a CJK character or a word) that runs are compared on."""
from dataclasses import dataclass

SPACES = ' \t\r\n'
SENTENCE_END = '。！？!?.;；…'


@dataclass
class Unit:
    text: str       # as shown: with punctuation, and a leading space before a word
    key: str        # what runs are compared on: lower case, without punctuation
    t0: float       # seconds into the session
    t1: float
    segment: int = 0  # which Whisper segment of its run it came from


def is_cjk(c):
    """Return whether a character is Chinese, Japanese or Korean."""
    o = ord(c)
    return (0x3040 <= o <= 0x30FF or 0x3400 <= o <= 0x9FFF or 0xAC00 <= o <= 0xD7AF or 0xF900 <= o <= 0xFAFF
            or 0x20000 <= o <= 0x2FFFF)


def is_kana(c):
    return 0x3040 <= ord(c) <= 0x30FF


def _utf8_len(lead):
    """Return the byte length of a UTF-8 character from its first byte (1 for invalid bytes)."""
    if lead < 0x80:
        return 1
    if lead >> 5 == 0x6:
        return 2
    if lead >> 4 == 0xE:
        return 3
    if lead >> 3 == 0x1E:
        return 4
    return 1


def decode_chars(data, times):
    """Yield (char, t0, t1, byte index) for UTF-8 bytes whose byte i has the time times[i] = (t0, t1)."""
    i = 0
    while i < len(data):
        n = _utf8_len(data[i])
        if i + n > len(data):
            break  # a character cut off at the end
        ch = bytes(data[i:i + n]).decode('utf-8', errors='ignore')
        if ch:
            yield ch, times[i][0], times[i + n - 1][1], i
        i += n


def segments2units(segments, offset):
    """
    Split a run's segments into units.

    Args:
        segments: list of engines.Segment (times relative to the transcribed audio)
        offset: where that audio starts in the session, s

    Returns:
        list of Unit with session times
    """
    data = bytearray()
    times = []
    seg_of_byte = []
    for k, seg in enumerate(segments):
        for piece in seg.pieces:
            data += piece.data
            times += [(offset + piece.t0, offset + piece.t1)] * len(piece.data)
            seg_of_byte += [k] * len(piece.data)
    units = []
    pending = ''  # punctuation waiting for the next unit
    space = False
    in_word = False
    for ch, t0, t1, i_byte in decode_chars(data, times):
        seg_idx = seg_of_byte[i_byte]
        if ch in SPACES:
            space, in_word = True, False
            continue
        cjk = is_cjk(ch)
        word = not cjk and ch.isalnum()
        if cjk or (word and not in_word):  # a new unit
            text = (' ' if space and not cjk else '') + pending + ch
            units.append(Unit(text, ch.lower() if word else ch, t0, t1, seg_idx))
            pending = ''
            in_word = word
        elif word:  # the rest of a word
            units[-1].text += ch
            units[-1].key += ch.lower()
            units[-1].t1 = t1
        elif units and not space:  # punctuation belongs to what it follows
            units[-1].text += ch
            units[-1].t1 = max(units[-1].t1, t1)
        else:
            pending += ch
            continue  # keep `space` for the unit that follows
        space = False
    return units


def units2text(units, strip=True):
    """Join units to text."""
    text = ''.join(u.text for u in units)
    return text.strip() if strip else text


def kept_prefix(hyp, kept):
    """
    Return how many units at the start of a run repeat the kept (already final) text.

    Whisper may hear that part a little differently each time, so its end is found by text,
    not by the rough token timestamps.
    """
    def same(a, b):
        return a.key == b.key

    n = 0
    while n < len(hyp) and n < len(kept) and same(hyp[n], kept[n]):
        n += 1
    if n == len(kept):
        return n  # the usual case

    def find(m):
        """Return where the last m kept units reappear in hyp, closest to where they should be, or -1."""
        if m > len(kept):
            return -1
        tail = kept[len(kept) - m:]
        for d in range(7):
            for p in (len(kept) - d, len(kept) + d):
                if m <= p <= len(hyp) and all(same(a, b) for a, b in zip(hyp[p - m:p], tail)):
                    return p
        return -1

    for m in (3, 2):
        p = find(m)
        if p >= 0:
            return p
    if len(hyp) >= len(kept) and n + 4 >= len(kept):
        return len(kept)  # only the last few units differ
    p = find(1)
    if p >= 0:
        return p
    return min(len(kept), len(hyp))  # no landmark: assume the same length
