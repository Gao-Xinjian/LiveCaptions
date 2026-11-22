"""Text post-processing: Traditional to Simplified Chinese."""
import sys

_converter = None
_converter_tried = False


def _get_converter():
    """Return a function converting Traditional to Simplified Chinese, or None if no converter is available."""
    global _converter, _converter_tried
    if _converter_tried:
        return _converter
    _converter_tried = True
    try:
        import opencc
        cc = opencc.OpenCC('t2s')
        _converter = cc.convert
    except Exception:
        if sys.platform == 'win32':
            _converter = _lcmap_t2s
    return _converter


def _lcmap_t2s(text):
    """Convert with the Windows API (fallback when OpenCC is not installed)."""
    import ctypes as C
    LCMAP_SIMPLIFIED_CHINESE = 0x02000000
    fn = C.windll.kernel32.LCMapStringEx
    n = fn('zh-CN', LCMAP_SIMPLIFIED_CHINESE, text, len(text), None, 0, None, None, 0)
    if n <= 0:
        return text
    buf = C.create_unicode_buffer(n)
    fn('zh-CN', LCMAP_SIMPLIFIED_CHINESE, text, len(text), buf, n, None, None, 0)
    return buf.value[:n]


def to_simplified(text):
    """Return text in Simplified Chinese characters (unchanged if no converter is available)."""
    conv = _get_converter()
    return conv(text) if conv and text else text
