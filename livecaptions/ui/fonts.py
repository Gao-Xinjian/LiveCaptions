"""One Chinese font next to the Latin one, so Qt does not take Chinese characters from whichever fonts have them."""
import sys

from PySide6.QtGui import QFont, QFontDatabase

CJK_FAMILIES = {   # Simplified Chinese UI fonts that come with the OS, in order of preference
    'win32': ['Microsoft YaHei UI', 'Microsoft YaHei'],
    'darwin': ['PingFang SC', 'Hiragino Sans GB', 'Heiti SC'],
}
CJK_FAMILIES_OTHER = ['Noto Sans CJK SC', 'Source Han Sans SC', 'WenQuanYi Micro Hei']
_cjk_family = None


def get_cjk_family():
    """Return the installed Simplified Chinese UI font, or '' if none is found (needs a Qt application)."""
    global _cjk_family
    if _cjk_family is None:
        installed = set(QFontDatabase.families())
        _cjk_family = next((f for f in CJK_FAMILIES.get(sys.platform, CJK_FAMILIES_OTHER) if f in installed), '')
    return _cjk_family


def add_cjk_fallback(font):
    """
    Return a copy of font that takes Chinese characters from one Chinese UI font.

    Without it Qt picks a font per character: on an English Windows some come from SimSun and others from the
    Japanese MS UI Gothic, so thick and thin characters mix.
    """
    out = QFont(font)
    families = list(font.families()) or [font.family()]
    cjk = get_cjk_family()
    if cjk and cjk not in families:
        out.setFamilies(families + [cjk])
    return out
