"""
Interface translation: tr() wraps QCoreApplication.translate, and a dictionary-backed QTranslator supplies the
Chinese text (no Qt Linguist tools needed). Strings missing from a table fall back to English.
"""
from PySide6.QtCore import QCoreApplication, QTranslator

CONTEXT = 'LiveCaptions'
_translator = None


def tr(text):
    """Translate an interface string into the current language."""
    return QCoreApplication.translate(CONTEXT, text) if text else text


class DictTranslator(QTranslator):
    def __init__(self, table):
        super().__init__()
        self.table = table

    def translate(self, context, source, disambiguation=None, n=-1):
        return self.table.get(source, source)  # PySide turns '' into an empty translation, not "missing"

    def isEmpty(self):
        return not self.table


def get_table(lang):
    """Return the translation table of a language ({} for English)."""
    if lang == 'zh_CN':
        from livecaptions.ui.i18n.zh_CN import TABLE
        return TABLE
    return {}


def install_language(app, lang):
    """Switch the interface language; widgets then get a LanguageChange event."""
    global _translator
    if _translator is not None:
        app.removeTranslator(_translator)
        _translator = None
    table = get_table(lang)
    if table:
        _translator = DictTranslator(table)
        app.installTranslator(_translator)
