"""
Global hotkeys. Windows: RegisterHotKey, so they work while another app has the focus.
Elsewhere: application shortcuts (they work while a LiveCaptions window is active).
"""
import ctypes as C
import sys

from PySide6.QtCore import QAbstractNativeEventFilter, QObject, Qt, Signal
from PySide6.QtGui import QKeySequence, QShortcut

WM_HOTKEY = 0x0312
MOD_ALT, MOD_CONTROL, MOD_SHIFT, MOD_WIN, MOD_NOREPEAT = 0x1, 0x2, 0x4, 0x8, 0x4000


def seq2win(seq_text):
    """Convert 'Ctrl+Alt+S' to (modifiers, virtual key) for RegisterHotKey, or None if not supported."""
    seq = QKeySequence(seq_text)
    if seq.isEmpty():
        return None
    combo = seq[0]
    key = combo.key()
    mods = combo.keyboardModifiers()
    win_mods = MOD_NOREPEAT
    if mods & Qt.KeyboardModifier.ControlModifier:
        win_mods |= MOD_CONTROL
    if mods & Qt.KeyboardModifier.AltModifier:
        win_mods |= MOD_ALT
    if mods & Qt.KeyboardModifier.ShiftModifier:
        win_mods |= MOD_SHIFT
    if mods & Qt.KeyboardModifier.MetaModifier:
        win_mods |= MOD_WIN
    k = key.value if hasattr(key, 'value') else int(key)
    if Qt.Key.Key_A.value <= k <= Qt.Key.Key_Z.value or Qt.Key.Key_0.value <= k <= Qt.Key.Key_9.value:
        vk = k  # same codes as ASCII
    elif Qt.Key.Key_F1.value <= k <= Qt.Key.Key_F24.value:
        vk = 0x70 + k - Qt.Key.Key_F1.value
    elif k == Qt.Key.Key_Space.value:
        vk = 0x20
    else:
        return None
    return win_mods, vk


class _WinFilter(QAbstractNativeEventFilter):
    def __init__(self, on_hotkey):
        super().__init__()
        self.on_hotkey = on_hotkey

    def nativeEventFilter(self, event_type, message):
        if event_type == b'windows_generic_MSG' or bytes(event_type) == b'windows_generic_MSG':
            from ctypes import wintypes
            msg = wintypes.MSG.from_address(int(message))
            if msg.message == WM_HOTKEY:
                self.on_hotkey(int(msg.wParam))
                return True, 0
        return False, 0


class HotkeyManager(QObject):
    triggered = Signal(str)  # action name

    def __init__(self, app, windows):
        """
        Args:
            app: the QApplication
            windows: widgets that get application shortcuts on systems without global hotkeys
        """
        super().__init__()
        self.app = app
        self.windows = windows
        self.ids = {}       # Windows hotkey id -> action
        self.shortcuts = []
        self.failed = []    # actions whose hotkey could not be registered
        self._filter = None
        if sys.platform == 'win32':
            self._filter = _WinFilter(self._on_win_hotkey)
            app.installNativeEventFilter(self._filter)

    def _on_win_hotkey(self, hk_id):
        if hk_id in self.ids:
            self.triggered.emit(self.ids[hk_id])

    def set_hotkeys(self, mapping):
        """Register {action: 'Ctrl+Alt+X'}; returns the actions that failed (key taken or unsupported)."""
        self.clear()
        for action, seq in mapping.items():
            if not seq:
                continue
            if sys.platform == 'win32':
                spec = seq2win(seq)
                hk_id = 0xA000 + len(self.ids)
                if spec and C.windll.user32.RegisterHotKey(None, hk_id, spec[0], spec[1]):
                    self.ids[hk_id] = action
                else:
                    self.failed.append(action)
            else:
                for w in self.windows:
                    sc = QShortcut(QKeySequence(seq), w)
                    sc.setContext(Qt.ShortcutContext.ApplicationShortcut)
                    sc.activated.connect(lambda a=action: self.triggered.emit(a))
                    self.shortcuts.append(sc)
        return list(self.failed)

    def clear(self):
        if sys.platform == 'win32':
            for hk_id in self.ids:
                C.windll.user32.UnregisterHotKey(None, hk_id)
        self.ids = {}
        for sc in self.shortcuts:
            sc.setEnabled(False)
            sc.deleteLater()
        self.shortcuts = []
        self.failed = []
