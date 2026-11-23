"""Tray / menu bar icon, and the app icon drawn in code (no image files)."""
from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QAction, QColor, QFont, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import QMenu, QSystemTrayIcon

from livecaptions.ui.i18n import tr


def draw_icon(size, color='#3A6FD8'):
    """Draw the icon (a 'CC' caption badge) at one size; also used for the packaged app's icon file."""
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor(color))
    p.drawRoundedRect(QRectF(0, size * 0.12, size, size * 0.76), size * 0.18, size * 0.18)
    font = QFont()
    font.setBold(True)
    font.setPixelSize(int(size * 0.42))
    p.setFont(font)
    p.setPen(QColor('white'))
    p.drawText(QRectF(0, 0, size, size), Qt.AlignmentFlag.AlignCenter, 'CC')
    p.end()
    return pm


def make_icon(state='idle'):
    """Return the app icon; state 'running' and 'paused' change its colour."""
    color = {'running': '#2E9E5B', 'paused': '#D08A1E'}.get(state, '#3A6FD8')
    icon = QIcon()
    for size in (16, 24, 32, 48, 64, 128, 256):
        icon.addPixmap(draw_icon(size, color))
    return icon


class Tray(QSystemTrayIcon):
    action = Signal(str)  # start_pause, stop, click, show_captions, show_main, open_folder, quit

    def __init__(self, parent=None):
        super().__init__(make_icon(), parent)
        self.menu = QMenu()
        self.acts = {}
        for name in ('start_pause', 'stop', None, 'click', 'show_captions', 'show_main', 'open_folder', None, 'quit'):
            if name is None:
                self.menu.addSeparator()
                continue
            act = QAction(self.menu)
            act.triggered.connect(lambda _=False, n=name: self.action.emit(n))
            self.menu.addAction(act)
            self.acts[name] = act
        self.acts['click'].setCheckable(True)
        self.setContextMenu(self.menu)
        self.activated.connect(self._on_activated)
        self.retranslate(False, False)

    def _on_activated(self, reason):
        if reason in (QSystemTrayIcon.ActivationReason.Trigger, QSystemTrayIcon.ActivationReason.DoubleClick):
            self.action.emit('show_main')

    def retranslate(self, running, paused):
        self.acts['start_pause'].setText(tr('Pause') if running and not paused else (tr('Resume') if paused else tr('Start')))
        self.acts['stop'].setText(tr('Stop'))
        self.acts['stop'].setEnabled(running)
        self.acts['click'].setText(tr('Click-through'))
        self.acts['show_captions'].setText(tr('Show / hide captions'))
        self.acts['show_main'].setText(tr('Show the main window'))
        self.acts['open_folder'].setText(tr('Open the session folder'))
        self.acts['quit'].setText(tr('Quit'))
        self.setToolTip('LiveCaptions')

    def set_state(self, running, paused):
        self.setIcon(make_icon('paused' if paused else 'running' if running else 'idle'))
        self.retranslate(running, paused)
