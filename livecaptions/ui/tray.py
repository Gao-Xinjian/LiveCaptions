"""Tray / menu bar icon, and the app icon drawn in code (no image files)."""
import math
from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QAction, QColor, QFont, QIcon, QPainter, QPixmap, QPainterPath, QPen
from PySide6.QtWidgets import QMenu, QSystemTrayIcon
from livecaptions.ui.i18n import tr


def draw_icon(size, color):
    """Draw the icon (an audio waveform badge) at one size; also used for the packaged app's icon file."""
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor(color))
    p.drawRoundedRect(QRectF(0, 0, size, size), size * 0.20, size * 0.20)

    path = QPainterPath()
    path.moveTo(size * 0.10, size * 0.50)
    path.lineTo(size * 0.20, size * 0.50)
    path.cubicTo(size * 0.245, size * 0.50, size * 0.255, size * 0.22, size * 0.300, size * 0.22)
    path.cubicTo(size * 0.345, size * 0.22, size * 0.355, size * 0.78, size * 0.400, size * 0.78)
    path.cubicTo(size * 0.445, size * 0.78, size * 0.455, size * 0.12, size * 0.500, size * 0.12)
    path.cubicTo(size * 0.545, size * 0.12, size * 0.555, size * 0.90, size * 0.600, size * 0.90)
    path.cubicTo(size * 0.645, size * 0.90, size * 0.655, size * 0.38, size * 0.700, size * 0.38)
    path.cubicTo(size * 0.745, size * 0.38, size * 0.755, size * 0.50, size * 0.800, size * 0.50)
    path.lineTo(size * 0.90, size * 0.50)

    p.setBrush(Qt.BrushStyle.NoBrush)
    p.setPen(QPen(QColor('white'), size * 0.055, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
    p.drawPath(path)

    p.end()
    return pm


def make_icon(state='idle'):
    """Return the app icon; state 'running' and 'paused' change its colour."""
    color = {'running': "#409E65", 'paused': "#C69956"}.get(state, "#6484C3")
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
