"""
Preview the app icon (draw_icon in livecaptions/ui/tray.py) at the sizes the OS uses, in the three states (idle,
running, paused), on light and dark backgrounds. Sizes up to 32 px are also shown enlarged, pixel by pixel.
The window redraws whenever tray.py is saved, so draw_icon can be edited while it is open; errors are shown in it.
"""
import argparse
import importlib
import os
import sys
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from PySide6.QtCore import QFileSystemWatcher, Qt, QTimer  # noqa: E402
from PySide6.QtGui import QColor, QFont, QPainter, QPixmap  # noqa: E402
from PySide6.QtWidgets import QApplication, QLabel, QScrollArea  # noqa: E402

import livecaptions.ui.tray as tray  # noqa: E402

SIZES = [16, 24, 32, 48, 64, 128, 256]
STATES = [('idle', '#3A6FD8'), ('running', '#2E9E5B'), ('paused', '#D08A1E')]
BACKGROUNDS = [('light', '#F0F0F0'), ('dark', '#202020')]
ZOOM = 4
PAD = 12


def make_sheet(draw_icon):
    """Return a pixmap with the icon at every size, state and background."""
    widths = [s + (s * ZOOM + PAD if s <= 32 else 0) + 2 * PAD for s in SIZES]
    row_h = max(SIZES) + 2 * PAD + 18
    sheet = QPixmap(sum(widths), row_h * len(STATES) * len(BACKGROUNDS))
    sheet.fill(QColor('white'))
    p = QPainter(sheet)
    font = QFont()
    font.setPixelSize(12)
    p.setFont(font)
    y = 0
    for state, color in STATES:
        for bg_name, bg in BACKGROUNDS:
            p.fillRect(0, y, sheet.width(), row_h, QColor(bg))
            p.setPen(QColor('#888888'))
            p.drawText(PAD, y + 14, f'{state} ({color}) on {bg_name}')
            x = 0
            for s, w in zip(SIZES, widths):
                pm = draw_icon(s, color)
                p.drawPixmap(x + PAD, y + 18 + PAD, pm)
                if s <= 32:
                    big = pm.scaled(s * ZOOM, s * ZOOM, Qt.AspectRatioMode.KeepAspectRatio,
                                    Qt.TransformationMode.FastTransformation)
                    p.drawPixmap(x + PAD + s + PAD, y + 18 + PAD, big)
                p.drawText(x + PAD, y + row_h - 4, f'{s} px')
                x += w
            y += row_h
    p.end()
    return sheet


class Preview(QScrollArea):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Icon preview: save livecaptions/ui/tray.py to redraw')
        self.label = QLabel()
        self.label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.setWidget(self.label)
        self.setWidgetResizable(True)
        self.watcher = QFileSystemWatcher([tray.__file__])
        self.watcher.fileChanged.connect(lambda _: QTimer.singleShot(200, self.reload))  # let the editor finish
        self.redraw()
        self.resize(min(1100, self.label.sizeHint().width() + 40), 900)

    def reload(self):
        if tray.__file__ not in self.watcher.files():   # editors that save by replacing the file
            self.watcher.addPath(tray.__file__)
        try:
            importlib.reload(tray)
        except Exception:
            self.show_error()
            return
        self.redraw()

    def redraw(self):
        try:
            self.label.setPixmap(make_sheet(tray.draw_icon))
            self.label.setStyleSheet('')
        except Exception:
            self.show_error()

    def show_error(self):
        self.label.setPixmap(QPixmap())
        self.label.setStyleSheet('color: #C00000; font-family: Consolas, Menlo, monospace;')
        self.label.setText(traceback.format_exc())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--save', default='', help='write the sheet to this PNG instead of opening a window (default: none)')
    parser.add_argument('--big', default='', help='also write the 1024 px icon (the packaged app\'s source) to this PNG (default: none)')
    args = parser.parse_args()
    if args.save:
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    app = QApplication(sys.argv[:1])
    if args.big:
        tray.draw_icon(1024).save(args.big)
    if args.save:
        make_sheet(tray.draw_icon).save(args.save)
        return 0
    win = Preview()
    win.show()
    return app.exec()


if __name__ == '__main__':
    sys.exit(main())
