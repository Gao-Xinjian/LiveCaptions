"""Caption display: final text is appended, only the grey unfinished tail is redrawn."""
from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtGui import QColor, QFont, QFontMetricsF, QPainter, QTextBlockFormat, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import QFrame, QMenu, QPlainTextEdit, QSizeGrip, QVBoxLayout, QWidget

from livecaptions.ui.i18n import tr


def get_scheme_colors(cc):
    """Return (text, background, unfinished text) colours of a CaptionCfg."""
    if cc.scheme == 'dark':
        return QColor('#FFFFFF'), QColor('#000000'), QColor('#9A9A9A')
    if cc.scheme == 'light':
        return QColor('#000000'), QColor('#FFFFFF'), QColor('#707070')
    return QColor(cc.fg), QColor(cc.bg), QColor(cc.tentative_fg)


def get_utf16_len(text):
    """Return the length of text in UTF-16 code units, which QTextDocument positions count."""
    return len(text.encode('utf-16-le')) // 2


class CaptionView(QPlainTextEdit):
    """Read-only text with a transparent background; committed text is append-only."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.setReadOnly(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)  # the wheel still scrolls back
        self.viewport().setAutoFillBackground(False)
        self.setStyleSheet('QPlainTextEdit { background: transparent; }')
        self._tent_len = 0  # UTF-16 length of the unfinished tail at the end of the document
        self._tent_text = ''
        self.fmt_final = QTextCharFormat()
        self.fmt_tent = QTextCharFormat()
        self.block_fmt = QTextBlockFormat()
        self.apply_style()

    def apply_style(self):
        cc = self.cfg.caption
        fg, _, tent = get_scheme_colors(cc)
        fg.setAlphaF(cc.text_opacity)
        tent.setAlphaF(cc.text_opacity)
        font = QFont(cc.font_family) if cc.font_family else QFont(self.font().family())
        font.setPointSize(cc.font_size)
        font.setBold(cc.bold)
        self.setFont(font)
        self.fmt_final = QTextCharFormat()
        self.fmt_final.setForeground(fg)
        self.fmt_final.setFont(font)
        self.fmt_tent = QTextCharFormat(self.fmt_final)
        self.fmt_tent.setForeground(tent)
        self.fmt_tent.setFontItalic(cc.tentative_italic)
        self.block_fmt = QTextBlockFormat()
        self.block_fmt.setLineHeight(cc.line_spacing * 100, QTextBlockFormat.LineHeightTypes.ProportionalHeight.value)
        self.block_fmt.setAlignment(Qt.AlignmentFlag.AlignHCenter if cc.align == 'center' else Qt.AlignmentFlag.AlignLeft)
        self.document().setMaximumBlockCount(max(10, cc.scrollback))
        # Restyle what is already there.
        cur = QTextCursor(self.document())
        cur.select(QTextCursor.SelectionType.Document)
        cur.mergeBlockFormat(self.block_fmt)
        final_fmt = QTextCharFormat(self.fmt_final)
        cur.setPosition(0)
        cur.setPosition(max(0, self._end() - self._tent_len), QTextCursor.MoveMode.KeepAnchor)
        cur.mergeCharFormat(final_fmt)
        self._replace_tentative(self._tent_text)

    def get_line_height(self):
        """Return the height of one caption line in pixels."""
        return QFontMetricsF(self.font()).lineSpacing() * self.cfg.caption.line_spacing

    def _end(self):
        return max(0, self.document().characterCount() - 1)

    def _at_bottom(self):
        sb = self.verticalScrollBar()
        return sb.value() >= sb.maximum() - 2

    def _replace_tentative(self, text):
        cur = QTextCursor(self.document())
        cur.setPosition(max(0, self._end() - self._tent_len))
        cur.movePosition(QTextCursor.MoveOperation.End, QTextCursor.MoveMode.KeepAnchor)
        cur.removeSelectedText()
        cur.insertText(text, self.fmt_tent)
        self._tent_len = get_utf16_len(text)
        self._tent_text = text

    def add_committed(self, text):
        """Append final text; '\n' starts a new line."""
        follow = self._at_bottom()
        self._replace_tentative('')
        cur = QTextCursor(self.document())
        cur.movePosition(QTextCursor.MoveOperation.End)
        parts = text.split('\n')
        for i, part in enumerate(parts):
            if i:
                cur.insertBlock(self.block_fmt, self.fmt_final)
            if part:
                if cur.block().length() <= 1:  # a new line starts without the space between words
                    part = part.lstrip()
                cur.insertText(part, self.fmt_final)
        self._tent_len = 0
        if follow:
            self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())

    def set_tentative(self, text):
        follow = self._at_bottom()
        cur = QTextCursor(self.document())
        cur.movePosition(QTextCursor.MoveOperation.End)
        if cur.block().length() <= 1:
            text = text.lstrip()
        self._replace_tentative(text)
        if follow:
            self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())

    def clear_all(self):
        self.clear()
        self._tent_len = 0
        self._tent_text = ''
        cur = QTextCursor(self.document())
        cur.setBlockFormat(self.block_fmt)


class CaptionFrame(QWidget):
    """Paints the rounded translucent background behind a CaptionView."""

    def __init__(self, cfg, view):
        super().__init__()
        self.cfg = cfg
        self.lay = QVBoxLayout(self)
        self.lay.setContentsMargins(14, 8, 14, 8)
        self.lay.addWidget(view)

    def paintEvent(self, event):
        cc = self.cfg.caption
        _, bg, _ = get_scheme_colors(cc)
        bg.setAlphaF(cc.bg_opacity)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(bg)
        p.drawRoundedRect(self.rect(), 10, 10)


class CaptionWindow(QWidget):
    """Frameless floating caption window: drag anywhere, resize at the corner, right-click for options."""
    toggled_click_through = Signal(bool)
    toggled_on_top = Signal(bool)
    dock_requested = Signal()
    geometry_changed = Signal(list)

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_MacAlwaysShowToolWindow)  # macOS hides tool windows of inactive apps
        self.setWindowTitle(tr('Captions'))
        self.frame = CaptionFrame(cfg, QWidget())
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.frame)
        self.grip = QSizeGrip(self)
        self.grip.resize(16, 16)
        self.view = None
        self._drag = None
        self.apply_flags()
        geo = cfg.caption.geometry
        if len(geo) == 4 and geo[2] > 50 and geo[3] > 30:
            self.setGeometry(*geo)
        else:
            self.resize(900, 140)

    def set_view(self, view):
        """Put the shared CaptionView into this window."""
        old = self.frame.lay.itemAt(0).widget() if self.frame.lay.count() else None
        if old is not None and old is not view:
            self.frame.lay.removeWidget(old)
            if not isinstance(old, CaptionView):
                old.deleteLater()
        self.view = view
        self.frame.lay.addWidget(view)
        view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        view.customContextMenuRequested.connect(self._menu)
        view.viewport().installEventFilter(self)
        view.show()

    def apply_flags(self):
        """Apply always-on-top and click-through (re-shows the window, as Qt requires)."""
        cc = self.cfg.caption
        visible = self.isVisible()
        flags = Qt.WindowType.Window | Qt.WindowType.FramelessWindowHint | Qt.WindowType.Tool
        if cc.on_top:
            flags |= Qt.WindowType.WindowStaysOnTopHint
        if cc.click_through:
            flags |= Qt.WindowType.WindowTransparentForInput
        self.setWindowFlags(flags)
        self.grip.setVisible(not cc.click_through)
        if visible:
            self.show()

    def fit_lines(self):
        """Resize the height to show the configured number of lines."""
        if self.view is None:
            return
        m = self.frame.lay.contentsMargins()
        h = int(self.view.get_line_height() * self.cfg.caption.max_lines + m.top() + m.bottom() + 12)
        self.resize(self.width(), h)

    def resizeEvent(self, event):
        self.grip.move(self.width() - self.grip.width(), self.height() - self.grip.height())
        super().resizeEvent(event)

    def moveEvent(self, event):
        super().moveEvent(event)
        self._emit_geometry()

    def _emit_geometry(self):
        g = self.geometry()
        self.geometry_changed.emit([g.x(), g.y(), g.width(), g.height()])

    def eventFilter(self, obj, event):
        # Drag the window with the left button anywhere on the text; selecting text needs Shift.
        t = event.type()
        if t == event.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton \
                and not (event.modifiers() & Qt.KeyboardModifier.ShiftModifier):
            if self.windowHandle() is not None and self.windowHandle().startSystemMove():
                return True
            self._drag = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            return True
        if t == event.Type.MouseMove and self._drag is not None:
            self.move(event.globalPosition().toPoint() - self._drag)
            return True
        if t == event.Type.MouseButtonRelease and self._drag is not None:
            self._drag = None
            return True
        return super().eventFilter(obj, event)

    def _menu(self, pos):
        cc = self.cfg.caption
        menu = QMenu(self)
        a_top = menu.addAction(tr('Always on top'))
        a_top.setCheckable(True)
        a_top.setChecked(cc.on_top)
        a_click = menu.addAction(tr('Click-through'))
        a_click.setCheckable(True)
        a_click.setChecked(cc.click_through)
        a_dock = menu.addAction(tr('Show captions inside the main window'))
        menu.addSeparator()
        a_copy = menu.addAction(tr('Copy all'))
        a_clear = menu.addAction(tr('Clear'))
        act = menu.exec(self.view.mapToGlobal(pos) if self.view else self.mapToGlobal(QPoint(0, 0)))
        if act is a_top:
            self.toggled_on_top.emit(a_top.isChecked())
        elif act is a_click:
            self.toggled_click_through.emit(a_click.isChecked())
        elif act is a_dock:
            self.dock_requested.emit()
        elif act is a_copy and self.view:
            self.view.selectAll()
            self.view.copy()
            self.view.moveCursor(QTextCursor.MoveOperation.End)
        elif act is a_clear and self.view:
            self.view.clear_all()
