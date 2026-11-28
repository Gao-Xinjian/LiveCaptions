"""Settings forms generated from the schema metadata (config/schema.py)."""
import dataclasses

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QKeySequence
from PySide6.QtWidgets import (QCheckBox, QColorDialog, QComboBox, QDoubleSpinBox, QFileDialog, QFontComboBox,
                               QFormLayout, QGridLayout, QGroupBox, QHBoxLayout, QKeySequenceEdit, QLabel, QLineEdit,
                               QPlainTextEdit, QPushButton, QSpinBox, QToolButton, QVBoxLayout, QWidget)

from livecaptions.ui.i18n import tr


class ColorButton(QPushButton):
    changed = Signal(str)

    def __init__(self, value):
        super().__init__()
        self.value = value
        self.clicked.connect(self._pick)
        self._paint()

    def _paint(self):
        self.setText(self.value)
        self.setStyleSheet(f'QPushButton {{ background: {self.value}; color: '
                           f'{"#000" if QColor(self.value).lightness() > 128 else "#fff"}; }}')

    def _pick(self):
        c = QColorDialog.getColor(QColor(self.value), self, tr('Choose a colour'))
        if c.isValid():
            self.value = c.name()
            self._paint()
            self.changed.emit(self.value)


class PathEdit(QWidget):
    changed = Signal(str)

    def __init__(self, value, is_dir, placeholder=''):
        super().__init__()
        self.is_dir = is_dir
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.edit = QLineEdit(value)
        self.edit.setPlaceholderText(placeholder)
        self.edit.editingFinished.connect(lambda: self.changed.emit(self.edit.text().strip()))
        btn = QPushButton(tr('Browse…'))
        btn.clicked.connect(self._browse)
        lay.addWidget(self.edit, 1)
        lay.addWidget(btn)

    def _browse(self):
        if self.is_dir:
            path = QFileDialog.getExistingDirectory(self, tr('Choose a folder'), self.edit.text())
        else:
            path, _ = QFileDialog.getOpenFileName(self, tr('Choose a file'), self.edit.text())
        if path:
            self.edit.setText(path)
            self.changed.emit(path)


class LinesEdit(QPlainTextEdit):
    changed = Signal(list)

    def __init__(self, lines):
        super().__init__('\n'.join(lines))
        self.setFixedHeight(110)
        self.textChanged.connect(lambda: self.changed.emit([s for s in self.toPlainText().splitlines() if s.strip()]))


class MultiCheck(QWidget):
    changed = Signal(list)

    def __init__(self, choices, values, columns=2):
        super().__init__()
        lay = QGridLayout(self)   # a few columns, not one long row that would widen the window
        lay.setContentsMargins(0, 0, 0, 0)
        self.boxes = []
        for i, (val, label) in enumerate(choices):
            cb = QCheckBox(tr(label))
            cb.setChecked(val in values)
            cb.toggled.connect(self._emit)
            self.boxes.append((val, cb))
            lay.addWidget(cb, i // columns, i % columns)
        lay.setColumnStretch(columns, 1)

    def _emit(self):
        self.changed.emit([v for v, cb in self.boxes if cb.isChecked()])


def allow_shrink(combo, chars=10):
    """
    Let a combo box shrink below its longest entry when the window is narrow (the closed box then cuts the text
    short; the open list still shows it whole). It keeps its natural width when there is room.
    """
    combo.setMinimumWidth(min(combo.sizeHint().width(), combo.fontMetrics().averageCharWidth() * chars + 40))
    return combo


def make_combo(choices, value, editable=False):
    """Return a QComboBox with (value, label) choices and value selected."""
    combo = QComboBox()
    combo.setEditable(editable)
    for val, label in choices:
        combo.addItem(tr(label), val)
    idx = combo.findData(value)
    if idx >= 0:
        combo.setCurrentIndex(idx)
    elif editable:
        combo.setEditText(str(value))
    return allow_shrink(combo)


class SectionForm(QWidget):
    """The options of one settings section; writes every change into the dataclass and emits changed(name)."""
    changed = Signal(str, str)  # section, option

    def __init__(self, section, obj, special=None, title=''):
        """
        Args:
            section: section name in Config (e.g. 'audio')
            obj: the section dataclass, edited in place
            special: {kind: factory(form, field, value) -> QWidget} for kinds such as 'device' or 'model'
            title: group title
        """
        super().__init__()
        self.section = section
        self.obj = obj
        self.special = special or {}
        self.rows = {}  # option -> (label widget, editor)
        self.tips = {}  # option -> its translated tooltip
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        box = QGroupBox(title) if title else QWidget()
        self.form = QFormLayout(box)
        self.form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        outer.addWidget(box)
        self.adv_box = QWidget()
        self.adv_form = QFormLayout(self.adv_box)
        self.adv_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        self.adv_btn = QToolButton()
        self.adv_btn.setText(tr('Advanced options'))
        self.adv_btn.setCheckable(True)
        self.adv_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.adv_btn.setArrowType(Qt.ArrowType.RightArrow)
        self.adv_btn.toggled.connect(self._toggle_adv)
        self.adv_box.setVisible(False)
        n_adv = 0
        for f in dataclasses.fields(obj):
            meta = f.metadata
            if meta.get('kind') == 'hidden' or not meta.get('label'):
                continue
            editor = self._make_editor(f, getattr(obj, f.name))
            if editor is None:
                continue
            tip = tr(meta.get('tip') or '')
            label = QLabel(tr(meta['label']) if not isinstance(editor, QCheckBox) else '')
            for w in (label, editor):
                w.setToolTip(tip)
            target = self.adv_form if meta.get('advanced') else self.form
            n_adv += bool(meta.get('advanced'))
            target.addRow(label, editor)
            self.rows[f.name] = (label, editor)
            self.tips[f.name] = tip
        if n_adv:
            outer.addWidget(self.adv_btn)
            outer.addWidget(self.adv_box)

    def _toggle_adv(self, on):
        self.adv_box.setVisible(on)
        self.adv_btn.setArrowType(Qt.ArrowType.DownArrow if on else Qt.ArrowType.RightArrow)

    def _set(self, name, value):
        if getattr(self.obj, name) != value:
            setattr(self.obj, name, value)
            self.changed.emit(self.section, name)

    def _make_editor(self, f, value):
        meta, name = f.metadata, f.name
        kind = meta.get('kind')
        if kind in self.special:
            return self.special[kind](self, f, value)
        if isinstance(value, bool):
            w = QCheckBox(tr(meta['label']))
            w.setChecked(value)
            w.toggled.connect(lambda v, n=name: self._set(n, bool(v)))
            return w
        if isinstance(value, int):
            w = QSpinBox()
            w.setRange(meta['lo'] if meta['lo'] is not None else -10 ** 9, meta['hi'] if meta['hi'] is not None else 10 ** 9)
            w.setSingleStep(int(meta['step'] or 1))
            if meta.get('unit'):
                w.setSuffix(' ' + tr(meta['unit']))
            w.setValue(value)
            w.valueChanged.connect(lambda v, n=name: self._set(n, int(v)))
            return w
        if isinstance(value, float):
            w = QDoubleSpinBox()
            w.setDecimals(meta['decimals'] if meta.get('decimals') is not None else 2)
            w.setRange(meta['lo'] if meta['lo'] is not None else -1e9, meta['hi'] if meta['hi'] is not None else 1e9)
            w.setSingleStep(meta['step'] or 0.1)
            if meta.get('unit'):
                w.setSuffix(' ' + tr(meta['unit']))
            w.setValue(value)
            w.valueChanged.connect(lambda v, n=name: self._set(n, float(v)))
            return w
        if isinstance(value, list):
            if kind == 'multi':
                w = MultiCheck(meta['choices'], value)
            else:
                w = LinesEdit(value)
            w.changed.connect(lambda v, n=name: self._set(n, list(v)))
            return w
        if meta.get('choices'):
            w = make_combo(meta['choices'], value, editable=kind == 'lang')
            if kind == 'lang':  # a listed language, or any code typed in
                def on_text(t, c=w, n=name):
                    idx = c.findText(t)
                    self._set(n, c.itemData(idx) if idx >= 0 else t.strip().lower())
                w.currentTextChanged.connect(on_text)
            else:
                w.currentIndexChanged.connect(lambda _, c=w, n=name: self._set(n, c.currentData()))
            return w
        if kind == 'color':
            w = ColorButton(value)
            w.changed.connect(lambda v, n=name: self._set(n, v))
            return w
        if kind in ('dir', 'file'):
            w = PathEdit(value, kind == 'dir', tr('Default'))
            w.changed.connect(lambda v, n=name: self._set(n, v))
            return w
        if kind == 'font':
            w = QFontComboBox()
            w.setEditable(True)
            if value:
                w.setCurrentText(value)
            else:
                w.setCurrentIndex(-1)
                w.lineEdit().setPlaceholderText(tr('Default'))
            w.currentTextChanged.connect(lambda v, n=name: self._set(n, v))
            return allow_shrink(w, 16)
        if kind == 'hotkey':
            w = QKeySequenceEdit(QKeySequence(value))
            w.setClearButtonEnabled(True)
            w.editingFinished.connect(lambda e=w, n=name: self._set(n, e.keySequence().toString(QKeySequence.SequenceFormat.PortableText)))
            return w
        w = QLineEdit(str(value))
        w.editingFinished.connect(lambda e=w, n=name: self._set(n, e.text()))
        return w

    def set_scope(self, backend, policy):
        """Grey out options that do not apply to this backend or streaming policy."""
        for name, (label, editor) in self.rows.items():
            meta = next(x for x in dataclasses.fields(self.obj) if x.name == name).metadata
            backends, policies = meta.get('backends'), meta.get('policies')
            tip = self.tips[name]
            if backends is not None and backend not in backends:
                tip = tr('Not used by this backend')
            elif policies is not None and policy not in policies:
                tip = tr('Not used by this streaming policy')
            ok = tip == self.tips[name]
            for w in (label, editor):
                w.setEnabled(ok)
                w.setToolTip(tip)

    def set_row_enabled(self, name, on):
        if name in self.rows:
            for w in self.rows[name]:
                w.setEnabled(on)

    def set_row_visible(self, name, on):
        if name in self.rows:
            for w in self.rows[name]:
                w.setVisible(on)
