"""Model selection (installed models, browse) and the download dialog."""
import os
import shutil
import threading

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QDialog, QFileDialog, QHBoxLayout, QHeaderView, QLabel,
                               QMessageBox, QProgressBar, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout,
                               QWidget)

from livecaptions.config.store import get_models_dir
from livecaptions.models import manager
from livecaptions.ui.i18n import tr


def fmt_bytes(n):
    """Format a size in bytes as MB or GB."""
    if n >= 1e9:
        return f'{n / 1e9:.2f} GB'
    return f'{n / 1e6:.1f} MB' if n < 1e7 else f'{n / 1e6:.0f} MB'


class ModelPicker(QWidget):
    """Combo of installed models of the current backend, with Browse, Download and Delete."""
    changed = Signal(str)

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        self.combo = QComboBox()
        self.combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.combo.currentIndexChanged.connect(self._on_pick)
        b_browse = QPushButton(tr('Browse…'))
        b_browse.clicked.connect(self._browse)
        b_dl = QPushButton(tr('Download…'))
        b_dl.clicked.connect(self._download)
        b_del = QPushButton(tr('Delete'))
        b_del.clicked.connect(self._delete)
        row.addWidget(self.combo, 1)
        for b in (b_browse, b_dl, b_del):
            row.addWidget(b)
        lay.addLayout(row)
        self.info = QLabel()
        self.info.setWordWrap(True)
        lay.addWidget(self.info)
        self.refresh()

    def refresh(self):
        """Fill the combo with installed models of the backend, plus the current path if it is elsewhere."""
        self.combo.blockSignals(True)
        self.combo.clear()
        backend = self.cfg.model.backend
        for m in manager.list_installed(get_models_dir(self.cfg)):
            if m.backend == backend:
                self.combo.addItem(f'{m.name}  ({fmt_bytes(m.size)})', m.path)
        cur = self.cfg.model.path
        if cur and self.combo.findData(cur) < 0:
            self.combo.addItem(os.path.basename(os.path.normpath(cur)) or cur, cur)
        idx = self.combo.findData(cur)
        if idx >= 0:
            self.combo.setCurrentIndex(idx)
        elif self.combo.count():
            self.combo.setCurrentIndex(-1)
        self.combo.blockSignals(False)
        self._show_info()

    def _show_info(self):
        path = self.cfg.model.path
        if not path:
            self.info.setText(tr('No model selected. Download one, or browse to a model file or folder.'))
            return
        fmt = manager.detect_format(path)
        if fmt is None:
            self.info.setText(tr('Not a model this app can read:') + ' ' + path)
        elif fmt == 'pt':
            self.info.setText(tr('openai-whisper .pt models are not supported yet.'))
        elif fmt != self.cfg.model.backend:
            self.info.setText(tr('This model is for the other backend:') + f' {fmt}')
        else:
            self.info.setText(path)

    def _on_pick(self, idx):
        path = self.combo.itemData(idx) if idx >= 0 else ''
        if path and path != self.cfg.model.path:
            self.cfg.model.path = path
            self._show_info()
            self.changed.emit(path)

    def _browse(self):
        start = get_models_dir(self.cfg)
        if self.cfg.model.backend == 'fasterwhisper':
            path = QFileDialog.getExistingDirectory(self, tr('Choose a CTranslate2 model folder'), start)
        else:
            path, _ = QFileDialog.getOpenFileName(self, tr('Choose a ggml model file'), start, 'ggml (*.bin);;*')
        if path:
            self.cfg.model.path = os.path.normpath(path)
            self.refresh()
            self.changed.emit(self.cfg.model.path)

    def _download(self):
        dlg = DownloadDialog(self.cfg, self)
        dlg.exec()
        if dlg.installed_path and manager.detect_format(dlg.installed_path) == self.cfg.model.backend:
            self.cfg.model.path = dlg.installed_path
            self.changed.emit(self.cfg.model.path)
        self.refresh()

    def _delete(self):
        path = self.combo.currentData()
        if not path or not os.path.exists(path):
            return
        if not path.startswith(get_models_dir(self.cfg)):
            QMessageBox.information(self, tr('Delete'), tr('Only models in the models folder can be deleted here.'))
            return
        ans = QMessageBox.question(self, tr('Delete'), tr('Delete this model from the disk? This cannot be undone.') + f'\n\n{path}')
        if ans == QMessageBox.StandardButton.Yes:
            manager.delete_model(path)
            if self.cfg.model.path == path:
                self.cfg.model.path = ''
                self.changed.emit('')
            self.refresh()


class _DlBridge(QObject):
    progress = Signal(int, int)
    finished = Signal(str, str)  # path, error


class DownloadDialog(QDialog):
    """Lists the registry models (plus the Silero VAD model) and downloads one with huggingface_hub."""

    def __init__(self, cfg, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self.setWindowTitle(tr('Download models'))
        self.resize(760, 520)
        self.installed_path = ''
        self.cancel = None
        self.bridge = _DlBridge()
        self.bridge.progress.connect(self._on_progress)
        self.bridge.finished.connect(self._on_finished)
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel(tr('Models go to:') + ' ' + get_models_dir(cfg)))
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels([tr('Model'), tr('Download'), tr('Memory (approx.)'), tr('Translate'), tr('Installed')])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        lay.addWidget(self.table, 1)
        self.bar = QProgressBar()
        self.bar.setVisible(False)
        lay.addWidget(self.bar)
        row = QHBoxLayout()
        self.status = QLabel()
        self.b_go = QPushButton(tr('Download'))
        self.b_go.clicked.connect(self._start)
        self.b_cancel = QPushButton(tr('Cancel download'))
        self.b_cancel.setEnabled(False)
        self.b_cancel.clicked.connect(self._cancel)
        b_close = QPushButton(tr('Close'))
        b_close.clicked.connect(self.close)
        row.addWidget(self.status, 1)
        for b in (self.b_go, self.b_cancel, b_close):
            row.addWidget(b)
        lay.addLayout(row)
        self._fill()

    def _fill(self):
        mdir = get_models_dir(self.cfg)
        self.entries = manager.get_registry(self.cfg.model.backend) + manager.get_registry('vad')
        self.table.setRowCount(len(self.entries))
        for r, e in enumerate(self.entries):
            if e.family == 'ten':
                name = tr('TEN VAD library (speech detection; Apache 2.0 with Agora\'s conditions)')
            elif e.backend == 'vad':
                name = tr('Silero VAD model (speech detection)')
            else:
                name = e.id
            cells = [name, fmt_bytes(e.size), fmt_bytes(e.get_ram_estimate()) if e.backend != 'vad' else '–',
                     '–' if e.backend == 'vad' else tr('yes') if e.can_translate else tr('no'),
                     '✓' if manager.is_installed(e, mdir) else '']
            for c, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if c:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if e.license:
                    item.setToolTip(tr('Licence:') + ' ' + e.license + f'\nhttps://github.com/{e.repo}')
                self.table.setItem(r, c, item)

    def _start(self):
        row = self.table.currentRow()
        if row < 0:
            self.status.setText(tr('Choose a model in the list first.'))
            return
        entry = self.entries[row]
        mdir = get_models_dir(self.cfg)
        try:
            free = shutil.disk_usage(mdir if os.path.exists(mdir) else os.path.dirname(mdir)).free
        except OSError:
            free = None
        if free is not None and free < entry.size * 1.05:
            self.status.setText(tr('Not enough free disk space.'))
            return
        self.cancel = threading.Event()
        self.bar.setVisible(True)
        self.bar.setRange(0, 1000)
        self.bar.setValue(0)
        self.b_go.setEnabled(False)
        self.b_cancel.setEnabled(True)
        self.status.setText(tr('Downloading') + f' {entry.id}…')
        mc = self.cfg.model

        def work():
            try:
                path = manager.download(entry, mdir, mc.hf_endpoint, mc.proxy,
                                        lambda d, t: self.bridge.progress.emit(d, t), self.cancel)
                self.bridge.finished.emit(path, '')
            except manager.DownloadCancelled:
                self.bridge.finished.emit('', 'cancelled')
            except Exception as e:
                self.bridge.finished.emit('', f'{type(e).__name__}: {e}')

        threading.Thread(target=work, daemon=True).start()

    def _on_progress(self, done, total):
        if total > 0:
            self.bar.setValue(int(1000 * min(1.0, done / total)))
            self.status.setText(f'{fmt_bytes(done)} / {fmt_bytes(total)}')

    def _on_finished(self, path, error):
        self.b_go.setEnabled(True)
        self.b_cancel.setEnabled(False)
        self.bar.setVisible(False)
        if error == 'cancelled':
            self.status.setText(tr('Cancelled. The download resumes where it stopped next time.'))
        elif error:
            self.status.setText(tr('Download failed:') + ' ' + error)
        else:
            self.status.setText(tr('Installed:') + ' ' + path)
            if manager.detect_format(path) != 'vad':
                self.installed_path = path
        self._fill()

    def _cancel(self):
        if self.cancel is not None:
            self.cancel.set()

    def closeEvent(self, event):
        self._cancel()
        super().closeEvent(event)
