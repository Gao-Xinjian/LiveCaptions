"""First-run dialog: shown while no model is installed; downloads a recommended model and the Silero VAD model."""
import os
import shutil
import threading

from PySide6.QtCore import QEvent, QObject, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QComboBox, QDialog, QFileDialog, QHBoxLayout, QLabel,
                               QProgressBar, QPushButton, QRadioButton, QVBoxLayout)

from livecaptions import native
from livecaptions.config.store import get_models_dir
from livecaptions.models import manager
from livecaptions.models.registry import find_entry
from livecaptions.ui.i18n import tr
from livecaptions.ui.model_widgets import fmt_bytes

CHOICES = [   # (registry id, what it is good for)
    ('large-v3-turbo-q5_0', 'accurate; best with a GPU'),
    ('small-q5_1', 'lighter; fine on the CPU'),
    ('base-q5_1', 'fastest, less accurate'),
]
SILERO_ID = 'silero-v6.2.0'   # the default VAD needs it
MIRROR = 'https://hf-mirror.com'


def needs_first_run(cfg):
    """Return whether no usable model is chosen and none is installed: the app cannot caption yet."""
    if cfg.model.path and os.path.exists(cfg.model.path):
        return False
    return not any(m.backend in ('whispercpp', 'fasterwhisper') for m in manager.list_installed(get_models_dir(cfg)))


def get_gpu_names():
    """Return the GPUs whisper.cpp can use (empty without lc_native)."""
    try:
        return [name for _, name, _ in native.list_gpus()]
    except Exception:
        return []


class _Bridge(QObject):
    progress = Signal(int, int)
    finished = Signal(str, str)   # model path ('' when only the VAD model was fetched), error


class FirstRunDialog(QDialog):
    def __init__(self, ctl):
        super().__init__(ctl.main)
        self.ctl = ctl
        self.cfg = ctl.cfg
        self.cancel = None
        self.bridge = _Bridge()
        self.bridge.progress.connect(self._on_progress)
        self.bridge.finished.connect(self._on_finished)
        self.entries = [(e, desc) for e, desc in ((find_entry(i), d) for i, d in CHOICES) if e is not None]
        self.gpus = get_gpu_names()
        self.resize(560, 0)
        lay = QVBoxLayout(self)
        self.title = QLabel()
        font = QFont(self.title.font())
        font.setPointSizeF(font.pointSizeF() * 1.4)
        font.setBold(True)
        self.title.setFont(font)
        lay.addWidget(self.title)
        # Interface language: switching retranslates this dialog and the main window at once.
        row = QHBoxLayout()
        self.l_lang = QLabel()
        self.lang = QComboBox()
        for text, code in (('English', 'en'), ('简体中文', 'zh_CN')):
            self.lang.addItem(text, code)
        self.lang.setCurrentIndex(max(0, self.lang.findData(self.cfg.general.ui_lang)))
        self.lang.currentIndexChanged.connect(self._on_lang)
        row.addWidget(self.l_lang)
        row.addWidget(self.lang)
        row.addStretch(1)
        lay.addLayout(row)
        self.intro = QLabel()
        self.intro.setWordWrap(True)
        lay.addWidget(self.intro)
        self.gpu = QLabel()   # own label: a wrapped label is sized for its first lines only
        lay.addWidget(self.gpu)
        # The models: the first if a GPU was found, else the second.
        self.group = QButtonGroup(self)
        self.radios = []
        for k in range(len(self.entries)):
            rb = QRadioButton()
            self.group.addButton(rb, k)
            self.radios.append(rb)
            lay.addWidget(rb)
        if self.radios:
            self.radios[0 if self.gpus or len(self.radios) == 1 else 1].setChecked(True)
        self.more = QLabel()
        self.more.setWordWrap(True)
        lay.addWidget(self.more)
        self.mirror = QCheckBox()
        self.mirror.setChecked(self.cfg.model.hf_endpoint.rstrip('/') == MIRROR)
        lay.addWidget(self.mirror)
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setVisible(False)
        lay.addWidget(self.bar)
        self.status = QLabel()
        self.status.setWordWrap(True)
        lay.addWidget(self.status)
        row = QHBoxLayout()
        self.b_browse = QPushButton()
        self.b_browse.clicked.connect(self._browse)
        self.b_later = QPushButton()
        self.b_later.clicked.connect(self.reject)
        self.b_go = QPushButton()
        self.b_go.setDefault(True)
        self.b_go.clicked.connect(self._start)
        row.addWidget(self.b_browse)
        row.addStretch(1)
        row.addWidget(self.b_later)
        row.addWidget(self.b_go)
        lay.addLayout(row)
        self.retranslate()

    def retranslate(self):
        self.setWindowTitle('LiveCaptions')
        self.title.setText(tr('Welcome to LiveCaptions'))
        self.l_lang.setText(tr('Interface language'))
        self.intro.setText(tr('Captions need a speech recognition model. It is not included: choose one to download '
                              '(it stays on this computer), or use a model file you already have.'))
        self.gpu.setText((tr('GPU found:') + ' ' + ', '.join(self.gpus)) if self.gpus
                         else tr('No GPU found; the CPU will do the work.'))
        for rb, (e, desc) in zip(self.radios, self.entries):
            rb.setText(f'{e.id}  ({fmt_bytes(e.size)}): {tr(desc)}')
        self.more.setText(tr('More models, also for faster-whisper, are on the Model page under Download….'))
        self.mirror.setText(tr('Download through hf-mirror.com (faster in mainland China)'))
        self.b_browse.setText(tr('Use a model file I have…'))
        self.b_later.setText(tr('Later'))
        self.b_go.setText(tr('Download and use'))

    def changeEvent(self, event):
        if event.type() == QEvent.Type.LanguageChange:
            self.retranslate()
        super().changeEvent(event)

    def _on_lang(self, idx):
        code = self.lang.itemData(idx)
        if code and code != self.cfg.general.ui_lang:
            self.cfg.general.ui_lang = code
            self.ctl.on_setting_changed('general', 'ui_lang')

    # Downloads.

    def _start(self):
        k = self.group.checkedId()
        if k < 0:
            return
        entry = self.entries[k][0]
        mdir = get_models_dir(self.cfg)
        try:
            free = shutil.disk_usage(mdir if os.path.exists(mdir) else os.path.dirname(mdir)).free
        except OSError:
            free = None
        if free is not None and free < entry.size * 1.05:
            self.status.setText(tr('Not enough free disk space.'))
            return
        self._download(entry)

    def _download(self, entry):
        """Fetch the Silero VAD model if missing, then entry (None = only the VAD model), on a worker thread."""
        if self.mirror.isChecked():
            self.cfg.model.hf_endpoint = MIRROR
        elif self.cfg.model.hf_endpoint.rstrip('/') == MIRROR:
            self.cfg.model.hf_endpoint = ''
        mdir, mc = get_models_dir(self.cfg), self.cfg.model
        self.cancel = threading.Event()
        for w in (self.b_go, self.b_browse, self.lang, self.mirror, *self.radios):
            w.setEnabled(False)
        self.b_later.setText(tr('Cancel'))
        self.bar.setValue(0)
        self.bar.setVisible(entry is not None)
        self.status.setText(tr('Downloading') + f' {entry.id if entry else tr("the speech detector")}…')

        def work():
            try:
                silero = find_entry(SILERO_ID)
                if silero is not None and not manager.is_installed(silero, mdir):
                    manager.download(silero, mdir, mc.hf_endpoint, mc.proxy, None, self.cancel)
                path = ''
                if entry is not None:
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
            self.status.setText(tr('Downloading') + f' {fmt_bytes(done)} / {fmt_bytes(total)}')

    def _on_finished(self, path, error):
        self.cancel = None
        if error:
            for w in (self.b_go, self.b_browse, self.lang, self.mirror, *self.radios):
                w.setEnabled(True)
            self.b_later.setText(tr('Later'))
            self.bar.setVisible(False)
            self.status.setText(tr('Cancelled. The download resumes where it stopped next time.') if error == 'cancelled'
                                else tr('Download failed:') + ' ' + error)
            return
        if path:
            self._use_model(path)
        self.accept()

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(self, tr('Choose a ggml model file'), os.path.expanduser('~'),
                                              'ggml (*.bin);;*')
        if not path:
            return
        path = os.path.normpath(path)
        if manager.detect_format(path) != 'whispercpp' and manager.detect_format(os.path.dirname(path)) == 'fasterwhisper':
            path = os.path.dirname(path)   # model.bin inside a CTranslate2 folder
        if manager.detect_format(path) not in ('whispercpp', 'fasterwhisper'):
            self.status.setText(tr('Not a model this app can read:') + ' ' + path)
            return
        self._use_model(path)
        silero = find_entry(SILERO_ID)
        if silero is not None and not manager.is_installed(silero, get_models_dir(self.cfg)):
            self._download(None)
        else:
            self.accept()

    def _use_model(self, path):
        backend = manager.detect_format(path)
        if backend != self.cfg.model.backend:
            self.cfg.model.backend = backend
            self.ctl.on_setting_changed('model', 'backend')
        self.cfg.model.path = path
        self.ctl.on_setting_changed('model', 'path')
        self.ctl.main.update_backend()   # refresh the model page

    def reject(self):
        if self.cancel is not None:   # Cancel during a download
            self.cancel.set()
            return
        super().reject()
