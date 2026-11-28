"""Main window: settings pages on the left, docked captions, monitor strip and Start/Pause/Stop at the bottom."""
import os
import sys

from PySide6.QtCore import QEvent, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QComboBox, QFileDialog, QGroupBox, QHBoxLayout, QInputDialog, QLabel, QListWidget,
                               QMainWindow, QMessageBox, QPlainTextEdit, QPushButton, QScrollArea, QSplitter,
                               QStackedWidget, QVBoxLayout, QWidget)

from livecaptions import native
from livecaptions.audio.devices import list_devices, supports_system_audio
from livecaptions.config import store
from livecaptions.engines.fasterwhisper import get_cuda_count
from livecaptions.segmentation.prompt import split_terms, terms2line
from livecaptions.ui.forms import SectionForm
from livecaptions.ui.i18n import tr
from livecaptions.ui.model_widgets import ModelPicker
from livecaptions.ui.monitor_strip import MonitorStrip
from livecaptions.ui.share_panel import SharePanel
from livecaptions.ui.tray import make_icon

# (page title, [(section, group title)])
PAGES = [
    ('Audio', [('audio', '')]),
    ('Model', [('model', '')]),
    ('Recognition', [('recog', '')]),
    ('Segmentation', [('stream', '')]),
    ('VAD & filters', [('vad', 'Speech detection'), ('filters', 'Hallucination filters')]),
    ('Terms', [('terms', '')]),
    ('Saving', [('save', '')]),
    ('Captions', [('caption', '')]),
    ('Sharing', [('share', '')]),
    ('Monitor', [('monitor', '')]),
    ('General', [('general', '')]),
]


class MainWindow(QMainWindow):
    setting_changed = Signal(str, str)
    terms_changed = Signal(str)

    def __init__(self, ctl):
        super().__init__()
        self.ctl = ctl
        self.cfg = ctl.cfg
        self.setWindowIcon(make_icon())
        self.forms = {}
        self.pickers = {}
        self._terms_timer = QTimer(self)
        self._terms_timer.setSingleShot(True)
        self._terms_timer.setInterval(600)
        self._terms_timer.timeout.connect(self._save_terms)
        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        self.splitter = QSplitter(Qt.Orientation.Vertical)
        top = QWidget()
        top_lay = QHBoxLayout(top)
        top_lay.setContentsMargins(0, 0, 0, 0)
        self.sidebar = QListWidget()
        self.sidebar.setFixedWidth(125)
        self.stack = QStackedWidget()
        self.sidebar.currentRowChanged.connect(self.stack.setCurrentIndex)
        top_lay.addWidget(self.sidebar)
        top_lay.addWidget(self.stack, 1)
        self.splitter.addWidget(top)
        self.dock_area = QWidget()  # holds the caption view when captions are docked
        self.dock_lay = QVBoxLayout(self.dock_area)
        self.dock_lay.setContentsMargins(0, 0, 0, 0)
        self.splitter.addWidget(self.dock_area)
        self.dock_area.setVisible(False)
        outer.addWidget(self.splitter, 1)
        self.monitor = MonitorStrip(self.cfg)
        outer.addWidget(self.monitor)
        outer.addLayout(self._build_controls())
        self.build_pages()
        self.retranslate_controls()
        self.resize(980, 720)
        self.setMinimumSize(920, 620)

    # Controls.

    def _build_controls(self):
        row = QHBoxLayout()
        self.b_start = QPushButton()
        self.b_start.setMinimumWidth(100)
        self.b_start.clicked.connect(self.ctl.start_or_pause)
        self.b_stop = QPushButton()
        self.b_stop.setMinimumWidth(100)
        self.b_stop.clicked.connect(self.ctl.stop)
        self.b_captions = QPushButton()
        self.b_captions.clicked.connect(lambda: self.ctl.on_tray('show_captions'))
        self.b_click = QPushButton()   # the way back from click-through without hotkeys or the tray
        self.b_click.setCheckable(True)
        self.b_click.clicked.connect(lambda on: self.ctl.set_caption_opt('click_through', on))
        self.b_folder = QPushButton()
        self.b_folder.clicked.connect(self.ctl.open_session_folder)
        self.status = QLabel()
        self.status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.status.setMinimumWidth(0)
        self.elapsed = QLabel('00:00:00')
        for button in (self.b_start, self.b_stop, self.b_captions, self.b_click, self.b_folder):
            button.setFixedSize(150, 40)
        row.addWidget(self.b_start)
        row.addWidget(self.b_stop)
        row.addWidget(self.status, 1)
        row.addWidget(self.elapsed)
        row.addWidget(self.b_captions)
        row.addWidget(self.b_click)
        row.addWidget(self.b_folder)
        return row

    def retranslate_controls(self):
        self.setWindowTitle('LiveCaptions')
        running, paused = self.ctl.is_running(), self.ctl.is_paused()
        self.b_start.setText(tr('Pause') if running and not paused else (tr('Resume') if paused else tr('Start')))
        self.b_stop.setText(tr('Stop'))
        self.b_stop.setEnabled(running)
        cc = self.cfg.caption
        self.b_captions.setText(tr('Show / hide captions'))
        self.b_click.setText(tr('Click-through'))
        self.b_click.setToolTip(tr('Clicks go to the windows below. Toggle back with the button in the main window or the tray menu.'))
        self.b_click.setChecked(cc.click_through)
        for b in (self.b_captions, self.b_click):   # docked captions live in this window
            b.setEnabled(not cc.docked)
        self.b_folder.setText(tr('Open folder'))

    def set_status(self, text):
        self.status.setText(text)

    # Pages.

    def build_pages(self):
        """(Re)build every settings page from the schema, keeping the selected page."""
        current = max(0, self.sidebar.currentRow())
        self.sidebar.blockSignals(True)
        self.sidebar.clear()
        while self.stack.count():
            w = self.stack.widget(0)
            self.stack.removeWidget(w)
            w.deleteLater()
        self.forms = {}
        self.pickers = {}
        special = {'device': self._make_device_picker, 'model': self._make_model_picker, 'gpu': self._make_gpu_picker,
                   'profile': self._make_profile_picker}
        for title, sections in PAGES:
            page = QWidget()
            lay = QVBoxLayout(page)
            for sec, group in sections:
                form = SectionForm(sec, getattr(self.cfg, sec), special, tr(group) if group else '')
                form.changed.connect(self.setting_changed)
                self.forms[sec] = form
                lay.addWidget(form)
            if title == 'Terms':
                lay.addWidget(self._build_terms_editor(), 1)
            if title == 'Sharing':
                self.share_panel = SharePanel(self.ctl)
                lay.addWidget(self.share_panel)
            if title == 'General':
                lay.addWidget(self._build_general_buttons())
                if get_platform_note():
                    lay.addWidget(QLabel(get_platform_note()))
            if title == 'Audio':
                if not supports_system_audio():
                    lay.addWidget(QLabel(tr('System audio capture needs Windows, or macOS 14.2 or later.')))
                elif sys.platform == 'darwin':
                    note = QLabel(tr('macOS asks for permission the first time: allow it in System Settings > Privacy & '
                                     'Security (microphone, and screen & system audio recording).'))
                    note.setWordWrap(True)
                    lay.addWidget(note)
            if title != 'Terms':
                lay.addStretch(1)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(page)
            self.stack.addWidget(scroll)
            self.sidebar.addItem(tr(title))
        self.sidebar.blockSignals(False)
        self.sidebar.setCurrentRow(min(current, self.sidebar.count() - 1))
        self.stack.setCurrentIndex(self.sidebar.currentRow())
        self.update_backend()

    def update_backend(self):
        """Grey out options of the other backend or policy and refresh the backend-specific pickers."""
        if 'model' in self.pickers:
            self.pickers['model'].refresh()
        self.refresh_gpus()
        self.update_scope()

    def update_scope(self):
        """Grey out options that the current backend or streaming policy does not use."""
        for form in self.forms.values():
            form.set_scope(self.cfg.model.backend, self.cfg.stream.policy)
        if 'audio' in self.forms:
            self.forms['audio'].set_row_enabled('tap_apps', sys.platform == 'darwin' and self.cfg.audio.source == 'system')
        self.update_translate_hint()
        self.update_vad_rows()

    def update_vad_rows(self):
        """Enable the options of the selected VAD method only."""
        form = self.forms.get('vad')
        if form is None:
            return
        method, cascade = self.cfg.vad.method, self.cfg.vad.cascade
        neural = method in ('silero', 'ten')
        energy = method == 'energy' or (neural and cascade)
        for name in ('floor_window_s', 'floor_percentile', 'margin_db', 'min_level_db', 'band_limit'):
            form.set_row_enabled(name, energy)
        for name in ('cascade', 'agc_on', 'agc_max_db'):
            form.set_row_enabled(name, neural)
        for name in ('silero_model', 'silero_threads'):
            form.set_row_enabled(name, method == 'silero')
        for name in ('ten_library', 'ten_hop'):
            form.set_row_enabled(name, method == 'ten')
        for name in ('threshold', 'neg_threshold', 'min_speech_ms'):
            form.set_row_enabled(name, method != 'off')
        self.update_caption_rows()

    def update_caption_rows(self):
        """Enable the colour pickers only for the custom colour scheme."""
        form = self.forms.get('caption')
        if form is not None:
            for name in ('fg', 'bg', 'tentative_fg'):
                form.set_row_enabled(name, self.cfg.caption.scheme == 'custom')

    def update_translate_hint(self, can_translate=None):
        form = self.forms.get('recog')
        if form is None:
            return
        name = os.path.basename(os.path.normpath(self.cfg.model.path)).lower() if self.cfg.model.path else ''
        if can_translate is None:
            can_translate = not any(k in name for k in ('turbo', 'distil', '.en'))
        form.set_row_enabled('translate', can_translate)

    def _make_device_picker(self, form, f, value):
        w = QWidget()
        lay = QHBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        combo = QComboBox()
        btn = QPushButton(tr('Refresh'))
        lay.addWidget(combo, 1)
        lay.addWidget(btn)
        self.pickers['device'] = combo

        def fill():
            combo.blockSignals(True)
            combo.clear()
            combo.addItem(tr('System default'), '')
            for d in list_devices(self.cfg.audio.source):
                combo.addItem(d.name + ('  ' + tr('(default)') if d.is_default else ''), d.id_hex)
            idx = combo.findData(self.cfg.audio.device_id)
            if idx < 0 and self.cfg.audio.device_id:
                combo.addItem(self.cfg.audio.device_name + ' ' + tr('(not found)'), self.cfg.audio.device_id)
                idx = combo.count() - 1
            combo.setCurrentIndex(max(0, idx))
            combo.blockSignals(False)

        def pick(idx):
            self.cfg.audio.device_id = combo.itemData(idx) or ''
            self.cfg.audio.device_name = combo.itemText(idx) if self.cfg.audio.device_id else ''
            form.changed.emit('audio', 'device_id')

        combo.currentIndexChanged.connect(pick)
        btn.clicked.connect(fill)
        self.refresh_devices = fill
        fill()
        return w

    def _make_model_picker(self, form, f, value):
        picker = ModelPicker(self.cfg)
        picker.changed.connect(lambda _: form.changed.emit('model', 'path'))
        self.pickers['model'] = picker
        return picker

    def _make_gpu_picker(self, form, f, value):
        combo = QComboBox()
        self.pickers['gpu'] = combo
        combo.currentIndexChanged.connect(lambda idx: self._pick_gpu(form, combo, idx))
        return combo

    def _pick_gpu(self, form, combo, idx):
        if idx >= 0 and combo.itemData(idx) is not None and combo.itemData(idx) != self.cfg.recog.gpu_device:
            self.cfg.recog.gpu_device = int(combo.itemData(idx))
            form.changed.emit('recog', 'gpu_device')

    def refresh_gpus(self):
        combo = self.pickers.get('gpu')
        if combo is None:
            return
        combo.blockSignals(True)
        combo.clear()
        if self.cfg.model.backend == 'fasterwhisper':
            gpus = [(i, f'CUDA {i}', 'discrete') for i in range(get_cuda_count())]
        else:
            gpus = native.list_gpus() if native.is_available() else []
        for i, desc, kind in gpus:
            combo.addItem(f'{desc}  ({tr(kind)})', i)
        if not gpus:
            combo.addItem(tr('No GPU found: the CPU is used'), None)
        idx = combo.findData(self.cfg.recog.gpu_device)
        combo.setCurrentIndex(max(0, idx))
        combo.blockSignals(False)

    def _make_profile_picker(self, form, f, value):
        w = QWidget()
        lay = QHBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        combo = QComboBox()
        b_new = QPushButton(tr('New…'))
        b_del = QPushButton(tr('Delete'))
        lay.addWidget(combo, 1)
        lay.addWidget(b_new)
        lay.addWidget(b_del)
        self.pickers['profile'] = combo

        def fill():
            combo.blockSignals(True)
            combo.clear()
            for name in store.list_profiles():
                combo.addItem(name, name)
            combo.setCurrentIndex(max(0, combo.findData(self.cfg.terms.profile)))
            combo.blockSignals(False)

        def pick(idx):
            self._flush_terms()
            self.cfg.terms.profile = combo.itemData(idx) or 'default'
            form.changed.emit('terms', 'profile')
            self.load_terms_editor()

        def new():
            name, ok = QInputDialog.getText(self, tr('New profile'), tr('Profile name (e.g. the course):'))
            if ok and name.strip():
                self._flush_terms()
                store.save_terms(name.strip(), '')
                self.cfg.terms.profile = name.strip()
                fill()
                form.changed.emit('terms', 'profile')
                self.load_terms_editor()

        def delete():
            name = self.cfg.terms.profile
            if name == 'default':
                return
            if QMessageBox.question(self, tr('Delete'), tr('Delete the profile') + f' "{name}"?') == QMessageBox.StandardButton.Yes:
                store.delete_profile(name)
                self.cfg.terms.profile = 'default'
                fill()
                form.changed.emit('terms', 'profile')
                self.load_terms_editor()

        combo.currentIndexChanged.connect(pick)
        b_new.clicked.connect(new)
        b_del.clicked.connect(delete)
        fill()
        return w

    def _build_terms_editor(self):
        box = QGroupBox(tr('Terms of this profile (separated by spaces, commas or 、)'))
        lay = QVBoxLayout(box)
        self.terms_edit = QPlainTextEdit()
        self.terms_edit.setPlaceholderText(tr('e.g. 突触 递质 抑制, or hippocampus, dendrite, synapse'))
        self.terms_edit.textChanged.connect(self._terms_timer.start)
        self.terms_count = QLabel()
        lay.addWidget(self.terms_edit, 1)
        lay.addWidget(self.terms_count)
        self.load_terms_editor()
        return box

    def load_terms_editor(self):
        self.terms_edit.blockSignals(True)
        self.terms_edit.setPlainText(store.load_terms(self.cfg.terms.profile))
        self.terms_edit.blockSignals(False)
        self._update_token_count()

    def _flush_terms(self):
        if self._terms_timer.isActive():
            self._terms_timer.stop()
            self._save_terms()

    def _save_terms(self):
        text = self.terms_edit.toPlainText()
        store.save_terms(self.cfg.terms.profile, text)
        self._update_token_count()
        self.terms_changed.emit(text)

    def _update_token_count(self):
        words = split_terms(self.terms_edit.toPlainText())
        n = self.ctl.count_tokens(terms2line(words)) if words else 0
        limit = self.cfg.terms.term_tokens
        if n is None:
            self.terms_count.setText(tr('Tokens are counted once a model is loaded.') + f' ({len(words)} ' + tr('terms') + ')')
        else:
            warn = '  ⚠ ' + tr('Too long: the last terms are left out.') if n > limit else ''
            self.terms_count.setText(f'{len(words)} ' + tr('terms') + f', {n} / {limit} ' + tr('tokens') + warn)

    def _build_general_buttons(self):
        box = QWidget()
        lay = QHBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)
        b_exp = QPushButton(tr('Export settings…'))
        b_imp = QPushButton(tr('Import settings…'))
        b_dir = QPushButton(tr('Open the settings folder'))
        b_reset = QPushButton(tr('Reset to defaults'))
        b_quit = QPushButton(tr('Quit'))
        b_exp.clicked.connect(self._export)
        b_imp.clicked.connect(self._import)
        b_dir.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(store.get_config_dir())))
        b_reset.clicked.connect(self._reset)
        b_quit.clicked.connect(self.ctl.quit)
        for b in (b_exp, b_imp, b_dir, b_reset):
            lay.addWidget(b)
        lay.addStretch(1)
        lay.addWidget(b_quit)
        return box

    def _export(self):
        path, _ = QFileDialog.getSaveFileName(self, tr('Export settings'), 'livecaptions.toml', 'TOML (*.toml)')
        if path:
            store.save_config(self.cfg, path)

    def _import(self):
        path, _ = QFileDialog.getOpenFileName(self, tr('Import settings'), '', 'TOML (*.toml)')
        if path:
            self.ctl.replace_config(store.load_config(path))

    def _reset(self):
        if QMessageBox.question(self, tr('Reset to defaults'), tr('Reset all settings to their defaults?')) == QMessageBox.StandardButton.Yes:
            self.ctl.replace_config(store.Config())

    # Docked captions.

    def set_docked_view(self, view):
        """Show the caption view inside this window (view) or not (None)."""
        if view is not None:
            self.dock_lay.addWidget(view)
            view.show()
            self.dock_area.setVisible(True)
            self.dock_area.setStyleSheet('')
        else:
            self.dock_area.setVisible(False)

    # Events.

    def changeEvent(self, event):
        if event.type() == QEvent.Type.LanguageChange:
            self.build_pages()
            self.retranslate_controls()
        super().changeEvent(event)

    def closeEvent(self, event):
        if self.ctl.is_running() and self.ctl.tray.isVisible():  # keep captioning from the tray
            self.hide()
            self.ctl.tray.showMessage('LiveCaptions', tr('Still running. Use the tray icon to show the window or quit.'))
            event.ignore()
            return
        event.accept()
        self.ctl.quit()


def get_platform_note():
    """Return a note about features that are not available on this OS yet."""
    if sys.platform == 'darwin':
        return tr('macOS: global hotkeys work only while a LiveCaptions window is active.')
    return ''
