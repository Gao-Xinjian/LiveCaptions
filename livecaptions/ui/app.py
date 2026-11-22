"""The controller: owns the settings, the pipeline, the windows, the tray and the hotkeys."""
import dataclasses
import os
import time

from PySide6.QtCore import QObject, QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QMessageBox

from livecaptions import native
from livecaptions.config import store
from livecaptions.config.schema import Config
from livecaptions.pipeline import EngineHolder, Pipeline
from livecaptions.power import apply_power_settings
from livecaptions.stats.sampler import ResourceSampler
from livecaptions.ui.caption_window import CaptionView, CaptionWindow
from livecaptions.ui.hotkeys import HotkeyManager
from livecaptions.ui.i18n import install_language, tr
from livecaptions.ui.settings_window import MainWindow
from livecaptions.ui.tray import Tray

CAPTION_REFIT = {'font_size', 'font_family', 'bold', 'line_spacing', 'max_lines'}


class EventBridge(QObject):
    """Carries pipeline events (emitted on the pipeline thread) to the Qt thread."""
    event = Signal(str, object)

    def emit(self, kind, payload=None):
        self.event.emit(kind, payload)


class Controller(QObject):
    def __init__(self, app, cfg):
        super().__init__()
        self.app = app
        self.cfg = cfg
        self.holder = EngineHolder()
        self.pipeline = None
        self.session_dir = ''
        self.last_session_dir = ''
        self.engine_desc = ''
        self.sampler = ResourceSampler()
        self.bridge = EventBridge()
        self.bridge.event.connect(self.on_event)
        install_language(app, cfg.general.ui_lang)
        native_ok = native.is_available()
        if native_ok:
            native.set_log_level(cfg.general.log_level)
        apply_power_settings(cfg.general)
        # Windows.
        self.view = CaptionView(cfg)
        self.caption_win = CaptionWindow(cfg)
        self.caption_win.toggled_click_through.connect(lambda on: self.set_caption_opt('click_through', on))
        self.caption_win.toggled_on_top.connect(lambda on: self.set_caption_opt('on_top', on))
        self.caption_win.dock_requested.connect(lambda: self.set_caption_opt('docked', True))
        self.caption_win.geometry_changed.connect(self._remember_geometry)
        self.tray = Tray()
        self.tray.action.connect(self.on_tray)
        self.main = MainWindow(self)
        self.main.setting_changed.connect(self.on_setting_changed)
        self.main.terms_changed.connect(self.on_terms_changed)
        self.hotkeys = HotkeyManager(app, [self.main, self.caption_win])
        self.hotkeys.triggered.connect(self.on_hotkey)
        # Timers.
        self.save_timer = QTimer(self)
        self.save_timer.setSingleShot(True)
        self.save_timer.setInterval(500)
        self.save_timer.timeout.connect(lambda: store.save_config(self.cfg))
        self.stats_timer = QTimer(self)
        self.stats_timer.timeout.connect(self._sample)
        self.stats_timer.start(int(cfg.monitor.sample_s * 1000))
        self.idle_timer = QTimer(self)
        self.idle_timer.timeout.connect(self._check_idle_unload)
        self.idle_timer.start(30000)
        # Show.
        self.apply_dock()
        self.register_hotkeys()
        self.tray.show()
        if not (cfg.general.start_minimized and self.tray.isVisible()):
            self.main.show()
        self.main.set_status(tr('Ready.') if native_ok else tr('lc_native is missing: run python scripts/build_native.py'))

    # State.

    def is_running(self):
        return self.pipeline is not None and self.pipeline.is_running()

    def is_paused(self):
        return self.is_running() and self.pipeline.is_paused()

    def _update_state(self):
        running, paused = self.is_running(), self.is_paused()
        self.main.retranslate_controls()
        self.tray.set_state(running, paused)
        self.main.setWindowIcon(self.tray.icon())

    # Start / pause / stop.

    def start_or_pause(self):
        if self.is_running():
            self.pipeline.set_paused(not self.pipeline.is_paused())
            self._update_state()
            return
        if not self.cfg.model.path:
            QMessageBox.information(self.main, 'LiveCaptions', tr('Choose or download a model first (Model page).'))
            self.main.sidebar.setCurrentRow(1)
            return
        store.save_config(self.cfg)
        if self.view.document().characterCount() > 1:
            self.view.add_committed('\n')
        self.sampler.reset_pipeline()
        self.main.monitor.wave.clear()
        self.pipeline = Pipeline(self.cfg, self.holder, self.bridge.emit, store.load_terms(self.cfg.terms.profile))
        self.pipeline.start()
        self.t_start = time.monotonic()
        if not self.cfg.caption.docked:
            self.caption_win.show()
        self._update_state()

    def stop(self):
        if self.is_running():
            self.pipeline.stop()
            self.main.set_status(tr('Stopping…'))

    def open_session_folder(self):
        path = self.session_dir or self.last_session_dir or store.get_save_dir(self.cfg)
        os.makedirs(path, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(path))

    def count_tokens(self, text):
        """Return the token count of text with the loaded model, or None if no model is loaded."""
        eng = self.holder.engine
        if eng is None or not eng.is_loaded():
            return None
        try:
            return len(eng.tokenize(text))
        except Exception:
            return None

    # Pipeline events.

    def on_event(self, kind, payload):
        if kind == 'commit':
            self.view.add_committed(payload)
        elif kind == 'tentative':
            self.view.set_tentative(payload)
        elif kind == 'level':
            self.main.monitor.add_level(payload)
        elif kind == 'status':
            text = {'loading': tr('Loading the model…'), 'paused': tr('Paused'),
                    'running': tr('Listening') + ' · ' + self.engine_desc}.get(payload, payload)
            self.main.set_status(text)
            self._update_state()
        elif kind == 'started':
            self.engine_desc = payload['engine']
            self.session_dir = payload['session']
            src = tr('microphone') if self.cfg.audio.source == 'mic' else tr('system audio')
            self.main.set_status(tr('Listening to the') + f' {src} · {self.engine_desc}')
            self.main.update_translate_hint(payload['can_translate'])
            self.main._update_token_count()
            self._update_state()
        elif kind == 'warning':
            if payload == 'rerouted':
                self.main.set_status(tr('The audio device changed; capture follows the new default device.'))
        elif kind == 'error':
            self.main.set_status(tr('Error:') + ' ' + str(payload))
            self.tray.showMessage('LiveCaptions', tr('Error:') + ' ' + str(payload))
        elif kind == 'stopped':
            self.last_session_dir = payload.get('session') or self.last_session_dir
            self.session_dir = ''
            if not self.main.status.text().startswith(tr('Error:')):
                saved = payload.get('session')
                self.main.set_status(tr('Stopped.') + (' ' + tr('Saved in') + ' ' + saved if saved else ''))
            self.pipeline.join(1.0)
            self._update_state()

    def _sample(self):
        mc = self.cfg.monitor
        if not (mc.enabled and self.main.isVisible()):
            return
        s = self.sampler.sample(self.pipeline if self.is_running() else None, mc.cpu_per_core, mc.show_gpu)
        self.main.monitor.show_sample(s)
        if self.is_running():
            secs = int(self.pipeline.audio_s)
            self.main.elapsed.setText(f'{secs // 3600:02d}:{secs // 60 % 60:02d}:{secs % 60:02d}')
            if mc.silent_warn_s and self.main.monitor.silent_s >= mc.silent_warn_s and not self.is_paused():
                self.main.set_status(tr('No sound from the input for') + f' {int(self.main.monitor.silent_s)} s')

    def _check_idle_unload(self):
        minutes = self.cfg.model.idle_unload_min
        if minutes > 0 and not self.is_running() and self.holder.is_loaded() \
                and time.monotonic() - self.holder.last_used > minutes * 60:
            self.holder.unload()
            self.main.set_status(tr('Model unloaded after being idle.'))

    # Settings.

    def on_setting_changed(self, section, name):
        self.save_timer.start()
        cc = self.cfg.caption
        if section == 'caption':
            if name in ('on_top', 'click_through'):
                self.caption_win.apply_flags()
                self.tray.acts['click'].setChecked(cc.click_through)
            elif name == 'docked':
                self.apply_dock()
            elif name.startswith('hk_'):
                self.register_hotkeys()
            else:
                if name == 'scheme':
                    self.main.update_caption_rows()
                self.view.apply_style()
                self.caption_win.frame.update()
                if name in CAPTION_REFIT:
                    self.caption_win.fit_lines()
        elif section == 'monitor':
            self.main.monitor.apply_cfg()
            self.stats_timer.setInterval(int(self.cfg.monitor.sample_s * 1000))
        elif section == 'general':
            if name == 'ui_lang':
                install_language(self.app, self.cfg.general.ui_lang)
                self.tray.set_state(self.is_running(), self.is_paused())
            elif name in ('low_priority', 'eco_qos'):
                apply_power_settings(self.cfg.general)
            elif name == 'log_level' and native.is_available():
                native.set_log_level(self.cfg.general.log_level)
        elif section == 'model' and name in ('backend', 'models_dir'):
            self.main.update_backend()
        elif section == 'model' and name == 'path':
            self.main.update_translate_hint()
        elif section == 'audio' and name == 'source':
            self.main.refresh_devices()
            self.main.update_scope()
        elif section == 'vad' and name in ('method', 'cascade'):
            self.main.update_vad_rows()
        elif section == 'stream' and name == 'policy':
            self.main.update_scope()
        elif section == 'terms':
            if self.is_running():
                self.pipeline.set_terms(store.load_terms(self.cfg.terms.profile))
            self.main._update_token_count()
        if self.is_running() and not self._is_live(section, name):
            self.main.set_status(tr('This change applies at the next Start.'))

    def _is_live(self, section, name):
        f = next((x for x in dataclasses.fields(getattr(self.cfg, section)) if x.name == name), None)
        return f is None or f.metadata.get('live', False) or section in ('caption', 'monitor', 'general')

    def on_terms_changed(self, text):
        if self.is_running():
            self.pipeline.set_terms(text)

    def replace_config(self, new_cfg):
        """Take over all values of new_cfg (import / reset); the running session keeps its non-live ones."""
        for sec in dataclasses.fields(Config):
            if sec.name != 'version':
                for f in dataclasses.fields(getattr(new_cfg, sec.name)):
                    setattr(getattr(self.cfg, sec.name), f.name, getattr(getattr(new_cfg, sec.name), f.name))
        store.save_config(self.cfg)
        install_language(self.app, self.cfg.general.ui_lang)
        self.main.build_pages()
        self.view.apply_style()
        self.caption_win.apply_flags()
        self.apply_dock()
        self.register_hotkeys()

    def set_caption_opt(self, name, value):
        setattr(self.cfg.caption, name, value)
        self.main.build_pages()  # show the new value in the form
        self.on_setting_changed('caption', name)

    def _remember_geometry(self, geo):
        if self.caption_win.isVisible():
            self.cfg.caption.geometry = geo
            self.save_timer.start()

    def apply_dock(self):
        """Put the caption view in the floating window or in the main window."""
        if self.cfg.caption.docked:
            self.caption_win.hide()
            self.main.set_docked_view(self.view)
        else:
            self.main.set_docked_view(None)
            self.caption_win.set_view(self.view)
            self.caption_win.fit_lines()
            if self.is_running():
                self.caption_win.show()
        self.view.apply_style()

    def register_hotkeys(self):
        cc = self.cfg.caption
        failed = self.hotkeys.set_hotkeys({'start_pause': cc.hk_start, 'stop': cc.hk_stop, 'click': cc.hk_click,
                                           'show_captions': cc.hk_show})
        if failed:
            self.main.set_status(tr('Some hotkeys could not be registered (used by another app?):') + ' ' + ', '.join(failed))

    # Tray and hotkeys.

    def on_hotkey(self, action):
        self.on_tray(action)

    def on_tray(self, action):
        if action == 'start_pause':
            self.start_or_pause()
        elif action == 'stop':
            self.stop()
        elif action == 'click':
            self.set_caption_opt('click_through', not self.cfg.caption.click_through)
        elif action == 'show_captions':
            if self.cfg.caption.docked:
                self.main.showNormal()
                self.main.activateWindow()
            elif self.caption_win.isVisible():
                self.caption_win.hide()
            else:
                self.caption_win.show()
        elif action == 'show_main':
            self.main.showNormal()
            self.main.raise_()
            self.main.activateWindow()
        elif action == 'open_folder':
            self.open_session_folder()
        elif action == 'quit':
            self.quit()

    def quit(self):
        if self.is_running():
            self.pipeline.stop()
            self.pipeline.join(5.0)
        self.hotkeys.clear()
        store.save_config(self.cfg)
        self.holder.unload()
        self.tray.hide()
        self.app.quit()
