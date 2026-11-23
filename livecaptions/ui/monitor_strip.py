"""Monitor strip: live waveform with speech detection, plus memory, CPU, inference and GPU load (PLAN.md §12.4)."""
import numpy as np
from PySide6.QtCore import QRectF, QTimer
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QHBoxLayout, QLabel, QSizePolicy, QWidget

from livecaptions.ui.i18n import tr

COLS_PER_S = 100  # one waveform column per 10 ms (pipeline.PEAK_COL)


class Waveform(QWidget):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.setMinimumHeight(36)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._resize_ring()
        self._dirty = False
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.set_fps(cfg.monitor.fps)

    def _resize_ring(self):
        n = int(self.cfg.monitor.wave_s * COLS_PER_S)
        self.peaks = np.zeros(n, np.float32)
        self.speech = np.zeros(n, bool)

    def set_fps(self, fps):
        self.timer.setInterval(int(1000 / max(1, fps)))

    def add(self, peaks, speech):
        if len(self.peaks) != int(self.cfg.monitor.wave_s * COLS_PER_S):
            self._resize_ring()
        k = min(len(peaks), len(self.peaks))
        if k == 0:
            return
        self.peaks = np.roll(self.peaks, -k)
        self.peaks[-k:] = peaks[-k:]
        self.speech = np.roll(self.speech, -k)
        self.speech[-k:] = speech
        self._dirty = True

    def clear(self):
        self.peaks[:] = 0
        self.speech[:] = False
        self.update()

    def _tick(self):
        # Repaint only when new audio arrived and the strip is visible (no wake-ups otherwise).
        if self._dirty and self.isVisible():
            self._dirty = False
            self.update()

    def showEvent(self, event):
        self.timer.start()
        super().showEvent(event)

    def hideEvent(self, event):
        self.timer.stop()
        super().hideEvent(event)

    def paintEvent(self, event):
        p = QPainter(self)
        w, h = self.width(), self.height()
        pal = self.palette()
        p.fillRect(self.rect(), pal.base())
        n = len(self.peaks)
        if n == 0 or w <= 0:
            return
        # Reduce the ring to one value per pixel column.
        idx = np.linspace(0, n, w + 1).astype(int)
        cols = np.maximum.reduceat(self.peaks, idx[:-1]) if n >= w else np.interp(np.arange(w), np.linspace(0, w, n), self.peaks)
        sp = np.maximum.reduceat(self.speech, idx[:-1]) if n >= w else np.zeros(w, bool)
        db = 20 * np.log10(np.maximum(cols, 1e-4))  # -80..0 dBFS
        amp = np.clip((db + 60) / 60, 0, 1) * (h / 2 - 2)
        if self.cfg.monitor.show_vad:
            speech_col = QColor(pal.highlight().color())
            speech_col.setAlphaF(0.25)
            x = 0
            while x < w:  # shade the runs of speech
                if sp[x]:
                    x0 = x
                    while x < w and sp[x]:
                        x += 1
                    p.fillRect(QRectF(x0, 0, x - x0, h), speech_col)
                x += 1
        p.setPen(QPen(pal.text().color(), 1))
        mid = h / 2
        for x in range(w):
            if amp[x] > 0.5:
                p.drawLine(int(x), int(mid - amp[x]), int(x), int(mid + amp[x]))


class MonitorStrip(QWidget):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.wave = Waveform(cfg)
        self.labels = {k: QLabel() for k in ('ram', 'cpu', 'busy', 'gpu', 'battery')}
        lay.addWidget(self.wave, 1)
        for lab in self.labels.values():
            lab.setMinimumWidth(70)
            lay.addWidget(lab)
        self.silent_s = 0.0
        self.apply_cfg()

    def apply_cfg(self):
        mc = self.cfg.monitor
        self.setVisible(mc.enabled)
        self.wave.setVisible(mc.show_wave)
        self.wave.set_fps(mc.fps)
        for key, on in (('ram', mc.show_ram), ('cpu', mc.show_cpu), ('busy', mc.show_busy), ('gpu', mc.show_gpu)):
            self.labels[key].setVisible(on)

    def add_level(self, payload):
        peaks = payload['peaks']
        self.wave.add(peaks, bool(payload['speech']))
        loud = len(peaks) and peaks.max() > 10 ** (-50 / 20)
        self.silent_s = 0.0 if loud else self.silent_s + len(peaks) / COLS_PER_S

    def show_sample(self, s):
        self.labels['ram'].setText(tr('RAM') + f' {s.rss / 2 ** 20:.0f} MB')
        self.labels['cpu'].setText(tr('CPU') + f' {s.cpu_pct:.0f}%')
        step = f' · {tr("run every")} {s.step_s:.1f} s' if s.step_adapted else ''
        self.labels['busy'].setText(tr('Inference') + f' {s.busy_pct:.0f}%' + step)
        self.labels['busy'].setToolTip(tr('Share of time spent in inference') + f'; RTF {s.rtf:.2f}' +
                                       ('\n' + tr('Runs are spaced out because inference fell behind.') if step else ''))
        self.labels['gpu'].setText(tr('GPU') + (f' {s.gpu_pct:.0f}%' if s.gpu_pct is not None else ' –'))
        self.labels['gpu'].setVisible(self.cfg.monitor.show_gpu and s.gpu_pct is not None)
        if s.battery_pct is not None:
            self.labels['battery'].setText(('🔌 ' if s.plugged else '🔋 ') + f'{s.battery_pct:.0f}%')
            self.labels['battery'].setVisible(True)
        else:
            self.labels['battery'].setVisible(False)
