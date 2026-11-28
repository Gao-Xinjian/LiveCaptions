"""
Monitor strip (PLAN.md §12.4): a live waveform of the captured audio, then RAM, CPU, GPU, inference load, real-time
factor and power, each in a slot of fixed width, so a hidden item or a changing number moves nothing else.
"""
import numpy as np
from PySide6.QtCore import QEvent, QLineF, QRectF, QTimer
from PySide6.QtGui import QColor, QFontMetrics, QPainter, QPen
from PySide6.QtWidgets import QHBoxLayout, QLabel, QSizePolicy, QWidget

from livecaptions.ui.i18n import tr

COLS_PER_S = 100                   # one waveform column per 10 ms (capture.LEVEL_SAMPLES)
WAVE_PENDING = QColor('#5f6368')   # audio inference has not covered yet
WAVE_DONE = QColor('#6484C3')      # audio inference has covered (or dropped as silence)
SPEECH = QColor('#3cb371')         # where the VAD heard speech
ADAPTED = QColor('#D08A1E')        # the load label while automatic adaptation spaces out the runs
SPEECH_BAR = 3                     # px under the waveform
SLOTS = ('ram', 'cpu', 'gpu', 'load', 'rtf', 'power')


def keep_space_when_hidden(widget):
    sp = widget.sizePolicy()
    sp.setRetainSizeWhenHidden(True)
    widget.setSizePolicy(sp)


class Waveform(QWidget):
    """
    The last seconds of captured audio, newest on the right. It reads the levels straight from the capture
    (source.levels), so it keeps moving while a model run blocks the pipeline; the pipeline adds where the VAD heard
    speech and how far inference has got (add_progress), in the same columns of capture position.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.setMinimumHeight(36)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        keep_space_when_hidden(self)
        self.source = None
        self.silent_s = 0.0
        self.clear()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.set_fps(cfg.monitor.fps)

    def set_fps(self, fps):
        self.timer.setInterval(int(1000 / max(1, fps)))

    def set_source(self, source):
        """Follow a capture (anything with levels(since)), or none."""
        self.source = source

    def clear(self):
        self.peaks = np.zeros(int(self.cfg.monitor.wave_s * COLS_PER_S), np.float32)
        self.total = 0         # column number just after the newest column
        self.done_col = 0
        self.speech = []       # [start, end) column runs the VAD took for speech, oldest first
        self.silent_s = 0.0
        self.update()

    def add_progress(self, payload):
        """Take the pipeline's news: whether the chunk just read was speech, and how far inference has got."""
        self.done_col = max(self.done_col, payload['done_col'])
        start, end = payload['start_col'], payload['end_col']
        if payload['speech'] and end > start:
            if self.speech and start <= self.speech[-1][1]:
                self.speech[-1][1] = max(self.speech[-1][1], end)
            else:
                self.speech.append([start, end])
        oldest = self.total - len(self.peaks)
        while self.speech and self.speech[0][1] < oldest:
            self.speech.pop(0)

    def _append(self, peaks, first):
        n = len(self.peaks)
        if n != int(self.cfg.monitor.wave_s * COLS_PER_S):   # the length setting changed
            self.peaks = np.zeros(int(self.cfg.monitor.wave_s * COLS_PER_S), np.float32)
            n = len(self.peaks)
        new = np.concatenate([np.zeros(max(0, first - self.total), np.float32), peaks])[-n:]   # a gap stays silent
        self.peaks = np.roll(self.peaks, -len(new))
        self.peaks[-len(new):] = new
        self.total = max(self.total, first + len(peaks))
        loud = peaks.max() > 10 ** (-50 / 20)
        self.silent_s = 0.0 if loud else self.silent_s + len(peaks) / COLS_PER_S

    def _tick(self):
        # Read what the capture got since the last tick; repaint only then, and only while visible.
        if self.source is None:
            return
        peaks, first = self.source.levels(self.total)
        if len(peaks):
            self._append(peaks, first)
            if self.isVisible():
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
        p.fillRect(self.rect(), self.palette().base())
        n = len(self.peaks)
        if n == 0 or w <= 0:
            return
        show_vad = self.cfg.monitor.show_vad
        wave_h = h - (SPEECH_BAR + 1 if show_vad else 0)
        # Each pixel column shows the loudest of its audio columns.
        x = np.arange(w)
        starts = x * n // w
        ends = np.maximum((x + 1) * n // w, starts + 1)
        cols = np.maximum.reduceat(self.peaks, starts)
        first_col = self.total - n
        done = first_col + ends - 1 < self.done_col
        db = 20 * np.log10(np.maximum(cols, 1e-4))   # -80..0 dBFS, shown from -60
        amp = np.clip((db + 60) / 60, 0, 1) * (wave_h / 2 - 1)
        mid = wave_h / 2
        for color, mask in ((WAVE_PENDING, ~done), (WAVE_DONE, done)):
            lines = [QLineF(i, mid - amp[i], i, mid + amp[i]) for i in np.flatnonzero(mask & (amp > 0.5))]
            if lines:
                p.setPen(QPen(color, 1))
                p.drawLines(lines)
        if show_vad:
            for start, end in self.speech:
                x0, x1 = (start - first_col) * w / n, (end - first_col) * w / n
                if x1 > 0 and x0 < w:
                    p.fillRect(QRectF(max(0.0, x0), h - SPEECH_BAR, min(w, x1) - max(0.0, x0), SPEECH_BAR), SPEECH)


class MonitorStrip(QWidget):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(14)
        self.wave = Waveform(cfg)
        lay.addWidget(self.wave, 1)
        self.labels = {}
        for key in SLOTS:
            lab = QLabel()
            keep_space_when_hidden(lab)
            self.labels[key] = lab
            lay.addWidget(lab)
        self._last = None
        self.retranslate()
        self.apply_cfg()

    @property
    def silent_s(self):
        return self.wave.silent_s

    def retranslate(self):
        """Give every slot the width of its longest text in the current language."""
        widest = {'ram': tr('RAM') + ' 88888 MB', 'cpu': tr('CPU') + ' 1888%', 'gpu': tr('GPU') + ' 100%',
                  'load': tr('Load') + ' 100%', 'rtf': 'RTF 8.88', 'power': tr('Power') + ' 100% ' + tr('(AC)')}
        for key, lab in self.labels.items():
            lab.setFixedWidth(QFontMetrics(lab.font()).horizontalAdvance(widest[key]) + 6)
        self.labels['rtf'].setToolTip(tr('Real-time factor: seconds of inference per second of audio. Below 1 keeps up.'))
        if self._last is not None:
            self.show_sample(self._last)

    def changeEvent(self, event):
        if event.type() == QEvent.Type.LanguageChange:
            self.retranslate()
        super().changeEvent(event)

    def apply_cfg(self):
        mc = self.cfg.monitor
        self.setVisible(mc.enabled)
        self.wave.setVisible(mc.show_wave)
        self.wave.set_fps(mc.fps)
        self.wave.update()
        for key, on in (('ram', mc.show_ram), ('cpu', mc.show_cpu), ('gpu', mc.show_gpu), ('load', mc.show_busy),
                        ('rtf', mc.show_rtf), ('power', mc.show_power)):
            self.labels[key].setVisible(on)

    def add_progress(self, payload):
        self.wave.add_progress(payload)

    def show_sample(self, s):
        self._last = s
        lab = self.labels
        lab['ram'].setText(tr('RAM') + f' {s.rss / 2 ** 20:.0f} MB')
        lab['cpu'].setText(tr('CPU') + f' {s.cpu_pct:.0f}%')
        lab['gpu'].setText(tr('GPU') + (f' {s.gpu_pct:.0f}%' if s.gpu_pct is not None else ' –'))
        lab['load'].setText(tr('Load') + f' {s.busy_pct:.0f}%')
        lab['load'].setStyleSheet(f'color: {ADAPTED.name()};' if s.step_adapted else '')   # runs are being spaced out
        lab['load'].setToolTip(tr('Share of time spent in inference') +
                               ('\n' + tr('Runs are spaced out because inference fell behind.') + ' ' + tr('run every') +
                                f' {s.step_s:.1f} s' if s.step_adapted else ''))
        lab['rtf'].setText(f'RTF {s.rtf:.2f}')
        if s.battery_pct is None:
            lab['power'].setText(tr('Power') + ' –')
        else:
            lab['power'].setText(tr('Power') + f' {s.battery_pct:.0f}%' + (' ' + tr('(AC)') if s.plugged else ''))
        lab['power'].setToolTip(tr('Plugged in') if s.plugged else tr('On battery') if s.battery_pct is not None else '')
