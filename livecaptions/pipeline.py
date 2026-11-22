"""
The pipeline thread: audio source -> VAD -> streaming policy -> engine -> filters -> writers -> events.

No Qt here: events go to a callback emit(kind, payload), which the UI turns into queued Qt signals.
Event kinds: status, error, warning, started, level, commit, tentative, stats, stopped.
"""
import threading
import time

import numpy as np

from livecaptions import SAMPLE_RATE
from livecaptions.audio.capture import FLAG_REROUTED, FLAG_STOPPED, NativeCapture
from livecaptions.config.store import get_models_dir
from livecaptions.engines import create_engine
from livecaptions.output.session import SessionWriter
from livecaptions.segmentation import create_policy
from livecaptions.vad import create_vad

PEAK_COL = 160  # samples per waveform column (10 ms)


class EngineHolder:
    """Keeps the loaded engine between sessions, so Start is instant when nothing changed."""

    def __init__(self):
        self.engine = None
        self.last_used = time.monotonic()
        self.lock = threading.Lock()

    def get(self, cfg):
        """Return an engine with cfg's model loaded (loads or reloads as needed)."""
        with self.lock:
            if self.engine is None or self.engine.name != cfg.model.backend:
                self.unload_locked()
                self.engine = create_engine(cfg.model.backend)
            self.engine.load(cfg.model.path, cfg.recog)
            self.last_used = time.monotonic()
            return self.engine

    def unload_locked(self):
        if self.engine is not None:
            self.engine.unload()
        self.engine = None

    def unload(self):
        with self.lock:
            self.unload_locked()

    def is_loaded(self):
        return self.engine is not None and self.engine.is_loaded()


class Pipeline:
    def __init__(self, cfg, holder, emit, terms_text='', source=None, save=True):
        """
        Args:
            cfg: Config, shared with the UI; live options are read on every use
            holder: EngineHolder
            emit: callback(kind, payload), called on the pipeline thread
            terms_text: the course terms
            source: an audio source (None = live capture as configured)
            save: whether to write a session folder
        """
        self.cfg = cfg
        self.holder = holder
        self.emit = emit
        self.terms_text = terms_text
        self.source = source
        self.save = save
        self.thread = None
        self.policy = None
        self._stop = threading.Event()
        self._paused = threading.Event()
        self._resume = threading.Event()
        # Counters read by the stats sampler (plain floats: atomic enough for display).
        self.audio_s = 0.0
        self.busy_s = 0.0
        self.runs = 0
        self.run_audio_s = 0.0
        self.latencies = []
        self.language = ''

    def start(self):
        self.thread = threading.Thread(target=self._run, name='pipeline', daemon=True)
        self.thread.start()

    def stop(self):
        self._stop.set()
        self._resume.set()
        if self.holder.engine is not None:
            self.holder.engine.abort()

    def set_paused(self, paused):
        if paused:
            self._resume.clear()
            self._paused.set()
        else:
            self._paused.clear()
            self._resume.set()

    def is_paused(self):
        return self._paused.is_set()

    def is_running(self):
        return self.thread is not None and self.thread.is_alive()

    def join(self, timeout=None):
        if self.thread is not None:
            self.thread.join(timeout)

    def set_terms(self, text):
        self.terms_text = text
        if self.policy is not None:
            self.policy.set_terms(text)

    def _handle(self, out, session):
        if out.committed:
            self.emit('commit', out.committed)
        if out.tentative is not None:
            self.emit('tentative', out.tentative)
        if session is not None and out.lines:
            session.write_lines(out.lines)
        self.busy_s += out.busy_s
        self.runs += out.runs
        self.run_audio_s += out.audio_s
        self.latencies = (self.latencies + out.latencies)[-200:]
        if out.language:
            self.language = out.language

    def _emit_level(self, chunk, vad, ep):
        n = len(chunk) // PEAK_COL * PEAK_COL
        peaks = np.abs(chunk[:n]).reshape(-1, PEAK_COL).max(axis=1) if n else np.zeros(0, np.float32)
        self.emit('level', {'peaks': peaks, 'vad': ep.last_prob, 'speech': ep.speaking})

    def _run(self):
        cfg = self.cfg
        source = vad = session = None
        captured = 0
        result_dir = ''
        try:
            # Load the model and the VAD.
            self.emit('status', 'loading')
            if not cfg.model.path:
                raise RuntimeError('No model selected: choose or download one in Model')
            engine = self.holder.get(cfg)
            vad = create_vad(cfg.vad, get_models_dir(cfg))
            self.policy = create_policy(engine, cfg, self.terms_text)
            if self._stop.is_set():
                return
            # Open the audio source and the session files.
            if self.source is None:
                self.source = NativeCapture(cfg.audio)
            source = self.source
            if self.save and (cfg.save.save_audio or cfg.save.save_text):
                session = SessionWriter(cfg, engine.describe() + ' · ' + cfg.model.path)
            source.start()
            self.emit('started', {'engine': engine.describe(), 'session': session.dir if session else '',
                                  'can_translate': engine.caps.can_translate})
            t_flags = 0.0
            while not self._stop.is_set():
                # Pause: make the text final, then wait (draining or closing the device).
                if self._paused.is_set():
                    self._handle(self.policy.flush(commit=cfg.stream.commit_on_pause), session)
                    self.emit('status', 'paused')
                    if cfg.audio.pause_close:
                        source.stop()
                        self._resume.wait()
                        if not self._stop.is_set():
                            source.start()
                    else:
                        while not self._resume.wait(0.2):
                            source.read(0)  # drop what arrives while paused
                    if self._stop.is_set():
                        break
                    vad.reset()
                    self.emit('status', 'running')
                    continue
                # Read and analyse the next chunk.
                chunk = source.read(100)
                if source.finished and not len(chunk):
                    break
                if cfg.audio.gain_db:
                    chunk = np.clip(chunk * np.float32(10 ** (cfg.audio.gain_db / 20)), -1.0, 1.0)
                captured += len(chunk)
                self.audio_s = captured / SAMPLE_RATE
                if session is not None:
                    session.write_audio(chunk)
                vres = vad.process(chunk)
                out = self.policy.process(chunk, vres, source.idle_seconds())
                self._handle(out, session)
                if len(chunk):
                    self._emit_level(chunk, vad, self.policy.ep)
                if session is not None:
                    session.maybe_flush()
                # Device problems.
                now = time.monotonic()
                if now - t_flags > 0.5:
                    t_flags = now
                    flags = source.flags()
                    if flags & FLAG_STOPPED:
                        raise RuntimeError('The audio device stopped (unplugged or disabled)')
                    if flags & FLAG_REROUTED:
                        self.emit('warning', 'rerouted')
            self._handle(self.policy.flush(commit=True), session)
        except Exception as e:  # reported to the UI; the session still closes cleanly
            self.emit('error', f'{type(e).__name__}: {e}')
        finally:
            if source is not None:
                try:
                    source.stop()
                finally:
                    source.close()
            if vad is not None:
                vad.close()
            if session is not None:
                result_dir = session.close(captured / SAMPLE_RATE)
            self.holder.last_used = time.monotonic()
            self.emit('stopped', {'session': result_dir, 'audio_s': captured / SAMPLE_RATE, 'runs': self.runs,
                                  'busy_s': self.busy_s})
