"""
Forwarding of the captions (PLAN.md §19.5): the bus numbers every caption event and hands it to the outputs.

Events are plain dicts, the same ones the share server sends to browsers and programs:
    {'type': 'final',   'seq': n, 'text': ..., 't': 12.3}               newly final text; '\\n' ends a line
    {'type': 'partial', 'text': ...}                                    the grey unfinished tail; replaces the last one
    {'type': 'line',    'seq': n, 'text': ..., 't0': 10.1, 't1': 14.8}  a finished caption line, as saved
    {'type': 'session', 'seq': n, 'state': 'started' | 'paused' | 'running' | 'stopped', 'name': ..., 'engine': ...,
     'time': '2026-11-27 10:15:00'}
Times are seconds of audio into the session. An output implements CaptionSink.deliver(); one with slow work (a
chatbot, a webhook) can derive from QueuedSink and implement handle() instead.
"""
import queue
import threading
import time
import traceback


class CaptionSink:
    """An output of the bus. deliver() is called on the Qt thread for every event and must return at once."""

    def deliver(self, event):
        raise NotImplementedError

    def close(self):
        """Stop the output: the app quits or the output is switched off."""


class QueuedSink(CaptionSink):
    """An output that does slow work: events wait in a queue and handle() runs on the sink's own thread."""

    def __init__(self, max_events=10000):
        self._queue = queue.Queue(max_events)
        self._thread = threading.Thread(target=self._loop, name=type(self).__name__, daemon=True)
        self._thread.start()

    def deliver(self, event):
        try:
            self._queue.put_nowait(event)
        except queue.Full:   # the output cannot keep up: drop events rather than hold up the captions
            pass

    def handle(self, event):
        """Process one event, on the sink's thread."""
        raise NotImplementedError

    def _loop(self):
        while True:
            event = self._queue.get()
            if event is None:
                return
            try:
                self.handle(event)
            except Exception:   # one failed event must not stop the output
                traceback.print_exc()

    def close(self):
        try:
            self._queue.put(None, timeout=1.0)
        except queue.Full:
            pass


class CaptionBus:
    """Numbers the caption events and hands them to every output. Used from the Qt thread only."""

    def __init__(self):
        self.sinks = []
        self.seq = 0
        self.partial_text = ''

    def add(self, sink):
        self.sinks.append(sink)

    def remove(self, sink):
        if sink in self.sinks:
            self.sinks.remove(sink)
        sink.close()

    def publish(self, event):
        if event['type'] != 'partial':
            self.seq += 1
            event['seq'] = self.seq
        for sink in list(self.sinks):
            sink.deliver(event)

    def send_final(self, text, t):
        if text:
            self.publish({'type': 'final', 'text': text, 't': round(t, 2)})

    def send_partial(self, text):
        if text != self.partial_text:
            self.partial_text = text
            self.publish({'type': 'partial', 'text': text})

    def send_lines(self, lines):
        for line in lines:
            self.publish({'type': 'line', 'text': line.text, 't0': round(line.t0, 2), 't1': round(line.t1, 2)})

    def send_session(self, state, **info):
        if state == 'stopped':
            self.send_partial('')
        self.publish({'type': 'session', 'state': state, 'time': time.strftime('%Y-%m-%d %H:%M:%S'), **info})

    def close(self):
        for sink in self.sinks:
            sink.close()
        self.sinks = []
