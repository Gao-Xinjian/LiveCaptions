"""
The share server (PLAN.md §19.5): a read-only web page with the live captions, an event stream and the transcript,
for this computer or, when allowed, the local network. Python's own http.server, one thread per connection.

    /                  the caption page for browsers
    /events            Server-Sent Events: a snapshot, then final, partial, line and session events (forward/bus.py),
                       and while nothing happens a ping every 10 s with the session, the number of the last event and
                       the length of the final text (characters), which the page checks against what it got
    /transcript.txt    the final text of the current session
    /transcript.json   {'session': ..., 'text': ..., 'lines': [{'text', 't0', 't1', 'seq'}], 'partial': ...}
    /api/state         {'session': ..., 'viewers': n, 'seq': n}
On the local network these need ?code=<access code> when the code is on; requests from this computer never do.
Nothing can be sent to the app through it.
"""
import json
import os
import secrets
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

import psutil

from livecaptions.forward.bus import CaptionSink

WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'web')
MAX_STREAMS = 64        # event streams at once: viewers and programs
PING_S = 10.0           # a ping keeps idle connections open and lets the page see that its connection still works
WRITE_TIMEOUT_S = 10.0  # a viewer that stops reading is dropped after this
FORBIDDEN_PAGE = '''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>LiveCaptions</title><body style="font-family:system-ui,sans-serif;padding:2em">
<p>Access code required. / 需要访问码。</p>
<form method="get" action="/"><input name="code" inputmode="numeric" autofocus> <button>OK</button></form>'''.encode('utf-8')


VIRTUAL_ADAPTERS = ('vethernet', 'wsl', 'hyper-v', 'virtualbox', 'vbox', 'vmware', 'docker', 'loopback', 'virbr',
                    'veth', 'br-', 'utun', 'bridge')   # name parts of adapters other devices cannot reach


def get_primary_address():
    """Return the IPv4 address this computer uses for its main network connection, or ''."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(('192.0.2.1', 9))   # a UDP "connect" only picks the route; nothing is sent
            return s.getsockname()[0]
    except OSError:
        return ''


def get_lan_addresses():
    """Return [(adapter name, IPv4 address)] other devices can reach, the main connection first."""
    stats = psutil.net_if_stats()
    out = []
    for name, addrs in psutil.net_if_addrs().items():
        if (name in stats and not stats[name].isup) or any(v in name.lower() for v in VIRTUAL_ADAPTERS):
            continue
        for a in addrs:
            if a.family == socket.AF_INET and not a.address.startswith(('127.', '169.254.')):
                out.append((name, a.address))
    primary = get_primary_address()
    return sorted(out, key=lambda x: (x[1] != primary, x[0]))


def make_code():
    """Return a new 6-digit access code."""
    return f'{secrets.randbelow(10 ** 6):06d}'


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = sys.platform != 'win32'   # on Windows it would let two programs share one port


class _StreamState:
    """Where one event stream is: its session, its position in the log, and the grey text it last got."""

    def __init__(self, sid, pos):
        self.sid = sid
        self.pos = pos            # index into the log; None = send a snapshot first
        self.partial_ver = -1
        self.t_partial = 0.0


class ShareServer(CaptionSink):
    """Keeps the current session's caption events and serves them; deliver() only stores them and wakes the streams."""

    def __init__(self, share_cfg):
        """
        Args:
            share_cfg: ShareCfg; lan and port are fixed for this server, the other options are read live

        Raises:
            OSError: the port is in use or not allowed
        """
        self.cfg = share_cfg
        self.lan = share_cfg.lan
        self.port = share_cfg.port
        self.viewers = 0
        self._cond = threading.Condition()
        self._closing = False
        self._session_id = 0
        self._session = {}
        self._log = []        # session, final and line events of the current session, in order
        self._text = []       # the final text pieces
        self._chars = 0       # their total length in characters, sent with pings so the page can check its text
        self._lines = []
        self._seq = 0
        self._partial = ''
        self._partial_ver = 0
        with open(os.path.join(WEB_DIR, 'index.html'), 'rb') as f:
            self.page = f.read()
        handler = type('Handler', (_Handler,), {'share': self})
        self.httpd = _Server(('0.0.0.0' if self.lan else '127.0.0.1', self.port), handler)
        self._thread = threading.Thread(target=self.httpd.serve_forever, args=(0.2,), name='share-server', daemon=True)
        self._thread.start()

    # The bus side (Qt thread).

    def deliver(self, event):
        with self._cond:
            kind = event['type']
            if kind == 'partial':
                self._partial = event['text']
                self._partial_ver += 1
            else:
                if kind == 'session' and event['state'] == 'started':   # a new session: viewers start afresh
                    self._session_id += 1
                    self._log, self._text, self._lines = [], [], []
                    self._chars = 0
                    self._partial = ''
                    self._partial_ver += 1
                if kind == 'session':
                    self._session = event
                elif kind == 'final':
                    self._text.append(event['text'])
                    self._chars += len(event['text'])
                elif kind == 'line':
                    self._lines.append(event)
                self._log.append(event)
                self._seq = event['seq']
            self._cond.notify_all()

    def get_links(self):
        """Return [(where, url)]: '' for this computer, then the adapter name of each local network address if allowed."""
        links = [('', f'http://127.0.0.1:{self.port}/')]
        if self.lan:
            query = f'?code={self.cfg.code}' if self.cfg.use_code and self.cfg.code else ''
            links += [(name, f'http://{ip}:{self.port}/{query}') for name, ip in get_lan_addresses()]
        return links

    def close(self):
        with self._cond:
            self._closing = True
            self._cond.notify_all()
        self.httpd.shutdown()
        self.httpd.server_close()

    # The connection side (one thread per connection).

    def add_viewer(self):
        with self._cond:
            if self.viewers >= MAX_STREAMS:
                return False
            self.viewers += 1
            return True

    def remove_viewer(self):
        with self._cond:
            self.viewers -= 1

    def get_text(self):
        with self._cond:
            return ''.join(self._text)

    def get_transcript(self):
        with self._cond:
            return {'session': self._session, 'text': ''.join(self._text), 'partial': self._partial,
                    'lines': [{k: e[k] for k in ('text', 't0', 't1', 'seq')} for e in self._lines]}

    def get_state(self):
        with self._cond:
            return {'session': self._session, 'viewers': self.viewers, 'seq': self._seq}

    def iter_stream(self, last_id):
        """
        Yield the SSE text of one connection: a snapshot, or only what it missed when it reconnects with the id of
        its last event (EventSource does this by itself), then the events as they come, until the server closes.
        """
        with self._cond:
            st = _StreamState(self._session_id, None)
            sid, _, seq = (last_id or '').partition('-')
            if sid == str(self._session_id) and seq.isdigit():
                st.pos = next((i for i, e in enumerate(self._log) if e['seq'] > int(seq)), len(self._log))
                st.partial_ver = self._partial_ver - 1   # resend the grey text
        while True:
            with self._cond:
                chunks = self._collect(st)
            if chunks is None:
                return
            yield ''.join(chunks)

    def _collect(self, st):
        """Wait (lock held) until one stream has something to send and return it as SSE text; None = closing."""
        while not self._closing:
            send_partial = self.cfg.send_partial
            if st.pos is None or st.sid != self._session_id:
                st.sid, st.pos = self._session_id, len(self._log)
                st.partial_ver, st.t_partial = self._partial_ver, time.monotonic()
                snap = {'type': 'snapshot', 'session': self._session, 'text': ''.join(self._text),
                        'partial': self._partial if send_partial else '', 'seq': self._seq}
                return [format_event('snapshot', snap, f'{st.sid}-{self._seq}')]
            partial_due = send_partial and st.partial_ver != self._partial_ver
            interval = max(0.1, self.cfg.partial_interval_s)
            if st.pos < len(self._log) or (partial_due and time.monotonic() - st.t_partial >= interval):
                chunks = [format_event(e['type'], e, f'{st.sid}-{e["seq"]}') for e in self._log[st.pos:]]
                st.pos = len(self._log)
                if partial_due:   # with new final text the grey tail goes at once, or the two would overlap
                    chunks.append(format_event('partial', {'type': 'partial', 'text': self._partial}))
                    st.partial_ver, st.t_partial = self._partial_ver, time.monotonic()
                return chunks
            timeout = max(0.01, interval - (time.monotonic() - st.t_partial)) if partial_due else PING_S
            if not self._cond.wait(timeout) and not partial_due:   # nothing new: tell the page where it should be
                return [format_event('ping', {'type': 'ping', 'session': st.sid, 'seq': self._seq, 'chars': self._chars})]
        return None


def format_event(name, data, event_id=None):
    """Return one Server-Sent Event; events with an id let a reconnecting client resume."""
    head = f'id: {event_id}\n' if event_id else ''
    return f'{head}event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n'


class _Handler(BaseHTTPRequestHandler):
    share = None   # the ShareServer, set on a subclass per server

    def log_message(self, fmt, *args):   # no access log in the app's log
        pass

    def do_GET(self):
        url = urlsplit(self.path)
        query = parse_qs(url.query)
        if url.path == '/favicon.ico':
            return self._send(204, b'', 'text/plain')
        if not self._is_allowed(query):
            return self._send(403, FORBIDDEN_PAGE, 'text/html; charset=utf-8')
        if url.path in ('/', '/index.html'):
            return self._send(200, self.share.page, 'text/html; charset=utf-8')
        if url.path == '/events':
            return self._stream(query)
        if url.path == '/transcript.txt':
            return self._send(200, self.share.get_text().encode('utf-8'), 'text/plain; charset=utf-8')
        if url.path == '/transcript.json':
            return self._send_json(self.share.get_transcript())
        if url.path == '/api/state':
            return self._send_json(self.share.get_state())
        return self._send(404, b'not found', 'text/plain')

    def _is_allowed(self, query):
        cfg = self.share.cfg
        if not self.share.lan or not cfg.use_code or not cfg.code:
            return True
        host = self.client_address[0]
        if host.startswith('127.') or host == '::1':   # this computer
            return True
        return secrets.compare_digest(query.get('code', [''])[0], cfg.code)

    def _send(self, status, body, content_type):
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, data):
        self._send(200, json.dumps(data, ensure_ascii=False).encode('utf-8'), 'application/json; charset=utf-8')

    def _stream(self, query):
        share = self.share
        if not share.add_viewer():
            return self._send(503, b'too many connections', 'text/plain')
        try:
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream; charset=utf-8')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.connection.settimeout(WRITE_TIMEOUT_S)
            self.wfile.write(b'retry: 2000\n\n')
            self.wfile.flush()
            last_id = self.headers.get('Last-Event-ID') or query.get('last', [''])[0]
            for text in share.iter_stream(last_id):
                self.wfile.write(text.encode('utf-8'))
                self.wfile.flush()
        except (OSError, ValueError):   # the viewer went away or stopped reading
            pass
        finally:
            share.remove_viewer()
