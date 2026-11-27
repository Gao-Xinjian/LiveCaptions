"""Entry point: single instance, Qt application, controller."""
import faulthandler
import getpass
import os
import sys

import platformdirs
from PySide6.QtCore import Qt
from PySide6.QtNetwork import QLocalServer, QLocalSocket
from PySide6.QtWidgets import QApplication

from livecaptions import APP_NAME
from livecaptions.config.store import load_config
from livecaptions.ui.fonts import add_cjk_fallback


def get_log_path():
    """Return the log file of the packaged app (the previous run's log is kept next to it)."""
    return os.path.join(platformdirs.user_log_dir(APP_NAME, appauthor=False), 'livecaptions.log')


def redirect_output_to_log():
    """
    Send Python's and lc_native's output to the log file: a packaged app has no console, so errors would vanish.
    Crashes in native code are logged too (faulthandler).
    """
    path = get_log_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        os.replace(path, os.path.splitext(path)[0] + '.previous.log')
    f = open(path, 'w', encoding='utf-8', buffering=1)
    for fd in (1, 2):   # the C runtime's stdout/stderr (whisper.cpp, ggml, the tap timings)
        try:
            os.dup2(f.fileno(), fd)
        except OSError:
            pass
    sys.stdout = sys.stderr = f
    faulthandler.enable(f)


def get_instance_key():
    """Return the local socket name of this user's instance."""
    user = ''.join(c for c in getpass.getuser() if c.isalnum()) or 'user'
    return f'{APP_NAME}-{user}'


def notify_running_instance(key):
    """Ask an already running instance to show itself; return whether one answered."""
    sock = QLocalSocket()
    sock.connectToServer(key)
    if sock.waitForConnected(300):
        sock.write(b'show')
        sock.waitForBytesWritten(300)
        sock.disconnectFromServer()
        return True
    return False


def exit_now(code):
    """
    End the process without the interpreter's teardown. The packaged macOS app hangs there after CTranslate2 has run
    (CI, 2026-11-23), and a process left behind keeps the single-instance socket, so the app could not start again.
    Called when settings and sessions are saved.
    """
    for f in (sys.stdout, sys.stderr):
        try:
            if f is not None:
                f.flush()
        except (OSError, ValueError):
            pass
    os._exit(code)


def main():
    if '--self-check' in sys.argv[1:]:
        from livecaptions.selfcheck import run_self_check
        exit_now(run_self_check(sys.argv[1:]))
    if getattr(sys, 'frozen', False):
        redirect_output_to_log()
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setQuitOnLastWindowClosed(False)  # the tray keeps the app alive
    app.setFont(add_cjk_fallback(app.font()))  # Chinese text in one font, not mixed per character
    key = get_instance_key()
    if notify_running_instance(key):
        return 0
    server = QLocalServer()
    QLocalServer.removeServer(key)  # a stale socket after a crash
    server.listen(key)

    from livecaptions.ui.app import Controller
    ctl = Controller(app, load_config())

    def on_connection():
        conn = server.nextPendingConnection()
        if conn is not None:
            conn.readyRead.connect(lambda: ctl.on_tray('show_main'))
    server.newConnection.connect(on_connection)
    code = app.exec()
    server.close()
    exit_now(code)


if __name__ == '__main__':
    sys.exit(main())
