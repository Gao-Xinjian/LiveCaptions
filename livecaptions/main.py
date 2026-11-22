"""Entry point: single instance, Qt application, controller."""
import getpass
import sys

from PySide6.QtCore import Qt
from PySide6.QtNetwork import QLocalServer, QLocalSocket
from PySide6.QtWidgets import QApplication

from livecaptions import APP_NAME
from livecaptions.config.store import load_config


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


def main():
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setQuitOnLastWindowClosed(False)  # the tray keeps the app alive
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
    return app.exec()


if __name__ == '__main__':
    sys.exit(main())
