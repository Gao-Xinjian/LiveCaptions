"""Under the sharing settings: whether the page is running, its links, the access code and the program interface."""
from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtGui import QDesktopServices, QGuiApplication
from PySide6.QtWidgets import QGridLayout, QGroupBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from livecaptions.ui.i18n import tr


class SharePanel(QGroupBox):
    def __init__(self, ctl):
        super().__init__(tr('Links'))
        self.ctl = ctl
        lay = QVBoxLayout(self)
        self.status = QLabel()
        self.status.setWordWrap(True)
        lay.addWidget(self.status)
        self.links = QWidget()
        self.links_lay = QGridLayout(self.links)
        self.links_lay.setContentsMargins(0, 0, 0, 0)
        self.links_lay.setColumnStretch(1, 1)   # the link takes the width, not the buttons
        lay.addWidget(self.links)
        row = QHBoxLayout()
        self.code = QLabel()
        self.code.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.b_code = QPushButton(tr('New code'))
        self.b_code.clicked.connect(self.ctl.new_share_code)
        row.addWidget(self.code)
        row.addWidget(self.b_code)
        row.addStretch(1)
        lay.addLayout(row)
        api = QLabel(tr('For programs, such as a chatbot: /events (live events), /transcript.txt and /transcript.json '
                        'under the same address.'))
        api.setWordWrap(True)
        lay.addWidget(api)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_status)
        self.timer.start(1000)
        self.refresh()

    def refresh(self):
        """Show the links of the running server (after it started, stopped or moved) and the access code."""
        while self.links_lay.count():
            self.links_lay.takeAt(0).widget().deleteLater()
        share = self.ctl.share
        for r, (where, url) in enumerate(share.get_links() if share is not None else []):
            label = QLabel(tr('Local network') + f' ({where})' if where else tr('This computer'))
            link = QLabel(url)
            link.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            b_copy = QPushButton(tr('Copy'))
            b_copy.clicked.connect(lambda _=False, u=url: QGuiApplication.clipboard().setText(u))
            b_open = QPushButton(tr('Open'))
            b_open.clicked.connect(lambda _=False, u=url: QDesktopServices.openUrl(QUrl(u)))
            for c, w in enumerate((label, link, b_copy, b_open)):
                self.links_lay.addWidget(w, r, c)
        sc = self.ctl.cfg.share
        show_code = share is not None and sc.lan and sc.use_code
        self.code.setText(tr('Access code:') + f' {sc.code}')
        for w in (self.code, self.b_code):
            w.setVisible(show_code)
        self.update_status()
        main = getattr(self.ctl, 'main', None)
        if main is not None:   # the links can make the page wider; measure once Qt has laid them out
            QTimer.singleShot(0, main.update_page_width)

    def update_status(self):
        share = self.ctl.share
        if not self.ctl.cfg.share.enabled:
            text = tr('Sharing is off.')
        elif share is None:
            text = tr('Could not start the page:') + ' ' + self.ctl.share_error
        else:
            text = tr('The page is running.') + ' ' + tr('Connected:') + f' {share.viewers}'
            if share.lan:
                text += '\n' + tr('Other devices must be on the same network and use a link below.')
        self.status.setText(text)
