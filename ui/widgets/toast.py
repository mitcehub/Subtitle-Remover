"""轻量浮动提示（替代 InfoBar）。"""

from PyQt6.QtCore import QEvent, QTimer, Qt
from PyQt6.QtWidgets import QLabel


class Toast(QLabel):
    def __init__(self, parent):
        super().__init__(parent)
        self.setObjectName('Toast')
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.hide()

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.hide)

        parent.installEventFilter(self)

    def show_message(self, text, msec=4000):
        self.setText(text)
        self.adjustSize()
        self._reposition()
        self.show()
        self.raise_()
        self._timer.start(msec)

    def _reposition(self):
        parent = self.parentWidget()
        if parent is None:
            return
        self.move((parent.width() - self.width()) // 2, 56)

    def eventFilter(self, obj, event):
        if obj is self.parentWidget() and event.type() == QEvent.Type.Resize:
            self._reposition()
        return super().eventFilter(obj, event)
