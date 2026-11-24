"""日志视图：只读富文本 + 自动滚动记忆。"""

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QTextOption
from PyQt6.QtWidgets import QTextEdit


class LogView(QTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('LogView')
        self.setReadOnly(True)
        self.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        self.setWordWrapMode(QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.document().setDocumentMargin(6)
        self.document().setMaximumBlockCount(2000)
        self.setMinimumHeight(80)
        self.auto_scroll = True
        self.verticalScrollBar().valueChanged.connect(self._on_scroll)

    def append_rich(self, html):
        self.append(html)
        if self.auto_scroll:
            sb = self.verticalScrollBar()
            sb.setValue(sb.maximum())

    def _on_scroll(self, value):
        sb = self.verticalScrollBar()
        if value >= sb.maximum():
            self.auto_scroll = True
        elif self.auto_scroll and value < sb.maximum():
            self.auto_scroll = False
