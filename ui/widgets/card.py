"""卡片容器。"""

from PyQt6.QtWidgets import QFrame, QLabel, QVBoxLayout


class Card(QFrame):
    """带可选标题的深色卡片。"""

    def __init__(self, title=None, parent=None, margins=(16, 14, 16, 16), spacing=12):
        super().__init__(parent)
        self.setObjectName('Card')
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(*margins)
        self._layout.setSpacing(spacing)
        if title:
            self._layout.addWidget(QLabel(title, objectName='CardTitle'))

    def add(self, widget, stretch=0):
        self._layout.addWidget(widget, stretch)
        return widget

    def add_layout(self, layout):
        self._layout.addLayout(layout)

    def add_stretch(self, stretch=1):
        self._layout.addStretch(stretch)

    @property
    def body(self):
        return self._layout


def divider():
    """字段之间的细分隔线。"""
    line = QFrame()
    line.setObjectName('Divider')
    line.setFixedHeight(1)
    return line
