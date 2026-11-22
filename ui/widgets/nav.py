"""左侧导航按钮：图标 + 选中指示条。"""

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtWidgets import QFrame, QToolButton

from ui.theme import tokens
from ui.widgets import icons


class NavButton(QToolButton):
    def __init__(self, icon_name, tooltip='', parent=None):
        super().__init__(parent)
        self._icon_name = icon_name
        self.setProperty('kind', 'nav')
        self.setCheckable(True)
        self.setAutoExclusive(True)
        self.setFixedSize(38, 38)
        self.setIconSize(QSize(19, 19))
        self.setIcon(icons.icon(icon_name, 19, tokens.TEXT_SECONDARY))
        self.setToolTip(tooltip)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        self._indicator = QFrame(self)
        self._indicator.setStyleSheet(
            f'background: {tokens.ACCENT}; border-radius: 1px;')
        self._indicator.setGeometry(0, 10, 3, 18)
        self._indicator.hide()

        self.toggled.connect(self._on_toggled)

    def _on_toggled(self, checked):
        self.setIcon(icons.icon(
            self._icon_name, 19,
            tokens.ACCENT if checked else tokens.TEXT_SECONDARY))
        self._indicator.setVisible(checked)
