"""按钮组件：主按钮 / 次按钮 / 幽灵按钮 / 图标按钮。"""

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtWidgets import QPushButton, QToolButton

from ui.widgets import icons


class _Button(QPushButton):
    def __init__(self, text, kind, icon_name=None, parent=None):
        super().__init__(text, parent)
        self.setProperty('kind', kind)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight(32)
        if icon_name:
            self.setIcon(icons.icon(icon_name, 16))
            self.setIconSize(QSize(16, 16))


class PrimaryButton(_Button):
    def __init__(self, text, icon_name=None, parent=None):
        super().__init__(text, 'primary', icon_name, parent)


class SecondaryButton(_Button):
    def __init__(self, text, icon_name=None, parent=None):
        super().__init__(text, 'secondary', icon_name, parent)


class GhostButton(_Button):
    def __init__(self, text, icon_name=None, parent=None):
        super().__init__(text, 'ghost', icon_name, parent)


class DangerButton(_Button):
    def __init__(self, text, icon_name=None, parent=None):
        super().__init__(text, 'danger', icon_name, parent)


class IconButton(QToolButton):
    """方形图标按钮（播放控制栏使用）。"""

    def __init__(self, icon_name, tooltip='', parent=None, size=18):
        super().__init__(parent)
        self.setIcon(icons.icon(icon_name, size))
        self.setIconSize(QSize(size, size))
        self.setToolTip(tooltip)
        self.setFixedSize(30, 30)
        self.setCursor(Qt.CursorShape.PointingHandCursor)


def set_active(button, active):
    """切换幽灵按钮的选中态（适应/填充等）。"""
    button.setProperty('active', 'true' if active else 'false')
    button.style().unpolish(button)
    button.style().polish(button)
