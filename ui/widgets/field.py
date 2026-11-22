"""设置字段：下拉框 / 滑杆 / 开关 / 路径选择。"""

from PyQt6.QtCore import Qt, QTimer, QPointF
from PyQt6.QtGui import QColor, QPainter, QPen, QPolygonF
from PyQt6.QtWidgets import (QComboBox, QFileDialog, QHBoxLayout, QLabel,
                             QSlider, QVBoxLayout, QWidget)

from infra.config import config
from ui.theme import tokens
from ui.widgets.button import GhostButton
from ui.widgets.toggle import Switch


class ComboBox(QComboBox):
    """去掉原生箭头，自绘一个与深色主题匹配的折角。"""

    def paintEvent(self, event):
        super().paintEvent(event)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(QColor(tokens.TEXT_MUTED))
        pen.setWidthF(1.5)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        p.setPen(pen)
        cx = self.width() - 14
        cy = self.height() / 2 - 1
        p.drawPolyline(QPolygonF([
            QPointF(cx - 4, cy - 2), QPointF(cx, cy + 2), QPointF(cx + 4, cy - 2),
        ]))


class _Field(QWidget):
    """标题 + 可选描述 + 控件主体。"""

    def __init__(self, title, desc=None, parent=None):
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(6)
        self._layout.addWidget(QLabel(title, objectName='FieldTitle'))
        if desc:
            hint = QLabel(desc, objectName='FieldDesc')
            hint.setWordWrap(True)
            self._layout.addWidget(hint)

    def add_body(self, widget):
        self._layout.addWidget(widget)


class ComboBoxField(_Field):
    def __init__(self, title, config_item, labels, desc=None, parent=None):
        super().__init__(title, desc, parent)
        self._item = config_item
        self._options = list(config_item.validator.options)

        self.combo = ComboBox()
        self.combo.addItems(labels)
        try:
            self.combo.setCurrentIndex(self._options.index(config_item.value))
        except ValueError:
            pass
        self.combo.currentIndexChanged.connect(self._on_changed)
        self.add_body(self.combo)

    def _on_changed(self, index):
        if 0 <= index < len(self._options):
            config.set(self._item, self._options[index])


class RangeField(_Field):
    """整数滑杆字段；拖动时只更新显示，停手后才落盘。"""

    def __init__(self, title, config_item, desc=None, parent=None):
        super().__init__(title, desc, parent)
        self._item = config_item
        validator = config_item.validator

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(10)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(int(validator.min), int(validator.max))
        self.slider.setValue(int(config_item.value))
        self.slider.setSingleStep(1)
        self.slider.setPageStep(1)

        self.value_label = QLabel(str(config_item.value), objectName='FieldValue')
        self.value_label.setFixedWidth(44)
        self.value_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        row.addWidget(self.slider, 1)
        row.addWidget(self.value_label)
        self._layout.addLayout(row)

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(250)
        self._timer.timeout.connect(self._commit)
        self.slider.valueChanged.connect(self._on_value)

    def _on_value(self, value):
        self.value_label.setText(str(value))
        self._timer.start()

    def _commit(self):
        config.set(self._item, self.slider.value())


class SwitchField(QWidget):
    """左标题右开关。"""

    def __init__(self, title, config_item, desc=None, parent=None):
        super().__init__(parent)
        self._item = config_item

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(12)

        text = QVBoxLayout()
        text.setContentsMargins(0, 0, 0, 0)
        text.setSpacing(4)
        text.addWidget(QLabel(title, objectName='FieldTitle'))
        if desc:
            hint = QLabel(desc, objectName='FieldDesc')
            hint.setWordWrap(True)
            text.addWidget(hint)
        row.addLayout(text, 1)

        self.switch = Switch()
        self.switch.setChecked(bool(config_item.value))
        self.switch.toggled.connect(self._on_toggled)
        row.addWidget(self.switch, 0, Qt.AlignmentFlag.AlignTop)

    def _on_toggled(self, checked):
        config.set(self._item, bool(checked))


class PathField(_Field):
    """路径选择：文本 + 选择按钮。"""

    def __init__(self, title, config_item, button_text, default_text='', parent=None):
        super().__init__(title, None, parent)
        self._item = config_item
        self._default_text = default_text

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(10)

        self.path_label = QLabel('', objectName='FieldDesc')
        self.path_label.setWordWrap(True)
        self.path_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        row.addWidget(self.path_label, 1)

        self.button = GhostButton(button_text)
        self.button.clicked.connect(self._choose)
        row.addWidget(self.button, 0, Qt.AlignmentFlag.AlignTop)

        self._layout.addLayout(row)
        self._refresh()

    def _refresh(self):
        value = self._item.value
        self.path_label.setText(value if value else self._default_text)
        self.path_label.setToolTip(value or '')

    def _choose(self):
        start = self._item.value or ''
        folder = QFileDialog.getExistingDirectory(self, self.button.text(), start)
        if not folder:
            return
        config.set(self._item, folder)
        self._refresh()
