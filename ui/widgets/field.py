"""设置字段：下拉框 / 滑杆 / 开关 / 路径选择。"""

from PyQt6.QtCore import Qt, QTimer, QPointF
from PyQt6.QtGui import QColor, QPainter, QPen, QPolygonF, QIntValidator
from PyQt6.QtWidgets import (QComboBox, QFileDialog, QHBoxLayout, QLabel,
                             QLineEdit, QSlider, QVBoxLayout, QWidget)

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


class NoWheelSlider(QSlider):
    """忽略滚轮事件的滑杆：鼠标在设置面板滚动时不应误改参数值。"""

    def wheelEvent(self, event):
        event.ignore()


class RangeField(_Field):
    """整数参数字段：滑杆 + 可输入数字框；改值后延迟落盘。"""

    def __init__(self, title, config_item, desc=None, parent=None):
        super().__init__(title, desc, parent)
        self._item = config_item
        validator = config_item.validator
        self._min = int(validator.min)
        self._max = int(validator.max)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(10)

        self.slider = NoWheelSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(self._min, self._max)
        self.slider.setValue(int(config_item.value))
        self.slider.setSingleStep(1)
        self.slider.setPageStep(1)

        # 数字框：可直接键入，限定在 [min, max]，回车或失焦后生效
        self.value_edit = QLineEdit(str(config_item.value), objectName='FieldValueEdit')
        self.value_edit.setFixedWidth(52)
        self.value_edit.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.value_edit.setValidator(QIntValidator(self._min, self._max, self))
        self.value_edit.setToolTip(f'可输入 {self._min} ~ {self._max} 的整数')

        row.addWidget(self.slider, 1)
        row.addWidget(self.value_edit)
        self._layout.addLayout(row)

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(250)
        self._timer.timeout.connect(self._commit)
        self.slider.valueChanged.connect(self._on_slider)
        self.value_edit.editingFinished.connect(self._on_edit)

    def _on_slider(self, value):
        # 滑杆变化时刷新输入框（阻塞信号避免递归）
        self.value_edit.blockSignals(True)
        self.value_edit.setText(str(value))
        self.value_edit.blockSignals(False)
        self._timer.start()

    def _on_edit(self):
        # 手动输入：校验并夹紧到范围，同步回滑杆
        try:
            value = int(self.value_edit.text())
        except ValueError:
            value = self.slider.value()
        value = max(self._min, min(self._max, value))
        self.value_edit.setText(str(value))
        if self.slider.value() != value:
            self.slider.setValue(value)  # 会触发 _on_slider → 重启定时器
        else:
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
