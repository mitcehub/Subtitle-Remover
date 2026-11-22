"""开关控件：自绘胶囊轨道 + 圆形滑块。"""

from PyQt6.QtCore import Qt, QPropertyAnimation, QRectF, pyqtProperty
from PyQt6.QtGui import QColor, QPainter
from PyQt6.QtWidgets import QAbstractButton

from ui.theme import tokens


class Switch(QAbstractButton):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setFixedSize(38, 22)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._knob = 0.0
        self._anim = QPropertyAnimation(self, b'knob', self)
        self._anim.setDuration(110)
        self.toggled.connect(self._animate)

    def _get_knob(self):
        return self._knob

    def _set_knob(self, value):
        self._knob = value
        self.update()

    knob = pyqtProperty(float, _get_knob, _set_knob)

    def _animate(self, checked):
        self._anim.stop()
        self._anim.setStartValue(self._knob)
        self._anim.setEndValue(1.0 if checked else 0.0)
        self._anim.start()

    def setChecked(self, checked):
        super().setChecked(checked)
        self._knob = 1.0 if checked else 0.0
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        radius = self.height() / 2

        track = QColor(tokens.ACCENT if self.isChecked() else '#3A4558')
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(track)
        p.drawRoundedRect(QRectF(0, 0, self.width(), self.height()), radius, radius)

        margin = 3.0
        d = self.height() - margin * 2
        x = margin + self._knob * (self.width() - d - margin * 2)
        p.setBrush(QColor('#F2F6FC') if self.isChecked() else QColor('#C6CEDA'))
        p.drawEllipse(QRectF(x, margin, d, d))
