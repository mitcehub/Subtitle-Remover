"""图标：用 QPainter 现场绘制，避免依赖第三方图标库与多套明暗资源。"""

from PyQt6.QtCore import Qt, QRectF, QPointF
from PyQt6.QtGui import (QIcon, QPixmap, QPainter, QPen, QColor, QPolygonF,
                         QFont)

from ui.theme import tokens

_CACHE = {}


def _make(name, size, color, text):
    key = (name, size, color, text)
    if key in _CACHE:
        return _CACHE[key]

    dpr = 2
    pm = QPixmap(size * dpr, size * dpr)
    pm.setDevicePixelRatio(dpr)
    pm.fill(Qt.GlobalColor.transparent)

    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.scale(dpr, dpr)

    c = QColor(color)
    s = float(size)
    pen = QPen(c)
    pen.setWidthF(max(1.4, s / 12))
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    p.setPen(pen)
    p.setBrush(Qt.BrushStyle.NoBrush)

    def pt(x, y):
        return QPointF(x * s, y * s)

    if name == 'play':
        p.setBrush(c)
        p.setPen(Qt.PenStyle.NoPen)
        p.drawPolygon(QPolygonF([pt(.34, .2), pt(.79, .5), pt(.34, .8)]))
    elif name == 'pause':
        p.setBrush(c)
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(QRectF(.32 * s, .22 * s, .13 * s, .56 * s), 1.5, 1.5)
        p.drawRoundedRect(QRectF(.55 * s, .22 * s, .13 * s, .56 * s), 1.5, 1.5)
    elif name == 'stop':
        p.setBrush(c)
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(QRectF(.3 * s, .3 * s, .4 * s, .4 * s), 2, 2)
    elif name == 'prev':
        p.setBrush(c)
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(QRectF(.26 * s, .24 * s, .1 * s, .52 * s), 1.5, 1.5)
        p.drawPolygon(QPolygonF([pt(.76, .24), pt(.4, .5), pt(.76, .76)]))
    elif name == 'next':
        p.setBrush(c)
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(QRectF(.64 * s, .24 * s, .1 * s, .52 * s), 1.5, 1.5)
        p.drawPolygon(QPolygonF([pt(.24, .24), pt(.6, .5), pt(.24, .76)]))
    elif name == 'skip_start':
        p.setBrush(c)
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(QRectF(.66 * s, .24 * s, .1 * s, .52 * s), 1.5, 1.5)
        p.drawPolygon(QPolygonF([pt(.58, .24), pt(.18, .5), pt(.58, .76)]))
    elif name == 'skip_end':
        p.setBrush(c)
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(QRectF(.24 * s, .24 * s, .1 * s, .52 * s), 1.5, 1.5)
        p.drawPolygon(QPolygonF([pt(.42, .24), pt(.82, .5), pt(.42, .76)]))
    elif name == 'folder':
        p.drawPolygon(QPolygonF([
            pt(.13, .3), pt(.13, .24), pt(.38, .24), pt(.46, .32),
            pt(.87, .32), pt(.87, .76), pt(.13, .76),
        ]))
    elif name == 'home':
        p.drawPolyline(QPolygonF([pt(.18, .5), pt(.5, .22), pt(.82, .5)]))
        p.drawPolyline(QPolygonF([
            pt(.28, .46), pt(.28, .79), pt(.72, .79), pt(.72, .46),
        ]))
    elif name == 'settings':
        for y, x in ((.3, .62), (.5, .38), (.7, .55)):
            p.drawLine(pt(.18, y), pt(.82, y))
            p.setBrush(c)
            p.drawEllipse(QPointF(x * s, y * s), .07 * s, .07 * s)
            p.setBrush(Qt.BrushStyle.NoBrush)
    elif name == 'speed':
        f = QFont()
        f.setPointSizeF(size * 0.44)
        f.setBold(True)
        p.setFont(f)
        p.setPen(c)
        p.drawText(QRectF(0, 0, s, s), Qt.AlignmentFlag.AlignCenter, text or '1x')
    else:
        p.drawEllipse(QRectF(.25 * s, .25 * s, .5 * s, .5 * s))

    p.end()
    icon = QIcon(pm)
    _CACHE[key] = icon
    return icon


def icon(name, size=18, color=None, text=None):
    """按名字取图标。"""
    return _make(name, size, color or tokens.TEXT_PRIMARY, text)


def speed_icon(speed, size=18):
    """播放速度图标：1x / 2x / 4x。"""
    return _make('speed', size, tokens.TEXT_PRIMARY, f'{speed}x')
