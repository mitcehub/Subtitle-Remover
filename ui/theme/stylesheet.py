"""由设计令牌生成全局 QSS。"""

from string import Template

from ui.theme import tokens


_QSS = Template("""
QWidget {
    color: $TEXT_PRIMARY;
    font-family: $FONT_FAMILY;
    font-size: 13px;
}

QMainWindow, QWidget#Root, QDialog {
    background: $BG_APP;
}

/* ---------- 卡片 ---------- */
QFrame#Card {
    background: $BG_CARD;
    border: 1px solid $BORDER;
    border-radius: $RADIUS;
}
QLabel#CardTitle {
    color: $TEXT_MUTED;
    font-size: 11px;
    font-weight: 600;
}
QFrame#Divider {
    background: $DIVIDER;
    border: none;
    max-height: 1px;
    min-height: 1px;
}

/* ---------- 文本 ---------- */
QLabel#FieldTitle { color: $TEXT_PRIMARY; font-size: 13px; }
QLabel#FieldDesc  { color: $TEXT_MUTED; font-size: 11px; }
QLabel#FieldValue { color: $ACCENT; font-family: $FONT_MONO; font-size: 12px; }
QLabel#Timecode   { color: $TEXT_SECONDARY; font-family: $FONT_MONO; font-size: 12px; }
QLabel#HwBadge    { color: $TEXT_MUTED; font-size: 11px; }
QLabel#Version    { color: $TEXT_MUTED; font-size: 11px; }
QLabel#AppTitle   { color: $TEXT_PRIMARY; font-size: 13px; font-weight: 600; }
QLabel#Toast {
    background: $BG_ELEVATED;
    border: 1px solid $BORDER_STRONG;
    border-radius: $RADIUS;
    padding: 10px 18px;
    color: $TEXT_PRIMARY;
}

/* ---------- 顶部栏 / 导航 ---------- */
QFrame#TitleBar {
    background: $BG_PANEL;
    border-bottom: 1px solid $BORDER;
}
QFrame#NavRail {
    background: #121722;
    border-right: 1px solid $BORDER;
}

/* ---------- 按钮 ---------- */
QPushButton {
    background: $BG_ELEVATED;
    border: 1px solid $BORDER;
    border-radius: $RADIUS_SM;
    padding: 6px 14px;
    color: $TEXT_PRIMARY;
}
QPushButton:hover { background: #313C50; border-color: $BORDER_STRONG; }
QPushButton:pressed { background: #263042; }
QPushButton:disabled {
    background: #1E2531;
    border-color: #262F3F;
    color: $TEXT_MUTED;
}

QPushButton[kind="primary"] {
    background: $ACCENT;
    border-color: $ACCENT;
    color: $TEXT_ON_ACCENT;
    font-weight: 600;
}
QPushButton[kind="primary"]:hover { background: $ACCENT_HOVER; border-color: $ACCENT_HOVER; }
QPushButton[kind="primary"]:pressed { background: $ACCENT_PRESSED; border-color: $ACCENT_PRESSED; }
QPushButton[kind="primary"]:disabled {
    background: #2A3446;
    border-color: $BORDER;
    color: $TEXT_MUTED;
}

QPushButton[kind="ghost"] {
    background: transparent;
    border: 1px solid transparent;
    color: $TEXT_SECONDARY;
    padding: 5px 10px;
}
QPushButton[kind="ghost"]:hover { background: $BG_CARD_HOVER; color: $TEXT_PRIMARY; }
QPushButton[kind="ghost"][active="true"] {
    background: $ACCENT_SOFT;
    color: $ACCENT;
}
QPushButton[kind="ghost"]:disabled {
    background: transparent;
    color: $TEXT_MUTED;
}

QPushButton[kind="danger"] {
    background: transparent;
    border: 1px solid $DANGER;
    color: $DANGER;
}
QPushButton[kind="danger"]:hover { background: rgba(229, 83, 75, 0.14); }

/* ---------- 图标按钮 ---------- */
QToolButton {
    background: transparent;
    border: none;
    border-radius: $RADIUS_SM;
    padding: 4px;
}
QToolButton:hover { background: $BG_CARD_HOVER; }
QToolButton:pressed { background: $BG_ELEVATED; }
QToolButton:disabled { background: transparent; }

QToolButton[kind="nav"] {
    padding: 0;
    border-radius: $RADIUS;
}
QToolButton[kind="nav"]:hover { background: $BG_CARD_HOVER; }
QToolButton[kind="nav"]:checked { background: $ACCENT_SOFT; }

/* ---------- 下拉框 ---------- */
QComboBox {
    background: $BG_INPUT;
    border: 1px solid $BORDER;
    border-radius: $RADIUS_SM;
    padding: 5px 10px;
    min-height: 22px;
}
QComboBox:hover { border-color: $BORDER_STRONG; }
QComboBox:focus { border-color: $ACCENT; }
QComboBox::drop-down { border: none; width: 24px; }
QComboBox::down-arrow { image: none; width: 0; height: 0; }
QComboBox QAbstractItemView {
    background: $BG_ELEVATED;
    border: 1px solid $BORDER_STRONG;
    border-radius: $RADIUS_SM;
    padding: 4px;
    selection-background-color: $ACCENT_SOFT;
    selection-color: $TEXT_PRIMARY;
}

/* ---------- 滑块 ---------- */
QSlider { min-height: 18px; }
QSlider::groove:horizontal {
    height: 4px;
    background: $BG_INPUT;
    border-radius: 2px;
}
QSlider::sub-page:horizontal {
    background: $ACCENT;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    width: 12px;
    height: 12px;
    margin: -5px 0;
    border-radius: 7px;
    background: $TEXT_PRIMARY;
    border: 2px solid $ACCENT;
}
QSlider::handle:horizontal:hover { background: $ACCENT_HOVER; }

/* ---------- 日志 ---------- */
QTextEdit#LogView {
    background: $BG_INPUT;
    border: 1px solid $BORDER;
    border-radius: $RADIUS_SM;
    padding: 6px;
    color: $TEXT_SECONDARY;
    selection-background-color: $ACCENT_SOFT;
}

/* ---------- 滚动条 ---------- */
QScrollBar:vertical {
    background: transparent;
    width: 10px;
    margin: 2px;
}
QScrollBar::handle:vertical {
    background: #39445A;
    border-radius: 4px;
    min-height: 28px;
}
QScrollBar::handle:vertical:hover { background: #4A5771; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }

QScrollBar:horizontal {
    background: transparent;
    height: 10px;
    margin: 2px;
}
QScrollBar::handle:horizontal {
    background: #39445A;
    border-radius: 4px;
    min-width: 28px;
}
QScrollBar::handle:horizontal:hover { background: #4A5771; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal { background: transparent; }

/* ---------- 其他 ---------- */
QToolTip {
    background: $BG_ELEVATED;
    color: $TEXT_PRIMARY;
    border: 1px solid $BORDER_STRONG;
    padding: 4px 8px;
}
QMenu {
    background: $BG_ELEVATED;
    border: 1px solid $BORDER_STRONG;
    border-radius: $RADIUS_SM;
    padding: 4px;
}
QMenu::item {
    padding: 6px 18px;
    border-radius: 4px;
}
QMenu::item:selected { background: $ACCENT_SOFT; }
QSplitter::handle { background: transparent; }
QSplitter::handle:hover { background: $ACCENT_SOFT; }
QScrollArea { border: none; background: transparent; }
QScrollArea > QWidget > QWidget { background: transparent; }
""")


def build_stylesheet():
    """返回应用级 QSS。"""
    values = {k: getattr(tokens, k) for k in dir(tokens) if k.isupper()}
    return _QSS.substitute(values)
