"""深色影音风设计令牌：颜色、圆角、字体、尺寸。

深蓝灰底 + 琥珀/橘色强调色，暖冷对比让预览区成为视觉主体，
同时避免青色系在长时间注视下的视觉疲劳。
"""

# ---- 背景层级（拉开明度差，增强层次） ----
BG_APP = "#10141C"          # 窗口底色（最深）
BG_PANEL = "#181E2A"        # 面板 / 顶栏
BG_CARD = "#212938"         # 卡片
BG_CARD_HOVER = "#2A3445"   # 卡片悬停
BG_INPUT = "#131924"        # 输入框 / 滑块槽
BG_ELEVATED = "#2E394D"     # 悬浮层（菜单、默认按钮）

# ---- 描边 ----
BORDER = "#323C4E"
BORDER_STRONG = "#465268"
DIVIDER = "#2B3444"

# ---- 文本 ----
TEXT_PRIMARY = "#EDF1F7"
TEXT_SECONDARY = "#AEB9CB"
TEXT_MUTED = "#7C889C"
TEXT_ON_ACCENT = "#241605"  # 橘底上的深色文字，保证对比度

# ---- 强调色：琥珀 / 橘 ----
ACCENT = "#F5A623"
ACCENT_HOVER = "#FFB84D"
ACCENT_PRESSED = "#D98E12"
ACCENT_SOFT = "rgba(245, 166, 35, 0.16)"

# ---- 语义色 ----
DANGER = "#F0615B"
SUCCESS = "#4CCB87"
WARNING = "#F5A623"         # 与强调色统一为琥珀
INFO = "#5B9CFF"

# ---- 圆角 ----
RADIUS_SM = "5px"
RADIUS = "8px"
RADIUS_LG = "12px"

# ---- 字体 ----
FONT_FAMILY = "Microsoft YaHei UI"
FONT_MONO = "Consolas"

# ---- 尺寸 ----
NAV_WIDTH = 56
