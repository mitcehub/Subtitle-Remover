"""主窗口：原生标题栏（含硬件信息）+ 左侧导航 + 页面堆栈。"""

import os
import shutil
import subprocess
import threading
import logging

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QMainWindow,
                             QStackedWidget, QVBoxLayout, QWidget)

from infra.config import config, tr, BASE_DIR, VERSION
from infra.process_manager import ProcessManager
from ui.pages.home_page import HomePage
from ui.pages.settings_page import SettingsPage
from ui.theme import tokens
from ui.widgets.nav import NavButton
from ui.widgets.toast import Toast

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    _hw_label_ready = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        # 使用系统原生标题栏：拖动、边缘缩放、窗口按钮都由系统提供，稳定可靠。
        # 硬件信息通过窗口标题文字展示（见 _apply_hw_text）。
        self._hw_text = ''
        self.setWindowTitle(self._build_title())
        self.resize(1246, 810)
        self.setMinimumSize(960, 640)
        self.setWindowIcon(QtGui.QIcon(
            os.path.join(BASE_DIR, 'ui', 'icon', 'subtitle-remover.ico')))

        root = QWidget()
        root.setObjectName('Root')
        self.setCentralWidget(root)

        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        body.addWidget(self._build_nav())

        self.stack = QStackedWidget()
        body.addWidget(self.stack, 1)
        outer.addLayout(body, 1)

        self.home_page = HomePage()
        self.settings_page = SettingsPage()
        self.stack.addWidget(self.home_page)
        self.stack.addWidget(self.settings_page)

        self.home_btn.setChecked(True)
        self.home_btn.clicked.connect(lambda: self._switch_page(0))
        self.settings_btn.clicked.connect(lambda: self._switch_page(1))

        self.toast = Toast(self)
        config.appRestartSig.connect(
            lambda: self.toast.show_message('配置已更新，重启后生效'))

        self.home_page.hardware_changed.connect(self._refresh_hardware_label)

        self._setup_hardware_badge()

    # ---- 导航 ----

    def _build_nav(self):
        rail = QFrame()
        rail.setObjectName('NavRail')
        rail.setFixedWidth(tokens.NAV_WIDTH)

        layout = QVBoxLayout(rail)
        layout.setContentsMargins(9, 10, 9, 10)
        layout.setSpacing(6)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)

        self.home_btn = NavButton('home', tr['SubtitleExtractorGUI']['Title'])
        self.settings_btn = NavButton('settings', tr['SubtitleExtractorGUI']['Setting'])
        layout.addWidget(self.home_btn)
        layout.addWidget(self.settings_btn)
        layout.addStretch(1)
        return rail

    def _switch_page(self, index):
        self.stack.setCurrentIndex(index)

    # ---- 硬件信息（显示在原生标题栏） ----

    def _build_title(self):
        """基础标题 + 当前硬件信息。"""
        base = "Subtitle Remover"
        return f"{base}  —  {self._hw_text}" if self._hw_text else base

    def _apply_hw_text(self, text):
        """更新硬件信息并刷新原生标题栏文字。"""
        self._hw_text = text or ''
        self.setWindowTitle(self._build_title())

    def _setup_hardware_badge(self):
        self._hw_query_running = False
        self._hw_label_ready.connect(self._apply_hw_text)
        self._refresh_hardware_label()

        self._hw_refresh_timer = QTimer(self)
        self._hw_refresh_timer.setInterval(2000)
        self._hw_refresh_timer.timeout.connect(self._refresh_hardware_label)
        self._hw_refresh_timer.start()

    def _refresh_hardware_label(self):
        """刷新硬件信息文本（读取当前配置 + 实际硬件）。

        nvidia-smi / torch 查询放到后台线程，避免阻塞 GUI 事件循环。
        """
        from infra.hardware import HardwareAccelerator

        hw = HardwareAccelerator.instance()

        if not config.hardwareAcceleration.value:
            self._apply_hw_text("CPU（GPU 加速已关闭）")
            return
        if hw.has_cuda():
            if self._hw_query_running:
                return
            self._hw_query_running = True
            threading.Thread(target=self._query_hardware_async, daemon=True).start()
            return
        if hw.has_mps():
            text = "MPS (Apple GPU)"
        elif hw.has_accelerator():
            text = hw.accelerator_name
        else:
            text = "CPU（无 GPU 加速器）"
        self._apply_hw_text(text)

    def _query_hardware_async(self):
        """后台线程：查询系统级显存占用并通过信号回传主线程。"""
        try:
            gpu_name, vram_text = self._query_nvidia_smi()
            if gpu_name and vram_text:
                text = f"GPU: {gpu_name}  |  VRAM: {vram_text}"
            else:
                # nvidia-smi 不可用时回退到 torch（只看主进程 context）
                try:
                    import torch
                    gpu_name = torch.cuda.get_device_name(0)
                    free_b, total_b = torch.cuda.mem_get_info()
                    used_gb = (total_b - free_b) / (1024 ** 3)
                    total_gb = total_b / (1024 ** 3)
                    text = f"GPU: {gpu_name}  |  VRAM: {used_gb:.1f}GB / {total_gb:.1f}GB"
                except Exception:
                    text = "GPU (CUDA)"
            self._hw_label_ready.emit(text)
        except Exception:
            # 退出时控件可能已销毁，忽略即可
            pass
        finally:
            self._hw_query_running = False

    def _query_nvidia_smi(self):
        """调用 nvidia-smi 查询系统级 GPU 名字和显存占用。"""
        nvidia_smi = shutil.which('nvidia-smi')
        if not nvidia_smi:
            return None, None
        try:
            result = subprocess.run(
                [nvidia_smi,
                 '--query-gpu=name,memory.used,memory.total',
                 '--format=csv,noheader,nounits'],
                capture_output=True, timeout=3, text=True
            )
            if result.returncode != 0 or not result.stdout.strip():
                return None, None
            parts = result.stdout.strip().split(', ')
            if len(parts) < 3:
                return None, None
            gpu_name = parts[0].strip()
            used_mb = float(parts[1])
            total_mb = float(parts[2])
            vram_text = f"{used_mb/1024:.1f}GB / {total_mb/1024:.1f}GB"
            return gpu_name, vram_text
        except Exception:
            return None, None

    # ---- 窗口位置 ----

    def load_window_position(self):
        try:
            screen = self.screen().availableGeometry()
            # 还原的尺寸不能超过可用桌面（高 DPI 缩放后 1246×810 会超出屏幕）
            extra_h = max(0, self.frameGeometry().height() - self.height())
            max_w = max(self.minimumWidth(), screen.width())
            max_h = max(self.minimumHeight(), screen.height() - extra_h)

            x = config.windowX.value
            y = config.windowY.value
            width = min(int(config.windowW.value or 1246), max_w)
            height = min(int(config.windowH.value or 810), max_h)

            if x is None or y is None:
                self.resize(width, height)
                self.center_window()
                return

            x = int(x)
            y = int(y)
            if (x >= screen.left() and y >= screen.top() and
                    x + width <= screen.right() + 1 and
                    y + height <= screen.bottom() + 1 - extra_h):
                self.setGeometry(x, y, width, height)
            else:
                self.resize(width, height)
                self.center_window()
        except Exception as e:
            logger.warning('window_position_load_failed: %s', e)
            self.center_window()

    def save_window_position(self):
        config.set(config.windowX, self.x())
        config.set(config.windowY, self.y())
        config.set(config.windowW, self.width())
        config.set(config.windowH, self.height())

    def center_window(self):
        screen_rect = QtWidgets.QApplication.primaryScreen().availableGeometry()
        window_rect = self.frameGeometry()
        window_rect.moveCenter(screen_rect.center())
        self.move(window_rect.topLeft())

    # ---- 关闭 ----

    def closeEvent(self, event):
        self.save_window_position()
        try:
            self.home_page.shutdown()
        finally:
            ProcessManager.instance().terminate_all()
        super().closeEvent(event)

    def keyPressEvent(self, event):
        if (event.key() == QtCore.Qt.Key.Key_C and
                event.modifiers() == QtCore.Qt.KeyboardModifier.ControlModifier):
            logger.warning('user_interrupt: Ctrl+C')
            self.close()
        else:
            super().keyPressEvent(event)
