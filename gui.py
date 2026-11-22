# -*- coding: utf-8 -*-
"""
@Author  : Fang Yao（原作者） / 改写：Jason Eric
@Time    : 2023/4/1 6:07 下午（原始时间）
@FileName: gui.py
@desc: 字幕去除器图形化界面（由 PySimpleGUI 改写为 PySide6）
"""

import sys
import os
import logging
import multiprocessing
import threading
from PyQt6.QtCore import Qt
from PyQt6 import QtCore, QtWidgets, QtGui
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout
from qfluentwidgets import (FluentWindow,
                          setTheme, Theme, setThemeColor, InfoBar, ToolButton, FluentIcon)

from qframelesswindow.utils import getSystemAccentColor
from infra.config import config, tr, VERSION
from ui.theme_listener import SystemThemeListener
from infra.process_manager import ProcessManager
from ui.advanced_setting_interface import AdvancedSettingInterface
from ui.home_interface import HomeInterface

# ============ 日志配置 ============
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.cache')
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, 'asr.log')

_root_logger = logging.getLogger()
_root_logger.setLevel(logging.DEBUG)

_fh = logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8')
_fh.setLevel(logging.DEBUG)
_fmt = logging.Formatter('%(asctime)s [%(levelname)-5s] %(name)s: %(message)s', datefmt='%H:%M:%S')
_fh.setFormatter(_fmt)
_root_logger.addHandler(_fh)

_ch = logging.StreamHandler()
_ch.setLevel(logging.WARNING)
_ch.setFormatter(_fmt)
_root_logger.addHandler(_ch)

# 配置 scenedetect 日志（默认被静默清空，重定向到文件）
try:
    import pyscenedetect
    _scenedetect_logger = logging.getLogger('pyscenedetect')
    _scenedetect_logger.setLevel(logging.DEBUG)
    _scenedetect_logger.handlers.clear()
    _scenedetect_logger.addHandler(_fh)
except ImportError:
    pass

logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('torch').setLevel(logging.WARNING)

logger = logging.getLogger('gui')

# 清理上次运行残留的临时文件（仅清理应用专属临时子目录，避免误删系统 TEMP）
APP_TEMP_DIR = os.path.join(LOG_DIR, 'temp')
if os.path.isdir(APP_TEMP_DIR):
    import shutil
    for fname in os.listdir(APP_TEMP_DIR):
        fpath = os.path.join(APP_TEMP_DIR, fname)
        try:
            if os.path.isfile(fpath):
                os.remove(fpath)
            elif os.path.isdir(fpath):
                shutil.rmtree(fpath, ignore_errors=True)
        except Exception:
            pass

logger.info('=' * 60)
logger.info('app_start: video subtitle remover v%s', '1.5')
logger.info('=' * 60)


class SubtitleExtractorGUI(FluentWindow): 
    _hw_label_ready = QtCore.pyqtSignal(str)

    def __init__(self):
        super().__init__()
        # 开启云母效果
        self.setMicaEffectEnabled(True)
        # 设置深色主题并跟随系统主题色
        setTheme(Theme.AUTO)
        setThemeColor(getSystemAccentColor(), save=True)

        # 初始化系统主题监听器并连接信号
        self.themeListener = SystemThemeListener(self)
        self.themeListener.start()

        # 设置窗口图标
        self.setWindowIcon(QtGui.QIcon(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui", "icon", "asr.ico")))
        # 窗口标题（自动同步到 titleLabel）
        self.setWindowTitle("Video Subtitle Remover")
        # 标题栏：紧凑高度 + logo 留边距
        self.titleBar.setFixedHeight(36)
        self.titleBar.hBoxLayout.setContentsMargins(12, 0, 0, 0)
        # 内容区顶部随标题栏高度
        self.widgetLayout.setContentsMargins(0, 36, 0, 0)
        # 创建界面布局
        self._create_layout()
        self._connectSignalToSlot()
        # 标题栏显示硬件加速信息
        self._setup_titlebar_hardware()

    def _setup_titlebar_hardware(self):
        """在标题栏标题右侧添加硬件加速信息标签。"""
        from PyQt6.QtWidgets import QLabel
        from PyQt6.QtCore import QTimer

        self._hw_label = QLabel(self.titleBar)
        self._hw_label.setStyleSheet("color: #999; font-size: 11px; font-weight: normal; background: transparent;")
        self._hw_query_running = False
        self._hw_label_ready.connect(self._hw_label.setText)

        self._refresh_hardware_label()

        # 插入到标题标签之后、弹性空间之前
        idx = self.titleBar.hBoxLayout.indexOf(self.titleBar.titleLabel)
        self.titleBar.hBoxLayout.insertWidget(idx + 1, self._hw_label, 0, Qt.AlignmentFlag.AlignVCenter)

        # 监听开关切换 → 实时刷新标题栏
        try:
            sw = self.homeInterface.setting_interface.hardware_acceleration.switchButton
            sw.checkedChanged.connect(self._refresh_hardware_label)
        except Exception:
            pass

        # 定时刷新 VRAM 占用（仅 CUDA 模式有动态数值可刷）
        # 刷新间隔 2 秒；开销极小（仅 torch.cuda.mem_get_info 一次调用）
        self._hw_refresh_timer = QTimer(self)
        self._hw_refresh_timer.setInterval(2000)
        self._hw_refresh_timer.timeout.connect(self._refresh_hardware_label)
        self._hw_refresh_timer.start()

    def _refresh_hardware_label(self):
        """刷新标题栏硬件信息文本（读取当前配置 + 实际硬件）。

        nvidia-smi / torch 查询放到后台线程，避免每 2 秒阻塞 GUI 事件循环。
        """
        from infra.hardware import HardwareAccelerator
        from infra.config import config

        hw = HardwareAccelerator.instance()

        if not config.hardwareAcceleration.value:
            self._hw_label.setText("  CPU（GPU 加速已关闭）  ")
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
        self._hw_label.setText(f"  {text}  ")

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
            self._hw_label_ready.emit(f"  {text}  ")
        finally:
            self._hw_query_running = False

    def _query_nvidia_smi(self):
        """调用 nvidia-smi 查询系统级 GPU 名字和显存占用。

        Returns:
            (gpu_name, vram_text) 二元组；失败返回 (None, None)。
            vram_text 形如 "12.3GB / 31.8GB"。
        """
        import subprocess, shutil
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

    def _connectSignalToSlot(self):
        config.appRestartSig.connect(self._showRestartTooltip)

    def _showRestartTooltip(self):
        """ show restart tooltip """
        InfoBar.success(
            'Updated successfully',
            'Configuration takes effect after restart',
            duration=5000,
            parent=self
        )

    def _create_layout(self):
        # 移除 FluentWindow 默认的导航栏
        if self.navigationInterface:
            self.hBoxLayout.removeWidget(self.navigationInterface)
            self.navigationInterface.deleteLater()
            self.navigationInterface = None

        # 从旧的 widgetLayout 中移除 stackedWidget
        self.widgetLayout.removeWidget(self.stackedWidget)

        # ============ 窗口级别左侧导航面板（仅图标按钮） ============
        left_panel = QWidget()
        left_panel.setMinimumWidth(48)
        left_panel.setMaximumWidth(48)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 8, 0, 0)
        left_layout.setSpacing(8)
        left_layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)

        self.home_btn = ToolButton(FluentIcon.HOME, self)
        self.settings_btn = ToolButton(FluentIcon.SETTING, self)
        left_layout.addWidget(self.home_btn)
        left_layout.addWidget(self.settings_btn)

        # 内容区：左面板 + stackedWidget
        self.widgetLayout.addWidget(left_panel)
        self.widgetLayout.addWidget(self.stackedWidget, 1)

        # 创建主页面和高级设置页面
        self.homeInterface = HomeInterface(self)
        self.homeInterface.setObjectName("HomeInterface")
        self.advancedSettingInterface = AdvancedSettingInterface(self)
        self.advancedSettingInterface.setObjectName("AdvancedSettingInterface")

        # 直接加入 stackedWidget
        self.stackedWidget.addWidget(self.homeInterface)
        self.stackedWidget.addWidget(self.advancedSettingInterface)
        self.stackedWidget.setCurrentWidget(self.homeInterface)

        # 连接按钮切换页面
        self.home_btn.clicked.connect(lambda: self.stackedWidget.setCurrentWidget(self.homeInterface))
        self.settings_btn.clicked.connect(lambda: self.stackedWidget.setCurrentWidget(self.advancedSettingInterface))

    def closeEvent(self, event):
        """程序关闭时保存窗口位置并清理资源"""
        self.save_window_position()
        # 停止主题监听线程：darkdetect.listener 为阻塞循环，无协作退出接口，
        # 只能用 terminate() 结束；wait 保证进程退出前线程已收尾。
        try:
            if self.themeListener and self.themeListener.isRunning():
                self.themeListener.terminate()
                self.themeListener.wait(2000)
        except Exception:
            pass
        ProcessManager.instance().terminate_all()
        super().closeEvent(event)

    def _onThemeChangedFinished(self):
        super()._onThemeChangedFinished()

    def save_window_position(self):
        """保存窗口位置到配置文件"""
        # 保存窗口位置和大小
        config.set(config.windowX, self.x())
        config.set(config.windowY, self.y())
        config.set(config.windowW, self.width())
        config.set(config.windowH, self.height())

    def load_window_position(self):
        # 尝试读取窗口位置
        try:
            x = config.windowX.value
            y = config.windowY.value
            width = config.windowW.value
            height = config.windowH.value

            if x is None or y is None:
                self.center_window()
                return

            # 确保窗口在屏幕内
            screen_rect = QtWidgets.QApplication.primaryScreen().availableGeometry()
            if (x >= 0 and y >= 0 and 
                x + width <= screen_rect.width() and 
                y + height <= screen_rect.height()):
                self.setGeometry(x, y, width, height)
            else:
                self.center_window()
        except Exception as e:
            logger.warning('window_position_load_failed: %s', e)
            self.center_window()
    
    def center_window(self):
        """将窗口居中显示"""
        screen_rect = QtWidgets.QApplication.primaryScreen().availableGeometry()
        window_rect = self.frameGeometry()
        center_point = screen_rect.center()
        window_rect.moveCenter(center_point)
        self.move(window_rect.topLeft())

    def keyPressEvent(self, event):
        """处理键盘事件"""
        # 检测Ctrl+C组合键
        if event.key() == QtCore.Qt.Key.Key_C and event.modifiers() == QtCore.Qt.KeyboardModifier.ControlModifier:
            logger.warning('user_interrupt: Ctrl+C')
            self.close()
        else:
            super().keyPressEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # 导航栏已移除，标题栏贴到窗口最左侧
        self.titleBar.move(0, 0)
        self.titleBar.resize(self.width(), self.titleBar.height())


if __name__ == '__main__':
    logger.info('qt_app_init')
    multiprocessing.set_start_method("spawn")
    QApplication.setHighDpiScaleFactorRoundingPolicy(
    Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    # 兼容 PyQt6：无 AA_EnableHighDpiScaling（PyQt6 默认启用高 DPI），仅设置不报错
    app = QtWidgets.QApplication(sys.argv)
    app.setAttribute(Qt.ApplicationAttribute.AA_DontCreateNativeWidgetSiblings)
    # 设置应用级图标，确保 Windows 任务栏显示正确
    _icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui", "icon", "asr.ico")
    app.setWindowIcon(QtGui.QIcon(_icon_path))
    window = SubtitleExtractorGUI()
    # 先设置透明, 再显示, 否则会有闪烁的效果
    window.setWindowOpacity(0.0)
    window.show()
    window.load_window_position()

    # Windows 任务栏图标修复：强制设置 AppUserModelID 并刷新任务栏图标
    if sys.platform == 'win32':
        import ctypes
        from ctypes import wintypes
        try:
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID('vsr.subtitleremover.1')
            # 通过原生 API 强制设置窗口图标（某些 PyQt6 / qfluentwidgets 版本会丢失任务栏图标）
            WM_SETICON = 0x0080
            ICON_BIG = 1
            ICON_SMALL = 0
            icon_pixmap = QtGui.QIcon(_icon_path).pixmap(256, 256)
            # 用延时确保 window 已完全创建
            from PyQt6.QtCore import QTimer
            def _force_taskbar_icon():
                try:
                    hicon = icon_pixmap.toHICON()
                    hwnd = int(window.winId())
                    ctypes.windll.user32.SendMessageW(hwnd, WM_SETICON, ICON_BIG, hicon)
                    ctypes.windll.user32.SendMessageW(hwnd, WM_SETICON, ICON_SMALL, hicon)
                except Exception:
                    pass
            QTimer.singleShot(200, _force_taskbar_icon)
        except Exception:
            pass

    # 使用动画效果逐渐显示窗口
    animation = QtCore.QPropertyAnimation(window, b"windowOpacity")
    animation.setDuration(300)  # 300毫秒的动画
    animation.setStartValue(0.0)
    animation.setEndValue(1.0)
    animation.start()
    app.exec()