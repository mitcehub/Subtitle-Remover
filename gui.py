# -*- coding: utf-8 -*-
"""
@desc: 字幕去除器图形化界面入口（PySide6/PyQt6 自绘深色主题）
"""

import sys
import os
import logging
import multiprocessing

from PyQt6.QtCore import Qt
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtWidgets import QApplication

from ui.main_window import MainWindow
from ui.theme import tokens
from ui.theme.stylesheet import build_stylesheet

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

ICON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "ui", "icon", "asr.ico")


def _fix_taskbar_icon(window):
    """Windows 任务栏图标修复。"""
    if sys.platform != 'win32':
        return
    import ctypes
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            'vsr.subtitleremover.1')
        WM_SETICON = 0x0080
        ICON_BIG = 1
        ICON_SMALL = 0
        icon_pixmap = QtGui.QIcon(ICON_PATH).pixmap(256, 256)

        def _force_taskbar_icon():
            try:
                hicon = icon_pixmap.toHICON()
                hwnd = int(window.winId())
                ctypes.windll.user32.SendMessageW(hwnd, WM_SETICON, ICON_BIG, hicon)
                ctypes.windll.user32.SendMessageW(hwnd, WM_SETICON, ICON_SMALL, hicon)
            except Exception:
                pass

        QtCore.QTimer.singleShot(200, _force_taskbar_icon)
    except Exception:
        pass


def main():
    logger.info('qt_app_init')
    multiprocessing.set_start_method("spawn")
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)

    app = QtWidgets.QApplication(sys.argv)
    # Fusion 底座让各平台控件外观一致，避免原生主题与深色 QSS 冲突
    app.setStyle('Fusion')
    app.setFont(QtGui.QFont(tokens.FONT_FAMILY, 9))
    app.setStyleSheet(build_stylesheet())
    app.setAttribute(Qt.ApplicationAttribute.AA_DontCreateNativeWidgetSiblings)
    app.setWindowIcon(QtGui.QIcon(ICON_PATH))

    window = MainWindow()
    window.setWindowOpacity(0.0)
    window.show()
    window.load_window_position()

    _fix_taskbar_icon(window)

    animation = QtCore.QPropertyAnimation(window, b"windowOpacity")
    animation.setDuration(300)
    animation.setStartValue(0.0)
    animation.setEndValue(1.0)
    animation.start()

    app.exec()


if __name__ == '__main__':
    main()
