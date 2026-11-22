from PyQt6 import QtCore

from qfluentwidgets import setTheme, qconfig, Theme
import darkdetect


class SystemThemeListener(QtCore.QThread):
    """ System theme listener """

    def __init__(self, parent=None):
        super().__init__(parent=parent)

    def run(self):
        darkdetect.listener(self._on_theme_changed)

    def _on_theme_changed(self, theme: str):
        theme = Theme.DARK if theme.lower() == "dark" else Theme.LIGHT
        if qconfig.themeMode.value != Theme.AUTO or theme == qconfig.theme:
            return
        # 通过 queued invoke 把主题变更传递到主线程，避免跨线程操作 Qt 对象
        QtCore.QMetaObject.invokeMethod(
            self, "_apply_theme_on_main_thread",
            QtCore.Qt.ConnectionType.QueuedConnection,
            QtCore.Q_ARG(str, theme.value)
        )

    @QtCore.pyqtSlot(str)
    def _apply_theme_on_main_thread(self, theme_name: str):
        """在主线程中应用主题变更"""
        # 不能用 setTheme(具体 DARK/LIGHT)：它内部 qconfig.set(themeMode, theme) 会把
        # themeMode 从 AUTO 覆盖为具体值，导致 _on_theme_changed 的 AUTO 守卫此后恒失败，
        # 主题只能跟随系统切换一次。setTheme(Theme.AUTO) 保持 themeMode=AUTO，
        # 由 qfluentwidgets 自行用 darkdetect 解析并应用当前系统主题。
        setTheme(Theme.AUTO)
