"""主工作页：视频预览、选区、播放控制、时间轴、任务执行。"""

import os
import cv2
import queue
import threading
import traceback
import logging
import types
from pathlib import Path
from datetime import datetime

from PyQt6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QSplitter
from PyQt6.QtCore import pyqtSlot, pyqtSignal, Qt, QTimer
from PyQt6 import QtWidgets, QtGui

from ui.component.video_display_component import VideoDisplayComponent
from ui.component.timeline_widget import TimelineWidget
from ui.widgets.button import (PrimaryButton, SecondaryButton, GhostButton,
                               IconButton, set_active)
from ui.widgets.card import Card
from ui.widgets.field import ComboBoxField, SwitchField
from ui.widgets.log_view import LogView
from ui.widgets import icons
from ui.playback_controller import PlaybackController
from app.pipeline import ProcessingPipeline
from infra.config import config, tr
from infra.process_manager import ProcessManager
from infra.utils import get_readable_path
from core.video_io.video_reader import create_video_capture
from core.inpaint.region import get_tight_inpaint_area
from ui.preview_decoder import PreviewDecoder

logger = logging.getLogger(__name__)


class HomePage(QWidget):
    hardware_changed = pyqtSignal()
    progress_signal = pyqtSignal(int, bool)
    append_log_signal = pyqtSignal(list)
    update_preview_with_comp_signal = pyqtSignal(list)
    task_error_signal = pyqtSignal(object)
    toggle_buttons_signal = pyqtSignal(bool)

    MODE_SINGLE = 0
    MODE_COMPARE = 1

    DISPLAY_FIT = 0
    DISPLAY_FILL = 1

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.video_path = None
        self.output_path = None
        self.video_cap = None
        self.output_cap = None
        self.fps = None
        self.frame_count = None
        self.frame_width = None
        self.frame_height = None
        self.current_mode = self.MODE_COMPARE
        self.display_mode = self.DISPLAY_FIT
        self._has_processed = False
        self._processing_finished_called = False

        self._stop_event = threading.Event()
        self._worker_thread = None
        # 任务代号：每次开始/停止自增，用于丢弃旧任务迟到的回调
        self._task_generation = 0
        self.running_process = None
        self._video_cap_lock = threading.Lock()

        # 存储最后一帧原片和处理结果（供 Compare 模式使用）
        self._last_ori_frame = None
        self._last_comp_frame = None
        self._preview_decoder = None
        self._preview_queue = queue.Queue(maxsize=2)
        self._proc_pid = None

        self.__init_widgets()
        self._playback = PlaybackController(self)
        self._pipeline = ProcessingPipeline()
        self.progress_signal.connect(self.update_progress)
        self.append_log_signal.connect(self.append_log)
        self.update_preview_with_comp_signal.connect(self.update_preview_with_comp)
        self.task_error_signal.connect(self.on_task_error)
        self.toggle_buttons_signal.connect(self._toggle_buttons)

    # ==================== 界面搭建 ====================

    def __init_widgets(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(0)

        # ============ 中间面板 ============
        center_panel = QWidget()
        center_layout = QVBoxLayout(center_panel)
        center_layout.setContentsMargins(0, 0, 8, 0)
        center_layout.setSpacing(6)

        self.video_display_component = VideoDisplayComponent(self)
        self.video_display_component.tracks_changed.connect(self._on_tracks_changed)
        center_layout.addWidget(self.video_display_component, 1)

        self.video_display = self.video_display_component.video_display
        self.video_slider = self.video_display_component.video_slider
        self.video_slider.valueChanged.connect(self.slider_changed)
        self.video_slider.valueChanged.connect(lambda v: self.timeline.set_current_frame(v))

        # 空格键切换播放/暂停
        self._shortcut_space = QtGui.QShortcut(QtGui.QKeySequence(Qt.Key.Key_Space), self)
        self._shortcut_space.activated.connect(self._toggle_playback)

        # 控制栏：时间码 | 播放控制居中 | 显示模式
        control_bar = QWidget()
        control_layout = QHBoxLayout(control_bar)
        control_layout.setContentsMargins(0, 0, 0, 0)
        control_layout.setSpacing(4)

        self.time_label = QtWidgets.QLabel('00:00:00 / 00:00:00', objectName='Timecode')
        self.time_label.setFixedWidth(180)
        self.time_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        control_layout.addWidget(self.time_label)

        control_layout.addStretch()

        self.jump_start_btn = IconButton('skip_start', tr['Main']['JumpToStart'], self)
        self.jump_start_btn.setEnabled(False)
        control_layout.addWidget(self.jump_start_btn)

        self.prev_btn = IconButton('prev', tr['Main']['PreviousFrame'], self)
        self.prev_btn.setEnabled(False)
        control_layout.addWidget(self.prev_btn)

        self.play_btn = IconButton('play', tr['Main']['Play'], self)
        self.play_btn.setEnabled(False)
        control_layout.addWidget(self.play_btn)

        self.speed_btn = IconButton('speed', tr['Main']['PlaySpeed'], self)
        self.speed_btn.setEnabled(False)
        control_layout.addWidget(self.speed_btn)

        self.next_btn = IconButton('next', tr['Main']['NextFrame'], self)
        self.next_btn.setEnabled(False)
        control_layout.addWidget(self.next_btn)

        self.jump_end_btn = IconButton('skip_end', tr['Main']['JumpToEnd'], self)
        self.jump_end_btn.setEnabled(False)
        control_layout.addWidget(self.jump_end_btn)

        control_layout.addStretch()

        # 显示模式
        self.fit_btn = GhostButton(tr['Main']['FitView'], parent=self)
        self.fill_btn = GhostButton(tr['Main']['FillView'], parent=self)
        self.fit_btn.setEnabled(False)
        self.fill_btn.setEnabled(False)
        control_layout.addWidget(self.fit_btn)
        control_layout.addWidget(self.fill_btn)

        self.fit_btn.clicked.connect(lambda: self.set_display_mode(self.DISPLAY_FIT))
        self.fill_btn.clicked.connect(lambda: self.set_display_mode(self.DISPLAY_FILL))

        # 播放控制图标缓存（供 PlaybackController 使用）
        self._play_icon = icons.icon('play', 18)
        self._pause_icon = icons.icon('pause', 18)
        self._speed1x_icon = icons.speed_icon(1, 18)
        self._speed2x_icon = icons.speed_icon(2, 18)
        self._speed4x_icon = icons.speed_icon(4, 18)
        self.speed_btn.setIcon(self._speed1x_icon)

        self.jump_start_btn.clicked.connect(lambda: self._playback.jump_to_start())
        self.prev_btn.clicked.connect(lambda: self._playback.step_backward())
        self.play_btn.clicked.connect(lambda: self._playback.toggle())
        self.speed_btn.clicked.connect(lambda: self._playback.cycle_speed())
        self.next_btn.clicked.connect(lambda: self._playback.step_forward())
        self.jump_end_btn.clicked.connect(lambda: self._playback.jump_to_end())

        # 键盘快捷键
        QtGui.QShortcut(QtGui.QKeySequence(Qt.Key.Key_Home), self).activated.connect(
            lambda: self._playback.jump_to_start())
        QtGui.QShortcut(QtGui.QKeySequence(Qt.Key.Key_End), self).activated.connect(
            lambda: self._playback.jump_to_end())
        QtGui.QShortcut(QtGui.QKeySequence(Qt.Key.Key_I), self).activated.connect(
            self._jump_to_in_point)
        QtGui.QShortcut(QtGui.QKeySequence(Qt.Key.Key_O), self).activated.connect(
            self._jump_to_out_point)

        center_layout.addWidget(control_bar)

        # 时间轴（固定高度）
        self.timeline = TimelineWidget(self)
        self.timeline.frame_selected.connect(self._on_timeline_frame_selected)
        self.timeline.preview_seek.connect(self._on_preview_seek)
        self.timeline.track_range_changed.connect(self._on_track_range_changed)
        # 未加载视频时也保留标尺 + 一条空白轨道的高度，避免时间轴被压成一条线
        self.timeline.setMinimumHeight(80)
        center_layout.addWidget(self.timeline, 0)

        self._preview_poll_timer = QTimer(self)
        self._preview_poll_timer.setInterval(30)
        self._preview_poll_timer.timeout.connect(self._poll_preview_frame)

        # ============ 右侧边栏 ============
        right_panel = QWidget()
        right_panel.setMinimumWidth(300)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(8, 0, 0, 0)
        right_layout.setSpacing(10)

        # 处理设置
        settings_card = Card('处理设置')
        mode_labels = []
        if tr.has_section('InpaintMode'):
            mode_labels = list(tr['InpaintMode'].values())
        options = list(config.inpaintMode.validator.options)
        labels = [mode_labels[i] if i < len(mode_labels) else str(opt.value)
                  for i, opt in enumerate(options)]
        self.inpaint_field = ComboBoxField(
            tr['SubtitleExtractorGUI']['InpaintMode'], config.inpaintMode, labels)
        self.inpaint_field.setToolTip(tr['SubtitleExtractorGUI']['InpaintModeDesc'])

        self.hw_field = SwitchField(
            tr['Setting']['HardwareAcceleration'], config.hardwareAcceleration,
            tr['Setting']['HardwareAccelerationDesc'])
        self.hardware_switch = self.hw_field.switch
        self.hardware_switch.toggled.connect(lambda _: self.hardware_changed.emit())

        settings_card.add(self.inpaint_field)
        settings_card.add(self.hw_field)
        right_layout.addWidget(settings_card)

        # 输出日志
        log_card = Card('输出日志')
        self.output_text = LogView()
        log_card.add(self.output_text, 1)
        right_layout.addWidget(log_card, 1)

        # 操作按钮
        action_card = Card(margins=(12, 10, 12, 10))
        action_layout = QHBoxLayout()
        action_layout.setContentsMargins(0, 0, 0, 0)
        action_layout.setSpacing(8)

        self.file_button = SecondaryButton(tr['SubtitleExtractorGUI']['Open'], 'folder')
        self.file_button.clicked.connect(self.open_file)
        self.file_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        action_layout.addWidget(self.file_button)

        self.run_button = PrimaryButton(tr['SubtitleExtractorGUI']['Run'], 'play')
        self.run_button.clicked.connect(self.run_button_clicked)
        action_layout.addWidget(self.run_button)

        self.stop_button = SecondaryButton(tr['SubtitleExtractorGUI']['Stop'], 'stop')
        self.stop_button.setVisible(False)
        self.stop_button.clicked.connect(self.stop_button_clicked)
        action_layout.addWidget(self.stop_button)

        action_card.add_layout(action_layout)
        right_layout.addWidget(action_card)

        # ============ 双栏 QSplitter（中栏 + 右栏） ============
        self.splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self.splitter.addWidget(center_panel)
        self.splitter.addWidget(right_panel)
        self.splitter.setStretchFactor(0, 3)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([900, 220])
        self.splitter.setCollapsible(0, False)
        self.splitter.setCollapsible(1, False)
        self.splitter.setHandleWidth(6)

        main_layout.addWidget(self.splitter, 1)

    # ==================== 时间标签更新 ====================

    def _update_time_label(self, frame_num, fps, total_frames):
        if fps <= 0 or total_frames <= 0:
            return
        fps_int = max(1, int(round(fps)))

        def tc(fn):
            fn_0 = max(0, min(fn - 1, (total_frames or 1) - 1))
            sec = fn_0 // fps_int
            fr = fn_0 % fps_int
            m, s = divmod(sec, 60)
            h, m = divmod(m, 60)
            if h > 0:
                return f'{h:02d}:{m:02d}:{s:02d}:{fr:02d}'
            return f'{m:02d}:{s:02d}:{fr:02d}'

        self.time_label.setText(f'{tc(frame_num)} / {tc(total_frames)}')

    # ==================== 模式切换 ====================

    def _set_mode(self, mode):
        self.current_mode = mode
        self.slider_changed(self.video_slider.value())

    def _set_disp_buttons_enabled(self, enabled):
        self.fit_btn.setEnabled(enabled)
        self.fill_btn.setEnabled(enabled)
        if enabled:
            set_active(self.fit_btn, self.display_mode == self.DISPLAY_FIT)
            set_active(self.fill_btn, self.display_mode == self.DISPLAY_FILL)
        else:
            set_active(self.fit_btn, False)
            set_active(self.fill_btn, False)

    def reset_processed_state(self):
        self._has_processed = False
        self._last_ori_frame = None
        self._last_comp_frame = None
        self.output_path = None
        self._playback._playback_speed = 1
        self.speed_btn.setIcon(self._speed1x_icon)
        self.speed_btn.setToolTip(tr['Main']['PlaySpeed'])
        if self.output_cap:
            self.output_cap.release()
            self.output_cap = None
        self._stop_event.clear()
        if self._playback.is_playing:
            self._playback.stop()
        self.video_display_component.clear_selections()
        self.output_text.clear()

    # ==================== 播放控制 ====================

    def _jump_to_in_point(self):
        tracks = self.video_display_component.get_tracks()
        min_start = None
        for t in tracks:
            if t.get("enabled", True):
                s = t.get("start", 1)
                if min_start is None or s < min_start:
                    min_start = s
        if min_start is not None:
            self.video_slider.setValue(min_start)

    def _jump_to_out_point(self):
        tracks = self.video_display_component.get_tracks()
        max_end = None
        for t in tracks:
            if t.get("enabled", True):
                e = t.get("end", self.video_slider.maximum())
                if max_end is None or e > max_end:
                    max_end = e
        if max_end is not None:
            self.video_slider.setValue(max_end)

    def _toggle_playback(self):
        self._playback.toggle()

    def _playback_tick(self):
        """播放定时器回调：顺序读取帧避免闪跳，播放结束后重置以允许重新播放。"""
        # 处理中 worker 可能正在释放/重建 VideoCapture，跳过本次 tick 避免跨线程竞态
        if self._worker_thread and self._worker_thread.is_alive():
            return
        if self.current_mode == self.MODE_COMPARE:
            # Compare 模式：顺序读取两个 cap（不再每帧 seek，避免闪跳）
            with self._video_cap_lock:
                ori_cap = self.video_cap
                comp_cap = self.output_cap if (
                    self._has_processed and self.output_cap and self.output_cap.isOpened()) else None
                if ori_cap is None or not ori_cap.isOpened():
                    self._playback.stop()
                    return
                ret_ori, ori_frame = ori_cap.read()
                ret_comp, comp_frame = (False, None)
                if comp_cap and comp_cap.isOpened():
                    ret_comp, comp_frame = comp_cap.read()
                if not ret_ori:
                    # 播放结束：重置到开头以允许重新播放（setValue 需在锁外，避免重入死锁）
                    self._playback.stop()
                    ori_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    if comp_cap:
                        comp_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    fi = None
                else:
                    fi = int(ori_cap.get(cv2.CAP_PROP_POS_FRAMES))
            if fi is None:
                self.video_slider.setValue(1)
                return
            if ori_frame is not None:
                self._last_ori_frame = ori_frame
            if comp_frame is not None:
                self._last_comp_frame = comp_frame
            display_ori = self._last_ori_frame
            display_comp = self._last_comp_frame if self._last_comp_frame is not None else display_ori
            if display_ori is not None:
                combined = self._concat_frames(display_ori, display_comp)
                self.video_display_component.update_video_display(combined, draw_selection=False)
            if self.fps and self.frame_count:
                self._update_time_label(fi, self.fps, self.frame_count)
            self.video_slider.blockSignals(True)
            self.video_slider.setValue(fi)
            self.video_slider.blockSignals(False)
            self.timeline.set_current_frame(fi)
            return
        # Single 模式：顺序读取（不 seek），流畅播放
        with self._video_cap_lock:
            cap = self.output_cap if (
                self._has_processed and self.output_cap and self.output_cap.isOpened()) else self.video_cap
            if cap is None or not cap.isOpened():
                self._playback.stop()
                return
            ret, frame = cap.read()
            if not ret:
                # 播放结束：重置到开头以允许重新播放（setValue 需在锁外，避免重入死锁）
                self._playback.stop()
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                fi = None
            else:
                fi = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if fi is None:
            self.video_slider.setValue(1)
            return
        if self.fps and self.frame_count:
            self._update_time_label(fi, self.fps, self.frame_count)
        self.video_slider.blockSignals(True)
        self.video_slider.setValue(fi)
        self.video_slider.blockSignals(False)
        self.timeline.set_current_frame(fi)
        self.video_display_component.update_video_display(frame)

    def _set_playback_buttons_enabled(self, enabled):
        self._playback.set_buttons_enabled(enabled)

    # ==================== 显示控制 ====================

    def set_display_mode(self, mode):
        self.display_mode = mode
        self.video_display_component.set_display_mode(mode)
        set_active(self.fit_btn, mode == self.DISPLAY_FIT)
        set_active(self.fill_btn, mode == self.DISPLAY_FILL)
        self.slider_changed(self.video_slider.value())

    # ==================== 视频加载与预览 ====================

    def slider_changed(self, value):
        if self._worker_thread and self._worker_thread.is_alive():
            return
        if self.fps and self.frame_count:
            self._update_time_label(value, self.fps, self.frame_count)

        if self.current_mode == self.MODE_COMPARE:
            # Compare: 左右拼接（原片 + 结果）
            ori_frame = None
            with self._video_cap_lock:
                if self.video_cap and self.video_cap.isOpened():
                    self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, value - 1))
                    ret, ori_frame = self.video_cap.read()
                    if not ret:
                        ori_frame = None
            comp_frame = None
            with self._video_cap_lock:
                if self.output_cap and self.output_cap.isOpened():
                    self.output_cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, value - 1))
                    ret, comp_frame = self.output_cap.read()
                    if not ret:
                        comp_frame = None
            # 处理中缓存帧兜底
            if ori_frame is None and self._last_ori_frame is not None:
                ori_frame = self._last_ori_frame
            if comp_frame is None and self._last_comp_frame is not None:
                comp_frame = self._last_comp_frame
            if ori_frame is not None and comp_frame is not None:
                combined = self._concat_frames(ori_frame, comp_frame)
                self.video_display_component.update_video_display(combined, draw_selection=False)
            elif ori_frame is not None:
                # 无处理结果时只显示原片
                self.video_display_component.update_video_display(ori_frame, draw_selection=False)
            return

        cap = None
        frame = None
        with self._video_cap_lock:
            cap = self.output_cap if (
                self._has_processed and self.output_cap and self.output_cap.isOpened()) else self.video_cap
            if cap is not None and cap.isOpened():
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, value - 1))
                ret, frame = cap.read()
                if not ret:
                    frame = None
                    logger.debug('slider_changed: frame=%d seek failed', value)
            else:
                logger.debug('slider_changed: cap not available for frame=%d', value)
        if frame is not None:
            self.update_preview(frame)
        else:
            self.video_display_component.clear_display()

    def update_preview(self, frame):
        """更新预览画面（set_video_parameters 已在 load_video 时设置一次）。"""
        self.video_display_component.update_video_display(frame)

    @staticmethod
    def _concat_frames(frame_a, frame_b):
        """将两帧等高拼接为对比图（高度不同时缩放到一致）。"""
        if frame_a is None:
            return frame_b
        if frame_b is None:
            return frame_a
        if frame_a.shape[0] != frame_b.shape[0]:
            h = min(frame_a.shape[0], frame_b.shape[0])
            frame_a = cv2.resize(frame_a, (int(frame_a.shape[1] * h / frame_a.shape[0]), h))
            frame_b = cv2.resize(frame_b, (int(frame_b.shape[1] * h / frame_b.shape[0]), h))
        return cv2.hconcat([frame_a, frame_b])

    def load_video(self, video_path):
        if self._worker_thread and self._worker_thread.is_alive():
            logger.warning('video_load_rejected_while_processing: %s', video_path)
            self.append_output('正在处理中，无法打开新视频，请先停止当前任务')
            return False
        logger.info('video_load_start: %s', video_path)
        self.video_path = video_path
        with self._video_cap_lock:
            if self.video_cap:
                self.video_cap.release()
                self.video_cap = None
            self.video_cap = create_video_capture(get_readable_path(self.video_path))
            if not self.video_cap.isOpened():
                self.video_cap = None
                logger.warning('video_open_failed: %s', video_path)
                return False
            ret, frame = self.video_cap.read()
            if not ret:
                self.video_cap.release()
                self.video_cap = None
                logger.warning('video_first_frame_failed: %s', video_path)
                return False
            self.frame_count = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.frame_height = int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.frame_width = int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.fps = self.video_cap.get(cv2.CAP_PROP_FPS)
            logger.info('video_loaded: size=%dx%d, fps=%.2f, frames=%d',
                        self.frame_width, self.frame_height, self.fps, self.frame_count)

        # 设置视频参数（帧宽高、fps、显示模式），供 VideoDisplayComponent 计算锁定比例等使用
        self.video_display_component.set_video_parameters(
            self.frame_width, self.frame_height,
            self.fps if self.fps is not None else 30,
            self.display_mode,
        )

        self._stop_preview_decoder()
        self._preview_decoder = PreviewDecoder(
            get_readable_path(self.video_path), self._preview_queue)
        self._preview_poll_timer.start()

        self.reset_processed_state()
        self.current_mode = self.MODE_SINGLE
        self._set_disp_buttons_enabled(False)
        self._set_playback_buttons_enabled(True)

        self.update_preview(frame)
        self.video_slider.setMaximum(self.frame_count)
        self.video_slider.setValue(1)
        self.video_display_component.set_dragger_enabled(True)
        self._update_time_label(1, self.fps, self.frame_count)
        self.timeline.set_data(self.video_display_component.get_tracks(), self.frame_count, self.fps)
        self.timeline.set_current_frame(1)
        return True

    # ==================== 轨道 / 时间轴回调 ====================

    def _on_tracks_changed(self, _ignored=None):
        self.timeline.set_data(self.video_display_component.get_tracks(), self.frame_count or 1, self.fps or 30)
        self.timeline.set_current_frame(self.video_slider.value())

    def _on_track_range_changed(self, _track_id, _start, _end):
        """时间轴拖动轨道边界：轨道字典为共享引用，这里刷新画面选框。"""
        self.video_display_component.update_preview_with_rect()
        self.timeline.set_current_frame(self.video_slider.value())

    def _on_timeline_frame_selected(self, frame):
        frame = max(1, min(frame, self.frame_count or 1))
        if self._worker_thread and self._worker_thread.is_alive():
            return
        if frame != self.video_slider.value():
            self.video_slider.setValue(frame)

    def _stop_preview_decoder(self):
        self._preview_poll_timer.stop()
        if self._preview_decoder:
            self._preview_decoder.stop()
            self._preview_decoder = None
        while not self._preview_queue.empty():
            try:
                self._preview_queue.get_nowait()
            except queue.Empty:
                break

    def _on_preview_seek(self, frame):
        if not self._preview_decoder:
            return
        fi = max(0, min(frame - 1, (self.frame_count or 1) - 1))
        self._preview_decoder.seek(fi)
        if not self._preview_poll_timer.isActive():
            self._preview_poll_timer.start()
        # 立即刷新选框（不等待解码器线程），滤镜使用当前滑块值+共享的轨道字典
        self.video_display_component.update_preview_with_rect()

    def _poll_preview_frame(self):
        if self._worker_thread and self._worker_thread.is_alive():
            return
        try:
            fi, frame = self._preview_queue.get_nowait()
        except queue.Empty:
            return
        self._last_ori_frame = frame
        self.video_slider.blockSignals(True)
        self.video_slider.setValue(fi + 1)
        self.video_slider.blockSignals(False)
        self.video_display_component.update_video_display(frame, draw_selection=True)
        if self.fps and self.frame_count:
            self._update_time_label(fi + 1, self.fps, self.frame_count)

    # ==================== 运行 / 停止 ====================

    def run_button_clicked(self):
        logger.info('run_button_clicked')
        if not self.video_path:
            self.append_output(tr['SubtitleExtractorGUI']['OpenVideoFirst'])
            return

        if not os.path.exists(self.video_path):
            logger.warning('file_not_found: %s', self.video_path)
            self.append_output(f"File not found: {self.video_path}")
            return

        logger.info('processing_start: path=%s, tracks=%s, ab_sections=%s',
                    self.video_path,
                    [t.get("id") for t in self.video_display_component.get_tracks()],
                    self.video_display_component.get_ab_sections())

        self._preview_poll_timer.stop()
        self._playback.stop()
        self.video_display_component.set_dragger_enabled(False)
        self.timeline.setEnabled(False)
        self._set_playback_buttons_enabled(False)
        # 处理期间禁止打开新视频，避免旧任务用新 video_path 配旧 output_path
        self.file_button.setEnabled(False)
        self.video_display_component.show_status(tr['Main']['WaitRendering'])

        # 新任务代号：旧 worker 迟到的回调会被丢弃
        self._task_generation += 1
        generation = self._task_generation

        # STTN 模式固定使用框选区域遮罩，双窗口直接对比
        self.current_mode = self.MODE_COMPARE

        # 坐标转换必须视图模式确定之后执行，
        # 避免 compare 模式拼接帧后覆盖 scaled_width/scaled_height 导致坐标映射错误
        tracks = self.video_display_component.get_tracks()
        track_data_px = []
        for t in tracks:
            rects = self.video_display_component.preview_coordinates_to_video_coordinates(
                [(t["ymin"], t["ymax"], t["xmin"], t["xmax"])])
            if rects:
                ymin_px, ymax_px, xmin_px, xmax_px = rects[0]
                track_data_px.append({
                    "id": t["id"],
                    "ymin": ymin_px, "ymax": ymax_px,
                    "xmin": xmin_px, "xmax": xmax_px,
                    "start": t["start"], "end": t["end"],
                    "enabled": t.get("enabled", True),
                })
        options = {
            'track_data': track_data_px,
            'sub_areas': self.video_display_component.preview_coordinates_to_video_coordinates(
                [(t["ymin"], t["ymax"], t["xmin"], t["xmax"]) for t in tracks if t.get("enabled", True)]
            ),
            'ab_sections': self.video_display_component.get_ab_sections(),
        }

        self.video_display_component.show_status(tr['Main']['WaitRendering'])

        try:
            self._stop_event.clear()
            self._processing_finished_called = False
            self.toggle_buttons_signal.emit(False)

            # 计算输出路径
            save_dir = config.saveDirectory.value if config.saveDirectory.value else os.path.dirname(self.video_path)
            stem = Path(self.video_path).stem
            ext = Path(self.video_path).suffix
            self.output_path = os.path.abspath(os.path.join(save_dir, f'{stem}_no_sub{ext}'))

            def task():
                try:
                    with self._video_cap_lock:
                        if self.video_cap:
                            self.video_cap.release()
                            self.video_cap = None

                    process = self.run_subtitle_remover_process(self.video_path, self.output_path, options)

                    if self._stop_event.is_set():
                        return

                    # 处理完成
                    if process.exitcode == 0:
                        self.progress_signal.emit(100, True)
                    else:
                        self.append_log_signal.emit([
                            tr['SubtitleExtractorGUI']['ErrorDuringProcessing'].format(
                                f"exit code: {process.exitcode}")])
                except Exception as e:
                    logger.error('processing_task_error: %s', traceback.format_exc())
                    self.append_log_signal.emit([f"Error: {e}"])
                finally:
                    # 只有当前任务才能恢复 UI，避免旧任务干扰新任务
                    if generation == self._task_generation:
                        self.toggle_buttons_signal.emit(True)

            self._worker_thread = threading.Thread(target=task, daemon=True)
            self._worker_thread.start()
        except Exception as e:
            logger.error('thread_start_error: %s', traceback.format_exc())
            self.append_log_signal.emit([f"Error: {e}"])
            self.toggle_buttons_signal.emit(True)

    def stop_button_clicked(self):
        try:
            self._stop_event.set()
            # 使旧任务的迟到回调失效
            self._task_generation += 1
            self.append_output(tr['Main']['StopProcessing'])
            logger.info('user_stop_processing')
            # 通过 ProcessManager 统一终止子进程
            if self._proc_pid is not None:
                ProcessManager.instance().terminate_by_pid(self._proc_pid)
            ProcessManager.instance().terminate_all()
            if self._pipeline:
                self._pipeline.stop()
        finally:
            # 必须 join 旧 worker，否则 is_alive() 守卫失效且旧信号会污染新任务
            worker = self._worker_thread
            if worker is not None and worker.is_alive():
                worker.join(timeout=5)
                if worker.is_alive():
                    logger.warning('worker_thread_still_alive_after_stop')
            self._proc_pid = None
            self._worker_thread = None
            self.file_button.setEnabled(True)
            self._reopen_video_cap()
            self.slider_changed(self.video_slider.value())
            self.video_display_component.hide_status()
            self.video_display_component.set_dragger_enabled(True)
            self.timeline.setEnabled(True)
            self._set_playback_buttons_enabled(True)
            self.run_button.setVisible(True)
            self.stop_button.setVisible(False)

    @pyqtSlot(bool)
    def _toggle_buttons(self, show_run):
        self.run_button.setVisible(show_run)
        self.stop_button.setVisible(not show_run)
        if show_run:
            self.file_button.setEnabled(True)
            self.video_display_component.hide_status()
            self.video_display_component.set_dragger_enabled(True)
            self.timeline.setEnabled(True)
            self._set_playback_buttons_enabled(True)
            # 处理失败/停止时 video_cap 已被释放，这里恢复源视频句柄
            self._reopen_video_cap()

    def _reopen_video_cap(self):
        if not self.video_path:
            return
        if self.video_cap is not None and self.video_cap.isOpened():
            return
        try:
            with self._video_cap_lock:
                self.video_cap = create_video_capture(self.video_path)
        except Exception:
            logger.exception('video_cap_reopen_failed')

    # ==================== 子进程处理 ====================

    def run_subtitle_remover_process(self, video_path, output_path, options):
        callbacks = types.SimpleNamespace(
            on_progress=self.progress_signal.emit,
            on_log=self.append_log_signal.emit,
            on_preview=self.update_preview_with_comp_signal.emit,
            on_error=self.task_error_signal.emit,
        )
        return self._pipeline.start(
            video_path, output_path, options, callbacks,
            on_started=self._on_process_started,
            stop_event=self._stop_event,
        )

    def _on_process_started(self, process):
        # 在 process.start() 之后、join() 之前记录 PID，否则处理期间 _proc_pid 恒为 None，无法按 PID 停止
        self._proc_pid = getattr(process, 'pid', None)
        logger.info('worker_process_started: pid=%s', self._proc_pid)

    # ==================== 进度 / 日志 / 错误 ====================

    @pyqtSlot(int, bool)
    def update_progress(self, progress_total, isFinished):
        try:
            if self.frame_count:
                pos = int(progress_total / 100 * self.frame_count)
                pos = max(1, min(pos, self.frame_count))
                if pos != self.video_slider.value():
                    self.video_slider.blockSignals(True)
                    self.video_slider.setValue(pos)
                    self.video_slider.blockSignals(False)
                self.timeline.set_current_frame(pos)
                self._update_time_label(pos, self.fps, self.frame_count)

            if isFinished:
                self.processing_finished()
        except Exception as e:
            logger.warning('progress_update_error: %s', e)

    @pyqtSlot()
    def processing_finished(self):
        if self._processing_finished_called:
            return
        # worker 线程未退出时 slider_changed 会早返回，延迟到线程结束后再处理
        if self._worker_thread and self._worker_thread.is_alive():
            QTimer.singleShot(200, self.processing_finished)
            return
        self._processing_finished_called = True
        logger.info('processing_finished_callback')
        logger.info('  video_path=%s', self.video_path)
        logger.info('  output_path=%s', self.output_path)
        self.run_button.setVisible(True)
        self.stop_button.setVisible(False)
        self.file_button.setEnabled(True)
        self.video_display_component.hide_status()
        self._set_playback_buttons_enabled(True)

        if self.output_path and os.path.exists(self.output_path):
            logger.info('output_video_open_start')
            self._has_processed = True
            with self._video_cap_lock:
                if self.output_cap:
                    self.output_cap.release()
                self.output_cap = create_video_capture(get_readable_path(self.output_path))
                if self.output_cap.isOpened():
                    logger.info('output_video_opened')
                else:
                    logger.warning('output_video_open_failed')
                    self.output_cap = None
                    self._has_processed = False

        logger.info('source_video_reopen')
        with self._video_cap_lock:
            if self.video_cap:
                self.video_cap.release()
            self.video_cap = create_video_capture(get_readable_path(self.video_path))
            if self.video_cap.isOpened():
                logger.info('source_video_opened')
            else:
                logger.warning('source_video_reopen_failed')

        # 处理完成，切换到单窗口预览输出结果
        self.current_mode = self.MODE_SINGLE

        # 重置时间指针到开头
        self.video_slider.setValue(1)
        if self.output_cap and self.output_cap.isOpened():
            out_frames = int(self.output_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if out_frames > 0:
                self.video_slider.setMaximum(out_frames)
        self.timeline.set_current_frame(1)
        self._update_time_label(1, self.fps, self.frame_count)

        # 刷新当前视图
        self.slider_changed(self.video_slider.value())
        # 处理完成后不再需要预览解码器（源视频），使用输出视频直接播放
        self._stop_preview_decoder()

    @pyqtSlot(list)
    def append_log(self, log):
        self.append_output(*log)

    def append_output(self, *args):
        text = ' '.join(str(arg) for arg in args).rstrip()
        if not text:
            return
        timestamp = datetime.now().strftime('%H:%M:%S')
        escaped = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        if '错误' in text or 'Error' in text or '失败' in text or 'Failed' in text:
            color = '#E5534B'
        elif '成功' in text or '完成' in text or 'Success' in text or 'Finished' in text:
            color = '#35C27B'
        elif '警告' in text or 'Warning' in text:
            color = '#E8A33D'
        else:
            color = '#9AA6B8'
        html = f'<span style="color:#6B7688;">[{timestamp}]</span> <span style="color:{color};">{escaped}</span><br>'
        self.output_text.append_rich(html)
        logger.info('output_log: %s', ' '.join(str(a) for a in args))

        # 开始移除字幕 → 自动切换到双窗口对比显示
        if '开始移除字幕' in text:
            self.current_mode = self.MODE_COMPARE

    @pyqtSlot(list)
    def update_preview_with_comp(self, args):
        """处理中实时预览"""
        if not args or len(args) < 2:
            return
        frame_ori, frame_comp = args
        if frame_ori is None or frame_comp is None:
            return
        # 只有处理仍在进行时才更新预览，避免停止后到达的排队信号再次禁用拖拽
        if not (self._worker_thread and self._worker_thread.is_alive()):
            return
        self._last_ori_frame = frame_ori
        self._last_comp_frame = frame_comp

        self.video_display_component.set_dragger_enabled(False)
        self._set_disp_buttons_enabled(True)

        # 处理中：在原始帧上叠加用户选择框 + 实际推理校准框
        if self._worker_thread and self._worker_thread.is_alive() and self.current_mode == self.MODE_COMPARE:
            tracks = self.video_display_component.get_tracks()
            if tracks:
                # 缩放到预览尺寸再画（帧从子进程来是全分辨率的，直接画在缩略图上线太细）
                PREVIEW_W = 800
                h_orig, w_orig = frame_ori.shape[:2]
                scale = PREVIEW_W / w_orig if w_orig > PREVIEW_W else 1.0
                if scale < 1.0:
                    preview_h = int(h_orig * scale)
                    ori_display = cv2.resize(frame_ori, (PREVIEW_W, preview_h), interpolation=cv2.INTER_AREA)
                    comp_small = cv2.resize(frame_comp, (PREVIEW_W, preview_h), interpolation=cv2.INTER_AREA)
                else:
                    ori_display = frame_ori.copy()
                    comp_small = frame_comp
                h, w = ori_display.shape[:2]

                # --- 收集像素坐标，分 track 累计 ---
                track_pixel_groups = {}  # track_id -> list of (ymin,ymax,xmin,xmax)
                enabled_tracks = []
                for t in tracks:
                    if not t.get("enabled", True):
                        continue
                    px = int(t["xmin"] * w)
                    py = int(t["ymin"] * h)
                    px2 = int(t["xmax"] * w)
                    py2 = int(t["ymax"] * h)
                    tid = t.get("id", 0)
                    track_pixel_groups.setdefault(tid, []).append((py, py2, px, px2))
                    enabled_tracks.append(t)

                logger.debug("draw_boxes: %d tracks, frame=%dx%d (scaled from %dx%d, scale=%.2f)",
                             len(track_pixel_groups), w, h, w_orig, h_orig, scale)

                # --- 用轨道颜色画用户选区（角标）+ 推理校准框（完整矩形），同色 ---
                # 固定画线参数（预览空间 ~800px 宽，不随视频缩放），确保在不同视频上大小一致
                FIXED_CORNER = 48      # 角标长度（固定像素，不按视频缩放）
                FIXED_PW = 2           # 线宽，与视频框选一致
                FONT_SCALE = 1.1       # 标签字号
                FONT_PW = 1            # 标签字粗细（FONT_HERSHEY_PLAIN 笔画细，设 2 会显粗）
                for tid, coords in track_pixel_groups.items():
                    t = next((t for t in enabled_tracks if str(t.get("id", "")) == str(tid)), None)
                    color_hex = t.get("color", "#FF6B35") if t else "#FF6B35"
                    bgr = (int(color_hex[5:7], 16), int(color_hex[3:5], 16), int(color_hex[1:3], 16))
                    # 1) 用户原始选区 → 四角括号（不标文字）
                    for (py, py2, px, px2) in coords:
                        cv2.line(ori_display, (px, py), (px + FIXED_CORNER, py), bgr, FIXED_PW)
                        cv2.line(ori_display, (px, py), (px, py + FIXED_CORNER), bgr, FIXED_PW)
                        cv2.line(ori_display, (px2, py), (px2 - FIXED_CORNER, py), bgr, FIXED_PW)
                        cv2.line(ori_display, (px2, py), (px2, py + FIXED_CORNER), bgr, FIXED_PW)
                        cv2.line(ori_display, (px, py2), (px + FIXED_CORNER, py2), bgr, FIXED_PW)
                        cv2.line(ori_display, (px, py2), (px, py2 - FIXED_CORNER), bgr, FIXED_PW)
                        cv2.line(ori_display, (px2, py2), (px2 - FIXED_CORNER, py2), bgr, FIXED_PW)
                        cv2.line(ori_display, (px2, py2), (px2, py2 - FIXED_CORNER), bgr, FIXED_PW)
                    # 2) 推理校准框 → 完整矩形（标注轨道的唯一标签）
                    try:
                        scaled_padding = max(4, int(30 * scale))
                        tight_areas = get_tight_inpaint_area(
                            coords, w, h, padding=scaled_padding)
                        for (ty1, ty2, tx1, tx2) in tight_areas:
                            cv2.rectangle(ori_display, (tx1, ty1), (tx2, ty2), bgr, FIXED_PW)
                            # 只在校准框上标一次 "轨道01"，去掉 ## 和 选/校
                            try:
                                label = f"轨道{int(tid):02d}"
                            except (ValueError, TypeError):
                                label = f"轨道{str(tid).zfill(2)}"
                            cv2.putText(ori_display, label,
                                        (tx1 + 4, ty1 - 10), cv2.FONT_HERSHEY_PLAIN,
                                        FONT_SCALE * 1.8, bgr, FONT_PW)
                    except Exception as e:
                        logger.error("tight_crop_preview_error: %s", e)

                combined = self._concat_frames(ori_display, comp_small)
            else:
                combined = self._concat_frames(frame_ori, frame_comp)
        elif self.current_mode == self.MODE_COMPARE:
            combined = self._concat_frames(frame_ori, frame_comp)
        else:
            combined = None

        if combined is not None:
            self.video_display_component.update_video_display(combined, draw_selection=False)
        else:
            self.video_display_component.update_video_display(frame_comp, draw_selection=False)
        self.video_display_component.hide_status()

    @pyqtSlot(object)
    def on_task_error(self, e):
        self.append_output(tr['SubtitleExtractorGUI']['ErrorDuringProcessing'].format(str(e)))

    # ==================== 打开文件 ====================

    def open_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            tr['SubtitleExtractorGUI']['Open'],
            "",
            "All Files (*.*);;Video Files (*.mp4 *.flv *.wmv *.avi *.mkv *.mov);;Image Files (*.jpg *.jpeg *.png *.bmp *.webp *.tiff)"
        )
        if path:
            if self.load_video(path):
                self.append_output(f"{tr['SubtitleExtractorGUI']['OpenVideoSuccess']}: {path}")
            else:
                self.append_output(f"{tr['SubtitleExtractorGUI']['OpenVideoFailed']}: {path}")

    # ==================== 关闭清理 ====================

    def shutdown(self):
        """窗口关闭时释放资源（由主窗口调用）。"""
        try:
            self._stop_event.set()
            ProcessManager.instance().terminate_all()
            if self._worker_thread and self._worker_thread.is_alive():
                self._worker_thread.join(timeout=5)

            for signal, slot in (
                (self.progress_signal, self.update_progress),
                (self.append_log_signal, self.append_log),
                (self.update_preview_with_comp_signal, self.update_preview_with_comp),
                (self.task_error_signal, self.on_task_error),
                (self.toggle_buttons_signal, self._toggle_buttons),
            ):
                try:
                    signal.disconnect(slot)
                except Exception:
                    pass

            self._stop_preview_decoder()

            with self._video_cap_lock:
                if self.video_cap:
                    self.video_cap.release()
                    self.video_cap = None
                if self.output_cap:
                    self.output_cap.release()
                    self.output_cap = None
        except Exception as e:
            logger.warning('shutdown_error: %s', e)
