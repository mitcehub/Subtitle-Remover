"""字幕去除核心编排器。"""

import shutil
import traceback
import os
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Callable, Optional
import cv2
import sys
import tempfile
import time
from tqdm import tqdm

from infra.config import config, tr, MODEL_PATH, InpaintMode
from infra.hardware import HardwareAccelerator
from infra.utils import get_readable_path
from core.inpaint.sttn import STTNAutoInpaint
from core.inpaint.e2fgvi import E2FGVIAutoInpaint
from core.inpaint.propainter import ProPainterAutoInpaint
from core.inpaint.region import create_mask
from core.video_io.ffmpeg import FFmpegCLI, merge_audio_to_video
from core.video_io.video_reader import create_video_capture
from core.video_io.video_writer import FFmpegVideoWriter

logger = logging.getLogger(__name__)


@dataclass
class ProcessingCallbacks:
    on_log: Callable[[str], None] = field(default_factory=lambda: lambda msg: print(msg))
    on_progress: Callable[[int, bool], None] = field(default_factory=lambda: lambda pct, done: None)
    on_preview: Callable[[object, object], None] = field(default_factory=lambda: lambda ori, comp: None)


class SubtitleRemover:
    # 处理中预览帧的最小发送间隔（秒），避免每帧 copy + 跨进程传输拖慢处理
    PREVIEW_MIN_INTERVAL = 0.1

    def __init__(self, video_path, gui_mode=False, callbacks: Optional[ProcessingCallbacks] = None):
        self.sub_areas = []
        self.gui_mode = gui_mode
        self.callbacks = callbacks or ProcessingCallbacks()
        self.hardware_accelerator = HardwareAccelerator.instance()
        self.hardware_accelerator.set_enabled(config.hardwareAcceleration.value)

        self.video_path = video_path
        # create_video_capture 内部已统一做 Windows 短路径转换
        self.video_cap = create_video_capture(video_path)
        self.vd_name = Path(video_path).stem
        self.frame_count = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT) + 0.5)
        if self.frame_count <= 0:
            fallback_count = FFmpegCLI.instance().get_frame_count(video_path)
            if fallback_count:
                self.frame_count = fallback_count
        self.fps = self.video_cap.get(cv2.CAP_PROP_FPS)
        self.frame_width = int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.size = (self.frame_width, self.frame_height)
        self.mask_size = (self.frame_height, self.frame_width)

        fourcc = int(self.video_cap.get(cv2.CAP_PROP_FOURCC))
        fourcc_str = ''.join(chr((fourcc >> 8 * i) & 0xFF) for i in range(4)).replace('\x00', '') if fourcc else 'unknown'
        logger.info('video_opened: path=%s, %dx%d, fps=%.2f, frames=%d, codec=%s',
                 video_path, self.frame_width, self.frame_height, self.fps, self.frame_count, fourcc_str)
        # 元数据读取完毕后立即释放句柄，避免处理期间重复占用文件
        self.video_cap.release()

        self.video_temp_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        self.video_temp_file.close()  # 立即关闭句柄，否则 Windows 上 ffmpeg 无法写入同一路径
        self.video_writer = None
        self.video_out_path = os.path.abspath(os.path.join(os.path.dirname(video_path), f'{self.vd_name}_no_sub.mp4'))

        self.progress_total = 0
        self.isFinished = False
        self.is_successful_merged = False
        self.ab_sections = None
        self.track_data = None
        self._last_preview_time = 0.0

    def _emit_log(self, *args):
        msg = ' '.join(str(a) for a in args)
        logger.info('[output] %s', msg)
        self.callbacks.on_log(msg)

    def _emit_progress(self):
        self.callbacks.on_progress(self.progress_total, self.isFinished)

    def _emit_preview(self, frame_ori, frame_comp):
        # 节流：预览只用于可视化，没必要每帧都做 copy + 跨进程传输
        now = time.time()
        if now - self._last_preview_time < self.PREVIEW_MIN_INTERVAL:
            return
        self._last_preview_time = now
        self.callbacks.on_preview(frame_ori, frame_comp)

    def update_progress(self, tbar, increment):
        tbar.update(increment)
        total = getattr(tbar, 'total', 0) or 0
        pct = int((tbar.n / total) * 100) if total > 0 else 0
        self.progress_total = max(0, min(100, pct))
        self._emit_progress()

    def run(self):
        start_time = time.time()
        logger.info('processing_start: frames=%d, fps=%s', self.frame_count, self.fps)

        if len(self.sub_areas) == 0:
            self._emit_log(tr['Main']['FullScreenProcessingNote'])
            self.sub_areas.append((0, self.frame_height, 0, self.frame_width))

        self._emit_log(tr['Main']['SubtitleArea'].format(self.sub_areas))
        ab_str = str(self.ab_sections) if self.ab_sections else tr['Main']['ABSectionAll']
        self._emit_log(tr['Main']['ABSection'].format(ab_str))
        os.makedirs(os.path.dirname(self.video_out_path), exist_ok=True)

        self.progress_total = 0
        tbar = tqdm(total=self.frame_count, unit='frame', position=0, file=sys.__stdout__, desc='Subtitle Removing')
        try:
            self._log_model()
            self._process_video(tbar)

            self.video_cap.release()
            if self.video_writer is not None:
                written = self.video_writer.frames_written
                if self.frame_count and written < self.frame_count:
                    # 不再用末帧补齐：CAP 帧数对 VFR 不可靠，补帧会插入重复尾巴
                    logger.warning('frame_count_mismatch: decoded=%d, cap_reported=%d',
                                   written, self.frame_count)
                self.video_writer.release()
                if self.video_writer.failed:
                    raise RuntimeError(
                        f'视频编码失败，输出文件可能不完整 (ffmpeg rc={self.video_writer.returncode})')
            self._merge_audio()

            elapsed = round(time.time() - start_time, 1)
            self._emit_log(tr['Main']['FinishedProcessing'].format(elapsed))
            self._emit_log(tr['Main']['SavedTo'].format(self.video_out_path))
            logger.info('processing_complete: output=%s, time=%.1fs', self.video_out_path, elapsed)
            self.isFinished = True
        except Exception:
            logger.exception('processing_error')
            self._emit_log(tr['SubtitleExtractorGUI']['ErrorDuringProcessing'].format(traceback.format_exc()))
            raise
        finally:
            self._cleanup(success=self.isFinished)

    def _process_video(self, tbar):
        mode = config.inpaintMode.value
        if mode == InpaintMode.STTN_AUTO:
            self._emit_log(tr['Main']['ProcessingStartRemovingSubtitles'])
            mask = create_mask(self.mask_size, list(self.sub_areas))
            self.video_writer = FFmpegVideoWriter(
                get_readable_path(self.video_temp_file.name), self.fps, self.size)
            sttn = STTNAutoInpaint(self.hardware_accelerator.device,
                                   config.sttnCheckpoint.value or MODEL_PATH,
                                   self.video_path)
            sttn(input_mask=mask, input_sub_remover=self, tbar=tbar,
                 gui_mode=self.gui_mode)
            return

        if mode == InpaintMode.E2FGVI:
            self._emit_log(tr['Main']['ProcessingStartRemovingSubtitles'])
            mask = create_mask(self.mask_size, list(self.sub_areas))
            self.video_writer = FFmpegVideoWriter(
                get_readable_path(self.video_temp_file.name), self.fps, self.size)
            e2fgvi = E2FGVIAutoInpaint(self.hardware_accelerator.device,
                                       config.e2fgviCheckpoint.value,
                                       self.video_path)
            e2fgvi(input_mask=mask, input_sub_remover=self, tbar=tbar,
                   gui_mode=self.gui_mode)
            return

        if mode != InpaintMode.PROPAINTER:
            raise ValueError(f'未知处理模型: {mode}')
        self._emit_log(tr['Main']['ProcessingStartRemovingSubtitles'])
        mask = create_mask(self.mask_size, list(self.sub_areas))
        self.video_writer = FFmpegVideoWriter(
            get_readable_path(self.video_temp_file.name), self.fps, self.size)
        propainter = ProPainterAutoInpaint(self.hardware_accelerator.device,
                                           config.propainterCheckpoint.value,
                                           self.video_path)
        propainter(input_mask=mask, input_sub_remover=self, tbar=tbar,
                   gui_mode=self.gui_mode)

    def _log_model(self):
        device = 'CPU'
        if self.hardware_accelerator.has_accelerator():
            name = self.hardware_accelerator.accelerator_name
            if self.hardware_accelerator.has_cuda() or self.hardware_accelerator.has_mps():
                device = name
            elif name == 'DirectML':
                device = 'DirectML'
        model_name = {
            InpaintMode.STTN_AUTO: 'STTN v4',
            InpaintMode.E2FGVI: 'E2FGVI HQ',
            InpaintMode.PROPAINTER: 'ProPainter',
        }.get(config.inpaintMode.value, str(config.inpaintMode.value))
        self._emit_log(tr['Main']['SubtitleRemoverModel'].format(f"{model_name} ({device})"))

    def _merge_audio(self):
        self.is_successful_merged = merge_audio_to_video(
            self.video_path, self.video_temp_file.name, self.video_out_path,
            FFmpegCLI.instance().ffmpeg_path
        )
        if not self.is_successful_merged:
            self._emit_log(f'音频合并失败，已输出无音频视频: {self.video_out_path}')
            logger.error('audio_merge_failed_fallback_copy: out=%s', self.video_out_path)
            try:
                shutil.copy2(self.video_temp_file.name, self.video_out_path)
            except IOError as e:
                self._emit_log(tr['Main']['CopyFileFailed'].format(self.video_temp_file.name, self.video_out_path, str(e)))
        self.video_temp_file.close()

    def _cleanup(self, success=False):
        try:
            self.video_cap.release()
        except Exception:
            pass
        try:
            if self.video_writer is not None:
                self.video_writer.release()
        except Exception:
            pass
        if success:
            self.isFinished = True
            self.progress_total = 100
        # 失败时不置 100%/完成，避免 UI 同时显示"错误"和"完成"
        self._emit_progress()
        try:
            self.video_temp_file.close()
        except Exception:
            pass
        if os.path.exists(self.video_temp_file.name):
            try:
                os.remove(self.video_temp_file.name)
            except Exception:
                pass
