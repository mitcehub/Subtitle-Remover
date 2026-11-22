"""视频写入：通过 FFmpeg 管道写入帧。"""

import subprocess
import threading
import logging

import numpy as np

from .ffmpeg import FFmpegCLI

logger = logging.getLogger(__name__)


class FFmpegVideoWriter:
    """
    通过 FFmpeg 管道写入帧，libx264 CRF 1 无损输出。
    接口兼容 cv2.VideoWriter（write/release）。
    """

    def __init__(self, output_path, fps, size):
        w, h = size
        cmd = [
            FFmpegCLI.instance().ffmpeg_path,
            '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{w}x{h}',
            '-pix_fmt', 'bgr24',
            '-r', str(fps),
            '-i', '-',
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-crf', '1',
            '-preset', 'medium',
            '-loglevel', 'error',
            output_path
        ]
        self._process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        self._returncode = None
        self._frames_written = 0
        self._size = (w, h)
        self._broken = False
        self._failed = False
        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stderr_thread.start()
        logger.info('video_writer_initialized: path=%s, fps=%s, %dx%d', output_path, fps, w, h)

    def _drain_stderr(self):
        """读取 FFmpeg stderr 到日志，避免 pipe buffer 满阻塞，同时保留诊断信息。"""
        try:
            for line in self._process.stderr:
                if line.strip():
                    logger.warning('[ffmpeg] %s', line.rstrip())
        except Exception:
            pass

    def write(self, frame):
        """写入一帧（numpy BGR 数组）。"""
        if self._broken:
            raise RuntimeError('FFmpegVideoWriter: pipe broken, write aborted')
        if frame is None or getattr(frame, 'shape', None) is None:
            raise ValueError('FFmpegVideoWriter.write received an empty frame')
        if frame.ndim == 2:
            frame = np.stack([frame] * 3, axis=-1)
        if frame.shape[0] != self._size[1] or frame.shape[1] != self._size[0]:
            raise ValueError(
                f'FFmpegVideoWriter frame size mismatch: got {frame.shape[1]}x{frame.shape[0]}, '
                f'expected {self._size[0]}x{self._size[1]}')
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        try:
            self._process.stdin.write(frame.tobytes())
        except BrokenPipeError:
            self._broken = True
            self._failed = True
            logger.error('video_writer_broken_pipe: frames_written=%d', self._frames_written)
            raise RuntimeError('FFmpegVideoWriter: pipe broken (ffmpeg exited early)')
        self._frames_written += 1

    @property
    def frames_written(self):
        return self._frames_written

    @property
    def failed(self):
        """FFmpeg 是否异常退出（非 0）或管道已断开。"""
        return self._failed

    @property
    def returncode(self):
        return self._returncode

    def release(self):
        """关闭管道并等待编码完成。"""
        if self._returncode is not None:
            return
        try:
            self._process.stdin.close()
        except BrokenPipeError:
            logger.warning('video_writer_broken_pipe_on_close')
        try:
            self._process.wait(timeout=600)
        except subprocess.TimeoutExpired:
            logger.warning('video_writer_timeout: terminating')
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.error('video_writer_terminate_timeout')
                self._failed = True
        self._returncode = self._process.returncode
        if self._returncode not in (0, None) and not self._broken:
            self._failed = True
            logger.error('video_writer_failed: rc=%s', self._returncode)
        logger.info('video_writer_released: frames=%d, rc=%s, size=%s', self._frames_written, self._returncode, self._size)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()
        return False
