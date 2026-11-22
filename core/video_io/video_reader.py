"""视频读取：创建 VideoCapture 与帧预读取。"""

import os
import queue
import threading
import logging

import cv2

from infra.utils import get_readable_path

logger = logging.getLogger(__name__)


def create_video_capture(path, api_preference=None):
    """创建 VideoCapture，并在打开时请求硬件加速解码。

    必须在 open 之前通过构造参数传入 CAP_PROP_HW_ACCELERATION，
    打开后再 set 在多数后端上无效（会静默回退软解）。
    打开失败时抛出 FileNotFoundError。
    """
    # 统一做短路径转换（仅 Windows 生效），避免含中文/Unicode 路径打开失败
    safe_path = get_readable_path(path) or path
    api = api_preference if api_preference is not None else cv2.CAP_ANY
    accel_params = []
    if hasattr(cv2, 'CAP_PROP_HW_ACCELERATION') and hasattr(cv2, 'VIDEO_ACCELERATION_ANY'):
        accel_params = [cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY]

    cap = None
    if accel_params:
        try:
            cand = cv2.VideoCapture(safe_path, api, accel_params)
            if cand.isOpened():
                cap = cand
            else:
                cand.release()
        except Exception:
            cap = None
    if cap is None:
        try:
            cap = cv2.VideoCapture(safe_path, api)
        except Exception:
            cap = cv2.VideoCapture(safe_path)
    if not cap.isOpened():
        cap.release()
        raise FileNotFoundError(f'cannot open video: {path}')

    hw_accel = None
    if hasattr(cv2, 'CAP_PROP_HW_ACCELERATION'):
        try:
            hw_accel = int(cap.get(cv2.CAP_PROP_HW_ACCELERATION))
        except Exception:
            pass
    logger.info('video_capture_created: path=%s, hw_accel=%s', path, hw_accel)
    return cap


class FramePrefetcher:
    """
    后台线程预解码视频帧，使 I/O 与模型推理重叠。
    接口兼容 cv2.VideoCapture（read/release）。
    """

    _SENTINEL = (False, None)
    # read() 阻塞超时（秒），避免损坏视频导致永久挂起
    _READ_TIMEOUT = 30

    def __init__(self, video_cap, buffer_size=10):
        self.cap = video_cap
        self._buffer = queue.Queue(maxsize=buffer_size)
        self._stopped = threading.Event()
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        logger.info('frame_prefetcher_started: buffer_size=%d', buffer_size)

    def _safe_put(self, item):
        """带超时的入队；停止后立即放弃，避免消费者退出时生产者永久阻塞。"""
        while not self._stopped.is_set():
            try:
                self._buffer.put(item, timeout=0.5)
                return True
            except queue.Full:
                continue
        return False

    def _read_loop(self):
        frames_read = 0
        try:
            while not self._stopped.is_set():
                ret, frame = self.cap.read()
                if not self._safe_put((ret, frame)):
                    return
                frames_read += 1
                if not ret:
                    break
            logger.info('frame_prefetcher_finished: frames_read=%d', frames_read)
        except Exception:
            logger.exception('frame_prefetcher_thread_error')
            self._safe_put((False, None))
            logger.info('frame_prefetcher_error: frames_before_error=%d', frames_read)

    def read(self):
        """读取下一帧，接口与 cv2.VideoCapture.read() 一致。

        超时抛 TimeoutError（而不是返回 (False, None)），
        避免调用方把"生产者卡死"误判成"视频结束"而静默截断输出。
        """
        try:
            result = self._buffer.get(timeout=self._READ_TIMEOUT)
        except queue.Empty:
            logger.error('frame_prefetcher_read_timeout: %ds', self._READ_TIMEOUT)
            raise TimeoutError(f'FramePrefetcher read timeout after {self._READ_TIMEOUT}s')
        ret, frame = result
        return ret, frame

    def get(self, prop_id):
        return self.cap.get(prop_id)

    def stop(self):
        """停止预读取，不释放底层 video_cap。"""
        self._stopped.set()
        qsize = self._buffer.qsize()
        try:
            while not self._buffer.empty():
                self._buffer.get_nowait()
        except queue.Empty:
            pass
        self._thread.join(timeout=5)
        if self._thread.is_alive():
            logger.warning('frame_prefetcher_thread_still_alive_after_join')
        logger.info('frame_prefetcher_stopped: buffer_drained=%d', qsize)

    def release(self):
        """停止预读取并释放底层 video_cap。"""
        self.stop()
        self.cap.release()
