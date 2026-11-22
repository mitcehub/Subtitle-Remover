"""视频预览解码器：后台线程按需解码帧并缓存。"""

import cv2
import queue
import threading
import logging

from core.video_io.video_reader import create_video_capture

logger = logging.getLogger(__name__)


class PreviewDecoder:
    # 预览上限宽度：缓存的是缩放后的帧，避免 4K 原图占满内存（120 帧 4K ≈ 3GB）
    MAX_PREVIEW_WIDTH = 1280

    def __init__(self, video_path, result_queue, max_cache=24):
        self.video_path = video_path
        self.queue = result_queue
        self.max_cache = max_cache
        self._target = -1
        self._running = True
        self._cache = {}
        self._order = []
        self._event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def seek(self, frame_idx):
        self._target = frame_idx
        self._event.set()

    def stop(self):
        self._running = False
        self._event.set()
        self._thread.join(timeout=2)

    def _run(self):
        cap = create_video_capture(self.video_path)
        cap_pos = -1
        while self._running:
            self._event.wait(timeout=0.5)
            self._event.clear()
            if not self._running:
                break
            fi = self._target
            if fi < 0:
                continue
            if fi in self._cache:
                self._put_result(fi, self._cache[fi])
                continue
            pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
            if 0 <= fi - pos <= 60 and cap_pos >= 0:
                for _ in range(int(fi - pos)):
                    cap.read()
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, frame = cap.read()
            if ret:
                cap_pos = fi
                frame = self._prepare(frame)
                if len(self._order) >= self.max_cache:
                    old = self._order.pop(0)
                    self._cache.pop(old, None)
                self._cache[fi] = frame
                self._order.append(fi)
                self._put_result(fi, frame)
        cap.release()

    @classmethod
    def _prepare(cls, frame):
        """缩放到预览上限宽度，控制缓存内存占用。"""
        if frame is None:
            return None
        h, w = frame.shape[:2]
        if w <= cls.MAX_PREVIEW_WIDTH:
            return frame
        scale = cls.MAX_PREVIEW_WIDTH / w
        return cv2.resize(frame, (cls.MAX_PREVIEW_WIDTH, int(h * scale)),
                          interpolation=cv2.INTER_AREA)

    def _put_result(self, fi, frame):
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break
        self.queue.put((fi, frame))
