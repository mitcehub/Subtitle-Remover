"""远程调用封装：多进程环境下的回调函数传递。"""

import multiprocessing
import threading
import logging
import pickle
import cv2
from enum import Enum

logger = logging.getLogger(__name__)


class Command(Enum):
    FINISH = 0
    PROGRESS = 1
    LOG = 2
    ERROR = 3
    UPDATE_PREVIEW_WITH_COMP = 4


class SubtitleRemoverRemoteCall:
    """
    远程回调函数类，用于在多进程环境中传递回调函数
    """
    def __init__(self):
        self.queue = multiprocessing.Queue()
        self.callbacks = {}
        self.running = True
        self._thread = threading.Thread(target=self.run, daemon=True)
        self._thread.start()

    def run(self):
        try:
            while self.running:
                try:
                    cmd, args = self.queue.get(block=True)
                except (EOFError, BrokenPipeError, ConnectionResetError, OSError, AssertionError):
                    break
                if cmd == Command.FINISH:
                    break
                callback = self.callbacks.get(cmd)
                if callback:
                    # 单个回调异常不应终止整个消息循环，否则后续 PROGRESS/LOG/FINISH/ERROR 全部丢失
                    try:
                        callback(*args)
                    except Exception:
                        logger.exception('RemoteCall callback failed for cmd=%s', cmd)
        except Exception:
            pass
        finally:
            self.running = False

    def stop(self):
        self.running = False
        try:
            # FINISH 排在已缓冲消息之后，读取线程会先派发剩余日志/进度再退出
            self.queue.put((Command.FINISH, (None,)))
        except Exception:
            pass
        if self._thread.is_alive():
            self._thread.join(timeout=3)
            if self._thread.is_alive():
                logger.warning('remote_call_thread_still_alive_after_stop')

    def register_update_progress_callback(self, callback):
        self.callbacks[Command.PROGRESS] = callback

    def register_log_callback(self, callback):
        self.callbacks[Command.LOG] = callback

    def register_update_preview_with_comp_callback(self, callback):
        self.callbacks[Command.UPDATE_PREVIEW_WITH_COMP] = callback

    def register_error_callback(self, callback):
        self.callbacks[Command.ERROR] = callback

    @staticmethod
    def remote_call_update_progress(queue, progress, isFinished):
        queue.put((Command.PROGRESS, (progress, isFinished,)))

    @staticmethod
    def remote_call_append_log(queue, *args):
        queue.put((Command.LOG, (*args,)))

    @staticmethod
    def remote_call_finish(queue, *args):
        queue.put((Command.FINISH, (None,)))

    @staticmethod
    def remote_call_catch_error(queue, e):
        # multiprocessing.Queue 的序列化发生在 feeder 线程，queue.put() 捕获不到 pickle 失败；
        # 先 pickle.dumps 预校验，失败则退化为字符串，避免错误通知被静默丢弃。
        try:
            pickle.dumps(e)
        except Exception:
            e = f'{type(e).__name__}: {e}'
        queue.put((Command.ERROR, (e,)))

    @staticmethod
    def _downscale_for_preview(frame, max_width=640):
        if frame is None:
            return None
        h, w = frame.shape[:2]
        if w <= max_width:
            return frame
        scale = max_width / w
        new_h = int(h * scale)
        return cv2.resize(frame, (max_width, new_h), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _downscale_arg(arg):
        """递归降采样：识别 list/tuple 内的帧并逐帧缩放，保持 list 契约。"""
        if isinstance(arg, cv2.UMat):
            arg = arg.get()
        if isinstance(arg, (list, tuple)):
            return [SubtitleRemoverRemoteCall._downscale_arg(x) for x in arg]
        if hasattr(arg, 'shape'):
            return SubtitleRemoverRemoteCall._downscale_for_preview(arg)
        return arg

    @staticmethod
    def remote_call_update_preview_with_comp(queue, *args):
        # pipeline 以 queue, [ori, comp] 传单个 list 参数；这里识别 list/tuple 并逐帧降采样，
        # 否则 list 没有 shape，_downscale_for_preview 永不执行，4K 原图会按原始大小 pickle 传输。
        downscaled_args = [SubtitleRemoverRemoteCall._downscale_arg(arg) for arg in args]
        queue.put((Command.UPDATE_PREVIEW_WITH_COMP, (*downscaled_args,)))
