"""处理流水线：子进程启动与管理。"""

import logging
import multiprocessing
import os
import traceback

from core.subtitle_remover import SubtitleRemover, ProcessingCallbacks
from app.subtitle_remover_remote_call import SubtitleRemoverRemoteCall
from infra.config import BASE_DIR
from infra.process_manager import ProcessManager

logger = logging.getLogger(__name__)

# 允许通过 options setattr 落地的属性白名单，防止覆盖关键属性
_ALLOWED_OPTION_KEYS = frozenset({
    'sub_areas', 'ab_sections', 'track_data', 'video_out_path',
})


class _UIProcessingLogHandler(logging.Handler):
    """Forward useful processing logs from the child process to the GUI."""

    _LOGGER_LABELS = {
        'core.inpaint.sttn': 'STTN',
        'core.inpaint.e2fgvi': 'E2FGVI',
        'core.inpaint.propainter': 'ProPainter',
        'core.inpaint.region': 'Mask',
        'core.subtitle_remover': 'Pipeline',
        'core.video_io.ffmpeg': 'FFmpeg',
        'core.video_io.video_reader': 'VideoReader',
        'core.video_io.video_writer': 'VideoWriter',
        'infra.hardware': 'Hardware',
    }
    _INFO_PREFIXES = (
        'core.inpaint.',
        'core.subtitle_remover',
        'core.video_io.',
        'infra.hardware',
    )

    def __init__(self, queue):
        super().__init__(logging.INFO)
        self.queue = queue

    def emit(self, record):
        try:
            message = record.getMessage()
            # These messages are already sent to the UI by _emit_log.
            if record.name == 'core.subtitle_remover' and message.startswith('[output]'):
                return
            if (record.levelno < logging.WARNING
                    and not record.name.startswith(self._INFO_PREFIXES)):
                return
            label = self._LOGGER_LABELS.get(
                record.name, record.name.rsplit('.', 1)[-1])
            text = f'[{record.levelname}] {label}: {message}'
            SubtitleRemoverRemoteCall.remote_call_append_log(self.queue, [text])
        except Exception:
            self.handleError(record)


def remover_process(queue, video_path, output_path, options):
    """子进程入口：运行字幕去除"""
    failed = False
    root_logger = logging.getLogger()
    previous_level = root_logger.level
    ui_log_handler = _UIProcessingLogHandler(queue)
    ui_log_handler.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)-5s] %(name)s: %(message)s',
        datefmt='%H:%M:%S'))
    root_logger.addHandler(ui_log_handler)
    if root_logger.level > logging.DEBUG:
        root_logger.setLevel(logging.DEBUG)

    # pythonw-launched workers do not inherit gui.py's FileHandler.
    log_path = os.path.join(BASE_DIR, '.cache', 'subtitle-remover.log')
    file_log_handler = None
    if not any(
            isinstance(handler, logging.FileHandler)
            and os.path.abspath(handler.baseFilename) == os.path.abspath(log_path)
            for handler in root_logger.handlers):
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        file_log_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
        file_log_handler.setLevel(logging.DEBUG)
        file_log_handler.setFormatter(logging.Formatter(
            '%(asctime)s [%(levelname)-5s] %(name)s: %(message)s',
            datefmt='%H:%M:%S'))
        root_logger.addHandler(file_log_handler)
    try:
        callbacks = ProcessingCallbacks(
            # 统一以 list 形式回传，匹配 GUI 端 pyqtSignal(list) 的声明
            on_log=lambda msg: SubtitleRemoverRemoteCall.remote_call_append_log(queue, [msg]),
            on_progress=lambda pct, done: SubtitleRemoverRemoteCall.remote_call_update_progress(queue, pct, done),
            on_preview=lambda ori, comp: SubtitleRemoverRemoteCall.remote_call_update_preview_with_comp(
                queue, [ori, comp]),
        )
        sr = SubtitleRemover(video_path, True, callbacks=callbacks)
        sr.video_out_path = output_path
        for key, value in options.items():
            if key in _ALLOWED_OPTION_KEYS:
                setattr(sr, key, value)
            else:
                logger.warning('option_key_not_allowed: %s', key)
        sr.run()
    except Exception as e:
        failed = True
        traceback.print_exc()
        SubtitleRemoverRemoteCall.remote_call_catch_error(queue, e)
    finally:
        root_logger.removeHandler(ui_log_handler)
        ui_log_handler.close()
        if file_log_handler is not None:
            root_logger.removeHandler(file_log_handler)
            file_log_handler.close()
        root_logger.setLevel(previous_level)
        SubtitleRemoverRemoteCall.remote_call_finish(queue)
    if failed:
        # 让退出码反映失败，父进程不会误判为"处理成功"
        raise SystemExit(1)


class ProcessingPipeline:
    """处理流水线管理器"""

    def __init__(self):
        self.running_process = None

    def start(self, video_path, output_path, options, callbacks, on_started=None, stop_event=None):
        """启动处理子进程

        on_started: 在 process.start() 之后、join() 之前回调，用于及时记录 PID。
        stop_event: threading.Event，在等待期间被设置时主动终止僵死子进程，
                    避免 worker 线程在 ffmpeg wait(600s) 等场景下无限阻塞。
        """
        caller = SubtitleRemoverRemoteCall()
        caller.register_update_progress_callback(callbacks.on_progress)
        caller.register_log_callback(callbacks.on_log)
        caller.register_update_preview_with_comp_callback(callbacks.on_preview)
        caller.register_error_callback(callbacks.on_error)

        process = multiprocessing.Process(
            target=remover_process,
            args=(caller.queue, video_path, output_path, options)
        )
        try:
            self.running_process = process
            process.start()
            ProcessManager.instance().add_process(process)
            if on_started is not None:
                on_started(process)
            # 循环 join：正常等待子进程结束；若收到停止信号则主动终止僵死子进程
            while True:
                process.join(timeout=0.5)
                if not process.is_alive():
                    break
                if stop_event is not None and stop_event.is_set():
                    logger.warning('stop_event_set_during_join: terminating pid=%s',
                                   getattr(process, 'pid', None))
                    ProcessManager.instance().terminate_by_process(process)
            logger.info('subprocess_exit: exitcode=%d', process.exitcode)
        finally:
            caller.stop()
            ProcessManager.instance().remove_process(f"Process:{id(process)}")
        return process

    def stop(self):
        """停止当前处理"""
        if self.running_process:
            ProcessManager.instance().terminate_by_process(self.running_process)
            self.running_process = None
