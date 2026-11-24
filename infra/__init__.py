"""基础设施层：配置、硬件、进程管理、工具函数。"""

from infra.config import config, tr, VERSION, BASE_DIR, MODEL_PATH
from infra.process_manager import ProcessManager
from infra.utils import VIDEO_EXTENSIONS, get_readable_path, is_frame_in_sections


def __getattr__(name):
    """Load hardware detection lazily so importing app configuration stays fast."""
    if name == 'HardwareAccelerator':
        from infra.hardware import HardwareAccelerator
        return HardwareAccelerator
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
