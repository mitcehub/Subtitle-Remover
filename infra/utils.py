"""通用工具函数：文件类型检测、路径处理。"""

import os
import sys
import ctypes

VIDEO_EXTENSIONS = {
    '.mp4', '.m4a', '.m4v', '.f4v', '.f4a', '.m4b', '.m4r', '.f4b', '.mov',
    '.3gp', '.3gp2', '.3g2', '.3gpp', '.3gpp2', '.ogg', '.oga', '.ogv', '.ogx',
    '.wmv', '.wma', '.asf', '.webm', '.flv', '.avi', '.gifv', '.mkv', '.rm',
    '.rmvb', '.vob', '.dvd', '.mpg', '.mpeg', '.mp2', '.mpe', '.mpv',
    '.m2v', '.svi', '.mxf', '.roq', '.nsv', '.f4p'
}


def get_readable_path(path):
    if sys.platform != 'win32':
        return path
    buf = ctypes.create_unicode_buffer(4096)
    ret = ctypes.windll.kernel32.GetShortPathNameW(path, buf, 4096)
    return buf.value if ret != 0 else path


def paths_refer_to_same_file(path_a, path_b):
    """Return whether two paths identify the same file, including symlinks/hardlinks."""
    if not path_a or not path_b:
        return False
    path_a = os.path.realpath(os.path.abspath(path_a))
    path_b = os.path.realpath(os.path.abspath(path_b))
    if os.path.normcase(path_a) == os.path.normcase(path_b):
        return True
    try:
        return os.path.samefile(path_a, path_b)
    except (OSError, ValueError):
        return False


def is_frame_in_sections(frame_no, sections):
    """判断帧号是否在 AB section 范围内。
    sections 为元组列表 [(start, end), ...]，半开区间 [start, end)。
    兼容旧版 range 对象（自动检测类型）。
    """
    if sections is None or len(sections) <= 0:
        return True
    for section in sections:
        if isinstance(section, tuple):
            start, end = section
            if start <= frame_no < end:
                return True
        else:
            # 兼容 range 对象
            if frame_no in section:
                return True
    return False
