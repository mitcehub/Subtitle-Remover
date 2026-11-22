"""配置管理：常量、枚举、轻量配置层、模型路径、翻译加载。

不依赖任何第三方 UI 框架：配置项自带校验，读写标准 JSON。
对外 API 保持兼容：
    config.<item>.value
    config.set(item, value)
    config.appRestartSig.connect(fn)
    config.interfaceTexts / config.get_sttn_max_load_num()
"""

import os
import json
import logging
import configparser
from enum import Enum, unique
from pathlib import Path

logger = logging.getLogger(__name__)

# ============================================================================
# 常量
# ============================================================================

VERSION = "1.5"
BASE_DIR = str(Path(os.path.abspath(__file__)).parent.parent)
_STTN_DIR = os.path.join(BASE_DIR, 'core', 'inpaint', 'models', 'sttn')
_MODEL_PATH_FIXED = os.path.join(_STTN_DIR, 'sttn.pt')
if os.path.isfile(_MODEL_PATH_FIXED):
    MODEL_PATH = _MODEL_PATH_FIXED
else:
    _MODEL_EXTS = ('.pt', '.pth', '.ckpt', '.bin')
    _STTN_FILES = os.listdir(_STTN_DIR) if os.path.isdir(_STTN_DIR) else []
    MODEL_PATH = next(
        (os.path.join(_STTN_DIR, f) for f in _STTN_FILES
         if os.path.splitext(f)[1].lower() in _MODEL_EXTS and not f.endswith('.bak')),
        _MODEL_PATH_FIXED  # fallback — will fail with clear error at load time
    )
    logger.warning('sttn_model_path_fallback: sttn.pt not found, using %s', MODEL_PATH)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# ============================================================================
# 枚举
# ============================================================================

@unique
class InpaintMode(Enum):
    STTN_AUTO = "sttn-auto"
    E2FGVI = "e2fgvi"
    PROPAINTER = "propainter"

# ============================================================================
# 模型路径（三个模型均为内嵌推理，权重随项目分发；可在 config.json 的 Models 段覆盖）
# ============================================================================

DEFAULT_E2FGVI_CKPT = os.path.join(
    BASE_DIR, 'core', 'inpaint', 'models', 'e2fgvi', 'e2fgvi.pth')
DEFAULT_PROPAINTER_CKPT = os.path.join(
    BASE_DIR, 'core', 'inpaint', 'models', 'propainter', 'propainter.pt')


def resolve_model_path(path):
    """模型权重路径兼容：相对路径按项目根目录解析，绝对路径原样返回。

    config.json 中存相对路径（便于迁移/多机部署），传给 torch.load 前
    统一转成基于 BASE_DIR 的绝对路径，避免依赖进程工作目录。
    """
    if not path:
        return path
    if os.path.isabs(path):
        return path
    return os.path.join(BASE_DIR, path)

# ============================================================================
# 校验器
# ============================================================================

class Validator:
    """默认校验器：原样返回。"""

    def validate(self, value):
        return value


class BoolValidator(Validator):
    def validate(self, value):
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in ('1', 'true', 'yes', 'on')
        return bool(value)


class RangeValidator(Validator):
    """整数区间校验；越界会被夹紧到 [min, max]。"""

    def __init__(self, minimum, maximum):
        self.min = minimum
        self.max = maximum

    def validate(self, value):
        try:
            value = int(value)
        except (TypeError, ValueError):
            value = self.min
        return max(self.min, min(self.max, value))


class OptionsValidator(Validator):
    """可选值校验；非法值回落到第一个选项。"""

    def __init__(self, options):
        self.options = list(options)

    def validate(self, value):
        return value if value in self.options else self.options[0]


class ConfigValidator(Validator):
    """通用校验器占位（路径、坐标等自由文本）。"""


class EnumSerializer:
    """JSON 字符串 <-> Enum 成员互转。"""

    def __init__(self, enum_cls):
        self.enum = enum_cls

    def to_json(self, value):
        return value.value if isinstance(value, self.enum) else value

    def from_json(self, value):
        if isinstance(value, self.enum):
            return value
        return self.enum(value)


# ============================================================================
# 配置项
# ============================================================================

class ConfigItem:
    """单个配置项：持有当前值并负责校验与持久化。"""

    def __init__(self, group, name, default, validator=None,
                 restart=False, serializer=None):
        self.group = group
        self.name = name
        self.default = default
        self.validator = validator or Validator()
        self.restart = restart
        self.serializer = serializer
        self._value = default
        self._owner = None

    @property
    def value(self):
        return self._value

    def set_value(self, value, save=True):
        validated = self.validator.validate(value)
        changed = validated != self._value
        self._value = validated
        if self._owner is not None:
            if save:
                self._owner.save()
            if changed and self.restart:
                self._owner.appRestartSig.emit()
        return changed

    # ---- 序列化 ----
    def dump(self):
        if self.serializer is not None:
            return self.serializer.to_json(self._value)
        return self._value

    def load(self, raw):
        if self.serializer is not None:
            raw = self.serializer.from_json(raw)
        self._value = self.validator.validate(raw)


class _Signal:
    """极简信号：只用于配置变更通知（不依赖 Qt）。"""

    def __init__(self):
        self._slots = []

    def connect(self, slot):
        if slot not in self._slots:
            self._slots.append(slot)

    def disconnect(self, slot=None):
        if slot is None:
            self._slots.clear()
        elif slot in self._slots:
            self._slots.remove(slot)

    def emit(self, *args):
        for slot in list(self._slots):
            try:
                slot(*args)
            except Exception:
                logger.exception('config_signal_slot_error')


# ============================================================================
# Config 类
# ============================================================================

class Config:
    # 仅列出实际随项目分发的翻译文件，避免选择后静默回退中文
    interfaceTexts = {
        '简体中文': 'ch',
        'English': 'en',
    }

    interface = ConfigItem("Window", "Interface", "ch",
                           OptionsValidator(interfaceTexts.values()), restart=True)

    windowX = ConfigItem("Window", "X", None)
    windowY = ConfigItem("Window", "Y", None)
    windowW = ConfigItem("Window", "Width", 1200)
    windowH = ConfigItem("Window", "Height", 900)

    subtitleSelectionAreas = ConfigItem("Main", "SubtitleSelectionAreas", "0.88,0.99,0.15,0.85")

    inpaintMode = ConfigItem("Main", "InpaintMode", InpaintMode.STTN_AUTO,
                             OptionsValidator(InpaintMode),
                             serializer=EnumSerializer(InpaintMode))

    e2fgviCheckpoint = ConfigItem("Models", "E2FGVICheckpoint", DEFAULT_E2FGVI_CKPT)
    propainterCheckpoint = ConfigItem("Models", "ProPainterCheckpoint", DEFAULT_PROPAINTER_CKPT)
    sttnCheckpoint = ConfigItem("Models", "STTNCheckpoint", MODEL_PATH)

    # 约束：NeighborStride × ReferenceLength ≤ MaxLoadNum 下限(50)，
    # 避免 sttn.py 运行时静默抬升 chunk 大小导致 UI 显示值与实际不符。
    # 当前 5 × 10 = 50，正好等于 MaxLoadNum 下限。
    # 区间规则：默认值固定为「最低效果/最低开销」一端，只允许朝提高效果的方向调整。
    # 数值越大效果越好的参数：默认值即下限；数值越小效果越好的参数：默认值即上限。
    sttnNeighborStride = ConfigItem("Sttn", "NeighborStride", 5, RangeValidator(2, 5))
    sttnReferenceLength = ConfigItem("Sttn", "ReferenceLength", 10, RangeValidator(2, 10))
    sttnMaxLoadNum = ConfigItem("Sttn", "MaxLoadNum", 50, RangeValidator(50, 300))

    e2fgviMaxFrames = ConfigItem("E2FGVI", "MaxFrames", 150, RangeValidator(150, 300))
    e2fgviNeighborStride = ConfigItem("E2FGVI", "NeighborStride", 5, RangeValidator(2, 5))
    e2fgviReferenceLength = ConfigItem("E2FGVI", "ReferenceLength", 10, RangeValidator(2, 10))

    propainterLocalFrames = ConfigItem("ProPainter", "LocalFrames", 10, RangeValidator(10, 20))
    propainterReferenceFrames = ConfigItem("ProPainter", "ReferenceFrames", 10, RangeValidator(10, 20))
    propainterStride = ConfigItem("ProPainter", "Stride", 5, RangeValidator(1, 5))

    hardwareAcceleration = ConfigItem("Main", "HardwareAcceleration", True, BoolValidator())

    checkUpdateOnStartup = ConfigItem("Main", "CheckUpdateOnStartup", True, BoolValidator())

    saveDirectory = ConfigItem("Main", "SaveDirectory", "", ConfigValidator())

    def __init__(self):
        self.appRestartSig = _Signal()
        self._items = []
        self._raw = {}
        self._path = None
        for klass in reversed(type(self).__mro__):
            for attr in vars(klass).values():
                if isinstance(attr, ConfigItem):
                    attr._owner = self
                    self._items.append(attr)

    def get_sttn_max_load_num(self):
        return max(self.sttnMaxLoadNum.value,
                   self.sttnNeighborStride.value * self.sttnReferenceLength.value)

    # ---- 读写 ----
    def load(self, path):
        self._path = path
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if not isinstance(data, dict):
                data = {}
        except FileNotFoundError:
            data = {}
        except Exception as e:
            logger.warning('config_load_failed: %s', e)
            data = {}
        self._raw = data
        for item in self._items:
            section = data.get(item.group)
            if isinstance(section, dict) and item.name in section:
                try:
                    item.load(section[item.name])
                except Exception as e:
                    logger.warning('config_item_load_failed: %s.%s: %s',
                                   item.group, item.name, e)
        self.save()

    def save(self):
        if not self._path:
            return
        data = dict(self._raw)
        for item in self._items:
            section = data.get(item.group)
            if not isinstance(section, dict):
                section = {}
                data[item.group] = section
            section[item.name] = item.dump()
        try:
            os.makedirs(os.path.dirname(self._path), exist_ok=True)
            tmp = self._path + '.tmp'
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            os.replace(tmp, self._path)
            self._raw = data
        except Exception:
            logger.exception('config_save_failed: %s', self._path)

    def set(self, item, value):
        item.set_value(value)


# ============================================================================
# 初始化
# ============================================================================

CONFIG_FILE = os.path.join(BASE_DIR, 'config', 'config.json')
config = Config()
config.load(CONFIG_FILE)

tr = configparser.ConfigParser()
TRANSLATION_FILE = os.path.join(BASE_DIR, 'config', 'translations', f"{config.interface.value}.ini")

# Try to load the requested translation file
try:
    tr.read(TRANSLATION_FILE, encoding='utf-8')
    if not tr.sections():
        logger.warning('translation_file_empty_or_missing: %s', TRANSLATION_FILE)
        # Try to load ch.ini as fallback
        fallback_file = os.path.join(BASE_DIR, 'config', 'translations', 'ch.ini')
        tr_fallback = configparser.ConfigParser()
        tr_fallback.read(fallback_file, encoding='utf-8')
        if tr_fallback.sections():
            tr = tr_fallback
            logger.info('Loaded fallback translation from ch.ini')
        else:
            logger.warning('Fallback translation file ch.ini also empty or missing')
except Exception as e:
    logger.warning('Failed to load translation file %s: %s', TRANSLATION_FILE, e)
    # Try to load ch.ini as fallback
    fallback_file = os.path.join(BASE_DIR, 'config', 'translations', 'ch.ini')
    tr_fallback = configparser.ConfigParser()
    try:
        tr_fallback.read(fallback_file, encoding='utf-8')
        if tr_fallback.sections():
            tr = tr_fallback
            logger.info('Loaded fallback translation from ch.ini')
        else:
            logger.warning('Fallback translation file ch.ini also empty or missing')
    except Exception as e2:
        logger.warning('Failed to load fallback translation file ch.ini: %s', e2)
