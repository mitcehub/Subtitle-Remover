"""核心纯逻辑回归测试（不依赖 GPU / 模型权重）。

运行：python -m unittest tests.test_regressions -v
"""

import unittest
from types import SimpleNamespace

import numpy as np

from core.inpaint.region import create_mask
from core.inpaint.sttn import STTNAutoInpaint
from core.inpaint.e2fgvi import E2FGVIAutoInpaint
from ui.coordinate_mapper import CoordinateMapper
from ui.preview_decoder import PreviewDecoder


class _Point:
    def __init__(self, x, y):
        self._x = x
        self._y = y

    def x(self):
        return self._x

    def y(self):
        return self._y


class TestCreateMask(unittest.TestCase):
    def test_out_of_bounds_coords_are_clamped(self):
        # 超出画面右下边界的选区不应生成越界 mask，也不应崩溃
        mask = create_mask((100, 200), [(90, 150, 150, 300)])
        self.assertEqual(mask.shape, (100, 200))
        self.assertEqual(mask[95, 195], 255)

    def test_float_coords_are_cast_to_int(self):
        mask = create_mask((100, 200), [(10.4, 20.6, 30.2, 40.8)])
        self.assertEqual(mask.shape, (100, 200))
        self.assertEqual(mask[15, 35], 255)


class TestChunkRangesAlignWithAbsoluteFrames(unittest.TestCase):
    """S1：AB 段起始帧 > 0 时，chunk 必须使用绝对帧号，主循环按绝对帧号消费读取。"""

    def _check(self, cls):
        obj = cls.__new__(cls)
        remover = SimpleNamespace(ab_sections=[(20, 30)])
        ranges = obj._build_chunk_ranges({'len': 40}, effective_gap=10,
                                         input_sub_remover=remover)
        self.assertEqual(ranges[0][0], 20)
        self.assertEqual(ranges[-1][1], 30)
        # 覆盖长度必须等于 AB 段长度，不能从 0 开始
        self.assertEqual(sum(e - s for s, e in ranges), 10)

    def test_sttn(self):
        self._check(STTNAutoInpaint)

    def test_e2fgvi(self):
        self._check(E2FGVIAutoInpaint)


class TestCoordinateMapper(unittest.TestCase):
    def setUp(self):
        self.mapper = CoordinateMapper()
        self.mapper.set_video_params(
            frame_width=1920, frame_height=1080,
            scaled_width=800, scaled_height=450,
            border_left=0, border_top=0, fill_mode=False)

    def test_pos_to_norm_keeps_right_edge(self):
        # 右/下边缘应映射到 1.0，而不是 (sw-1)/sw
        self.assertAlmostEqual(self.mapper.pos_to_norm(_Point(800, 450))[1], 1.0)
        self.assertAlmostEqual(self.mapper.pos_to_norm(_Point(800, 450))[0], 1.0)

    def test_pos_to_norm_clamps_negative(self):
        y, x = self.mapper.pos_to_norm(_Point(-20, -10))
        self.assertEqual((y, x), (0.0, 0.0))

    def test_roundtrip(self):
        rects = [(0.1, 0.2, 0.3, 0.4)]
        video = self.mapper.preview_to_video(rects)
        self.assertEqual(len(video), 1)
        ymin, ymax, xmin, xmax = video[0]
        self.assertLess(ymin, ymax)
        self.assertLess(xmin, xmax)


class TestPreviewDecoderCache(unittest.TestCase):
    """S7：预览缓存必须缩放，避免 4K 原图占满内存。"""

    def test_large_frame_is_downscaled(self):
        frame = np.zeros((2160, 3840, 3), dtype=np.uint8)
        out = PreviewDecoder._prepare(frame)
        self.assertEqual(out.shape[1], PreviewDecoder.MAX_PREVIEW_WIDTH)
        self.assertLess(out.shape[0], 2160)

    def test_small_frame_kept(self):
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        out = PreviewDecoder._prepare(frame)
        self.assertIs(out, frame)

    def test_none_passthrough(self):
        self.assertIsNone(PreviewDecoder._prepare(None))


if __name__ == '__main__':
    unittest.main()
