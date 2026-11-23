"""核心纯逻辑回归测试（不依赖 GPU / 模型权重）。

运行：python -m unittest tests.test_regressions -v
"""

import io
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from core.cli import parse_args
from core.subtitle_remover import ProcessingCallbacks, SubtitleRemover
from core.inpaint.region import create_mask
from core.inpaint.sttn import STTNAutoInpaint
from core.inpaint.e2fgvi import E2FGVIAutoInpaint
from core.inpaint.propainter import (
    _iter_video_chunks,
    _release_reference_cache,
    _select_reference_indices,
    ProPainterAutoInpaint,
)
from infra.utils import get_inpaint_output_name, paths_refer_to_same_file
from ui.coordinate_mapper import CoordinateMapper
from ui.preview_decoder import PreviewDecoder
from ui.pages.home_page import HomePage


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


class TestProPainterReferenceSelection(unittest.TestCase):
    def test_top_k_matches_stable_full_sort(self):
        rng = np.random.default_rng(12)
        index = rng.integers(0, 256, size=(23, 18, 32), dtype=np.uint8)
        anchor = index[7]
        excluded = [6, 7, 8]
        count = 5
        scores = np.abs(index.astype(np.float32) - anchor.astype(np.float32)).mean(
            axis=(1, 2))
        scores[excluded] = np.inf
        expected = sorted(np.argsort(scores, kind='stable')[:count].tolist())

        self.assertEqual(
            _select_reference_indices(index, anchor, excluded, count, tile_size=4),
            expected)

    def test_equal_scores_preserve_stable_tie_order(self):
        index = np.zeros((12, 18, 32), dtype=np.uint8)
        self.assertEqual(
            _select_reference_indices(index, index[0], [0, 1], 4, tile_size=3),
            [2, 3, 4, 5])

    def test_processing_chunks_cover_cores_once_with_overlap(self):
        chunks = list(_iter_video_chunks(total=50, chunk_frames=20, overlap=3))

        self.assertEqual(
            chunks,
            [(0, 20, 0, 23), (20, 40, 17, 43), (40, 50, 37, 50)])
        cores = [frame for start, end, _, _ in chunks for frame in range(start, end)]
        self.assertEqual(cores, list(range(50)))

    def test_reference_cache_reads_only_requested_chunk(self):
        class FakeCapture:
            def __init__(self, frames):
                self.frames = frames
                self.pos = 0
                self.released = False

            def set(self, _prop, frame_index):
                self.pos = int(frame_index)
                return True

            def get(self, _prop):
                return self.pos

            def read(self):
                if self.pos >= len(self.frames):
                    return False, None
                frame = self.frames[self.pos]
                self.pos += 1
                return True, frame

            def release(self):
                self.released = True

        frames = [np.full((24, 32, 3), i, dtype=np.uint8) for i in range(10)]
        cap = FakeCapture(frames)
        obj = ProPainterAutoInpaint.__new__(ProPainterAutoInpaint)
        obj.video_path = 'fake-video'
        with mock.patch('core.inpaint.propainter.create_video_capture',
                        return_value=cap):
            index, ref_mm, ref_path, _ = obj._build_index_and_refs(
                (0, 0, 32, 24), 3, 8)
        try:
            self.assertEqual(index.shape, (5, 18, 32))
            self.assertEqual(ref_mm.shape[0], 5)
            self.assertEqual(cap.pos, 8)
            self.assertTrue(cap.released)
        finally:
            _release_reference_cache(ref_mm, ref_path)

    def test_chunk_processor_emits_each_core_frame_once(self):
        class FakeCapture:
            def __init__(self, frames):
                self.frames = frames
                self.pos = 0

            def read(self):
                if self.pos >= len(self.frames):
                    return False, None
                frame = self.frames[self.pos]
                self.pos += 1
                return True, frame

            def release(self):
                pass

        class FakeInpaint:
            device = 'cpu'

            def inpaint_window(self, win, num_local):
                b, _, _, h, w = win.shape
                return win[:, :num_local], np.zeros(
                    (b, num_local, h, w), dtype=np.float32)

        class FakeWriter:
            def __init__(self):
                self.frames = []

            def write(self, frame):
                self.frames.append(frame.copy())

        frames = [
            np.full((24, 32, 3), i, dtype=np.uint8)
            for i in range(5)
        ]
        obj = ProPainterAutoInpaint.__new__(ProPainterAutoInpaint)
        obj.video_path = 'fake-video'
        obj.inpaint = FakeInpaint()
        writer = FakeWriter()
        index = np.zeros((5, 18, 32), dtype=np.uint8)
        refs = np.zeros((5, 120, 640, 3), dtype=np.uint8)
        cap = FakeCapture(frames)

        with mock.patch('core.inpaint.propainter.create_video_capture',
                        return_value=cap):
            written = obj._process_chunk(
                writer, index, refs, (640, 120), (0, 0, 32, 24),
                video_total=5, sections=[(0, 5)], local=3, ref_frames=1,
                stride=2, core_start=0, core_end=5,
                context_start=0, context_end=5,
                input_sub_remover=None, tbar=None, gui_mode=False)

        self.assertEqual(written, 5)
        self.assertEqual(len(writer.frames), 5)
        for actual, expected in zip(writer.frames, frames):
            np.testing.assert_array_equal(actual, expected)


class TestVideoWriterBuffer(unittest.TestCase):
    def test_write_accepts_non_contiguous_frame_without_bytes_copy(self):
        from core.video_io.video_writer import FFmpegVideoWriter

        writer = FFmpegVideoWriter.__new__(FFmpegVideoWriter)
        writer._broken = False
        writer._failed = False
        writer._frames_written = 0
        writer._size = (2, 2)
        writer._process = SimpleNamespace(stdin=io.BytesIO())
        frame = np.arange(2 * 4 * 3, dtype=np.uint8).reshape(2, 4, 3)[:, ::2]

        writer.write(frame)

        self.assertEqual(writer._process.stdin.getvalue(), frame.copy().tobytes())
        self.assertEqual(writer.frames_written, 1)


class TestSafeOutputPath(unittest.TestCase):
    """输出路径以 O_EXCL 原子预留，避免覆盖文件或跟随既有符号链接。"""

    def test_existing_file_gets_suffix(self):
        with tempfile.TemporaryDirectory() as d:
            path = HomePage._safe_output_path(d, 'video', '.mp4', 'e2fgvi')
            self.assertEqual(os.path.basename(path), 'video_e2fgvi_no_sub.mp4')
            self.assertTrue(os.path.isfile(path))
            # 创建一个同名文件后再次生成，应换名而非复用
            with open(path, 'wb'):
                pass
            path2 = HomePage._safe_output_path(d, 'video', '.mp4', 'e2fgvi')
            self.assertEqual(os.path.basename(path2), 'video_e2fgvi_no_sub_1.mp4')
            self.assertTrue(os.path.isfile(path2))

    def test_symlink_is_avoided(self):
        with tempfile.TemporaryDirectory() as d:
            target = os.path.join(d, 'secret.txt')
            with open(target, 'wb'):
                pass
            link = os.path.join(d, 'video_e2fgvi_no_sub.mp4')
            try:
                os.symlink(target, link)
            except (OSError, NotImplementedError):
                self.skipTest('symlink not supported')
            path = HomePage._safe_output_path(d, 'video', '.mp4', 'e2fgvi')
            self.assertNotEqual(path, link)
            self.assertEqual(os.path.basename(path), 'video_e2fgvi_no_sub_1.mp4')

    def test_model_names_are_stable_and_short(self):
        from infra.config import InpaintMode

        self.assertEqual(get_inpaint_output_name(InpaintMode.STTN_AUTO), 'sttn')
        self.assertEqual(get_inpaint_output_name(InpaintMode.E2FGVI), 'e2fgvi')
        self.assertEqual(get_inpaint_output_name(InpaintMode.PROPAINTER), 'propainter')


class TestInputOutputPathSafety(unittest.TestCase):
    def test_same_path_is_detected(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'video.mp4')
            with open(path, 'wb'):
                pass
            self.assertTrue(paths_refer_to_same_file(path, path))

    def test_hardlink_to_input_is_detected(self):
        with tempfile.TemporaryDirectory() as d:
            source = os.path.join(d, 'input.mp4')
            alias = os.path.join(d, 'output.mp4')
            with open(source, 'wb') as f:
                f.write(b'video')
            try:
                os.link(source, alias)
            except (OSError, NotImplementedError):
                self.skipTest('hardlinks not supported')
            self.assertTrue(paths_refer_to_same_file(source, alias))

    def test_different_paths_are_allowed(self):
        with tempfile.TemporaryDirectory() as d:
            source = os.path.join(d, 'input.mp4')
            output = os.path.join(d, 'output.mp4')
            self.assertFalse(paths_refer_to_same_file(source, output))

    def test_cli_rejects_output_equal_to_input(self):
        with tempfile.TemporaryDirectory() as d:
            source = os.path.join(d, 'input.mp4')
            with open(source, 'wb'):
                pass
            with mock.patch.object(sys, 'argv', ['cli', source, '--output', source]):
                with self.assertRaises(SystemExit) as exc:
                    parse_args()
            self.assertEqual(exc.exception.code, 2)


class TestAudioMergeFallback(unittest.TestCase):
    def test_failed_fallback_copy_raises(self):
        with tempfile.TemporaryDirectory() as d:
            temp_video = os.path.join(d, 'processed.mp4')
            with open(temp_video, 'wb') as f:
                f.write(b'processed')
            output = os.path.join(d, 'output.mp4')
            obj = SubtitleRemover.__new__(SubtitleRemover)
            obj.video_path = os.path.join(d, 'input.mp4')
            obj.video_out_path = output
            obj.video_temp_file = SimpleNamespace(name=temp_video, close=mock.Mock())
            obj.callbacks = ProcessingCallbacks(on_log=lambda _msg: None)

            with mock.patch('core.subtitle_remover.FFmpegCLI.instance',
                            return_value=SimpleNamespace(ffmpeg_path='ffmpeg')):
                with mock.patch('core.subtitle_remover.merge_audio_to_video', return_value=False):
                    with mock.patch('core.subtitle_remover.shutil.copy2',
                                    side_effect=PermissionError('denied')):
                        with self.assertRaises(RuntimeError):
                            obj._merge_audio()
            obj.video_temp_file.close.assert_called_once()

    def test_successful_fallback_copy_produces_output(self):
        with tempfile.TemporaryDirectory() as d:
            temp_video = os.path.join(d, 'processed.mp4')
            with open(temp_video, 'wb') as f:
                f.write(b'processed')
            output = os.path.join(d, 'output.mp4')
            obj = SubtitleRemover.__new__(SubtitleRemover)
            obj.video_path = os.path.join(d, 'input.mp4')
            obj.video_out_path = output
            obj.video_temp_file = SimpleNamespace(name=temp_video, close=mock.Mock())
            obj.callbacks = ProcessingCallbacks(on_log=lambda _msg: None)

            with mock.patch('core.subtitle_remover.FFmpegCLI.instance',
                            return_value=SimpleNamespace(ffmpeg_path='ffmpeg')):
                with mock.patch('core.subtitle_remover.merge_audio_to_video', return_value=False):
                    obj._merge_audio()
            with open(output, 'rb') as f:
                self.assertEqual(f.read(), b'processed')
            obj.video_temp_file.close.assert_called_once()


class TestProcessingCallbackThrottling(unittest.TestCase):
    def test_preview_slot_limits_full_frame_copies(self):
        obj = SubtitleRemover.__new__(SubtitleRemover)
        obj._last_preview_time = 0.0
        with mock.patch('core.subtitle_remover.time.monotonic',
                        side_effect=[1.0, 1.05, 1.11]):
            self.assertTrue(obj._should_emit_preview())
            self.assertFalse(obj._should_emit_preview())
            self.assertTrue(obj._should_emit_preview())

    def test_progress_callbacks_are_throttled(self):
        obj = SubtitleRemover.__new__(SubtitleRemover)
        obj._last_progress_time = 0.0
        obj._last_progress_pct = -1
        obj.progress_total = 0
        obj.isFinished = False
        obj.callbacks = ProcessingCallbacks(on_progress=mock.Mock())
        bar = SimpleNamespace(n=0, total=1000)
        bar.update = lambda increment: setattr(bar, 'n', bar.n + increment)

        with mock.patch('core.subtitle_remover.time.monotonic',
                        side_effect=[1.0, 1.01, 1.2]):
            obj.update_progress(bar, 1)
            obj.update_progress(bar, 1)
            obj.update_progress(bar, 1)

        self.assertEqual(obj.callbacks.on_progress.call_count, 2)


if __name__ == '__main__':
    unittest.main()
