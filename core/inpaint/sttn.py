"""STTN 去字幕模型定义与推理引擎。"""

import os
import math
import time
import gc
import logging
import contextlib
from typing import List

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from infra.config import config
from infra.utils import get_model_output_stem, is_frame_in_sections
from core.inpaint.region import get_inpaint_area_by_mask, get_tight_inpaint_area
from core.video_io.ffmpeg import FFmpegCLI
from core.video_io.video_reader import FramePrefetcher, create_video_capture

logger = logging.getLogger(__name__)

# STTN 模型输入尺寸
MODEL_W = 640
MODEL_H = 120
# split_h 计算系数：split_h = int(W * SPLIT_H_RATIO / SPLIT_H_DIVISOR)
SPLIT_H_NUMERATOR = 3
SPLIT_H_DENOMINATOR = 16
# GC 间隔：每 N 个 chunk 执行一次 gc.collect + empty_cache
GC_INTERVAL = 10
# prior 后处理（对齐训练仓库 infer_subtitle.py 2026-08-23 定稿生产值）：
# PRIOR_DILATE：连续概率场 max-pool 膨胀核（5≈全分辨率 ~2px），抬升字母空隙/
# 汉字空腔/笔画边缘的低概率区，消除软融合残留（漏字）；1=关闭。
# PRIOR_THRESHOLD：历史二值化阈值。软融合下不参与出像素，仅作语义保留与日志。
PRIOR_THRESHOLD = 0.30
PRIOR_DILATE = 5


# ============================================================================
# 模型组件
# ============================================================================

def _init_weights(module, gain=0.02):
    classname = module.__class__.__name__
    if 'InstanceNorm2d' in classname:
        if hasattr(module, 'weight') and module.weight is not None:
            nn.init.constant_(module.weight.data, 1.0)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias.data, 0.0)
    elif hasattr(module, 'weight') and ('Conv' in classname or 'Linear' in classname):
        nn.init.normal_(module.weight.data, 0.0, gain)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias.data, 0.0)


class Attention(nn.Module):
    def forward(self, query, key, value, m=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        if m is not None:
            scores.masked_fill(m, torch.finfo(scores.dtype).min)
        # 对齐训练仓库：softmax 禁用 autocast 自动 FP32 promote，保持输入精度
        # （BF16 显存优化口径）；非 cuda/cpu 后端用 nullcontext 兜底
        if scores.device.type in ('cuda', 'cpu'):
            ac_off = torch.autocast(device_type=scores.device.type, enabled=False)
        else:
            ac_off = contextlib.nullcontext()
        with ac_off:
            p_attn = F.softmax(scores, dim=-1)
        return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, patch_size, d_model):
        super().__init__()
        self.patch_size = patch_size
        self.query_embedding = nn.Conv2d(d_model, d_model, kernel_size=1, padding=0)
        self.value_embedding = nn.Conv2d(d_model, d_model, kernel_size=1, padding=0)
        self.key_embedding = nn.Conv2d(d_model, d_model, kernel_size=1, padding=0)
        self.output_linear = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True))
        self.attention = Attention()

    def forward(self, x, m=None, b=1, c=256):
        bt, _, h, w = x.size()
        t = bt // b
        d_k = c // len(self.patch_size)
        output = []
        _query = self.query_embedding(x)
        _key = self.key_embedding(x)
        _value = self.value_embedding(x)
        for (width, height), query, key, value in zip(self.patch_size,
                                                       torch.chunk(_query, len(self.patch_size), dim=1),
                                                       torch.chunk(_key, len(self.patch_size), dim=1),
                                                       torch.chunk(_value, len(self.patch_size), dim=1)):
            out_w, out_h = w // width, h // height
            if m is not None:
                mm = m.view(b, t, 1, out_h, height, out_w, width)
                mm = mm.permute(0, 1, 3, 5, 2, 4, 6).contiguous().view(b, t * out_h * out_w, height * width)
                mm = (mm.mean(-1) > 0.5 + 1e-7).unsqueeze(1).expand(-1, t * out_h * out_w, -1)
            else:
                mm = None
            query = query.view(b, t, d_k, out_h, height, out_w, width).permute(0, 1, 3, 5, 2, 4, 6).contiguous().view(b, t * out_h * out_w, d_k * height * width)
            key = key.view(b, t, d_k, out_h, height, out_w, width).permute(0, 1, 3, 5, 2, 4, 6).contiguous().view(b, t * out_h * out_w, d_k * height * width)
            value = value.view(b, t, d_k, out_h, height, out_w, width).permute(0, 1, 3, 5, 2, 4, 6).contiguous().view(b, t * out_h * out_w, d_k * height * width)
            y, _ = self.attention(query, key, value, mm)
            y = y.view(b, t, out_h, out_w, d_k, height, width).permute(0, 1, 4, 2, 5, 3, 6).contiguous().view(bt, d_k, h, w)
            output.append(y)
        return self.output_linear(torch.cat(output, 1))


class FeedForward(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=2, dilation=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True))

    def forward(self, x):
        return self.conv(x)


class TransformerBlock(nn.Module):
    def __init__(self, patchsize, hidden=256):
        super().__init__()
        self.attention = MultiHeadedAttention(patchsize, d_model=hidden)
        self.feed_forward = FeedForward(hidden)

    def forward(self, x):
        x_val, b, c = x['x'], x['b'], x['c']
        m = x.get('m')
        x_val = x_val + self.attention(x_val, m, b, c)
        x_val = x_val + self.feed_forward(x_val)
        return {'x': x_val, 'm': m, 'b': b, 'c': c}


class Deconv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=1, padding=padding)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True))


class TextPriorHead(nn.Module):
    """v4 STTN prior head（2026-08-27，与训练仓库 sttn_subtitle.py 对齐）：
    - 输入 H/2 encoder 64ch 特征 + RGB 下采样 + 可选 ±k 时域统计 2ch
    - 4 层 conv(96ch)+BN+LeakyReLU(0.2)，可选上下文膨胀卷积
    - 单级 deconv 2x 上采样回全分辨率
    """

    def __init__(self, in_dim=64, hidden_dim=96, temporal=True,
                 context_dilation=True):
        super().__init__()
        self.temporal = temporal
        self.context_dilation = context_dilation
        extra = 2 if temporal else 0
        self.conv1 = nn.Conv2d(in_dim + 3 + extra, hidden_dim, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        self.conv3 = nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(hidden_dim)
        self.conv4 = nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(hidden_dim)
        if context_dilation:
            self.ctx_conv1 = nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 2, dilation=2)
            self.ctx_bn1 = nn.BatchNorm2d(hidden_dim)
            self.ctx_conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 4, dilation=4)
            self.ctx_bn2 = nn.BatchNorm2d(hidden_dim)
        self.deconv1 = Deconv(hidden_dim, 1, padding=1)

    def forward(self, x, rgb=None, temporal=None):
        if rgb is not None:
            rgb = F.interpolate(rgb, size=x.shape[-2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, rgb], dim=1)
        if temporal is not None:
            temporal = F.interpolate(temporal, size=x.shape[-2:],
                                     mode='bilinear', align_corners=False)
            x = torch.cat([x, temporal], dim=1)
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2)
        if self.context_dilation:
            x = F.leaky_relu(self.ctx_bn1(self.ctx_conv1(x)), 0.2)
            x = F.leaky_relu(self.ctx_bn2(self.ctx_conv2(x)), 0.2)
        return self.deconv1(x)


class SubtitleRemovalGenerator(nn.Module):
    def __init__(self, prior_temporal=True, context_dilation=True):
        super().__init__()
        channel = 256
        patch_size = [(80, 15), (32, 6), (10, 5), (5, 3)]
        self.prior_temporal = prior_temporal
        self.prior_context = context_dilation
        self.transformer = nn.Sequential(*[TransformerBlock(patch_size, hidden=channel) for _ in range(8)])
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 64, 3, 1, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, 3, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, channel, 3, 1, 1), nn.LeakyReLU(0.2, True),
        )
        self.decoder = nn.Sequential(
            Deconv(channel, 128, 3, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 64, 3, 1, 1), nn.LeakyReLU(0.2, True),
            Deconv(64, 64, 3, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 3, 3, 1, 1)
        )
        self.prior_head = TextPriorHead(
            in_dim=64, hidden_dim=96, temporal=prior_temporal,
            context_dilation=context_dilation)
        # 与训练仓库一致：-1 表示按当前时域窗口取半窗（legacy）；新 checkpoint
        # 会携带自己的 _temporal_radius。
        self.register_buffer('_temporal_radius', torch.tensor(-1))
        self.apply(_init_weights)

    def compute_temporal_maps(self, frames_t):
        """时域稳定性图：ch0=±k 帧灰度 std，ch1=|时域中值-当前帧|。"""
        b, t, c, h, w = frames_t.shape
        radius = int(self._temporal_radius.item())
        k = max((t - 1) // 2, 0) if radius < 0 else min(radius, max(t - 1, 0))
        if k == 0:
            return frames_t.new_zeros(b * t, 2, h, w)
        x = frames_t.mean(dim=2, keepdim=True).transpose(1, 2)
        win = F.pad(x, (0, 0, 0, 0, k, k), mode='replicate').unfold(2, 2 * k + 1, 1)
        maps = torch.cat([
            torch.tanh(win.std(dim=-1) * 5),
            torch.tanh((win.median(dim=-1).values - x).abs() * 5),
        ], dim=1)
        return maps.permute(0, 2, 1, 3, 4).contiguous().reshape(b * t, 2, h, w)

    def encode(self, frames_flat):
        """手动逐层执行 encoder（键名保持 encoder.0~7 不变）。

        Returns:
            (low_feat, enc_feat)：low_feat 为 H/2 64ch 特征（prior_head 输入），
            enc_feat 为 H/4 256ch 特征（transformer 输入，与原版一致）。
        """
        h = self.encoder[0](frames_flat)
        h = self.encoder[1](h)
        h = self.encoder[2](h)
        h = self.encoder[3](h)
        low_feat = h
        h = self.encoder[4](h)
        h = self.encoder[5](h)
        h = self.encoder[6](h)
        enc_feat = self.encoder[7](h)
        return low_feat, enc_feat


# ============================================================================
# 推理引擎
# ============================================================================

class STTNInpaint:
    def __init__(self, device, model_path):
        self.device = device
        self.model = SubtitleRemovalGenerator(
            prior_temporal=True, context_dilation=True).to(device)

        # 记录模型文件的路径、大小和 SHA-256 前缀（前后各 1MB，避免完整读取 67MB）
        import hashlib
        _mp = model_path
        _size = os.path.getsize(_mp) if os.path.isfile(_mp) else -1
        _sha_prefix = ''
        try:
            with open(_mp, 'rb') as _fh:
                _head = _fh.read(1 << 20)       # 前 1MB
                _fh.seek(-(1 << 20), 2)          # 后 1MB
                _tail = _fh.read(1 << 20)
            _sha_prefix = hashlib.sha256(_head + _tail).hexdigest()[:16]
        except Exception:
            pass
        logger.info('sttn_model_file: path=%s, size=%d, sha256=%s', _mp, _size, _sha_prefix)

        ckpt = torch.load(model_path, map_location='cpu', weights_only=True)
        state_dict = ckpt.get('netG', ckpt)
        try:
            self.model.load_state_dict(state_dict, strict=True)
        except RuntimeError as exc:
            logger.error('sttn_model_weight_mismatch: %s', exc)
            raise RuntimeError(
                'STTN 模型权重与当前 v4 架构不匹配；请使用训练仓库 '
                'gen_300000/gen_278000（含 temporal + context prior head）权重。'
            ) from exc
        self.model.eval()
        # CUDA 上启用 bfloat16 autocast（对齐训练仓库推理口径，transformer 提速）；
        # 其他后端（CPU/MPS/DirectML）保持 fp32
        self._use_bf16 = False
        if self.device.type == 'cuda':
            try:
                self._use_bf16 = torch.cuda.is_bf16_supported()
            except Exception:
                self._use_bf16 = False
        self.neighbor_stride = config.sttnNeighborStride.value
        self.ref_length = config.sttnReferenceLength.value
        logger.info('sttn_model_loaded: device=%s, neighbor_stride=%d, ref_length=%d, '
                    'prior_threshold=%.2f(reserved), prior_dilate=%d, bf16=%s',
                    device, self.neighbor_stride, self.ref_length,
                    PRIOR_THRESHOLD, PRIOR_DILATE, self._use_bf16)

    def _get_ref_index(self, neighbor_ids, length):
        return [i for i in range(0, length, self.ref_length) if i not in neighbor_ids]

    def _inpaint(self, frames: List[np.ndarray]):
        """执行 STTN 推理。

        Args:
            frames: 缩放后的帧列表 (BGR, MODEL_W×MODEL_H)
        Returns:
            (comp_outputs, prior_probs):
                comp_outputs 为模型分辨率软融合结果（decoder 输出与连续 prior
                概率加权后再做滑窗重叠平均，float32 [0,1]），
                prior_probs 为膨胀后的连续 prior 概率场（float32 [0,1]，每帧一张）。
                调用方在原始分辨率用连续概率完成二次软融合，避免非文字区重采样模糊。
        """
        frame_length = len(frames)
        rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0 for f in frames]
        chunk_t = torch.from_numpy(np.stack(rgb_frames)).float().permute(0, 3, 1, 2).unsqueeze(0).to(self.device) * 2 - 1

        comp_frames = [None] * frame_length
        weights = [0] * frame_length

        # bf16 autocast 仅在 CUDA 且硬件支持时启用，其余后端为 no-op
        autocast_ctx = (torch.autocast('cuda', dtype=torch.bfloat16)
                        if self._use_bf16 else contextlib.nullcontext())

        with torch.no_grad():
            frames_flat = chunk_t.view(-1, 3, MODEL_H, MODEL_W)

            # v4：先算 ±k 时域稳定性图（与训练 forward 同源），再走 prior_head
            with autocast_ctx:
                low_feat, enc_feat = self.model.encode(frames_flat)
                temporal_maps = None
                if getattr(self.model, 'prior_temporal', False):
                    temporal_maps = self.model.compute_temporal_maps(chunk_t)
                prior_logits = self.model.prior_head(
                    low_feat, rgb=frames_flat, temporal=temporal_maps)

            prior_prob = torch.sigmoid(prior_logits).float()
            # 概率场膨胀：max-pool 抬升空洞/边缘低概率区（字母内部空隙、汉字空腔、
            # 笔画边缘一圈），缓解软融合"检测到了但保留原图"的漏字
            if PRIOR_DILATE > 1:
                prior_prob = F.max_pool2d(prior_prob, PRIOR_DILATE, stride=1,
                                          padding=PRIOR_DILATE // 2)
            prior_np = prior_prob[:, 0].cpu().numpy()

            for f in range(0, frame_length, self.neighbor_stride):
                neighbor_ids = list(range(max(0, f - self.neighbor_stride), min(frame_length, f + self.neighbor_stride + 1)))
                l_t = len(neighbor_ids)
                all_ids = neighbor_ids + self._get_ref_index(neighbor_ids, frame_length)

                feat_win = enc_feat[all_ids]
                # transformer 无 attention mask（新模型全程看完整原图）
                with autocast_ctx:
                    feat_win = self.model.transformer({'x': feat_win, 'b': 1, 'c': enc_feat.shape[1]})['x']
                    output = torch.tanh(self.model.decoder(feat_win)).float()

                outputs_cpu = output[:l_t].detach().permute(0, 2, 3, 1).cpu().numpy()
                for i in range(l_t):
                    idx = neighbor_ids[i]
                    out = (outputs_cpu[i] * 0.5 + 0.5).astype(np.float32)
                    # 模型分辨率软融合：连续 prior 概率加权（与训练 blend 对齐），
                    # 避免硬切侵蚀细笔画边缘；融合后再参与滑窗重叠平均
                    p = prior_np[idx][..., None]
                    blended = (p * out + (1 - p) * rgb_frames[idx]).clip(0, 1).astype(np.float32)

                    if comp_frames[idx] is None:
                        comp_frames[idx] = blended
                        weights[idx] = 1
                    else:
                        comp_frames[idx] += blended
                        weights[idx] += 1

        comp_outputs = [
            (comp_frames[i] / weights[i]).clip(0, 1).astype(np.float32) if comp_frames[i] is not None
            else rgb_frames[i]
            for i in range(frame_length)
        ]
        return comp_outputs, prior_np


class STTNAutoInpaint:
    def __init__(self, device, model_path, video_path, clip_gap=None):
        self.sttn_inpaint = STTNInpaint(device, model_path)
        self.video_path = video_path
        self.video_out_path = os.path.join(
            os.path.dirname(os.path.abspath(video_path)),
            f"{get_model_output_stem(os.path.splitext(os.path.basename(video_path))[0], 'sttn-auto')}.mp4"
        )
        raw_max_load = config.sttnMaxLoadNum.value
        min_chunk = config.sttnNeighborStride.value * config.sttnReferenceLength.value
        self.clip_gap = max(raw_max_load, min_chunk)
        if self.clip_gap > raw_max_load:
            logger.info('sttn_max_load_num: user=%d raised to %d (min_chunk=neighbor_stride*ref_length=%d*%d=%d)',
                        raw_max_load, self.clip_gap,
                        config.sttnNeighborStride.value, config.sttnReferenceLength.value, min_chunk)

    def _build_chunk_ranges(self, frame_info, effective_gap, input_sub_remover=None):
        """生成 chunk 帧范围列表 [(start, end), ...]。

        按 effective_gap 等分。AB section 限制：
        只在 AB section 覆盖的帧区间内生成 chunk，
        超出范围的帧在主循环中被 skip 跳过（非 chunk 边界）。
        """
        total_len = frame_info['len']

        # 确定 AB section 范围
        ab_start, ab_end = 0, total_len
        if input_sub_remover is not None and input_sub_remover.ab_sections:
            ab_sections = input_sub_remover.ab_sections
            ab_start = max(0, ab_sections[0][0])
            ab_end = min(total_len, ab_sections[-1][1])
        proc_len = ab_end - ab_start
        if proc_len <= 0:
            return []

        total_chunks = (proc_len + effective_gap - 1) // effective_gap
        return [(ab_start + i * effective_gap,
                 min(ab_start + (i + 1) * effective_gap, ab_end))
                for i in range(total_chunks)]

    def __call__(self, input_mask=None, input_sub_remover=None, tbar=None, gui_mode=False):
        reader = None
        writer = None
        prefetcher = None
        total_written = 0
        try:
            reader = create_video_capture(self.video_path)
            prefetcher = FramePrefetcher(reader)
            frame_info = {
                'W': int(reader.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5),
                'H': int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5),
                'fps': reader.get(cv2.CAP_PROP_FPS),
                'len': int(reader.get(cv2.CAP_PROP_FRAME_COUNT) + 0.5)
            }
            if frame_info['len'] <= 0:
                fallback_len = FFmpegCLI.instance().get_frame_count(self.video_path) or 0
                if fallback_len > 0:
                    frame_info['len'] = fallback_len
                    logger.info('sttn_frame_count_fallback: %d', fallback_len)

            if input_sub_remover is not None:
                ab_sections = input_sub_remover.ab_sections
                track_data = getattr(input_sub_remover, 'track_data', None)
                writer = input_sub_remover.video_writer
            else:
                ab_sections = track_data = None
                writer = cv2.VideoWriter(self.video_out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                         frame_info['fps'], (frame_info['W'], frame_info['H']))

            split_h = int(frame_info['W'] * SPLIT_H_NUMERATOR / SPLIT_H_DENOMINATOR)
            if input_mask is None:
                # 无 mask 时用空 mask（主流程总会传入 input_mask）
                _, global_mask = cv2.threshold(
                    np.zeros((frame_info['H'], frame_info['W']), dtype=np.uint8),
                    127, 1, cv2.THRESH_BINARY)
                global_mask = global_mask[:, :, None]
            else:
                _, global_mask = cv2.threshold(input_mask, 127, 1, cv2.THRESH_BINARY)
                global_mask = global_mask[:, :, None]

            global_area = get_inpaint_area_by_mask(frame_info['W'], frame_info['H'], split_h, global_mask)
            logger.info('sttn_auto: using full-width crop from mask, %d areas', len(global_area))

            effective_gap = self.clip_gap
            # 生成 chunk 列表 [(start, end), ...]
            chunk_ranges = self._build_chunk_ranges(frame_info, effective_gap,
                                                     input_sub_remover)
            total_written = 0
            logger.info('chunk_plan: total_chunks=%d, total_frames=%d',
                        len(chunk_ranges), frame_info['len'])

            cursor = 0  # prefetcher 下一个将读出的绝对帧号
            end_of_stream = False

            def write_original_until(target_frame):
                """顺序读取并原样写出 [cursor, target_frame) 的帧。

                读取器只能顺序解码，必须显式消费掉 AB 段之前的帧，
                否则 chunk 的绝对帧号会与实际图像错位。
                """
                nonlocal cursor, total_written
                while cursor < target_frame:
                    ok, image = prefetcher.read()
                    if not ok:
                        return False
                    writer.write(image)
                    total_written += 1
                    cursor += 1
                    if input_sub_remover is not None:
                        if tbar:
                            input_sub_remover.update_progress(tbar, 1)
                        if gui_mode:
                            input_sub_remover._emit_preview(image, image)
                    elif tbar is not None:
                        tbar.update(1)
                return True

            for chunk_idx, (start_f, end_f) in enumerate(chunk_ranges):
                tqdm.write(f'Processing: {start_f + 1} - {end_f} / Total: {frame_info["len"]}')

                if not write_original_until(start_f):
                    end_of_stream = True
                    break

                chunk_area = global_area
                chunk_tight = False
                if track_data:
                    areas = [(t["ymin"], t["ymax"], t["xmin"], t["xmax"])
                             for t in track_data if t.get("enabled", True) and t["start"] <= end_f and t["end"] >= start_f + 1]
                    if areas:
                        chunk_area = get_tight_inpaint_area(areas, frame_info['W'], frame_info['H'])
                        chunk_tight = True
                    else:
                        chunk_area = []

                frames_hr, frames, comps = [], {}, {k: [] for k in range(len(chunk_area))}
                valid_count = 0

                for j in range(start_f, end_f):
                    ok, image = prefetcher.read()
                    if not ok:
                        end_of_stream = True
                        break
                    cursor += 1
                    frames_hr.append(image)
                    valid_count += 1
                    if is_frame_in_sections(j, ab_sections):
                        for k in range(len(chunk_area)):
                            a = chunk_area[k]
                            if chunk_tight:
                                frames.setdefault(k, []).append(cv2.resize(image[a[0]:a[1], a[2]:a[3], :], (MODEL_W, MODEL_H)))
                            else:
                                frames.setdefault(k, []).append(cv2.resize(image[a[0]:a[1], :], (MODEL_W, MODEL_H)))

                if valid_count == 0:
                    if end_of_stream:
                        break
                    continue

                for k in range(len(chunk_area)):
                    comps[k] = self.sttn_inpaint._inpaint(frames[k]) if frames.get(k) else []

                if chunk_area:
                    processed_map = {}
                    idx = 0
                    for j in range(start_f, end_f):
                        if j - start_f < valid_count and is_frame_in_sections(j, ab_sections):
                            processed_map[j - start_f] = idx
                            idx += 1

                    for j in range(valid_count):
                        preview_due = (
                            gui_mode and input_sub_remover is not None
                            and input_sub_remover._should_emit_preview())
                        original = frames_hr[j].copy() if preview_due else None
                        frame = frames_hr[j]

                        # 按帧判断是否在 track 时间范围内，避免 chunk 边界帧被多余处理
                        do_composite = j in processed_map
                        if do_composite and track_data:
                            frame_number = start_f + j
                            frame_tracks = [t for t in track_data if t.get("enabled", True)
                                            and t["start"] <= frame_number + 1 <= t["end"]]
                            if not frame_tracks:
                                do_composite = False  # 无轨道覆盖该帧 → 不合成

                        if do_composite:
                            ci = processed_map[j]
                            for k in range(len(chunk_area)):
                                if comps[k] and ci < len(comps[k][0]):
                                    a = chunk_area[k]
                                    area_h = a[1] - a[0]
                                    comp_preds, prior_probs = comps[k]
                                    # 高分辨率二次软融合：模型分辨率融合结果上采样后，
                                    # 与连续 prior 概率（上采样，不二值化）加权，
                                    # 文字区取修复结果，非文字区保留原始高清像素
                                    if chunk_tight:
                                        area_w = a[3] - a[2]
                                        pred_hr = cv2.resize(comp_preds[ci], (area_w, area_h), interpolation=cv2.INTER_LINEAR)
                                        prior_hr = cv2.resize(prior_probs[ci], (area_w, area_h), interpolation=cv2.INTER_LINEAR)[..., None]
                                        orig_hr = cv2.cvtColor(frame[a[0]:a[1], a[2]:a[3], :], cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                                        comp_rgb = (prior_hr * pred_hr + (1 - prior_hr) * orig_hr).clip(0, 1)
                                        comp = (comp_rgb * 255).astype(np.uint8)
                                        comp = cv2.cvtColor(comp, cv2.COLOR_RGB2BGR)
                                        frame[a[0]:a[1], a[2]:a[3], :] = comp
                                    else:
                                        pred_hr = cv2.resize(comp_preds[ci], (frame_info['W'], area_h), interpolation=cv2.INTER_LINEAR)
                                        prior_hr = cv2.resize(prior_probs[ci], (frame_info['W'], area_h), interpolation=cv2.INTER_LINEAR)[..., None]
                                        orig_hr = cv2.cvtColor(frame[a[0]:a[1], :, :], cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                                        comp_rgb = (prior_hr * pred_hr + (1 - prior_hr) * orig_hr).clip(0, 1)
                                        comp = (comp_rgb * 255).astype(np.uint8)
                                        comp = cv2.cvtColor(comp, cv2.COLOR_RGB2BGR)
                                        frame[a[0]:a[1], :, :] = comp
                        writer.write(frame)
                        total_written += 1
                        if input_sub_remover:
                            if tbar:
                                input_sub_remover.update_progress(tbar, 1)
                            if original is not None:
                                input_sub_remover._emit_reserved_preview(original, frame)
                else:
                    for j in range(valid_count):
                        writer.write(frames_hr[j])
                        total_written += 1
                        if input_sub_remover:
                            if tbar:
                                input_sub_remover.update_progress(tbar, 1)
                            if gui_mode:
                                if input_sub_remover._should_emit_preview():
                                    input_sub_remover._emit_reserved_preview(
                                        frames_hr[j], frames_hr[j])

                if end_of_stream:
                    break
                del frames_hr, frames, comps
                if chunk_idx % GC_INTERVAL == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            if not end_of_stream:
                # AB 段之后的帧原样写出；读到 EOF 为止，不依赖可能不准的帧总数
                while True:
                    ok, image = prefetcher.read()
                    if not ok:
                        break
                    writer.write(image)
                    total_written += 1
                    if input_sub_remover is not None:
                        if tbar:
                            input_sub_remover.update_progress(tbar, 1)
                        if gui_mode:
                            input_sub_remover._emit_preview(image, image)
                    elif tbar is not None:
                        tbar.update(1)
        except Exception as e:
            logger.error('sttn_process_error: %s', e)
            raise
        finally:
            if prefetcher is not None:
                prefetcher.release()
            if writer and input_sub_remover is None:
                writer.release()
            logger.info('sttn_process_end: written=%d', total_written)
