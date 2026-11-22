"""ProPainter 去字幕模型定义与推理引擎（自包含单文件，对齐 STTN/E2FGVI 生产口径）。

整合训练仓库 official_propainter 的：
  RAFT 光流（raft.py / corr.py / extractor.py / update.py / utils.py）
  ProPainter 主干（encoder 5ch / 双向传播 / sparse transformer / decoder）
  字幕双趟（prior 检测趟 + mask 条件修复趟）
以及推理流式逻辑（core/video.py stream_propainter）。

与 STTN/E2FGVI 的关键差异（必须按 ProPainter 自身特性，不能照搬）：
- **没有 chunk 概念**：流式滑窗 local=10 + ref=10，stride=5；
- ref 帧不是固定间隔采样，而是用 18×32 灰度缩略图做**全局相似度选择**；
- RAFT 光流需独立权重 raft-things.pth，在含字原帧上现算；
- encoder 输入 5 通道（RGB + mask_in + mask_updated），输入归一化到 [-1,1]；
- 输出软融合：final = p × pred + (1−p) × 原帧（p = sigmoid(prior_logits)）。
"""

import gc
import logging
import math
import os
import tempfile
from functools import reduce

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from tqdm import tqdm

from infra.config import config
from core.inpaint.region import get_inpaint_area_by_mask, get_tight_inpaint_area
from core.video_io.video_reader import create_video_capture

logger = logging.getLogger(__name__)

# ProPainter 模型输入尺寸（与 STTN / E2FGVI / 训练仓库共享常量一致）
MODEL_W = 640
MODEL_H = 120
# split_h 计算系数：split_h = int(W * SPLIT_H_NUMERATOR / SPLIT_H_DENOMINATOR)
SPLIT_H_NUMERATOR = 3
SPLIT_H_DENOMINATOR = 16
# GC 间隔：每 N 个窗口执行一次 gc.collect + empty_cache
GC_INTERVAL = 10
# 一次前向批多少个独立窗口（实测 5090 上 batching 无收益，2 持平、4 反而更慢）
WINDOW_BATCH = 1
# prior 后处理（对齐训练仓库 infer_subtitle.py 生产值）：
# PRIOR_DILATE：连续概率场 max-pool 膨胀核，抬升字母空隙/汉字空腔/笔画边缘的
# 低概率区，消除软融合残留（漏字）；1=关闭。
PRIOR_DILATE = 5
# prior 二值化阈值：ProPainter 是 mask 条件式架构，修复趟必须拿到二值 mask
MASK_THRESHOLD = 0.30
# RAFT 迭代次数（训练仓库推理出厂值）
RAFT_ITERS = 20
# RAFT 金字塔要求空间尺寸整除 64
RAFT_PAD_MULTIPLE = 64

DEFAULT_RAFT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'models', 'propainter', 'raft-things.pth')


# ============================================================================
# 基础工具
# ============================================================================

def constant_init(module, val=0, bias=0):
    """常量初始化（替代 mmcv.cnn.constant_init）。"""
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def flow_warp(x, flow, interpolation='bilinear', padding_mode='zeros',
              align_corners=True):
    """用光流 warp 图像/特征图。flow: [n, h, w, 2]。"""
    if x.size()[-2:] != flow.size()[1:3]:
        raise ValueError(f'The spatial sizes of input ({x.size()[-2:]}) and '
                         f'flow ({flow.size()[1:3]}) are not the same.')
    _, _, h, w = x.size()
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w),
                                    indexing='ij')
    grid = torch.stack((grid_x, grid_y), 2).type_as(x)
    grid.requires_grad = False

    grid_flow = grid + flow
    grid_flow_x = 2.0 * grid_flow[:, :, :, 0] / max(w - 1, 1) - 1.0
    grid_flow_y = 2.0 * grid_flow[:, :, :, 1] / max(h - 1, 1) - 1.0
    grid_flow = torch.stack((grid_flow_x, grid_flow_y), dim=3)
    return F.grid_sample(x, grid_flow, mode=interpolation,
                         padding_mode=padding_mode, align_corners=align_corners)


def length_sq(x):
    return torch.sum(torch.square(x), dim=1, keepdim=True)


def fbConsistencyCheck(flow_fw, flow_bw, alpha1=0.01, alpha2=0.5):
    """前后向光流一致性校验，返回 [b,1,h,w] 有效掩码。"""
    flow_bw_warped = flow_warp(flow_bw, flow_fw.permute(0, 2, 3, 1))  # wb(wf(x))
    flow_diff_fw = flow_fw + flow_bw_warped  # wf + wb(wf(x))
    mag_sq_fw = length_sq(flow_fw) + length_sq(flow_bw_warped)
    occ_thresh_fw = alpha1 * mag_sq_fw + alpha2
    return (length_sq(flow_diff_fw) < occ_thresh_fw).to(flow_fw)


class ModulatedDeformConv2d(nn.Module):
    """ModulatedDeformConv2d 兼容实现（基于 torchvision.ops.deform_conv2d）。"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, deform_groups=1, bias=True,
                 **_):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        self.groups = groups
        self.deform_groups = deform_groups

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            fan_in = self.weight.size(1) * self.weight[0].numel()
            bound = 1 / (fan_in ** 0.5)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, offset, mask):
        return torchvision.ops.deform_conv2d(
            x, offset, self.weight, self.bias, self.stride,
            self.padding, self.dilation, mask)


class BaseNetwork(nn.Module):
    def __init__(self):
        super().__init__()

    def init_weights(self, init_type='normal', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if classname.find('InstanceNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight.data, 1.0)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1
                                           or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'xavier_uniform':
                    nn.init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'none':
                    m.reset_parameters()
                else:
                    raise NotImplementedError(
                        'initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)
        for m in self.children():
            if hasattr(m, 'init_weights'):
                m.init_weights(init_type, gain)

    def print_network(self):
        num_params = 0
        for p in self.parameters():
            num_params += p.numel()
        logger.info('propainter_network_params: %d (%.2fM)', num_params,
                    num_params / 1e6)


class TemporalPriorMixin:
    """时域窗口语义版本化（与 STTN/E2FGVI 同款）。"""
    def init_temporal_radius(self, radius, mode='auto'):
        if mode not in ('auto', 'fixed', 'legacy') or int(radius) < 0:
            raise ValueError('prior_temporal_mode must be auto/fixed/legacy and radius >= 0')
        self._temporal_mode = mode
        self._configured_radius = int(radius)
        self._temporal_radius_value = -1 if mode == 'legacy' else int(radius)
        self.register_buffer('_temporal_radius', torch.tensor(self._temporal_radius_value))

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        key = prefix + '_temporal_radius'
        if self._temporal_mode == 'fixed':
            value = self._configured_radius
        elif self._temporal_mode == 'legacy':
            value = -1
        else:
            value = int(state_dict[key]) if key in state_dict else -1
        state_dict[key] = torch.tensor(value, dtype=torch.long)
        self._temporal_radius_value = value
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def compute_temporal_maps(self, frames_t):
        b, t, c, h, w = frames_t.shape
        k = max((t - 1) // 2, 0) if self._temporal_radius_value < 0 else min(
            self._temporal_radius_value, max(t - 1, 0))
        if k == 0:
            return frames_t.new_zeros(b * t, 2, h, w)
        x = frames_t.mean(dim=2, keepdim=True).transpose(1, 2)
        win = F.pad(x, (0, 0, 0, 0, k, k), mode='replicate').unfold(2, 2 * k + 1, 1)
        maps = torch.cat([torch.tanh(win.std(dim=-1) * 5),
                          torch.tanh((win.median(dim=-1).values - x).abs() * 5)], dim=1)
        return maps.permute(0, 2, 1, 3, 4).contiguous().reshape(b * t, 2, h, w)


# ============================================================================
# ProPainter 主干
# ============================================================================

class DeformableAlignment(ModulatedDeformConv2d):
    """二阶可变形对齐模块（max_residue_magnitude=3，对齐 ProPainter 官方）。"""
    def __init__(self, *args, **kwargs):
        self.max_residue_magnitude = kwargs.pop('max_residue_magnitude', 3)
        super().__init__(*args, **kwargs)

        self.conv_offset = nn.Sequential(
            nn.Conv2d(2 * self.out_channels + 2 + 1 + 2, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, 27 * self.deform_groups, 3, 1, 1),
        )
        self.init_offset()

    def init_offset(self):
        constant_init(self.conv_offset[-1], val=0, bias=0)

    def forward(self, x, cond_feat, flow):
        out = self.conv_offset(cond_feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = self.max_residue_magnitude * torch.tanh(torch.cat((o1, o2), dim=1))
        offset = offset + flow.flip(1).repeat(1, offset.size(1) // 2, 1, 1)
        mask = torch.sigmoid(mask)
        return torchvision.ops.deform_conv2d(
            x, offset, self.weight, self.bias, self.stride,
            self.padding, self.dilation, mask)


class BidirectionalPropagation(nn.Module):
    """双向特征传播（learnable=False 用于图像域，learnable=True 用于特征域）。"""
    def __init__(self, channel, learnable=True):
        super().__init__()
        self.deform_align = nn.ModuleDict()
        self.backbone = nn.ModuleDict()
        self.channel = channel
        self.prop_list = ['backward_1', 'forward_1']
        self.learnable = learnable

        if self.learnable:
            for module in self.prop_list:
                self.deform_align[module] = DeformableAlignment(
                    channel, channel, 3, padding=1, deform_groups=16)
                self.backbone[module] = nn.Sequential(
                    nn.Conv2d(2 * channel + 2, channel, 3, 1, 1),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    nn.Conv2d(channel, channel, 3, 1, 1),
                )
            self.fuse = nn.Sequential(
                nn.Conv2d(2 * channel + 2, channel, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(channel, channel, 3, 1, 1),
            )

    def binary_mask(self, mask, th=0.1):
        mask[mask > th] = 1
        mask[mask <= th] = 0
        return mask.to(mask)

    def forward(self, x, flows_forward, flows_backward, mask, interpolation='bilinear'):
        """x: [b,t,c,h,w] → (outputs_b, outputs_f, outputs, masks_f)。"""
        b, t, c, h, w = x.shape
        feats, masks = {}, {}
        feats['input'] = [x[:, i, :, :, :] for i in range(0, t)]
        masks['input'] = [mask[:, i, :, :, :] for i in range(0, t)]

        cache_list = ['input'] + self.prop_list

        for p_i, module_name in enumerate(self.prop_list):
            feats[module_name] = []
            masks[module_name] = []

            if 'backward' in module_name:
                frame_idx = list(range(0, t))[::-1]
                flow_idx = frame_idx
                flows_for_prop = flows_forward
                flows_for_check = flows_backward
            else:
                frame_idx = list(range(0, t))
                flow_idx = list(range(-1, t - 1))
                flows_for_prop = flows_backward
                flows_for_check = flows_forward

            for i, idx in enumerate(frame_idx):
                feat_current = feats[cache_list[p_i]][idx]
                mask_current = masks[cache_list[p_i]][idx]

                if i == 0:
                    feat_prop = feat_current
                    mask_prop = mask_current
                else:
                    flow_prop = flows_for_prop[:, flow_idx[i], :, :, :]
                    flow_check = flows_for_check[:, flow_idx[i], :, :, :]
                    flow_vaild_mask = fbConsistencyCheck(flow_prop, flow_check)
                    feat_warped = flow_warp(feat_prop, flow_prop.permute(0, 2, 3, 1),
                                            interpolation)

                    if self.learnable:
                        cond = torch.cat([feat_current, feat_warped, flow_prop,
                                          flow_vaild_mask, mask_current], dim=1)
                        feat_prop = self.deform_align[module_name](feat_prop, cond, flow_prop)
                        mask_prop = mask_current
                    else:
                        mask_prop_valid = flow_warp(mask_prop, flow_prop.permute(0, 2, 3, 1))
                        mask_prop_valid = self.binary_mask(mask_prop_valid)
                        union_vaild_mask = self.binary_mask(
                            mask_current * flow_vaild_mask * (1 - mask_prop_valid))
                        feat_prop = union_vaild_mask * feat_warped + \
                            (1 - union_vaild_mask) * feat_current
                        mask_prop = self.binary_mask(
                            mask_current * (1 - (flow_vaild_mask * (1 - mask_prop_valid))))

                if self.learnable:
                    feat = torch.cat([feat_current, feat_prop, mask_current], dim=1)
                    feat_prop = feat_prop + self.backbone[module_name](feat)

                feats[module_name].append(feat_prop)
                masks[module_name].append(mask_prop)

            if 'backward' in module_name:
                feats[module_name] = feats[module_name][::-1]
                masks[module_name] = masks[module_name][::-1]

        outputs_b = torch.stack(feats['backward_1'], dim=1).view(-1, c, h, w)
        outputs_f = torch.stack(feats['forward_1'], dim=1).view(-1, c, h, w)

        if self.learnable:
            mask_in = mask.reshape(-1, 2, h, w)
            masks_f = None
            outputs = self.fuse(torch.cat([outputs_b, outputs_f, mask_in], dim=1)) + \
                x.reshape(-1, c, h, w)
        else:
            masks_f = torch.stack(masks['forward_1'], dim=1)
            outputs = outputs_f

        return (outputs_b.view(b, -1, c, h, w), outputs_f.view(b, -1, c, h, w),
                outputs.view(b, -1, c, h, w), masks_f)


class Encoder(nn.Module):
    """5 通道编码器（RGB + mask_in + mask_updated）。"""
    def __init__(self):
        super().__init__()
        self.group = [1, 2, 4, 8, 1]
        self.layers = nn.ModuleList([
            nn.Conv2d(5, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1, groups=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(640, 512, kernel_size=3, stride=1, padding=1, groups=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(768, 384, kernel_size=3, stride=1, padding=1, groups=4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(640, 256, kernel_size=3, stride=1, padding=1, groups=8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 128, kernel_size=3, stride=1, padding=1, groups=1),
            nn.LeakyReLU(0.2, inplace=True)
        ])

    def forward(self, x):
        bt, c, _, _ = x.size()
        out = x
        for i, layer in enumerate(self.layers):
            if i == 8:
                x0 = out
                _, _, h, w = x0.size()
            if i > 8 and i % 2 == 0:
                g = self.group[(i - 8) // 2]
                xx = x0.view(bt, g, -1, h, w)
                o = out.view(bt, g, -1, h, w)
                out = torch.cat([xx, o], 2).view(bt, -1, h, w)
            out = layer(out)
        return out


class deconv(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size=3, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(input_channel, output_channel,
                              kernel_size=kernel_size, stride=1, padding=padding)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        return self.conv(x)


# ============================================================================
# 稀疏时域 Transformer
# ============================================================================

class SoftSplit(nn.Module):
    def __init__(self, channel, hidden, kernel_size, stride, padding):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.t2t = nn.Unfold(kernel_size=kernel_size, stride=stride, padding=padding)
        c_in = reduce((lambda x, y: x * y), kernel_size) * channel
        self.embedding = nn.Linear(c_in, hidden)

    def forward(self, x, b, output_size):
        f_h = int((output_size[0] + 2 * self.padding[0] -
                   (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1)
        f_w = int((output_size[1] + 2 * self.padding[1] -
                   (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1)
        feat = self.t2t(x)
        feat = feat.permute(0, 2, 1)
        feat = self.embedding(feat)
        return feat.view(b, -1, f_h, f_w, feat.size(2))


class SoftComp(nn.Module):
    def __init__(self, channel, hidden, kernel_size, stride, padding):
        super().__init__()
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        c_out = reduce((lambda x, y: x * y), kernel_size) * channel
        self.embedding = nn.Linear(hidden, c_out)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias_conv = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)

    def forward(self, x, t, output_size):
        b_, _, _, _, c_ = x.shape
        x = x.view(b_, -1, c_)
        feat = self.embedding(x)
        b, _, c = feat.size()
        feat = feat.view(b * t, -1, c).permute(0, 2, 1)
        feat = F.fold(feat, output_size=output_size, kernel_size=self.kernel_size,
                      stride=self.stride, padding=self.padding)
        return self.bias_conv(feat)


class FusionFeedForward(nn.Module):
    def __init__(self, dim, hidden_dim=1960, t2t_params=None):
        super().__init__()
        self.fc1 = nn.Sequential(nn.Linear(dim, hidden_dim))
        self.fc2 = nn.Sequential(nn.GELU(), nn.Linear(hidden_dim, dim))
        assert t2t_params is not None
        self.t2t_params = t2t_params
        self.kernel_shape = reduce((lambda x, y: x * y), t2t_params['kernel_size'])

    def forward(self, x, output_size):
        n_vecs = 1
        for i, d in enumerate(self.t2t_params['kernel_size']):
            n_vecs *= int((output_size[i] + 2 * self.t2t_params['padding'][i] -
                           (d - 1) - 1) / self.t2t_params['stride'][i] + 1)

        x = self.fc1(x)
        b, n, c = x.size()
        normalizer = x.new_ones(b, n, self.kernel_shape).view(
            -1, n_vecs, self.kernel_shape).permute(0, 2, 1)
        normalizer = F.fold(normalizer, output_size=output_size,
                            kernel_size=self.t2t_params['kernel_size'],
                            padding=self.t2t_params['padding'],
                            stride=self.t2t_params['stride'])
        x = F.fold(x.view(-1, n_vecs, c).permute(0, 2, 1), output_size=output_size,
                   kernel_size=self.t2t_params['kernel_size'],
                   padding=self.t2t_params['padding'],
                   stride=self.t2t_params['stride'])
        x = F.unfold(x / normalizer, kernel_size=self.t2t_params['kernel_size'],
                     padding=self.t2t_params['padding'],
                     stride=self.t2t_params['stride']).permute(
                         0, 2, 1).contiguous().view(b, n, c)
        return self.fc2(x)


def window_partition(x, window_size, n_head):
    B, T, H, W, C = x.shape
    x = x.view(B, T, H // window_size[0], window_size[0],
               W // window_size[1], window_size[1], n_head, C // n_head)
    return x.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()


class SparseWindowAttention(nn.Module):
    def __init__(self, dim, n_head, window_size, pool_size=(4, 4), qkv_bias=True,
                 attn_drop=0., proj_drop=0., pooling_token=True):
        super().__init__()
        assert dim % n_head == 0
        self.key = nn.Linear(dim, dim, qkv_bias)
        self.query = nn.Linear(dim, dim, qkv_bias)
        self.value = nn.Linear(dim, dim, qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.proj = nn.Linear(dim, dim)
        self.n_head = n_head
        self.window_size = window_size
        self.pooling_token = pooling_token
        if self.pooling_token:
            ks, stride = pool_size, pool_size
            self.pool_layer = nn.Conv2d(dim, dim, kernel_size=ks, stride=stride,
                                        padding=(0, 0), groups=dim)
            self.pool_layer.weight.data.fill_(1. / (pool_size[0] * pool_size[1]))
            self.pool_layer.bias.data.fill_(0)
        self.expand_size = tuple((i + 1) // 2 for i in window_size)

        if any(i > 0 for i in self.expand_size):
            mask_tl = torch.ones(self.window_size[0], self.window_size[1])
            mask_tl[:-self.expand_size[0], :-self.expand_size[1]] = 0
            mask_tr = torch.ones(self.window_size[0], self.window_size[1])
            mask_tr[:-self.expand_size[0], self.expand_size[1]:] = 0
            mask_bl = torch.ones(self.window_size[0], self.window_size[1])
            mask_bl[self.expand_size[0]:, :-self.expand_size[1]] = 0
            mask_br = torch.ones(self.window_size[0], self.window_size[1])
            mask_br[self.expand_size[0]:, self.expand_size[1]:] = 0
            masrool_k = torch.stack((mask_tl, mask_tr, mask_bl, mask_br), 0).flatten(0)
            self.register_buffer("valid_ind_rolled",
                                 masrool_k.nonzero(as_tuple=False).view(-1))

        self.max_pool = nn.MaxPool2d(window_size, window_size, (0, 0))

    def forward(self, x, mask=None, T_ind=None, attn_mask=None):
        b, t, h, w, c = x.shape
        w_h, w_w = self.window_size[0], self.window_size[1]
        c_head = c // self.n_head
        n_wh = math.ceil(h / self.window_size[0])
        n_ww = math.ceil(w / self.window_size[1])
        new_h = n_wh * self.window_size[0]
        new_w = n_ww * self.window_size[1]
        pad_r = new_w - w
        pad_b = new_h - h
        if pad_r > 0 or pad_b > 0:
            x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b, 0, 0), mode='constant', value=0)
            mask = F.pad(mask, (0, 0, 0, pad_r, 0, pad_b, 0, 0),
                         mode='constant', value=0)

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        win_q = window_partition(q.contiguous(), self.window_size, self.n_head).view(
            b, n_wh * n_ww, self.n_head, t, w_h * w_w, c_head)
        win_k = window_partition(k.contiguous(), self.window_size, self.n_head).view(
            b, n_wh * n_ww, self.n_head, t, w_h * w_w, c_head)
        win_v = window_partition(v.contiguous(), self.window_size, self.n_head).view(
            b, n_wh * n_ww, self.n_head, t, w_h * w_w, c_head)

        if any(i > 0 for i in self.expand_size):
            (k_tl, v_tl) = map(
                lambda a: torch.roll(a, shifts=(-self.expand_size[0], -self.expand_size[1]),
                                     dims=(2, 3)), (k, v))
            (k_tr, v_tr) = map(
                lambda a: torch.roll(a, shifts=(-self.expand_size[0], self.expand_size[1]),
                                     dims=(2, 3)), (k, v))
            (k_bl, v_bl) = map(
                lambda a: torch.roll(a, shifts=(self.expand_size[0], -self.expand_size[1]),
                                     dims=(2, 3)), (k, v))
            (k_br, v_br) = map(
                lambda a: torch.roll(a, shifts=(self.expand_size[0], self.expand_size[1]),
                                     dims=(2, 3)), (k, v))

            (k_tl_w, k_tr_w, k_bl_w, k_br_w) = map(
                lambda a: window_partition(a, self.window_size, self.n_head).view(
                    b, n_wh * n_ww, self.n_head, t, w_h * w_w, c_head),
                (k_tl, k_tr, k_bl, k_br))
            (v_tl_w, v_tr_w, v_bl_w, v_br_w) = map(
                lambda a: window_partition(a, self.window_size, self.n_head).view(
                    b, n_wh * n_ww, self.n_head, t, w_h * w_w, c_head),
                (v_tl, v_tr, v_bl, v_br))
            rool_k = torch.cat((k_tl_w, k_tr_w, k_bl_w, k_br_w), 4).contiguous()
            rool_v = torch.cat((v_tl_w, v_tr_w, v_bl_w, v_br_w), 4).contiguous()
            rool_k = rool_k[:, :, :, :, self.valid_ind_rolled]
            rool_v = rool_v[:, :, :, :, self.valid_ind_rolled]
            roll_N = rool_k.shape[4]
            rool_k = rool_k.view(b, n_wh * n_ww, self.n_head, t, roll_N, c // self.n_head)
            rool_v = rool_v.view(b, n_wh * n_ww, self.n_head, t, roll_N, c // self.n_head)
            win_k = torch.cat((win_k, rool_k), dim=4)
            win_v = torch.cat((win_v, rool_v), dim=4)

        if self.pooling_token:
            pool_x = self.pool_layer(x.view(b * t, new_h, new_w, c).permute(0, 3, 1, 2))
            _, _, p_h, p_w = pool_x.shape
            pool_x = pool_x.permute(0, 2, 3, 1).view(b, t, p_h, p_w, c)
            pool_k = self.key(pool_x).unsqueeze(1).repeat(1, n_wh * n_ww, 1, 1, 1, 1)
            pool_k = pool_k.view(b, n_wh * n_ww, t, p_h, p_w, self.n_head, c_head).permute(
                0, 1, 5, 2, 3, 4, 6)
            pool_k = pool_k.contiguous().view(
                b, n_wh * n_ww, self.n_head, t, p_h * p_w, c_head)
            win_k = torch.cat((win_k, pool_k), dim=4)
            pool_v = self.value(pool_x).unsqueeze(1).repeat(1, n_wh * n_ww, 1, 1, 1, 1)
            pool_v = pool_v.view(b, n_wh * n_ww, t, p_h, p_w, self.n_head, c_head).permute(
                0, 1, 5, 2, 3, 4, 6)
            pool_v = pool_v.contiguous().view(
                b, n_wh * n_ww, self.n_head, t, p_h * p_w, c_head)
            win_v = torch.cat((win_v, pool_v), dim=4)

        out = torch.zeros_like(win_q)
        l_t = mask.size(1)

        mask = self.max_pool(mask.view(b * l_t, new_h, new_w))
        mask = mask.view(b, l_t, n_wh * n_ww)
        mask = torch.sum(mask, dim=1)

        for i in range(win_q.shape[0]):
            mask_ind_i = mask[i].nonzero(as_tuple=False).view(-1)
            mask_n = len(mask_ind_i)
            if mask_n > 0:
                win_q_t = win_q[i, mask_ind_i].view(mask_n, self.n_head, t * w_h * w_w, c_head)
                win_k_t = win_k[i, mask_ind_i]
                win_v_t = win_v[i, mask_ind_i]
                if T_ind is not None:
                    win_k_t = win_k_t[:, :, T_ind.view(-1)].view(
                        mask_n, self.n_head, -1, c_head)
                    win_v_t = win_v_t[:, :, T_ind.view(-1)].view(
                        mask_n, self.n_head, -1, c_head)
                else:
                    win_k_t = win_k_t.view(n_wh * n_ww, self.n_head, t * w_h * w_w, c_head)
                    win_v_t = win_v_t.view(n_wh * n_ww, self.n_head, t * w_h * w_w, c_head)

                att_t = (win_q_t @ win_k_t.transpose(-2, -1)) * (1.0 / math.sqrt(win_q_t.size(-1)))
                att_t = F.softmax(att_t, dim=-1)
                att_t = self.attn_drop(att_t)
                y_t = att_t @ win_v_t
                out[i, mask_ind_i] = y_t.view(-1, self.n_head, t, w_h * w_w, c_head)

            unmask_ind_i = (mask[i] == 0).nonzero(as_tuple=False).view(-1)
            win_q_s = win_q[i, unmask_ind_i]
            win_k_s = win_k[i, unmask_ind_i, :, :, :w_h * w_w]
            win_v_s = win_v[i, unmask_ind_i, :, :, :w_h * w_w]

            att_s = (win_q_s @ win_k_s.transpose(-2, -1)) * (1.0 / math.sqrt(win_q_s.size(-1)))
            att_s = F.softmax(att_s, dim=-1)
            att_s = self.attn_drop(att_s)
            y_s = att_s @ win_v_s
            out[i, unmask_ind_i] = y_s

        out = out.view(b, n_wh, n_ww, self.n_head, t, w_h, w_w, c_head)
        out = out.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous().view(b, t, new_h, new_w, c)

        if pad_r > 0 or pad_b > 0:
            out = out[:, :, :h, :w, :]

        return self.proj_drop(self.proj(out))


class TemporalSparseTransformer(nn.Module):
    def __init__(self, dim, n_head, window_size, pool_size,
                 norm_layer=nn.LayerNorm, t2t_params=None):
        super().__init__()
        self.window_size = window_size
        self.attention = SparseWindowAttention(dim, n_head, window_size, pool_size)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.mlp = FusionFeedForward(dim, t2t_params=t2t_params)

    def forward(self, x, fold_x_size, mask=None, T_ind=None):
        B, T, H, W, C = x.shape
        shortcut = x
        x = self.norm1(x)
        att_x = self.attention(x, mask, T_ind)
        x = shortcut + att_x
        y = self.norm2(x)
        x = x + self.mlp(y.view(B, T * H * W, C), fold_x_size).view(B, T, H, W, C)
        return x


class TemporalSparseTransformerBlock(nn.Module):
    def __init__(self, dim, n_head, window_size, pool_size, depths, t2t_params=None):
        super().__init__()
        blocks = [TemporalSparseTransformer(dim, n_head, window_size, pool_size,
                                            t2t_params=t2t_params)
                  for _ in range(depths)]
        self.transformer = nn.Sequential(*blocks)
        self.depths = depths

    def forward(self, x, fold_x_size, l_mask=None, t_dilation=2):
        assert self.depths % t_dilation == 0, 'wrong t_dilation input.'
        T = x.size(1)
        T_ind = [torch.arange(i, T, t_dilation) for i in range(t_dilation)] * \
            (self.depths // t_dilation)
        for i in range(0, self.depths):
            x = self.transformer[i](x, fold_x_size, l_mask, T_ind[i])
        return x


# ============================================================================
# 生成器主干
# ============================================================================

class InpaintGenerator(BaseNetwork):
    def __init__(self, init_weights=True, model_path=None):
        super().__init__()
        channel = 128
        hidden = 512

        self.encoder = Encoder()

        self.decoder = nn.Sequential(
            deconv(channel, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            deconv(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1))

        kernel_size = (7, 7)
        padding = (3, 3)
        stride = (3, 3)
        t2t_params = {'kernel_size': kernel_size, 'stride': stride, 'padding': padding}
        self.ss = SoftSplit(channel, hidden, kernel_size, stride, padding)
        self.sc = SoftComp(channel, hidden, kernel_size, stride, padding)
        self.max_pool = nn.MaxPool2d(kernel_size, stride, padding)

        self.img_prop_module = BidirectionalPropagation(3, learnable=False)
        self.feat_prop_module = BidirectionalPropagation(128, learnable=True)

        self.transformers = TemporalSparseTransformerBlock(
            dim=hidden, n_head=4, window_size=(5, 9), pool_size=(4, 4),
            depths=8, t2t_params=t2t_params)

        if init_weights:
            self.init_weights()

        if model_path is not None:
            try:
                ckpt = torch.load(model_path, map_location='cpu', weights_only=True)
            except TypeError:
                ckpt = torch.load(model_path, map_location='cpu')
            self.load_state_dict(ckpt, strict=True)

        self.print_network()


class SubtitleRemovalGenerator(TemporalPriorMixin, InpaintGenerator):
    """ProPainter 主干 + 独立 RGB prior 头（字幕双趟：检测 + 修复）。"""
    def __init__(self, init_weights=True, model_path=None, prior_temporal=True,
                 prior_temporal_k=2, context_dilation=False, prior_temporal_mode='auto'):
        super().__init__(init_weights=init_weights, model_path=model_path)
        self.prior_temporal = prior_temporal
        self.init_temporal_radius(prior_temporal_k, prior_temporal_mode)
        self.prior_temporal_k = prior_temporal_k
        self.prior_context = context_dilation

        self.prior_head = TextPriorHead(in_dim=64, hidden_dim=96,
                                        temporal=prior_temporal,
                                        context_dilation=context_dilation)
        # 检测前端独立于 mask 条件式修复主干（只吃 RGB，不读 mask 条件特征）
        self.prior_rgb_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        if init_weights:
            self.init_weights()
        self.init_prior_rgb_from_repair_encoder()

    @torch.no_grad()
    def init_prior_rgb_from_repair_encoder(self):
        """用 5ch 主干的 RGB 切片初始化 prior stem（零 mask 时等价于旧前端）。"""
        src0, src2 = self.encoder.layers[0], self.encoder.layers[2]
        dst0, dst2 = self.prior_rgb_encoder[0], self.prior_rgb_encoder[2]
        dst0.weight.copy_(src0.weight[:, :3])
        dst0.bias.copy_(src0.bias)
        dst2.weight.copy_(src2.weight)
        dst2.bias.copy_(src2.bias)

    def _encoder_low_feat(self, frames_flat):
        return self.prior_rgb_encoder(frames_flat)

    def prior_pass(self, frames, num_local_frames=None):
        """prior 趟：RGB-only H/2 encoder → prior_head → [B,T,1,H,W] logits。"""
        b, t, _, ori_h, ori_w = frames.size()
        frames_flat = frames.reshape(b * t, 3, ori_h, ori_w)
        low_feat = self._encoder_low_feat(frames_flat)
        temporal = None
        if self.prior_temporal:
            if num_local_frames is not None and num_local_frames < t:
                tm = frames.new_zeros(b, t, 2, ori_h, ori_w)
                tm[:, :num_local_frames] = self.compute_temporal_maps(
                    frames[:, :num_local_frames]).view(
                        b, num_local_frames, 2, ori_h, ori_w)
                temporal = tm.view(b * t, 2, ori_h, ori_w)
            else:
                temporal = self.compute_temporal_maps(frames)
        logits = self.prior_head(low_feat, rgb=frames_flat, temporal=temporal)
        return logits.view(b, t, 1, ori_h, ori_w)

    def repair_with_mask(self, frames, flows_bi, masks, num_local_frames):
        """修复趟：mask 条件全管线，返回 pred [B,l_t,3,H,W]（[-1,1]）。"""
        l_t = num_local_frames
        b, t, _, ori_h, ori_w = frames.size()

        enc_feat = self._encoder_pass(frames.reshape(b * t, 3, ori_h, ori_w),
                                      masks.reshape(b * t, 1, ori_h, ori_w),
                                      masks.reshape(b * t, 1, ori_h, ori_w))
        _, c, h, w = enc_feat.size()
        local_feat = enc_feat.view(b, t, c, h, w)[:, :l_t, ...]
        ref_feat = enc_feat.view(b, t, c, h, w)[:, l_t:, ...]
        fold_feat_size = (h, w)

        ds_flows_f = F.interpolate(flows_bi[0].reshape(-1, 2, ori_h, ori_w),
                                   scale_factor=1 / 4, mode='bilinear',
                                   align_corners=False).view(b, l_t - 1, 2, h, w) / 4.0
        ds_flows_b = F.interpolate(flows_bi[1].reshape(-1, 2, ori_h, ori_w),
                                   scale_factor=1 / 4, mode='bilinear',
                                   align_corners=False).view(b, l_t - 1, 2, h, w) / 4.0
        if masks.dim() != 5 or masks.shape[0] != b or masks.shape[1] != t:
            raise ValueError(
                f"masks 须为 [B,T,1,H,W] 且与 frames 帧数一致, "
                f"实际 {tuple(masks.shape)} (frames b={b} t={t})")
        ds_mask_in = F.interpolate(masks.reshape(-1, 1, ori_h, ori_w),
                                   scale_factor=1 / 4, mode='nearest').view(b, t, 1, h, w)
        ds_mask_in_local = ds_mask_in[:, :l_t]
        ds_mask_updated_local = F.interpolate(
            masks[:, :l_t].reshape(-1, 1, ori_h, ori_w), scale_factor=1 / 4,
            mode='nearest').view(b, l_t, 1, h, w)

        if self.training:
            mask_pool_l = self.max_pool(ds_mask_in.view(-1, 1, h, w))
            mask_pool_l = mask_pool_l.view(b, t, 1, mask_pool_l.size(-2),
                                           mask_pool_l.size(-1))
        else:
            mask_pool_l = self.max_pool(ds_mask_in_local.reshape(-1, 1, h, w))
            mask_pool_l = mask_pool_l.view(b, l_t, 1, mask_pool_l.size(-2),
                                           mask_pool_l.size(-1))

        prop_mask_in = torch.cat([ds_mask_in_local, ds_mask_updated_local], dim=2)
        _, _, local_feat, _ = self.feat_prop_module(local_feat, ds_flows_f, ds_flows_b,
                                                    prop_mask_in)
        enc_feat = torch.cat((local_feat, ref_feat), dim=1)

        trans_feat = self.ss(enc_feat.view(-1, c, h, w), b, fold_feat_size)
        mask_pool_l = mask_pool_l.permute(0, 1, 3, 4, 2).contiguous()
        trans_feat = self.transformers(trans_feat, fold_feat_size, mask_pool_l)
        trans_feat = self.sc(trans_feat, t, fold_feat_size)
        trans_feat = trans_feat.view(b, t, -1, h, w)

        enc_feat = enc_feat + trans_feat

        if self.training:
            output = self.decoder(enc_feat.view(-1, c, h, w))
            output = torch.tanh(output).view(b, t, 3, ori_h, ori_w)
        else:
            output = self.decoder(enc_feat[:, :l_t].reshape(-1, c, h, w))
            output = torch.tanh(output).view(b, l_t, 3, ori_h, ori_w)
        return output

    def _encoder_pass(self, frames_flat, m_in, m_upd):
        return self.encoder(torch.cat([frames_flat, m_in, m_upd], dim=1))


class TextPriorHead(nn.Module):
    """v4 文字检测头：H/2 64ch 特征 + RGB 下采样 + 可选 ±k 时域统计 2ch。"""
    def __init__(self, in_dim=64, hidden_dim=96, temporal=False, context_dilation=False):
        super().__init__()
        self.temporal = temporal
        extra = 2 if temporal else 0
        self.conv1 = nn.Conv2d(in_dim + 3 + extra, hidden_dim, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        self.conv3 = nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(hidden_dim)
        self.conv4 = nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(hidden_dim)
        self.context_dilation = context_dilation
        if context_dilation:
            self.ctx_conv1 = nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 2, dilation=2)
            self.ctx_bn1 = nn.BatchNorm2d(hidden_dim)
            self.ctx_conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 4, dilation=4)
            self.ctx_bn2 = nn.BatchNorm2d(hidden_dim)
        self.deconv1 = deconv(hidden_dim, 1, padding=1)

    def forward(self, x, rgb=None, temporal=None):
        if rgb is not None:
            rgb = F.interpolate(rgb, size=x.shape[-2:], mode='bilinear',
                                align_corners=False)
            parts = [x, rgb]
        else:
            parts = [x]
        if temporal is not None:
            temporal = F.interpolate(temporal, size=x.shape[-2:], mode='bilinear',
                                     align_corners=False)
            parts.append(temporal)
        x = torch.cat(parts, dim=1)
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2)
        if self.context_dilation:
            x = F.leaky_relu(self.ctx_bn1(self.ctx_conv1(x)), 0.2)
            x = F.leaky_relu(self.ctx_bn2(self.ctx_conv2(x)), 0.2)
        return self.deconv1(x)


# ============================================================================
# RAFT 光流（weights/raft-things.pth，small=False）
# ============================================================================

class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.BatchNorm2d(planes)
        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.InstanceNorm2d(planes)
        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not stride == 1:
                self.norm3 = nn.Sequential()

        if stride == 1:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))
        if self.downsample is not None:
            x = self.downsample(x)
        return self.relu(x + y)


class BasicEncoder(nn.Module):
    def __init__(self, output_dim=128, norm_fn='batch', dropout=0.0):
        super().__init__()
        self.norm_fn = norm_fn
        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)
        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(64)
        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(64)
        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 64
        self.layer1 = self._make_layer(64, stride=1)
        self.layer2 = self._make_layer(96, stride=2)
        self.layer3 = self._make_layer(128, stride=2)
        self.conv2 = nn.Conv2d(128, output_dim, kernel_size=1)

        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)
        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.conv2(x)

        if self.training and self.dropout is not None:
            x = self.dropout(x)
        if is_list:
            x = torch.split(x, [batch_dim, batch_dim], dim=0)
        return x


class FlowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super().__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192 + 128):
        super().__init__()
        self.convz1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))
        self.convr1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))
        self.convq1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))
        self.convz2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))
        self.convr2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))
        self.convq2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q

        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r * h, x], dim=1)))
        return (1 - z) * h + z * q


class BasicMotionEncoder(nn.Module):
    def __init__(self, corr_levels=4, corr_radius=4):
        super().__init__()
        cor_planes = corr_levels * (2 * corr_radius + 1) ** 2
        self.convc1 = nn.Conv2d(cor_planes, 256, 1, padding=0)
        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)
        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv = nn.Conv2d(64 + 192, 128 - 2, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))
        out = F.relu(self.conv(torch.cat([cor, flo], dim=1)))
        return torch.cat([out, flow], dim=1)


class BasicUpdateBlock(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=128):
        super().__init__()
        self.encoder = BasicMotionEncoder()
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=128 + hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)
        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64 * 9, 1, padding=0))

    def forward(self, net, inp, corr, flow, upsample=True):
        motion_features = self.encoder(flow, corr)
        inp = torch.cat([inp, motion_features], dim=1)
        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)
        mask = .25 * self.mask(net)
        return net, mask, delta_flow


def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1, 1], dim=-1)
    xgrid = 2 * xgrid / (W - 1) - 1
    ygrid = 2 * ygrid / (H - 1) - 1
    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)
    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()
    return img


def coords_grid(batch, ht, wd):
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd), indexing='ij')
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


def upflow8(flow, mode='bilinear'):
    new_size = (8 * flow.shape[2], 8 * flow.shape[3])
    return 8 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)


class CorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []
        corr = CorrBlock.corr(fmap1, fmap2)
        batch, h1, w1, dim, h2, w2 = corr.shape
        corr = corr.reshape(batch * h1 * w1, dim, h2, w2)
        self.corr_pyramid.append(corr)
        for _ in range(self.num_levels - 1):
            corr = F.avg_pool2d(corr, 2, stride=2)
            self.corr_pyramid.append(corr)

    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape
        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            dx = torch.linspace(-r, r, 2 * r + 1)
            dy = torch.linspace(-r, r, 2 * r + 1)
            delta = torch.stack(torch.meshgrid(dy, dx, indexing='ij'), dim=-1).to(
                coords.device)
            centroid_lvl = coords.reshape(batch * h1 * w1, 1, 1, 2) / 2 ** i
            delta_lvl = delta.view(1, 2 * r + 1, 2 * r + 1, 2)
            coords_lvl = centroid_lvl + delta_lvl
            corr = bilinear_sampler(corr, coords_lvl)
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)
        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def corr(fmap1, fmap2):
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht * wd)
        fmap2 = fmap2.view(batch, dim, ht * wd)
        corr = torch.matmul(fmap1.transpose(1, 2), fmap2)
        corr = corr.view(batch, ht, wd, 1, ht, wd)
        return corr / math.sqrt(dim)


class RAFT(nn.Module):
    """RAFT（small=False，corr_levels=4，corr_radius=4）。"""
    def __init__(self, corr_levels=4, corr_radius=4):
        super().__init__()
        self.hidden_dim = 128
        self.context_dim = 128
        self.corr_levels = corr_levels
        self.corr_radius = corr_radius
        self.fnet = BasicEncoder(output_dim=256, norm_fn='instance')
        self.cnet = BasicEncoder(output_dim=self.hidden_dim + self.context_dim,
                                 norm_fn='batch')
        self.update_block = BasicUpdateBlock(hidden_dim=self.hidden_dim)

    def initialize_flow(self, img):
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H // 8, W // 8).to(img.device)
        coords1 = coords_grid(N, H // 8, W // 8).to(img.device)
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)
        up_flow = F.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)
        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8 * H, 8 * W)

    def forward(self, image1, image2, iters=12, flow_init=None, test_mode=True):
        image1 = image1.contiguous()
        image2 = image2.contiguous()
        hdim = self.hidden_dim
        cdim = self.context_dim

        fmap1, fmap2 = self.fnet([image1, image2])
        fmap1 = fmap1.float()
        fmap2 = fmap2.float()

        corr_fn = CorrBlock(fmap1, fmap2, radius=self.corr_radius)

        cnet = self.cnet(image1)
        net, inp = torch.split(cnet, [hdim, cdim], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(image1)
        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_up = None
        for _ in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1)
            flow = coords1 - coords0
            net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)
            coords1 = coords1 + delta_flow
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)

        if test_mode:
            return coords1 - coords0, flow_up
        return flow_up


def raft_bi_padded(raft, frames01, iters=RAFT_ITERS, multiple=RAFT_PAD_MULTIPLE):
    """RAFT 双向光流；空间尺寸不足 multiple 时 replicate 补边，算完裁回。

    frames01: [b,t,3,h,w]，取值 [0,1]。返回 (flows_f, flows_b) [b,t-1,2,h,w]。
    """
    b, t, c, h, w = frames01.shape
    if t < 2:
        empty = frames01.new_zeros(b, 0, 2, h, w)
        return empty, empty.clone()

    ph = (multiple - h % multiple) % multiple
    pw = (multiple - w % multiple) % multiple
    x = frames01
    if ph or pw:
        x = F.pad(frames01, (0, pw, 0, ph, 0, 0), mode='replicate')

    with torch.no_grad():
        x1 = x[:, :-1].reshape(-1, c, x.shape[-2], x.shape[-1])
        x2 = x[:, 1:].reshape(-1, c, x.shape[-2], x.shape[-1])
        _, f_f = raft(x1, x2, iters=iters, test_mode=True)
        _, f_b = raft(x2, x1, iters=iters, test_mode=True)

    if ph or pw:
        f_f = f_f[..., :h, :w]
        f_b = f_b[..., :h, :w]
    return (f_f.view(b, t - 1, 2, h, w), f_b.view(b, t - 1, 2, h, w))


# ============================================================================
# 部署层栈（与 STTN/E2FGVI 同款）
# ============================================================================

def _fp_speckle_filter(prob, peak_min=0.45):
    """连通域散斑过滤：仅保留含峰值 ≥ peak_min 核的成分，孤立背景雾斑置零。"""
    m = (prob > 0.02).astype(np.uint8)
    if m.max() == 0:
        return prob
    core = prob >= peak_min
    if not core.any():
        return prob
    n_f, lab_f = cv2.connectedComponents(m, connectivity=8)
    keep = np.zeros(n_f, dtype=bool)
    keep[np.unique(lab_f[core])] = True
    keep[0] = False
    if keep.all():
        return prob
    return prob * keep[lab_f].astype(np.float32)


def deploy_stack(prob, prior_dilate, fp_filter):
    """部署栈：膨胀(max-pool) → 散斑过滤。返回处理后的概率场 [H,W]。"""
    if prior_dilate > 1:
        t = torch.from_numpy(prob)[None, None]
        prob = F.max_pool2d(t, prior_dilate, stride=1,
                            padding=prior_dilate // 2)[0, 0].numpy()
    if fp_filter:
        prob = _fp_speckle_filter(prob)
    return prob


def deploy_stack_batch(probs, prior_dilate, fp_filter):
    """批量版部署栈，probs [B,T,H,W]；膨胀一次性完成。"""
    if prior_dilate > 1:
        h, w = probs.shape[-2:]
        t = torch.from_numpy(probs).reshape(-1, 1, h, w)
        probs = F.max_pool2d(t, prior_dilate, stride=1,
                             padding=prior_dilate // 2).reshape(probs.shape).numpy()
    if fp_filter:
        h, w = probs.shape[-2:]
        probs = np.stack([_fp_speckle_filter(p) for p in probs.reshape(-1, h, w)]
                         ).reshape(probs.shape)
    return probs


# ============================================================================
# 推理引擎
# ============================================================================

class ProPainterInpaint:
    """模型封装：加载主模型 + RAFT，单窗口推理。"""

    def __init__(self, device, model_path, raft_path=None):
        self.device = device
        raft_path = raft_path or DEFAULT_RAFT_PATH

        state = torch.load(model_path, map_location='cpu', weights_only=True)
        if 'netG' in state:
            state = state['netG']
        state = {(k[len('module.'):] if k.startswith('module.') else k): v
                 for k, v in state.items()}

        pk = state.get('prior_head.conv1.weight')
        prior_temporal = bool(pk is not None and pk.shape[1] >= 69)
        context_dilation = any(k.startswith('prior_head.ctx_conv') for k in state)

        self.model = SubtitleRemovalGenerator(
            init_weights=False,
            prior_temporal=prior_temporal,
            prior_temporal_k=2,
            context_dilation=context_dilation,
            prior_temporal_mode='auto').to(device)
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing:
            logger.warning('propainter_ckpt_missing_keys(%d): %s', len(missing), missing[:10])
        if unexpected:
            logger.warning('propainter_ckpt_unexpected_keys(%d): %s',
                           len(unexpected), unexpected[:10])
        self.model.eval()

        # 仅反序列化张量/基础类型，避免恶意权重文件通过 pickle 执行任意代码（CWE-502）
        try:
            raft_state = torch.load(raft_path, map_location='cpu', weights_only=True)
        except TypeError:
            # 兼容不支持 weights_only 参数的旧版 PyTorch
            raft_state = torch.load(raft_path, map_location='cpu')
        if isinstance(raft_state, dict) and 'state_dict' in raft_state:
            raft_state = raft_state['state_dict']
        raft_state = {(k[len('module.'):] if k.startswith('module.') else k): v
                      for k, v in raft_state.items()}
        self.raft = RAFT()
        self.raft.load_state_dict(raft_state)
        self.raft.to(device).eval()

        self.local_frames = config.propainterLocalFrames.value
        self.ref_frames = config.propainterReferenceFrames.value
        self.stride = config.propainterStride.value
        logger.info('propainter_model_loaded: device=%s, prior_temporal=%s, '
                    'context_dilation=%s, local=%d, ref=%d, stride=%d',
                    device, prior_temporal, context_dilation,
                    self.local_frames, self.ref_frames, self.stride)

    def inpaint_window(self, win, num_local):
        """批次窗口推理：win [B,T,3,H,W] ∈ [-1,1]，num_local = local 帧数。

        返回 (pred [B,l_t,3,H,W] ∈ [-1,1], deployed [B,l_t,H,W])。
        B 个窗口互相独立，语义与逐窗口推理完全一致，只是把 GPU 喂饱。
        """
        flows = raft_bi_padded(self.raft, (win[:, :num_local] + 1) / 2, iters=RAFT_ITERS)
        probability = torch.sigmoid(self.model.prior_pass(win, num_local))
        probs = probability[:, :, 0].float().cpu().numpy()
        deployed = deploy_stack_batch(probs, PRIOR_DILATE, 1)
        mask = torch.from_numpy(deployed)[:, :, None].to(win.device) > MASK_THRESHOLD
        pred = self.model.repair_with_mask(win, flows, mask.float(), num_local)
        return pred, deployed[:, :num_local]


def _normalize_sections(ab_sections, total):
    if not ab_sections:
        return [(0, total)]
    secs = []
    for s in ab_sections:
        if isinstance(s, tuple):
            a, b = int(s[0]), int(s[1])
        else:
            a, b = int(s.start), int(s.stop)
        a = max(0, a)
        b = min(total, b)
        if b > a:
            secs.append((a, b))
    if not secs:
        return [(0, total)]
    secs.sort()
    merged = [list(secs[0])]
    for a, b in secs[1:]:
        if a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged]


def _build_starts(sections, local, stride):
    starts = []
    for a, b in sections:
        span = b - a
        if span <= local:
            starts.append(a)
            continue
        s = list(range(a, b - local + 1, stride))
        if s[-1] + local < b:
            s.append(b - local)
        starts.extend(s)
    return starts


class ProPainterAutoInpaint:
    """生产端入口：对齐 STTNAutoInpaint / E2FGVIAutoInpaint 的 __call__ 契约。"""

    def __init__(self, device, model_path, video_path, raft_path=None):
        self.inpaint = ProPainterInpaint(device, model_path, raft_path)
        self.video_path = video_path
        self.video_out_path = os.path.join(
            os.path.dirname(os.path.abspath(video_path)),
            f"{os.path.basename(video_path).rsplit('.', 1)[0]}_no_sub.mp4")

    # ---------------------------------------------------------------- 工具
    @staticmethod
    def _resolve_crop_box(input_mask, sub_areas, W, H):
        """推导单一裁剪框 (x1,y1,x2,y2)：优先选区合并，其次 mask 条带。"""
        if sub_areas:
            boxes = [(int(a[0]), int(a[1]), int(a[2]), int(a[3])) for a in sub_areas]
            ymin = max(0, min(b[0] for b in boxes))
            ymax = min(H, max(b[1] for b in boxes))
            xmin = max(0, min(b[2] for b in boxes))
            xmax = min(W, max(b[3] for b in boxes))
            areas = get_tight_inpaint_area([(ymin, ymax, xmin, xmax)], W, H)
            if areas:
                y1, y2, x1, x2 = areas[0]
                if x2 > x1 and y2 > y1:
                    return x1, y1, x2, y2
        if input_mask is not None:
            split_h = int(W * SPLIT_H_NUMERATOR / SPLIT_H_DENOMINATOR)
            areas = get_inpaint_area_by_mask(W, H, split_h, input_mask)
            if areas:
                y1 = min(a[0] for a in areas)
                y2 = max(a[1] for a in areas)
                if y2 > y1:
                    return 0, y1, W, y2
        return 0, 0, W, H

    def _build_index_and_refs(self, crop_box, total_hint):
        """一趟顺序解码，同时产出：18×32 灰度索引 + 全片 ref 帧缓存文件。

        ProPainter 的 ref 帧是全局相似度挑选的，若运行时逐窗口随机 seek 解码，
        长视频会慢一个数量级。这里把每帧裁剪图（模型分辨率 RGB）顺序写进临时
        文件，推理时按下标直接读，彻底消除随机 seek。
        """
        x1, y1, x2, y2 = crop_box
        ref_size = (MODEL_W, MODEL_H)
        budget = 8 * 1024 ** 3
        if total_hint and total_hint * MODEL_W * MODEL_H * 3 > budget:
            ref_size = (MODEL_W // 2, MODEL_H // 2)
            logger.info('propainter_ref_cache_half_res: total=%d', total_hint)

        cap = create_video_capture(self.video_path)
        fd, ref_path = tempfile.mkstemp(prefix='pp_refs_', suffix='.dat')
        os.close(fd)
        thumbs = []
        try:
            with open(ref_path, 'wb') as fh:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    rgb = cv2.cvtColor(
                        cv2.resize(frame[y1:y2, x1:x2], ref_size,
                                   interpolation=cv2.INTER_LINEAR),
                        cv2.COLOR_BGR2RGB)
                    fh.write(rgb.tobytes())
                    small = rgb if ref_size == (MODEL_W, MODEL_H) else cv2.resize(
                        rgb, (MODEL_W, MODEL_H), interpolation=cv2.INTER_LINEAR)
                    thumbs.append(cv2.resize(cv2.cvtColor(small, cv2.COLOR_RGB2GRAY),
                                             (32, 18), interpolation=cv2.INTER_AREA))
        finally:
            cap.release()
        if not thumbs:
            raise ValueError('视频没有可解码帧')
        total = len(thumbs)
        ref_mm = np.memmap(ref_path, dtype=np.uint8, mode='r',
                           shape=(total, ref_size[1], ref_size[0], 3))
        return np.stack(thumbs), ref_mm, ref_path, ref_size

    # ---------------------------------------------------------------- 主流程
    def __call__(self, input_mask=None, input_sub_remover=None, tbar=None, gui_mode=False):
        cap = reader = writer = None
        ref_mm = ref_path = None
        written = 0
        try:
            reader = create_video_capture(self.video_path)
            W = int(reader.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
            H = int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
            fps = reader.get(cv2.CAP_PROP_FPS) or 30.0
            total_hint = int(reader.get(cv2.CAP_PROP_FRAME_COUNT) + 0.5)
            reader.release()
            reader = None

            if input_sub_remover is not None:
                ab_sections = input_sub_remover.ab_sections
                sub_areas = getattr(input_sub_remover, 'sub_areas', None)
                writer = input_sub_remover.video_writer
            else:
                ab_sections, sub_areas = None, None
                writer = cv2.VideoWriter(self.video_out_path,
                                         cv2.VideoWriter_fourcc(*'mp4v'),
                                         fps, (W, H))

            crop_box = self._resolve_crop_box(input_mask, sub_areas, W, H)
            x1, y1, x2, y2 = crop_box
            crop_w, crop_h = x2 - x1, y2 - y1
            logger.info('propainter_auto: crop=(%d,%d)-(%d,%d), size=%dx%d',
                        x1, y1, x2, y2, crop_w, crop_h)

            index, ref_mm, ref_path, ref_size = self._build_index_and_refs(
                crop_box, total_hint)
            total = len(index)
            sections = _normalize_sections(ab_sections, total)
            local = min(self.inpaint.local_frames, total)
            ref_frames = self.inpaint.ref_frames
            stride = max(1, min(self.inpaint.stride, local))
            starts = _build_starts(sections, local, stride)
            logger.info('propainter_auto: total=%d, sections=%d, local=%d, ref=%d, '
                        'stride=%d, windows=%d', total, len(sections), local,
                        ref_frames, stride, len(starts))

            # 第二趟：顺序解码，每帧只解一次；本地帧直接复用，ref 帧读缓存文件
            cap = create_video_capture(self.video_path)
            pending = {}
            read_pos = 0

            def fill(upto):
                nonlocal read_pos
                while read_pos <= upto:
                    ok, fr = cap.read()
                    if not ok:
                        break
                    pending[read_pos] = fr
                    read_pos += 1

            sums, weights = {}, {}
            write_index = 0

            def emit(frame_idx):
                nonlocal written
                frame = pending.pop(frame_idx, None)
                if frame is None:
                    raise RuntimeError(f'输出帧 {frame_idx} 解码失败')
                original = frame.copy() if gui_mode else None
                if frame_idx in sums:
                    crop_out = (sums.pop(frame_idx) / weights.pop(frame_idx)
                                * 255).clip(0, 255).astype(np.uint8)
                    frame[y1:y2, x1:x2] = cv2.cvtColor(crop_out, cv2.COLOR_RGB2BGR)
                writer.write(frame)
                written += 1
                if input_sub_remover is not None:
                    if tbar:
                        input_sub_remover.update_progress(tbar, 1)
                    if gui_mode and original is not None:
                        input_sub_remover._emit_preview(original, frame)

            wi = 0
            n_starts = len(starts)
            while wi < n_starts:
                # 把连续的若干独立窗口凑成一批（同 l_t 才能拼 batch）
                l_t = min(local, total - starts[wi])
                group = []
                while wi < n_starts and len(group) < WINDOW_BATCH:
                    start = starts[wi]
                    if min(local, total - start) != l_t:
                        break
                    group.append(start)
                    wi += 1

                # 批次第一个窗口之前（含区块间隙）的帧按原样顺序写出
                while write_index < group[0]:
                    fill(write_index)
                    emit(write_index)
                    write_index += 1

                fill(group[-1] + l_t - 1)

                wins, metas = [], []
                for start in group:
                    local_ids = list(range(start, start + l_t))

                    anchor = index[start + l_t // 2].astype(np.float32)
                    scores = np.abs(index.astype(np.float32) - anchor).mean(axis=(1, 2))
                    scores[local_ids] = np.inf
                    count = min(ref_frames, total - l_t)
                    refs = (sorted(np.argsort(scores, kind='stable')[:count].tolist())
                            if count > 0 else [])

                    crops = [cv2.resize(cv2.cvtColor(pending[i][y1:y2, x1:x2],
                                                    cv2.COLOR_BGR2RGB),
                                        (MODEL_W, MODEL_H),
                                        interpolation=cv2.INTER_LINEAR)
                             for i in local_ids]
                    for i in refs:
                        img = np.asarray(ref_mm[i])
                        if ref_size != (MODEL_W, MODEL_H):
                            img = cv2.resize(img, (MODEL_W, MODEL_H),
                                             interpolation=cv2.INTER_LINEAR)
                        crops.append(img)
                    arr = np.stack(crops).astype(np.float32) / 127.5 - 1.0
                    wins.append(torch.from_numpy(arr).permute(0, 3, 1, 2))
                    metas.append(local_ids)

                win = torch.stack(wins, 0).to(self.inpaint.device)

                with torch.no_grad():
                    pred, deployed = self.inpaint.inpaint_window(win, l_t)

                predicted = (pred.float().cpu().numpy().transpose(0, 1, 3, 4, 2) + 1) / 2
                if not np.isfinite(predicted).all():
                    raise RuntimeError('推理结果包含 NaN/Inf')

                for bi, local_ids in enumerate(metas):
                    for k, frame_id in enumerate(local_ids):
                        repaired = cv2.resize(predicted[bi, k], (crop_w, crop_h),
                                              interpolation=cv2.INTER_LINEAR)
                        prob = cv2.resize(deployed[bi, k], (crop_w, crop_h),
                                          interpolation=cv2.INTER_LINEAR)[..., None]
                        original = cv2.cvtColor(pending[frame_id][y1:y2, x1:x2],
                                                cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                        blended = prob * repaired + (1 - prob) * original
                        if frame_id in sums:
                            sums[frame_id] += blended
                            weights[frame_id] += 1
                        else:
                            sums[frame_id] = blended
                            weights[frame_id] = 1

                del win, pred

                flush_until = starts[wi] if wi < n_starts else total
                while write_index < flush_until:
                    fill(write_index)
                    emit(write_index)
                    write_index += 1

                if wi % 20 == 0 or wi >= n_starts:
                    tqdm.write(f'Processing: {group[0] + 1} - '
                               f'{group[-1] + l_t} / Total: {total} '
                               f'(stride {stride})')

                if (wi // max(WINDOW_BATCH, 1)) % GC_INTERVAL == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            while write_index < total:
                fill(write_index)
                emit(write_index)
                write_index += 1

        except Exception as e:
            logger.error('propainter_process_error: %s', e)
            raise
        finally:
            if cap is not None:
                cap.release()
            if ref_mm is not None:
                try:
                    ref_mm._mmap.close()
                except Exception:
                    pass
                del ref_mm
            if ref_path and os.path.exists(ref_path):
                try:
                    os.remove(ref_path)
                except OSError:
                    pass
            if reader is not None:
                reader.release()
            if writer is not None and input_sub_remover is None:
                writer.release()
            logger.info('propainter_process_end: written=%d', written)
