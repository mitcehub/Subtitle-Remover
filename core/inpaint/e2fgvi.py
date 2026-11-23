"""E2FGVI 去字幕模型定义与推理引擎（自包含单文件，对齐 STTN 生产口径）。

整合训练仓库 official_e2fgvi 的模型定义（encoder / prior_head / SPyNet /
特征传播 / temporal focal transformer / decoder）与推理逻辑，去除训练专用
代码（Discriminator / spectral_norm / FlowCompletionLoss / 光流可视化）。

推理核心：含字原帧单趟编码（无涂黑、无 attention 屏蔽），prior_head 识别
文字仅用于输出软融合写回；窗口 = 连续 local 帧 + ref 帧按 ref_length 采样，
对齐官方 E2FGVI test.py。
"""

import os
import gc
import logging
import contextlib
from typing import List
from functools import reduce
import math

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torchvision.ops import deform_conv2d as _deform_conv2d

from infra.config import config
from infra.utils import is_frame_in_sections
from core.inpaint.region import get_inpaint_area_by_mask, get_tight_inpaint_area
from core.video_io.ffmpeg import FFmpegCLI
from core.video_io.video_reader import FramePrefetcher, create_video_capture

logger = logging.getLogger(__name__)

# E2FGVI 模型输入尺寸（与 STTN / 训练仓库 generate_data 共享常量一致）
MODEL_W = 640
MODEL_H = 120
# split_h 计算系数：split_h = int(W * SPLIT_H_NUMERATOR / SPLIT_H_DENOMINATOR)
SPLIT_H_NUMERATOR = 3
SPLIT_H_DENOMINATOR = 16
# GC 间隔：每 N 个 chunk 执行一次 gc.collect + empty_cache
GC_INTERVAL = 10
# prior 后处理（对齐训练仓库 infer_subtitle.py 生产值）：
# PRIOR_DILATE：连续概率场 max-pool 膨胀核，抬升字母空隙/汉字空腔/笔画边缘的
# 低概率区，消除软融合残留（漏字）；1=关闭。
PRIOR_DILATE = 5


# ============================================================================
# mmcv 兼容层（ModulatedDeformConv2d / ConvModule / constant_init）
# ============================================================================

class ModulatedDeformConv2d(nn.Module):
    """ModulatedDeformConv2d 兼容实现（基于 torchvision.ops.deform_conv2d）。"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, deform_groups=1, bias=True,
                 **kwargs):
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
        return modulated_deform_conv2d(
            x, offset, mask, self.weight, self.bias,
            self.stride, self.padding, self.dilation,
            self.groups, self.deform_groups)


def modulated_deform_conv2d(x, offset, mask, weight, bias=None,
                            stride=1, padding=0, dilation=1,
                            groups=1, deform_groups=1):
    """逐采样点调制；torchvision 从 weight/offset 形状推导两种 groups。"""
    return _deform_conv2d(
        x, offset, weight, bias, stride=stride, padding=padding,
        dilation=dilation, mask=mask)


class ConvModule(nn.Module):
    """Conv2d + BatchNorm2d + Activation（与 mmcv 命名兼容）。"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias='auto', act_cfg=dict(type='ReLU'),
                 norm_cfg=dict(type='BN'), **_):
        super().__init__()
        has_bias = (norm_cfg is None) if bias == 'auto' else bias
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding,
                              dilation=dilation, groups=groups,
                              bias=has_bias)
        if norm_cfg:
            if norm_cfg.get('type') == 'BN':
                self.bn = nn.BatchNorm2d(out_channels)
            else:
                self.bn = nn.Identity()
        else:
            self.bn = nn.Identity()
        if act_cfg:
            act_type = act_cfg.get('type', 'ReLU')
            if act_type == 'ReLU':
                self.activate = nn.ReLU(inplace=act_cfg.get('inplace', True))
            elif act_type == 'LeakyReLU':
                self.activate = nn.LeakyReLU(negative_slope=act_cfg.get('negative_slope', 0.1),
                                             inplace=act_cfg.get('inplace', True))
            else:
                self.activate = nn.Identity()
        else:
            self.activate = nn.Identity()

    def forward(self, x):
        return self.activate(self.bn(self.conv(x)))


def constant_init(module, val=0, bias=0):
    """常量初始化（替代 mmcv.cnn.constant_init）。"""
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


# ============================================================================
# 光流工具与 SPyNet
# ============================================================================

def flow_warp(x, flow, interpolation='bilinear', padding_mode='zeros',
              align_corners=True):
    """用光流 warp 图像/特征图。flow: [n, h, w, 2]。"""
    if x.size()[-2:] != flow.size()[1:3]:
        raise ValueError(f'The spatial sizes of input ({x.size()[-2:]}) and '
                         f'flow ({flow.size()[1:3]}) are not the same.')
    _, _, h, w = x.size()
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w), indexing='ij')
    grid = torch.stack((grid_x, grid_y), 2).type_as(x)
    grid.requires_grad = False

    grid_flow = grid + flow
    grid_flow_x = 2.0 * grid_flow[:, :, :, 0] / max(w - 1, 1) - 1.0
    grid_flow_y = 2.0 * grid_flow[:, :, :, 1] / max(h - 1, 1) - 1.0
    grid_flow = torch.stack((grid_flow_x, grid_flow_y), dim=3)
    return F.grid_sample(x, grid_flow, mode=interpolation,
                         padding_mode=padding_mode, align_corners=align_corners)


class SPyNetBasicModule(nn.Module):
    """SPyNet 基础模块。"""
    def __init__(self):
        super().__init__()
        self.basic_module = nn.Sequential(
            ConvModule(8, 32, 7, 1, 3, norm_cfg=None, act_cfg=dict(type='ReLU')),
            ConvModule(32, 64, 7, 1, 3, norm_cfg=None, act_cfg=dict(type='ReLU')),
            ConvModule(64, 32, 7, 1, 3, norm_cfg=None, act_cfg=dict(type='ReLU')),
            ConvModule(32, 16, 7, 1, 3, norm_cfg=None, act_cfg=dict(type='ReLU')),
            ConvModule(16, 2, 7, 1, 3, norm_cfg=None, act_cfg=None))

    def forward(self, tensor_input):
        return self.basic_module(tensor_input)


class SPyNet(nn.Module):
    """SPyNet 光流网络（不加载官方预训练，权重由 checkpoint 提供）。"""
    def __init__(self):
        super().__init__()
        self.basic_module = nn.ModuleList(
            [SPyNetBasicModule() for _ in range(6)])
        self.register_buffer(
            'mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer(
            'std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def compute_flow(self, ref, supp):
        n, _, h, w = ref.size()
        ref = [(ref - self.mean) / self.std]
        supp = [(supp - self.mean) / self.std]
        for _ in range(5):
            ref.append(F.avg_pool2d(ref[-1], kernel_size=2, stride=2,
                                    count_include_pad=False))
            supp.append(F.avg_pool2d(supp[-1], kernel_size=2, stride=2,
                                     count_include_pad=False))
        ref = ref[::-1]
        supp = supp[::-1]

        flow = ref[0].new_zeros(n, 2, h // 32, w // 32)
        for level in range(len(ref)):
            if level == 0:
                flow_up = flow
            else:
                flow_up = F.interpolate(flow, scale_factor=2, mode='bilinear',
                                        align_corners=True) * 2.0
            flow = flow_up + self.basic_module[level](torch.cat([
                ref[level],
                flow_warp(supp[level],
                          flow_up.permute(0, 2, 3, 1).contiguous(),
                          padding_mode='border'), flow_up
            ], 1))
        return flow

    def forward(self, ref, supp):
        h, w = ref.shape[2:4]
        w_up = w if (w % 32) == 0 else 32 * (w // 32 + 1)
        h_up = h if (h % 32) == 0 else 32 * (h // 32 + 1)
        ref = F.interpolate(ref, size=(h_up, w_up), mode='bilinear',
                            align_corners=False)
        supp = F.interpolate(supp, size=(h_up, w_up), mode='bilinear',
                             align_corners=False)
        flow = F.interpolate(self.compute_flow(ref, supp), size=(h, w),
                             mode='bilinear', align_corners=False)
        flow[:, 0, :, :] *= float(w) / float(w_up)
        flow[:, 1, :, :] *= float(h) / float(h_up)
        return flow


# ============================================================================
# 特征传播（BasicVSR++ 二阶可变形对齐）
# ============================================================================

class SecondOrderDeformableAlignment(ModulatedDeformConv2d):
    """二阶可变形对齐模块。"""
    def __init__(self, *args, **kwargs):
        self.max_residue_magnitude = kwargs.pop('max_residue_magnitude', 10)
        super().__init__(*args, **kwargs)

        self.conv_offset = nn.Sequential(
            nn.Conv2d(3 * self.out_channels + 4, self.out_channels, 3, 1, 1),
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

    def forward(self, x, extra_feat, flow_1, flow_2):
        extra_feat = torch.cat([extra_feat, flow_1, flow_2], dim=1)
        out = self.conv_offset(extra_feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)

        offset = self.max_residue_magnitude * torch.tanh(
            torch.cat((o1, o2), dim=1))
        offset_1, offset_2 = torch.chunk(offset, 2, dim=1)
        offset_1 = offset_1 + flow_1.flip(1).repeat(1, offset_1.size(1) // 2, 1, 1)
        offset_2 = offset_2 + flow_2.flip(1).repeat(1, offset_2.size(1) // 2, 1, 1)
        offset = torch.cat([offset_1, offset_2], dim=1)

        mask = torch.sigmoid(mask)
        return modulated_deform_conv2d(x, offset, mask, self.weight, self.bias,
                                       self.stride, self.padding, self.dilation,
                                       self.groups, self.deform_groups)


class BidirectionalPropagation(nn.Module):
    def __init__(self, channel):
        super().__init__()
        modules = ['backward_', 'forward_']
        self.deform_align = nn.ModuleDict()
        self.backbone = nn.ModuleDict()
        self.channel = channel

        for i, module in enumerate(modules):
            self.deform_align[module] = SecondOrderDeformableAlignment(
                2 * channel, channel, 3, padding=1, deform_groups=16)
            self.backbone[module] = nn.Sequential(
                nn.Conv2d((2 + i) * channel, channel, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Conv2d(channel, channel, 3, 1, 1),
            )

        self.fusion = nn.Conv2d(2 * channel, channel, 1, 1, 0)

    def forward(self, x, flows_backward, flows_forward):
        """x: [b, t, c, h, w]，返回 [b, t, c, h, w]。"""
        b, t, c, h, w = x.shape
        feats = {}
        feats['spatial'] = [x[:, i, :, :, :] for i in range(0, t)]

        for module_name in ['backward_', 'forward_']:
            feats[module_name] = []

            frame_idx = range(0, t)
            flow_idx = range(-1, t - 1)
            mapping_idx = list(range(0, len(feats['spatial'])))
            mapping_idx += mapping_idx[::-1]

            if 'backward' in module_name:
                frame_idx = frame_idx[::-1]
                flows = flows_backward
            else:
                flows = flows_forward

            feat_prop = x.new_zeros(b, self.channel, h, w)
            for i, idx in enumerate(frame_idx):
                feat_current = feats['spatial'][mapping_idx[idx]]

                if i > 0:
                    flow_n1 = flows[:, flow_idx[i], :, :, :]
                    cond_n1 = flow_warp(feat_prop, flow_n1.permute(0, 2, 3, 1))

                    feat_n2 = torch.zeros_like(feat_prop)
                    flow_n2 = torch.zeros_like(flow_n1)
                    cond_n2 = torch.zeros_like(cond_n1)
                    if i > 1:
                        feat_n2 = feats[module_name][-2]
                        flow_n2 = flows[:, flow_idx[i - 1], :, :, :]
                        flow_n2 = flow_n1 + flow_warp(
                            flow_n2, flow_n1.permute(0, 2, 3, 1))
                        cond_n2 = flow_warp(feat_n2, flow_n2.permute(0, 2, 3, 1))

                    cond = torch.cat([cond_n1, feat_current, cond_n2], dim=1)
                    feat_prop = torch.cat([feat_prop, feat_n2], dim=1)
                    feat_prop = self.deform_align[module_name](feat_prop, cond,
                                                               flow_n1, flow_n2)

                feat = [feat_current] + [
                    feats[k][idx]
                    for k in feats if k not in ['spatial', module_name]
                ] + [feat_prop]

                feat = torch.cat(feat, dim=1)
                feat_prop = feat_prop + self.backbone[module_name](feat)
                feats[module_name].append(feat_prop)

            if 'backward' in module_name:
                feats[module_name] = feats[module_name][::-1]

        outputs = []
        for i in range(0, t):
            align_feats = [feats[k].pop(0) for k in feats if k != 'spatial']
            align_feats = torch.cat(align_feats, dim=1)
            outputs.append(self.fusion(align_feats))

        return torch.stack(outputs, dim=1) + x


# ============================================================================
# Temporal Focal Transformer
# ============================================================================

class SoftSplit(nn.Module):
    def __init__(self, channel, hidden, kernel_size, stride, padding, t2t_param):
        super().__init__()
        self.kernel_size = kernel_size
        self.t2t = nn.Unfold(kernel_size=kernel_size, stride=stride, padding=padding)
        c_in = reduce((lambda x, y: x * y), kernel_size) * channel
        self.embedding = nn.Linear(c_in, hidden)
        self.t2t_param = t2t_param

    def forward(self, x, b, output_size):
        f_h = int((output_size[0] + 2 * self.t2t_param['padding'][0] -
                   (self.t2t_param['kernel_size'][0] - 1) - 1) /
                  self.t2t_param['stride'][0] + 1)
        f_w = int((output_size[1] + 2 * self.t2t_param['padding'][1] -
                   (self.t2t_param['kernel_size'][1] - 1) - 1) /
                  self.t2t_param['stride'][1] + 1)
        feat = self.t2t(x)
        feat = feat.permute(0, 2, 1)
        feat = self.embedding(feat)
        feat = feat.view(b, -1, f_h, f_w, feat.size(2))
        return feat


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
        feat = self.bias_conv(feat)
        return feat


class FusionFeedForward(nn.Module):
    def __init__(self, d_model, n_vecs=None, t2t_params=None):
        super().__init__()
        hd = 1960
        self.conv1 = nn.Sequential(nn.Linear(d_model, hd))
        self.conv2 = nn.Sequential(nn.GELU(), nn.Linear(hd, d_model))
        assert t2t_params is not None and n_vecs is not None
        self.t2t_params = t2t_params

    def forward(self, x, output_size):
        n_vecs = 1
        for i, d in enumerate(self.t2t_params['kernel_size']):
            n_vecs *= int((output_size[i] + 2 * self.t2t_params['padding'][i] -
                           (d - 1) - 1) / self.t2t_params['stride'][i] + 1)

        x = self.conv1(x)
        b, n, c = x.size()
        normalizer = x.new_ones(b, n, 49).view(-1, n_vecs, 49).permute(0, 2, 1)
        normalizer = F.fold(normalizer, output_size=output_size,
                            kernel_size=self.t2t_params['kernel_size'],
                            padding=self.t2t_params['padding'],
                            stride=self.t2t_params['stride'])

        x = F.fold(x.view(-1, n_vecs, c).permute(0, 2, 1),
                   output_size=output_size,
                   kernel_size=self.t2t_params['kernel_size'],
                   padding=self.t2t_params['padding'],
                   stride=self.t2t_params['stride'])

        x = F.unfold(x / normalizer, kernel_size=self.t2t_params['kernel_size'],
                     padding=self.t2t_params['padding'],
                     stride=self.t2t_params['stride']).permute(
                         0, 2, 1).contiguous().view(b, n, c)
        x = self.conv2(x)
        return x


def window_partition(x, window_size):
    B, T, H, W, C = x.shape
    x = x.view(B, T, H // window_size[0], window_size[0], W // window_size[1],
               window_size[1], C)
    windows = x.permute(0, 2, 4, 1, 3, 5, 6).contiguous().view(
        -1, T * window_size[0] * window_size[1], C)
    return windows


def window_partition_noreshape(x, window_size):
    B, T, H, W, C = x.shape
    x = x.view(B, T, H // window_size[0], window_size[0], W // window_size[1],
               window_size[1], C)
    windows = x.permute(0, 2, 4, 1, 3, 5, 6).contiguous()
    return windows


def window_reverse(windows, window_size, T, H, W):
    B = int(windows.shape[0] / (H * W / window_size[0] / window_size[1]))
    x = windows.view(B, H // window_size[0], W // window_size[1], T,
                     window_size[0], window_size[1], -1)
    x = x.permute(0, 3, 1, 4, 2, 5, 6).contiguous().view(B, T, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """Temporal focal window attention。"""
    def __init__(self, dim, expand_size, window_size, focal_window,
                 focal_level, num_heads, qkv_bias, pool_method):
        super().__init__()
        self.dim = dim
        self.expand_size = expand_size
        self.window_size = window_size
        self.pool_method = pool_method
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.focal_level = focal_level
        self.focal_window = focal_window

        if any(i > 0 for i in self.expand_size) and focal_level > 0:
            mask_tl = torch.ones(self.window_size[0], self.window_size[1])
            mask_tl[:-self.expand_size[0], :-self.expand_size[1]] = 0
            mask_tr = torch.ones(self.window_size[0], self.window_size[1])
            mask_tr[:-self.expand_size[0], self.expand_size[1]:] = 0
            mask_bl = torch.ones(self.window_size[0], self.window_size[1])
            mask_bl[self.expand_size[0]:, :-self.expand_size[1]] = 0
            mask_br = torch.ones(self.window_size[0], self.window_size[1])
            mask_br[self.expand_size[0]:, self.expand_size[1]:] = 0
            mask_rolled = torch.stack((mask_tl, mask_tr, mask_bl, mask_br),
                                      0).flatten(0)
            self.register_buffer("valid_ind_rolled",
                                 mask_rolled.nonzero(as_tuple=False).view(-1))

        if pool_method != "none" and focal_level > 1:
            self.unfolds = nn.ModuleList()
            for k in range(focal_level - 1):
                stride = 2 ** k
                kernel_size = tuple(2 * (i // 2) + 2 ** k + (2 ** k - 1)
                                    for i in self.focal_window)
                self.unfolds += [
                    nn.Unfold(kernel_size=kernel_size, stride=stride,
                              padding=tuple(i // 2 for i in kernel_size))
                ]
                if k > 0:
                    mask = torch.zeros(kernel_size)
                    mask[(2 ** k) - 1:, (2 ** k) - 1:] = 1
                    self.register_buffer(
                        "valid_ind_unfold_{}".format(k),
                        mask.flatten(0).nonzero(as_tuple=False).view(-1))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x_all, mask_all=None):
        x = x_all[0]

        B, T, nH, nW, C = x.shape
        qkv = self.qkv(x).reshape(B, T, nH, nW, 3, C).permute(4, 0, 1, 2, 3, 5).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]

        (q_windows, k_windows, v_windows) = map(
            lambda t: window_partition(t, self.window_size).view(
                -1, T, self.window_size[0] * self.window_size[1], self.num_heads,
                C // self.num_heads).permute(0, 3, 1, 2, 4).contiguous().view(
                    -1, self.num_heads, T * self.window_size[0] * self.window_size[1],
                    C // self.num_heads), (q, k, v))

        if any(i > 0 for i in self.expand_size) and self.focal_level > 0:
            (k_tl, v_tl) = map(
                lambda t: torch.roll(t, shifts=(-self.expand_size[0], -self.expand_size[1]),
                                     dims=(2, 3)), (k, v))
            (k_tr, v_tr) = map(
                lambda t: torch.roll(t, shifts=(-self.expand_size[0], self.expand_size[1]),
                                     dims=(2, 3)), (k, v))
            (k_bl, v_bl) = map(
                lambda t: torch.roll(t, shifts=(self.expand_size[0], -self.expand_size[1]),
                                     dims=(2, 3)), (k, v))
            (k_br, v_br) = map(
                lambda t: torch.roll(t, shifts=(self.expand_size[0], self.expand_size[1]),
                                     dims=(2, 3)), (k, v))

            (k_tl_windows, k_tr_windows, k_bl_windows, k_br_windows) = map(
                lambda t: window_partition(t, self.window_size).view(
                    -1, T, self.window_size[0] * self.window_size[1], self.num_heads,
                    C // self.num_heads), (k_tl, k_tr, k_bl, k_br))
            (v_tl_windows, v_tr_windows, v_bl_windows, v_br_windows) = map(
                lambda t: window_partition(t, self.window_size).view(
                    -1, T, self.window_size[0] * self.window_size[1], self.num_heads,
                    C // self.num_heads), (v_tl, v_tr, v_bl, v_br))
            k_rolled = torch.cat((k_tl_windows, k_tr_windows, k_bl_windows, k_br_windows),
                                 2).permute(0, 3, 1, 2, 4).contiguous()
            v_rolled = torch.cat((v_tl_windows, v_tr_windows, v_bl_windows, v_br_windows),
                                 2).permute(0, 3, 1, 2, 4).contiguous()

            k_rolled = k_rolled[:, :, :, self.valid_ind_rolled]
            v_rolled = v_rolled[:, :, :, self.valid_ind_rolled]
            temp_N = k_rolled.shape[3]
            k_rolled = k_rolled.view(-1, self.num_heads, T * temp_N, C // self.num_heads)
            v_rolled = v_rolled.view(-1, self.num_heads, T * temp_N, C // self.num_heads)
            k_rolled = torch.cat((k_windows, k_rolled), 2)
            v_rolled = torch.cat((v_windows, v_rolled), 2)
        else:
            k_rolled = k_windows
            v_rolled = v_windows

        if self.pool_method != "none" and self.focal_level > 1:
            k_pooled = []
            v_pooled = []
            for k in range(self.focal_level - 1):
                stride = 2 ** k
                x_window_pooled = x_all[k + 1].permute(0, 3, 1, 2, 4).contiguous()
                nWh, nWw = x_window_pooled.shape[2:4]

                mask = x_window_pooled.new(T, nWh, nWw).fill_(1)
                unfolded_mask = self.unfolds[k](mask.unsqueeze(1)).view(
                    1, T, self.unfolds[k].kernel_size[0], self.unfolds[k].kernel_size[1], -1).permute(4, 1, 2, 3, 0).contiguous().\
                    view(nWh * nWw // stride // stride, -1, 1)

                if k > 0:
                    valid_ind_unfold_k = getattr(self, "valid_ind_unfold_{}".format(k))
                    unfolded_mask = unfolded_mask[:, valid_ind_unfold_k]

                x_window_masks = unfolded_mask.flatten(1).unsqueeze(0)
                x_window_masks = x_window_masks.masked_fill(
                    x_window_masks == 0, float(-100.0)).masked_fill(
                        x_window_masks > 0, float(0.0))
                mask_all[k + 1] = x_window_masks

                qkv_pooled = self.qkv(x_window_pooled).reshape(
                    B, T, nWh, nWw, 3, C).permute(4, 0, 1, 5, 2, 3).view(
                        3, -1, C, nWh, nWw).contiguous()
                k_pooled_k, v_pooled_k = qkv_pooled[1], qkv_pooled[2]

                (k_pooled_k, v_pooled_k) = map(
                    lambda t: self.unfolds[k](t).view(
                        B, T, C, self.unfolds[k].kernel_size[0],
                        self.unfolds[k].kernel_size[1], -1)
                    .permute(0, 5, 1, 3, 4, 2).contiguous().view(
                        -1, T, self.unfolds[k].kernel_size[0] * self.unfolds[k].kernel_size[1],
                        self.num_heads, C // self.num_heads).permute(0, 3, 1, 2, 4).contiguous(),
                    (k_pooled_k, v_pooled_k))

                if k > 0:
                    (k_pooled_k, v_pooled_k) = map(
                        lambda t: t[:, :, :, valid_ind_unfold_k], (k_pooled_k, v_pooled_k))

                k_pooled_k = k_pooled_k.view(
                    -1, self.num_heads, T * self.unfolds[k].kernel_size[0] *
                    self.unfolds[k].kernel_size[1], C // self.num_heads)
                v_pooled_k = v_pooled_k.view(
                    -1, self.num_heads, T * self.unfolds[k].kernel_size[0] *
                    self.unfolds[k].kernel_size[1], C // self.num_heads)

                k_pooled += [k_pooled_k]
                v_pooled += [v_pooled_k]

            k_all = torch.cat([k_rolled] + k_pooled, 2)
            v_all = torch.cat([v_rolled] + v_pooled, 2)
        else:
            k_all = k_rolled
            v_all = v_rolled

        N = k_all.shape[-2]
        q_windows = q_windows * self.scale
        attn = (q_windows @ k_all.transpose(-2, -1))
        window_area = T * self.window_size[0] * self.window_size[1]
        window_area_rolled = k_rolled.shape[2]

        if self.pool_method != "none" and self.focal_level > 1:
            offset = window_area_rolled
            for k in range(self.focal_level - 1):
                bias = tuple((i + 2 ** k - 1) for i in self.focal_window)
                if mask_all[k + 1] is not None:
                    attn[:, :, :window_area, offset:(offset + (T * bias[0] * bias[1]))] = \
                        attn[:, :, :window_area, offset:(offset + (T * bias[0] * bias[1]))] + \
                        mask_all[k + 1][:, :, None, None, :].repeat(
                            attn.shape[0] // mask_all[k + 1].shape[1], 1, 1, 1, 1).view(
                                -1, 1, 1, mask_all[k + 1].shape[-1])
                offset += T * bias[0] * bias[1]

        if mask_all[0] is not None:
            nW = mask_all[0].shape[0]
            attn = attn.view(attn.shape[0] // nW, nW, self.num_heads, window_area, N)
            attn[:, :, :, :, :window_area] = attn[:, :, :, :, :window_area] + mask_all[0][None, :, None, :, :]
            attn = attn.view(-1, self.num_heads, window_area, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        x = (attn @ v_all).transpose(1, 2).reshape(attn.shape[0], window_area, C)
        x = self.proj(x)
        return x


class TemporalFocalTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=(5, 9), mlp_ratio=4.,
                 qkv_bias=True, pool_method="fc", focal_level=2,
                 focal_window=(5, 9), norm_layer=nn.LayerNorm, n_vecs=None,
                 t2t_params=None):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.expand_size = tuple(i // 2 for i in window_size)
        self.mlp_ratio = mlp_ratio
        self.pool_method = pool_method
        self.focal_level = focal_level
        self.focal_window = focal_window

        self.window_size_glo = self.window_size

        self.pool_layers = nn.ModuleList()
        if self.pool_method != "none":
            for k in range(self.focal_level - 1):
                window_size_glo = tuple(math.floor(i / (2 ** k)) for i in self.window_size_glo)
                self.pool_layers.append(
                    nn.Linear(window_size_glo[0] * window_size_glo[1], 1))
                self.pool_layers[-1].weight.data.fill_(
                    1. / (window_size_glo[0] * window_size_glo[1]))
                self.pool_layers[-1].bias.data.fill_(0)

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim, expand_size=self.expand_size,
                                    window_size=self.window_size,
                                    focal_window=focal_window,
                                    focal_level=focal_level, num_heads=num_heads,
                                    qkv_bias=qkv_bias, pool_method=pool_method)
        self.norm2 = norm_layer(dim)
        self.mlp = FusionFeedForward(dim, n_vecs=n_vecs, t2t_params=t2t_params)

    def forward(self, x):
        output_size = x[1]
        x = x[0]

        B, T, H, W, C = x.shape
        shortcut = x
        x = self.norm1(x)
        shifted_x = x

        x_windows_all = [shifted_x]
        x_window_masks_all = [None]

        if self.focal_level > 1 and self.pool_method != "none":
            for k in range(self.focal_level - 1):
                window_size_glo = tuple(math.floor(i / (2 ** k)) for i in self.window_size_glo)
                pooled_h = math.ceil(H / window_size_glo[0]) * (2 ** k)
                pooled_w = math.ceil(W / window_size_glo[1]) * (2 ** k)
                H_pool = pooled_h * window_size_glo[0]
                W_pool = pooled_w * window_size_glo[1]

                x_level_k = shifted_x
                if H > H_pool:
                    trim_t = (H - H_pool) // 2
                    trim_b = H - H_pool - trim_t
                    x_level_k = x_level_k[:, :, trim_t:-trim_b]
                elif H < H_pool:
                    pad_t = (H_pool - H) // 2
                    pad_b = H_pool - H - pad_t
                    x_level_k = F.pad(x_level_k, (0, 0, 0, 0, pad_t, pad_b))

                if W > W_pool:
                    trim_l = (W - W_pool) // 2
                    trim_r = W - W_pool - trim_l
                    x_level_k = x_level_k[:, :, :, trim_l:-trim_r]
                elif W < W_pool:
                    pad_l = (W_pool - W) // 2
                    pad_r = W_pool - W - pad_l
                    x_level_k = F.pad(x_level_k, (0, 0, pad_l, pad_r))

                x_windows_noreshape = window_partition_noreshape(
                    x_level_k.contiguous(), window_size_glo)
                nWh, nWw = x_windows_noreshape.shape[1:3]
                x_windows_noreshape = x_windows_noreshape.view(
                    B, nWh, nWw, T, window_size_glo[0] * window_size_glo[1],
                    C).transpose(4, 5)
                x_windows_pooled = self.pool_layers[k](
                    x_windows_noreshape).flatten(-2)
                x_windows_all += [x_windows_pooled]
                x_window_masks_all += [None]

        attn_windows = self.attn(x_windows_all, mask_all=x_window_masks_all)

        attn_windows = attn_windows.view(-1, T, self.window_size[0],
                                         self.window_size[1], C)
        shifted_x = window_reverse(attn_windows, self.window_size, T, H, W)

        x = shortcut + shifted_x
        y = self.norm2(x)
        x = x + self.mlp(y.view(B, T * H * W, C), output_size).view(B, T, H, W, C)

        return x, output_size


# ============================================================================
# 时域先验（TemporalPriorMixin）
# ============================================================================

class TemporalPriorMixin:
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
# 模型主体
# ============================================================================

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


class TextPriorHead(nn.Module):
    """v4 文字检测头：输入 H/2 64ch 特征 + RGB 下采样 + 可选 ±k 时域统计 2ch。"""
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
            rgb = F.interpolate(rgb, size=x.shape[-2:], mode='bilinear', align_corners=False)
            parts = [x, rgb]
        else:
            parts = [x]
        if temporal is not None:
            temporal = F.interpolate(temporal, size=x.shape[-2:],
                                     mode='bilinear', align_corners=False)
            parts.append(temporal)
        x = torch.cat(parts, dim=1)
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2)
        if self.context_dilation:
            x = F.leaky_relu(self.ctx_bn1(self.ctx_conv1(x)), 0.2)
            x = F.leaky_relu(self.ctx_bn2(self.ctx_conv2(x)), 0.2)
        x = self.deconv1(x)
        return x


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.group = [1, 2, 4, 8, 1]
        self.layers = nn.ModuleList([
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
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

    def forward_mid(self, x):
        """前 8 层 → mid_feat (H/4 256ch) + low_feat (H/2 64ch)。"""
        out = x
        low_feat = None
        for i in range(8):
            out = self.layers[i](out)
            if i == 3:
                low_feat = out
        return out, low_feat

    def forward(self, x, mid_feat=None):
        bt, c, _, _ = x.size()
        out = mid_feat if mid_feat is not None else x
        start = 8 if mid_feat is not None else 0
        for i in range(start, len(self.layers)):
            if i == 8:
                x0 = out
                _, _, h, w = x0.size()
            if i > 8 and i % 2 == 0:
                g = self.group[(i - 8) // 2]
                x_tmp = x0.view(bt, g, -1, h, w)
                o = out.view(bt, g, -1, h, w)
                out = torch.cat([x_tmp, o], 2).view(bt, -1, h, w)
            out = self.layers[i](out)
        return out


class deconv(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size=3, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(input_channel, output_channel,
                              kernel_size=kernel_size, stride=1, padding=padding)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        return self.conv(x)


class SubtitleRemovalGenerator(TemporalPriorMixin, BaseNetwork):
    """含字原帧单趟编码（无涂黑、无 attention 屏蔽）；prior 仅用于监督与输出融合。"""
    def __init__(self, init_weights=True, prior_temporal=False, prior_temporal_k=2,
                 context_dilation=False, prior_temporal_mode='auto'):
        super().__init__()
        channel = 256
        hidden = 512

        self.encoder = Encoder()
        self.decoder = nn.Sequential(
            deconv(128, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            deconv(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
        )
        self.feat_prop_module = BidirectionalPropagation(channel // 2)

        kernel_size = (7, 7)
        padding = (3, 3)
        stride = (3, 3)
        # 仅用于计算 n_vecs 占位；transformer block 内 FusionFeedForward 按实际
        # output_size 动态计算，不影响权重形状。实际推理输入 640×120 → 特征 160×30。
        output_size = (30, 160)
        t2t_params = {'kernel_size': kernel_size, 'stride': stride, 'padding': padding}
        self.ss = SoftSplit(channel // 2, hidden, kernel_size, stride, padding,
                            t2t_param=t2t_params)
        self.sc = SoftComp(channel // 2, hidden, kernel_size, stride, padding)

        n_vecs = 1
        for i, d in enumerate(kernel_size):
            n_vecs *= int((output_size[i] + 2 * padding[i] - (d - 1) - 1) / stride[i] + 1)

        blocks = []
        depths = 8
        for _ in range(depths):
            blocks.append(TemporalFocalTransformerBlock(
                dim=hidden, num_heads=4, window_size=(5, 9),
                focal_level=2, focal_window=(5, 9),
                n_vecs=n_vecs, t2t_params=t2t_params, pool_method="fc"))
        self.transformer = nn.Sequential(*blocks)

        self.update_spynet = SPyNet()

        self.prior_head = TextPriorHead(in_dim=64, hidden_dim=96,
                                        temporal=prior_temporal,
                                        context_dilation=context_dilation)
        self.prior_temporal = prior_temporal
        self.init_temporal_radius(prior_temporal_k, prior_temporal_mode)
        self.prior_temporal_k = prior_temporal_k
        self.prior_context = context_dilation

        if init_weights:
            self.init_weights()
            for m in self.modules():
                if isinstance(m, SecondOrderDeformableAlignment):
                    m.init_offset()

    def forward(self, frames, num_local_frames, mask=None, debug=False,
                mid_feat=None, prior_logits_pre=None):
        """frames: [B, T, 3, H, W] 含字帧 [-1, 1]；返回 (output, pred_flows, prior_logits)。"""
        if (mid_feat is None) != (prior_logits_pre is None):
            raise ValueError('mid_feat 与 prior_logits_pre 必须成对传入（或同时为 None）')
        l_t = num_local_frames
        b, t, ori_c, ori_h, ori_w = frames.size()
        assert ori_h % 4 == 0 and ori_w % 4 == 0, \
            f'输入尺寸需能被 4 整除，当前 H={ori_h} W={ori_w}'

        frames_flat = frames.view(b * t, 3, ori_h, ori_w)
        if mid_feat is not None:
            enc_mid = mid_feat
        else:
            enc_mid, enc_low = self.encoder.forward_mid(frames_flat)

        if prior_logits_pre is not None:
            prior_logits = prior_logits_pre.view(b, t, 1, ori_h, ori_w)
        else:
            temporal_maps = None
            if self.prior_temporal:
                tm = torch.zeros(b, t, 2, ori_h, ori_w,
                                 device=frames.device, dtype=frames.dtype)
                tm[:, :l_t] = self.compute_temporal_maps(
                    frames[:, :l_t]).view(b, l_t, 2, ori_h, ori_w)
                temporal_maps = tm.view(b * t, 2, ori_h, ori_w)
            prior_logits = self.prior_head(enc_low, rgb=frames_flat,
                                           temporal=temporal_maps)
            prior_logits = prior_logits.view(b, t, 1, ori_h, ori_w)

        flow_input = (frames[:, :l_t, ...] + 1) / 2
        pred_flows = self.forward_bidirect_flow(flow_input)

        enc_feat = self.encoder(frames_flat, mid_feat=enc_mid)
        _, c, h, w = enc_feat.size()
        fold_output_size = (h, w)

        local_feat = enc_feat.view(b, t, c, h, w)[:, :l_t, ...]
        ref_feat = enc_feat.view(b, t, c, h, w)[:, l_t:, ...]
        local_feat = self.feat_prop_module(local_feat, pred_flows[0], pred_flows[1])
        enc_feat = torch.cat((local_feat, ref_feat), dim=1)

        trans_feat = self.ss(enc_feat.view(-1, c, h, w), b, fold_output_size)
        trans_feat = self.transformer([trans_feat, fold_output_size])
        trans_feat = self.sc(trans_feat[0], t, fold_output_size)
        trans_feat = trans_feat.view(b, t, -1, h, w)
        enc_feat = enc_feat + trans_feat

        output = self.decoder(enc_feat.view(b * t, c, h, w))
        output = torch.tanh(output)

        return output, pred_flows, prior_logits

    def forward_bidirect_flow(self, local_frames):
        b, l_t, c, h, w = local_frames.size()
        local_frames = F.interpolate(local_frames.view(-1, c, h, w),
                                     scale_factor=1 / 4, mode='bilinear',
                                     align_corners=True, recompute_scale_factor=True)
        local_frames = local_frames.view(b, l_t, c, h // 4, w // 4)
        mlf_1 = local_frames[:, :-1, ...].reshape(-1, c, h // 4, w // 4)
        mlf_2 = local_frames[:, 1:, ...].reshape(-1, c, h // 4, w // 4)
        pred_flows_forward = self.update_spynet(mlf_1, mlf_2)
        pred_flows_backward = self.update_spynet(mlf_2, mlf_1)
        pred_flows_forward = pred_flows_forward.view(b, l_t - 1, 2, h // 4, w // 4)
        pred_flows_backward = pred_flows_backward.view(b, l_t - 1, 2, h // 4, w // 4)
        return pred_flows_forward, pred_flows_backward


# ============================================================================
# 推理引擎
# ============================================================================

class E2FGVIInpaint:
    def __init__(self, device, model_path):
        self.device = device
        self.model = SubtitleRemovalGenerator(
            prior_temporal=True, prior_temporal_k=2,
            prior_temporal_mode='auto', context_dilation=True).to(device)

        import hashlib
        _mp = model_path
        _size = os.path.getsize(_mp) if os.path.isfile(_mp) else -1
        _sha_prefix = ''
        try:
            with open(_mp, 'rb') as _fh:
                _head = _fh.read(1 << 20)
                _fh.seek(-(1 << 20), 2)
                _tail = _fh.read(1 << 20)
            _sha_prefix = hashlib.sha256(_head + _tail).hexdigest()[:16]
        except Exception:
            pass
        logger.info('e2fgvi_model_file: path=%s, size=%d, sha256=%s', _mp, _size, _sha_prefix)

        ckpt = torch.load(model_path, map_location='cpu', weights_only=True)
        state_dict = ckpt.get('netG', ckpt)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()

        self._use_bf16 = False
        if self.device.type == 'cuda':
            try:
                self._use_bf16 = torch.cuda.is_bf16_supported()
            except Exception:
                self._use_bf16 = False
        self.neighbor_stride = config.e2fgviNeighborStride.value
        self.ref_length = config.e2fgviReferenceLength.value
        logger.info('e2fgvi_model_loaded: device=%s, neighbor_stride=%d, ref_length=%d, '
                    'prior_dilate=%d, bf16=%s',
                    device, self.neighbor_stride, self.ref_length,
                    PRIOR_DILATE, self._use_bf16)

    def _get_ref_index(self, neighbor_ids, length):
        return [i for i in range(0, length, self.ref_length) if i not in neighbor_ids]

    def _inpaint(self, frames: List[np.ndarray]):
        """执行 E2FGVI 推理。

        Args:
            frames: 缩放后的帧列表 (BGR, MODEL_W×MODEL_H)
        Returns:
            (comp_outputs, prior_probs): 模型分辨率软融合结果 + 膨胀后连续 prior 概率。
        """
        frame_length = len(frames)
        rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0 for f in frames]
        chunk_t = torch.from_numpy(np.stack(rgb_frames)).float().permute(0, 3, 1, 2).unsqueeze(0).to(self.device) * 2 - 1

        comp_frames = [None] * frame_length
        weights = [0] * frame_length

        autocast_ctx = (torch.autocast('cuda', dtype=torch.bfloat16)
                        if self._use_bf16 else contextlib.nullcontext())

        with torch.no_grad():
            frames_flat = chunk_t.view(-1, 3, MODEL_H, MODEL_W)

            # 整 chunk 只算一次 prior（所有窗口共享，避免逐窗覆盖不一致）
            with autocast_ctx:
                enc_mid_c, enc_low_c = self.model.encoder.forward_mid(frames_flat)
                temporal_maps = None
                if getattr(self.model, 'prior_temporal', False):
                    temporal_maps = self.model.compute_temporal_maps(chunk_t)
                prior_logits_c = self.model.prior_head(enc_low_c, rgb=frames_flat,
                                                       temporal=temporal_maps)

            prior_prob_c = torch.sigmoid(prior_logits_c.float())
            if PRIOR_DILATE > 1:
                prior_prob_c = F.max_pool2d(prior_prob_c, PRIOR_DILATE, stride=1,
                                            padding=PRIOR_DILATE // 2)
            prior_np = prior_prob_c[:, 0].cpu().numpy()

            for f in range(0, frame_length, self.neighbor_stride):
                neighbor_ids = list(range(max(0, f - self.neighbor_stride),
                                          min(frame_length, f + self.neighbor_stride + 1)))
                l_t = len(neighbor_ids)
                all_ids = neighbor_ids + self._get_ref_index(neighbor_ids, frame_length)
                window_t = chunk_t[:, all_ids]

                with autocast_ctx:
                    pred, _, _ = self.model(
                        window_t, num_local_frames=l_t,
                        mid_feat=enc_mid_c[all_ids],
                        prior_logits_pre=prior_logits_c[all_ids])
                pred = pred.float().view(1, -1, 3, MODEL_H, MODEL_W)

                for i in range(l_t):
                    idx = neighbor_ids[i]
                    out = pred[0, i].cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5
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


class E2FGVIAutoInpaint:
    def __init__(self, device, model_path, video_path, clip_gap=None):
        self.e2fgvi_inpaint = E2FGVIInpaint(device, model_path)
        self.video_path = video_path
        self.video_out_path = os.path.join(
            os.path.dirname(os.path.abspath(video_path)),
            f"{os.path.basename(video_path).rsplit('.', 1)[0]}_no_sub.mp4"
        )
        raw_max_load = config.e2fgviMaxFrames.value
        min_chunk = config.e2fgviNeighborStride.value * config.e2fgviReferenceLength.value
        self.clip_gap = max(raw_max_load, min_chunk)
        if self.clip_gap > raw_max_load:
            logger.info('e2fgvi_max_frames: user=%d raised to %d (min_chunk=neighbor_stride*ref_length=%d*%d=%d)',
                        raw_max_load, self.clip_gap,
                        config.e2fgviNeighborStride.value, config.e2fgviReferenceLength.value, min_chunk)

    def _build_chunk_ranges(self, frame_info, effective_gap, input_sub_remover=None):
        total_len = frame_info['len']
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
                    logger.info('e2fgvi_frame_count_fallback: %d', fallback_len)

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
                _, global_mask = cv2.threshold(
                    np.zeros((frame_info['H'], frame_info['W']), dtype=np.uint8),
                    127, 1, cv2.THRESH_BINARY)
                global_mask = global_mask[:, :, None]
            else:
                _, global_mask = cv2.threshold(input_mask, 127, 1, cv2.THRESH_BINARY)
                global_mask = global_mask[:, :, None]

            global_area = get_inpaint_area_by_mask(frame_info['W'], frame_info['H'], split_h, global_mask)
            logger.info('e2fgvi_auto: using full-width crop from mask, %d areas', len(global_area))

            effective_gap = self.clip_gap
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
                    comps[k] = self.e2fgvi_inpaint._inpaint(frames[k]) if frames.get(k) else []

                if chunk_area:
                    processed_map = {}
                    idx = 0
                    for j in range(start_f, end_f):
                        if j - start_f < valid_count and is_frame_in_sections(j, ab_sections):
                            processed_map[j - start_f] = idx
                            idx += 1

                    for j in range(valid_count):
                        original = frames_hr[j].copy() if gui_mode else None
                        frame = frames_hr[j]

                        do_composite = j in processed_map
                        if do_composite and track_data:
                            frame_number = start_f + j
                            frame_tracks = [t for t in track_data if t.get("enabled", True)
                                            and t["start"] <= frame_number + 1 <= t["end"]]
                            if not frame_tracks:
                                do_composite = False

                        if do_composite:
                            ci = processed_map[j]
                            for k in range(len(chunk_area)):
                                if comps[k] and ci < len(comps[k][0]):
                                    a = chunk_area[k]
                                    area_h = a[1] - a[0]
                                    comp_preds, prior_probs = comps[k]
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
                            if original is not None and gui_mode:
                                input_sub_remover._emit_preview(original, frame)
                else:
                    for j in range(valid_count):
                        writer.write(frames_hr[j])
                        total_written += 1
                        if input_sub_remover:
                            if tbar:
                                input_sub_remover.update_progress(tbar, 1)
                            if gui_mode:
                                input_sub_remover._emit_preview(frames_hr[j], frames_hr[j])

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
            logger.error('e2fgvi_process_error: %s', e)
            raise
        finally:
            if prefetcher is not None:
                prefetcher.release()
            if writer and input_sub_remover is None:
                writer.release()
            logger.info('e2fgvi_process_end: written=%d', total_written)
