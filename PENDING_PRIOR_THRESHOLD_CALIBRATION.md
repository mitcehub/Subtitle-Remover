# 待办：prior_threshold 新模型校准

> 创建时间：2026-08-08
> 状态：**已关闭（2026-08-24，随 450K 模型部署失效）**

## 关闭说明

2026-08-24 部署训练仓库 gen_450000.pth（v3 prior_head 架构）并同步推理融合方式后，
本待办所述「prior_threshold 二值化出像素」的前提已不成立：

- 推理融合改为**软融合**：连续 sigmoid 概率场 + max-pool 膨胀（prior_dilate=5）直接加权，
  **不再有任何二值化步骤参与出像素**（模型分辨率与高分辨率两级均如此）；
- prior_threshold=0.30 仅作语义保留与日志记录，不再影响输出（对齐训练仓库
  infer_subtitle.py 2026-08-23 定稿生产配置）；
- 膨胀态阈值扫描已在训练仓库完成（2026-08-23，gen_450000 + dilate5）：
  0.30→0.05 仅换 +0.28dB hole / +0.008 R，代价是误检面积大增，不值——0.30 即最优区起点。

若未来重新引入二值化路径（如新增硬遮罩模式），再按原校准方法重开此待办。

## 背景

`prior_threshold` 是 STTN-auto 模型推理后处理的关键参数，控制 `prior_head` 输出的 sigmoid 概率被二值化为"文字区"的阈值：

```python
prior_bin = (torch.sigmoid(prior_logits) > prior_threshold).astype(np.float32)
```

prior_mask 决定高清融合时的区域切分：
- `prior=1`（文字区）→ 取 decoder 修复结果
- `prior=0`（非文字区）→ 保留原始高清像素

阈值偏高 → 漏检文字（残留字幕）；阈值偏低 → 误检文字（非文字区被替换，可能糊化背景细节）。

## 当前状态

| 项 | 值 | 来源 |
|---|---|---|
| 当前阈值 | **0.30** | 训练仓库 `infer_subtitle.py` 默认值，76K 帧 IoU 扫描最优 |
| 阈值出现位置 | `core/inpaint/sttn.py` 的 `PRIOR_DILATE` / `MASK_THRESHOLD` 常量 | 模型分辨率内部 prior 后处理 |
| 训练仓库验证最优 | 0.30 | `validate_subtitle.py` 注释「76K 扫描最优，R 0.9805」 |

**问题**：0.30 是训练仓库在 val 集上扫出的，**本项目部署的模型权重与训练 val 集分布可能不同**，需要在本项目实际使用的模型 + 真实视频数据上重新校准。

## 校准方法（参考训练仓库 `scan_prior_threshold`）

训练仓库 `official_sttn/validate_subtitle.py` 的 `scan_prior_threshold()`（L402-497）提供了标准流程：

1. **准备 val 集**：用 `generate_val_clips.py` 生成 .pt（含 subbed/clean/glyph），或本项目自带一批带 GT mask 的测试视频
2. **遍历候选阈值**：`[0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]`（步长 0.1 粗扫）→ 在最优区间 0.02 步长细扫
3. **每个阈值计算指标**：
   - IoU = TP / (TP + FP + FN)
   - Precision = TP / (TP + FP)
   - Recall = TP / (TP + FN)
   - prior 在模型分辨率计算，上采样到 crop 全分辨率后与 GT glyph mask 比较（`F.interpolate` bilinear → `> 0.5` 二值化，与推理融合一致）
4. **选阈值**：以 **Recall 优先**（漏字比误检更影响观感），目标 Recall ≥ 0.98，在此前提下取 IoU 最高的阈值

## 需要处理的点

- [ ] 准备本项目测试集（10-20 段含硬字幕的真实视频 + 对应 GT mask）
- [ ] 移植/编写 `scan_prior_threshold` 脚本到本项目（`tools/` 目录）
- [ ] 在本项目部署的模型权重上跑扫描，记录 IoU/P/R 曲线
- [ ] 确定最终阈值，更新 `core/inpaint/sttn.py` 中 `MASK_THRESHOLD` 的硬编码值
- [ ] （可选）将阈值提取为 config.json 可配置项，便于后续不重新打包即调整

## 相关代码位置

| 文件 | 行号 | 说明 |
|---|---|---|
| [sttn.py](file:///e:/AI%20Tool/video%20subtitle%20remover/core/inpaint/sttn.py) | `MASK_THRESHOLD` | prior 二值化阈值 |
| [propainter.py](file:///e:/AI%20Tool/video%20subtitle%20remover/core/inpaint/propainter.py) | `MASK_THRESHOLD` | mask 条件修复趟的二值化阈值 |
| 训练仓库 `validate_subtitle.py` | L402-497 | `scan_prior_threshold` 参考实现 |

## 备注

- 本项目还有 4 处 crop 尺寸的 `> 0.5` 二值化（L275、L289、L534、L543），那是 prior 上采样后的再二值化，**不是 prior_threshold**，校准时不要动这 4 处。
- 训练时 prior 的监督是 GT glyph mask（精确字形 + dilate 1px），推理时 prior_head 学到的就是逼近这个分布，所以校准的 GT 也应是精确字形 mask，不是 bbox 矩形。
