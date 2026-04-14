#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ResNet-18 模型定义文件。

设计目标：
1. 单独维护 ResNet-18 结构，不把训练逻辑写进模型文件；
2. 支持 ImageNet 预训练权重初始化；
3. 支持自定义输入通道数，适配多模态 2.5D MRI 输入；
4. 支持自定义输出类别数，便于后续 baseline 与其他实验复用；
5. 支持在分类头前插入 Dropout，用于正则化、缓解过拟合；
6. 支持分层参数组（Layerwise Parameter Groups），便于对骨干网络与
   分类头使用差异化学习率，防止预训练特征被过大的学习率破坏。

推荐用法：

```python
from models.resnet_18 import ResNet18Classifier

# 构建模型
model = ResNet18Classifier(
    in_channels=35,      # 例如 6 模态 * 5 切片 + 5 VOI 通道 = 35
    num_classes=2,
    pretrained=True,
    dropout_p=0.5,       # 分类头前 Dropout 概率，0.0 表示不使用
)

# 获取分层参数组，对骨干网络使用较小的学习率
param_groups = model.get_param_groups(base_lr=1e-4, lr_mult=0.1)
optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-2)
```

分层学习率策略说明：
  ResNet-18 骨干网络按深度分为 4 个阶段（layer1~layer4），加上 conv1/bn1
  组成的浅层特征提取器，以及最后的全连接分类头（fc）。迁移学习时推荐：

  - 浅层（conv1 + layer1）：学习率最小，保留 ImageNet 通用特征；
  - 中层（layer2 + layer3）：学习率适中；
  - 深层（layer4）：学习率较大，适应领域特征；
  - 分类头（fc）：学习率最大，随机初始化需快速收敛。

  通过 `get_param_groups(base_lr, lr_mult)` 自动按上述策略分配学习率，
  其中 `base_lr` 为分类头学习率，`lr_mult`（默认 0.1）为骨干相对倍率。
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


def _adapt_first_conv_weight(
    pretrained_weight: torch.Tensor,
    new_in_channels: int,
) -> torch.Tensor:
    """将 ImageNet 预训练的第一层卷积权重适配到新的输入通道数。

    参数：
    - pretrained_weight:
      原始 ResNet-18 第一层卷积权重，形状为 [64, 3, 7, 7]
    - new_in_channels:
      新模型期望的输入通道数

    返回：
    - 适配后的卷积权重，形状为 [64, new_in_channels, 7, 7]

    适配策略：
    1. 若 `new_in_channels == 3`，直接返回原权重；
    2. 若 `new_in_channels == 1`，对 RGB 通道求均值，得到单通道初始化；
    3. 若 `new_in_channels > 3`，重复使用原始 3 通道权重并截断到目标通道数；
    4. 为避免通道数变化后卷积输出幅值失衡，再按 `3 / new_in_channels` 做缩放。
    """

    out_channels, old_in_channels, kernel_h, kernel_w = pretrained_weight.shape
    if old_in_channels != 3:
        raise ValueError(
            f"Unexpected pretrained conv1 shape: expected 3 input channels, got {old_in_channels}."
        )

    if new_in_channels == 3:
        return pretrained_weight

    if new_in_channels == 1:
        return pretrained_weight.mean(dim=1, keepdim=True)

    repeat_times = (new_in_channels + old_in_channels - 1) // old_in_channels
    expanded_weight = pretrained_weight.repeat(1, repeat_times, 1, 1)
    expanded_weight = expanded_weight[:, :new_in_channels, :, :]

    scale = old_in_channels / float(new_in_channels)
    expanded_weight = expanded_weight * scale

    if expanded_weight.shape != (out_channels, new_in_channels, kernel_h, kernel_w):
        raise RuntimeError(
            "Failed to adapt conv1 weight to the requested input channels. "
            f"Expected {(out_channels, new_in_channels, kernel_h, kernel_w)}, "
            f"got {tuple(expanded_weight.shape)}."
        )

    return expanded_weight


def ResNet18(
    in_channels: int,
    num_classes: int = 2,
    pretrained: bool = True,
    dropout_p: float = 0.0,
) -> nn.Module:
    """构建适用于当前项目的 ResNet-18。

    参数：
    - in_channels:
      输入通道数。
      对于本项目，使用 6 模态 × 5 切片 + 5 VOI 通道，则 `in_channels = 35`。
    - num_classes:
      输出类别数。当前 IDH 二分类通常设为 2。
    - pretrained:
      是否加载 ImageNet 预训练权重。
    - dropout_p:
      分类头前 Dropout 的 dropout 概率（0.0 表示不使用 Dropout）。
      在训练集远大于验证集时（如当前约 6000 blocks 对应 14 名验证患者），
      适当的 Dropout（建议 0.3～0.5）可显著抑制过拟合。

    返回：
    - 已完成输入输出适配的 `torchvision` ResNet-18 模型。
    """

    if in_channels <= 0:
        raise ValueError(f"in_channels must be positive, got {in_channels}.")
    if num_classes <= 0:
        raise ValueError(f"num_classes must be positive, got {num_classes}.")
    if not (0.0 <= dropout_p < 1.0):
        raise ValueError(f"dropout_p must be in [0.0, 1.0), got {dropout_p}.")

    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)

    # 若输入通道数不是标准 RGB 3 通道，则替换第一层卷积。
    if in_channels != 3:
        old_conv = model.conv1
        new_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )

        if pretrained:
            with torch.no_grad():
                adapted_weight = _adapt_first_conv_weight(old_conv.weight.data, in_channels)
                new_conv.weight.copy_(adapted_weight)

        model.conv1 = new_conv

    # 替换最后的全连接层，使输出类别数与当前任务一致。
    # 若 dropout_p > 0，在线性层前插入 Dropout 以正则化分类头，抑制过拟合。
    in_features = model.fc.in_features
    if dropout_p > 0.0:
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features, num_classes),
        )
    else:
        model.fc = nn.Linear(in_features, num_classes)

    return model


class ResNet18Classifier(nn.Module):
    """一个轻量封装，便于训练脚本直接实例化并通过 CLI 参数配置模型。

    封装了 ResNet18() 工厂函数的所有参数，统一对外暴露。
    额外提供 get_param_groups() 方法，支持分层差异化学习率。
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int = 2,
        pretrained: bool = True,
        dropout_p: float = 0.0,
    ) -> None:
        """
        参数：
        - in_channels: 输入通道数（如 35 = 6 模态 × 5 切片 + 5 VOI 通道）。
        - num_classes: 输出类别数，IDH 二分类设为 2。
        - pretrained: 是否使用 ImageNet 预训练权重。
        - dropout_p: 分类头前 Dropout 概率，0.0 表示不使用。
        """
        super().__init__()
        self.model = ResNet18(
            in_channels=in_channels,
            num_classes=num_classes,
            pretrained=pretrained,
            dropout_p=dropout_p,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播，输出分类 logits。"""

        return self.model(x)

    def get_param_groups(
        self,
        base_lr: float,
        lr_mult: float = 0.1,
    ) -> list[dict]:
        """将模型参数按层次分组，返回可直接传入优化器的参数组列表。

        ResNet-18 骨干按深度分为 5 组（浅→深），分类头单独一组，共 6 组。
        骨干各组的学习率 = base_lr × lr_mult × 层级系数，保证浅层改动最小。

        分组与学习率分配：

          组别            层                学习率
          ──────────────────────────────────────────────────
          group_stem     conv1 + bn1       base_lr × lr_mult × 0.2
          group_layer1   layer1            base_lr × lr_mult × 0.4
          group_layer2   layer2            base_lr × lr_mult × 0.6
          group_layer3   layer3            base_lr × lr_mult × 0.8
          group_layer4   layer4            base_lr × lr_mult × 1.0
          group_head     fc                base_lr
          ──────────────────────────────────────────────────

        参数：
        - base_lr:
          分类头（fc 层）的学习率，通常取优化器全局 lr（如 args.lr）。
        - lr_mult:
          骨干网络相对于分类头的学习率倍率（默认 0.1）。
          例如 base_lr=1e-4、lr_mult=0.1，则骨干最深层学习率为 1e-5。

        返回：
        - 包含 6 个字典的列表，每个字典形如
          {"params": [...], "lr": float, "name": str}，
          可直接传入 torch.optim.AdamW / SGD 等优化器。

        使用示例：
        ```python
        param_groups = model.get_param_groups(base_lr=args.lr, lr_mult=0.1)
        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=args.weight_decay,
        )
        ```

        注意：
        - 仅包含 requires_grad=True 的参数，冻结层会被自动跳过；
        - 若某组内所有参数均被冻结，该组仍会出现在列表中（params 为空列表），
          优化器对空组无任何影响，不影响训练；
        - 建议配合 CosineAnnealingLR 或 ReduceLROnPlateau 调度器使用，
          调度器会按比例同步调整各组学习率。
        """
        m = self.model  # 内部 torchvision ResNet-18 实例

        # 定义各骨干段及其相对学习率系数（越浅系数越小）
        backbone_groups = [
            ("stem",   [m.conv1, m.bn1], 0.2),
            ("layer1", [m.layer1],       0.4),
            ("layer2", [m.layer2],       0.6),
            ("layer3", [m.layer3],       0.8),
            ("layer4", [m.layer4],       1.0),
        ]

        param_groups: list[dict] = []

        for group_name, modules, scale in backbone_groups:
            params = [
                p
                for mod in modules
                for p in mod.parameters()
                if p.requires_grad
            ]
            param_groups.append({
                "params": params,
                "lr":     base_lr * lr_mult * scale,
                "name":   f"backbone_{group_name}",
            })

        # 分类头：使用完整的 base_lr
        head_params = [p for p in m.fc.parameters() if p.requires_grad]
        param_groups.append({
            "params": head_params,
            "lr":     base_lr,
            "name":   "head_fc",
        })

        return param_groups


if __name__ == "__main__":
    # 简单自检：验证模型可以被正常构建并处理一个假输入。
    # 同时测试 dropout_p 参数是否生效（含 Dropout 层和不含 Dropout 层两种情况）。
    for dp in (0.0, 0.5):
        dummy_model = ResNet18(in_channels=35, num_classes=2, pretrained=False, dropout_p=dp)
        dummy_input = torch.randn(2, 35, 224, 224)
        dummy_output = dummy_model(dummy_input)
        print(f"[dropout_p={dp}] Model build check passed.")
        print(f"  Input shape : {tuple(dummy_input.shape)}")
        print(f"  Output shape: {tuple(dummy_output.shape)}")
        print(f"  fc layer    : {dummy_model.fc}")

    # 测试 ResNet18Classifier.get_param_groups()
    print("\n--- get_param_groups() 测试 ---")
    clf = ResNet18Classifier(in_channels=35, num_classes=2, pretrained=False, dropout_p=0.0)
    base_lr = 1e-4
    groups = clf.get_param_groups(base_lr=base_lr, lr_mult=0.1)
    total_params = 0
    for g in groups:
        n_params = sum(p.numel() for p in g["params"])
        total_params += n_params
        print(f"  [{g['name']:20s}]  lr={g['lr']:.2e}  params={n_params:,}")
    model_params = sum(p.numel() for p in clf.parameters() if p.requires_grad)
    assert total_params == model_params, (
        f"参数数量不一致：分组合计 {total_params:,} ≠ 模型总参数 {model_params:,}"
    )
    print(f"  参数总量一致性检查通过（共 {total_params:,} 个可训练参数）")
