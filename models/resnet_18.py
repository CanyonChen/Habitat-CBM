#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ResNet-18 模型定义文件。

设计目标：
1. 单独维护 ResNet-18 结构，不把训练逻辑写进模型文件；
2. 支持 ImageNet 预训练权重初始化；
3. 支持自定义输入通道数，适配多模态 2.5D MRI 输入；
4. 支持自定义输出类别数，便于后续 baseline 与其他实验复用；
5. 支持在分类头前插入 Dropout，用于正则化、缓解过拟合。

推荐用法：

```python
from models.resnet_18 import ResNet18

model = ResNet18(
    in_channels=35,      # 例如 6 模态 * 5 切片 + 5 VOI 通道 = 35
    num_classes=2,
    pretrained=True,
    dropout_p=0.5,       # 分类头前 Dropout 概率，0.0 表示不使用
)
```
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
