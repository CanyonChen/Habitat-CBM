#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Habitat-CBM 模型定义（纯网络结构版本）。

设计边界：
1. 本文件只负责模型结构与前向接口（X->C->Y）；
2. 训练相关逻辑（loss、stage 冻结策略、优化器参数组）已解耦到
   `repo/srcs/train_habitat_CBM.py`；
3. 推理/验证相关逻辑（概率输出、TTI 干预、患者级聚合）已解耦到
   `repo/srcs/eval_habitat_CBM.py`。
"""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    # 当脚本以 `repo` 为 import 根目录运行时可用
    from models.resnet_18 import ResNet18
except ImportError:  # pragma: no cover - 兼容包内相对导入
    from .resnet_18 import ResNet18


class ResNet18Backbone(nn.Module):
    """仅输出 feature 向量 z 的 ResNet-18 骨干。"""

    def __init__(self, in_channels: int, pretrained: bool = True) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}.")

        base = ResNet18(
            in_channels=in_channels,
            num_classes=2,  # 占位参数，CBM 不使用 base.fc 做分类
            pretrained=pretrained,
            dropout_p=0.0,
        )

        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = base.avgpool
        self.out_dim = 512

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """提取 CNN 特征并输出 [B, 512]。"""

        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        z = self.avgpool(x).flatten(1)
        return z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)


class HabitatCBM(nn.Module):
    """Habitat-CBM 主模型（硬概念瓶颈，纯结构）。

    结构：
        z = encoder(x)                    # [B, 512]
        c_hat = concept_head(z)           # [B, n_concepts]
        y_logit = linear(dropout(c_hat))  # [B, 1]
    """

    LABEL_HEAD_TYPE = "dropout_linear"

    def __init__(
        self,
        in_channels: int,
        n_concepts: int = 8,
        concept_hidden_dim: int = 256,
        label_hidden_dim: int = 32,
        dropout_p: float | None = None,
        concept_dropout_p: float | None = None,
        label_dropout_p: float | None = None,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}.")
        if n_concepts <= 0:
            raise ValueError(f"n_concepts must be positive, got {n_concepts}.")
        if concept_hidden_dim <= 0:
            raise ValueError(
                f"concept_hidden_dim must be positive, got {concept_hidden_dim}."
            )

        shared_dropout = float(dropout_p) if dropout_p is not None else None
        if shared_dropout is not None and not (0.0 <= shared_dropout < 1.0):
            raise ValueError(
                f"dropout_p must be in [0.0, 1.0), got {shared_dropout}."
            )

        if concept_dropout_p is None:
            concept_dropout_p = shared_dropout if shared_dropout is not None else 0.3
        if label_dropout_p is None:
            label_dropout_p = shared_dropout if shared_dropout is not None else 0.1

        if not (0.0 <= concept_dropout_p < 1.0):
            raise ValueError(
                "concept_dropout_p must be in [0.0, 1.0), "
                f"got {concept_dropout_p}."
            )
        if not (0.0 <= label_dropout_p < 1.0):
            raise ValueError(
                f"label_dropout_p must be in [0.0, 1.0), got {label_dropout_p}."
            )

        self.n_concepts = n_concepts
        # `label_hidden_dim` 保留在接口中，仅用于兼容旧配置/旧 checkpoint 元数据。
        self.label_hidden_dim = int(label_hidden_dim)
        self.label_head_type = self.LABEL_HEAD_TYPE
        self.encoder = ResNet18Backbone(in_channels=in_channels, pretrained=pretrained)
        self.concept_head = nn.Sequential(
            nn.Linear(self.encoder.out_dim, concept_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(concept_dropout_p),
            nn.Linear(concept_hidden_dim, n_concepts),
        )
        self.label_head = nn.Sequential(
            nn.Dropout(label_dropout_p),
            nn.Linear(n_concepts, 1),
        )

    def _validate_concept_tensor(self, c: torch.Tensor, tensor_name: str) -> None:
        """检查概念张量形状是否与 n_concepts 匹配。"""

        if c.ndim != 2:
            raise ValueError(
                f"{tensor_name} must be 2D [B, {self.n_concepts}], got shape {tuple(c.shape)}."
            )
        if c.shape[1] != self.n_concepts:
            raise ValueError(
                f"{tensor_name} second dim must be {self.n_concepts}, got {c.shape[1]}."
            )

    def forward_x_to_c(self, x: torch.Tensor) -> torch.Tensor:
        """仅执行 X -> C。"""

        z = self.encoder(x)
        c_hat = self.concept_head(z)
        return c_hat

    def forward_c_to_y(self, c: torch.Tensor) -> torch.Tensor:
        """仅执行 C -> Y。"""

        self._validate_concept_tensor(c, tensor_name="c")
        return self.label_head(c)

    def forward_x_to_cy(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """执行完整前向 X -> C -> Y。"""

        z = self.encoder(x)
        c_hat = self.concept_head(z)
        y_logit = self.label_head(c_hat)  # hard bottleneck: y 仅依赖 c_hat
        return {
            "z": z,
            "c_hat": c_hat,
            "y_logit": y_logit,
        }

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """默认前向接口，返回 z/c_hat/y_logit。"""

        return self.forward_x_to_cy(x)


if __name__ == "__main__":
    # 最小自检：仅检查结构与前向维度契约。
    torch.manual_seed(42)
    model = HabitatCBM(
        in_channels=35,
        n_concepts=8,
        concept_hidden_dim=256,
        label_hidden_dim=32,
        concept_dropout_p=0.3,
        label_dropout_p=0.1,
        pretrained=False,
    )
    x = torch.randn(2, 35, 224, 224)
    out = model.forward_x_to_cy(x)
    print("HabitatCBM (pure model) self-check passed.")
    print(f"  label_head    : {model.label_head_type}")
    print(f"  z shape       : {tuple(out['z'].shape)}")
    print(f"  c_hat shape   : {tuple(out['c_hat'].shape)}")
    print(f"  y_logit shape : {tuple(out['y_logit'].shape)}")
