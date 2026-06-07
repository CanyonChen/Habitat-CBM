#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""3D Habitat-CBM model using MedicalNet-18 as the image backbone."""

from __future__ import annotations

from pathlib import Path
import sys

import torch
import torch.nn as nn

try:
    from models.medicalnet_18 import MedicalNet18
except ImportError:  # pragma: no cover
    try:
        from .medicalnet_18 import MedicalNet18
    except ImportError:  # pragma: no cover
        CURRENT_DIR = Path(__file__).resolve().parent
        if str(CURRENT_DIR) not in sys.path:
            sys.path.insert(0, str(CURRENT_DIR))
        from medicalnet_18 import MedicalNet18


class MedicalNet18Backbone(nn.Module):
    """MedicalNet-18 backbone that returns a [B, 512] feature vector."""

    def __init__(
        self,
        in_channels: int = 4,
        pretrain_path: str | Path | None = None,
    ) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}.")

        base = MedicalNet18(
            in_channels=in_channels,
            num_classes=2,
            pretrain_path=pretrain_path,
            dropout_p=0.0,
            shortcut_type="A",
        )
        self.in_channels = in_channels
        self.pretrain_info = base.pretrain_info
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = base.avgpool
        self.out_dim = base.out_dim

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"x must be 5D [B,C,D,H,W], got shape {tuple(x.shape)}.")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"x channel dim must be {self.in_channels}, got {x.shape[1]}."
            )

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.avgpool(x).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)


class HabitatCBM3D(nn.Module):
    """Hard concept bottleneck model with a 3D MedicalNet-18 encoder."""

    LABEL_HEAD_TYPE = "dropout_linear"

    def __init__(
        self,
        in_channels: int = 4,
        n_concepts: int = 5,
        concept_hidden_dim: int = 256,
        label_hidden_dim: int = 32,
        dropout_p: float | None = None,
        concept_dropout_p: float | None = None,
        label_dropout_p: float | None = None,
        pretrain_path: str | Path | None = None,
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
            raise ValueError(f"dropout_p must be in [0.0, 1.0), got {shared_dropout}.")

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
        self.label_hidden_dim = int(label_hidden_dim)
        self.label_head_type = self.LABEL_HEAD_TYPE
        self.encoder = MedicalNet18Backbone(
            in_channels=in_channels,
            pretrain_path=pretrain_path,
        )
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
        if c.ndim != 2:
            raise ValueError(
                f"{tensor_name} must be 2D [B, {self.n_concepts}], "
                f"got shape {tuple(c.shape)}."
            )
        if c.shape[1] != self.n_concepts:
            raise ValueError(
                f"{tensor_name} second dim must be {self.n_concepts}, got {c.shape[1]}."
            )

    def forward_x_to_c(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.concept_head(z)

    def forward_c_to_y(self, c: torch.Tensor) -> torch.Tensor:
        self._validate_concept_tensor(c, tensor_name="c")
        return self.label_head(c)

    def forward_x_to_cy(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.encoder(x)
        c_hat = self.concept_head(z)
        y_logit = self.label_head(c_hat)
        return {
            "z": z,
            "c_hat": c_hat,
            "y_logit": y_logit,
        }

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.forward_x_to_cy(x)


if __name__ == "__main__":
    torch.manual_seed(42)
    model = HabitatCBM3D(
        in_channels=4,
        n_concepts=5,
        concept_hidden_dim=256,
        label_hidden_dim=32,
        concept_dropout_p=0.3,
        label_dropout_p=0.1,
        pretrain_path=None,
    )
    model.eval()
    x = torch.randn(2, 4, 16, 64, 64)
    with torch.no_grad():
        out = model.forward_x_to_cy(x)
    print("HabitatCBM3D self-check passed.")
    print(f"  label_head    : {model.label_head_type}")
    print(f"  z shape       : {tuple(out['z'].shape)}")
    print(f"  c_hat shape   : {tuple(out['c_hat'].shape)}")
    print(f"  y_logit shape : {tuple(out['y_logit'].shape)}")
