#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MedicalNet-18 adapted for 4-channel 3D MRI classification.

This module ports the MedicalNet 3D ResNet-18 backbone from
``MedicalNet/models/resnet.py`` and replaces the original segmentation head
with global average pooling plus a lightweight classification head.

Expected input layout is ``[B, 4, D, H, W]`` with channels ordered as
T1, T1ce, T2, T2FLAIR.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "BasicBlock",
    "MedicalNet18",
    "adapt_first_conv_weight",
]


def conv3x3x3(
    in_planes: int,
    out_planes: int,
    stride: int | tuple[int, int, int] = 1,
    dilation: int = 1,
) -> nn.Conv3d:
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        dilation=dilation,
        bias=False,
    )


def downsample_basic_block(
    x: torch.Tensor,
    planes: int,
    stride: int | tuple[int, int, int],
) -> torch.Tensor:
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    channel_padding = planes - out.shape[1]
    if channel_padding <= 0:
        return out[:, :planes]

    zero_pads = out.new_zeros(
        out.shape[0],
        channel_padding,
        out.shape[2],
        out.shape[3],
        out.shape[4],
    )
    return torch.cat([out, zero_pads], dim=1)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int | tuple[int, int, int] = 1,
        dilation: int = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)
        return out


class _ShortcutA(nn.Module):
    def __init__(
        self,
        planes: int,
        stride: int | tuple[int, int, int],
    ) -> None:
        super().__init__()
        self.planes = planes
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return downsample_basic_block(x, planes=self.planes, stride=self.stride)


def adapt_first_conv_weight(
    pretrained_weight: torch.Tensor,
    new_in_channels: int,
) -> torch.Tensor:
    """Adapt MedicalNet conv1 weights to a new number of input channels."""

    if pretrained_weight.ndim != 5:
        raise ValueError(
            "Expected 3D conv weight with shape [out,in,kD,kH,kW], "
            f"got {tuple(pretrained_weight.shape)}."
        )
    if new_in_channels <= 0:
        raise ValueError(f"new_in_channels must be positive, got {new_in_channels}.")

    out_channels, old_in_channels, kernel_d, kernel_h, kernel_w = pretrained_weight.shape
    if old_in_channels == new_in_channels:
        return pretrained_weight

    if new_in_channels == 1:
        return pretrained_weight.mean(dim=1, keepdim=True)

    repeat_times = (new_in_channels + old_in_channels - 1) // old_in_channels
    expanded_weight = pretrained_weight.repeat(1, repeat_times, 1, 1, 1)
    expanded_weight = expanded_weight[:, :new_in_channels, :, :, :]
    expanded_weight = expanded_weight * (old_in_channels / float(new_in_channels))

    expected_shape = (out_channels, new_in_channels, kernel_d, kernel_h, kernel_w)
    if tuple(expanded_weight.shape) != expected_shape:
        raise RuntimeError(
            "Failed to adapt conv1 weight. "
            f"Expected {expected_shape}, got {tuple(expanded_weight.shape)}."
        )
    return expanded_weight


def _safe_torch_load(path: str | Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _extract_state_dict(checkpoint: Any) -> Mapping[str, torch.Tensor]:
    if not isinstance(checkpoint, Mapping):
        raise TypeError(
            "Expected checkpoint to be a mapping or contain a state_dict mapping, "
            f"got {type(checkpoint)!r}."
        )

    state_dict = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state_dict, Mapping):
        raise TypeError("checkpoint['state_dict'] must be a mapping.")
    return state_dict


def _normalize_state_key(key: str) -> str:
    while key.startswith("module."):
        key = key[len("module.") :]
    return key


class MedicalNet18(nn.Module):
    """3D ResNet-18 backbone from MedicalNet with a classification head."""

    out_dim = 512

    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 2,
        pretrain_path: str | Path | None = None,
        dropout_p: float = 0.0,
        shortcut_type: str = "A",
    ) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}.")
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}.")
        if not (0.0 <= dropout_p < 1.0):
            raise ValueError(f"dropout_p must be in [0.0, 1.0), got {dropout_p}.")
        if shortcut_type not in {"A", "B"}:
            raise ValueError(f"shortcut_type must be 'A' or 'B', got {shortcut_type!r}.")

        self.inplanes = 64
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.shortcut_type = shortcut_type

        self.conv1 = nn.Conv3d(
            in_channels,
            64,
            kernel_size=7,
            stride=(2, 2, 2),
            padding=(3, 3, 3),
            bias=False,
        )
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(BasicBlock, 64, 2, shortcut_type)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, shortcut_type, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, shortcut_type, stride=1, dilation=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, shortcut_type, stride=1, dilation=4)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        if dropout_p > 0.0:
            self.classifier = nn.Sequential(
                nn.Dropout(p=dropout_p),
                nn.Linear(self.out_dim, num_classes),
            )
        else:
            self.classifier = nn.Linear(self.out_dim, num_classes)

        self._init_weights()
        self.pretrain_info: dict[str, object] | None = None
        if pretrain_path is not None:
            self.pretrain_info = self.load_medicalnet_pretrain(pretrain_path)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out")
            elif isinstance(module, nn.BatchNorm3d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

    def _make_layer(
        self,
        block: type[BasicBlock],
        planes: int,
        blocks: int,
        shortcut_type: str,
        stride: int = 1,
        dilation: int = 1,
    ) -> nn.Sequential:
        downsample: nn.Module | None = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == "A":
                downsample = _ShortcutA(
                    planes=planes * block.expansion,
                    stride=stride,
                )
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm3d(planes * block.expansion),
                )

        layers: list[nn.Module] = [
            block(
                self.inplanes,
                planes,
                stride=stride,
                dilation=dilation,
                downsample=downsample,
            )
        ]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def load_medicalnet_pretrain(self, pretrain_path: str | Path) -> dict[str, object]:
        checkpoint = _safe_torch_load(pretrain_path)
        state_dict = _extract_state_dict(checkpoint)
        model_state = self.state_dict()
        filtered_state: dict[str, torch.Tensor] = {}
        skipped: list[str] = []
        adapted: list[str] = []

        for raw_key, value in state_dict.items():
            key = _normalize_state_key(str(raw_key))
            if key.startswith("conv_seg") or key.startswith("classifier"):
                skipped.append(key)
                continue
            if key not in model_state:
                skipped.append(key)
                continue
            if not isinstance(value, torch.Tensor):
                skipped.append(key)
                continue

            target = model_state[key]
            if key == "conv1.weight" and tuple(value.shape) != tuple(target.shape):
                value = adapt_first_conv_weight(value, self.in_channels)
                adapted.append(key)

            if tuple(value.shape) != tuple(target.shape):
                skipped.append(key)
                continue
            filtered_state[key] = value

        missing, unexpected = self.load_state_dict(filtered_state, strict=False)
        return {
            "path": str(pretrain_path),
            "loaded_keys": len(filtered_state),
            "adapted_keys": adapted,
            "skipped_keys": skipped,
            "missing_keys": list(missing),
            "unexpected_keys": list(unexpected),
        }

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
        z = self.forward_features(x)
        return self.classifier(z)


if __name__ == "__main__":
    torch.manual_seed(42)
    model = MedicalNet18(in_channels=4, num_classes=2, pretrain_path=None)
    model.eval()
    x = torch.randn(2, 4, 16, 64, 64)
    with torch.no_grad():
        z = model.forward_features(x)
        logits = model(x)
    print("MedicalNet18 self-check passed.")
    print(f"  z shape      : {tuple(z.shape)}")
    print(f"  logits shape : {tuple(logits.shape)}")
