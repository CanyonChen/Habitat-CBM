#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""3D transform builders for Habitat-CBM volume samples."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F

try:
    from monai.transforms import (
        Compose,
        EnsureTyped,
        RandAffined,
        RandFlipd,
        RandGaussianNoised,
        RandGibbsNoised,
        RandScaleIntensityd,
        RandStdShiftIntensityd,
        Resized,
    )
except ImportError as exc:  # pragma: no cover - only exercised when MONAI is missing.
    Compose = None
    EnsureTyped = None
    RandAffined = None
    RandFlipd = None
    RandGaussianNoised = None
    RandGibbsNoised = None
    RandScaleIntensityd = None
    RandStdShiftIntensityd = None
    Resized = None
    _MONAI_IMPORT_ERROR = exc
else:
    _MONAI_IMPORT_ERROR = None


@dataclass(frozen=True)
class MonaiVolumeAugmentConfig:
    """Config for 3D MRI volume augmentation."""

    enabled: bool = True
    affine_prob: float = 0.5
    rotate_deg: float = 10.0
    translate_px: float = 6.0
    scale_range: float = 0.10
    flip_prob: float = 0.5
    intensity_scale_prob: float = 0.5
    intensity_scale: float = 0.15
    intensity_shift_prob: float = 0.5
    intensity_shift: float = 0.10
    gaussian_noise_prob: float = 0.25
    gaussian_noise_std: float = 0.03
    gibbs_noise_prob: float = 0.0
    gibbs_noise_alpha: float = 0.3

    def to_dict(self) -> Dict[str, float | bool]:
        return asdict(self)


class ResizeVolumeTransform:
    """Small MONAI-free fallback for deterministic resize + tensor conversion."""

    def __init__(self, spatial_size: Sequence[int] | None) -> None:
        self.spatial_size = _normalize_spatial_size(spatial_size)

    def __call__(self, sample: Dict[str, object]) -> Dict[str, object]:
        image = torch.as_tensor(sample["image"], dtype=torch.float32)
        if image.ndim != 4:
            raise ValueError(f"Expected image [C,D,H,W], got {tuple(image.shape)}.")
        if self.spatial_size is not None:
            image = F.interpolate(
                image.unsqueeze(0),
                size=self.spatial_size,
                mode="trilinear",
                align_corners=False,
            ).squeeze(0)
        output = dict(sample)
        output["image"] = image
        return output


def _normalize_spatial_size(
    spatial_size: Union[int, Sequence[int], None],
) -> tuple[int, int, int] | None:
    if spatial_size is None:
        return None
    if isinstance(spatial_size, int):
        if spatial_size <= 0:
            return None
        return (int(spatial_size), int(spatial_size), int(spatial_size))
    if len(spatial_size) != 3:
        raise ValueError(f"spatial_size must be [D,H,W], got {spatial_size!r}.")
    parsed = tuple(int(v) for v in spatial_size)
    if any(v <= 0 for v in parsed):
        return None
    return parsed


def _require_monai_for_random_aug(config: MonaiVolumeAugmentConfig) -> None:
    if _MONAI_IMPORT_ERROR is None:
        return
    if config.enabled:
        raise ImportError(
            "MONAI is required for random 3D augmentation. "
            "Install it with `pip install -r habitat_CBM/repo/requirements.txt` "
            "or set train.use_monai_augmentation=false."
        ) from _MONAI_IMPORT_ERROR


def build_monai_volume_transforms(
    config: MonaiVolumeAugmentConfig,
    spatial_size: Union[int, Sequence[int], None] = None,
) -> Dict[str, object]:
    """Build train/val/test transforms for channel-first 3D volume samples."""

    target_size = _normalize_spatial_size(spatial_size)
    _require_monai_for_random_aug(config)

    if _MONAI_IMPORT_ERROR is not None:
        fallback = ResizeVolumeTransform(target_size)
        return {"train": fallback, "val": fallback, "test": fallback}

    resize_transform = None
    if target_size is not None:
        resize_transform = Resized(
            keys=["image"],
            spatial_size=target_size,
            mode=["trilinear"],
        )
    typed = EnsureTyped(keys=["image"], dtype=torch.float32, track_meta=False)

    eval_ops = []
    if resize_transform is not None:
        eval_ops.append(resize_transform)
    eval_ops.append(typed)
    eval_transform = Compose(eval_ops)

    if not config.enabled:
        return {"train": eval_transform, "val": eval_transform, "test": eval_transform}

    train_ops = []
    if resize_transform is not None:
        train_ops.append(resize_transform)

    use_affine = (
        config.affine_prob > 0.0
        and (config.rotate_deg > 0.0 or config.translate_px > 0.0 or config.scale_range > 0.0)
    )
    if use_affine:
        rotate = float(np.deg2rad(config.rotate_deg))
        train_ops.append(
            RandAffined(
                keys=["image"],
                prob=float(config.affine_prob),
                rotate_range=(rotate, rotate, rotate),
                translate_range=(
                    float(config.translate_px),
                    float(config.translate_px),
                    float(config.translate_px),
                ),
                scale_range=(
                    float(config.scale_range),
                    float(config.scale_range),
                    float(config.scale_range),
                ),
                mode=["bilinear"],
                padding_mode=["zeros"],
                spatial_size=None,
                cache_grid=False,
            )
        )

    if config.flip_prob > 0.0:
        train_ops.append(RandFlipd(keys=["image"], prob=float(config.flip_prob), spatial_axis=(0, 1, 2)))
    if config.intensity_scale_prob > 0.0 and config.intensity_scale > 0.0:
        train_ops.append(
            RandScaleIntensityd(
                keys=["image"],
                factors=float(config.intensity_scale),
                prob=float(config.intensity_scale_prob),
                channel_wise=True,
            )
        )
    if config.intensity_shift_prob > 0.0 and config.intensity_shift > 0.0:
        train_ops.append(
            RandStdShiftIntensityd(
                keys=["image"],
                factors=float(config.intensity_shift),
                prob=float(config.intensity_shift_prob),
                nonzero=True,
                channel_wise=True,
            )
        )
    if config.gaussian_noise_prob > 0.0 and config.gaussian_noise_std > 0.0:
        train_ops.append(
            RandGaussianNoised(
                keys=["image"],
                prob=float(config.gaussian_noise_prob),
                mean=0.0,
                std=float(config.gaussian_noise_std),
            )
        )
    if config.gibbs_noise_prob > 0.0 and config.gibbs_noise_alpha > 0.0:
        train_ops.append(
            RandGibbsNoised(
                keys=["image"],
                prob=float(config.gibbs_noise_prob),
                alpha=(0.0, float(config.gibbs_noise_alpha)),
            )
        )

    train_ops.append(typed)
    return {"train": Compose(train_ops), "val": eval_transform, "test": eval_transform}

