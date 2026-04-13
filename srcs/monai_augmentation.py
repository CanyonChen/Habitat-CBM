#!/usr/bin/env python3
"""
MONAI augmentation builders for 2.5D multimodal MRI blocks.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict

import numpy as np
import torch

try:
    from monai.transforms import (
        Compose,
        EnsureTyped,
        RandAffined,
        RandFlipd,
        RandScaleIntensityd,
        RandStdShiftIntensityd,
    )
except ImportError as exc:  # pragma: no cover - exercised only when MONAI is missing.
    Compose = None
    EnsureTyped = None
    RandAffined = None
    RandFlipd = None
    RandScaleIntensityd = None
    RandStdShiftIntensityd = None
    _MONAI_IMPORT_ERROR = exc
else:
    _MONAI_IMPORT_ERROR = None


@dataclass(frozen=True)
class MonaiAugmentConfig:
    """Config for a conservative MRI augmentation policy."""

    enabled: bool = True
    affine_prob: float = 0.5
    rotate_deg: float = 10.0
    translate_px: float = 8.0
    scale_range: float = 0.1
    flip_prob: float = 0.5
    intensity_scale_prob: float = 0.3
    intensity_scale: float = 0.1
    intensity_shift_prob: float = 0.3
    intensity_shift: float = 0.1

    def to_dict(self) -> Dict[str, float | bool]:
        """Return a JSON-safe config dictionary."""

        return asdict(self)


def _require_monai() -> None:
    """Raise a clear error message when MONAI is not installed."""

    if _MONAI_IMPORT_ERROR is None:
        return
    raise ImportError(
        "MONAI is required for the augmentation pipeline. "
        "Install it with `pip install -r habitat_CBM/repo/requirements.txt` "
        "or `pip install monai`."
    ) from _MONAI_IMPORT_ERROR


def build_monai_block_transforms(
    config: MonaiAugmentConfig,
    has_mask: bool,
) -> Dict[str, object]:
    """Build train/val/test transforms for 2D channel-first block samples."""

    _require_monai()

    keys = ["image"]
    if has_mask:
        keys.append("mask")

    typed = EnsureTyped(keys=keys, dtype=torch.float32, track_meta=False)
    eval_transform = Compose([typed])

    if not config.enabled:
        return {
            "train": eval_transform,
            "val": eval_transform,
            "test": eval_transform,
        }

    train_ops = []

    use_affine = (
        config.affine_prob > 0.0
        and (config.rotate_deg > 0.0 or config.translate_px > 0.0 or config.scale_range > 0.0)
    )
    if use_affine:
        interp_modes = tuple(["bilinear"] + (["nearest"] if has_mask else []))
        padding_modes = tuple(["zeros"] * len(keys))
        train_ops.append(
            RandAffined(
                keys=keys,
                prob=config.affine_prob,
                rotate_range=(float(np.deg2rad(config.rotate_deg)),),
                translate_range=(float(config.translate_px), float(config.translate_px)),
                scale_range=(float(config.scale_range), float(config.scale_range)),
                mode=interp_modes,
                padding_mode=padding_modes,
                spatial_size=None,
                cache_grid=False,
            )
        )

    if config.flip_prob > 0.0:
        train_ops.append(
            RandFlipd(
                keys=keys,
                prob=config.flip_prob,
                spatial_axis=1,
            )
        )

    if config.intensity_scale_prob > 0.0 and config.intensity_scale > 0.0:
        train_ops.append(
            RandScaleIntensityd(
                keys=["image"],
                factors=float(config.intensity_scale),
                prob=config.intensity_scale_prob,
                channel_wise=False,
            )
        )

    if config.intensity_shift_prob > 0.0 and config.intensity_shift > 0.0:
        train_ops.append(
            RandStdShiftIntensityd(
                keys=["image"],
                factors=float(config.intensity_shift),
                prob=config.intensity_shift_prob,
                nonzero=True,
                channel_wise=False,
            )
        )

    train_ops.append(typed)

    return {
        "train": Compose(train_ops),
        "val": eval_transform,
        "test": eval_transform,
    }
