#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Habitat-CBM 三阶段训练主脚本。

关键能力：
1. JSON 主配置 + CLI 覆盖；
2. Stage1/Stage2/Stage3 完整训练与早停；
3. stage1_best.pt / stage2_best.pt / stage3_best.pt + 分阶段日志导出；
4. 训练完成后自动执行患者级评估导出。
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW, Optimizer, RMSprop, SGD
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    ExponentialLR,
    MultiStepLR,
    ReduceLROnPlateau,
    StepLR,
)

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
PROJECT_ROOT = REPO_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.habitat_CBM import HabitatCBM
from srcs.data_loader_habitat_CBM import (
    ConceptScaler,
    HabitatIDHBlockDataset,
    PatientConceptDataset,
    build_habitat_cbm_dataloaders,
    build_habitat_cbm_datasets,
    concept_names_to_columns,
    load_concept_scaler,
    resolve_concept_names,
)
from srcs.eval_habitat_CBM import run_full_evaluation
from srcs.monai_augmentation import MonaiAugmentConfig, build_monai_block_transforms

TrainStage = Literal["stage1", "stage2", "stage3"]
SchedulerStepMode = Literal["none", "epoch", "metric"]
MODEL_NAME = "habitat_cbm"


@dataclass(frozen=True)
class LossBundle:
    total: torch.Tensor
    concept: torch.Tensor
    label: torch.Tensor


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _save_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _as_mapping(value: object, *, name: str) -> Dict[str, object]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"Config '{name}' must be an object.")
    return dict(value)


def _normalize_key(value: object) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def _optional_path(value: object) -> Optional[Path]:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    return Path(text)


def _get_float_pair(value: object, default: Tuple[float, float], *, name: str) -> Tuple[float, float]:
    if value is None:
        return default
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise ValueError(f"Config '{name}' must be a two-item numeric sequence.")
    return float(value[0]), float(value[1])


def _resolve_run_id(run_id: Optional[str]) -> str:
    if run_id:
        return run_id
    return time.strftime("%Y%m%d_%H%M%S")


def normalize_stage_name(stage: str) -> TrainStage:
    key = stage.strip().lower().replace("-", "").replace("_", "")
    mapping = {
        "1": "stage1",
        "stage1": "stage1",
        "xtoc": "stage1",
        "x2c": "stage1",
        "concept": "stage1",
        "2": "stage2",
        "stage2": "stage2",
        "ctoy": "stage2",
        "c2y": "stage2",
        "labelhead": "stage2",
        "3": "stage3",
        "stage3": "stage3",
        "joint": "stage3",
        "jointfinetune": "stage3",
    }
    if key not in mapping:
        raise ValueError(f"Unsupported stage: {stage}")
    return mapping[key]  # type: ignore[return-value]


def _set_requires_grad(module: nn.Module, flag: bool) -> None:
    for p in module.parameters():
        p.requires_grad = flag


_VALID_ENCODER_LAYERS = ("conv1", "bn1", "layer1", "layer2", "layer3", "layer4")


def _parse_freeze_encoder_layers(value: object) -> List[str]:
    """解析 freeze_encoder_layers 配置，返回需要冻结的 encoder 层名列表。

    支持以下格式：
    - null / None / 空列表：不冻结任何层
    - JSON 数组：["conv1", "layer1", "layer2"]
    - 逗号分隔字符串："conv1,layer1,layer2"
    可用层名：conv1、bn1、layer1、layer2、layer3、layer4。
    """
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped or stripped.lower() == "none":
            return []
        names = [n.strip() for n in stripped.split(",") if n.strip() and n.strip().lower() != "none"]
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        names = [str(n).strip() for n in value if str(n).strip() and str(n).strip().lower() != "none"]
    else:
        raise ValueError(
            f"freeze_encoder_layers must be null, a list, or a comma-separated string. Got: {value!r}"
        )

    invalid = [n for n in names if n not in _VALID_ENCODER_LAYERS]
    if invalid:
        raise ValueError(
            f"Invalid freeze_encoder_layers name(s): {invalid}. "
            f"Valid names: {list(_VALID_ENCODER_LAYERS)}"
        )
    return names


def _apply_stage1_encoder_freeze(model: HabitatCBM, freeze_layer_names: List[str]) -> None:
    """对 Stage1 encoder 中指定层执行冻结，并打印冻结摘要。"""
    _apply_encoder_freeze(model, freeze_layer_names, stage="stage1")


def _apply_encoder_freeze(model: HabitatCBM, freeze_layer_names: List[str], stage: str = "stage") -> None:
    """对 encoder 中指定层执行冻结，并打印冻结摘要。可用于任意训练阶段。"""
    if not freeze_layer_names:
        trainable = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
        print(f"[{stage}] encoder freeze: none (full fine-tune, trainable={trainable:,})")
        return

    frozen_params = 0
    for layer_name in freeze_layer_names:
        layer = getattr(model.encoder, layer_name, None)
        if layer is None:
            raise ValueError(
                f"freeze_encoder_layers: layer '{layer_name}' not found in encoder. "
                f"Valid names: {list(_VALID_ENCODER_LAYERS)}"
            )
        for p in layer.parameters():
            p.requires_grad = False
            frozen_params += p.numel()

    trainable = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    print(
        f"[{stage}] encoder freeze: {freeze_layer_names} "
        f"(frozen={frozen_params:,}, trainable={trainable:,})"
    )


def set_train_stage(model: HabitatCBM, stage: str) -> TrainStage:
    stage_name = normalize_stage_name(stage)
    if stage_name == "stage1":
        _set_requires_grad(model.encoder, True)
        _set_requires_grad(model.concept_head, True)
        _set_requires_grad(model.label_head, False)
    elif stage_name == "stage2":
        _set_requires_grad(model.encoder, False)
        _set_requires_grad(model.concept_head, False)
        _set_requires_grad(model.label_head, True)
    else:
        _set_requires_grad(model.encoder, True)
        _set_requires_grad(model.concept_head, True)
        _set_requires_grad(model.label_head, True)
    return stage_name


def _iter_trainable_params(module: nn.Module) -> List[nn.Parameter]:
    return [p for p in module.parameters() if p.requires_grad]


def get_param_groups(
    model: HabitatCBM,
    lr_encoder: float,
    lr_concept_head: float,
    lr_label_head: float,
) -> List[Dict[str, object]]:
    groups: List[Dict[str, object]] = []

    encoder_params = _iter_trainable_params(model.encoder)
    if encoder_params:
        groups.append({"params": encoder_params, "lr": lr_encoder, "name": "encoder"})

    concept_params = _iter_trainable_params(model.concept_head)
    if concept_params:
        groups.append({"params": concept_params, "lr": lr_concept_head, "name": "concept_head"})

    label_params = _iter_trainable_params(model.label_head)
    if label_params:
        groups.append({"params": label_params, "lr": lr_label_head, "name": "label_head"})

    if not groups:
        raise RuntimeError("No trainable params found.")
    return groups


def build_optimizer(
    model: HabitatCBM,
    optimizer_config: Mapping[str, object],
    train_config: Mapping[str, object],
) -> Optimizer:
    cfg = _as_mapping(optimizer_config, name="optimizer")
    base_lr = float(cfg.get("lr", train_config.get("lr", 1e-4)))
    lr_encoder = float(cfg.get("lr_encoder", train_config.get("lr_encoder", base_lr)))
    lr_concept_head = float(cfg.get("lr_concept_head", train_config.get("lr_concept_head", base_lr)))
    lr_label_head = float(cfg.get("lr_label_head", train_config.get("lr_label_head", base_lr)))
    weight_decay = float(cfg.get("weight_decay", train_config.get("weight_decay", 1e-2)))

    param_groups = get_param_groups(
        model=model,
        lr_encoder=lr_encoder,
        lr_concept_head=lr_concept_head,
        lr_label_head=lr_label_head,
    )
    name = _normalize_key(cfg.get("name", "adamw"))

    if name == "adamw":
        betas = _get_float_pair(cfg.get("betas"), (0.9, 0.999), name="optimizer.betas")
        return AdamW(
            param_groups,
            lr=base_lr,
            weight_decay=weight_decay,
            betas=betas,
            eps=float(cfg.get("eps", 1e-8)),
            amsgrad=bool(cfg.get("amsgrad", False)),
        )
    if name == "adam":
        betas = _get_float_pair(cfg.get("betas"), (0.9, 0.999), name="optimizer.betas")
        return Adam(
            param_groups,
            lr=base_lr,
            weight_decay=weight_decay,
            betas=betas,
            eps=float(cfg.get("eps", 1e-8)),
            amsgrad=bool(cfg.get("amsgrad", False)),
        )
    if name == "sgd":
        return SGD(
            param_groups,
            lr=base_lr,
            momentum=float(cfg.get("momentum", 0.9)),
            dampening=float(cfg.get("dampening", 0.0)),
            weight_decay=weight_decay,
            nesterov=bool(cfg.get("nesterov", False)),
        )
    if name == "rmsprop":
        return RMSprop(
            param_groups,
            lr=base_lr,
            alpha=float(cfg.get("alpha", 0.99)),
            eps=float(cfg.get("eps", 1e-8)),
            weight_decay=weight_decay,
            momentum=float(cfg.get("momentum", 0.0)),
            centered=bool(cfg.get("centered", False)),
        )

    raise ValueError("Unsupported optimizer. Use one of: adamw, adam, sgd, rmsprop. " f"Got: {cfg.get('name')}")


def build_scheduler(
    optimizer: Optimizer,
    scheduler_config: Mapping[str, object],
    epochs: int,
) -> Tuple[Optional[object], SchedulerStepMode]:
    cfg = _as_mapping(scheduler_config, name="scheduler")
    enabled = bool(cfg.get("enabled", True))
    name = _normalize_key(cfg.get("name", "none"))
    if not enabled or name in {"", "none", "null", "off", "disabled"}:
        return None, "none"

    if name in {"cosine", "cosine_annealing", "cosineannealing"}:
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=int(cfg.get("t_max", max(epochs, 1))),
            eta_min=float(cfg.get("eta_min", 1e-6)),
        )
        return scheduler, "epoch"

    if name in {"cosine_warm_restarts", "cosine_annealing_warm_restarts", "warm_restarts"}:
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=int(cfg.get("t_0", max(epochs, 1))),
            T_mult=int(cfg.get("t_mult", 1)),
            eta_min=float(cfg.get("eta_min", 1e-6)),
        )
        return scheduler, "epoch"

    if name in {"step", "step_lr", "steplr"}:
        scheduler = StepLR(
            optimizer,
            step_size=int(cfg.get("step_size", max(epochs // 3, 1))),
            gamma=float(cfg.get("gamma", 0.1)),
        )
        return scheduler, "epoch"

    if name in {"multistep", "multi_step", "multistep_lr", "multi_step_lr"}:
        milestones_raw = cfg.get("milestones", [max(epochs // 2, 1), max((epochs * 3) // 4, 1)])
        if not isinstance(milestones_raw, Sequence) or isinstance(milestones_raw, (str, bytes)):
            raise ValueError("Config 'scheduler.milestones' must be a numeric sequence.")
        milestones = [int(item) for item in milestones_raw]
        scheduler = MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=float(cfg.get("gamma", 0.1)),
        )
        return scheduler, "epoch"

    if name in {"exponential", "exponential_lr", "explr"}:
        scheduler = ExponentialLR(
            optimizer,
            gamma=float(cfg.get("gamma", 0.95)),
        )
        return scheduler, "epoch"

    if name in {"plateau", "reduce_on_plateau", "reduce_lr_on_plateau"}:
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode=str(cfg.get("mode", "max")),
            factor=float(cfg.get("factor", 0.5)),
            patience=int(cfg.get("patience", 5)),
            threshold=float(cfg.get("threshold", 1e-4)),
            min_lr=float(cfg.get("min_lr", 1e-6)),
        )
        return scheduler, "metric"

    raise ValueError(
        "Unsupported scheduler. Use one of: none, cosine_annealing, "
        "cosine_annealing_warm_restarts, step, multistep, exponential, reduce_on_plateau. "
        f"Got: {cfg.get('name')}"
    )


def _get_current_lrs(optimizer: Optimizer) -> Dict[str, float]:
    lrs: Dict[str, float] = {}
    for idx, group in enumerate(optimizer.param_groups):
        name = str(group.get("name", f"group{idx}"))
        lrs[f"lr_{name}"] = float(group["lr"])
    return lrs


def _validate_concept_tensor(c: torch.Tensor, n_concepts: int, name: str) -> None:
    if c.ndim != 2:
        raise ValueError(f"{name} must be [B,{n_concepts}], got {tuple(c.shape)}")
    if c.shape[1] != n_concepts:
        raise ValueError(f"{name} dim mismatch: expected {n_concepts}, got {c.shape[1]}")


def _reshape_binary_target(y_true: torch.Tensor) -> torch.Tensor:
    if y_true.ndim == 1:
        y = y_true.unsqueeze(1)
    elif y_true.ndim == 2 and y_true.shape[1] == 1:
        y = y_true
    else:
        raise ValueError(f"y_true must be [B] or [B,1], got {tuple(y_true.shape)}")
    return y.float()


def compute_concept_loss(
    model: HabitatCBM,
    c_hat: torch.Tensor,
    c_true_std: torch.Tensor,
    loss_config: Optional[Mapping[str, object]] = None,
) -> torch.Tensor:
    _validate_concept_tensor(c_hat, model.n_concepts, "c_hat")
    _validate_concept_tensor(c_true_std, model.n_concepts, "c_true_std")
    if c_hat.shape != c_true_std.shape:
        raise ValueError(f"c_hat/c_true_std shape mismatch: {tuple(c_hat.shape)} vs {tuple(c_true_std.shape)}")

    cfg = _as_mapping(loss_config, name="loss.concept")
    loss_name = _normalize_key(cfg.get("name", "mse"))
    reduction = str(cfg.get("reduction", "mean"))

    if loss_name in {"mse", "l2", "mse_loss"}:
        return F.mse_loss(c_hat, c_true_std, reduction=reduction)
    if loss_name in {"mae", "l1", "l1_loss"}:
        return F.l1_loss(c_hat, c_true_std, reduction=reduction)
    if loss_name in {"smooth_l1", "smooth_l1_loss", "huber", "huber_loss"}:
        beta = float(cfg.get("beta", cfg.get("delta", 1.0)))
        return F.smooth_l1_loss(c_hat, c_true_std, beta=beta, reduction=reduction)

    raise ValueError(
        "Unsupported concept loss. Use one of: mse, l1/mae, smooth_l1/huber. "
        f"Got: {cfg.get('name')}"
    )


def compute_label_loss(
    y_logit: torch.Tensor,
    y_true: torch.Tensor,
    pos_weight: Optional[torch.Tensor] = None,
    loss_config: Optional[Mapping[str, object]] = None,
) -> torch.Tensor:
    if y_logit.ndim != 2 or y_logit.shape[1] != 1:
        raise ValueError(f"y_logit must be [B,1], got {tuple(y_logit.shape)}")
    cfg = _as_mapping(loss_config, name="loss.label")
    y_target = _reshape_binary_target(y_true).to(device=y_logit.device, dtype=y_logit.dtype)
    if y_target.shape[0] != y_logit.shape[0]:
        raise ValueError(
            f"Batch mismatch: y_logit={y_logit.shape[0]} vs y_true={y_target.shape[0]}"
        )

    label_smoothing = float(cfg.get("label_smoothing", 0.0))
    if not (0.0 <= label_smoothing < 1.0):
        raise ValueError(f"label_smoothing must be in [0, 1), got {label_smoothing}")
    if label_smoothing > 0.0:
        y_target = y_target * (1.0 - label_smoothing) + 0.5 * label_smoothing

    use_pos_weight = bool(cfg.get("use_pos_weight", True))
    if pos_weight is not None and use_pos_weight:
        if pos_weight.ndim == 0:
            pw = pos_weight.reshape(1)
        elif pos_weight.ndim == 1 and pos_weight.numel() == 1:
            pw = pos_weight
        else:
            raise ValueError(f"pos_weight must be scalar or [1], got {tuple(pos_weight.shape)}")
        pw = pw.to(device=y_logit.device, dtype=y_logit.dtype)
    else:
        pw = None

    loss_name = _normalize_key(cfg.get("name", "bce_with_logits"))
    reduction = str(cfg.get("reduction", "mean"))
    if loss_name in {"bce", "bce_with_logits", "binary_cross_entropy_with_logits"}:
        return F.binary_cross_entropy_with_logits(
            y_logit,
            y_target,
            pos_weight=pw,
            reduction=reduction,
        )

    if loss_name in {"focal", "focal_bce", "focal_bce_with_logits"}:
        gamma = float(cfg.get("gamma", 2.0))
        alpha_value = cfg.get("alpha", None)
        bce = F.binary_cross_entropy_with_logits(
            y_logit,
            y_target,
            pos_weight=pw,
            reduction="none",
        )
        prob = torch.sigmoid(y_logit)
        p_t = prob * y_target + (1.0 - prob) * (1.0 - y_target)
        focal_factor = (1.0 - p_t).clamp_min(0.0).pow(gamma)
        loss = focal_factor * bce
        if alpha_value is not None:
            alpha = float(alpha_value)
            if not (0.0 <= alpha <= 1.0):
                raise ValueError(f"focal alpha must be in [0, 1], got {alpha}")
            alpha_t = alpha * y_target + (1.0 - alpha) * (1.0 - y_target)
            loss = alpha_t * loss
        if reduction == "mean":
            return loss.mean()
        if reduction == "sum":
            return loss.sum()
        if reduction == "none":
            return loss
        raise ValueError(f"Unsupported focal reduction: {reduction}")

    raise ValueError(
        "Unsupported label loss. Use one of: bce_with_logits, focal_bce_with_logits. "
        f"Got: {cfg.get('name')}"
    )


def compute_joint_loss(
    model: HabitatCBM,
    c_hat: torch.Tensor,
    c_true_std: torch.Tensor,
    y_logit: torch.Tensor,
    y_true: torch.Tensor,
    lambda_c: float,
    lambda_y: float,
    pos_weight: Optional[torch.Tensor] = None,
    concept_loss_config: Optional[Mapping[str, object]] = None,
    label_loss_config: Optional[Mapping[str, object]] = None,
) -> LossBundle:
    if lambda_c < 0 or lambda_y < 0:
        raise ValueError(f"lambda_c/lambda_y must be non-negative, got {lambda_c}/{lambda_y}")
    c_loss = compute_concept_loss(
        model,
        c_hat,
        c_true_std,
        loss_config=concept_loss_config,
    )
    y_loss = compute_label_loss(
        y_logit,
        y_true,
        pos_weight=pos_weight,
        loss_config=label_loss_config,
    )
    total = lambda_c * c_loss + lambda_y * y_loss
    return LossBundle(total=total, concept=c_loss, label=y_loss)


def stage1_train_step(
    model: HabitatCBM,
    x: torch.Tensor,
    c_true_std: torch.Tensor,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """兼容旧接口：Stage1 单步前向与概念损失。"""
    out = model.forward_x_to_cy(x)
    loss_c = compute_concept_loss(model=model, c_hat=out["c_hat"], c_true_std=c_true_std)
    return out, loss_c


def stage2_train_step(
    model: HabitatCBM,
    c_true_std: torch.Tensor,
    y_true: torch.Tensor,
    pos_weight: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """兼容旧接口：Stage2 单步前向与标签损失。"""
    y_logit = model.forward_c_to_y(c_true_std)
    loss_y = compute_label_loss(y_logit=y_logit, y_true=y_true, pos_weight=pos_weight)
    return y_logit, loss_y


def stage3_train_step(
    model: HabitatCBM,
    x: torch.Tensor,
    c_true_std: torch.Tensor,
    y_true: torch.Tensor,
    lambda_c: float = 0.5,
    lambda_y: float = 1.0,
    pos_weight: Optional[torch.Tensor] = None,
) -> Tuple[Dict[str, torch.Tensor], LossBundle]:
    """兼容旧接口：Stage3 单步前向与联合损失。"""
    out = model.forward_x_to_cy(x)
    losses = compute_joint_loss(
        model=model,
        c_hat=out["c_hat"],
        c_true_std=c_true_std,
        y_logit=out["y_logit"],
        y_true=y_true,
        lambda_c=lambda_c,
        lambda_y=lambda_y,
        pos_weight=pos_weight,
    )
    return out, losses


def _extract_loss_configs(cfg: Mapping[str, object], train_cfg: Mapping[str, object]) -> Tuple[Dict[str, object], Dict[str, object], Dict[str, object]]:
    loss_cfg = _as_mapping(cfg.get("loss", {}), name="loss")
    concept_loss_cfg = _as_mapping(loss_cfg.get("concept", {}), name="loss.concept")
    label_loss_cfg = _as_mapping(loss_cfg.get("label", {}), name="loss.label")
    joint_loss_cfg = _as_mapping(loss_cfg.get("joint", {}), name="loss.joint")

    # Backward compatibility with the previous flat train config.
    if "lambda_c" not in joint_loss_cfg and "lambda_c" in train_cfg:
        joint_loss_cfg["lambda_c"] = train_cfg["lambda_c"]
    if "lambda_y" not in joint_loss_cfg and "lambda_y" in train_cfg:
        joint_loss_cfg["lambda_y"] = train_cfg["lambda_y"]

    concept_loss_cfg.setdefault("name", "mse")
    concept_loss_cfg.setdefault("reduction", "mean")
    label_loss_cfg.setdefault("name", "bce_with_logits")
    label_loss_cfg.setdefault("reduction", "mean")
    label_loss_cfg.setdefault("use_pos_weight", True)
    joint_loss_cfg.setdefault("lambda_c", 0.5)
    joint_loss_cfg.setdefault("lambda_y", 1.0)

    return concept_loss_cfg, label_loss_cfg, joint_loss_cfg


def _compute_input_channels(data_cfg: Mapping[str, object]) -> int:
    modalities = tuple(data_cfg.get("modalities", ("t1", "t1ce", "t2", "t2flair", "adc", "cbf")))
    block_depth = int(data_cfg.get("block_depth", 5))
    append_voi_mask = bool(data_cfg.get("append_voi_mask", True))
    return len(modalities) * block_depth + (block_depth if append_voi_mask else 0)


def _resolve_model_dropouts(model_cfg: Mapping[str, object]) -> Tuple[float, float]:
    shared_dropout = model_cfg.get("dropout_p", None)
    if shared_dropout is not None:
        shared = float(shared_dropout)
        concept_dropout = float(model_cfg.get("concept_dropout_p", shared))
        label_dropout = float(model_cfg.get("label_dropout_p", shared))
    else:
        concept_dropout = float(model_cfg.get("concept_dropout_p", 0.3))
        label_dropout = float(model_cfg.get("label_dropout_p", 0.1))
    return concept_dropout, label_dropout


def _compute_pos_weight_from_train_patients(
    dataset: HabitatIDHBlockDataset,
    device: torch.device,
    label_loss_config: Mapping[str, object],
) -> Optional[torch.Tensor]:
    use_pos_weight = bool(label_loss_config.get("use_pos_weight", True))
    if not use_pos_weight:
        return None

    manual_value = label_loss_config.get("manual_pos_weight", None)
    if manual_value is not None:
        value = float(manual_value)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"manual_pos_weight must be a positive finite float, got {manual_value}")
        return torch.tensor([value], dtype=torch.float32, device=device)

    positive = 0
    negative = 0
    for case in dataset.patient_cases.values():
        if int(case.label_id) == 1:
            positive += 1
        else:
            negative += 1

    if positive <= 0:
        raise ValueError("Train split has no positive class; pos_weight cannot be computed.")
    if negative <= 0:
        raise ValueError("Train split has no negative class; pos_weight cannot be computed.")

    # BCEWithLogits pos_weight = N_negative / N_positive
    value = float(negative / positive)
    return torch.tensor([value], dtype=torch.float32, device=device)


def _build_stage2_patient_dataloaders(
    datasets: Mapping[str, HabitatIDHBlockDataset],
    batch_size: int,
    num_workers: int,
) -> Dict[str, DataLoader]:
    return {
        "train": DataLoader(
            PatientConceptDataset(datasets["train"]),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        ),
        "val": DataLoader(
            PatientConceptDataset(datasets["val"]),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        ),
    }


def _resolve_stage_label_loss_config(
    base_label_loss_cfg: Mapping[str, object],
    stage_cfg: Mapping[str, object],
    stage: TrainStage,
) -> Dict[str, object]:
    stage_override = _as_mapping(stage_cfg.get("label_loss", {}), name=f"stages.{stage}.label_loss")
    resolved = {**base_label_loss_cfg, **stage_override}
    resolved.setdefault("name", "bce_with_logits")
    resolved.setdefault("reduction", "mean")
    resolved.setdefault("use_pos_weight", True)
    return resolved


def _resolve_stage_monitor_metric(
    stage_cfg: Mapping[str, object],
    stage: TrainStage,
) -> Optional[str]:
    if stage != "stage2":
        return None

    metric = _normalize_key(stage_cfg.get("monitor_metric", "label_loss"))
    valid = {"label_loss", "auc"}
    if metric not in valid:
        raise ValueError(
            f"Unsupported stages.{stage}.monitor_metric: {metric}. "
            f"Use one of {sorted(valid)}."
        )
    return metric


def _extract_loader_labels(dataloader) -> Optional[np.ndarray]:
    dataset = getattr(dataloader, "dataset", None)
    if isinstance(dataset, PatientConceptDataset):
        labels = [
            int(dataset.block_dataset.patient_cases[patient_id].label_id)
            for patient_id in dataset.patient_ids
        ]
        return np.asarray(labels, dtype=np.int64)

    if hasattr(dataset, "sample_index") and hasattr(dataset, "patient_cases"):
        labels = [
            int(dataset.patient_cases[item.patient_id].label_id)
            for item in dataset.sample_index
        ]
        return np.asarray(labels, dtype=np.int64)
    return None


def _summarize_loader_sampling(dataloader) -> Dict[str, object]:
    sampler = getattr(dataloader, "sampler", None)
    sampler_type = type(sampler).__name__ if sampler is not None else "None"
    labels = _extract_loader_labels(dataloader)
    pos_ratio = float("nan")
    neg_ratio = float("nan")

    if labels is not None and labels.size > 0:
        weights_raw = getattr(sampler, "weights", None)
        if weights_raw is not None:
            weights = torch.as_tensor(weights_raw, dtype=torch.float64).detach().cpu().numpy()
            if weights.shape[0] == labels.shape[0]:
                weight_sum = float(weights.sum())
                if weight_sum > 0.0:
                    pos_ratio = float(weights[labels == 1].sum() / weight_sum)
                    neg_ratio = float(weights[labels == 0].sum() / weight_sum)

        if not np.isfinite(pos_ratio) or not np.isfinite(neg_ratio):
            total = float(labels.size)
            pos_ratio = float((labels == 1).sum() / total)
            neg_ratio = float((labels == 0).sum() / total)

    return {
        "sampler_type": sampler_type,
        "pos_ratio": pos_ratio,
        "neg_ratio": neg_ratio,
    }


def _safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y_true, y_prob))


def find_youden_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """在 val 集上用 Youden Index (Sensitivity + Specificity - 1) 最大化选取最优阈值。

    Args:
        y_true: 真实标签 (0/1)，shape [N]
        y_prob: 预测概率，shape [N]

    Returns:
        使 Youden Index 最大的阈值；如果类别少于 2 则回退到 0.5。
    """
    if len(np.unique(y_true)) < 2:
        return 0.5

    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    # Youden J = Sensitivity + Specificity - 1 = tpr + (1 - fpr) - 1 = tpr - fpr
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    best_threshold = float(thresholds[best_idx])
    # 限制到合理范围，避免 roc_curve 返回 threshold > 1 的边界值
    best_threshold = float(np.clip(best_threshold, 1e-6, 1.0 - 1e-6))
    return best_threshold


def collect_val_patient_probs(
    model: HabitatCBM,
    val_loader,
    device: torch.device,
    topk_pool: int,
    threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """在 val 集上收集患者级预测概率，用于 Youden Index 阈值选取。

    Returns:
        (y_true_patient, y_prob_patient) 两个 numpy 数组。
    """
    model.eval()
    patient_ids: List[str] = []
    probs: List[float] = []
    y_true_blocks: List[int] = []

    with torch.no_grad():
        for batch in val_loader:
            x = batch["image"].to(device=device, dtype=torch.float32, non_blocking=True)
            y_true = batch["label"].to(device=device, dtype=torch.float32, non_blocking=True)
            out = model.forward_x_to_cy(x)
            y_prob = torch.sigmoid(out["y_logit"]).squeeze(1)

            batch_size = int(x.shape[0])
            batch_patient_ids = batch["patient_id"]
            for idx in range(batch_size):
                patient_ids.append(str(batch_patient_ids[idx]))
                probs.append(float(y_prob[idx].item()))
                y_true_blocks.append(int(y_true[idx].item()))

    y_true_p, y_prob_p, _ = _patient_level_probs(
        patient_ids=patient_ids,
        probs=probs,
        y_trues=y_true_blocks,
        topk_pool=topk_pool,
        threshold=threshold,
    )
    return y_true_p, y_prob_p


def _patient_level_probs(
    patient_ids: List[str],
    probs: List[float],
    y_trues: List[int],
    topk_pool: int,
    threshold: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    buckets: Dict[str, List[Tuple[float, int]]] = {}
    for pid, prob, y_true in zip(patient_ids, probs, y_trues):
        buckets.setdefault(pid, []).append((float(prob), int(y_true)))

    y_true_patient: List[int] = []
    y_prob_patient: List[float] = []
    y_pred_patient: List[int] = []

    for pid in sorted(buckets.keys()):
        items = buckets[pid]
        gt_values = {item[1] for item in items}
        if len(gt_values) != 1:
            raise ValueError(f"Inconsistent y_true across blocks for patient {pid}")

        if topk_pool > 0 and len(items) > topk_pool:
            items = sorted(items, key=lambda item: abs(item[0] - 0.5), reverse=True)[:topk_pool]

        prob_mean = float(np.mean([item[0] for item in items]))
        y_true_i = int(next(iter(gt_values)))
        y_pred_i = int(prob_mean >= threshold)

        y_true_patient.append(y_true_i)
        y_prob_patient.append(prob_mean)
        y_pred_patient.append(y_pred_i)

    return (
        np.asarray(y_true_patient, dtype=np.int64),
        np.asarray(y_prob_patient, dtype=np.float64),
        np.asarray(y_pred_patient, dtype=np.int64),
    )


def _compute_patient_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

    auc = _safe_auc(y_true, y_prob)
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sen = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    spe = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    return {
        "auc": auc,
        "acc": acc,
        "f1": f1,
        "sen": sen,
        "spe": spe,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def _build_transforms(data_cfg: Mapping[str, object], train_cfg: Mapping[str, object]) -> Dict[str, object]:
    has_mask = bool(data_cfg.get("require_voi", True)) or bool(data_cfg.get("append_voi_mask", True)) or bool(
        data_cfg.get("mask_background_with_voi", False)
    )

    aug_cfg = MonaiAugmentConfig(
        enabled=bool(train_cfg.get("use_monai_augmentation", True)),
        affine_prob=float(train_cfg.get("aug_affine_prob", 0.9)),
        rotate_deg=float(train_cfg.get("aug_rotate_deg", 30.0)),
        translate_px=float(train_cfg.get("aug_translate_px", 20.0)),
        scale_range=float(train_cfg.get("aug_scale_range", 0.25)),
        flip_prob=float(train_cfg.get("aug_flip_prob", 0.5)),
        intensity_scale_prob=float(train_cfg.get("aug_intensity_scale_prob", 0.5)),
        intensity_scale=float(train_cfg.get("aug_intensity_scale", 0.25)),
        intensity_shift_prob=float(train_cfg.get("aug_intensity_shift_prob", 0.5)),
        intensity_shift=float(train_cfg.get("aug_intensity_shift", 0.25)),
        gaussian_noise_prob=float(train_cfg.get("aug_gaussian_noise_prob", 0.5)),
        gaussian_noise_std=float(train_cfg.get("aug_gaussian_noise_std", 0.05)),
        gibbs_noise_prob=float(train_cfg.get("aug_gibbs_noise_prob", 0.3)),
        gibbs_noise_alpha=float(train_cfg.get("aug_gibbs_noise_alpha", 0.5)),
    )

    return build_monai_block_transforms(
        config=aug_cfg,
        has_mask=has_mask,
        spatial_size=(int(data_cfg.get("resize_height", 224)), int(data_cfg.get("resize_width", 224))),
    )


def _load_json(path: Path) -> Dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Config must be a JSON object.")
    return payload


def _apply_cli_overrides(cfg: Dict[str, object], args: argparse.Namespace) -> Dict[str, object]:
    paths_cfg = dict(cfg.get("paths", {}))
    train_cfg = dict(cfg.get("train", {}))
    logging_cfg = dict(cfg.get("logging", {}))
    stages_cfg = _as_mapping(cfg.get("stages", {}), name="stages")

    if args.run_id is not None:
        logging_cfg["run_id"] = args.run_id
    if args.output_root is not None:
        paths_cfg["runs_root"] = str(args.output_root)
        paths_cfg["results_root"] = str(args.output_root)
    if args.checkpoint_root is not None:
        paths_cfg["checkpoint_root"] = str(args.checkpoint_root)
    if args.device is not None:
        train_cfg["device"] = args.device
    if args.epochs_stage1 is not None:
        train_cfg["epochs_stage1"] = int(args.epochs_stage1)
        stage1_cfg = _as_mapping(stages_cfg.get("stage1", {}), name="stages.stage1")
        stage1_cfg["epochs"] = int(args.epochs_stage1)
        stages_cfg["stage1"] = stage1_cfg
    if args.epochs_stage2 is not None:
        train_cfg["epochs_stage2"] = int(args.epochs_stage2)
        stage2_cfg = _as_mapping(stages_cfg.get("stage2", {}), name="stages.stage2")
        stage2_cfg["epochs"] = int(args.epochs_stage2)
        stages_cfg["stage2"] = stage2_cfg
    if args.epochs_stage3 is not None:
        train_cfg["epochs_stage3"] = int(args.epochs_stage3)
        stage3_cfg = _as_mapping(stages_cfg.get("stage3", {}), name="stages.stage3")
        stage3_cfg["epochs"] = int(args.epochs_stage3)
        stages_cfg["stage3"] = stage3_cfg
    if args.batch_size is not None:
        train_cfg["batch_size"] = int(args.batch_size)

    cfg = dict(cfg)
    cfg["paths"] = paths_cfg
    cfg["train"] = train_cfg
    cfg["logging"] = logging_cfg
    cfg["stages"] = stages_cfg
    return cfg


def _validate_config(cfg: Mapping[str, object]) -> None:
    required_sections = {"paths", "data", "model", "train", "eval", "intervention", "logging"}
    missing = sorted(required_sections - set(cfg.keys()))
    if missing:
        raise ValueError(f"Config missing sections: {missing}")

    paths_cfg = cfg["paths"]
    if not isinstance(paths_cfg, Mapping):
        raise ValueError("Config 'paths' must be object")
    required_paths = {"split_base_root", "concept_label_csv", "concept_scaler_json", "runs_root", "results_root"}
    missing_paths = sorted(required_paths - set(paths_cfg.keys()))
    if missing_paths:
        raise ValueError(f"Config paths missing keys: {missing_paths}")


def _resolve_child_path(base_dir: Path, value: object, default_name: str) -> Path:
    path = _optional_path(value)
    if path is None:
        return base_dir / default_name
    if path.is_absolute():
        return path
    return base_dir / path


def _build_stage_configs(
    cfg: Mapping[str, object],
    train_cfg: Mapping[str, object],
    checkpoint_root: Path,
    run_dir: Path,
) -> List[Tuple[TrainStage, int, Path, Path, Dict[str, object]]]:
    stages_cfg = _as_mapping(cfg.get("stages", {}), name="stages")
    defaults: Dict[TrainStage, Tuple[str, str, str]] = {
        "stage1": ("epochs_stage1", "stage1_best.pt", "stage1_concept_log.csv"),
        "stage2": ("epochs_stage2", "stage2_best.pt", "stage2_label_head_log.csv"),
        "stage3": ("epochs_stage3", "stage3_best.pt", "stage3_joint_log.csv"),
    }

    stage_items: List[Tuple[TrainStage, int, Path, Path, Dict[str, object]]] = []
    for stage in ("stage1", "stage2", "stage3"):
        stage_cfg = _as_mapping(stages_cfg.get(stage, {}), name=f"stages.{stage}")
        epochs_key, default_ckpt, default_log = defaults[stage]
        epochs = int(stage_cfg.get("epochs", train_cfg.get(epochs_key, 20 if stage != "stage3" else 30)))
        checkpoint_path = _resolve_child_path(
            checkpoint_root,
            stage_cfg.get("checkpoint_path", stage_cfg.get("checkpoint_name")),
            default_ckpt,
        )
        log_path = _resolve_child_path(
            run_dir,
            stage_cfg.get("log_csv_path", stage_cfg.get("log_csv")),
            default_log,
        )
        stage_items.append((stage, epochs, checkpoint_path, log_path, stage_cfg))

    return stage_items


def _save_checkpoint(
    path: Path,
    epoch: int,
    stage: TrainStage,
    model: HabitatCBM,
    optimizer: Optimizer,
    scheduler: object,
    best_score: float,
    run_id: str,
    model_config: Mapping[str, object],
    train_config: Mapping[str, object],
    optimizer_config: Mapping[str, object],
    scheduler_config: Mapping[str, object],
    loss_config: Mapping[str, object],
    monitor_name: str,
) -> None:
    scheduler_state = None
    if scheduler is not None and hasattr(scheduler, "state_dict"):
        scheduler_state = scheduler.state_dict()

    payload = {
        "epoch": int(epoch),
        "stage": stage,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler_state,
        "best_score": float(best_score),
        "run_id": run_id,
        "model_config": dict(model_config),
        "train_config": dict(train_config),
        "optimizer_config": dict(optimizer_config),
        "scheduler_config": dict(scheduler_config),
        "loss_config": dict(loss_config),
        "monitor_name": monitor_name,
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _load_checkpoint_model_only(model: HabitatCBM, checkpoint_path: Path, device: torch.device) -> Dict[str, object]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")
    state_dict = payload.get("model_state_dict")
    if state_dict is None:
        raise ValueError(f"Checkpoint missing model_state_dict: {checkpoint_path}")
    model.load_state_dict(state_dict, strict=True)
    return dict(payload)


def _train_stage1_epoch(
    model: HabitatCBM,
    dataloader,
    optimizer: Optimizer,
    device: torch.device,
    concept_loss_config: Mapping[str, object],
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_count = 0

    for batch in dataloader:
        x = batch["image"].to(device=device, dtype=torch.float32, non_blocking=True)
        c_true_std = batch["concept_true_std"].to(device=device, dtype=torch.float32, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        out = model.forward_x_to_cy(x)
        loss = compute_concept_loss(
            model=model,
            c_hat=out["c_hat"],
            c_true_std=c_true_std,
            loss_config=concept_loss_config,
        )
        loss.backward()
        optimizer.step()

        batch_size = int(x.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_count += batch_size

    return {"concept_loss": total_loss / max(total_count, 1)}


def _eval_stage1_epoch(
    model: HabitatCBM,
    dataloader,
    device: torch.device,
    concept_loss_config: Mapping[str, object],
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_count = 0

    with torch.no_grad():
        for batch in dataloader:
            x = batch["image"].to(device=device, dtype=torch.float32, non_blocking=True)
            c_true_std = batch["concept_true_std"].to(device=device, dtype=torch.float32, non_blocking=True)
            out = model.forward_x_to_cy(x)
            loss = compute_concept_loss(
                model=model,
                c_hat=out["c_hat"],
                c_true_std=c_true_std,
                loss_config=concept_loss_config,
            )

            batch_size = int(x.shape[0])
            total_loss += float(loss.item()) * batch_size
            total_count += batch_size

    return {"concept_loss": total_loss / max(total_count, 1)}


def _train_stage2_epoch(
    model: HabitatCBM,
    dataloader,
    optimizer: Optimizer,
    device: torch.device,
    pos_weight: Optional[torch.Tensor],
    concept_noise_std: float,
    concept_noise_std_vector: Optional[torch.Tensor],
    label_loss_config: Mapping[str, object],
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_count = 0

    for batch in dataloader:
        c_true_std = batch["concept_true_std"].to(device=device, dtype=torch.float32, non_blocking=True)
        y_true = batch["label"].to(device=device, dtype=torch.float32, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if concept_noise_std_vector is not None:
            noise_std = concept_noise_std_vector.to(device=device, dtype=c_true_std.dtype).view(1, -1)
            c_input = c_true_std + torch.randn_like(c_true_std) * noise_std
        elif concept_noise_std > 0.0:
            c_input = c_true_std + torch.randn_like(c_true_std) * float(concept_noise_std)
        else:
            c_input = c_true_std
        y_logit = model.forward_c_to_y(c_input)
        loss = compute_label_loss(
            y_logit=y_logit,
            y_true=y_true,
            pos_weight=pos_weight,
            loss_config=label_loss_config,
        )
        loss.backward()
        optimizer.step()

        batch_size = int(c_true_std.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_count += batch_size

    return {"label_loss": total_loss / max(total_count, 1)}


@torch.no_grad()
def _estimate_stage2_residual_noise_std(
    model: HabitatCBM,
    dataloader,
    device: torch.device,
    base_noise_std: float,
    eval_transform: object = None,
    min_noise_std: float = 0.0,
    max_noise_std: Optional[float] = None,
) -> Tuple[Optional[torch.Tensor], Dict[str, object]]:
    """Estimate per-concept Stage2 noise from Stage1 train residuals.

    Residuals are computed at patient level:
        mean_block(c_hat_stage1) - c_true_std

    The returned noise vector preserves the average configured noise strength
    (`base_noise_std`) while scaling each concept by its residual std ratio.
    """

    if base_noise_std <= 0.0:
        return None, {
            "mode": "disabled",
            "base_noise_std": float(base_noise_std),
            "num_patients": 0,
        }
    if min_noise_std < 0.0:
        raise ValueError(f"min_noise_std must be >= 0, got {min_noise_std}")
    if max_noise_std is not None and max_noise_std <= 0.0:
        raise ValueError(f"max_noise_std must be positive when set, got {max_noise_std}")
    if max_noise_std is not None and max_noise_std < min_noise_std:
        raise ValueError(
            f"max_noise_std must be >= min_noise_std, got {max_noise_std} < {min_noise_std}"
        )

    dataset = getattr(dataloader, "dataset", None)
    original_transform = getattr(dataset, "transform", None) if dataset is not None else None
    swapped_transform = dataset is not None and hasattr(dataset, "transform") and eval_transform is not None

    pred_sums: Dict[str, torch.Tensor] = {}
    true_by_patient: Dict[str, torch.Tensor] = {}
    counts: Dict[str, int] = {}

    was_training = model.training
    model.eval()
    if swapped_transform:
        dataset.transform = eval_transform
    try:
        for batch in dataloader:
            x = batch["image"].to(device=device, dtype=torch.float32, non_blocking=True)
            c_true_std = batch["concept_true_std"].to(device=device, dtype=torch.float32, non_blocking=True)
            out = model.forward_x_to_cy(x)
            c_hat = out["c_hat"].detach().cpu()
            c_true_cpu = c_true_std.detach().cpu()
            patient_ids = batch["patient_id"]

            for idx in range(int(c_hat.shape[0])):
                patient_id = str(patient_ids[idx])
                if patient_id not in pred_sums:
                    pred_sums[patient_id] = c_hat[idx].clone()
                    true_by_patient[patient_id] = c_true_cpu[idx].clone()
                    counts[patient_id] = 1
                else:
                    pred_sums[patient_id] = pred_sums[patient_id] + c_hat[idx]
                    counts[patient_id] += 1
    finally:
        if swapped_transform:
            dataset.transform = original_transform
        if was_training:
            model.train()

    if len(pred_sums) < 2:
        raise RuntimeError(
            "Need at least two patients to estimate residual-aware concept noise, "
            f"got {len(pred_sums)}."
        )

    residual_rows: List[torch.Tensor] = []
    for patient_id in sorted(pred_sums.keys()):
        pred_mean = pred_sums[patient_id] / float(counts[patient_id])
        residual_rows.append(pred_mean - true_by_patient[patient_id])

    residuals = torch.stack(residual_rows, dim=0).float()
    residual_mean = residuals.mean(dim=0)
    residual_std = residuals.std(dim=0, unbiased=False)
    residual_std_mean = residual_std.mean().clamp_min(1e-8)

    noise_std = residual_std / residual_std_mean * float(base_noise_std)
    if min_noise_std > 0.0:
        noise_std = torch.clamp(noise_std, min=float(min_noise_std))
    if max_noise_std is not None:
        noise_std = torch.clamp(noise_std, max=float(max_noise_std))

    summary = {
        "mode": "residual",
        "base_noise_std": float(base_noise_std),
        "min_noise_std": float(min_noise_std),
        "max_noise_std": float(max_noise_std) if max_noise_std is not None else None,
        "num_patients": int(len(pred_sums)),
        "residual_mean": [float(v) for v in residual_mean.tolist()],
        "residual_std": [float(v) for v in residual_std.tolist()],
        "noise_std": [float(v) for v in noise_std.tolist()],
        "noise_std_mean": float(noise_std.mean().item()),
        "noise_std_min": float(noise_std.min().item()),
        "noise_std_max": float(noise_std.max().item()),
    }
    return noise_std.to(device=device, dtype=torch.float32), summary


def _eval_stage2_epoch(
    model: HabitatCBM,
    dataloader,
    device: torch.device,
    pos_weight: Optional[torch.Tensor],
    topk_pool: int,
    threshold: float,
    label_loss_config: Mapping[str, object],
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_count = 0

    patient_ids: List[str] = []
    probs: List[float] = []
    y_true_blocks: List[int] = []

    with torch.no_grad():
        for batch in dataloader:
            c_true_std = batch["concept_true_std"].to(device=device, dtype=torch.float32, non_blocking=True)
            y_true = batch["label"].to(device=device, dtype=torch.float32, non_blocking=True)
            y_logit = model.forward_c_to_y(c_true_std)
            y_prob = torch.sigmoid(y_logit).squeeze(1)

            loss = compute_label_loss(
                y_logit=y_logit,
                y_true=y_true,
                pos_weight=pos_weight,
                loss_config=label_loss_config,
            )
            batch_size = int(c_true_std.shape[0])
            total_loss += float(loss.item()) * batch_size
            total_count += batch_size

            batch_patient_ids = batch["patient_id"]
            for idx in range(batch_size):
                patient_ids.append(str(batch_patient_ids[idx]))
                probs.append(float(y_prob[idx].item()))
                y_true_blocks.append(int(y_true[idx].item()))

    y_true_p, y_prob_p, y_pred_p = _patient_level_probs(
        patient_ids=patient_ids,
        probs=probs,
        y_trues=y_true_blocks,
        topk_pool=topk_pool,
        threshold=threshold,
    )
    metrics = _compute_patient_metrics(y_true=y_true_p, y_prob=y_prob_p, y_pred=y_pred_p)
    return {
        "label_loss": total_loss / max(total_count, 1),
        "auc": metrics["auc"],
        "acc": metrics["acc"],
        "f1": metrics["f1"],
        "sen": metrics["sen"],
        "spe": metrics["spe"],
    }


def _train_stage3_epoch(
    model: HabitatCBM,
    dataloader,
    optimizer: Optimizer,
    device: torch.device,
    lambda_c: float,
    lambda_y: float,
    pos_weight: Optional[torch.Tensor],
    concept_loss_config: Mapping[str, object],
    label_loss_config: Mapping[str, object],
) -> Dict[str, float]:
    model.train()
    total = 0.0
    total_c = 0.0
    total_y = 0.0
    total_count = 0

    for batch in dataloader:
        x = batch["image"].to(device=device, dtype=torch.float32, non_blocking=True)
        c_true_std = batch["concept_true_std"].to(device=device, dtype=torch.float32, non_blocking=True)
        y_true = batch["label"].to(device=device, dtype=torch.float32, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        out = model.forward_x_to_cy(x)
        losses = compute_joint_loss(
            model=model,
            c_hat=out["c_hat"],
            c_true_std=c_true_std,
            y_logit=out["y_logit"],
            y_true=y_true,
            lambda_c=lambda_c,
            lambda_y=lambda_y,
            pos_weight=pos_weight,
            concept_loss_config=concept_loss_config,
            label_loss_config=label_loss_config,
        )
        losses.total.backward()
        optimizer.step()

        batch_size = int(x.shape[0])
        total += float(losses.total.item()) * batch_size
        total_c += float(losses.concept.item()) * batch_size
        total_y += float(losses.label.item()) * batch_size
        total_count += batch_size

    return {
        "total_loss": total / max(total_count, 1),
        "concept_loss": total_c / max(total_count, 1),
        "label_loss": total_y / max(total_count, 1),
    }


def _eval_stage3_epoch(
    model: HabitatCBM,
    dataloader,
    device: torch.device,
    lambda_c: float,
    lambda_y: float,
    pos_weight: Optional[torch.Tensor],
    topk_pool: int,
    threshold: float,
    concept_loss_config: Mapping[str, object],
    label_loss_config: Mapping[str, object],
) -> Dict[str, float]:
    model.eval()

    total = 0.0
    total_c = 0.0
    total_y = 0.0
    total_count = 0

    patient_ids: List[str] = []
    probs: List[float] = []
    y_true_blocks: List[int] = []

    with torch.no_grad():
        for batch in dataloader:
            x = batch["image"].to(device=device, dtype=torch.float32, non_blocking=True)
            c_true_std = batch["concept_true_std"].to(device=device, dtype=torch.float32, non_blocking=True)
            y_true = batch["label"].to(device=device, dtype=torch.float32, non_blocking=True)

            out = model.forward_x_to_cy(x)
            losses = compute_joint_loss(
                model=model,
                c_hat=out["c_hat"],
                c_true_std=c_true_std,
                y_logit=out["y_logit"],
                y_true=y_true,
                lambda_c=lambda_c,
                lambda_y=lambda_y,
                pos_weight=pos_weight,
                concept_loss_config=concept_loss_config,
                label_loss_config=label_loss_config,
            )
            y_prob = torch.sigmoid(out["y_logit"]).squeeze(1)

            batch_size = int(x.shape[0])
            total += float(losses.total.item()) * batch_size
            total_c += float(losses.concept.item()) * batch_size
            total_y += float(losses.label.item()) * batch_size
            total_count += batch_size

            batch_patient_ids = batch["patient_id"]
            for idx in range(batch_size):
                patient_ids.append(str(batch_patient_ids[idx]))
                probs.append(float(y_prob[idx].item()))
                y_true_blocks.append(int(y_true[idx].item()))

    y_true_p, y_prob_p, y_pred_p = _patient_level_probs(
        patient_ids=patient_ids,
        probs=probs,
        y_trues=y_true_blocks,
        topk_pool=topk_pool,
        threshold=threshold,
    )
    metrics = _compute_patient_metrics(y_true=y_true_p, y_prob=y_prob_p, y_pred=y_pred_p)

    return {
        "total_loss": total / max(total_count, 1),
        "concept_loss": total_c / max(total_count, 1),
        "label_loss": total_y / max(total_count, 1),
        "auc": metrics["auc"],
        "acc": metrics["acc"],
        "f1": metrics["f1"],
        "sen": metrics["sen"],
        "spe": metrics["spe"],
    }


def _train_stage_loop(
    stage: TrainStage,
    model: HabitatCBM,
    train_loader,
    val_loader,
    optimizer: Optimizer,
    scheduler: object,
    scheduler_step_mode: SchedulerStepMode,
    device: torch.device,
    epochs: int,
    early_stop_patience: int,
    early_stop_min_delta: float,
    checkpoint_path: Path,
    log_csv_path: Path,
    run_id: str,
    model_config: Mapping[str, object],
    train_config: Mapping[str, object],
    optimizer_config: Mapping[str, object],
    scheduler_config: Mapping[str, object],
    loss_config: Mapping[str, object],
    concept_loss_config: Mapping[str, object],
    label_loss_config: Mapping[str, object],
    topk_pool: int,
    threshold: float,
    lambda_c: float,
    lambda_y: float,
    pos_weight: Optional[torch.Tensor],
    concept_noise_std: float,
    concept_noise_std_vector: Optional[torch.Tensor],
    concept_noise_summary: Optional[Mapping[str, object]],
    stage_monitor_metric: Optional[str],
) -> Tuple[int, float, List[Dict[str, object]]]:
    best_epoch = -1
    best_score = -float("inf")
    best_tie_break_score = -float("inf")
    no_improve = 0
    history: List[Dict[str, object]] = []
    sampling_summary = _summarize_loader_sampling(train_loader)
    effective_pos_weight_value = float(pos_weight.item()) if pos_weight is not None else None

    for epoch in range(1, epochs + 1):
        if stage == "stage1":
            train_stats = _train_stage1_epoch(
                model,
                train_loader,
                optimizer,
                device,
                concept_loss_config=concept_loss_config,
            )
            val_stats = _eval_stage1_epoch(
                model,
                val_loader,
                device,
                concept_loss_config=concept_loss_config,
            )
            monitor_metric_name = "val_concept_loss"
            monitor_metric_raw = float(val_stats["concept_loss"])
            selection_score = -monitor_metric_raw
            scheduler_metric = monitor_metric_raw
        elif stage == "stage2":
            train_stats = _train_stage2_epoch(
                model,
                train_loader,
                optimizer,
                device,
                pos_weight,
                concept_noise_std=concept_noise_std,
                concept_noise_std_vector=concept_noise_std_vector,
                label_loss_config=label_loss_config,
            )
            val_stats = _eval_stage2_epoch(
                model,
                val_loader,
                device,
                pos_weight,
                topk_pool=topk_pool,
                threshold=threshold,
                label_loss_config=label_loss_config,
            )
            stage2_monitor_metric = stage_monitor_metric or "label_loss"
            val_auc = float(val_stats["auc"])
            val_label_loss = float(val_stats["label_loss"])
            if stage2_monitor_metric == "label_loss":
                monitor_metric_name = "val_label_loss"
                monitor_metric_raw = val_label_loss
                selection_score = -monitor_metric_raw
                scheduler_metric = selection_score
                monitor_tie_break_name = "val_auc"
                monitor_tie_break_raw = val_auc
                tie_break_score = val_auc if np.isfinite(val_auc) else -float("inf")
            else:
                if np.isnan(val_auc):
                    monitor_metric_name = "val_label_loss(fallback)"
                    monitor_metric_raw = val_label_loss
                    selection_score = -monitor_metric_raw
                    scheduler_metric = selection_score
                    monitor_tie_break_name = None
                    monitor_tie_break_raw = None
                    tie_break_score = -float("inf")
                else:
                    monitor_metric_name = "val_auc"
                    monitor_metric_raw = val_auc
                    selection_score = monitor_metric_raw
                    scheduler_metric = monitor_metric_raw
                    monitor_tie_break_name = "val_label_loss"
                    monitor_tie_break_raw = val_label_loss
                    tie_break_score = -val_label_loss
        else:
            train_stats = _train_stage3_epoch(
                model,
                train_loader,
                optimizer,
                device,
                lambda_c=lambda_c,
                lambda_y=lambda_y,
                pos_weight=pos_weight,
                concept_loss_config=concept_loss_config,
                label_loss_config=label_loss_config,
            )
            val_stats = _eval_stage3_epoch(
                model,
                val_loader,
                device,
                lambda_c=lambda_c,
                lambda_y=lambda_y,
                pos_weight=pos_weight,
                topk_pool=topk_pool,
                threshold=threshold,
                concept_loss_config=concept_loss_config,
                label_loss_config=label_loss_config,
            )
            if np.isnan(float(val_stats["auc"])):
                monitor_metric_name = "val_total_loss(fallback)"
                monitor_metric_raw = float(val_stats["total_loss"])
                selection_score = -monitor_metric_raw
                scheduler_metric = selection_score
            else:
                monitor_metric_name = "val_auc"
                monitor_metric_raw = float(val_stats["auc"])
                selection_score = monitor_metric_raw
                scheduler_metric = monitor_metric_raw

        improved = (selection_score - best_score) > float(early_stop_min_delta)
        if (
            not improved
            and stage == "stage2"
            and monitor_tie_break_name is not None
            and abs(selection_score - best_score) <= float(early_stop_min_delta)
            and tie_break_score > best_tie_break_score
        ):
            improved = True
        if improved:
            best_score = selection_score
            best_tie_break_score = tie_break_score if stage == "stage2" else -float("inf")
            best_epoch = epoch
            no_improve = 0
            _save_checkpoint(
                path=checkpoint_path,
                epoch=epoch,
                stage=stage,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                best_score=best_score,
                run_id=run_id,
                model_config=model_config,
                train_config=train_config,
                optimizer_config=optimizer_config,
                scheduler_config=scheduler_config,
                loss_config=loss_config,
                monitor_name=monitor_metric_name,
            )
        else:
            no_improve += 1

        row: Dict[str, object] = {
            "epoch": epoch,
            "stage": stage,
            "monitor_metric_name": monitor_metric_name,
            "monitor_metric_raw": monitor_metric_raw,
            "selection_score": selection_score,
            "scheduler_metric": scheduler_metric,
            "is_best": int(improved),
            "best_score_so_far": best_score,
            "effective_pos_weight": effective_pos_weight_value,
            "train_sampler_type": sampling_summary["sampler_type"],
            "train_pos_sampling_ratio": sampling_summary["pos_ratio"],
            "train_neg_sampling_ratio": sampling_summary["neg_ratio"],
        }
        if stage == "stage2":
            row["monitor_tie_break_name"] = monitor_tie_break_name
            row["monitor_tie_break_raw"] = monitor_tie_break_raw
            if concept_noise_summary is not None:
                row["concept_noise_mode"] = concept_noise_summary.get("mode")
                row["concept_noise_std_mean"] = concept_noise_summary.get("noise_std_mean", concept_noise_std)
                row["concept_noise_std_min"] = concept_noise_summary.get("noise_std_min", concept_noise_std)
                row["concept_noise_std_max"] = concept_noise_summary.get("noise_std_max", concept_noise_std)
        row.update({f"train_{k}": float(v) for k, v in train_stats.items()})
        row.update({f"val_{k}": float(v) for k, v in val_stats.items()})
        row.update(_get_current_lrs(optimizer))
        history.append(row)

        pos_weight_text = f"{effective_pos_weight_value:.4f}" if effective_pos_weight_value is not None else "disabled"
        tie_break_text = ""
        if stage == "stage2" and monitor_tie_break_name is not None and monitor_tie_break_raw is not None:
            tie_break_text = f" tie={monitor_tie_break_name} raw={monitor_tie_break_raw:.6f}"
        print(
            f"[{stage}] epoch={epoch:03d} metric={monitor_metric_name} raw={monitor_metric_raw:.6f} "
            f"select={selection_score:.6f} best={best_score:.6f} improved={improved} "
            f"{tie_break_text}"
            f"sampler={sampling_summary['sampler_type']} "
            f"pos_ratio={sampling_summary['pos_ratio']:.3f} "
            f"neg_ratio={sampling_summary['neg_ratio']:.3f} "
            f"pos_weight={pos_weight_text}"
        )

        if scheduler is not None:
            if scheduler_step_mode == "metric":
                scheduler.step(scheduler_metric)
            elif scheduler_step_mode == "epoch":
                scheduler.step()

        if early_stop_patience > 0 and no_improve >= early_stop_patience:
            print(
                f"[{stage}] early stop at epoch={epoch} "
                f"(patience={early_stop_patience}, min_delta={early_stop_min_delta})"
            )
            break

    if best_epoch < 0:
        raise RuntimeError(f"{stage} finished without best checkpoint.")

    # 导出阶段日志
    fieldnames = sorted({key for row in history for key in row.keys()})
    _write_csv(log_csv_path, fieldnames=fieldnames, rows=history)
    return best_epoch, best_score, history


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Habitat-CBM with JSON config + CLI overrides.")
    parser.add_argument("--config", type=Path, default=CURRENT_DIR / "args_train_habitat_CBM.json")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--checkpoint-root", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--epochs-stage1", type=int, default=None)
    parser.add_argument("--epochs-stage2", type=int, default=None)
    parser.add_argument("--epochs-stage3", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    return parser


def main() -> None:
    args = _build_argparser().parse_args()
    cfg = _load_json(args.config)
    _validate_config(cfg)
    cfg = _apply_cli_overrides(cfg, args)

    paths_cfg = dict(cfg["paths"])
    data_cfg = dict(cfg["data"])
    model_cfg = dict(cfg["model"])
    train_cfg = dict(cfg["train"])
    eval_cfg = dict(cfg["eval"])
    logging_cfg = dict(cfg["logging"])
    optimizer_cfg = _as_mapping(cfg.get("optimizer", {}), name="optimizer")
    scheduler_cfg = _as_mapping(cfg.get("scheduler", {}), name="scheduler")
    concept_loss_cfg, label_loss_cfg, joint_loss_cfg = _extract_loss_configs(cfg, train_cfg)
    base_loss_cfg = {
        "concept": concept_loss_cfg,
        "label": label_loss_cfg,
        "joint": joint_loss_cfg,
    }

    run_id = _resolve_run_id(logging_cfg.get("run_id"))
    seed = int(train_cfg.get("seed", 42))
    _set_seed(seed)

    runs_root = Path(paths_cfg.get("runs_root", PROJECT_ROOT / "runs" / "03_habitat_cbm"))
    results_root = Path(paths_cfg.get("results_root", PROJECT_ROOT / "results" / "03_habitat_cbm"))
    checkpoint_root_cfg = _optional_path(paths_cfg.get("checkpoint_root"))
    checkpoint_root = checkpoint_root_cfg if checkpoint_root_cfg is not None else runs_root / run_id / "checkpoints"

    run_dir = runs_root / run_id
    result_dir = results_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    # 保存最终生效配置快照
    cfg_snapshot_path = run_dir / f"run_config_{MODEL_NAME}_{run_id}.json"
    _save_json(cfg_snapshot_path, cfg)

    device = torch.device(str(train_cfg.get("device", "cpu")))

    split_base_root = Path(paths_cfg["split_base_root"])
    concept_label_csv = Path(paths_cfg["concept_label_csv"])
    concept_scaler_json = Path(paths_cfg["concept_scaler_json"])
    selected_concept_names = resolve_concept_names(model_cfg.get("selected_concepts"))
    concept_columns = concept_names_to_columns(selected_concept_names)

    transform_map = _build_transforms(data_cfg=data_cfg, train_cfg=train_cfg)
    datasets = build_habitat_cbm_datasets(
        split_base_root=split_base_root,
        modalities=tuple(data_cfg.get("modalities", ("t1", "t1ce", "t2", "t2flair", "adc", "cbf"))),
        require_voi=bool(data_cfg.get("require_voi", True)),
        append_voi_mask=bool(data_cfg.get("append_voi_mask", True)),
        mask_background_with_voi=bool(data_cfg.get("mask_background_with_voi", False)),
        block_depth=int(data_cfg.get("block_depth", 5)),
        slice_axis=int(data_cfg.get("slice_axis", 2)),
        intensity_norm=str(data_cfg.get("intensity_norm", "zscore")),
        min_nonzero_voxels=int(data_cfg.get("min_nonzero_voxels", 16)),
        cache_volumes=bool(data_cfg.get("cache_volumes", True)),
        concept_label_csv=concept_label_csv,
        concept_scaler_json=concept_scaler_json,
        concept_columns=concept_columns,
        transform_map=transform_map,
    )
    dataloaders = build_habitat_cbm_dataloaders(
        datasets=datasets,
        batch_size=int(train_cfg.get("batch_size", 8)),
        num_workers=int(train_cfg.get("num_workers", 4)),
        train_shuffle=True,
        patient_balanced_sampling=bool(train_cfg.get("patient_balanced_sampling", False)),
    )
    scaler: ConceptScaler = load_concept_scaler(
        concept_scaler_json,
        concept_names=selected_concept_names,
    )
    effective_n_concepts = len(scaler.concept_names)
    configured_n_concepts = int(model_cfg.get("n_concepts", effective_n_concepts))
    if configured_n_concepts != effective_n_concepts:
        raise ValueError(
            "model.n_concepts does not match selected concepts/scaler dimension: "
            f"{configured_n_concepts} vs {effective_n_concepts}"
        )
    in_channels = int(model_cfg.get("in_channels", _compute_input_channels(data_cfg)))
    concept_dropout_p, label_dropout_p = _resolve_model_dropouts(model_cfg)

    model = HabitatCBM(
        in_channels=in_channels,
        n_concepts=effective_n_concepts,
        concept_hidden_dim=int(model_cfg.get("concept_hidden_dim", 256)),
        label_hidden_dim=int(model_cfg.get("label_hidden_dim", 32)),
        concept_dropout_p=concept_dropout_p,
        label_dropout_p=label_dropout_p,
        pretrained=bool(model_cfg.get("pretrained", True)),
    ).to(device)

    raw_pos_weight = _compute_pos_weight_from_train_patients(
        datasets["train"],
        device=device,
        label_loss_config={"use_pos_weight": True, "manual_pos_weight": None},
    )

    # 固定监控口径
    threshold = float(eval_cfg.get("threshold", 0.5))
    topk_pool = int(eval_cfg.get("topk_pool", 0))
    lambda_c = float(joint_loss_cfg.get("lambda_c", 0.5))
    lambda_y = float(joint_loss_cfg.get("lambda_y", 1.0))

    stage_cfgs = _build_stage_configs(
        cfg=cfg,
        train_cfg=train_cfg,
        checkpoint_root=checkpoint_root,
        run_dir=run_dir,
    )

    stage_summary: Dict[str, Dict[str, object]] = {}
    stage_checkpoint_paths: Dict[str, Path] = {}
    stage_loss_summary: Dict[str, Dict[str, Dict[str, object]]] = {}

    for stage_name, epochs, ckpt_path, log_path, stage_cfg in stage_cfgs:
        stage = set_train_stage(model, stage_name)
        stage_freeze_layer_names: List[str] = []

        # Stage1：根据配置冻结 encoder 浅层，抑制小样本过拟合
        if stage == "stage1":
            stage_freeze_layer_names = _parse_freeze_encoder_layers(
                stage_cfg.get("freeze_encoder_layers", None)
            )
            _apply_encoder_freeze(model, stage_freeze_layer_names, stage="stage1")

        # Stage3：同样支持冻结 encoder 浅层，防止联合微调时特征提取层大幅偏移
        if stage == "stage3":
            stage_freeze_layer_names = _parse_freeze_encoder_layers(
                stage_cfg.get("freeze_encoder_layers", None)
            )
            _apply_encoder_freeze(model, stage_freeze_layer_names, stage="stage3")

        stage2_patient_level = bool(stage_cfg.get("patient_level", False))
        stage2_concept_noise_mode = _normalize_key(stage_cfg.get("concept_noise_mode", "gaussian"))
        if stage2_concept_noise_mode not in {"none", "off", "disabled", "gaussian", "residual"}:
            raise ValueError(
                f"Unsupported stages.{stage}.concept_noise_mode: {stage2_concept_noise_mode}. "
                "Use one of: none, gaussian, residual."
            )
        stage2_concept_noise_std = float(stage_cfg.get("concept_noise_std", 0.0))
        stage2_concept_noise_min_std = float(stage_cfg.get("concept_noise_min_std", 0.0))
        max_noise_raw = stage_cfg.get("concept_noise_max_std", None)
        stage2_concept_noise_max_std = float(max_noise_raw) if max_noise_raw is not None else None
        stage_batch_size = int(stage_cfg.get("batch_size", train_cfg.get("batch_size", 8)))
        if stage_batch_size <= 0:
            raise ValueError(f"stages.{stage}.batch_size must be positive, got {stage_batch_size}")
        if stage2_concept_noise_std < 0.0:
            raise ValueError(f"stages.{stage}.concept_noise_std must be >= 0, got {stage2_concept_noise_std}")
        stage_monitor_metric = _resolve_stage_monitor_metric(stage_cfg, stage)
        stage_label_loss_cfg = _resolve_stage_label_loss_config(label_loss_cfg, stage_cfg, stage)
        stage_effective_loss_cfg = {
            "concept": dict(concept_loss_cfg),
            "label": dict(stage_label_loss_cfg),
            "joint": dict(joint_loss_cfg),
        }
        stage_effective_pos_weight = _compute_pos_weight_from_train_patients(
            datasets["train"],
            device=device,
            label_loss_config=stage_label_loss_cfg,
        )

        if stage == "stage2" and stage2_patient_level:
            stage2_patient_dataloaders = _build_stage2_patient_dataloaders(
                datasets=datasets,
                batch_size=stage_batch_size,
                num_workers=int(train_cfg.get("num_workers", 4)),
            )
            stage_train_loader = stage2_patient_dataloaders["train"]
            stage_val_loader = stage2_patient_dataloaders["val"]
        elif stage_batch_size != int(train_cfg.get("batch_size", 8)):
            stage_dataloaders = build_habitat_cbm_dataloaders(
                datasets=datasets,
                batch_size=stage_batch_size,
                num_workers=int(train_cfg.get("num_workers", 4)),
                train_shuffle=True,
                patient_balanced_sampling=bool(train_cfg.get("patient_balanced_sampling", False)),
            )
            stage_train_loader = stage_dataloaders["train"]
            stage_val_loader = stage_dataloaders["val"]
        else:
            stage_train_loader = dataloaders["train"]
            stage_val_loader = dataloaders["val"]

        stage_concept_noise_std_vector: Optional[torch.Tensor] = None
        stage_concept_noise_summary: Optional[Dict[str, object]] = None
        if stage == "stage2":
            if stage2_concept_noise_mode in {"none", "off", "disabled"} or stage2_concept_noise_std <= 0.0:
                stage2_concept_noise_std = 0.0
                stage_concept_noise_summary = {
                    "mode": "disabled",
                    "base_noise_std": float(stage2_concept_noise_std),
                }
            elif stage2_concept_noise_mode == "residual":
                residual_loader = DataLoader(
                    datasets["train"],
                    batch_size=int(train_cfg.get("batch_size", 8)),
                    shuffle=False,
                    num_workers=int(train_cfg.get("num_workers", 4)),
                    pin_memory=torch.cuda.is_available(),
                )
                stage_concept_noise_std_vector, stage_concept_noise_summary = _estimate_stage2_residual_noise_std(
                    model=model,
                    dataloader=residual_loader,
                    device=device,
                    base_noise_std=stage2_concept_noise_std,
                    eval_transform=transform_map.get("val") if transform_map else None,
                    min_noise_std=stage2_concept_noise_min_std,
                    max_noise_std=stage2_concept_noise_max_std,
                )
                print(
                    "[stage2] residual-aware concept noise std: "
                    f"mean={stage_concept_noise_summary['noise_std_mean']:.6f} "
                    f"min={stage_concept_noise_summary['noise_std_min']:.6f} "
                    f"max={stage_concept_noise_summary['noise_std_max']:.6f}"
                )
            else:
                stage_concept_noise_summary = {
                    "mode": "gaussian",
                    "base_noise_std": float(stage2_concept_noise_std),
                    "noise_std_mean": float(stage2_concept_noise_std),
                    "noise_std_min": float(stage2_concept_noise_std),
                    "noise_std_max": float(stage2_concept_noise_std),
                }

        stage_optimizer_cfg = {
            **optimizer_cfg,
            **_as_mapping(stage_cfg.get("optimizer", {}), name=f"stages.{stage}.optimizer"),
        }
        stage_scheduler_cfg = {
            **scheduler_cfg,
            **_as_mapping(stage_cfg.get("scheduler", {}), name=f"stages.{stage}.scheduler"),
        }
        optimizer = build_optimizer(
            model=model,
            optimizer_config=stage_optimizer_cfg,
            train_config=train_cfg,
        )
        scheduler, scheduler_step_mode = build_scheduler(
            optimizer=optimizer,
            scheduler_config=stage_scheduler_cfg,
            epochs=epochs,
        )

        best_epoch, best_score, _ = _train_stage_loop(
            stage=stage,
            model=model,
            train_loader=stage_train_loader,
            val_loader=stage_val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scheduler_step_mode=scheduler_step_mode,
            device=device,
            epochs=epochs,
            early_stop_patience=int(train_cfg.get("early_stop_patience", 10)),
            early_stop_min_delta=float(train_cfg.get("early_stop_min_delta", 0.0)),
            checkpoint_path=ckpt_path,
            log_csv_path=log_path,
            run_id=run_id,
            model_config={
                "in_channels": in_channels,
                "n_concepts": effective_n_concepts,
                "selected_concepts": list(scaler.concept_names),
                "concept_hidden_dim": int(model_cfg.get("concept_hidden_dim", 256)),
                "label_hidden_dim": int(model_cfg.get("label_hidden_dim", 32)),
                "label_head_type": getattr(model, "label_head_type", "unknown"),
                "concept_dropout_p": concept_dropout_p,
                "label_dropout_p": label_dropout_p,
            },
            train_config=train_cfg,
            optimizer_config=stage_optimizer_cfg,
            scheduler_config=stage_scheduler_cfg,
            loss_config=stage_effective_loss_cfg,
            concept_loss_config=concept_loss_cfg,
            label_loss_config=stage_label_loss_cfg,
            topk_pool=topk_pool,
            threshold=threshold,
            lambda_c=lambda_c,
            lambda_y=lambda_y,
            pos_weight=stage_effective_pos_weight,
            concept_noise_std=stage2_concept_noise_std,
            concept_noise_std_vector=stage_concept_noise_std_vector,
            concept_noise_summary=stage_concept_noise_summary,
            stage_monitor_metric=stage_monitor_metric,
        )

        # 下一阶段从当前阶段最佳权重继续
        _load_checkpoint_model_only(model, ckpt_path, device=device)
        stage_loss_summary[stage] = stage_effective_loss_cfg

        stage_summary[stage] = {
            "best_epoch": int(best_epoch),
            "best_score": float(best_score),
            "checkpoint": str(ckpt_path),
            "log_csv": str(log_path),
            "optimizer": dict(stage_optimizer_cfg),
            "scheduler": dict(stage_scheduler_cfg),
            "scheduler_step_mode": scheduler_step_mode,
            "batch_size": stage_batch_size,
            "stage2_patient_level": stage2_patient_level if stage == "stage2" else False,
            "stage2_concept_noise_mode": stage2_concept_noise_mode if stage == "stage2" else "none",
            "stage2_concept_noise_std": stage2_concept_noise_std if stage == "stage2" else 0.0,
            "stage2_concept_noise_summary": stage_concept_noise_summary if stage == "stage2" else None,
            "stage_monitor_metric": stage_monitor_metric,
            "effective_pos_weight": float(stage_effective_pos_weight.item()) if stage_effective_pos_weight is not None else None,
            "label_loss_config": dict(stage_label_loss_cfg),
            "frozen_encoder_layers": stage_freeze_layer_names,
        }
        stage_checkpoint_paths[stage] = ckpt_path

    stage3_ckpt = stage_checkpoint_paths.get("stage3", checkpoint_root / "stage3_best.pt")
    _load_checkpoint_model_only(model, stage3_ckpt, device=device)

    # ── 用 val 集 Youden Index 动态选取最优阈值 ──────────────────────────────
    print("[eval] Computing optimal threshold via Youden Index on val set ...")
    val_y_true, val_y_prob = collect_val_patient_probs(
        model=model,
        val_loader=dataloaders["val"],
        device=device,
        topk_pool=topk_pool,
        threshold=threshold,  # 此处 threshold 仅用于 topk 聚合，不影响 Youden 计算
    )
    youden_threshold = find_youden_threshold(val_y_true, val_y_prob)
    print(
        f"[eval] Fixed threshold={threshold:.4f}  →  Youden threshold={youden_threshold:.4f} "
        f"(val n={len(val_y_true)}, pos={int(val_y_true.sum())}, neg={int((1-val_y_true).sum())})"
    )
    eval_threshold = youden_threshold
    # ─────────────────────────────────────────────────────────────────────────

    include_splits = tuple(eval_cfg.get("include_splits", ["train", "val", "test"]))
    figure_include_splits = tuple(eval_cfg.get("figure_include_splits", include_splits))
    exported = run_full_evaluation(
        model=model,
        dataloaders=dataloaders,
        scaler=scaler,
        output_dir=result_dir,
        run_id=run_id,
        checkpoint_name=stage3_ckpt.name,
        threshold=eval_threshold,
        topk_pool=topk_pool,
        device=device,
        include_splits=include_splits,
        config_path=str(args.config),
        export_png=bool(eval_cfg.get("export_png", False)),
        figure_include_splits=figure_include_splits,
        figure_dpi=int(eval_cfg.get("figure_dpi", 150)),
    )

    run_summary = {
        "model": MODEL_NAME,
        "run_id": run_id,
        "seed": seed,
        "device": str(device),
        "architecture": {
            "label_head_type": getattr(model, "label_head_type", "unknown"),
            "label_hidden_dim_config": int(model_cfg.get("label_hidden_dim", 32)),
        },
        "paths": {
            "run_dir": str(run_dir),
            "result_dir": str(result_dir),
            "checkpoint_root": str(checkpoint_root),
            "config_snapshot": str(cfg_snapshot_path),
        },
        "data": {
            "split_base_root": str(split_base_root),
            "concept_label_csv": str(concept_label_csv),
            "concept_scaler_json": str(concept_scaler_json),
            "input_channels": in_channels,
            "n_concepts": effective_n_concepts,
            "selected_concepts": list(scaler.concept_names),
            "train_patients": len(datasets["train"].patient_cases),
            "val_patients": len(datasets["val"].patient_cases),
            "test_patients": len(datasets["test"].patient_cases),
        },
        "class_balance": {
            "patient_level_pos_weight_raw": float(raw_pos_weight.item()) if raw_pos_weight is not None else None,
        },
        "threshold": {
            "fixed_threshold": threshold,
            "youden_threshold": float(youden_threshold),
            "eval_threshold_used": float(eval_threshold),
        },
        "loss": {
            "base": base_loss_cfg,
            "stage_effective": stage_loss_summary,
        },
        "optimizer": dict(optimizer_cfg),
        "scheduler": dict(scheduler_cfg),
        "stage_summary": stage_summary,
        "evaluation_exports": {k: str(v) for k, v in exported.items()},
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    run_summary_path = result_dir / f"run_train_summary_{MODEL_NAME}_{run_id}.json"
    _save_json(run_summary_path, run_summary)

    print("Training finished.")
    print(f"  run_id       : {run_id}")
    print(f"  run_dir      : {run_dir}")
    print(f"  result_dir   : {result_dir}")
    print(f"  stage3_ckpt  : {stage3_ckpt}")
    print(f"  run_summary  : {run_summary_path}")


if __name__ == "__main__":
    main()
