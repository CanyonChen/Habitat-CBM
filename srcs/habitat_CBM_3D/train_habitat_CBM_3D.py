#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Three-stage training entrypoint for Habitat-CBM 3D."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parents[1]
PROJECT_ROOT = REPO_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.habitat_CBM_3D import HabitatCBM3D  # noqa: E402
from srcs.data_loader_habitat_CBM import (  # noqa: E402
    ConceptScaler,
    concept_names_to_columns,
    load_concept_scaler,
    resolve_concept_names,
)
from srcs.habitat_CBM_3D.data_loader_habitat_CBM_3D import (  # noqa: E402
    DEFAULT_3D_CROP_MARGIN,
    DEFAULT_3D_MODALITIES,
    DEFAULT_3D_TARGET_SHAPE,
    build_habitat_cbm_3d_dataloaders,
    build_habitat_cbm_3d_datasets,
)
from srcs.habitat_CBM_3D.eval_habitat_CBM_3D import run_full_evaluation  # noqa: E402
from srcs.habitat_CBM_3D.monai_augmentation_3d import (  # noqa: E402
    MonaiVolumeAugmentConfig,
    build_monai_volume_transforms,
)
from srcs.train_habitat_CBM import (  # noqa: E402
    _apply_cli_overrides,
    _apply_encoder_freeze,
    _as_mapping,
    _build_stage2_patient_dataloaders,
    _build_stage_configs,
    _compute_pos_weight_from_train_patients,
    _estimate_stage2_residual_noise_std,
    _extract_loss_configs,
    _is_stage3_variant,
    _load_checkpoint_model_only,
    _normalize_key,
    _optional_path,
    _parse_freeze_encoder_layers,
    _resolve_model_dropouts,
    _resolve_run_id,
    _resolve_stage_joint_loss_config,
    _resolve_stage_label_loss_config,
    _resolve_stage_monitor_metric,
    _save_json,
    _set_seed,
    _train_stage_loop,
    build_optimizer,
    build_scheduler,
    collect_val_patient_probs,
    find_youden_threshold,
    set_train_stage,
)

MODEL_NAME = "habitat_cbm_3d"


def _load_json(path: Path) -> Dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Config must be a JSON object.")
    return payload


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


def _parse_int_triple(
    value: object,
    default: Sequence[int],
    *,
    name: str,
    allow_zero: bool = False,
) -> Tuple[int, int, int]:
    if value is None:
        items = tuple(int(v) for v in default)
    elif isinstance(value, str):
        items = tuple(int(v.strip()) for v in value.split(",") if v.strip())
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        items = tuple(int(v) for v in value)
    else:
        raise ValueError(f"{name} must be a three-item sequence or comma-separated string.")
    if len(items) != 3:
        raise ValueError(f"{name} must contain exactly three integers [D,H,W], got {items!r}.")
    if allow_zero:
        if any(v < 0 for v in items):
            raise ValueError(f"{name} values must be non-negative, got {items!r}.")
    elif any(v <= 0 for v in items):
        raise ValueError(f"{name} values must be positive, got {items!r}.")
    return items


def _build_transforms(data_cfg: Mapping[str, object], train_cfg: Mapping[str, object]) -> Dict[str, object]:
    aug_cfg = MonaiVolumeAugmentConfig(
        enabled=bool(train_cfg.get("use_monai_augmentation", True)),
        affine_prob=float(train_cfg.get("aug_affine_prob", 0.5)),
        rotate_deg=float(train_cfg.get("aug_rotate_deg", 10.0)),
        translate_px=float(train_cfg.get("aug_translate_px", 6.0)),
        scale_range=float(train_cfg.get("aug_scale_range", 0.10)),
        flip_prob=float(train_cfg.get("aug_flip_prob", 0.5)),
        intensity_scale_prob=float(train_cfg.get("aug_intensity_scale_prob", 0.5)),
        intensity_scale=float(train_cfg.get("aug_intensity_scale", 0.15)),
        intensity_shift_prob=float(train_cfg.get("aug_intensity_shift_prob", 0.5)),
        intensity_shift=float(train_cfg.get("aug_intensity_shift", 0.10)),
        gaussian_noise_prob=float(train_cfg.get("aug_gaussian_noise_prob", 0.25)),
        gaussian_noise_std=float(train_cfg.get("aug_gaussian_noise_std", 0.03)),
        gibbs_noise_prob=float(train_cfg.get("aug_gibbs_noise_prob", 0.0)),
        gibbs_noise_alpha=float(train_cfg.get("aug_gibbs_noise_alpha", 0.3)),
    )
    target_shape = _parse_int_triple(data_cfg.get("target_shape"), DEFAULT_3D_TARGET_SHAPE, name="data.target_shape")
    return build_monai_volume_transforms(config=aug_cfg, spatial_size=target_shape)


def _compute_input_channels(data_cfg: Mapping[str, object]) -> int:
    modalities = tuple(data_cfg.get("modalities", DEFAULT_3D_MODALITIES))
    if not modalities:
        raise ValueError("data.modalities must not be empty.")
    return len(modalities)


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Habitat-CBM 3D with JSON config + CLI overrides.")
    parser.add_argument("--config", type=Path, default=CURRENT_DIR / "args_train_habitat_CBM_3D.json")
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
    base_loss_cfg = {"concept": concept_loss_cfg, "label": label_loss_cfg, "joint": joint_loss_cfg}

    run_id = _resolve_run_id(logging_cfg.get("run_id"))
    seed = int(train_cfg.get("seed", 42))
    _set_seed(seed)

    runs_root = Path(paths_cfg.get("runs_root", PROJECT_ROOT / "results" / "habitat_CBM_3D"))
    results_root = Path(paths_cfg.get("results_root", PROJECT_ROOT / "results" / "habitat_CBM_3D"))
    checkpoint_root_cfg = _optional_path(paths_cfg.get("checkpoint_root"))
    checkpoint_root = checkpoint_root_cfg if checkpoint_root_cfg is not None else runs_root / run_id / "checkpoints"
    run_dir = runs_root / run_id
    result_dir = results_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    cfg_snapshot_path = run_dir / f"run_config_{MODEL_NAME}_{run_id}.json"
    _save_json(cfg_snapshot_path, cfg)
    device = torch.device(str(train_cfg.get("device", "cpu")))

    split_base_root = Path(paths_cfg["split_base_root"])
    concept_label_csv = Path(paths_cfg["concept_label_csv"])
    concept_scaler_json = Path(paths_cfg["concept_scaler_json"])
    selected_concept_names = resolve_concept_names(model_cfg.get("selected_concepts"))
    concept_columns = concept_names_to_columns(selected_concept_names)
    modalities = tuple(data_cfg.get("modalities", DEFAULT_3D_MODALITIES))
    target_shape = _parse_int_triple(data_cfg.get("target_shape"), DEFAULT_3D_TARGET_SHAPE, name="data.target_shape")
    crop_margin = _parse_int_triple(
        data_cfg.get("crop_margin"),
        DEFAULT_3D_CROP_MARGIN,
        name="data.crop_margin",
        allow_zero=True,
    )

    transform_map = _build_transforms(data_cfg=data_cfg, train_cfg=train_cfg)
    datasets = build_habitat_cbm_3d_datasets(
        split_base_root=split_base_root,
        modalities=modalities,
        require_voi=bool(data_cfg.get("require_voi", True)),
        crop_with_voi=bool(data_cfg.get("crop_with_voi", True)),
        crop_margin=crop_margin,
        intensity_norm=str(data_cfg.get("intensity_norm", "zscore")),
        cache_volumes=bool(data_cfg.get("cache_volumes", False)),
        concept_label_csv=concept_label_csv,
        concept_scaler_json=concept_scaler_json,
        concept_columns=concept_columns,
        transform_map=transform_map,
    )
    dataloaders = build_habitat_cbm_3d_dataloaders(
        datasets=datasets,
        batch_size=int(train_cfg.get("batch_size", 1)),
        num_workers=int(train_cfg.get("num_workers", 2)),
        train_shuffle=True,
        patient_balanced_sampling=bool(train_cfg.get("patient_balanced_sampling", False)),
    )
    scaler: ConceptScaler = load_concept_scaler(concept_scaler_json, concept_names=selected_concept_names)
    effective_n_concepts = len(scaler.concept_names)
    configured_n_concepts = int(model_cfg.get("n_concepts", effective_n_concepts))
    if configured_n_concepts != effective_n_concepts:
        raise ValueError(
            "model.n_concepts does not match selected concepts/scaler dimension: "
            f"{configured_n_concepts} vs {effective_n_concepts}"
        )
    in_channels = int(model_cfg.get("in_channels", _compute_input_channels(data_cfg)))
    expected_channels = _compute_input_channels(data_cfg)
    if in_channels != expected_channels:
        raise ValueError(
            "Habitat-CBM 3D does not append VOI as an input channel; "
            f"model.in_channels must equal len(data.modalities)={expected_channels}, got {in_channels}."
        )
    concept_dropout_p, label_dropout_p = _resolve_model_dropouts(model_cfg)
    pretrain_path = model_cfg.get("pretrain_path", None)
    pretrain_path = None if pretrain_path in (None, "") else str(pretrain_path)

    model = HabitatCBM3D(
        in_channels=in_channels,
        n_concepts=effective_n_concepts,
        concept_hidden_dim=int(model_cfg.get("concept_hidden_dim", 256)),
        label_hidden_dim=int(model_cfg.get("label_hidden_dim", 32)),
        concept_dropout_p=concept_dropout_p,
        label_dropout_p=label_dropout_p,
        pretrain_path=pretrain_path,
    ).to(device)

    raw_pos_weight = _compute_pos_weight_from_train_patients(
        datasets["train"],
        device=device,
        label_loss_config={"use_pos_weight": True, "manual_pos_weight": None},
    )
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
        if stage == "stage1":
            stage_freeze_layer_names = _parse_freeze_encoder_layers(stage_cfg.get("freeze_encoder_layers", None))
            _apply_encoder_freeze(model, stage_freeze_layer_names, stage="stage1")
        if stage == "stage3a":
            print("[stage3a] trainable policy: label_head only (encoder and concept_head frozen)")
        if stage in {"stage3", "stage3b"}:
            stage_freeze_layer_names = _parse_freeze_encoder_layers(stage_cfg.get("freeze_encoder_layers", None))
            _apply_encoder_freeze(model, stage_freeze_layer_names, stage=stage)

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
        stage_batch_size = int(stage_cfg.get("batch_size", train_cfg.get("batch_size", 1)))
        if stage_batch_size <= 0:
            raise ValueError(f"stages.{stage}.batch_size must be positive, got {stage_batch_size}")
        stage_monitor_metric = _resolve_stage_monitor_metric(stage_cfg, stage)
        stage_label_loss_cfg = _resolve_stage_label_loss_config(label_loss_cfg, stage_cfg, stage)
        stage_joint_loss_cfg = _resolve_stage_joint_loss_config(joint_loss_cfg, stage_cfg, stage)
        stage_lambda_c = float(stage_joint_loss_cfg.get("lambda_c", lambda_c))
        stage_lambda_y = float(stage_joint_loss_cfg.get("lambda_y", lambda_y))

        stage3_concept_guard_enabled = False
        stage3_reference_concept_loss: Optional[float] = None
        stage3_concept_loss_max_delta = 0.0
        if _is_stage3_variant(stage):
            stage3_concept_guard_enabled = bool(stage_cfg.get("concept_guard_enabled", True))
            stage3_concept_loss_max_delta = float(stage_cfg.get("concept_guard_max_delta", 0.10))
            explicit_reference_loss = stage_cfg.get("concept_guard_reference_loss", None)
            if explicit_reference_loss is not None:
                stage3_reference_concept_loss = float(explicit_reference_loss)
            elif "stage1" in stage_summary:
                stage3_reference_concept_loss = -float(stage_summary["stage1"]["best_score"])
            if stage3_concept_guard_enabled and stage3_reference_concept_loss is None:
                print(f"[{stage}] concept guard requested but no Stage1 reference loss is available; disabling.")
                stage3_concept_guard_enabled = False
            elif stage3_concept_guard_enabled:
                print(
                    f"[{stage}] concept guard: require val_concept_loss <= "
                    f"{stage3_reference_concept_loss + stage3_concept_loss_max_delta:.6f} "
                    f"(reference={stage3_reference_concept_loss:.6f}, delta={stage3_concept_loss_max_delta:.6f})"
                )

        stage_effective_loss_cfg = {
            "concept": dict(concept_loss_cfg),
            "label": dict(stage_label_loss_cfg),
            "joint": dict(stage_joint_loss_cfg),
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
                num_workers=int(train_cfg.get("num_workers", 2)),
            )
            stage_train_loader = stage2_patient_dataloaders["train"]
            stage_val_loader = stage2_patient_dataloaders["val"]
        elif stage_batch_size != int(train_cfg.get("batch_size", 1)):
            stage_dataloaders = build_habitat_cbm_3d_dataloaders(
                datasets=datasets,
                batch_size=stage_batch_size,
                num_workers=int(train_cfg.get("num_workers", 2)),
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
                stage_concept_noise_summary = {"mode": "disabled", "base_noise_std": 0.0}
            elif stage2_concept_noise_mode == "residual":
                residual_loader = DataLoader(
                    datasets["train"],
                    batch_size=int(train_cfg.get("batch_size", 1)),
                    shuffle=False,
                    num_workers=int(train_cfg.get("num_workers", 2)),
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
        optimizer = build_optimizer(model=model, optimizer_config=stage_optimizer_cfg, train_config=train_cfg)
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
            early_stop_patience=int(stage_cfg.get("early_stop_patience", train_cfg.get("early_stop_patience", 10))),
            early_stop_min_delta=float(stage_cfg.get("early_stop_min_delta", train_cfg.get("early_stop_min_delta", 0.0))),
            checkpoint_path=ckpt_path,
            log_csv_path=log_path,
            run_id=run_id,
            model_config={
                "model_name": MODEL_NAME,
                "in_channels": in_channels,
                "target_shape": list(target_shape),
                "modalities": list(modalities),
                "crop_with_voi": bool(data_cfg.get("crop_with_voi", True)),
                "crop_margin": list(crop_margin),
                "pretrain_path": pretrain_path,
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
            lambda_c=stage_lambda_c,
            lambda_y=stage_lambda_y,
            pos_weight=stage_effective_pos_weight,
            concept_noise_std=stage2_concept_noise_std,
            concept_noise_std_vector=stage_concept_noise_std_vector,
            concept_noise_summary=stage_concept_noise_summary,
            stage_monitor_metric=stage_monitor_metric,
            stage3_concept_guard_enabled=stage3_concept_guard_enabled,
            stage3_reference_concept_loss=stage3_reference_concept_loss,
            stage3_concept_loss_max_delta=stage3_concept_loss_max_delta,
        )

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
            "trainable_policy": (
                "label_head_only"
                if stage in {"stage2", "stage3a"}
                else "joint_finetune"
                if _is_stage3_variant(stage)
                else "concept_finetune"
            ),
            "stage3_concept_guard_enabled": stage3_concept_guard_enabled if _is_stage3_variant(stage) else False,
            "stage3_concept_guard_reference_loss": stage3_reference_concept_loss if _is_stage3_variant(stage) else None,
            "stage3_concept_guard_max_delta": stage3_concept_loss_max_delta if _is_stage3_variant(stage) else 0.0,
            "effective_pos_weight": float(stage_effective_pos_weight.item()) if stage_effective_pos_weight is not None else None,
            "label_loss_config": dict(stage_label_loss_cfg),
            "joint_loss_config": dict(stage_joint_loss_cfg),
            "frozen_encoder_layers": stage_freeze_layer_names,
        }
        stage_checkpoint_paths[stage] = ckpt_path
        if stage == "stage3b":
            stage_loss_summary["stage3"] = stage_effective_loss_cfg
            stage_summary["stage3"] = dict(stage_summary[stage])
            stage_checkpoint_paths["stage3"] = ckpt_path

    stage3_ckpt = stage_checkpoint_paths.get("stage3", checkpoint_root / "stage3_best.pt")
    _load_checkpoint_model_only(model, stage3_ckpt, device=device)

    print("[eval] Computing optimal threshold via Youden Index on val set ...")
    val_y_true, val_y_prob = collect_val_patient_probs(
        model=model,
        val_loader=dataloaders["val"],
        device=device,
        topk_pool=topk_pool,
        threshold=threshold,
    )
    youden_threshold = find_youden_threshold(val_y_true, val_y_prob)
    print(
        f"[eval] Fixed threshold={threshold:.4f} -> Youden threshold={youden_threshold:.4f} "
        f"(val n={len(val_y_true)}, pos={int(val_y_true.sum())}, neg={int((1-val_y_true).sum())})"
    )

    include_splits = tuple(eval_cfg.get("include_splits", ["train", "val", "test"]))
    figure_include_splits = tuple(eval_cfg.get("figure_include_splits", include_splits))
    exported = run_full_evaluation(
        model=model,
        dataloaders=dataloaders,
        scaler=scaler,
        output_dir=result_dir,
        run_id=run_id,
        checkpoint_name=stage3_ckpt.name,
        threshold=youden_threshold,
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
            "pretrain_path": pretrain_path,
            "pretrain_info": getattr(model.encoder, "pretrain_info", None),
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
            "target_shape": list(target_shape),
            "modalities": list(modalities),
            "crop_margin": list(crop_margin),
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
            "eval_threshold_used": float(youden_threshold),
        },
        "loss": {"base": base_loss_cfg, "stage_effective": stage_loss_summary},
        "optimizer": dict(optimizer_cfg),
        "scheduler": dict(scheduler_cfg),
        "stage_summary": stage_summary,
        "evaluation_exports": {key: str(value) for key, value in exported.items()},
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    run_summary_path = result_dir / f"run_train_summary_{MODEL_NAME}_{run_id}.json"
    _save_json(run_summary_path, run_summary)

    print("3D training finished.")
    print(f"  run_id       : {run_id}")
    print(f"  run_dir      : {run_dir}")
    print(f"  result_dir   : {result_dir}")
    print(f"  stage3_ckpt  : {stage3_ckpt}")
    print(f"  run_summary  : {run_summary_path}")


if __name__ == "__main__":
    main()
