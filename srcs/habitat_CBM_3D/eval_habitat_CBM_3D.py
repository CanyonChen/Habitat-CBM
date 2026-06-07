#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Full evaluation entrypoint for Habitat-CBM 3D checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import torch

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import srcs.eval_habitat_CBM as eval2d  # noqa: E402
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
from srcs.habitat_CBM_3D.monai_augmentation_3d import (  # noqa: E402
    MonaiVolumeAugmentConfig,
    build_monai_volume_transforms,
)

MODEL_NAME = "habitat_cbm_3d"


def _load_json(path: Path) -> Dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"Config JSON not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Config JSON must be an object.")
    return payload


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


def _resolve_run_id(run_id: str | None) -> str:
    return run_id if run_id else time.strftime("%Y%m%d_%H%M%S")


def build_eval_datasets_and_loaders(
    split_base_root: Path,
    concept_label_csv: Path,
    concept_scaler_json: Path,
    data_cfg: Mapping[str, object],
    batch_size: int,
    num_workers: int,
    concept_columns: Sequence[str] | None = None,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    target_shape = _parse_int_triple(data_cfg.get("target_shape"), DEFAULT_3D_TARGET_SHAPE, name="data.target_shape")
    transform_map = build_monai_volume_transforms(
        config=MonaiVolumeAugmentConfig(enabled=False),
        spatial_size=target_shape,
    )
    datasets = build_habitat_cbm_3d_datasets(
        split_base_root=split_base_root,
        modalities=tuple(data_cfg.get("modalities", DEFAULT_3D_MODALITIES)),
        require_voi=bool(data_cfg.get("require_voi", True)),
        crop_with_voi=bool(data_cfg.get("crop_with_voi", True)),
        crop_margin=_parse_int_triple(
            data_cfg.get("crop_margin"),
            DEFAULT_3D_CROP_MARGIN,
            name="data.crop_margin",
            allow_zero=True,
        ),
        intensity_norm=str(data_cfg.get("intensity_norm", "zscore")),
        cache_volumes=bool(data_cfg.get("cache_volumes", False)),
        concept_label_csv=concept_label_csv,
        concept_scaler_json=concept_scaler_json,
        concept_columns=concept_columns,
        transform_map=transform_map,
    )
    loaders = build_habitat_cbm_3d_dataloaders(
        datasets=datasets,
        batch_size=batch_size,
        num_workers=num_workers,
        train_shuffle=False,
    )
    return datasets, loaders


def run_full_evaluation(
    model: HabitatCBM3D,
    dataloaders: Mapping[str, object],
    scaler: ConceptScaler,
    output_dir: Path,
    run_id: str,
    checkpoint_name: str,
    threshold: float,
    topk_pool: int,
    device: torch.device,
    include_splits: Sequence[str],
    config_path: str,
    export_png: bool = False,
    figure_include_splits: Optional[Sequence[str]] = None,
    figure_dpi: int = 150,
) -> Dict[str, Path]:
    previous_model_name = eval2d.MODEL_NAME
    eval2d.MODEL_NAME = MODEL_NAME
    try:
        split_outputs: Dict[str, Dict[str, object]] = {}
        for split in include_splits:
            if split not in dataloaders:
                continue
            split_outputs[split] = eval2d.evaluate_split(
                model=model,
                dataloader=dataloaders[split],
                scaler=scaler,
                split=split,
                run_id=run_id,
                checkpoint_name=checkpoint_name,
                threshold=threshold,
                topk_pool=topk_pool,
                device=device,
            )
        return eval2d.export_evaluation_outputs(
            split_outputs=split_outputs,
            output_dir=output_dir,
            run_id=run_id,
            checkpoint_name=checkpoint_name,
            threshold=threshold,
            config_path=config_path,
            export_png=export_png,
            figure_include_splits=figure_include_splits,
            figure_dpi=figure_dpi,
        )
    finally:
        eval2d.MODEL_NAME = previous_model_name


def _load_model_config_from_checkpoint(checkpoint_path: Path) -> Dict[str, object]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("Unsupported checkpoint format.")
    model_cfg = payload.get("model_config", {})
    if not isinstance(model_cfg, Mapping):
        raise ValueError("Checkpoint missing model_config.")
    return dict(model_cfg)


def _load_model_from_checkpoint(checkpoint_path: Path, device: torch.device) -> HabitatCBM3D:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("Unsupported checkpoint format.")
    model_cfg = payload.get("model_config", {})
    if not isinstance(model_cfg, Mapping):
        raise ValueError("Checkpoint missing model_config.")
    concept_dropout_p, label_dropout_p = eval2d._resolve_model_dropouts(model_cfg)
    checkpoint_selected_concepts = model_cfg.get("selected_concepts")
    if "n_concepts" in model_cfg:
        checkpoint_n_concepts = int(model_cfg["n_concepts"])
    elif checkpoint_selected_concepts is not None:
        checkpoint_n_concepts = len(resolve_concept_names(checkpoint_selected_concepts))
    else:
        checkpoint_n_concepts = 5

    model = HabitatCBM3D(
        in_channels=int(model_cfg.get("in_channels", 4)),
        n_concepts=checkpoint_n_concepts,
        concept_hidden_dim=int(model_cfg.get("concept_hidden_dim", 256)),
        label_hidden_dim=int(model_cfg.get("label_hidden_dim", 32)),
        concept_dropout_p=concept_dropout_p,
        label_dropout_p=label_dropout_p,
        pretrain_path=None,
    ).to(device)
    state_dict = payload.get("model_state_dict")
    if state_dict is None:
        raise ValueError("Checkpoint missing model_state_dict.")
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Habitat-CBM 3D full evaluation script.")
    parser.add_argument("--config", type=Path, default=CURRENT_DIR / "args_train_habitat_CBM_3D.json")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--topk-pool", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--splits", type=str, default="val,test")
    return parser


def main() -> None:
    args = _build_argparser().parse_args()
    cfg = _load_json(args.config)
    paths_cfg = cfg.get("paths", {})
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    eval_cfg = cfg.get("eval", {})
    train_cfg = cfg.get("train", {})
    if not isinstance(paths_cfg, Mapping):
        paths_cfg = {}
    if not isinstance(data_cfg, Mapping):
        data_cfg = {}
    if not isinstance(model_cfg, Mapping):
        model_cfg = {}
    if not isinstance(eval_cfg, Mapping):
        eval_cfg = {}
    if not isinstance(train_cfg, Mapping):
        train_cfg = {}

    checkpoint_model_cfg = _load_model_config_from_checkpoint(args.checkpoint)
    selected_concept_names = resolve_concept_names(
        model_cfg.get("selected_concepts", checkpoint_model_cfg.get("selected_concepts"))
    )
    concept_columns = concept_names_to_columns(selected_concept_names)
    split_base_root = Path(paths_cfg.get("split_base_root", REPO_ROOT.parent / "dataset" / "splited_data"))
    concept_label_csv = Path(paths_cfg.get("concept_label_csv", REPO_ROOT.parent / "dataset" / "concept_label" / "concept_labels.csv"))
    concept_scaler_json = Path(paths_cfg.get("concept_scaler_json", REPO_ROOT.parent / "dataset" / "concept_label" / "concept_scaler_stats.json"))

    batch_size = int(args.batch_size if args.batch_size is not None else train_cfg.get("batch_size", 1))
    num_workers = int(args.num_workers if args.num_workers is not None else train_cfg.get("num_workers", 2))
    scaler = load_concept_scaler(concept_scaler_json, concept_names=selected_concept_names)
    _, loaders = build_eval_datasets_and_loaders(
        split_base_root=split_base_root,
        concept_label_csv=concept_label_csv,
        concept_scaler_json=concept_scaler_json,
        data_cfg=data_cfg,
        batch_size=batch_size,
        num_workers=num_workers,
        concept_columns=concept_columns,
    )

    device = torch.device(args.device)
    model = _load_model_from_checkpoint(args.checkpoint, device=device)
    run_id = _resolve_run_id(args.run_id)
    output_dir = args.output_dir or Path(paths_cfg.get("results_root", REPO_ROOT.parent / "results" / "habitat_CBM_3D")) / f"eval_{run_id}"
    splits = tuple(item.strip() for item in args.splits.split(",") if item.strip())
    threshold = float(args.threshold if args.threshold is not None else eval_cfg.get("threshold", 0.5))
    topk_pool = int(args.topk_pool if args.topk_pool is not None else eval_cfg.get("topk_pool", 0))

    exported = run_full_evaluation(
        model=model,
        dataloaders=loaders,
        scaler=scaler,
        output_dir=output_dir,
        run_id=run_id,
        checkpoint_name=args.checkpoint.name,
        threshold=threshold,
        topk_pool=topk_pool,
        device=device,
        include_splits=splits,
        config_path=str(args.config),
        export_png=bool(eval_cfg.get("export_png", False)),
        figure_include_splits=tuple(eval_cfg.get("figure_include_splits", splits)),
        figure_dpi=int(eval_cfg.get("figure_dpi", 150)),
    )
    print("3D evaluation exported:")
    for key, path in exported.items():
        print(f"  {key}: {path}")


if __name__ == "__main__":
    main()
