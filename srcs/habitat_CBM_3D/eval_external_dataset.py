#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate a Habitat-CBM 3D checkpoint on a legacy splited_data external set.

The external set is expected to use the existing 2D/legacy directory protocol:

splited_data/
  train/{conventional,functional}/{mutant,wild_type}/<patient_id>/
  val/{conventional,functional}/{mutant,wild_type}/<patient_id>/
  test/{conventional,functional}/{mutant,wild_type}/<patient_id>/

For 3D evaluation, only T1/T1c/T2/FLAIR are used as image channels. The
functional VOI is used for cropping when crop_with_voi=true, but it is not
passed as an image channel.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import srcs.eval_habitat_CBM as eval2d  # noqa: E402
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
from srcs.habitat_CBM_3D.eval_habitat_CBM_3D import (  # noqa: E402
    _load_json,
    _load_model_config_from_checkpoint,
    _load_model_from_checkpoint,
    _parse_int_triple,
)
from srcs.habitat_CBM_3D.monai_augmentation_3d import (  # noqa: E402
    MonaiVolumeAugmentConfig,
    build_monai_volume_transforms,
)

MODEL_NAME = "habitat_cbm_3d_external_splited_data"
DEFAULT_CONFIG = CURRENT_DIR / "args_train_habitat_CBM_3D_tuned_v1_noaug.json"
DEFAULT_EXTERNAL_SPLIT_ROOT = Path("/root/autodl-tmp/habitat_CBM/dataset/splited_data")


def _resolve_run_id(run_id: str | None) -> str:
    return run_id if run_id else f"external_splited_data_{time.strftime('%Y%m%d_%H%M%S')}"


def _split_csv(value: str) -> Tuple[str, ...]:
    splits = tuple(item.strip().lower() for item in value.split(",") if item.strip())
    invalid = [split for split in splits if split not in {"train", "val", "test"}]
    if invalid:
        raise ValueError(f"Unsupported split(s): {invalid}. Valid values are train,val,test.")
    if not splits:
        raise ValueError("At least one split must be selected.")
    return splits


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _save_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _maybe_auto_external_concept_label_csv(
    external_split_root: Path,
    explicit_path: Optional[Path],
    no_concepts: bool,
) -> Optional[Path]:
    if no_concepts:
        return None
    if explicit_path is not None:
        return explicit_path
    candidate = external_split_root.parent / "concept_label" / "concept_labels.csv"
    return candidate if candidate.is_file() else None


def _resolve_concept_scaler_json(
    *,
    explicit_path: Optional[Path],
    paths_cfg: Mapping[str, object],
    concept_label_csv: Optional[Path],
) -> Optional[Path]:
    if concept_label_csv is None:
        return None
    if explicit_path is not None:
        return explicit_path
    configured = paths_cfg.get("concept_scaler_json")
    if configured:
        return Path(str(configured))
    raise ValueError(
        "Concept labels were enabled, but no scaler JSON was provided. "
        "Pass --concept-scaler-json, or use --no-concepts for classification-only evaluation."
    )


def _build_external_loaders(
    *,
    external_split_root: Path,
    concept_label_csv: Optional[Path],
    concept_scaler_json: Optional[Path],
    concept_columns: Sequence[str],
    data_cfg: Mapping[str, object],
    batch_size: int,
    num_workers: int,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    target_shape = _parse_int_triple(
        data_cfg.get("target_shape"),
        DEFAULT_3D_TARGET_SHAPE,
        name="data.target_shape",
    )
    transform_map = build_monai_volume_transforms(
        config=MonaiVolumeAugmentConfig(enabled=False),
        spatial_size=target_shape,
    )
    datasets = build_habitat_cbm_3d_datasets(
        split_base_root=external_split_root,
        manifest_csv=None,
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
        patient_balanced_sampling=False,
    )
    return datasets, loaders


def _concept_row(
    *,
    patient_id: str,
    split: str,
    y_true: int,
    c_pred_std: np.ndarray,
    c_true_std: np.ndarray,
    c_true_raw: np.ndarray,
    scaler: ConceptScaler,
) -> Dict[str, object]:
    c_pred_raw = scaler.destandardize(c_pred_std)
    row: Dict[str, object] = {
        "patient_id": patient_id,
        "split": split,
        "y_true": y_true,
    }
    for idx, concept_name in enumerate(scaler.concept_names):
        row[f"{concept_name}_true_std"] = float(c_true_std[idx])
        row[f"{concept_name}_pred_std"] = float(c_pred_std[idx])
        row[f"{concept_name}_abs_error_std"] = float(abs(c_pred_std[idx] - c_true_std[idx]))
        row[f"{concept_name}_true_raw"] = float(c_true_raw[idx])
        row[f"{concept_name}_pred_raw"] = float(c_pred_raw[idx])
        row[f"{concept_name}_abs_error_raw"] = float(abs(c_pred_raw[idx] - c_true_raw[idx]))
    return row


def _concept_mae_metrics(rows: Sequence[Mapping[str, object]], scaler: Optional[ConceptScaler]) -> Dict[str, float]:
    if not rows or scaler is None:
        return {}
    metrics: Dict[str, float] = {}
    std_values: List[float] = []
    raw_values: List[float] = []
    for concept_name in scaler.concept_names:
        std_key = f"{concept_name}_abs_error_std"
        raw_key = f"{concept_name}_abs_error_raw"
        concept_std = [float(row[std_key]) for row in rows]
        concept_raw = [float(row[raw_key]) for row in rows]
        metrics[f"{concept_name}_mae_std"] = float(np.mean(concept_std, dtype=np.float64))
        metrics[f"{concept_name}_mae_raw"] = float(np.mean(concept_raw, dtype=np.float64))
        std_values.extend(concept_std)
        raw_values.extend(concept_raw)
    metrics["concept_mae_std_mean"] = float(np.mean(std_values, dtype=np.float64))
    metrics["concept_mae_raw_mean"] = float(np.mean(raw_values, dtype=np.float64))
    return metrics


@torch.no_grad()
def evaluate_external_split(
    *,
    model,
    dataloader,
    split: str,
    threshold: float,
    device: torch.device,
    scaler: Optional[ConceptScaler],
) -> Dict[str, object]:
    model.eval()
    prediction_rows: List[Dict[str, object]] = []
    concept_rows: List[Dict[str, object]] = []
    total_bce = 0.0
    total_count = 0

    for batch in dataloader:
        images = batch["image"].to(device=device, dtype=torch.float32, non_blocking=True)
        y_true = batch["label"].to(device=device, dtype=torch.float32, non_blocking=True)

        out = model.forward_x_to_cy(images)
        y_logit = out["y_logit"].squeeze(1)
        y_prob = torch.sigmoid(y_logit)
        c_pred_std = out["c_hat"]

        bce = F.binary_cross_entropy_with_logits(y_logit, y_true)
        batch_size = int(y_true.shape[0])
        total_bce += float(bce.item()) * batch_size
        total_count += batch_size

        patient_ids = batch["patient_id"]
        for idx in range(batch_size):
            patient_id = str(patient_ids[idx])
            true_label = int(y_true[idx].item())
            prob = float(y_prob[idx].item())
            prediction_rows.append(
                {
                    "patient_id": patient_id,
                    "split": split,
                    "y_true": true_label,
                    "prob_idh_mut": prob,
                    "pred_label": int(prob >= threshold),
                }
            )

        if scaler is not None and "concept_true_std" in batch and "concept_true_raw" in batch:
            c_true_std = batch["concept_true_std"].to(dtype=torch.float32)
            c_true_raw = batch["concept_true_raw"].to(dtype=torch.float32)
            c_pred_np = c_pred_std.detach().cpu().numpy().astype(np.float32)
            c_true_std_np = c_true_std.detach().cpu().numpy().astype(np.float32)
            c_true_raw_np = c_true_raw.detach().cpu().numpy().astype(np.float32)
            for idx in range(batch_size):
                concept_rows.append(
                    _concept_row(
                        patient_id=str(patient_ids[idx]),
                        split=split,
                        y_true=int(y_true[idx].item()),
                        c_pred_std=c_pred_np[idx],
                        c_true_std=c_true_std_np[idx],
                        c_true_raw=c_true_raw_np[idx],
                        scaler=scaler,
                    )
                )

    metrics = eval2d.compute_patient_metrics(prediction_rows)
    metrics["loss_bce"] = total_bce / max(total_count, 1)
    metrics["num_patients"] = len(prediction_rows)
    metrics.update(_concept_mae_metrics(concept_rows, scaler))
    return {
        "split": split,
        "patient_prediction_rows": prediction_rows,
        "patient_concept_rows": concept_rows,
        "metrics": metrics,
    }


def _metric_row(split: str, metrics: Mapping[str, object]) -> Dict[str, object]:
    row: Dict[str, object] = {
        "split": split,
        "num_patients": metrics.get("num_patients", 0),
        "loss_bce": metrics.get("loss_bce", float("nan")),
        "auc": metrics.get("auc", float("nan")),
        "acc": metrics.get("acc", float("nan")),
        "sen": metrics.get("sen", float("nan")),
        "spe": metrics.get("spe", float("nan")),
        "f1": metrics.get("f1", float("nan")),
        "tn": metrics.get("tn", 0),
        "fp": metrics.get("fp", 0),
        "fn": metrics.get("fn", 0),
        "tp": metrics.get("tp", 0),
    }
    for key in sorted(metrics):
        if key.startswith("c") and ("_mae_" in key):
            row[key] = metrics[key]
    for key in ("concept_mae_std_mean", "concept_mae_raw_mean"):
        if key in metrics:
            row[key] = metrics[key]
    return row


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a Habitat-CBM 3D checkpoint on legacy splited_data as an external test set."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--external-split-root", type=Path, default=DEFAULT_EXTERNAL_SPLIT_ROOT)
    parser.add_argument("--splits", type=str, default="train,val,test")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument(
        "--concept-label-csv",
        type=Path,
        default=None,
        help=(
            "Optional external concept_labels.csv. If omitted, the script tries "
            "<external-split-root>/../concept_label/concept_labels.csv and falls back to classification-only."
        ),
    )
    parser.add_argument(
        "--concept-scaler-json",
        type=Path,
        default=None,
        help=(
            "Scaler used to standardize external concepts. Defaults to paths.concept_scaler_json "
            "from --config when concept labels are enabled."
        ),
    )
    parser.add_argument("--no-concepts", action="store_true", help="Run classification-only external evaluation.")
    return parser


def main() -> None:
    args = _build_argparser().parse_args()
    cfg = _load_json(args.config)
    paths_cfg = cfg.get("paths", {})
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("train", {})
    eval_cfg = cfg.get("eval", {})
    if not isinstance(paths_cfg, Mapping):
        paths_cfg = {}
    if not isinstance(data_cfg, Mapping):
        data_cfg = {}
    if not isinstance(model_cfg, Mapping):
        model_cfg = {}
    if not isinstance(train_cfg, Mapping):
        train_cfg = {}
    if not isinstance(eval_cfg, Mapping):
        eval_cfg = {}

    checkpoint_model_cfg = _load_model_config_from_checkpoint(args.checkpoint)
    selected_concepts = resolve_concept_names(
        checkpoint_model_cfg.get("selected_concepts", model_cfg.get("selected_concepts"))
    )
    concept_columns = concept_names_to_columns(selected_concepts)

    concept_label_csv = _maybe_auto_external_concept_label_csv(
        external_split_root=args.external_split_root,
        explicit_path=args.concept_label_csv,
        no_concepts=bool(args.no_concepts),
    )
    concept_scaler_json = _resolve_concept_scaler_json(
        explicit_path=args.concept_scaler_json,
        paths_cfg=paths_cfg,
        concept_label_csv=concept_label_csv,
    )
    scaler = (
        load_concept_scaler(concept_scaler_json, concept_names=selected_concepts)
        if concept_scaler_json is not None
        else None
    )

    batch_size = int(args.batch_size if args.batch_size is not None else train_cfg.get("batch_size", 1))
    num_workers = int(args.num_workers if args.num_workers is not None else train_cfg.get("num_workers", 2))
    threshold = float(args.threshold if args.threshold is not None else eval_cfg.get("threshold", 0.5))
    splits = _split_csv(args.splits)

    _, loaders = _build_external_loaders(
        external_split_root=args.external_split_root,
        concept_label_csv=concept_label_csv,
        concept_scaler_json=concept_scaler_json,
        concept_columns=concept_columns,
        data_cfg=data_cfg,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    requested_device = args.device if args.device is not None else str(train_cfg.get("device", "cuda"))
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        requested_device = "cpu"
    device = torch.device(requested_device)
    model = _load_model_from_checkpoint(args.checkpoint, device=device)

    run_id = _resolve_run_id(args.run_id)
    results_root_value = paths_cfg.get("results_root") or REPO_ROOT.parent / "results" / "habitat_CBM_3D"
    results_root = Path(results_root_value)
    output_dir = args.output_dir or results_root / f"external_splited_data_{run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    prediction_rows: List[Dict[str, object]] = []
    concept_rows: List[Dict[str, object]] = []
    metric_rows: List[Dict[str, object]] = []

    for split in splits:
        output = evaluate_external_split(
            model=model,
            dataloader=loaders[split],
            split=split,
            threshold=threshold,
            device=device,
            scaler=scaler,
        )
        prediction_rows.extend(output["patient_prediction_rows"])
        concept_rows.extend(output["patient_concept_rows"])
        metric_rows.append(_metric_row(split, output["metrics"]))

    if len(splits) > 1:
        overall_metrics = eval2d.compute_patient_metrics(prediction_rows)
        overall_metrics["num_patients"] = len(prediction_rows)
        total_loss = 0.0
        total_patients = 0
        for row in metric_rows:
            n_patients = int(row.get("num_patients", 0))
            total_loss += float(row.get("loss_bce", 0.0)) * n_patients
            total_patients += n_patients
        overall_metrics["loss_bce"] = total_loss / max(total_patients, 1)
        overall_metrics.update(_concept_mae_metrics(concept_rows, scaler))
        metric_rows.append(_metric_row("overall", overall_metrics))

    pred_path = output_dir / f"patient_predictions_{MODEL_NAME}_{run_id}.csv"
    metric_path = output_dir / f"metrics_{MODEL_NAME}_{run_id}.csv"
    concept_path = output_dir / f"concept_predictions_{MODEL_NAME}_{run_id}.csv"
    summary_path = output_dir / f"run_summary_{MODEL_NAME}_{run_id}.json"

    pred_fields = ("patient_id", "split", "y_true", "prob_idh_mut", "pred_label")
    _write_csv(pred_path, pred_fields, prediction_rows)

    metric_fields = sorted({key for row in metric_rows for key in row.keys()})
    ordered_metric_fields = [
        key
        for key in (
            "split",
            "num_patients",
            "loss_bce",
            "auc",
            "acc",
            "sen",
            "spe",
            "f1",
            "tn",
            "fp",
            "fn",
            "tp",
            "concept_mae_std_mean",
            "concept_mae_raw_mean",
        )
        if key in metric_fields
    ]
    ordered_metric_fields.extend(key for key in metric_fields if key not in ordered_metric_fields)
    _write_csv(metric_path, ordered_metric_fields, metric_rows)

    exported: Dict[str, str] = {
        "patient_predictions": str(pred_path),
        "metrics": str(metric_path),
        "run_summary": str(summary_path),
    }
    if concept_rows:
        concept_fields = list(concept_rows[0].keys())
        _write_csv(concept_path, concept_fields, concept_rows)
        exported["concept_predictions"] = str(concept_path)

    summary = {
        "model_name": MODEL_NAME,
        "run_id": run_id,
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "external_split_root": str(args.external_split_root),
        "splits": list(splits),
        "device": str(device),
        "threshold": threshold,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "selected_concepts": list(selected_concepts),
        "concept_label_csv": str(concept_label_csv) if concept_label_csv is not None else None,
        "concept_scaler_json": str(concept_scaler_json) if concept_scaler_json is not None else None,
        "classification_only": concept_label_csv is None,
        "metrics": {str(row["split"]): row for row in metric_rows},
        "exports": exported,
    }
    _save_json(summary_path, summary)

    print("External 3D evaluation exported:")
    for key, path in exported.items():
        print(f"  {key}: {path}")
    print("Metrics:")
    for row in metric_rows:
        print(
            "  {split}: n={num_patients} auc={auc:.4f} acc={acc:.4f} "
            "sen={sen:.4f} spe={spe:.4f} f1={f1:.4f}".format(**row)
        )


if __name__ == "__main__":
    main()
