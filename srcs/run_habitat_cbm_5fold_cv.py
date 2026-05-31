#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare and optionally run patient-level stratified 5-fold CV for Habitat-CBM.

The script does not change train_habitat_CBM.py. For each fold it creates:
- a train/val/test split directory compatible with the existing data loader;
- fold-specific concept_labels.csv with an updated split column;
- fold-specific concept_scaler_stats.json fitted only on that fold's train split;
- a full JSON training config consumed by train_habitat_CBM.py;
- a fold_metadata.json record and a top-level cv_plan.json.

Default behavior is preparation only. Add --run to launch the five training jobs
sequentially after all fold assets are written.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import stat
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
PROJECT_ROOT = REPO_ROOT.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_split import (  # noqa: E402
    PatientRecord,
    materialize_split_dirs,
    read_records,
    summarise,
    validate_layout,
    write_manifest,
    write_subset_csv,
    write_text_list,
)

MODEL_NAME = "habitat_cbm"
DEFAULT_N_FOLDS = 5
DEFAULT_STD_FLOOR = 1e-6
CONCEPT_IDS = tuple(f"c{i}" for i in range(1, 9))


def _load_json(path: Path) -> Dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"JSON not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _save_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"CSV not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return [dict(row) for row in reader]


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _as_dict(value: object, *, name: str) -> Dict[str, object]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"Config field '{name}' must be an object.")
    return dict(value)


def _normalize_patient_id(value: object) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError("Empty patient id.")
    if text.isdigit():
        return f"{int(text):03d}"
    return text


def _repo_relative_or_base(default_path: Path, fallback: Path) -> Path:
    return default_path if default_path.exists() else fallback


def _infer_dataset_root(base_cfg: Mapping[str, object]) -> Path:
    local_dataset = PROJECT_ROOT / "dataset"
    if (local_dataset / "idh.csv").exists() or (local_dataset / "images" / "idh.csv").exists():
        return local_dataset

    paths_cfg = _as_dict(base_cfg.get("paths", {}), name="paths")
    concept_label = paths_cfg.get("concept_label_csv")
    if concept_label:
        concept_label_path = Path(str(concept_label))
        if concept_label_path.name:
            return concept_label_path.parent.parent

    split_root = paths_cfg.get("split_base_root")
    if split_root:
        return Path(str(split_root)).parent

    return local_dataset


def _default_concept_label_csv(base_cfg: Mapping[str, object]) -> Path:
    local_path = PROJECT_ROOT / "dataset" / "concept_label" / "concept_labels.csv"
    paths_cfg = _as_dict(base_cfg.get("paths", {}), name="paths")
    fallback = Path(str(paths_cfg.get("concept_label_csv", local_path)))
    return _repo_relative_or_base(local_path, fallback)


def _default_concept_scaler_json(base_cfg: Mapping[str, object]) -> Path:
    local_path = PROJECT_ROOT / "dataset" / "concept_label" / "concept_scaler_stats.json"
    paths_cfg = _as_dict(base_cfg.get("paths", {}), name="paths")
    fallback = Path(str(paths_cfg.get("concept_scaler_json", local_path)))
    return _repo_relative_or_base(local_path, fallback)


def _safe_float(value: object, *, field: str, patient_id: str) -> float:
    text = str(value).strip()
    if text == "":
        raise ValueError(f"Empty value for {field} (patient={patient_id}).")
    try:
        parsed = float(text)
    except ValueError as exc:
        raise ValueError(f"Invalid float for {field} (patient={patient_id}): {text}") from exc
    if not np.isfinite(parsed):
        raise ValueError(f"Non-finite value for {field} (patient={patient_id}): {parsed}")
    return parsed


def _concept_columns(rows: Sequence[Mapping[str, str]]) -> Tuple[str, ...]:
    if not rows:
        raise ValueError("Concept label CSV is empty.")
    columns = tuple(col for col in rows[0].keys() if col.endswith("_true"))
    expected = tuple(f"{cid}_true" for cid in CONCEPT_IDS)
    missing = [col for col in expected if col not in columns]
    if missing:
        raise ValueError(f"Concept label CSV missing columns: {missing}")
    return expected


def _load_source_mapping(reference_scaler_json: Path, concept_columns: Sequence[str]) -> Dict[str, str]:
    fallback = {col[:-5]: col for col in concept_columns}
    if not reference_scaler_json.is_file():
        return fallback
    payload = _load_json(reference_scaler_json)
    concepts = payload.get("concepts", {})
    if not isinstance(concepts, Mapping):
        return fallback
    mapping: Dict[str, str] = {}
    for col in concept_columns:
        cid = col[:-5]
        item = concepts.get(cid, {})
        source = item.get("source") if isinstance(item, Mapping) else None
        mapping[cid] = str(source) if source else fallback[cid]
    return mapping


def _validate_concept_rows(
    concept_rows: Sequence[Mapping[str, str]],
    records: Sequence[PatientRecord],
    concept_columns: Sequence[str],
) -> Dict[str, Dict[str, str]]:
    by_patient: Dict[str, Dict[str, str]] = {}
    for row in concept_rows:
        patient_id = _normalize_patient_id(row.get("patient_id", ""))
        if patient_id in by_patient:
            raise ValueError(f"Duplicate patient in concept labels: {patient_id}")
        by_patient[patient_id] = dict(row)

    missing = [record.patient_id for record in records if record.patient_id not in by_patient]
    if missing:
        raise ValueError(f"Concept labels missing patients: {missing[:20]}")

    for record in records:
        row = by_patient[record.patient_id]
        y_true = int(float(str(row.get("y_true", "")).strip()))
        if y_true != int(record.idh_label):
            raise ValueError(
                f"IDH mismatch for patient {record.patient_id}: "
                f"idh.csv={record.idh_label}, concept_labels={y_true}"
            )
        for col in concept_columns:
            _safe_float(row[col], field=col, patient_id=record.patient_id)

    return by_patient


def _assign_round_robin_folds(
    records: Sequence[PatientRecord],
    *,
    n_folds: int,
    seed: int,
) -> List[List[PatientRecord]]:
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2.")

    by_label: Dict[str, List[PatientRecord]] = {}
    for record in records:
        by_label.setdefault(record.idh_label, []).append(record)

    for label, group in by_label.items():
        if len(group) < n_folds:
            raise ValueError(f"Class {label} has fewer patients ({len(group)}) than folds ({n_folds}).")

    rng = random.Random(seed)
    folds: List[List[PatientRecord]] = [[] for _ in range(n_folds)]
    for label in sorted(by_label):
        group = list(by_label[label])
        rng.shuffle(group)
        for index, record in enumerate(group):
            folds[index % n_folds].append(record)

    return [sorted(fold, key=lambda item: int(item.patient_id)) for fold in folds]


def _split_train_val(
    trainval_records: Sequence[PatientRecord],
    *,
    val_ratio: float,
    seed: int,
) -> Tuple[List[PatientRecord], List[PatientRecord]]:
    if not 0.0 < val_ratio < 0.5:
        raise ValueError(f"val_ratio must be in (0, 0.5), got {val_ratio}")

    rng = random.Random(seed)
    by_label: Dict[str, List[PatientRecord]] = {}
    for record in trainval_records:
        by_label.setdefault(record.idh_label, []).append(record)

    train: List[PatientRecord] = []
    val: List[PatientRecord] = []
    for label in sorted(by_label):
        group = list(by_label[label])
        rng.shuffle(group)
        n_val = int(round(len(group) * val_ratio))
        if len(group) >= 2:
            n_val = min(max(n_val, 1), len(group) - 1)
        else:
            n_val = 0
        val.extend(group[:n_val])
        train.extend(group[n_val:])

    return (
        sorted(train, key=lambda item: int(item.patient_id)),
        sorted(val, key=lambda item: int(item.patient_id)),
    )


def _counter_payload(records: Sequence[PatientRecord]) -> Dict[str, int]:
    counts = Counter(record.idh_label for record in records)
    return {
        "total": len(records),
        "wild_type": int(counts.get("0", 0)),
        "mutant": int(counts.get("1", 0)),
    }


def _write_fold_manifests(
    fold_dir: Path,
    splits: Mapping[str, Sequence[PatientRecord]],
    *,
    seed: int,
) -> None:
    manifests_root = fold_dir / "split" / "manifests"
    manifests_root.mkdir(parents=True, exist_ok=True)
    write_manifest(manifests_root / "split_assignments.csv", dict(splits), seed)
    fieldnames = list(next(iter(splits.values()))[0].raw.keys())
    for split_name in ("train", "val", "test"):
        records = list(splits[split_name])
        write_text_list(manifests_root / f"{split_name}_ids.txt", [record.patient_id for record in records])
        write_subset_csv(manifests_root / f"{split_name}.csv", records, fieldnames)


def _rewrite_concept_assets(
    *,
    output_dir: Path,
    concept_by_patient: Mapping[str, Mapping[str, str]],
    splits: Mapping[str, Sequence[PatientRecord]],
    concept_columns: Sequence[str],
    source_mapping: Mapping[str, str],
    std_floor: float,
) -> Tuple[Path, Path, Path]:
    split_by_patient: Dict[str, str] = {}
    for split_name, records in splits.items():
        for record in records:
            split_by_patient[record.patient_id] = split_name

    label_rows: List[Dict[str, object]] = []
    for split_name in ("train", "val", "test"):
        for record in splits[split_name]:
            source = concept_by_patient[record.patient_id]
            row: Dict[str, object] = {
                "patient_id": record.patient_id,
                "split": split_by_patient[record.patient_id],
                "y_true": int(record.idh_label),
                "label_name": record.idh_class,
            }
            for col in concept_columns:
                row[col] = _safe_float(source[col], field=col, patient_id=record.patient_id)
            label_rows.append(row)

    label_csv = output_dir / "concept_labels.csv"
    stats_csv = output_dir / "concept_statistics.csv"
    scaler_json = output_dir / "concept_scaler_stats.json"
    label_fields = ["patient_id", "split", "y_true", "label_name", *concept_columns]
    _write_csv(label_csv, label_fields, label_rows)

    stats_rows: List[Dict[str, object]] = []
    for concept_col in concept_columns:
        cid = concept_col[:-5]
        for split_name in ("all", "train", "val", "test"):
            rows = label_rows if split_name == "all" else [row for row in label_rows if row["split"] == split_name]
            values = np.asarray([float(row[concept_col]) for row in rows], dtype=np.float64)
            stats_rows.append(
                {
                    "concept_id": cid,
                    "source_column": source_mapping.get(cid, concept_col),
                    "split": split_name,
                    "count": int(values.size),
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=0)),
                    "min": float(values.min()),
                    "p01": float(np.percentile(values, 1)),
                    "p50": float(np.percentile(values, 50)),
                    "p99": float(np.percentile(values, 99)),
                    "max": float(values.max()),
                }
            )
    _write_csv(
        stats_csv,
        ["concept_id", "source_column", "split", "count", "mean", "std", "min", "p01", "p50", "p99", "max"],
        stats_rows,
    )

    train_rows = [row for row in label_rows if row["split"] == "train"]
    concepts: Dict[str, Dict[str, object]] = {}
    for concept_col in concept_columns:
        cid = concept_col[:-5]
        values = np.asarray([float(row[concept_col]) for row in train_rows], dtype=np.float64)
        std_raw = float(values.std(ddof=0))
        concepts[cid] = {
            "mean": float(values.mean()),
            "std": float(std_raw if std_raw > std_floor else std_floor),
            "std_raw": std_raw,
            "source": source_mapping.get(cid, concept_col),
        }
    _save_json(
        scaler_json,
        {
            "fit_split": "train",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "std_floor": float(std_floor),
            "concepts": concepts,
        },
    )
    return label_csv, stats_csv, scaler_json


def _deep_update_path_config(
    base_cfg: Mapping[str, object],
    *,
    split_root: Path,
    concept_label_csv: Path,
    concept_scaler_json: Path,
    fold_dir: Path,
    run_id: str,
    train_seed: int,
    device: Optional[str],
    num_workers: Optional[int],
    epochs_stage1: Optional[int],
    epochs_stage2: Optional[int],
    epochs_stage3: Optional[int],
) -> Dict[str, object]:
    cfg: Dict[str, object] = json.loads(json.dumps(base_cfg, ensure_ascii=False))
    paths_cfg = _as_dict(cfg.get("paths", {}), name="paths")
    paths_cfg["split_base_root"] = str(split_root.resolve())
    paths_cfg["concept_label_csv"] = str(concept_label_csv.resolve())
    paths_cfg["concept_scaler_json"] = str(concept_scaler_json.resolve())
    paths_cfg["runs_root"] = str((fold_dir / "runs").resolve())
    paths_cfg["results_root"] = str((fold_dir / "results").resolve())
    paths_cfg["checkpoint_root"] = None
    cfg["paths"] = paths_cfg

    train_cfg = _as_dict(cfg.get("train", {}), name="train")
    train_cfg["seed"] = int(train_seed)
    if device is not None:
        train_cfg["device"] = device
    if num_workers is not None:
        train_cfg["num_workers"] = int(num_workers)
    cfg["train"] = train_cfg

    stages_cfg = _as_dict(cfg.get("stages", {}), name="stages")
    if epochs_stage1 is not None:
        stage1 = _as_dict(stages_cfg.get("stage1", {}), name="stages.stage1")
        stage1["epochs"] = int(epochs_stage1)
        stages_cfg["stage1"] = stage1
    if epochs_stage2 is not None:
        stage2 = _as_dict(stages_cfg.get("stage2", {}), name="stages.stage2")
        stage2["epochs"] = int(epochs_stage2)
        stages_cfg["stage2"] = stage2
    if epochs_stage3 is not None:
        stage3 = _as_dict(stages_cfg.get("stage3", {}), name="stages.stage3")
        stage3["epochs"] = int(epochs_stage3)
        for step_name in ("step_a", "step_b"):
            if step_name in stage3 and isinstance(stage3[step_name], Mapping):
                step_cfg = dict(stage3[step_name])
                step_cfg["epochs"] = int(epochs_stage3)
                stage3[step_name] = step_cfg
        stages_cfg["stage3"] = stage3
    cfg["stages"] = stages_cfg

    logging_cfg = _as_dict(cfg.get("logging", {}), name="logging")
    logging_cfg["run_id"] = run_id
    cfg["logging"] = logging_cfg
    return cfg


def _shell_quote(path: Path | str) -> str:
    text = str(path)
    return "'" + text.replace("'", "'\"'\"'") + "'"


def _read_fold_test_metrics(metrics_path: Path, fold_name: str) -> Optional[Dict[str, object]]:
    if not metrics_path.is_file():
        return None
    rows = _read_csv(metrics_path)
    for row in rows:
        if row.get("split") != "test":
            continue
        metric = {"fold": fold_name}
        for key, value in row.items():
            if key in {"auc", "acc", "sen", "spe", "f1", "loss_bce_block", "threshold"}:
                metric[key] = float(value)
            elif key in {"num_patients", "num_blocks"}:
                metric[key] = int(float(value))
            else:
                metric[key] = value
        return metric
    return None


def _write_cv_metric_summary(output_root: Path, fold_plans: Sequence[Mapping[str, object]]) -> None:
    rows: List[Dict[str, object]] = []
    for plan in fold_plans:
        metrics = _read_fold_test_metrics(Path(str(plan["metrics_csv"])), str(plan["fold_name"]))
        if metrics is not None:
            rows.append(metrics)
    if not rows:
        return

    fieldnames = sorted({key for row in rows for key in row.keys()})
    _write_csv(output_root / "cv_test_metrics_by_fold.csv", fieldnames, rows)

    aggregate: Dict[str, object] = {"num_completed_folds": len(rows), "metrics": {}}
    for metric_name in ("auc", "acc", "sen", "spe", "f1"):
        values = np.asarray([float(row[metric_name]) for row in rows], dtype=np.float64)
        aggregate["metrics"][metric_name] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if values.size >= 2 else 0.0,
            "min": float(values.min()),
            "median": float(np.median(values)),
            "max": float(values.max()),
        }
    _save_json(output_root / "cv_test_metrics_aggregate.json", aggregate)


def build_argparser() -> argparse.ArgumentParser:
    base_config_default = CURRENT_DIR / "args_train_habitat_CBM.json"
    parser = argparse.ArgumentParser(
        description="Prepare and optionally run 5-fold CV for Habitat-CBM."
    )
    parser.add_argument("--base-config", type=Path, default=base_config_default)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--concept-label-csv", type=Path, default=None)
    parser.add_argument("--concept-scaler-json", type=Path, default=None)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "results" / "habitat_CBM_5fold_cv",
    )
    parser.add_argument("--n-folds", type=int, default=DEFAULT_N_FOLDS)
    parser.add_argument("--seed", type=int, default=2026, help="Seed for fold and validation split generation.")
    parser.add_argument(
        "--val-ratio-within-trainval",
        type=float,
        default=0.125,
        help="Validation ratio within the 80 percent non-test portion. 0.125 gives about 70/10/20.",
    )
    parser.add_argument("--train-seed", type=int, default=None, help="Training seed written into every fold JSON.")
    parser.add_argument(
        "--link-mode",
        choices=("symlink", "copy", "none"),
        default="symlink",
        help="How to materialize split folders. Use symlink for CV unless the server forbids it.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing fold directories.")
    parser.add_argument(
        "--skip-layout-validation",
        action="store_true",
        help="Skip conventional/functional image folder validation before materializing splits.",
    )
    parser.add_argument("--std-floor", type=float, default=DEFAULT_STD_FLOOR)
    parser.add_argument("--run-id-prefix", type=str, default="habitat_cbm_cv5")
    parser.add_argument("--train-script", type=Path, default=CURRENT_DIR / "train_habitat_CBM.py")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--epochs-stage1", type=int, default=None)
    parser.add_argument("--epochs-stage2", type=int, default=None)
    parser.add_argument("--epochs-stage3", type=int, default=None)
    parser.add_argument("--run", action="store_true", help="Run train_habitat_CBM.py for each fold after preparation.")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    base_cfg = _load_json(args.base_config)

    dataset_root = args.dataset_root if args.dataset_root is not None else _infer_dataset_root(base_cfg)
    concept_label_csv = (
        args.concept_label_csv if args.concept_label_csv is not None else _default_concept_label_csv(base_cfg)
    )
    concept_scaler_json = (
        args.concept_scaler_json
        if args.concept_scaler_json is not None
        else _default_concept_scaler_json(base_cfg)
    )
    output_root = args.output_root.resolve()
    train_seed = int(args.train_seed) if args.train_seed is not None else int(_as_dict(base_cfg.get("train", {}), name="train").get("seed", 42))

    if args.run and args.link_mode == "none":
        raise ValueError("--run cannot be used with --link-mode none; training needs materialized patient folders.")

    records = read_records(dataset_root / "idh.csv")
    if args.link_mode != "none" and not args.skip_layout_validation:
        validate_layout(dataset_root, records, require_voi=bool(_as_dict(base_cfg.get("data", {}), name="data").get("require_voi", True)))

    concept_rows = _read_csv(concept_label_csv)
    concept_columns = _concept_columns(concept_rows)
    concept_by_patient = _validate_concept_rows(concept_rows, records, concept_columns)
    source_mapping = _load_source_mapping(concept_scaler_json, concept_columns)

    output_root.mkdir(parents=True, exist_ok=True)
    test_folds = _assign_round_robin_folds(records, n_folds=args.n_folds, seed=args.seed)
    all_patient_ids = {record.patient_id for record in records}

    fold_plans: List[Dict[str, object]] = []
    commands: List[List[str]] = []
    for fold_index, test_records in enumerate(test_folds, start=1):
        fold_name = f"fold_{fold_index:02d}"
        fold_dir = output_root / "folds" / fold_name
        if fold_dir.exists():
            if not args.overwrite:
                raise FileExistsError(f"Fold directory already exists: {fold_dir}. Use --overwrite to replace it.")
            shutil.rmtree(fold_dir)
        fold_dir.mkdir(parents=True, exist_ok=True)

        test_ids = {record.patient_id for record in test_records}
        trainval_records = [record for record in records if record.patient_id not in test_ids]
        train_records, val_records = _split_train_val(
            trainval_records,
            val_ratio=args.val_ratio_within_trainval,
            seed=args.seed + fold_index * 1009,
        )
        split_ids = {record.patient_id for record in train_records + val_records + test_records}
        if split_ids != all_patient_ids:
            raise RuntimeError(f"{fold_name} does not cover exactly all patients.")

        splits = {"train": train_records, "val": val_records, "test": test_records}
        split_root = fold_dir / "split"
        _write_fold_manifests(fold_dir, splits, seed=args.seed)
        materialize_split_dirs(dataset_root, split_root, splits, args.link_mode)

        concept_dir = fold_dir / "concept_label"
        fold_label_csv, fold_stats_csv, fold_scaler_json = _rewrite_concept_assets(
            output_dir=concept_dir,
            concept_by_patient=concept_by_patient,
            splits=splits,
            concept_columns=concept_columns,
            source_mapping=source_mapping,
            std_floor=args.std_floor,
        )

        run_id = f"{args.run_id_prefix}_seed{args.seed}_fold{fold_index:02d}"
        fold_train_config = _deep_update_path_config(
            base_cfg,
            split_root=split_root,
            concept_label_csv=fold_label_csv,
            concept_scaler_json=fold_scaler_json,
            fold_dir=fold_dir,
            run_id=run_id,
            train_seed=train_seed,
            device=args.device,
            num_workers=args.num_workers,
            epochs_stage1=args.epochs_stage1,
            epochs_stage2=args.epochs_stage2,
            epochs_stage3=args.epochs_stage3,
        )
        train_config_path = fold_dir / f"train_config_{MODEL_NAME}_{fold_name}.json"
        _save_json(train_config_path, fold_train_config)

        command = [str(args.python), str(args.train_script), "--config", str(train_config_path)]
        result_dir = fold_dir / "results" / run_id
        plan = {
            "fold_name": fold_name,
            "fold_index": fold_index,
            "run_id": run_id,
            "train_config_json": str(train_config_path),
            "fold_metadata_json": str(fold_dir / "fold_metadata.json"),
            "split_root": str(split_root),
            "concept_label_csv": str(fold_label_csv),
            "concept_statistics_csv": str(fold_stats_csv),
            "concept_scaler_json": str(fold_scaler_json),
            "result_dir": str(result_dir),
            "metrics_csv": str(result_dir / f"metrics_{MODEL_NAME}_{run_id}.csv"),
            "patient_predictions_csv": str(result_dir / f"patient_predictions_{MODEL_NAME}_{run_id}.csv"),
            "command": command,
            "counts": {
                "train": _counter_payload(train_records),
                "val": _counter_payload(val_records),
                "test": _counter_payload(test_records),
            },
            "patient_ids": {
                "train": [record.patient_id for record in train_records],
                "val": [record.patient_id for record in val_records],
                "test": [record.patient_id for record in test_records],
            },
        }
        _save_json(fold_dir / "fold_metadata.json", plan)
        fold_plans.append(plan)
        commands.append(command)

        print(f"[{fold_name}]")
        print(summarise(splits))
        print(f"config: {train_config_path}")

    cv_plan = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "base_config": str(args.base_config.resolve()),
        "dataset_root": str(dataset_root.resolve()),
        "source_concept_label_csv": str(concept_label_csv.resolve()),
        "source_concept_scaler_json": str(concept_scaler_json.resolve()),
        "output_root": str(output_root),
        "n_folds": int(args.n_folds),
        "seed": int(args.seed),
        "train_seed": int(train_seed),
        "val_ratio_within_trainval": float(args.val_ratio_within_trainval),
        "link_mode": args.link_mode,
        "folds": fold_plans,
    }
    _save_json(output_root / "cv_plan.json", cv_plan)

    run_script_path = output_root / "run_all_folds.sh"
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", f"cd {_shell_quote(REPO_ROOT)}", ""]
    for command in commands:
        lines.append(" ".join(_shell_quote(item) for item in command))
    run_script_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    current_mode = run_script_path.stat().st_mode
    run_script_path.chmod(current_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    print(f"\nCV assets written to: {output_root}")
    print(f"Plan JSON: {output_root / 'cv_plan.json'}")
    print(f"Run script: {run_script_path}")

    if args.run:
        for command in commands:
            print("\n[run] " + " ".join(_shell_quote(item) for item in command))
            subprocess.run(command, check=True)
        _write_cv_metric_summary(output_root, fold_plans)
        print(f"\nCompleted CV run. Metric summary directory: {output_root}")
    else:
        print("Preparation only. Add --run or execute run_all_folds.sh to train all folds.")


if __name__ == "__main__":
    main()
