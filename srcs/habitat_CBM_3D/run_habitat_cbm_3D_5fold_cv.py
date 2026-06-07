#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare and optionally run manifest-based 5-fold CV for UCSF Habitat-CBM 3D."""

from __future__ import annotations

import argparse
import copy
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
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parents[1]
PROJECT_ROOT = REPO_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from srcs.habitat_CBM_3D.ucsf_pdgm_3d_utils import MANIFEST_FIELDS, write_csv_rows  # noqa: E402

MODEL_NAME = "habitat_cbm_3d"
DEFAULT_N_FOLDS = 5
DEFAULT_STD_FLOOR = 1e-6


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare manifest-based 5-fold CV for Habitat-CBM 3D.")
    parser.add_argument("--base-config", type=Path, default=CURRENT_DIR / "args_train_habitat_CBM_3D.json")
    parser.add_argument("--manifest-csv", type=Path, default=None)
    parser.add_argument("--concept-label-csv", type=Path, default=None)
    parser.add_argument("--concept-scaler-json", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "results" / "habitat_CBM_3D_5fold_cv")
    parser.add_argument("--n-folds", type=int, default=DEFAULT_N_FOLDS)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--val-ratio-within-trainval", type=float, default=0.125)
    parser.add_argument("--train-seed", type=int, default=None)
    parser.add_argument("--std-floor", type=float, default=DEFAULT_STD_FLOOR)
    parser.add_argument("--run-id-prefix", type=str, default="habitat_cbm_3d_cv5")
    parser.add_argument("--train-script", type=Path, default=CURRENT_DIR / "train_habitat_CBM_3D.py")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--epochs-stage1", type=int, default=None)
    parser.add_argument("--epochs-stage2", type=int, default=None)
    parser.add_argument("--epochs-stage3", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--run", action="store_true")
    return parser


def _load_json(path: Path) -> Dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON must be an object: {path}")
    return payload


def _save_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return [dict(row) for row in reader]


def _shell_quote(value: Path | str) -> str:
    text = str(value)
    return "'" + text.replace("'", "'\"'\"'") + "'"


def _concept_columns(rows: Sequence[Mapping[str, str]]) -> List[str]:
    if not rows:
        raise ValueError("Concept label CSV is empty.")
    cols = sorted([col for col in rows[0].keys() if col.lower().startswith("c") and col.lower().endswith("_true")])
    if not cols:
        raise ValueError("No concept columns found in concept label CSV.")
    return cols


def _assign_round_robin_folds(rows: Sequence[Mapping[str, str]], n_folds: int, seed: int) -> List[List[Mapping[str, str]]]:
    if n_folds <= 1:
        raise ValueError("n_folds must be > 1.")
    rng = random.Random(seed)
    groups: Dict[int, List[Mapping[str, str]]] = {0: [], 1: []}
    for row in rows:
        groups[int(float(row["y_true"]))].append(row)
    folds: List[List[Mapping[str, str]]] = [[] for _ in range(n_folds)]
    for group in groups.values():
        shuffled = list(group)
        rng.shuffle(shuffled)
        for idx, row in enumerate(shuffled):
            folds[idx % n_folds].append(row)
    return [sorted(fold, key=lambda item: str(item["patient_id"])) for fold in folds]


def _split_train_val(rows: Sequence[Mapping[str, str]], val_ratio: float, seed: int) -> tuple[List[Mapping[str, str]], List[Mapping[str, str]]]:
    if not (0.0 < val_ratio < 1.0):
        raise ValueError("val_ratio_within_trainval must be in (0,1).")
    rng = random.Random(seed)
    train: List[Mapping[str, str]] = []
    val: List[Mapping[str, str]] = []
    groups: Dict[int, List[Mapping[str, str]]] = {0: [], 1: []}
    for row in rows:
        groups[int(float(row["y_true"]))].append(row)
    for group in groups.values():
        shuffled = list(group)
        rng.shuffle(shuffled)
        n_val = int(round(len(shuffled) * val_ratio))
        if len(shuffled) >= 2:
            n_val = min(max(n_val, 1), len(shuffled) - 1)
        val.extend(shuffled[:n_val])
        train.extend(shuffled[n_val:])
    return (
        sorted(train, key=lambda item: str(item["patient_id"])),
        sorted(val, key=lambda item: str(item["patient_id"])),
    )


def _counter_payload(rows: Sequence[Mapping[str, str]]) -> Dict[str, int]:
    counter = Counter(int(float(row["y_true"])) for row in rows)
    return {"total": len(rows), "wild_type": counter.get(0, 0), "mutant": counter.get(1, 0)}


def _build_scaler(rows: Sequence[Mapping[str, str]], concept_columns: Sequence[str], source_mapping: Mapping[str, str], std_floor: float) -> Dict[str, object]:
    train_rows = [row for row in rows if row["split"] == "train"]
    if not train_rows:
        raise ValueError("No train rows available for fold scaler.")
    concepts: Dict[str, Dict[str, object]] = {}
    for col in concept_columns:
        concept_id = col[:-5].lower() if col.lower().endswith("_true") else col.lower()
        values = np.asarray([float(row[col]) for row in train_rows], dtype=np.float64)
        mean = float(values.mean())
        std_raw = float(values.std(ddof=0))
        concepts[concept_id] = {
            "mean": mean,
            "std": std_raw if std_raw > std_floor else float(std_floor),
            "std_raw": std_raw,
            "source": source_mapping.get(concept_id, col),
        }
    return {
        "fit_split": "train",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "std_floor": float(std_floor),
        "concepts": concepts,
    }


def _stats_rows(rows: Sequence[Mapping[str, str]], concept_columns: Sequence[str], source_mapping: Mapping[str, str]) -> List[Dict[str, object]]:
    output: List[Dict[str, object]] = []
    for split in ("all", "train", "val", "test"):
        split_rows = rows if split == "all" else [row for row in rows if row["split"] == split]
        if not split_rows:
            continue
        for col in concept_columns:
            concept_id = col[:-5].lower() if col.lower().endswith("_true") else col.lower()
            values = np.asarray([float(row[col]) for row in split_rows], dtype=np.float64)
            output.append(
                {
                    "concept_id": concept_id,
                    "source_column": source_mapping.get(concept_id, col),
                    "split": split,
                    "count": int(values.size),
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=0)),
                    "min": float(values.min()),
                    "p01": float(np.percentile(values, 1.0)),
                    "p50": float(np.percentile(values, 50.0)),
                    "p99": float(np.percentile(values, 99.0)),
                    "max": float(values.max()),
                }
            )
    return output


def _source_mapping_from_scaler(path: Path, concept_columns: Sequence[str]) -> Dict[str, str]:
    payload = _load_json(path)
    concepts = payload.get("concepts", {})
    if not isinstance(concepts, Mapping):
        return {col[:-5].lower(): col for col in concept_columns}
    mapping: Dict[str, str] = {}
    for col in concept_columns:
        concept_id = col[:-5].lower()
        item = concepts.get(concept_id, {})
        mapping[concept_id] = str(item.get("source", col)) if isinstance(item, Mapping) else col
    return mapping


def _rewrite_concept_assets(
    concept_rows_by_patient: Mapping[str, Mapping[str, str]],
    fold_rows: Sequence[Mapping[str, str]],
    output_dir: Path,
    concept_columns: Sequence[str],
    source_mapping: Mapping[str, str],
    std_floor: float,
) -> tuple[Path, Path, Path]:
    label_rows: List[Dict[str, object]] = []
    for row in fold_rows:
        patient_id = str(row["patient_id"])
        source = concept_rows_by_patient[patient_id]
        out_row: Dict[str, object] = {
            "patient_id": patient_id,
            "split": row["split"],
            "y_true": int(float(row["y_true"])),
            "label_name": row["label_name"],
        }
        for col in concept_columns:
            out_row[col] = source[col]
        label_rows.append(out_row)
    label_rows = sorted(label_rows, key=lambda item: (str(item["split"]), str(item["patient_id"])))
    label_csv = output_dir / "concept_labels.csv"
    stats_csv = output_dir / "concept_statistics.csv"
    scaler_json = output_dir / "concept_scaler_stats.json"
    write_csv_rows(label_csv, ("patient_id", "split", "y_true", "label_name", *concept_columns), label_rows)
    write_csv_rows(
        stats_csv,
        ("concept_id", "source_column", "split", "count", "mean", "std", "min", "p01", "p50", "p99", "max"),
        _stats_rows(label_rows, concept_columns, source_mapping),
    )
    _save_json(scaler_json, _build_scaler(label_rows, concept_columns, source_mapping, std_floor))
    return label_csv, stats_csv, scaler_json


def _stage_cfg(cfg: Dict[str, object], stage_name: str) -> Dict[str, object]:
    stages = cfg.setdefault("stages", {})
    if not isinstance(stages, dict):
        raise ValueError("base config stages must be an object.")
    stage = stages.setdefault(stage_name, {})
    if not isinstance(stage, dict):
        raise ValueError(f"base config stages.{stage_name} must be an object.")
    return stage


def _build_fold_config(
    base_cfg: Mapping[str, object],
    manifest_csv: Path,
    concept_label_csv: Path,
    concept_scaler_json: Path,
    fold_dir: Path,
    run_id: str,
    train_seed: int,
    args: argparse.Namespace,
) -> Dict[str, object]:
    cfg = copy.deepcopy(dict(base_cfg))
    paths = cfg.setdefault("paths", {})
    if not isinstance(paths, dict):
        raise ValueError("base config paths must be an object.")
    paths["manifest_csv"] = str(manifest_csv)
    paths["split_base_root"] = None
    paths["concept_label_csv"] = str(concept_label_csv)
    paths["concept_scaler_json"] = str(concept_scaler_json)
    paths["runs_root"] = str(fold_dir / "runs")
    paths["results_root"] = str(fold_dir / "results")
    paths["checkpoint_root"] = None

    logging = cfg.setdefault("logging", {})
    if isinstance(logging, dict):
        logging["run_id"] = run_id
    train = cfg.setdefault("train", {})
    if not isinstance(train, dict):
        raise ValueError("base config train must be an object.")
    train["seed"] = int(train_seed)
    if args.device is not None:
        train["device"] = args.device
    if args.num_workers is not None:
        train["num_workers"] = int(args.num_workers)
    if args.epochs_stage1 is not None:
        _stage_cfg(cfg, "stage1")["epochs"] = int(args.epochs_stage1)
    if args.epochs_stage2 is not None:
        _stage_cfg(cfg, "stage2")["epochs"] = int(args.epochs_stage2)
    if args.epochs_stage3 is not None:
        stage3 = _stage_cfg(cfg, "stage3")
        stage3["epochs"] = int(args.epochs_stage3)
        for substage in ("step_a", "step_b"):
            if isinstance(stage3.get(substage), dict):
                stage3[substage]["epochs"] = int(args.epochs_stage3)
    return cfg


def main() -> None:
    args = _build_argparser().parse_args()
    base_cfg = _load_json(args.base_config)
    paths_cfg = base_cfg.get("paths", {})
    if not isinstance(paths_cfg, Mapping):
        raise ValueError("base config paths must be an object.")
    manifest_csv = args.manifest_csv or Path(str(paths_cfg["manifest_csv"]))
    concept_label_csv = args.concept_label_csv or Path(str(paths_cfg["concept_label_csv"]))
    concept_scaler_json = args.concept_scaler_json or Path(str(paths_cfg["concept_scaler_json"]))
    train_cfg = base_cfg.get("train", {})
    if not isinstance(train_cfg, Mapping):
        train_cfg = {}
    train_seed = int(args.train_seed if args.train_seed is not None else train_cfg.get("seed", 42))

    manifest_rows = _read_csv(manifest_csv)
    if not manifest_rows:
        raise ValueError(f"Manifest is empty: {manifest_csv}")
    missing = sorted(set(MANIFEST_FIELDS) - set(manifest_rows[0].keys()))
    if missing:
        raise ValueError(f"Manifest missing columns: {missing}")
    concept_rows = _read_csv(concept_label_csv)
    concept_columns = _concept_columns(concept_rows)
    concept_rows_by_patient = {row["patient_id"]: row for row in concept_rows}
    missing_concepts = sorted(set(row["patient_id"] for row in manifest_rows) - set(concept_rows_by_patient))
    if missing_concepts:
        raise ValueError(f"Manifest patients missing concept labels, first 10: {missing_concepts[:10]}")
    source_mapping = _source_mapping_from_scaler(concept_scaler_json, concept_columns)

    output_root = args.output_root.resolve()
    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output root already exists: {output_root}. Use --overwrite.")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    test_folds = _assign_round_robin_folds(manifest_rows, n_folds=int(args.n_folds), seed=int(args.seed))
    all_ids = {row["patient_id"] for row in manifest_rows}
    fold_plans: List[Dict[str, object]] = []
    commands: List[List[str]] = []
    for fold_idx, test_rows in enumerate(test_folds, start=1):
        fold_name = f"fold_{fold_idx:02d}"
        fold_dir = output_root / "folds" / fold_name
        fold_dir.mkdir(parents=True, exist_ok=True)
        test_ids = {row["patient_id"] for row in test_rows}
        trainval_rows = [row for row in manifest_rows if row["patient_id"] not in test_ids]
        train_rows, val_rows = _split_train_val(trainval_rows, float(args.val_ratio_within_trainval), seed=int(args.seed) + fold_idx * 1009)
        split_rows: List[Dict[str, str]] = []
        for split_name, rows in (("train", train_rows), ("val", val_rows), ("test", test_rows)):
            for row in rows:
                out = dict(row)
                out["split"] = split_name
                split_rows.append(out)
        if {row["patient_id"] for row in split_rows} != all_ids:
            raise RuntimeError(f"{fold_name} does not cover exactly all patients.")
        split_rows = sorted(split_rows, key=lambda item: (item["split"], item["patient_id"]))

        fold_manifest = fold_dir / "manifest.csv"
        write_csv_rows(fold_manifest, MANIFEST_FIELDS, split_rows)
        concept_dir = fold_dir / "concept_label"
        fold_label_csv, fold_stats_csv, fold_scaler_json = _rewrite_concept_assets(
            concept_rows_by_patient=concept_rows_by_patient,
            fold_rows=split_rows,
            output_dir=concept_dir,
            concept_columns=concept_columns,
            source_mapping=source_mapping,
            std_floor=float(args.std_floor),
        )
        run_id = f"{args.run_id_prefix}_seed{args.seed}_fold{fold_idx:02d}"
        fold_config = _build_fold_config(
            base_cfg=base_cfg,
            manifest_csv=fold_manifest,
            concept_label_csv=fold_label_csv,
            concept_scaler_json=fold_scaler_json,
            fold_dir=fold_dir,
            run_id=run_id,
            train_seed=train_seed,
            args=args,
        )
        config_path = fold_dir / f"train_config_{MODEL_NAME}_{fold_name}.json"
        _save_json(config_path, fold_config)
        command = [str(args.python), str(args.train_script), "--config", str(config_path)]
        plan = {
            "fold_name": fold_name,
            "run_id": run_id,
            "manifest_csv": str(fold_manifest),
            "concept_label_csv": str(fold_label_csv),
            "concept_statistics_csv": str(fold_stats_csv),
            "concept_scaler_json": str(fold_scaler_json),
            "train_config_json": str(config_path),
            "command": command,
            "counts": {
                "train": _counter_payload(train_rows),
                "val": _counter_payload(val_rows),
                "test": _counter_payload(test_rows),
            },
        }
        _save_json(fold_dir / "fold_metadata.json", plan)
        fold_plans.append(plan)
        commands.append(command)
        print(f"[{fold_name}] train={plan['counts']['train']} val={plan['counts']['val']} test={plan['counts']['test']}")

    cv_plan = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "base_config": str(args.base_config.resolve()),
        "source_manifest_csv": str(manifest_csv.resolve()),
        "source_concept_label_csv": str(concept_label_csv.resolve()),
        "source_concept_scaler_json": str(concept_scaler_json.resolve()),
        "output_root": str(output_root),
        "n_folds": int(args.n_folds),
        "seed": int(args.seed),
        "train_seed": train_seed,
        "val_ratio_within_trainval": float(args.val_ratio_within_trainval),
        "folds": fold_plans,
    }
    _save_json(output_root / "cv_plan.json", cv_plan)

    run_script = output_root / "run_all_folds.sh"
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", f"cd {_shell_quote(REPO_ROOT)}", ""]
    lines.extend(" ".join(_shell_quote(item) for item in command) for command in commands)
    run_script.write_text("\n".join(lines) + "\n", encoding="utf-8")
    run_script.chmod(run_script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    print(f"\n3D manifest CV assets written to: {output_root}")
    print(f"Plan JSON: {output_root / 'cv_plan.json'}")
    print(f"Run script: {run_script}")
    if args.run:
        for command in commands:
            print("\n[run] " + " ".join(_shell_quote(item) for item in command))
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()

