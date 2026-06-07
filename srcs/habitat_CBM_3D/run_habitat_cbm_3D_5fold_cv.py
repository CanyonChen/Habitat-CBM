#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare and optionally run patient-level 5-fold CV for Habitat-CBM 3D."""

from __future__ import annotations

import argparse
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parents[1]
PROJECT_ROOT = REPO_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from srcs.run_habitat_cbm_5fold_cv import (  # noqa: E402
    DEFAULT_N_FOLDS,
    DEFAULT_STD_FLOOR,
    _assign_round_robin_folds,
    _as_dict,
    _concept_columns,
    _counter_payload,
    _deep_update_path_config,
    _default_concept_label_csv,
    _default_concept_scaler_json,
    _infer_dataset_root,
    _load_json,
    _load_source_mapping,
    _read_csv,
    _rewrite_concept_assets,
    _shell_quote,
    _split_train_val,
    _validate_concept_rows,
    _write_cv_metric_summary,
    _write_fold_manifests,
    _save_json,
)
from srcs.data_split import (  # noqa: E402
    materialize_split_dirs,
    read_records,
    summarise,
    validate_layout,
)

MODEL_NAME = "habitat_cbm_3d"


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare and optionally run 5-fold CV for Habitat-CBM 3D.")
    parser.add_argument("--base-config", type=Path, default=CURRENT_DIR / "args_train_habitat_CBM_3D.json")
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--concept-label-csv", type=Path, default=None)
    parser.add_argument("--concept-scaler-json", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "results" / "habitat_CBM_3D_5fold_cv")
    parser.add_argument("--n-folds", type=int, default=DEFAULT_N_FOLDS)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--val-ratio-within-trainval", type=float, default=0.125)
    parser.add_argument("--train-seed", type=int, default=None)
    parser.add_argument("--link-mode", choices=("symlink", "copy", "none"), default="symlink")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-layout-validation", action="store_true")
    parser.add_argument("--std-floor", type=float, default=DEFAULT_STD_FLOOR)
    parser.add_argument("--run-id-prefix", type=str, default="habitat_cbm_3d_cv5")
    parser.add_argument("--train-script", type=Path, default=CURRENT_DIR / "train_habitat_CBM_3D.py")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--epochs-stage1", type=int, default=None)
    parser.add_argument("--epochs-stage2", type=int, default=None)
    parser.add_argument("--epochs-stage3", type=int, default=None)
    parser.add_argument("--run", action="store_true")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    base_cfg = _load_json(args.base_config)
    dataset_root = args.dataset_root if args.dataset_root is not None else _infer_dataset_root(base_cfg)
    concept_label_csv = args.concept_label_csv if args.concept_label_csv is not None else _default_concept_label_csv(base_cfg)
    concept_scaler_json = (
        args.concept_scaler_json if args.concept_scaler_json is not None else _default_concept_scaler_json(base_cfg)
    )
    output_root = args.output_root.resolve()
    train_seed = int(
        args.train_seed
        if args.train_seed is not None
        else _as_dict(base_cfg.get("train", {}), name="train").get("seed", 42)
    )
    if args.run and args.link_mode == "none":
        raise ValueError("--run cannot be used with --link-mode none; training needs materialized patient folders.")

    records = read_records(dataset_root / "idh.csv")
    if args.link_mode != "none" and not args.skip_layout_validation:
        validate_layout(
            dataset_root,
            records,
            require_voi=bool(_as_dict(base_cfg.get("data", {}), name="data").get("require_voi", True)),
        )
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

    print(f"\n3D CV assets written to: {output_root}")
    print(f"Plan JSON: {output_root / 'cv_plan.json'}")
    print(f"Run script: {run_script_path}")
    if args.run:
        for command in commands:
            print("\n[run] " + " ".join(_shell_quote(item) for item in command))
            subprocess.run(command, check=True)
        _write_cv_metric_summary(output_root, fold_plans)
        print(f"\nCompleted 3D CV run. Metric summary directory: {output_root}")
    else:
        print("Preparation only. Add --run or execute run_all_folds.sh to train all folds.")


if __name__ == "__main__":
    main()

