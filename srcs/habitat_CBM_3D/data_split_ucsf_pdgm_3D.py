#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a manifest-driven train/val/test split for UCSF-PDGM 3D data."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

from ucsf_pdgm_3d_utils import (
    DEFAULT_UCSF_ROOT,
    ID_TO_LABEL,
    MANIFEST_FIELDS,
    REQUIRED_MANIFEST_PATH_KEYS,
    idh_to_binary_label,
    is_followup_id,
    list_nifti_stems,
    modality_path,
    read_ucsf_metadata,
    resolve_nifti_stem,
    str2bool,
    write_csv_rows,
)


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a UCSF-PDGM manifest split for Habitat-CBM 3D.")
    parser.add_argument("--ucsf-root", type=Path, default=DEFAULT_UCSF_ROOT)
    parser.add_argument("--metadata-csv", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=Path("/root/autodl-tmp/habitat_CBM/dataset/ucsf_pdgm_3d"))
    parser.add_argument("--manifest-name", type=str, default="manifest_ucsf_pdgm_3d.csv")
    parser.add_argument("--include-followup", type=str2bool, default=False)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _ensure_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    ratios = (train_ratio, val_ratio, test_ratio)
    if any(value <= 0.0 for value in ratios):
        raise ValueError("train/val/test ratios must all be positive.")
    total = sum(ratios)
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError(f"train/val/test ratios must sum to 1.0, got {total:.6f}.")


def _stratified_split(
    rows: Sequence[Mapping[str, object]],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Dict[str, List[Mapping[str, object]]]:
    rng = random.Random(seed)
    grouped: Dict[int, List[Mapping[str, object]]] = {0: [], 1: []}
    for row in rows:
        grouped[int(row["y_true"])].append(row)

    splits: Dict[str, List[Mapping[str, object]]] = {"train": [], "val": [], "test": []}
    for label_id, group in grouped.items():
        if not group:
            raise ValueError(f"No UCSF cases found for label {label_id}.")
        shuffled = list(group)
        rng.shuffle(shuffled)
        n_total = len(shuffled)
        n_train = int(round(n_total * train_ratio))
        n_val = int(round(n_total * val_ratio))
        if n_total >= 3:
            n_train = min(max(n_train, 1), n_total - 2)
            n_val = min(max(n_val, 1), n_total - n_train - 1)
        n_test = n_total - n_train - n_val
        if n_test <= 0:
            raise ValueError(
                f"Label {label_id} leaves no test samples: "
                f"total={n_total}, train={n_train}, val={n_val}, test={n_test}."
            )
        splits["train"].extend(shuffled[:n_train])
        splits["val"].extend(shuffled[n_train : n_train + n_val])
        splits["test"].extend(shuffled[n_train + n_val :])

    for split_name in splits:
        splits[split_name] = sorted(splits[split_name], key=lambda item: str(item["patient_id"]))
    return splits


def _build_rows(ucsf_root: Path, metadata_csv: Path | None, include_followup: bool) -> tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    metadata_rows = read_ucsf_metadata(ucsf_root, metadata_csv)
    available_stems = list_nifti_stems(ucsf_root)
    available_stem_set = set(available_stems)
    rows: List[Dict[str, object]] = []
    skipped: List[Dict[str, object]] = []
    seen_patients: set[str] = set()

    for meta_row in metadata_rows:
        patient_id = str(meta_row["ID"]).strip()
        followup = is_followup_id(patient_id)
        if followup and not include_followup:
            skipped.append({"patient_id": patient_id, "reason": "followup"})
            continue
        if patient_id in seen_patients:
            raise ValueError(f"Duplicate UCSF metadata ID after filtering: {patient_id}")
        seen_patients.add(patient_id)

        nifti_stem = resolve_nifti_stem(patient_id, available_stems)
        if nifti_stem not in available_stem_set:
            skipped.append({"patient_id": patient_id, "reason": f"missing_nifti_dir:{nifti_stem}"})
            continue

        paths = {key: modality_path(ucsf_root, nifti_stem, key) for key in REQUIRED_MANIFEST_PATH_KEYS}
        missing = [key for key, path in paths.items() if not path.is_file()]
        if missing:
            skipped.append({"patient_id": patient_id, "reason": "missing_files:" + ",".join(missing)})
            continue

        y_true = idh_to_binary_label(meta_row["IDH"])
        row: Dict[str, object] = {
            "patient_id": patient_id,
            "nifti_stem": nifti_stem,
            "split": "",
            "y_true": y_true,
            "label_name": ID_TO_LABEL[y_true],
            "idh_raw": str(meta_row["IDH"]).strip(),
            "is_followup": int(followup),
        }
        for key, path in paths.items():
            row[f"{key}_path"] = str(path)
        rows.append(row)

    if not rows:
        raise ValueError("No usable UCSF cases found after filtering and file validation.")
    return rows, skipped


def _summary_payload(
    rows: Sequence[Mapping[str, object]],
    skipped: Sequence[Mapping[str, object]],
    splits: Mapping[str, Sequence[Mapping[str, object]]],
    args: argparse.Namespace,
    manifest_path: Path,
) -> Dict[str, object]:
    total_counter = Counter(int(row["y_true"]) for row in rows)
    split_counts: Dict[str, Dict[str, int]] = {}
    for split_name, split_rows in splits.items():
        counter = Counter(int(row["y_true"]) for row in split_rows)
        split_counts[split_name] = {
            "total": len(split_rows),
            "wild_type": counter.get(0, 0),
            "mutant": counter.get(1, 0),
        }
    skipped_counter = Counter(str(row["reason"]).split(":", 1)[0] for row in skipped)
    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "ucsf_root": str(args.ucsf_root),
        "metadata_csv": str(args.metadata_csv or args.ucsf_root / "UCSF-PDGM-metadata_v5.csv"),
        "manifest_csv": str(manifest_path),
        "include_followup": bool(args.include_followup),
        "seed": int(args.seed),
        "ratios": {
            "train": float(args.train_ratio),
            "val": float(args.val_ratio),
            "test": float(args.test_ratio),
        },
        "usable_cases": len(rows),
        "usable_label_counts": {
            "wild_type": total_counter.get(0, 0),
            "mutant": total_counter.get(1, 0),
        },
        "split_counts": split_counts,
        "skipped_cases": len(skipped),
        "skipped_reasons": dict(skipped_counter),
    }


def main() -> None:
    args = _build_argparser().parse_args()
    _ensure_ratios(args.train_ratio, args.val_ratio, args.test_ratio)
    output_root = args.output_root
    manifest_path = output_root / args.manifest_name
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(f"Manifest already exists: {manifest_path}. Use --overwrite.")
    output_root.mkdir(parents=True, exist_ok=True)

    rows, skipped = _build_rows(args.ucsf_root, args.metadata_csv, include_followup=bool(args.include_followup))
    splits = _stratified_split(rows, args.train_ratio, args.val_ratio, args.seed)
    manifest_rows: List[Dict[str, object]] = []
    for split_name in ("train", "val", "test"):
        for row in splits[split_name]:
            out_row = dict(row)
            out_row["split"] = split_name
            manifest_rows.append(out_row)
    manifest_rows = sorted(manifest_rows, key=lambda item: (str(item["split"]), str(item["patient_id"])))
    write_csv_rows(manifest_path, MANIFEST_FIELDS, manifest_rows)

    skipped_path = output_root / "manifest_skipped_cases.csv"
    write_csv_rows(skipped_path, ("patient_id", "reason"), skipped)
    summary = _summary_payload(rows, skipped, splits, args, manifest_path)
    summary_path = output_root / "manifest_summary_ucsf_pdgm_3d.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("UCSF-PDGM 3D manifest split generated:")
    print(f"  manifest : {manifest_path}")
    print(f"  summary  : {summary_path}")
    print(f"  skipped  : {skipped_path}")
    for split_name in ("train", "val", "test"):
        split_summary = summary["split_counts"][split_name]
        print(
            f"  {split_name}: total={split_summary['total']} "
            f"wild_type={split_summary['wild_type']} mutant={split_summary['mutant']}"
        )


if __name__ == "__main__":
    main()
