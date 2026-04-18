#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
构建 Habitat-CBM 训练用概念标签资产。

输入：
- concept_proxy_features_{run_id}.csv

输出：
- concept_labels.csv
- concept_statistics.csv
- concept_scaler_stats.json
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np

SOURCE_MAPPING = {
    "c1": "c1_h1_t1ce_firstorder_mean",
    "c2": "c2_h23_t1ce_firstorder_mean",
    "c3": "c3_whole_tumor_shape_sphericity",
    "c4": "c4_whole_tumor_flair_ce_volume_ratio",
    "c5": "c5_h12_adc_10percentile",
    "c6": "c6_h1_cbf_95percentile",
    "c7": "c7_h1_volume_ratio",
    "c8": "c8_h2_volume_ratio",
}
CONCEPT_IDS = tuple(SOURCE_MAPPING.keys())


@dataclass(frozen=True)
class Record:
    patient_id: str
    split: str
    y_true: int
    label_name: str
    concept_values: Dict[str, float]


def _normalize_patient_id(value: object) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError("Empty patient_id encountered.")
    return text


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return [dict(row) for row in reader]


def _safe_float(value: str, *, field: str, patient_id: str) -> float:
    text = str(value).strip()
    if text == "":
        raise ValueError(f"Empty value for {field} (patient={patient_id}).")
    try:
        parsed = float(text)
    except ValueError as exc:
        raise ValueError(
            f"Invalid numeric value for {field} (patient={patient_id}): {text}"
        ) from exc
    if not np.isfinite(parsed):
        raise ValueError(
            f"Non-finite numeric value for {field} (patient={patient_id}): {parsed}"
        )
    return parsed


def _validate_and_parse(rows: Sequence[Mapping[str, str]]) -> List[Record]:
    if not rows:
        raise ValueError("Input concept proxy CSV is empty.")

    required_cols = {"patient_id", "split", "y_true", "label_name", *SOURCE_MAPPING.values()}
    missing = sorted(required_cols - set(rows[0].keys()))
    if missing:
        raise ValueError(f"Missing required columns in concept proxy CSV: {missing}")

    records: List[Record] = []
    seen_patients: set[str] = set()
    for row in rows:
        patient_id = _normalize_patient_id(row["patient_id"])
        if patient_id in seen_patients:
            raise ValueError(f"Duplicate patient_id detected: {patient_id}")
        seen_patients.add(patient_id)

        split = str(row["split"]).strip().lower()
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Invalid split '{split}' for patient {patient_id}")

        y_true = int(_safe_float(row["y_true"], field="y_true", patient_id=patient_id))
        if y_true not in {0, 1}:
            raise ValueError(f"y_true must be 0/1 for patient {patient_id}, got {y_true}")

        label_name = str(row["label_name"]).strip()
        concept_values = {
            concept_id: _safe_float(
                row[source_col], field=source_col, patient_id=patient_id
            )
            for concept_id, source_col in SOURCE_MAPPING.items()
        }

        records.append(
            Record(
                patient_id=patient_id,
                split=split,
                y_true=y_true,
                label_name=label_name,
                concept_values=concept_values,
            )
        )

    return records


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _concept_array(records: Sequence[Record], concept_id: str) -> np.ndarray:
    return np.asarray([record.concept_values[concept_id] for record in records], dtype=np.float64)


def _build_statistics(records: Sequence[Record]) -> List[Dict[str, object]]:
    stats_rows: List[Dict[str, object]] = []
    for concept_id in CONCEPT_IDS:
        values = _concept_array(records, concept_id)
        stats_rows.append(
            {
                "concept_id": concept_id,
                "source_column": SOURCE_MAPPING[concept_id],
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

    split_groups: Dict[str, List[Record]] = {"train": [], "val": [], "test": []}
    for record in records:
        split_groups[record.split].append(record)

    for split_name, split_records in split_groups.items():
        if not split_records:
            continue
        for concept_id in CONCEPT_IDS:
            values = _concept_array(split_records, concept_id)
            stats_rows.append(
                {
                    "concept_id": concept_id,
                    "source_column": SOURCE_MAPPING[concept_id],
                    "count": int(values.size),
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=0)),
                    "min": float(values.min()),
                    "p01": float(np.percentile(values, 1.0)),
                    "p50": float(np.percentile(values, 50.0)),
                    "p99": float(np.percentile(values, 99.0)),
                    "max": float(values.max()),
                    "split": split_name,
                }
            )
    return stats_rows


def _build_scaler(records: Sequence[Record], std_floor: float) -> Dict[str, object]:
    train_records = [record for record in records if record.split == "train"]
    if not train_records:
        raise ValueError("No train split rows found; cannot fit concept scaler.")

    concepts: Dict[str, Dict[str, object]] = {}
    for concept_id in CONCEPT_IDS:
        values = _concept_array(train_records, concept_id)
        mean = float(values.mean())
        std_raw = float(values.std(ddof=0))
        std = std_raw if std_raw > std_floor else std_floor
        concepts[concept_id] = {
            "mean": mean,
            "std": std,
            "std_raw": std_raw,
            "source": SOURCE_MAPPING[concept_id],
        }

    return {
        "fit_split": "train",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "std_floor": float(std_floor),
        "concepts": concepts,
    }


def build_assets(
    concept_proxy_csv: Path,
    output_label_csv: Path,
    output_stats_csv: Path,
    output_scaler_json: Path,
    std_floor: float,
) -> None:
    rows = _read_csv(concept_proxy_csv)
    records = _validate_and_parse(rows)

    label_rows: List[Dict[str, object]] = []
    for record in sorted(records, key=lambda item: (item.split, item.patient_id)):
        row: Dict[str, object] = {
            "patient_id": record.patient_id,
            "split": record.split,
            "y_true": record.y_true,
            "label_name": record.label_name,
        }
        for concept_id in CONCEPT_IDS:
            row[f"{concept_id}_true"] = record.concept_values[concept_id]
        label_rows.append(row)

    label_fields = ["patient_id", "split", "y_true", "label_name", *[f"{cid}_true" for cid in CONCEPT_IDS]]
    _write_csv(output_label_csv, label_fields, label_rows)

    stats_rows = _build_statistics(records)
    stats_fields = [
        "concept_id",
        "source_column",
        "split",
        "count",
        "mean",
        "std",
        "min",
        "p01",
        "p50",
        "p99",
        "max",
    ]
    normalized_stats_rows: List[Dict[str, object]] = []
    for row in stats_rows:
        normalized = {key: row.get(key, "all") for key in stats_fields}
        normalized_stats_rows.append(normalized)
    _write_csv(output_stats_csv, stats_fields, normalized_stats_rows)

    scaler_payload = _build_scaler(records, std_floor=std_floor)
    output_scaler_json.parent.mkdir(parents=True, exist_ok=True)
    output_scaler_json.write_text(
        json.dumps(scaler_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Habitat-CBM concept label assets.")
    parser.add_argument("--concept-proxy-csv", type=Path, required=True)
    parser.add_argument("--output-label-csv", type=Path, required=True)
    parser.add_argument("--output-stats-csv", type=Path, required=True)
    parser.add_argument("--output-scaler-json", type=Path, required=True)
    parser.add_argument(
        "--std-floor",
        type=float,
        default=1e-6,
        help="Minimum std to avoid divide-by-zero in concept standardization.",
    )
    return parser


def main() -> None:
    args = _build_argparser().parse_args()
    build_assets(
        concept_proxy_csv=args.concept_proxy_csv,
        output_label_csv=args.output_label_csv,
        output_stats_csv=args.output_stats_csv,
        output_scaler_json=args.output_scaler_json,
        std_floor=args.std_floor,
    )
    print("Habitat-CBM concept assets generated:")
    print(f"  - labels : {args.output_label_csv}")
    print(f"  - stats  : {args.output_stats_csv}")
    print(f"  - scaler : {args.output_scaler_json}")


if __name__ == "__main__":
    main()
