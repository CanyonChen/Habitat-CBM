#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build UCSF-PDGM 3D Habitat-CBM concept label assets from proxy features."""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np

from ucsf_pdgm_3d_utils import DEFAULT_CONCEPT_SOURCE_MAPPING, write_csv_rows

CONCEPT_IDS = tuple(DEFAULT_CONCEPT_SOURCE_MAPPING.keys())


@dataclass(frozen=True)
class Record:
    patient_id: str
    split: str
    y_true: int
    label_name: str
    concept_values: Dict[str, float]


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build UCSF 3D Habitat-CBM concept labels/scaler.")
    parser.add_argument("--concept-proxy-csv", type=Path, required=True)
    parser.add_argument("--source-mapping-json", type=Path, default=None)
    parser.add_argument("--output-label-csv", type=Path, default=Path("/root/autodl-tmp/habitat_CBM/dataset/ucsf_pdgm_3d/concept_label/concept_labels.csv"))
    parser.add_argument("--output-stats-csv", type=Path, default=Path("/root/autodl-tmp/habitat_CBM/dataset/ucsf_pdgm_3d/concept_label/concept_statistics.csv"))
    parser.add_argument("--output-scaler-json", type=Path, default=Path("/root/autodl-tmp/habitat_CBM/dataset/ucsf_pdgm_3d/concept_label/concept_scaler_stats.json"))
    parser.add_argument("--std-floor", type=float, default=1e-6)
    return parser


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return [dict(row) for row in reader]


def _load_source_mapping(path: Path | None) -> Dict[str, str]:
    if path is None:
        return dict(DEFAULT_CONCEPT_SOURCE_MAPPING)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "concept_source_mapping" in payload:
        payload = payload["concept_source_mapping"]
    if not isinstance(payload, Mapping):
        raise ValueError(f"Invalid source mapping JSON: {path}")
    mapping = {str(key).strip().lower(): str(value).strip() for key, value in payload.items()}
    missing = sorted(set(CONCEPT_IDS) - set(mapping.keys()))
    if missing:
        raise ValueError(f"Source mapping missing concept ids: {missing}")
    return {concept_id: mapping[concept_id] for concept_id in CONCEPT_IDS}


def _safe_float(value: object, *, field: str, patient_id: str) -> float:
    text = str(value).strip()
    if text == "":
        raise ValueError(f"Empty value for {field} (patient={patient_id}).")
    try:
        parsed = float(text)
    except ValueError as exc:
        raise ValueError(f"Invalid numeric value for {field} (patient={patient_id}): {text}") from exc
    if not np.isfinite(parsed):
        raise ValueError(f"Non-finite numeric value for {field} (patient={patient_id}): {parsed}")
    return parsed


def _parse_records(rows: Sequence[Mapping[str, str]], source_mapping: Mapping[str, str]) -> List[Record]:
    if not rows:
        raise ValueError("Concept proxy CSV is empty.")
    required = {"patient_id", "split", "y_true", "label_name", *source_mapping.values()}
    missing = sorted(required - set(rows[0].keys()))
    if missing:
        raise ValueError(f"Concept proxy CSV missing required columns: {missing}")

    records: List[Record] = []
    seen: set[str] = set()
    for row in rows:
        patient_id = str(row["patient_id"]).strip()
        if not patient_id:
            raise ValueError("Empty patient_id encountered.")
        if patient_id in seen:
            raise ValueError(f"Duplicate patient_id in concept proxy CSV: {patient_id}")
        seen.add(patient_id)
        split = str(row["split"]).strip().lower()
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Invalid split for patient {patient_id}: {split}")
        y_true = int(_safe_float(row["y_true"], field="y_true", patient_id=patient_id))
        if y_true not in {0, 1}:
            raise ValueError(f"y_true must be 0/1 for patient {patient_id}, got {y_true}")
        concept_values = {
            concept_id: _safe_float(row[source_col], field=source_col, patient_id=patient_id)
            for concept_id, source_col in source_mapping.items()
        }
        records.append(
            Record(
                patient_id=patient_id,
                split=split,
                y_true=y_true,
                label_name=str(row["label_name"]).strip(),
                concept_values=concept_values,
            )
        )
    return records


def _concept_array(records: Sequence[Record], concept_id: str) -> np.ndarray:
    return np.asarray([record.concept_values[concept_id] for record in records], dtype=np.float64)


def _stats_rows(records: Sequence[Record], source_mapping: Mapping[str, str]) -> List[Dict[str, object]]:
    output: List[Dict[str, object]] = []
    groups: Dict[str, Sequence[Record]] = {"all": records}
    for split in ("train", "val", "test"):
        split_records = [record for record in records if record.split == split]
        if split_records:
            groups[split] = split_records
    for split_name, split_records in groups.items():
        for concept_id in CONCEPT_IDS:
            values = _concept_array(split_records, concept_id)
            output.append(
                {
                    "concept_id": concept_id,
                    "source_column": source_mapping[concept_id],
                    "split": split_name,
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


def _build_scaler(records: Sequence[Record], source_mapping: Mapping[str, str], std_floor: float) -> Dict[str, object]:
    train_records = [record for record in records if record.split == "train"]
    if not train_records:
        raise ValueError("No train rows found; cannot fit concept scaler.")
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
            "source": source_mapping[concept_id],
        }
    return {
        "fit_split": "train",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "std_floor": float(std_floor),
        "concepts": concepts,
    }


def build_assets(
    concept_proxy_csv: Path,
    source_mapping_json: Path | None,
    output_label_csv: Path,
    output_stats_csv: Path,
    output_scaler_json: Path,
    std_floor: float,
) -> None:
    source_mapping = _load_source_mapping(source_mapping_json)
    records = _parse_records(_read_csv(concept_proxy_csv), source_mapping)

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
    write_csv_rows(
        output_label_csv,
        ("patient_id", "split", "y_true", "label_name", *[f"{concept_id}_true" for concept_id in CONCEPT_IDS]),
        label_rows,
    )
    write_csv_rows(
        output_stats_csv,
        ("concept_id", "source_column", "split", "count", "mean", "std", "min", "p01", "p50", "p99", "max"),
        _stats_rows(records, source_mapping),
    )
    output_scaler_json.parent.mkdir(parents=True, exist_ok=True)
    output_scaler_json.write_text(
        json.dumps(_build_scaler(records, source_mapping, std_floor), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    args = _build_argparser().parse_args()
    build_assets(
        concept_proxy_csv=args.concept_proxy_csv,
        source_mapping_json=args.source_mapping_json,
        output_label_csv=args.output_label_csv,
        output_stats_csv=args.output_stats_csv,
        output_scaler_json=args.output_scaler_json,
        std_floor=float(args.std_floor),
    )
    print("UCSF 3D Habitat-CBM concept assets generated:")
    print(f"  labels : {args.output_label_csv}")
    print(f"  stats  : {args.output_stats_csv}")
    print(f"  scaler : {args.output_scaler_json}")


if __name__ == "__main__":
    main()
