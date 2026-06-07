#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extract UCSF-PDGM 3D habitat radiomics proxies for Habitat-CBM concepts."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Mapping

import nibabel as nib
import numpy as np

from ucsf_pdgm_3d_utils import (
    DEFAULT_CONCEPT_SOURCE_MAPPING,
    RADIOMICS_MODALITIES,
    read_manifest_records,
    write_csv_rows,
)


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract UCSF habitat radiomics/concept proxy features.")
    parser.add_argument("--manifest-csv", type=Path, required=True)
    parser.add_argument("--habitat-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("/root/autodl-tmp/habitat_CBM/results/ucsf_pdgm_3d_radiomics"))
    parser.add_argument("--selected-habitat", type=str, default="h23")
    parser.add_argument("--splits", type=str, default="train,val,test")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--max-patients", type=int, default=None)
    return parser


def _resolve_run_id(run_id: str | None) -> str:
    return run_id if run_id else time.strftime("%Y%m%d_%H%M%S")


def _load_array(path: Path) -> tuple[nib.spatialimages.SpatialImage, np.ndarray]:
    image = nib.load(str(path))
    array = np.asarray(image.get_fdata(dtype=np.float32), dtype=np.float32)
    if array.ndim != 3:
        raise ValueError(f"Expected 3D NIfTI, got shape={array.shape}: {path}")
    return image, array


def _mask_path(habitat_root: Path, split: str, patient_id: str, mask_name: str) -> Path:
    return habitat_root / split / patient_id / f"{mask_name}.nii.gz"


def _voxel_volume_cm3(image: nib.spatialimages.SpatialImage) -> float:
    zooms = image.header.get_zooms()[:3]
    volume_mm3 = float(zooms[0] * zooms[1] * zooms[2])
    if not np.isfinite(volume_mm3) or volume_mm3 <= 0.0:
        return 1.0 / 1000.0
    return volume_mm3 / 1000.0


def _firstorder(values: np.ndarray) -> Dict[str, float]:
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("Cannot compute firstorder stats from an empty/invalid ROI.")
    return {
        "count": float(values.size),
        "mean": float(values.mean(dtype=np.float64)),
        "std": float(values.std(dtype=np.float64)),
        "min": float(values.min()),
        "p10": float(np.percentile(values, 10.0)),
        "p50": float(np.percentile(values, 50.0)),
        "p90": float(np.percentile(values, 90.0)),
        "p95": float(np.percentile(values, 95.0)),
        "max": float(values.max()),
    }


def _bbox_fill_ratio(mask: np.ndarray) -> float:
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        return 0.0
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0) + 1
    bbox_volume = int(np.prod(maxs - mins))
    if bbox_volume <= 0:
        return 0.0
    return float(coords.shape[0] / bbox_volume)


def _extract_record(record, habitat_root: Path, selected_habitat: str) -> tuple[Dict[str, object], Dict[str, object]]:
    selected_path = _mask_path(habitat_root, record.split, record.patient_id, selected_habitat)
    h123_path = _mask_path(habitat_root, record.split, record.patient_id, "h123")
    h3_path = _mask_path(habitat_root, record.split, record.patient_id, "h3")
    if not selected_path.is_file():
        raise FileNotFoundError(f"Selected habitat mask not found: {selected_path}")
    if not h123_path.is_file():
        raise FileNotFoundError(f"Whole tumor habitat mask not found: {h123_path}")
    if not h3_path.is_file():
        raise FileNotFoundError(f"H3 habitat mask not found: {h3_path}")

    _, selected_arr = _load_array(selected_path)
    _, whole_arr = _load_array(h123_path)
    _, h3_arr = _load_array(h3_path)
    selected_mask = selected_arr > 0
    whole_mask = whole_arr > 0
    h3_mask = h3_arr > 0
    selected_count = int(selected_mask.sum())
    whole_count = int(whole_mask.sum())
    h3_count = int(h3_mask.sum())
    if selected_count <= 0:
        raise ValueError(f"Selected habitat mask is empty for {record.patient_id}: {selected_path}")
    if whole_count <= 0:
        raise ValueError(f"Whole tumor mask is empty for {record.patient_id}: {h123_path}")

    row: Dict[str, object] = {
        "patient_id": record.patient_id,
        "nifti_stem": record.nifti_stem,
        "split": record.split,
        "y_true": record.y_true,
        "label_name": record.label_name,
        "selected_habitat": selected_habitat,
        "selected_habitat_path": str(selected_path),
        "selected_voxels": selected_count,
        "whole_tumor_voxels": whole_count,
        "h3_voxels": h3_count,
        "selected_volume_ratio": float(selected_count / whole_count),
        "h3_volume_ratio": float(h3_count / whole_count),
        "whole_tumor_bbox_fill_ratio": _bbox_fill_ratio(whole_mask),
    }

    voxel_volume_cm3 = None
    for modality in RADIOMICS_MODALITIES:
        image, volume = _load_array(record.paths[modality])
        if tuple(volume.shape) != tuple(selected_mask.shape):
            raise ValueError(
                f"Shape mismatch for {record.patient_id} {modality}: "
                f"image={volume.shape}, mask={selected_mask.shape}"
            )
        if voxel_volume_cm3 is None:
            voxel_volume_cm3 = _voxel_volume_cm3(image)
        stats = _firstorder(volume[selected_mask])
        for key, value in stats.items():
            row[f"{modality}_selected_firstorder_{key}"] = value

    tumor_volume_cm3 = float(whole_count * (voxel_volume_cm3 if voxel_volume_cm3 is not None else 0.001))
    concept_row: Dict[str, object] = {
        "patient_id": record.patient_id,
        "split": record.split,
        "y_true": record.y_true,
        "label_name": record.label_name,
        "selected_habitat": selected_habitat,
        "c1_selected_t1ce_firstorder_mean": row["t1ce_selected_firstorder_mean"],
        "c2_selected_t2flair_firstorder_mean": row["t2flair_selected_firstorder_mean"],
        "c3_whole_tumor_bbox_fill_ratio": row["whole_tumor_bbox_fill_ratio"],
        "c4_tumor_volume_log1p_cm3": float(math.log1p(tumor_volume_cm3)),
        "c5_selected_adc_10percentile": row["adc_selected_firstorder_p10"],
        "c6_selected_adc_95percentile": row["adc_selected_firstorder_p95"],
        "c7_selected_volume_ratio": row["selected_volume_ratio"],
        "c8_h3_volume_ratio": row["h3_volume_ratio"],
    }
    return row, concept_row


def main() -> None:
    args = _build_argparser().parse_args()
    run_id = _resolve_run_id(args.run_id)
    selected_habitat = args.selected_habitat.strip().lower()
    split_filter = {item.strip().lower() for item in args.splits.split(",") if item.strip()}
    records = [
        record
        for record in read_manifest_records(args.manifest_csv)
        if "all" in split_filter or record.split in split_filter
    ]
    if args.max_patients is not None:
        records = records[: int(args.max_patients)]
    if not records:
        raise ValueError("No records selected for radiomics extraction.")

    output_root = args.output_root / run_id
    output_root.mkdir(parents=True, exist_ok=True)
    feature_rows: List[Dict[str, object]] = []
    concept_rows: List[Dict[str, object]] = []
    failures: List[Dict[str, object]] = []
    for record in records:
        try:
            feature_row, concept_row = _extract_record(record, args.habitat_root, selected_habitat)
            feature_rows.append(feature_row)
            concept_rows.append(concept_row)
        except Exception as exc:
            failures.append(
                {
                    "patient_id": record.patient_id,
                    "split": record.split,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    feature_path = output_root / f"radiomics_features_{selected_habitat}_{run_id}.csv"
    concept_path = output_root / f"concept_proxy_features_{run_id}.csv"
    failure_path = output_root / f"radiomics_failures_{run_id}.csv"
    mapping_path = output_root / f"concept_source_mapping_{run_id}.json"

    feature_fields = list(feature_rows[0].keys()) if feature_rows else [
        "patient_id",
        "nifti_stem",
        "split",
        "y_true",
        "label_name",
        "selected_habitat",
    ]
    concept_fields = [
        "patient_id",
        "split",
        "y_true",
        "label_name",
        "selected_habitat",
        *DEFAULT_CONCEPT_SOURCE_MAPPING.values(),
    ]
    write_csv_rows(feature_path, feature_fields, feature_rows)
    write_csv_rows(concept_path, concept_fields, concept_rows)
    write_csv_rows(failure_path, ("patient_id", "split", "error_type", "error"), failures)
    mapping_path.write_text(
        json.dumps(
            {
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "selected_habitat": selected_habitat,
                "concept_source_mapping": DEFAULT_CONCEPT_SOURCE_MAPPING,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print("UCSF habitat radiomics proxy extraction finished:")
    print(f"  features : {feature_path}")
    print(f"  concepts : {concept_path}")
    print(f"  mapping  : {mapping_path}")
    print(f"  failures : {failure_path}")
    print(f"  exported : {len(concept_rows)}")
    print(f"  failed   : {len(failures)}")


if __name__ == "__main__":
    main()
