#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared UCSF-PDGM v5 helpers for the Habitat-CBM 3D pipeline."""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

DEFAULT_UCSF_ROOT = Path("/root/autodl-tmp/habitat_CBM/PKG _UCSF_PDGM_Version_5")
DEFAULT_METADATA_NAME = "UCSF-PDGM-metadata_v5.csv"
DEFAULT_NIFTI_SUBDIR = "UCSF-PDGM-v5"
DEFAULT_SPLITS = ("train", "val", "test")

MODEL_MODALITIES = ("t1", "t1ce", "t2", "t2flair")
RADIOMICS_MODALITIES = ("t1", "t1ce", "t2", "t2flair", "adc")
REQUIRED_MANIFEST_PATH_KEYS = (*MODEL_MODALITIES, "adc", "tumor_seg")

MODALITY_SUFFIXES: Dict[str, str] = {
    "t1": "_T1.nii.gz",
    "t1ce": "_T1c.nii.gz",
    "t2": "_T2.nii.gz",
    "t2flair": "_FLAIR.nii.gz",
    "adc": "_ADC.nii.gz",
    "tumor_seg": "_tumor_segmentation.nii.gz",
}

DEFAULT_CONCEPT_SOURCE_MAPPING = {
    "c1": "c1_selected_t1ce_firstorder_mean",
    "c2": "c2_selected_t2flair_firstorder_mean",
    "c3": "c3_whole_tumor_bbox_fill_ratio",
    "c4": "c4_tumor_volume_log1p_cm3",
    "c5": "c5_selected_adc_10percentile",
    "c6": "c6_selected_adc_95percentile",
    "c7": "c7_selected_volume_ratio",
    "c8": "c8_h3_volume_ratio",
}

MANIFEST_FIELDS = (
    "patient_id",
    "nifti_stem",
    "split",
    "y_true",
    "label_name",
    "idh_raw",
    "is_followup",
    "t1_path",
    "t1ce_path",
    "t2_path",
    "t2flair_path",
    "adc_path",
    "tumor_seg_path",
)

LABEL_TO_ID = {"wild_type": 0, "mutant": 1}
ID_TO_LABEL = {0: "wild_type", 1: "mutant"}


@dataclass(frozen=True)
class UCSFManifestRecord:
    patient_id: str
    nifti_stem: str
    split: str
    y_true: int
    label_name: str
    idh_raw: str
    is_followup: bool
    paths: Dict[str, Path]


def str2bool(value: str) -> bool:
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return [dict(row) for row in reader]


def write_csv_rows(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def is_followup_id(patient_id: str) -> bool:
    return "_FU" in str(patient_id).strip().upper()


def idh_to_binary_label(idh_value: object) -> int:
    text = str(idh_value).strip().lower()
    if not text:
        raise ValueError("Empty IDH value encountered.")
    return 0 if text == "wildtype" else 1


def resolve_nifti_stem(metadata_id: str, available_stems: Sequence[str]) -> str:
    """Resolve metadata ID to the UCSF nifti stem.

    Metadata uses both `UCSF-PDGM-004` and already-padded/follow-up ids such as
    `UCSF-PDGM-0429_FU003d`; nifti directories use the stem plus `_nifti`.
    """

    patient_id = str(metadata_id).strip()
    if not patient_id:
        raise ValueError("Empty UCSF patient ID encountered.")
    available = set(available_stems)
    if patient_id in available:
        return patient_id

    match = re.match(r"^(UCSF-PDGM-)(\d+)(.*)$", patient_id)
    if match is None:
        return patient_id
    candidate = f"{match.group(1)}{int(match.group(2)):04d}{match.group(3)}"
    return candidate


def list_nifti_stems(ucsf_root: Path) -> List[str]:
    nifti_root = ucsf_root / DEFAULT_NIFTI_SUBDIR
    if not nifti_root.is_dir():
        raise FileNotFoundError(f"Missing UCSF nifti directory: {nifti_root}")
    stems: List[str] = []
    for path in sorted(nifti_root.iterdir()):
        if path.is_dir() and path.name.endswith("_nifti"):
            stems.append(path.name[: -len("_nifti")])
    if not stems:
        raise ValueError(f"No *_nifti patient directories found under {nifti_root}")
    return stems


def modality_path(ucsf_root: Path, nifti_stem: str, modality_key: str) -> Path:
    suffix = MODALITY_SUFFIXES[modality_key]
    return ucsf_root / DEFAULT_NIFTI_SUBDIR / f"{nifti_stem}_nifti" / f"{nifti_stem}{suffix}"


def read_ucsf_metadata(ucsf_root: Path, metadata_csv: Optional[Path] = None) -> List[Dict[str, str]]:
    metadata_path = metadata_csv or (ucsf_root / DEFAULT_METADATA_NAME)
    if not metadata_path.is_file():
        raise FileNotFoundError(f"UCSF metadata CSV not found: {metadata_path}")
    rows = _read_csv_rows(metadata_path)
    if not rows:
        raise ValueError(f"UCSF metadata CSV is empty: {metadata_path}")
    required = {"ID", "IDH"}
    missing = sorted(required - set(rows[0].keys()))
    if missing:
        raise ValueError(f"UCSF metadata CSV missing required columns: {missing}")
    return rows


def manifest_row_to_record(row: Mapping[str, str]) -> UCSFManifestRecord:
    patient_id = str(row["patient_id"]).strip()
    split = str(row["split"]).strip().lower()
    if split not in DEFAULT_SPLITS:
        raise ValueError(f"Invalid split for patient {patient_id}: {split}")
    y_true = int(float(row["y_true"]))
    if y_true not in ID_TO_LABEL:
        raise ValueError(f"Invalid y_true for patient {patient_id}: {y_true}")
    label_name = str(row.get("label_name", ID_TO_LABEL[y_true])).strip()
    paths: Dict[str, Path] = {}
    for key in REQUIRED_MANIFEST_PATH_KEYS:
        field = f"{key}_path"
        value = str(row.get(field, "")).strip()
        if not value:
            raise ValueError(f"Manifest row for patient {patient_id} missing {field}.")
        paths[key] = Path(value)
    return UCSFManifestRecord(
        patient_id=patient_id,
        nifti_stem=str(row["nifti_stem"]).strip(),
        split=split,
        y_true=y_true,
        label_name=label_name,
        idh_raw=str(row.get("idh_raw", "")).strip(),
        is_followup=str(row.get("is_followup", "0")).strip().lower() in {"1", "true", "yes", "y"},
        paths=paths,
    )


def read_manifest_records(manifest_csv: Path, split: Optional[str] = None) -> List[UCSFManifestRecord]:
    if not manifest_csv.is_file():
        raise FileNotFoundError(f"Manifest CSV not found: {manifest_csv}")
    rows = _read_csv_rows(manifest_csv)
    if not rows:
        raise ValueError(f"Manifest CSV is empty: {manifest_csv}")
    missing = sorted(set(MANIFEST_FIELDS) - set(rows[0].keys()))
    if missing:
        raise ValueError(f"Manifest CSV missing required columns: {missing}")
    records = [manifest_row_to_record(row) for row in rows]
    if split is not None:
        split_name = split.strip().lower()
        records = [record for record in records if record.split == split_name]
    return records
