#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Patient-level 3D data loader for Habitat-CBM 3D experiments."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

import sys

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from srcs.data_loader import (  # noqa: E402
    LABEL_TO_ID,
    PatientCase,
    collect_patient_image_files,
    discover_modality_files,
    discover_patient_dirs,
    discover_voi_file,
    ensure_tensor,
    load_nifti_array,
    load_voi_array,
    normalize_volume,
    str2bool,
)
from srcs.data_loader_habitat_CBM import (  # noqa: E402
    ConceptEntry,
    ConceptScaler,
    concept_names_to_columns,
    load_concept_labels,
    load_concept_scaler,
    resolve_concept_names,
)
from srcs.habitat_CBM_3D.ucsf_pdgm_3d_utils import (  # noqa: E402
    read_manifest_records,
)

DEFAULT_3D_MODALITIES = ("t1", "t1ce", "t2", "t2flair")
DEFAULT_3D_TARGET_SHAPE = (32, 128, 128)  # [D,H,W]
DEFAULT_3D_CROP_MARGIN = (4, 16, 16)  # [D,H,W]
DEFAULT_CONCEPT_NAMES = ("c1", "c2", "c3", "c4", "c6")
DEFAULT_CONCEPT_COLUMNS = tuple(f"{name}_true" for name in DEFAULT_CONCEPT_NAMES)


@dataclass(frozen=True)
class VolumeSample:
    patient_id: str


def _parse_int_triple(value: object, default: Sequence[int], *, name: str) -> Tuple[int, int, int]:
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
    if any(v < 0 for v in items):
        raise ValueError(f"{name} values must be non-negative, got {items!r}.")
    return items


def _voi_bbox_hwd(voi: np.ndarray, margin_dhw: Sequence[int]) -> Tuple[slice, slice, slice]:
    foreground = np.argwhere(np.asarray(voi) > 0)
    if foreground.size == 0:
        raise ValueError("VOI mask is empty; cannot crop a 3D volume.")
    margin_d, margin_h, margin_w = (int(v) for v in margin_dhw)
    h_min, w_min, d_min = foreground.min(axis=0)
    h_max, w_max, d_max = foreground.max(axis=0) + 1
    h0 = max(int(h_min) - margin_h, 0)
    h1 = min(int(h_max) + margin_h, int(voi.shape[0]))
    w0 = max(int(w_min) - margin_w, 0)
    w1 = min(int(w_max) + margin_w, int(voi.shape[1]))
    d0 = max(int(d_min) - margin_d, 0)
    d1 = min(int(d_max) + margin_d, int(voi.shape[2]))
    return slice(h0, h1), slice(w0, w1), slice(d0, d1)


class HabitatIDHVolumeDataset(Dataset):
    """3D patient-level Dataset with concept labels for Habitat-CBM 3D."""

    def __init__(
        self,
        split_root: str | Path | None = None,
        manifest_csv: str | Path | None = None,
        split_name: str | None = None,
        modalities: Sequence[str] = DEFAULT_3D_MODALITIES,
        require_voi: bool = True,
        crop_with_voi: bool = True,
        crop_margin: Sequence[int] = DEFAULT_3D_CROP_MARGIN,
        intensity_norm: str = "zscore",
        cache_volumes: bool = True,
        return_metadata: bool = True,
        transform=None,
        concept_label_csv: str | Path | None = None,
        concept_scaler_json: str | Path | None = None,
        concept_columns: Sequence[str] = DEFAULT_CONCEPT_COLUMNS,
    ) -> None:
        self.split_root = Path(split_root) if split_root is not None else None
        self.manifest_csv = Path(manifest_csv) if manifest_csv is not None else None
        self.split_name = (
            str(split_name).strip().lower()
            if split_name is not None
            else (self.split_root.name.strip().lower() if self.split_root is not None else None)
        )
        if self.split_root is None and self.manifest_csv is None:
            raise ValueError("Either split_root or manifest_csv must be provided.")
        if self.manifest_csv is not None and self.split_name not in {"train", "val", "test"}:
            raise ValueError("split_name must be one of train/val/test when manifest_csv is used.")
        self.modalities = tuple(str(item).strip().lower() for item in modalities if str(item).strip())
        if not self.modalities:
            raise ValueError("modalities must not be empty.")
        self.require_voi = bool(require_voi or crop_with_voi)
        self.crop_with_voi = bool(crop_with_voi)
        self.crop_margin = _parse_int_triple(crop_margin, DEFAULT_3D_CROP_MARGIN, name="crop_margin")
        self.intensity_norm = str(intensity_norm)
        self.cache_volumes = bool(cache_volumes)
        self.return_metadata = bool(return_metadata)
        self.transform = transform
        self.concept_columns = tuple(concept_columns)
        self.enable_concepts = concept_label_csv is not None or concept_scaler_json is not None
        self._volume_cache: Dict[str, Dict[str, np.ndarray]] = {}
        self._voi_cache: Dict[str, np.ndarray] = {}
        self._concept_map: Dict[str, ConceptEntry] = {}
        self._concept_scaler: Optional[ConceptScaler] = None

        self.patient_cases = self._build_patient_cases()
        self.patient_ids = tuple(sorted(self.patient_cases.keys()))
        self.sample_index = [VolumeSample(patient_id=patient_id) for patient_id in self.patient_ids]

        if self.enable_concepts:
            if concept_label_csv is None or concept_scaler_json is None:
                raise ValueError(
                    "When concept mode is enabled, both concept_label_csv and concept_scaler_json are required."
                )
            concept_names = tuple(
                str(col).strip().lower()[:-5] if str(col).strip().lower().endswith("_true") else str(col).strip().lower()
                for col in self.concept_columns
            )
            self._concept_map = load_concept_labels(
                concept_label_csv=concept_label_csv,
                concept_columns=self.concept_columns,
            )
            self._concept_scaler = load_concept_scaler(
                concept_scaler_json,
                concept_names=concept_names,
            )
            self._validate_concept_alignment()

    @property
    def concept_dim(self) -> int:
        return len(self.concept_columns)

    def _build_patient_cases(self) -> Dict[str, PatientCase]:
        if self.manifest_csv is not None:
            return self._build_patient_cases_from_manifest()
        if self.split_root is None:
            raise RuntimeError("split_root is required for legacy directory mode.")
        patient_dir_map = discover_patient_dirs(self.split_root)
        patient_cases: Dict[str, PatientCase] = {}
        for patient_id, entry in sorted(patient_dir_map.items()):
            label_name = str(entry["label_name"])
            all_files = collect_patient_image_files(entry)
            modality_paths = discover_modality_files(entry, self.modalities, all_files=all_files)
            voi_path = discover_voi_file(entry) if self.require_voi else None
            patient_cases[patient_id] = PatientCase(
                patient_id=patient_id,
                label_name=label_name,
                label_id=LABEL_TO_ID[label_name],
                modality_paths=modality_paths,
                voi_path=voi_path,
            )
        return patient_cases

    def _build_patient_cases_from_manifest(self) -> Dict[str, PatientCase]:
        if self.manifest_csv is None or self.split_name is None:
            raise RuntimeError("manifest_csv and split_name are required for manifest mode.")
        records = read_manifest_records(self.manifest_csv, split=self.split_name)
        if not records:
            raise ValueError(f"No records found for split={self.split_name} in manifest {self.manifest_csv}.")
        patient_cases: Dict[str, PatientCase] = {}
        for record in records:
            modality_paths = {}
            for modality in self.modalities:
                if modality not in record.paths:
                    raise KeyError(f"Manifest for patient {record.patient_id} has no modality path: {modality}")
                path = record.paths[modality]
                if not path.is_file():
                    raise FileNotFoundError(f"Missing modality file for patient {record.patient_id}: {path}")
                modality_paths[modality] = path
            voi_path = record.paths["tumor_seg"] if self.require_voi else None
            if voi_path is not None and not voi_path.is_file():
                raise FileNotFoundError(f"Missing tumor segmentation for patient {record.patient_id}: {voi_path}")
            patient_cases[record.patient_id] = PatientCase(
                patient_id=record.patient_id,
                label_name=record.label_name,
                label_id=int(record.y_true),
                modality_paths=modality_paths,
                voi_path=voi_path,
            )
        return patient_cases

    def _validate_concept_alignment(self) -> None:
        if self._concept_scaler is None:
            raise RuntimeError("Concept scaler is not initialized.")
        if self.split_name is None:
            raise RuntimeError("Dataset split name is not initialized.")
        expected_split = self.split_name
        if self._concept_scaler.mean.shape[0] != self.concept_dim:
            raise ValueError(
                "Concept dimension mismatch between concept_columns and scaler stats: "
                f"{self.concept_dim} vs {self._concept_scaler.mean.shape[0]}"
            )
        for patient_id, case in self.patient_cases.items():
            if patient_id not in self._concept_map:
                raise KeyError(f"Patient {patient_id} exists in dataset but missing in concept labels.")
            entry = self._concept_map[patient_id]
            if entry.y_true != case.label_id:
                raise ValueError(
                    f"Label mismatch for patient {patient_id}: "
                    f"dataset label={case.label_id}, concept y_true={entry.y_true}."
                )
            if entry.split and entry.split != expected_split:
                raise ValueError(
                    f"Split mismatch for patient {patient_id}: dataset split={expected_split}, "
                    f"concept split={entry.split}."
                )
            if entry.concept_true_raw.shape != (self.concept_dim,):
                raise ValueError(
                    f"Concept shape mismatch for patient {patient_id}: "
                    f"expected {(self.concept_dim,)}, got {entry.concept_true_raw.shape}."
                )

    def _get_concepts(self, patient_id: str) -> Tuple[np.ndarray, np.ndarray]:
        if self._concept_scaler is None:
            raise RuntimeError("Concept scaler is not initialized.")
        raw = self._concept_map[patient_id].concept_true_raw.astype(np.float32)
        std = self._concept_scaler.standardize(raw).astype(np.float32)
        return raw, std

    def _get_patient_volumes(self, patient_id: str) -> Dict[str, np.ndarray]:
        if self.cache_volumes and patient_id in self._volume_cache:
            return self._volume_cache[patient_id]
        case = self.patient_cases[patient_id]
        volumes: Dict[str, np.ndarray] = {}
        reference_shape: Optional[Tuple[int, int, int]] = None
        for modality, path in case.modality_paths.items():
            volume = normalize_volume(load_nifti_array(path), self.intensity_norm)
            if volume.ndim != 3:
                raise ValueError(
                    f"Modality '{modality}' of patient {patient_id} is not 3D: "
                    f"shape={volume.shape}, path={path}"
                )
            if reference_shape is None:
                reference_shape = tuple(volume.shape)
            elif tuple(volume.shape) != reference_shape:
                raise ValueError(
                    f"Shape mismatch within patient {patient_id}: "
                    f"expected {reference_shape}, got {volume.shape} for {modality}"
                )
            volumes[modality] = volume.astype(np.float32, copy=False)
        if self.cache_volumes:
            self._volume_cache[patient_id] = volumes
        return volumes

    def _get_patient_voi(self, patient_id: str) -> np.ndarray:
        if not self.require_voi:
            raise RuntimeError("VOI is disabled for this dataset instance.")
        if self.cache_volumes and patient_id in self._voi_cache:
            return self._voi_cache[patient_id]
        case = self.patient_cases[patient_id]
        if case.voi_path is None:
            raise FileNotFoundError(f"Missing VOI path for patient {patient_id}.")
        voi = load_voi_array(case.voi_path)
        if voi.ndim != 3:
            raise ValueError(
                f"VOI of patient {patient_id} is not 3D: shape={voi.shape}, path={case.voi_path}"
            )
        voi = (voi > 0).astype(np.float32)
        if self.cache_volumes:
            self._voi_cache[patient_id] = voi
        return voi

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, index: int) -> Dict[str, object]:
        patient_id = self.patient_ids[index]
        case = self.patient_cases[patient_id]
        volumes = self._get_patient_volumes(patient_id)
        crop_slices = None
        if self.crop_with_voi:
            voi = self._get_patient_voi(patient_id)
            first_shape = next(iter(volumes.values())).shape
            if tuple(voi.shape) != tuple(first_shape):
                raise ValueError(
                    f"VOI shape mismatch for patient {patient_id}: "
                    f"voi={tuple(voi.shape)}, image={tuple(first_shape)}"
                )
            crop_slices = _voi_bbox_hwd(voi, self.crop_margin)

        channel_volumes: List[np.ndarray] = []
        for modality in self.modalities:
            volume = volumes[modality]
            if crop_slices is not None:
                volume = volume[crop_slices]
            # Existing arrays are [H,W,D]; MedicalNet expects [C,D,H,W].
            channel_volumes.append(np.transpose(volume, (2, 0, 1)).astype(np.float32, copy=False))

        sample_dict: Dict[str, object] = {
            "image": np.stack(channel_volumes, axis=0).astype(np.float32),
            "label": case.label_id,
        }
        if self.transform is not None:
            sample_dict = dict(self.transform(sample_dict))

        image_tensor = ensure_tensor(sample_dict["image"], dtype=torch.float32)
        if image_tensor.ndim != 4:
            raise ValueError(f"Expected image tensor [C,D,H,W], got {tuple(image_tensor.shape)}.")
        label_tensor = ensure_tensor(sample_dict["label"], dtype=torch.long)
        if label_tensor.ndim != 0:
            label_tensor = label_tensor.reshape(()).to(dtype=torch.long)

        output: Dict[str, object] = {
            "image": image_tensor,
            "label": label_tensor,
            "patient_id": patient_id,
            "slice_index": 0,
        }
        if self.enable_concepts:
            concept_raw, concept_std = self._get_concepts(patient_id)
            output["concept_true_raw"] = torch.as_tensor(concept_raw, dtype=torch.float32)
            output["concept_true_std"] = torch.as_tensor(concept_std, dtype=torch.float32)
        if self.return_metadata:
            output["paths"] = {
                **{key: str(value) for key, value in case.modality_paths.items()},
                **({"voi": str(case.voi_path), "tumor_seg": str(case.voi_path)} if case.voi_path is not None else {}),
            }
        return output

    def summary(self) -> Dict[str, object]:
        patient_count = len(self.patient_cases)
        mutant_count = sum(case.label_id == 1 for case in self.patient_cases.values())
        return {
            "split_root": str(self.split_root) if self.split_root is not None else None,
            "manifest_csv": str(self.manifest_csv) if self.manifest_csv is not None else None,
            "split_name": self.split_name,
            "patient_count": patient_count,
            "sample_count": len(self),
            "modalities": list(self.modalities),
            "require_voi": self.require_voi,
            "crop_with_voi": self.crop_with_voi,
            "crop_margin_dhw": list(self.crop_margin),
            "intensity_norm": self.intensity_norm,
            "transform_enabled": self.transform is not None,
            "mutant_count": mutant_count,
            "wild_type_count": patient_count - mutant_count,
        }


class PatientConceptDataset(Dataset):
    """Patient-level concept-only dataset used by Stage2."""

    def __init__(self, volume_dataset: HabitatIDHVolumeDataset) -> None:
        if not volume_dataset.enable_concepts:
            raise ValueError("PatientConceptDataset requires concept-enabled volume dataset.")
        self.block_dataset = volume_dataset
        self.volume_dataset = volume_dataset
        self.patient_ids = sorted(volume_dataset.patient_cases.keys())

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, index: int) -> Dict[str, object]:
        patient_id = self.patient_ids[index]
        patient_case = self.volume_dataset.patient_cases[patient_id]
        concept_raw, concept_std = self.volume_dataset._get_concepts(patient_id)
        return {
            "patient_id": patient_id,
            "label": torch.tensor(patient_case.label_id, dtype=torch.long),
            "concept_true_raw": torch.as_tensor(concept_raw, dtype=torch.float32),
            "concept_true_std": torch.as_tensor(concept_std, dtype=torch.float32),
        }


def build_patient_balanced_sampler(dataset: HabitatIDHVolumeDataset) -> WeightedRandomSampler:
    if len(dataset.sample_index) == 0:
        raise ValueError("Cannot build sampler for empty dataset.")
    class_patient_counts = Counter(case.label_id for case in dataset.patient_cases.values())
    weights: List[float] = []
    for sample in dataset.sample_index:
        label_id = int(dataset.patient_cases[sample.patient_id].label_id)
        class_count = float(class_patient_counts[label_id])
        if class_count <= 0.0:
            raise ValueError(f"Class {label_id} has no patients for balanced sampling.")
        weights.append(1.0 / class_count)
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
    )


def build_habitat_cbm_3d_datasets(
    split_base_root: str | Path | None = None,
    manifest_csv: str | Path | None = None,
    modalities: Sequence[str] = DEFAULT_3D_MODALITIES,
    require_voi: bool = True,
    crop_with_voi: bool = True,
    crop_margin: Sequence[int] = DEFAULT_3D_CROP_MARGIN,
    intensity_norm: str = "zscore",
    cache_volumes: bool = True,
    concept_label_csv: str | Path | None = None,
    concept_scaler_json: str | Path | None = None,
    concept_columns: Sequence[str] | None = None,
    transform_map: Optional[Mapping[str, object]] = None,
) -> Dict[str, HabitatIDHVolumeDataset]:
    base_root = Path(split_base_root) if split_base_root is not None else None
    manifest_path = Path(manifest_csv) if manifest_csv is not None else None
    if base_root is None and manifest_path is None:
        raise ValueError("Either split_base_root or manifest_csv must be provided.")
    datasets: Dict[str, HabitatIDHVolumeDataset] = {}
    for split in ("train", "val", "test"):
        split_root = None
        if manifest_path is None:
            assert base_root is not None
            split_root = base_root / split
            if not split_root.is_dir():
                raise FileNotFoundError(f"Missing split directory: {split_root}")
        transform = transform_map[split] if transform_map and split in transform_map else None
        datasets[split] = HabitatIDHVolumeDataset(
            split_root=split_root,
            manifest_csv=manifest_path,
            split_name=split,
            modalities=modalities,
            require_voi=require_voi,
            crop_with_voi=crop_with_voi,
            crop_margin=crop_margin,
            intensity_norm=intensity_norm,
            cache_volumes=cache_volumes,
            return_metadata=True,
            transform=transform,
            concept_label_csv=concept_label_csv,
            concept_scaler_json=concept_scaler_json,
            concept_columns=concept_columns if concept_columns is not None else DEFAULT_CONCEPT_COLUMNS,
        )
    return datasets


def build_habitat_cbm_3d_dataloaders(
    datasets: Mapping[str, HabitatIDHVolumeDataset],
    batch_size: int,
    num_workers: int,
    train_shuffle: bool = True,
    patient_balanced_sampling: bool = False,
) -> Dict[str, DataLoader]:
    train_sampler = build_patient_balanced_sampler(datasets["train"]) if patient_balanced_sampling else None
    return {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=False if train_sampler is not None else train_shuffle,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        ),
    }


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Habitat-CBM 3D data loader debug CLI.")
    parser.add_argument("--split-root", type=Path, default=None)
    parser.add_argument("--manifest-csv", type=Path, default=None)
    parser.add_argument("--split", type=str, default="train", choices=("train", "val", "test"))
    parser.add_argument("--modalities", type=str, default=",".join(DEFAULT_3D_MODALITIES))
    parser.add_argument("--require-voi", type=str2bool, default=True)
    parser.add_argument("--crop-with-voi", type=str2bool, default=True)
    parser.add_argument("--crop-margin", type=str, default=",".join(str(v) for v in DEFAULT_3D_CROP_MARGIN))
    parser.add_argument("--intensity-norm", choices=("zscore", "minmax", "none"), default="zscore")
    parser.add_argument("--cache-volumes", type=str2bool, default=True)
    parser.add_argument("--concept-label-csv", type=Path, default=None)
    parser.add_argument("--concept-scaler-json", type=Path, default=None)
    parser.add_argument("--selected-concepts", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=3)
    return parser


def _main() -> None:
    args = _build_argparser().parse_args()
    selected_concepts = resolve_concept_names(args.selected_concepts) if args.selected_concepts is not None else DEFAULT_CONCEPT_NAMES
    dataset = HabitatIDHVolumeDataset(
        split_root=args.split_root,
        manifest_csv=args.manifest_csv,
        split_name=args.split if args.manifest_csv is not None else None,
        modalities=tuple(item.strip().lower() for item in args.modalities.split(",") if item.strip()),
        require_voi=args.require_voi,
        crop_with_voi=args.crop_with_voi,
        crop_margin=_parse_int_triple(args.crop_margin, DEFAULT_3D_CROP_MARGIN, name="crop_margin"),
        intensity_norm=args.intensity_norm,
        cache_volumes=args.cache_volumes,
        concept_label_csv=args.concept_label_csv,
        concept_scaler_json=args.concept_scaler_json,
        concept_columns=concept_names_to_columns(selected_concepts),
    )
    print("Dataset summary:")
    for key, value in sorted(dataset.summary().items()):
        print(f"  {key}: {value}")
    print("\nSample preview:")
    for idx in range(min(args.max_samples, len(dataset))):
        sample = dataset[idx]
        line = (
            f"[{idx}] patient={sample['patient_id']} image_shape={tuple(sample['image'].shape)} "
            f"label={int(sample['label'])}"
        )
        if "concept_true_std" in sample:
            line += f" concept_dim={tuple(sample['concept_true_std'].shape)}"
        print(line)


if __name__ == "__main__":
    _main()
