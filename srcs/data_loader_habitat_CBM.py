#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Habitat-CBM 数据配置器。

设计目标：
1. 保持 baseline `srcs/data_loader.py` 不变；
2. 在 CBM 侧直接扩展 `HabitatIDHBlockDataset`，新增 concept 标签接入能力；
3. 统一输出 image/label + concept_true_raw/concept_true_std，供 train/eval/intervention 共用。
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

import sys

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# 复用已有通用加载器，避免改动 baseline 路径。
from srcs.data_loader import (
    DEFAULT_MODALITIES,
    HabitatIDHBlockDataset as _BaseHabitatIDHBlockDataset,
    str2bool,
)

DEFAULT_CONCEPT_NAMES = tuple(f"c{i}" for i in range(1, 9))
DEFAULT_CONCEPT_COLUMNS = tuple(f"{name}_true" for name in DEFAULT_CONCEPT_NAMES)


@dataclass(frozen=True)
class ConceptScaler:
    """概念标准化统计。"""

    concept_names: Tuple[str, ...]
    mean: np.ndarray  # [K]
    std: np.ndarray  # [K]
    fit_split: str
    source_columns: Tuple[str, ...]

    def standardize(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        if values.shape != self.mean.shape:
            raise ValueError(
                f"Concept value shape mismatch: expected {self.mean.shape}, got {values.shape}."
            )
        return (values - self.mean) / self.std

    def destandardize(self, values_std: np.ndarray) -> np.ndarray:
        values_std = np.asarray(values_std, dtype=np.float32)
        if values_std.shape != self.mean.shape:
            raise ValueError(
                f"Concept std value shape mismatch: expected {self.mean.shape}, got {values_std.shape}."
            )
        return values_std * self.std + self.mean


@dataclass(frozen=True)
class ConceptEntry:
    """每位患者对应的一条概念标签记录。"""

    patient_id: str
    split: str
    y_true: int
    label_name: str
    concept_true_raw: np.ndarray  # [K]


def _normalize_patient_id(value: object) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError("Empty patient_id found in concept labels.")
    return text


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return [dict(row) for row in reader]


def load_concept_labels(
    concept_label_csv: str | Path,
    concept_columns: Sequence[str] = DEFAULT_CONCEPT_COLUMNS,
) -> Dict[str, ConceptEntry]:
    """读取 concept_labels.csv 并按 patient_id 建索引。"""

    path = Path(concept_label_csv)
    if not path.is_file():
        raise FileNotFoundError(f"Concept label CSV not found: {path}")

    rows = _read_csv_rows(path)
    required = {"patient_id", "split", "y_true", "label_name", *concept_columns}
    if not rows:
        raise ValueError(f"Concept label CSV is empty: {path}")

    missing = sorted(required - set(rows[0].keys()))
    if missing:
        raise ValueError(f"Missing required concept label columns: {missing}")

    mapping: Dict[str, ConceptEntry] = {}
    for row in rows:
        patient_id = _normalize_patient_id(row["patient_id"])
        if patient_id in mapping:
            raise ValueError(f"Duplicate patient_id in concept labels: {patient_id}")

        split = str(row["split"]).strip().lower()
        label_name = str(row["label_name"]).strip()

        try:
            y_true = int(float(row["y_true"]))
        except ValueError as exc:
            raise ValueError(
                f"Invalid y_true for patient {patient_id}: {row['y_true']}"
            ) from exc
        if y_true not in {0, 1}:
            raise ValueError(f"y_true must be 0/1 for patient {patient_id}, got {y_true}.")

        values: List[float] = []
        for col in concept_columns:
            value_text = str(row[col]).strip()
            if value_text == "":
                raise ValueError(f"Empty concept value '{col}' for patient {patient_id}.")
            try:
                value = float(value_text)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid concept value '{col}' for patient {patient_id}: {value_text}"
                ) from exc
            if not np.isfinite(value):
                raise ValueError(
                    f"Non-finite concept value '{col}' for patient {patient_id}: {value}"
                )
            values.append(value)

        mapping[patient_id] = ConceptEntry(
            patient_id=patient_id,
            split=split,
            y_true=y_true,
            label_name=label_name,
            concept_true_raw=np.asarray(values, dtype=np.float32),
        )

    return mapping


def load_concept_scaler(concept_scaler_json: str | Path) -> ConceptScaler:
    """读取 concept_scaler_stats.json。"""

    path = Path(concept_scaler_json)
    if not path.is_file():
        raise FileNotFoundError(f"Concept scaler JSON not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if "concepts" not in payload:
        raise ValueError(f"Invalid scaler JSON: missing 'concepts' in {path}")

    concepts = payload["concepts"]
    if not isinstance(concepts, Mapping):
        raise ValueError("Invalid scaler JSON: 'concepts' must be a mapping.")

    concept_names = sorted(concepts.keys(), key=lambda item: int(item[1:]) if item[1:].isdigit() else item)
    if not concept_names:
        raise ValueError("Invalid scaler JSON: no concept entries found.")

    means: List[float] = []
    stds: List[float] = []
    source_columns: List[str] = []
    for concept_name in concept_names:
        item = concepts[concept_name]
        if not isinstance(item, Mapping):
            raise ValueError(f"Invalid scaler JSON entry for {concept_name}")

        mean = float(item.get("mean"))
        std = float(item.get("std"))
        source = str(item.get("source", f"{concept_name}_true"))

        if not np.isfinite(mean):
            raise ValueError(f"Non-finite mean for {concept_name}: {mean}")
        if not np.isfinite(std):
            raise ValueError(f"Non-finite std for {concept_name}: {std}")
        if std <= 0:
            raise ValueError(f"Std must be positive for {concept_name}, got {std}")

        means.append(mean)
        stds.append(std)
        source_columns.append(source)

    fit_split = str(payload.get("fit_split", "train"))
    return ConceptScaler(
        concept_names=tuple(concept_names),
        mean=np.asarray(means, dtype=np.float32),
        std=np.asarray(stds, dtype=np.float32),
        fit_split=fit_split,
        source_columns=tuple(source_columns),
    )


class HabitatIDHBlockDataset(_BaseHabitatIDHBlockDataset):
    """CBM 扩展版 Dataset。"""

    def __init__(
        self,
        split_root: str | Path,
        modalities: Sequence[str] = DEFAULT_MODALITIES,
        require_voi: bool = True,
        append_voi_mask: bool = True,
        mask_background_with_voi: bool = False,
        block_depth: int = 5,
        slice_axis: int = 2,
        intensity_norm: str = "zscore",
        min_nonzero_voxels: int = 16,
        cache_volumes: bool = True,
        return_metadata: bool = True,
        transform=None,
        concept_label_csv: str | Path | None = None,
        concept_scaler_json: str | Path | None = None,
        concept_columns: Sequence[str] = DEFAULT_CONCEPT_COLUMNS,
    ) -> None:
        super().__init__(
            split_root=split_root,
            modalities=modalities,
            require_voi=require_voi,
            append_voi_mask=append_voi_mask,
            mask_background_with_voi=mask_background_with_voi,
            block_depth=block_depth,
            slice_axis=slice_axis,
            intensity_norm=intensity_norm,
            min_nonzero_voxels=min_nonzero_voxels,
            cache_volumes=cache_volumes,
            return_metadata=return_metadata,
            transform=transform,
        )

        self.concept_columns = tuple(concept_columns)
        self.enable_concepts = concept_label_csv is not None or concept_scaler_json is not None
        self._concept_map: Dict[str, ConceptEntry] = {}
        self._concept_scaler: Optional[ConceptScaler] = None

        if self.enable_concepts:
            if concept_label_csv is None or concept_scaler_json is None:
                raise ValueError(
                    "When concept mode is enabled, both concept_label_csv and concept_scaler_json are required."
                )
            self._concept_map = load_concept_labels(
                concept_label_csv=concept_label_csv,
                concept_columns=self.concept_columns,
            )
            self._concept_scaler = load_concept_scaler(concept_scaler_json)
            self._validate_concept_alignment()

    @property
    def concept_dim(self) -> int:
        return len(self.concept_columns)

    def _validate_concept_alignment(self) -> None:
        if self._concept_scaler is None:
            raise RuntimeError("Concept scaler is not initialized.")

        expected_split = self.split_root.name.strip().lower()
        if self._concept_scaler.mean.shape[0] != self.concept_dim:
            raise ValueError(
                "Concept dimension mismatch between concept_columns and scaler stats: "
                f"{self.concept_dim} vs {self._concept_scaler.mean.shape[0]}"
            )

        for patient_id, case in self.patient_cases.items():
            if patient_id not in self._concept_map:
                raise KeyError(
                    f"Patient {patient_id} exists in dataset but missing in concept labels."
                )
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
        entry = self._concept_map[patient_id]
        raw = entry.concept_true_raw.astype(np.float32)
        std = self._concept_scaler.standardize(raw).astype(np.float32)
        return raw, std

    def __getitem__(self, index: int) -> Dict[str, object]:
        output = super().__getitem__(index)
        if not self.enable_concepts:
            return output

        patient_id = str(output["patient_id"]) if "patient_id" in output else str(
            self.sample_index[index].patient_id
        )
        concept_raw, concept_std = self._get_concepts(patient_id)
        output["concept_true_raw"] = torch.as_tensor(concept_raw, dtype=torch.float32)
        output["concept_true_std"] = torch.as_tensor(concept_std, dtype=torch.float32)
        return output


class PatientConceptDataset(Dataset):
    """患者级 Stage2 数据集：每位患者仅保留一条概念样本。"""

    def __init__(self, block_dataset: HabitatIDHBlockDataset) -> None:
        if not block_dataset.enable_concepts:
            raise ValueError("PatientConceptDataset requires concept-enabled block dataset.")
        self.block_dataset = block_dataset
        self.patient_ids = sorted(block_dataset.patient_cases.keys())

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, index: int) -> Dict[str, object]:
        patient_id = self.patient_ids[index]
        patient_case = self.block_dataset.patient_cases[patient_id]
        concept_raw, concept_std = self.block_dataset._get_concepts(patient_id)
        return {
            "patient_id": patient_id,
            "label": torch.tensor(patient_case.label_id, dtype=torch.long),
            "concept_true_raw": torch.as_tensor(concept_raw, dtype=torch.float32),
            "concept_true_std": torch.as_tensor(concept_std, dtype=torch.float32),
        }


def build_patient_balanced_sampler(dataset: HabitatIDHBlockDataset) -> WeightedRandomSampler:
    """构建患者均衡 + 类别均衡的 block 级采样器。"""

    if len(dataset.sample_index) == 0:
        raise ValueError("Cannot build sampler for empty dataset.")

    patient_block_counts = Counter(item.patient_id for item in dataset.sample_index)
    class_patient_counts = Counter(case.label_id for case in dataset.patient_cases.values())
    for label_id, count in class_patient_counts.items():
        if count <= 0:
            raise ValueError(f"Class {label_id} has no patients for balanced sampling.")

    weights: List[float] = []
    for sample in dataset.sample_index:
        patient_id = sample.patient_id
        label_id = int(dataset.patient_cases[patient_id].label_id)
        patient_blocks = float(patient_block_counts[patient_id])
        class_patients = float(class_patient_counts[label_id])
        if patient_blocks <= 0.0 or class_patients <= 0.0:
            raise ValueError(
                "Invalid patient/class count when building balanced sampler: "
                f"patient={patient_id}, label={label_id}, blocks={patient_blocks}, class_patients={class_patients}"
            )
        weights.append(1.0 / (patient_blocks * class_patients))

    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
    )


def build_habitat_cbm_datasets(
    split_base_root: str | Path,
    modalities: Sequence[str] = DEFAULT_MODALITIES,
    require_voi: bool = True,
    append_voi_mask: bool = True,
    mask_background_with_voi: bool = False,
    block_depth: int = 5,
    slice_axis: int = 2,
    intensity_norm: str = "zscore",
    min_nonzero_voxels: int = 16,
    cache_volumes: bool = True,
    concept_label_csv: str | Path | None = None,
    concept_scaler_json: str | Path | None = None,
    transform_map: Optional[Mapping[str, object]] = None,
) -> Dict[str, HabitatIDHBlockDataset]:
    """构建 train/val/test 三个 CBM 数据集。"""

    base_root = Path(split_base_root)
    datasets: Dict[str, HabitatIDHBlockDataset] = {}
    for split in ("train", "val", "test"):
        split_root = base_root / split
        if not split_root.is_dir():
            raise FileNotFoundError(f"Missing split directory: {split_root}")

        transform = transform_map[split] if transform_map and split in transform_map else None
        datasets[split] = HabitatIDHBlockDataset(
            split_root=split_root,
            modalities=modalities,
            require_voi=require_voi,
            append_voi_mask=append_voi_mask,
            mask_background_with_voi=mask_background_with_voi,
            block_depth=block_depth,
            slice_axis=slice_axis,
            intensity_norm=intensity_norm,
            min_nonzero_voxels=min_nonzero_voxels,
            cache_volumes=cache_volumes,
            return_metadata=True,
            transform=transform,
            concept_label_csv=concept_label_csv,
            concept_scaler_json=concept_scaler_json,
        )
    return datasets


def build_habitat_cbm_dataloaders(
    datasets: Mapping[str, HabitatIDHBlockDataset],
    batch_size: int,
    num_workers: int,
    train_shuffle: bool = True,
    patient_balanced_sampling: bool = False,
) -> Dict[str, DataLoader]:
    """构建 CBM 训练/验证/测试 DataLoader。"""

    train_sampler = None
    if patient_balanced_sampling:
        train_sampler = build_patient_balanced_sampler(datasets["train"])

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
    parser = argparse.ArgumentParser(description="Habitat-CBM data loader debug CLI.")
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--block-depth", type=int, default=5)
    parser.add_argument("--require-voi", type=str2bool, default=True)
    parser.add_argument("--append-voi-mask", type=str2bool, default=True)
    parser.add_argument("--mask-background-with-voi", type=str2bool, default=False)
    parser.add_argument("--slice-axis", type=int, default=2)
    parser.add_argument("--intensity-norm", choices=("zscore", "minmax", "none"), default="zscore")
    parser.add_argument("--min-nonzero-voxels", type=int, default=16)
    parser.add_argument("--cache-volumes", type=str2bool, default=True)
    parser.add_argument("--concept-label-csv", type=Path, default=None)
    parser.add_argument("--concept-scaler-json", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=3)
    return parser


def _main() -> None:
    args = _build_argparser().parse_args()
    dataset = HabitatIDHBlockDataset(
        split_root=args.split_root,
        block_depth=args.block_depth,
        require_voi=args.require_voi,
        append_voi_mask=args.append_voi_mask,
        mask_background_with_voi=args.mask_background_with_voi,
        slice_axis=args.slice_axis,
        intensity_norm=args.intensity_norm,
        min_nonzero_voxels=args.min_nonzero_voxels,
        cache_volumes=args.cache_volumes,
        return_metadata=True,
        concept_label_csv=args.concept_label_csv,
        concept_scaler_json=args.concept_scaler_json,
    )

    summary = dataset.summary()
    print("Dataset summary:")
    for key in sorted(summary.keys()):
        print(f"  {key}: {summary[key]}")
    print(f"  concept_enabled: {dataset.enable_concepts}")
    if dataset.enable_concepts:
        print(f"  concept_dim: {dataset.concept_dim}")

    print("\nSample preview:")
    for i in range(min(args.max_samples, len(dataset))):
        sample = dataset[i]
        line = (
            f"[{i}] patient={sample.get('patient_id')} slice={sample.get('slice_index')} "
            f"image_shape={tuple(sample['image'].shape)} label={int(sample['label'])}"
        )
        if "concept_true_raw" in sample:
            raw = sample["concept_true_raw"].tolist()
            std = sample["concept_true_std"].tolist()
            line += f" concept_raw={raw} concept_std={std}"
        print(line)


if __name__ == "__main__":
    _main()
