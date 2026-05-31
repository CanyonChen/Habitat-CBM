#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
导出指定患者的中心 VOI 切片 PNG。

功能说明：
1. 从 `dataset/splited_data/manifests/split_assignments.csv` 自动定位患者所在 split 与标签目录；
2. 读取六模态 MRI（t1/t1ce/t2/t2flair/adc/cbf）和 canonical `functional/voi`；
3. 按 `axis=2` 在 VOI 非零切片中选取最中心的 5 张（若不足 5 张则全部导出）；
4. 导出六模态 ROI overlay PNG；
5. 导出 H1/H2/H3 生境 RGB overlay PNG，其中 H1=红、H2=绿、H3=蓝。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
PROJECT_ROOT = REPO_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from srcs.data_loader import (  # noqa: E402
    discover_modality_files,
    discover_voi_file,
    extract_slice_2d,
    load_nifti_array,
    normalize_volume,
)

DEFAULT_MODALITIES = ("t1", "t1ce", "t2", "t2flair", "adc", "cbf")
DEFAULT_PATIENT_IDS = ("003", "005")
DEFAULT_SPLIT_BASE_ROOT = PROJECT_ROOT / "dataset" / "splited_data"
DEFAULT_HABITAT_ROOT = PROJECT_ROOT / "dataset" / "habitat_masks"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "get_png"
ROI_COLOR = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
HABITAT_COLORS: Mapping[str, np.ndarray] = {
    "h1": np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
    "h2": np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
    "h3": np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
}


@dataclass(frozen=True)
class PatientLocation:
    patient_id: str
    split_name: str
    label_name: str


@dataclass(frozen=True)
class PatientExportSummary:
    patient_id: str
    split_name: str
    label_name: str
    selected_slices: Tuple[int, ...]
    roi_modalities: Tuple[str, ...]
    roi_image_count: int
    habitat_image_count: int
    habitat_background_modality: str


@dataclass(frozen=True)
class PatientPayload:
    location: PatientLocation
    modality_paths: Dict[str, Path]
    voi_path: Path
    habitat_paths: Dict[str, Path]
    modality_volumes: Dict[str, np.ndarray]
    voi_volume: np.ndarray
    habitat_volumes: Dict[str, np.ndarray]


def normalize_patient_id(value: object) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError("patient_id 不能为空。")
    if text.isdigit():
        return text.zfill(3)
    return text


def parse_patient_ids(raw_value: str) -> Tuple[str, ...]:
    patient_ids = [normalize_patient_id(item) for item in raw_value.split(",") if item.strip()]
    if not patient_ids:
        raise ValueError("--patient-ids 至少要包含一个患者编号。")

    deduplicated: List[str] = []
    for patient_id in patient_ids:
        if patient_id not in deduplicated:
            deduplicated.append(patient_id)
    return tuple(deduplicated)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="导出指定患者的 ROI / Habitat PNG 可视化结果。")
    parser.add_argument(
        "--split-base-root",
        type=Path,
        default=DEFAULT_SPLIT_BASE_ROOT,
        help="包含 train/val/test 和 manifests 的数据划分根目录。",
    )
    parser.add_argument(
        "--habitat-root",
        type=Path,
        default=DEFAULT_HABITAT_ROOT,
        help="Habitat masks 根目录。",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="PNG 输出根目录。",
    )
    parser.add_argument(
        "--patient-ids",
        type=str,
        default=",".join(DEFAULT_PATIENT_IDS),
        help="逗号分隔的患者编号，例如 003,005。",
    )
    parser.add_argument(
        "--modalities",
        type=str,
        default=",".join(DEFAULT_MODALITIES),
        help="要导出的模态列表，逗号分隔。",
    )
    parser.add_argument(
        "--num-slices",
        type=int,
        default=5,
        help="每位患者导出的中心切片数量。",
    )
    parser.add_argument(
        "--slice-axis",
        type=int,
        choices=(0, 1, 2),
        default=2,
        help="切片轴，默认与当前项目一致使用 axis=2。",
    )
    parser.add_argument(
        "--intensity-norm",
        choices=("zscore", "minmax", "none"),
        default="zscore",
        help="读取后用于可视化底图的体数据归一化方式。",
    )
    parser.add_argument(
        "--habitat-background-modality",
        type=str,
        default="t1ce",
        help="Habitat overlay 底图所使用的模态。",
    )
    return parser


def load_patient_locations(manifest_path: Path) -> Dict[str, PatientLocation]:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"找不到 split manifest: {manifest_path}")

    mapping: Dict[str, PatientLocation] = {}
    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"manifest 缺少表头: {manifest_path}")

        for row in reader:
            patient_id = normalize_patient_id(row.get("patient_id", ""))
            split_name = str(row.get("split", "")).strip().lower()
            label_name = str(row.get("idh_class", "")).strip()
            if not split_name or not label_name:
                raise ValueError(f"manifest 行缺少 split 或 label 信息: {row}")
            mapping[patient_id] = PatientLocation(
                patient_id=patient_id,
                split_name=split_name,
                label_name=label_name,
            )
    return mapping


def resolve_mask_file(mask_dir: Path, mask_name: str) -> Path:
    candidates = sorted(mask_dir.glob(f"{mask_name}.nii*"))
    if not candidates:
        raise FileNotFoundError(f"找不到 habitat 掩膜 `{mask_name}`: {mask_dir}")
    if len(candidates) > 1:
        candidate_text = "\n".join(str(item) for item in candidates)
        raise ValueError(
            f"发现多个 `{mask_name}` 候选文件，请保留唯一文件：\n{candidate_text}"
        )
    return candidates[0]


def validate_same_shape(reference_name: str, reference_volume: np.ndarray, target_name: str, target_volume: np.ndarray) -> None:
    if tuple(reference_volume.shape) != tuple(target_volume.shape):
        raise ValueError(
            f"体数据 shape 不一致：{reference_name}={reference_volume.shape}, "
            f"{target_name}={target_volume.shape}"
        )


def load_patient_payload(
    split_base_root: Path,
    habitat_root: Path,
    location: PatientLocation,
    modalities: Sequence[str],
    intensity_norm: str,
) -> PatientPayload:
    split_root = split_base_root / location.split_name
    patient_dirs = {
        "conventional": split_root / "conventional" / location.label_name / location.patient_id,
        "functional": split_root / "functional" / location.label_name / location.patient_id,
    }
    for branch_name, branch_dir in patient_dirs.items():
        if not branch_dir.is_dir():
            raise FileNotFoundError(
                f"患者 {location.patient_id} 缺少 {branch_name} 目录: {branch_dir}"
            )

    modality_paths = discover_modality_files(patient_dirs, modalities)
    voi_path = discover_voi_file(patient_dirs)

    habitat_patient_dir = habitat_root / location.split_name / location.label_name / location.patient_id
    if not habitat_patient_dir.is_dir():
        raise FileNotFoundError(f"找不到 habitat 目录: {habitat_patient_dir}")
    habitat_paths = {
        name: resolve_mask_file(habitat_patient_dir, name)
        for name in ("h1", "h2", "h3")
    }

    modality_volumes = {
        modality: normalize_volume(load_nifti_array(path), intensity_norm)
        for modality, path in modality_paths.items()
    }
    first_modality = modalities[0]
    reference_volume = modality_volumes[first_modality]
    voi_volume = (load_nifti_array(voi_path) > 0).astype(np.float32)
    validate_same_shape(first_modality, reference_volume, "voi", voi_volume)

    habitat_volumes: Dict[str, np.ndarray] = {}
    for name, path in habitat_paths.items():
        habitat_volume = (load_nifti_array(path) > 0).astype(np.float32)
        validate_same_shape(first_modality, reference_volume, name, habitat_volume)
        habitat_volumes[name] = habitat_volume

    for modality, volume in modality_volumes.items():
        if volume.ndim != 3:
            raise ValueError(
                f"患者 {location.patient_id} 的模态 `{modality}` 不是 3D 体数据: {volume.shape}"
            )
        validate_same_shape(first_modality, reference_volume, modality, volume)

    return PatientPayload(
        location=location,
        modality_paths=modality_paths,
        voi_path=voi_path,
        habitat_paths=habitat_paths,
        modality_volumes=modality_volumes,
        voi_volume=voi_volume,
        habitat_volumes=habitat_volumes,
    )


def select_center_slice_indices(voi_volume: np.ndarray, axis: int, max_slices: int) -> Tuple[int, ...]:
    if max_slices <= 0:
        raise ValueError("max_slices 必须大于 0。")

    valid_slices: List[Tuple[int, int]] = []
    num_slices = voi_volume.shape[axis]
    for slice_index in range(num_slices):
        slice_2d = extract_slice_2d(voi_volume, slice_index, axis)
        voxel_count = int(np.count_nonzero(slice_2d > 0))
        if voxel_count > 0:
            valid_slices.append((slice_index, voxel_count))

    if not valid_slices:
        raise ValueError("VOI 中没有任何有效切片，无法导出 PNG。")

    weighted_center = float(
        sum(slice_index * voxel_count for slice_index, voxel_count in valid_slices)
        / sum(voxel_count for _, voxel_count in valid_slices)
    )
    ranked = sorted(
        valid_slices,
        key=lambda item: (abs(item[0] - weighted_center), -item[1], item[0]),
    )
    chosen = sorted(slice_index for slice_index, _ in ranked[:max_slices])
    return tuple(chosen)


def robust_normalize_slice(slice_2d: np.ndarray) -> np.ndarray:
    array = np.asarray(slice_2d, dtype=np.float32)
    finite_mask = np.isfinite(array)
    if not np.any(finite_mask):
        return np.zeros_like(array, dtype=np.float32)

    focus = array[finite_mask]
    nonzero_focus = focus[np.abs(focus) > 1e-8]
    if nonzero_focus.size > 0:
        focus = nonzero_focus

    lower = float(np.percentile(focus, 1.0))
    upper = float(np.percentile(focus, 99.0))
    if upper - lower < 1e-8:
        lower = float(focus.min())
        upper = float(focus.max())
    if upper - lower < 1e-8:
        return np.zeros_like(array, dtype=np.float32)

    clipped = np.clip(array, lower, upper)
    normalized = (clipped - lower) / (upper - lower)
    normalized[~finite_mask] = 0.0
    return normalized.astype(np.float32)


def to_rgb(slice_2d: np.ndarray) -> np.ndarray:
    base = robust_normalize_slice(slice_2d)
    return np.repeat(base[..., None], 3, axis=2)


def compute_mask_edge(mask_2d: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask_2d > 0, dtype=bool)
    if not np.any(mask):
        return np.zeros_like(mask, dtype=bool)

    padded = np.pad(mask, ((1, 1), (1, 1)), mode="constant", constant_values=False)
    center = padded[1:-1, 1:-1]
    neighbor_views = [
        padded[:-2, :-2],
        padded[:-2, 1:-1],
        padded[:-2, 2:],
        padded[1:-1, :-2],
        padded[1:-1, 2:],
        padded[2:, :-2],
        padded[2:, 1:-1],
        padded[2:, 2:],
    ]
    interior = center.copy()
    for neighbor in neighbor_views:
        interior &= neighbor
    return center & (~interior)


def apply_color_overlay(base_rgb: np.ndarray, mask_2d: np.ndarray, color: np.ndarray, alpha: float) -> np.ndarray:
    output = np.asarray(base_rgb, dtype=np.float32).copy()
    mask = np.asarray(mask_2d > 0, dtype=bool)
    if np.any(mask):
        output[mask] = (1.0 - alpha) * output[mask] + alpha * color
    return output


def paint_mask_edge(base_rgb: np.ndarray, mask_2d: np.ndarray, color: np.ndarray) -> np.ndarray:
    output = np.asarray(base_rgb, dtype=np.float32).copy()
    edge = compute_mask_edge(mask_2d)
    if np.any(edge):
        output[edge] = color
    return output


def compose_roi_overlay(image_slice: np.ndarray, voi_slice: np.ndarray) -> np.ndarray:
    output = to_rgb(image_slice)
    output = apply_color_overlay(output, voi_slice, ROI_COLOR, alpha=0.28)
    output = paint_mask_edge(output, voi_slice, ROI_COLOR)
    return np.clip(output, 0.0, 1.0)


def compose_habitat_overlay(
    background_slice: np.ndarray,
    voi_slice: np.ndarray,
    habitat_slices: Mapping[str, np.ndarray],
) -> np.ndarray:
    output = to_rgb(background_slice)
    for habitat_name in ("h1", "h2", "h3"):
        output = apply_color_overlay(
            output,
            habitat_slices[habitat_name],
            HABITAT_COLORS[habitat_name],
            alpha=0.42,
        )
        output = paint_mask_edge(output, habitat_slices[habitat_name], HABITAT_COLORS[habitat_name])
    output = paint_mask_edge(output, voi_slice, np.asarray([1.0, 1.0, 1.0], dtype=np.float32))
    return np.clip(output, 0.0, 1.0)


def save_png(image_rgb: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(output_path, np.clip(image_rgb, 0.0, 1.0))


def export_patient_pngs(
    payload: PatientPayload,
    output_root: Path,
    modalities: Sequence[str],
    selected_slices: Sequence[int],
    slice_axis: int,
    habitat_background_modality: str,
) -> PatientExportSummary:
    patient_root = output_root / payload.location.patient_id
    roi_root = patient_root / "roi_overlays"
    habitat_root = patient_root / "habitat_overlays"

    roi_count = 0
    habitat_count = 0
    for slice_index in selected_slices:
        voi_slice = extract_slice_2d(payload.voi_volume, slice_index, slice_axis)

        for modality in modalities:
            image_slice = extract_slice_2d(payload.modality_volumes[modality], slice_index, slice_axis)
            roi_overlay = compose_roi_overlay(image_slice, voi_slice)
            roi_path = (
                roi_root
                / modality
                / f"patient_{payload.location.patient_id}_{payload.location.split_name}_{modality}_slice_{slice_index:03d}_roi.png"
            )
            save_png(roi_overlay, roi_path)
            roi_count += 1

        habitat_slices = {
            name: extract_slice_2d(volume, slice_index, slice_axis)
            for name, volume in payload.habitat_volumes.items()
        }
        background_slice = extract_slice_2d(
            payload.modality_volumes[habitat_background_modality],
            slice_index,
            slice_axis,
        )
        habitat_overlay = compose_habitat_overlay(background_slice, voi_slice, habitat_slices)
        habitat_path = (
            habitat_root
            / f"patient_{payload.location.patient_id}_{payload.location.split_name}_{habitat_background_modality}_slice_{slice_index:03d}_habitat_rgb.png"
        )
        save_png(habitat_overlay, habitat_path)
        habitat_count += 1

    return PatientExportSummary(
        patient_id=payload.location.patient_id,
        split_name=payload.location.split_name,
        label_name=payload.location.label_name,
        selected_slices=tuple(int(item) for item in selected_slices),
        roi_modalities=tuple(str(item) for item in modalities),
        roi_image_count=roi_count,
        habitat_image_count=habitat_count,
        habitat_background_modality=habitat_background_modality,
    )


def main() -> None:
    args = build_argparser().parse_args()
    patient_ids = parse_patient_ids(args.patient_ids)
    modalities = tuple(item.strip().lower() for item in args.modalities.split(",") if item.strip())
    if not modalities:
        raise ValueError("至少要指定一个导出模态。")
    if args.habitat_background_modality not in modalities:
        raise ValueError(
            "--habitat-background-modality 必须包含在 --modalities 中，"
            f"当前为 {args.habitat_background_modality}，可选 {modalities}"
        )

    manifest_path = args.split_base_root / "manifests" / "split_assignments.csv"
    location_map = load_patient_locations(manifest_path)

    summaries: List[PatientExportSummary] = []
    for patient_id in patient_ids:
        if patient_id not in location_map:
            raise KeyError(f"manifest 中不存在患者 {patient_id}: {manifest_path}")

        location = location_map[patient_id]
        print(
            f"[Info] 处理患者 {patient_id} | split={location.split_name} | label={location.label_name}",
            flush=True,
        )
        payload = load_patient_payload(
            split_base_root=args.split_base_root,
            habitat_root=args.habitat_root,
            location=location,
            modalities=modalities,
            intensity_norm=args.intensity_norm,
        )
        selected_slices = select_center_slice_indices(
            payload.voi_volume,
            axis=args.slice_axis,
            max_slices=args.num_slices,
        )
        if len(selected_slices) < args.num_slices:
            print(
                f"[Warn] 患者 {patient_id} 只有 {len(selected_slices)} 张有效 VOI 切片，已全部导出。",
                flush=True,
            )
        print(
            f"[Info] 患者 {patient_id} 选中的中心切片: {list(selected_slices)}",
            flush=True,
        )
        summary = export_patient_pngs(
            payload=payload,
            output_root=args.output_root,
            modalities=modalities,
            selected_slices=selected_slices,
            slice_axis=args.slice_axis,
            habitat_background_modality=args.habitat_background_modality,
        )
        summaries.append(summary)

    args.output_root.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_root / "export_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "patient_ids": list(patient_ids),
                "modalities": list(modalities),
                "slice_axis": args.slice_axis,
                "num_slices": args.num_slices,
                "intensity_norm": args.intensity_norm,
                "habitat_background_modality": args.habitat_background_modality,
                "summaries": [asdict(item) for item in summaries],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[Done] PNG 已导出到: {args.output_root}", flush=True)
    print(f"[Done] 汇总信息已保存: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
