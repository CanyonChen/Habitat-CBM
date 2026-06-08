#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Inspect internal/external 3D data-domain differences for Habitat-CBM.

This script reads NIfTI headers and volumes, writes tabular summaries first, and
optionally creates plots afterwards. It does not run model inference.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from srcs.data_loader import load_nifti_array, load_voi_array, normalize_volume  # noqa: E402
from srcs.habitat_CBM_3D.data_loader_habitat_CBM_3D import (  # noqa: E402
    DEFAULT_3D_CROP_MARGIN,
    DEFAULT_3D_MODALITIES,
    HabitatIDHVolumeDataset,
    _voi_bbox_hwd,
)


DEFAULT_CONFIG = CURRENT_DIR / "args_train_habitat_CBM_3D_tuned_v3_noaug_b8_ep100_es10.json"
DEFAULT_EXTERNAL_SPLIT_ROOT = Path("/root/autodl-tmp/habitat_CBM/dataset/splited_data")
SUMMARY_NUMERIC_FIELDS = (
    "shape_h",
    "shape_w",
    "shape_d",
    "spacing_h",
    "spacing_w",
    "spacing_d",
    "voi_voxels",
    "voi_fraction",
    "crop_h",
    "crop_w",
    "crop_d",
    "crop_fraction",
    "raw_nonzero_mean",
    "raw_nonzero_std",
    "raw_nonzero_p1",
    "raw_nonzero_p50",
    "raw_nonzero_p99",
    "raw_crop_mean",
    "raw_crop_std",
    "norm_crop_mean",
    "norm_crop_std",
    "norm_crop_p1",
    "norm_crop_p50",
    "norm_crop_p99",
)


def _load_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Config JSON must be an object: {path}")
    return payload


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


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


def _parse_csv_list(value: str, *, allowed: Optional[set[str]] = None) -> Tuple[str, ...]:
    items = tuple(item.strip().lower() for item in value.split(",") if item.strip())
    if not items:
        raise ValueError("Expected at least one comma-separated item.")
    if allowed is not None:
        invalid = [item for item in items if item not in allowed]
        if invalid:
            raise ValueError(f"Invalid item(s): {invalid}; allowed={sorted(allowed)}")
    return items


def _spacing(path: Path) -> Tuple[float, float, float]:
    try:
        import nibabel as nib

        image = nib.load(str(path))
        zooms = tuple(float(v) for v in image.header.get_zooms()[:3])
        if len(zooms) == 3:
            return zooms
    except Exception:
        pass
    return (float("nan"), float("nan"), float("nan"))


def _finite_stats(values: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "p1": float("nan"),
            "p5": float("nan"),
            "p50": float("nan"),
            "p95": float("nan"),
            "p99": float("nan"),
        }
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p1": float(np.percentile(arr, 1)),
        "p5": float(np.percentile(arr, 5)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }


def _prefixed_stats(prefix: str, values: np.ndarray) -> Dict[str, float]:
    stats = _finite_stats(values)
    return {f"{prefix}_{key}": value for key, value in stats.items()}


def _nonzero_values(volume: np.ndarray) -> np.ndarray:
    arr = np.asarray(volume, dtype=np.float32)
    mask = np.abs(arr) > 1e-8
    return arr[mask] if np.any(mask) else arr.reshape(-1)


def _build_dataset(
    *,
    domain: str,
    split: str,
    manifest_csv: Optional[Path],
    split_base_root: Optional[Path],
    modalities: Sequence[str],
    require_voi: bool,
    crop_with_voi: bool,
    crop_margin: Sequence[int],
    intensity_norm: str,
) -> HabitatIDHVolumeDataset:
    split_root = split_base_root / split if split_base_root is not None else None
    return HabitatIDHVolumeDataset(
        split_root=split_root,
        manifest_csv=manifest_csv,
        split_name=split,
        modalities=modalities,
        require_voi=require_voi,
        crop_with_voi=crop_with_voi,
        crop_margin=crop_margin,
        intensity_norm=intensity_norm,
        cache_volumes=False,
        return_metadata=True,
        transform=None,
        concept_label_csv=None,
        concept_scaler_json=None,
    )


def inspect_dataset(
    *,
    domain: str,
    dataset: HabitatIDHVolumeDataset,
    modalities: Sequence[str],
    crop_margin: Sequence[int],
    max_patients: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    patient_ids = list(dataset.patient_ids)
    if max_patients > 0:
        patient_ids = patient_ids[:max_patients]

    for patient_id in patient_ids:
        case = dataset.patient_cases[patient_id]
        try:
            voi = load_voi_array(case.voi_path) if case.voi_path is not None else None
            if voi is None:
                raise FileNotFoundError(f"Missing VOI path for {patient_id}")
            crop_slices = _voi_bbox_hwd(voi, crop_margin)
            h_slice, w_slice, d_slice = crop_slices
            crop_h = int(h_slice.stop - h_slice.start)
            crop_w = int(w_slice.stop - w_slice.start)
            crop_d = int(d_slice.stop - d_slice.start)
            voi_voxels = int(np.sum(voi > 0))
            total_voxels = int(np.prod(voi.shape))
            crop_voxels = int(crop_h * crop_w * crop_d)
            base_row = {
                "domain": domain,
                "split": str(dataset.split_name),
                "patient_id": patient_id,
                "label_name": case.label_name,
                "y_true": int(case.label_id),
                "voi_path": str(case.voi_path),
                "voi_shape_hwd": "x".join(str(int(v)) for v in voi.shape),
                "voi_voxels": voi_voxels,
                "voi_fraction": float(voi_voxels / total_voxels) if total_voxels else float("nan"),
                "bbox_h0": int(h_slice.start),
                "bbox_h1": int(h_slice.stop),
                "bbox_w0": int(w_slice.start),
                "bbox_w1": int(w_slice.stop),
                "bbox_d0": int(d_slice.start),
                "bbox_d1": int(d_slice.stop),
                "crop_h": crop_h,
                "crop_w": crop_w,
                "crop_d": crop_d,
                "crop_voxels": crop_voxels,
                "crop_fraction": float(crop_voxels / total_voxels) if total_voxels else float("nan"),
            }
            for modality in modalities:
                path = case.modality_paths[modality]
                raw = load_nifti_array(path)
                norm = normalize_volume(raw, dataset.intensity_norm)
                if tuple(raw.shape) != tuple(voi.shape):
                    raise ValueError(f"{modality} shape {tuple(raw.shape)} does not match VOI {tuple(voi.shape)}")
                crop_raw = raw[crop_slices]
                crop_norm = norm[crop_slices]
                spacing_h, spacing_w, spacing_d = _spacing(path)
                shape_h, shape_w, shape_d = (int(v) for v in raw.shape)
                row = {
                    **base_row,
                    "modality": modality,
                    "image_path": str(path),
                    "shape_h": shape_h,
                    "shape_w": shape_w,
                    "shape_d": shape_d,
                    "shape_hwd": f"{shape_h}x{shape_w}x{shape_d}",
                    "spacing_h": spacing_h,
                    "spacing_w": spacing_w,
                    "spacing_d": spacing_d,
                    **_prefixed_stats("raw_nonzero", _nonzero_values(raw)),
                    **_prefixed_stats("raw_crop", crop_raw.reshape(-1)),
                    **_prefixed_stats("norm_crop", crop_norm.reshape(-1)),
                }
                rows.append(row)
        except Exception as exc:
            failures.append(
                {
                    "domain": domain,
                    "split": str(dataset.split_name),
                    "patient_id": patient_id,
                    "label_name": case.label_name,
                    "y_true": int(case.label_id),
                    "reason": type(exc).__name__,
                    "message": str(exc),
                }
            )
    return rows, failures


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["domain"]), str(row["split"]), str(row["modality"]))].append(row)

    summary_rows: List[Dict[str, Any]] = []
    for (domain, split, modality), group in sorted(grouped.items()):
        patients = {str(row["patient_id"]) for row in group}
        positives = {str(row["patient_id"]) for row in group if int(row["y_true"]) == 1}
        summary: Dict[str, Any] = {
            "domain": domain,
            "split": split,
            "modality": modality,
            "n_rows": len(group),
            "n_patients": len(patients),
            "n_positive": len(positives),
            "n_negative": len(patients) - len(positives),
            "prevalence": float(len(positives) / len(patients)) if patients else float("nan"),
        }
        for field in SUMMARY_NUMERIC_FIELDS:
            values: List[float] = []
            for row in group:
                try:
                    value = float(row.get(field, "nan"))
                except (TypeError, ValueError):
                    value = float("nan")
                if math.isfinite(value):
                    values.append(value)
            arr = np.asarray(values, dtype=np.float64)
            if arr.size == 0:
                summary[f"{field}_mean"] = float("nan")
                summary[f"{field}_std"] = float("nan")
                summary[f"{field}_median"] = float("nan")
                summary[f"{field}_min"] = float("nan")
                summary[f"{field}_max"] = float("nan")
            else:
                summary[f"{field}_mean"] = float(np.mean(arr))
                summary[f"{field}_std"] = float(np.std(arr))
                summary[f"{field}_median"] = float(np.median(arr))
                summary[f"{field}_min"] = float(np.min(arr))
                summary[f"{field}_max"] = float(np.max(arr))
        summary_rows.append(summary)
    return summary_rows


def export_figures(output_dir: Path, rows: Sequence[Mapping[str, Any]], *, dpi: int) -> Dict[str, str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - depends on runtime
        print(f"[WARN] matplotlib unavailable; skipping figures: {exc}")
        return {}

    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    generated: Dict[str, str] = {}
    fields = ("voi_fraction", "crop_fraction", "raw_nonzero_mean", "raw_nonzero_std", "norm_crop_std")
    domains = sorted({str(row["domain"]) for row in rows})
    for field in fields:
        values_by_domain: List[np.ndarray] = []
        labels: List[str] = []
        for domain in domains:
            values = []
            seen = set()
            for row in rows:
                if str(row["domain"]) != domain:
                    continue
                patient_modality = (row["patient_id"], row.get("modality", ""))
                if patient_modality in seen:
                    continue
                seen.add(patient_modality)
                try:
                    value = float(row.get(field, "nan"))
                except (TypeError, ValueError):
                    value = float("nan")
                if math.isfinite(value):
                    values.append(value)
            if values:
                values_by_domain.append(np.asarray(values, dtype=np.float64))
                labels.append(domain)
        if not values_by_domain:
            continue
        path = figure_dir / f"domain_{field}.png"
        fig, ax = plt.subplots(figsize=(5.5, 4.2), dpi=dpi)
        ax.boxplot(values_by_domain, labels=labels, showmeans=True)
        ax.set_title(f"Domain comparison: {field}")
        ax.set_ylabel(field)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        generated[field] = str(path)
    return generated


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect 3D Habitat-CBM internal/external data-domain stats.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--internal-manifest-csv", type=Path, default=None)
    parser.add_argument("--external-split-root", type=Path, default=DEFAULT_EXTERNAL_SPLIT_ROOT)
    parser.add_argument("--domains", type=str, default="internal,external")
    parser.add_argument("--splits", type=str, default="train,val,test")
    parser.add_argument("--modalities", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-patients-per-split", type=int, default=0, help="0 means all patients.")
    parser.add_argument("--export-plots", action="store_true")
    parser.add_argument("--dpi", type=int, default=300)
    return parser


def main() -> None:
    args = _build_argparser().parse_args()
    cfg = _load_json(args.config)
    paths_cfg = cfg.get("paths", {})
    data_cfg = cfg.get("data", {})
    if not isinstance(paths_cfg, Mapping):
        paths_cfg = {}
    if not isinstance(data_cfg, Mapping):
        data_cfg = {}

    domains = _parse_csv_list(args.domains, allowed={"internal", "external"})
    splits = _parse_csv_list(args.splits, allowed={"train", "val", "test"})
    modalities = (
        _parse_csv_list(args.modalities)
        if args.modalities is not None
        else tuple(str(item).strip().lower() for item in data_cfg.get("modalities", DEFAULT_3D_MODALITIES))
    )
    crop_margin = _parse_int_triple(data_cfg.get("crop_margin"), DEFAULT_3D_CROP_MARGIN, name="data.crop_margin")
    intensity_norm = str(data_cfg.get("intensity_norm", "zscore"))
    require_voi = bool(data_cfg.get("require_voi", True))
    crop_with_voi = bool(data_cfg.get("crop_with_voi", True))

    internal_manifest = args.internal_manifest_csv
    if internal_manifest is None and paths_cfg.get("manifest_csv"):
        internal_manifest = Path(str(paths_cfg["manifest_csv"]))

    all_rows: List[Dict[str, Any]] = []
    all_failures: List[Dict[str, Any]] = []
    dataset_summaries: List[Dict[str, Any]] = []

    for domain in domains:
        for split in splits:
            if domain == "internal":
                if internal_manifest is None:
                    raise ValueError("Internal domain requested but no manifest CSV was provided.")
                dataset = _build_dataset(
                    domain=domain,
                    split=split,
                    manifest_csv=internal_manifest,
                    split_base_root=None,
                    modalities=modalities,
                    require_voi=require_voi,
                    crop_with_voi=crop_with_voi,
                    crop_margin=crop_margin,
                    intensity_norm=intensity_norm,
                )
            else:
                dataset = _build_dataset(
                    domain=domain,
                    split=split,
                    manifest_csv=None,
                    split_base_root=args.external_split_root,
                    modalities=modalities,
                    require_voi=require_voi,
                    crop_with_voi=crop_with_voi,
                    crop_margin=crop_margin,
                    intensity_norm=intensity_norm,
                )
            dataset_summaries.append({"domain": domain, **dataset.summary()})
            rows, failures = inspect_dataset(
                domain=domain,
                dataset=dataset,
                modalities=modalities,
                crop_margin=crop_margin,
                max_patients=int(args.max_patients_per_split),
            )
            all_rows.extend(rows)
            all_failures.extend(failures)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    patient_stats_csv = output_dir / "domain_patient_volume_stats.csv"
    summary_csv = output_dir / "domain_summary_stats.csv"
    failures_csv = output_dir / "domain_inspection_failures.csv"
    summary_json = output_dir / "domain_shift_summary.json"

    patient_fields = [
        "domain",
        "split",
        "patient_id",
        "label_name",
        "y_true",
        "modality",
        "shape_hwd",
        "shape_h",
        "shape_w",
        "shape_d",
        "spacing_h",
        "spacing_w",
        "spacing_d",
        "voi_shape_hwd",
        "voi_voxels",
        "voi_fraction",
        "bbox_h0",
        "bbox_h1",
        "bbox_w0",
        "bbox_w1",
        "bbox_d0",
        "bbox_d1",
        "crop_h",
        "crop_w",
        "crop_d",
        "crop_voxels",
        "crop_fraction",
        "raw_nonzero_mean",
        "raw_nonzero_std",
        "raw_nonzero_p1",
        "raw_nonzero_p5",
        "raw_nonzero_p50",
        "raw_nonzero_p95",
        "raw_nonzero_p99",
        "raw_crop_mean",
        "raw_crop_std",
        "raw_crop_p1",
        "raw_crop_p5",
        "raw_crop_p50",
        "raw_crop_p95",
        "raw_crop_p99",
        "norm_crop_mean",
        "norm_crop_std",
        "norm_crop_p1",
        "norm_crop_p5",
        "norm_crop_p50",
        "norm_crop_p95",
        "norm_crop_p99",
        "image_path",
        "voi_path",
    ]
    summary_rows = summarize_rows(all_rows)
    summary_fields = list(summary_rows[0].keys()) if summary_rows else ["domain", "split", "modality"]
    _write_csv(patient_stats_csv, patient_fields, all_rows)
    _write_csv(summary_csv, summary_fields, summary_rows)
    _write_csv(failures_csv, ["domain", "split", "patient_id", "label_name", "y_true", "reason", "message"], all_failures)

    output_files = {
        "patient_volume_stats": str(patient_stats_csv),
        "summary_stats": str(summary_csv),
        "failures": str(failures_csv),
        "summary_json": str(summary_json),
    }
    summary_payload: Dict[str, Any] = {
        "config": str(args.config),
        "domains": list(domains),
        "splits": list(splits),
        "modalities": list(modalities),
        "internal_manifest_csv": str(internal_manifest) if internal_manifest is not None else None,
        "external_split_root": str(args.external_split_root),
        "crop_margin_dhw": list(crop_margin),
        "intensity_norm": intensity_norm,
        "max_patients_per_split": int(args.max_patients_per_split),
        "n_patient_modality_rows": len(all_rows),
        "n_failures": len(all_failures),
        "dataset_summaries": dataset_summaries,
        "files": output_files,
        "plots_requested": bool(args.export_plots),
    }
    _write_json(summary_json, summary_payload)

    generated_figures: Dict[str, str] = {}
    if args.export_plots:
        generated_figures = export_figures(output_dir, all_rows, dpi=args.dpi)
        if generated_figures:
            summary_payload["generated_figures"] = generated_figures
            summary_payload["files"].update({f"figure_{key}": value for key, value in generated_figures.items()})
            _write_json(summary_json, summary_payload)

    print(f"[OK] Domain shift inspection exported to {output_dir}")
    print(f"[OK] Patient stats: {patient_stats_csv}")
    print(f"[OK] Summary stats: {summary_csv}")
    if all_failures:
        print(f"[WARN] Inspection failures: {len(all_failures)} ({failures_csv})")
    if generated_figures:
        print(f"[OK] Figures: {len(generated_figures)}")


if __name__ == "__main__":
    main()
