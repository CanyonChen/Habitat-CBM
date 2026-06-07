#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build ADC-only habitat masks for UCSF-PDGM 3D cases."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import nibabel as nib
import numpy as np

from ucsf_pdgm_3d_utils import read_manifest_records, str2bool, write_csv_rows

try:
    from sklearn.cluster import KMeans as SklearnKMeans
except Exception:  # pragma: no cover - dependency may be absent in syntax-only environments.
    SklearnKMeans = None  # type: ignore[assignment]

HABITAT_NAMES = ("h1", "h2", "h3", "h12", "h13", "h23", "h123")


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build UCSF ADC-only habitat masks.")
    parser.add_argument("--manifest-csv", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("/root/autodl-tmp/habitat_CBM/dataset/ucsf_pdgm_3d/habitat_masks"))
    parser.add_argument("--splits", type=str, default="train,val,test")
    parser.add_argument("--selected-habitat", type=str, default="h23", choices=HABITAT_NAMES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-init", type=int, default=20)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--kmeans-backend", type=str, default="sklearn", choices=("sklearn", "torch-cuda", "numpy", "auto"))
    parser.add_argument("--zscore-std-atol", type=float, default=0.0)
    parser.add_argument("--zscore-std-rtol", type=float, default=1e-6)
    parser.add_argument("--constant-adc-policy", type=str, default="fail", choices=("fail", "whole-tumor-selected"))
    parser.add_argument("--min-voi-voxels", type=int, default=32)
    parser.add_argument("--max-patients", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--overwrite", type=str2bool, default=False)
    return parser


def _load_nifti(path: Path) -> tuple[nib.spatialimages.SpatialImage, np.ndarray]:
    image = nib.load(str(path))
    array = np.asarray(image.get_fdata(dtype=np.float32), dtype=np.float32)
    if array.ndim != 3:
        raise ValueError(f"NIfTI must be 3D, got shape={array.shape}: {path}")
    return image, array


def _adc_stats(values: np.ndarray, std_atol: float, std_rtol: float) -> tuple[float, float, float]:
    mean = float(np.mean(values, dtype=np.float64))
    std = float(np.std(values, dtype=np.float64))
    max_abs = float(np.max(np.abs(values)))
    scale = max(abs(mean), max_abs, np.finfo(np.float32).tiny)
    min_std = max(float(std_atol), float(std_rtol) * scale)
    return mean, std, min_std


def _zscore(values: np.ndarray, std_atol: float, std_rtol: float) -> tuple[np.ndarray, float, float, float]:
    mean, std, min_std = _adc_stats(values, std_atol=std_atol, std_rtol=std_rtol)
    if not np.isfinite(mean) or not np.isfinite(std) or std <= min_std:
        raise ValueError(f"ADC values have invalid mean/std: mean={mean}, std={std}, min_std={min_std}")
    return ((values - mean) / std).astype(np.float32), mean, std, min_std


def _fallback_kmeans_1d(values_z: np.ndarray, seed: int, max_iter: int) -> tuple[np.ndarray, np.ndarray, float, int, str]:
    rng = np.random.default_rng(seed)
    if values_z.size < 3:
        raise ValueError("Need at least 3 VOI voxels for k=3.")
    centers = np.percentile(values_z, [15.0, 50.0, 85.0]).astype(np.float32)
    centers += rng.normal(0.0, 1e-4, size=centers.shape).astype(np.float32)
    labels = np.zeros(values_z.shape[0], dtype=np.int64)
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        dist = np.abs(values_z[:, None] - centers[None, :])
        labels = np.argmin(dist, axis=1).astype(np.int64)
        new_centers = centers.copy()
        for cluster_idx in range(3):
            mask = labels == cluster_idx
            if np.any(mask):
                new_centers[cluster_idx] = float(values_z[mask].mean())
            else:
                farthest_idx = int(np.argmax(np.min(dist, axis=1)))
                new_centers[cluster_idx] = float(values_z[farthest_idx])
        if float(np.max(np.abs(new_centers - centers))) <= 1e-5:
            centers = new_centers
            break
        centers = new_centers
    inertia = float(np.sum((values_z - centers[labels]) ** 2, dtype=np.float64))
    return labels, centers.astype(np.float32), inertia, n_iter, "numpy-1d"


def _torch_cuda_available() -> bool:
    try:
        import torch
    except Exception:
        return False
    return bool(torch.cuda.is_available())


def _torch_cuda_kmeans_1d(
    values_z: np.ndarray,
    seed: int,
    n_init: int,
    max_iter: int,
) -> tuple[np.ndarray, np.ndarray, float, int, str]:
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("torch-cuda K-means backend requires PyTorch.") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("torch-cuda K-means backend requested, but CUDA is not available.")
    if values_z.size < 3:
        raise ValueError("Need at least 3 VOI voxels for k=3.")

    device = torch.device("cuda")
    x = torch.as_tensor(values_z.reshape(-1, 1), dtype=torch.float32, device=device)
    x_flat = x.reshape(-1)
    init_count = max(1, int(n_init))
    max_iter = int(max_iter)

    base_centers = np.percentile(values_z, [15.0, 50.0, 85.0]).astype(np.float32)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    centers = torch.as_tensor(base_centers, dtype=torch.float32, device=device).repeat(init_count, 1)
    centers += torch.randn((init_count, 3), generator=generator, device=device) * 1e-4

    labels = torch.zeros((init_count, x.shape[0]), dtype=torch.long, device=device)
    n_iter = 0
    x_batched = x.unsqueeze(0).expand(init_count, -1, -1)
    for n_iter in range(1, max_iter + 1):
        center_view = centers.reshape(init_count, 1, 3)
        dist = x_batched.pow(2) + center_view.pow(2) - 2.0 * torch.matmul(x_batched, center_view)
        labels = torch.argmin(dist, dim=2)
        new_centers = centers.clone()
        for init_idx in range(init_count):
            init_dist = dist[init_idx]
            for cluster_idx in range(3):
                mask = labels[init_idx] == cluster_idx
                if bool(mask.any()):
                    new_centers[init_idx, cluster_idx] = x_flat[mask].mean()
                else:
                    farthest_idx = torch.argmax(torch.min(init_dist, dim=1).values)
                    new_centers[init_idx, cluster_idx] = x_flat[farthest_idx]
        if float(torch.max(torch.abs(new_centers - centers)).item()) <= 1e-5:
            centers = new_centers
            break
        centers = new_centers

    center_view = centers.reshape(init_count, 1, 3)
    final_dist = x_batched.pow(2) + center_view.pow(2) - 2.0 * torch.matmul(x_batched, center_view)
    final_labels = torch.argmin(final_dist, dim=2)
    inertias = final_dist.gather(2, final_labels.unsqueeze(2)).sum(dim=(1, 2))
    best_idx = int(torch.argmin(inertias).item())
    return (
        final_labels[best_idx].detach().cpu().numpy().astype(np.int64),
        centers[best_idx].detach().cpu().numpy().astype(np.float32),
        float(inertias[best_idx].item()),
        n_iter,
        "torch-cuda-cublas",
    )


def _run_kmeans(values_z: np.ndarray, seed: int, n_init: int, max_iter: int, backend: str) -> tuple[np.ndarray, np.ndarray, float, int, str]:
    backend = backend.strip().lower()
    if backend == "auto":
        backend = "torch-cuda" if _torch_cuda_available() else "sklearn"
    if backend == "torch-cuda":
        return _torch_cuda_kmeans_1d(values_z, seed=seed, n_init=n_init, max_iter=max_iter)
    if backend == "numpy":
        return _fallback_kmeans_1d(values_z, seed=seed, max_iter=max_iter)

    x = values_z.reshape(-1, 1)
    if SklearnKMeans is not None:
        model = SklearnKMeans(
            n_clusters=3,
            init="k-means++",
            n_init=int(n_init),
            max_iter=int(max_iter),
            random_state=int(seed),
            algorithm="lloyd",
        )
        labels = model.fit_predict(x).astype(np.int64)
        centers = np.asarray(model.cluster_centers_, dtype=np.float32).reshape(3)
        return labels, centers, float(model.inertia_), int(model.n_iter_), "sklearn"
    if backend == "sklearn":
        raise RuntimeError("sklearn K-means backend requested, but scikit-learn is not available.")
    return _fallback_kmeans_1d(values_z, seed=seed, max_iter=max_iter)


def _map_labels_by_adc_center(labels_raw: np.ndarray, adc_values: np.ndarray) -> tuple[np.ndarray, Dict[str, float], Dict[str, int]]:
    centers_raw: Dict[int, float] = {}
    counts_raw: Dict[int, int] = {}
    for cluster_idx in range(3):
        mask = labels_raw == cluster_idx
        if not np.any(mask):
            raise ValueError(f"K-means cluster {cluster_idx} is empty.")
        centers_raw[cluster_idx] = float(adc_values[mask].mean(dtype=np.float64))
        counts_raw[cluster_idx] = int(mask.sum())

    ordered = sorted(centers_raw, key=lambda idx: centers_raw[idx])
    raw_to_habitat = {ordered[0]: 1, ordered[1]: 2, ordered[2]: 3}
    labels_mapped = np.zeros(labels_raw.shape, dtype=np.uint8)
    for raw_label, habitat_label in raw_to_habitat.items():
        labels_mapped[labels_raw == raw_label] = np.uint8(habitat_label)
    centers_mapped = {
        "h1_adc_center": centers_raw[ordered[0]],
        "h2_adc_center": centers_raw[ordered[1]],
        "h3_adc_center": centers_raw[ordered[2]],
    }
    counts_mapped = {
        "h1_voxels": counts_raw[ordered[0]],
        "h2_voxels": counts_raw[ordered[1]],
        "h3_voxels": counts_raw[ordered[2]],
    }
    return labels_mapped, centers_mapped, counts_mapped


def _make_masks(label_volume: np.ndarray, voi: np.ndarray) -> Dict[str, np.ndarray]:
    h1 = label_volume == 1
    h2 = label_volume == 2
    h3 = label_volume == 3
    return {
        "h1": h1,
        "h2": h2,
        "h3": h3,
        "h12": h1 | h2,
        "h13": h1 | h3,
        "h23": h2 | h3,
        "h123": np.asarray(voi) > 0,
    }


def _save_mask(mask: np.ndarray, reference_img: nib.spatialimages.SpatialImage, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = reference_img.header.copy()
    header.set_data_dtype(np.uint8)
    image = nib.Nifti1Image(mask.astype(np.uint8), affine=reference_img.affine, header=header)
    nib.save(image, str(path))


def _process_record(
    record,
    output_root: Path,
    selected_habitat: str,
    seed: int,
    n_init: int,
    max_iter: int,
    kmeans_backend: str,
    zscore_std_atol: float,
    zscore_std_rtol: float,
    constant_adc_policy: str,
    min_voi_voxels: int,
    overwrite: bool,
) -> Dict[str, object]:
    patient_out = output_root / record.split / record.patient_id
    if patient_out.exists():
        if not overwrite:
            raise FileExistsError(f"Patient habitat output already exists: {patient_out}")
        shutil.rmtree(patient_out)

    adc_img, adc = _load_nifti(record.paths["adc"])
    _, tumor_seg = _load_nifti(record.paths["tumor_seg"])
    if tuple(adc.shape) != tuple(tumor_seg.shape):
        raise ValueError(
            f"ADC/tumor segmentation shape mismatch for {record.patient_id}: "
            f"adc={adc.shape}, seg={tumor_seg.shape}"
        )
    voi = np.asarray(tumor_seg > 0)
    voi_count = int(voi.sum())
    if voi_count < min_voi_voxels:
        raise ValueError(f"VOI too small for {record.patient_id}: {voi_count} voxels.")

    adc_values = adc[voi]
    finite_mask = np.isfinite(adc_values)
    if int(finite_mask.sum()) < min_voi_voxels:
        raise ValueError(
            f"ADC has too few finite VOI voxels for {record.patient_id}: "
            f"{int(finite_mask.sum())}."
        )
    adc_values_finite = adc_values[finite_mask].astype(np.float32, copy=False)
    adc_mean, adc_std, zscore_min_std = _adc_stats(
        adc_values_finite,
        std_atol=zscore_std_atol,
        std_rtol=zscore_std_rtol,
    )
    habitat_build_note = ""
    if np.isfinite(adc_mean) and np.isfinite(adc_std) and adc_std <= zscore_min_std and constant_adc_policy == "whole-tumor-selected":
        label_volume = np.zeros(adc.shape, dtype=np.uint8)
        label_volume[voi] = 2
        centers_z = np.zeros(3, dtype=np.float32)
        inertia = 0.0
        n_iter = 0
        backend = "constant-adc-whole-tumor-fallback"
        centers_mapped = {
            "h1_adc_center": adc_mean,
            "h2_adc_center": adc_mean,
            "h3_adc_center": adc_mean,
        }
        counts_mapped = {
            "h1_voxels": 0,
            "h2_voxels": voi_count,
            "h3_voxels": 0,
        }
        habitat_build_note = "constant_adc_in_voi; h2=whole_tumor; h23=whole_tumor"
    else:
        values_z, adc_mean, adc_std, zscore_min_std = _zscore(
            adc_values_finite,
            std_atol=zscore_std_atol,
            std_rtol=zscore_std_rtol,
        )
        labels_raw, centers_z, inertia, n_iter, backend = _run_kmeans(
            values_z,
            seed=seed,
            n_init=n_init,
            max_iter=max_iter,
            backend=kmeans_backend,
        )
        labels_mapped, centers_mapped, counts_mapped = _map_labels_by_adc_center(labels_raw, adc_values_finite)

        flat_label_volume = np.zeros(voi_count, dtype=np.uint8)
        flat_label_volume[finite_mask] = labels_mapped
        label_volume = np.zeros(adc.shape, dtype=np.uint8)
        label_volume[voi] = flat_label_volume
    masks = _make_masks(label_volume, voi)
    for name, mask in masks.items():
        _save_mask(mask, adc_img, patient_out / f"{name}.nii.gz")

    selected_count = int(masks[selected_habitat].sum())
    row: Dict[str, object] = {
        "patient_id": record.patient_id,
        "nifti_stem": record.nifti_stem,
        "split": record.split,
        "y_true": record.y_true,
        "label_name": record.label_name,
        "selected_habitat": selected_habitat,
        "selected_habitat_path": str(patient_out / f"{selected_habitat}.nii.gz"),
        "voi_voxels": voi_count,
        "selected_voxels": selected_count,
        "selected_volume_ratio": float(selected_count / voi_count),
        "adc_mean_in_voi": adc_mean,
        "adc_std_in_voi": adc_std,
        "zscore_min_std": zscore_min_std,
        "kmeans_backend": backend,
        "kmeans_inertia": inertia,
        "kmeans_iterations": n_iter,
        "kmeans_centers_z": json.dumps([float(item) for item in centers_z.tolist()]),
        "habitat_build_note": habitat_build_note,
        **centers_mapped,
        **counts_mapped,
    }
    return row


def _process_record_worker(args: tuple[object, Path, str, int, int, int, str, float, float, str, int, bool]) -> Dict[str, object]:
    (
        record,
        output_root,
        selected_habitat,
        seed,
        n_init,
        max_iter,
        kmeans_backend,
        zscore_std_atol,
        zscore_std_rtol,
        constant_adc_policy,
        min_voi_voxels,
        overwrite,
    ) = args
    return _process_record(
        record=record,
        output_root=output_root,
        selected_habitat=selected_habitat,
        seed=seed,
        n_init=n_init,
        max_iter=max_iter,
        kmeans_backend=kmeans_backend,
        zscore_std_atol=zscore_std_atol,
        zscore_std_rtol=zscore_std_rtol,
        constant_adc_policy=constant_adc_policy,
        min_voi_voxels=min_voi_voxels,
        overwrite=overwrite,
    )


def main() -> None:
    args = _build_argparser().parse_args()
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
        raise ValueError("No manifest records selected for habitat building.")

    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    qc_rows: List[Dict[str, object]] = []
    failure_rows: List[Dict[str, object]] = []
    worker_args = [
        (
            record,
            output_root,
            selected_habitat,
            int(args.seed),
            int(args.n_init),
            int(args.max_iter),
            str(args.kmeans_backend),
            float(args.zscore_std_atol),
            float(args.zscore_std_rtol),
            str(args.constant_adc_policy),
            int(args.min_voi_voxels),
            bool(args.overwrite),
        )
        for record in records
    ]
    num_workers = max(1, int(args.num_workers))
    if num_workers == 1:
        for item in worker_args:
            record = item[0]
            try:
                qc_rows.append(_process_record_worker(item))
            except Exception as exc:
                failure_rows.append(
                    {
                        "patient_id": record.patient_id,
                        "split": record.split,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_to_record = {executor.submit(_process_record_worker, item): item[0] for item in worker_args}
            for completed, future in enumerate(as_completed(future_to_record), start=1):
                record = future_to_record[future]
                try:
                    qc_rows.append(future.result())
                except Exception as exc:
                    failure_rows.append(
                        {
                            "patient_id": record.patient_id,
                            "split": record.split,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    )
                if completed % 25 == 0 or completed == len(future_to_record):
                    print(
                        f"[progress] completed={completed}/{len(future_to_record)} "
                        f"processed={len(qc_rows)} failed={len(failure_rows)}",
                        flush=True,
                    )

    qc_rows = sorted(qc_rows, key=lambda item: (str(item["split"]), str(item["patient_id"])))
    failure_rows = sorted(
        failure_rows,
        key=lambda item: (str(item["split"]), str(item["patient_id"]), str(item["error_type"])),
    )

    qc_path = output_root / "habitat_qc_ucsf_pdgm_3d.csv"
    fail_path = output_root / "habitat_failures_ucsf_pdgm_3d.csv"
    qc_fields = (
        "patient_id",
        "nifti_stem",
        "split",
        "y_true",
        "label_name",
        "selected_habitat",
        "selected_habitat_path",
        "voi_voxels",
        "selected_voxels",
        "selected_volume_ratio",
        "adc_mean_in_voi",
        "adc_std_in_voi",
        "zscore_min_std",
        "kmeans_backend",
        "kmeans_inertia",
        "kmeans_iterations",
        "kmeans_centers_z",
        "habitat_build_note",
        "h1_adc_center",
        "h2_adc_center",
        "h3_adc_center",
        "h1_voxels",
        "h2_voxels",
        "h3_voxels",
    )
    write_csv_rows(qc_path, qc_fields, qc_rows)
    write_csv_rows(fail_path, ("patient_id", "split", "error_type", "error"), failure_rows)
    summary_path = output_root / "habitat_summary_ucsf_pdgm_3d.json"
    summary_path.write_text(
        json.dumps(
            {
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "manifest_csv": str(args.manifest_csv),
                "output_root": str(output_root),
                "selected_habitat": selected_habitat,
                "kmeans_backend": str(args.kmeans_backend),
                "constant_adc_policy": str(args.constant_adc_policy),
                "processed": len(qc_rows),
                "failed": len(failure_rows),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print("UCSF ADC-only habitat masks generated:")
    print(f"  output_root: {output_root}")
    print(f"  qc         : {qc_path}")
    print(f"  failures   : {fail_path}")
    print(f"  processed  : {len(qc_rows)}")
    print(f"  failed     : {len(failure_rows)}")


if __name__ == "__main__":
    main()
