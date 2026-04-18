#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Habitat-CBM 概念层评估脚本。

输入：
- patient_concepts_habitat_cbm_runXX.csv

输出：
- concept_metrics_by_concept.csv
- concept_metrics_overall.csv
- concept_error_distribution.csv
- concept_error_rank.csv
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

DEFAULT_CONCEPT_IDS = tuple(f"c{i}" for i in range(1, 9))


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return [dict(row) for row in reader]


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _safe_float(text: str, *, field: str, patient_id: str) -> float:
    value_text = str(text).strip()
    if value_text == "":
        raise ValueError(f"Empty value for {field} (patient={patient_id})")
    try:
        value = float(value_text)
    except ValueError as exc:
        raise ValueError(
            f"Invalid numeric value for {field} (patient={patient_id}): {value_text}"
        ) from exc
    if not np.isfinite(value):
        raise ValueError(
            f"Non-finite numeric value for {field} (patient={patient_id}): {value}"
        )
    return value


def _pearson_with_reason(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, str]:
    if y_true.size < 2:
        return float("nan"), "insufficient_samples"
    std_true = float(y_true.std(ddof=0))
    std_pred = float(y_pred.std(ddof=0))
    if std_true <= 1e-12:
        return float("nan"), "zero_variance_true"
    if std_pred <= 1e-12:
        return float("nan"), "zero_variance_pred"
    corr = float(np.corrcoef(y_true, y_pred)[0, 1])
    return corr, "ok"


def _r2_with_reason(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, str]:
    if y_true.size < 2:
        return float("nan"), "insufficient_samples"
    y_mean = float(np.mean(y_true))
    ss_tot = float(np.sum((y_true - y_mean) ** 2))
    if ss_tot <= 1e-12:
        return float("nan"), "zero_variance_true"
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    return 1.0 - ss_res / ss_tot, "ok"


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, object]:
    abs_error = np.abs(y_pred - y_true)
    sq_error = (y_pred - y_true) ** 2

    mae = float(np.mean(abs_error))
    rmse = float(math.sqrt(float(np.mean(sq_error))))
    r2, r2_reason = _r2_with_reason(y_true, y_pred)
    pearson, pearson_reason = _pearson_with_reason(y_true, y_pred)

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "r2_reason": r2_reason,
        "pearson_r": pearson,
        "pearson_reason": pearson_reason,
    }


def evaluate_concepts(
    patient_concepts_csv: Path,
    output_dir: Path,
    split_filter: str,
    use_scale: str,
) -> None:
    rows = _read_csv(patient_concepts_csv)
    if not rows:
        raise ValueError("patient_concepts CSV is empty.")

    if split_filter != "all":
        rows = [row for row in rows if str(row.get("split", "")).strip().lower() == split_filter]
        if not rows:
            raise ValueError(f"No rows found for split='{split_filter}'")

    if use_scale not in {"raw", "std"}:
        raise ValueError(f"Unsupported scale: {use_scale}")

    # 自动发现概念编号（优先 c1..c8）
    concept_ids = [cid for cid in DEFAULT_CONCEPT_IDS if f"{cid}_true_{use_scale}" in rows[0]]
    if not concept_ids:
        raise ValueError(
            f"No concept columns found for scale='{use_scale}'. "
            f"Expected columns like c1_true_{use_scale}, c1_pred_{use_scale}."
        )

    metrics_rows: List[Dict[str, object]] = []
    distribution_rows: List[Dict[str, object]] = []

    for concept_id in concept_ids:
        true_col = f"{concept_id}_true_{use_scale}"
        pred_col = f"{concept_id}_pred_{use_scale}"
        abs_col = f"{concept_id}_abs_error_{use_scale}"

        y_true = np.asarray(
            [_safe_float(row[true_col], field=true_col, patient_id=str(row.get("patient_id", ""))) for row in rows],
            dtype=np.float64,
        )
        y_pred = np.asarray(
            [_safe_float(row[pred_col], field=pred_col, patient_id=str(row.get("patient_id", ""))) for row in rows],
            dtype=np.float64,
        )

        metrics = _compute_metrics(y_true, y_pred)
        metrics_rows.append(
            {
                "concept_id": concept_id,
                "n_patients": int(y_true.size),
                **metrics,
            }
        )

        for row in rows:
            patient_id = str(row.get("patient_id", "")).strip()
            y_t = _safe_float(row[true_col], field=true_col, patient_id=patient_id)
            y_p = _safe_float(row[pred_col], field=pred_col, patient_id=patient_id)
            if abs_col in row and str(row[abs_col]).strip() != "":
                abs_err = _safe_float(row[abs_col], field=abs_col, patient_id=patient_id)
            else:
                abs_err = float(abs(y_p - y_t))
            distribution_rows.append(
                {
                    "patient_id": patient_id,
                    "split": str(row.get("split", "")).strip(),
                    "y_true": str(row.get("y_true", "")).strip(),
                    "concept_id": concept_id,
                    "true_value": y_t,
                    "pred_value": y_p,
                    "abs_error": abs_err,
                    "run_id": str(row.get("run_id", "")).strip(),
                    "checkpoint_name": str(row.get("checkpoint_name", "")).strip(),
                }
            )

    overall = {
        "n_concepts": int(len(metrics_rows)),
        "n_patients": int(len(rows)),
        "scale": use_scale,
        "split": split_filter,
        "mae_mean": float(np.nanmean([float(row["mae"]) for row in metrics_rows])),
        "rmse_mean": float(np.nanmean([float(row["rmse"]) for row in metrics_rows])),
        "r2_mean": float(np.nanmean([float(row["r2"]) for row in metrics_rows])),
        "pearson_r_mean": float(np.nanmean([float(row["pearson_r"]) for row in metrics_rows])),
    }

    # 偏差排序：按平均绝对误差从大到小
    rank_rows: List[Dict[str, object]] = []
    for concept_id in concept_ids:
        concept_errors = [
            float(item["abs_error"]) for item in distribution_rows if item["concept_id"] == concept_id
        ]
        concept_errors_np = np.asarray(concept_errors, dtype=np.float64)
        rank_rows.append(
            {
                "concept_id": concept_id,
                "mae": float(np.mean(concept_errors_np)),
                "median_abs_error": float(np.median(concept_errors_np)),
                "iqr_abs_error": float(
                    np.percentile(concept_errors_np, 75.0) - np.percentile(concept_errors_np, 25.0)
                ),
            }
        )
    rank_rows.sort(key=lambda row: float(row["mae"]), reverse=True)
    for idx, row in enumerate(rank_rows, start=1):
        row["rank"] = idx

    suffix = f"{split_filter}_{use_scale}"
    _write_csv(
        output_dir / f"concept_metrics_by_concept_{suffix}.csv",
        [
            "concept_id",
            "n_patients",
            "mae",
            "rmse",
            "r2",
            "r2_reason",
            "pearson_r",
            "pearson_reason",
        ],
        metrics_rows,
    )
    _write_csv(
        output_dir / f"concept_metrics_overall_{suffix}.csv",
        list(overall.keys()),
        [overall],
    )
    _write_csv(
        output_dir / f"concept_error_distribution_{suffix}.csv",
        [
            "patient_id",
            "split",
            "y_true",
            "concept_id",
            "true_value",
            "pred_value",
            "abs_error",
            "run_id",
            "checkpoint_name",
        ],
        distribution_rows,
    )
    _write_csv(
        output_dir / f"concept_error_rank_{suffix}.csv",
        ["concept_id", "mae", "median_abs_error", "iqr_abs_error", "rank"],
        rank_rows,
    )

    # 同步写入无后缀别名（便于论文目录固定引用）
    _write_csv(
        output_dir / "concept_metrics_by_concept.csv",
        [
            "concept_id",
            "n_patients",
            "mae",
            "rmse",
            "r2",
            "r2_reason",
            "pearson_r",
            "pearson_reason",
        ],
        metrics_rows,
    )
    _write_csv(output_dir / "concept_metrics_overall.csv", list(overall.keys()), [overall])
    _write_csv(
        output_dir / "concept_error_distribution.csv",
        [
            "patient_id",
            "split",
            "y_true",
            "concept_id",
            "true_value",
            "pred_value",
            "abs_error",
            "run_id",
            "checkpoint_name",
        ],
        distribution_rows,
    )
    _write_csv(
        output_dir / "concept_error_rank.csv",
        ["concept_id", "mae", "median_abs_error", "iqr_abs_error", "rank"],
        rank_rows,
    )


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Habitat-CBM concept predictions.")
    parser.add_argument("--patient-concepts-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", type=str, default="test", choices=("train", "val", "test", "all"))
    parser.add_argument("--scale", type=str, default="raw", choices=("raw", "std"))
    return parser


def main() -> None:
    args = _build_argparser().parse_args()
    evaluate_concepts(
        patient_concepts_csv=args.patient_concepts_csv,
        output_dir=args.output_dir,
        split_filter=args.split,
        use_scale=args.scale,
    )
    print("Concept evaluation exported:")
    print(f"  output_dir: {args.output_dir}")
    print(f"  split     : {args.split}")
    print(f"  scale     : {args.scale}")


if __name__ == "__main__":
    main()
