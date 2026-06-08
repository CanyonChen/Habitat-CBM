#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prediction-level diagnostics for Habitat-CBM 3D runs.

The script reuses exported patient prediction CSVs. It writes CSV/JSON/Markdown
diagnostics first, then optionally exports figures.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


MODEL_NAME = "habitat_cbm_3d"
EXTERNAL_MODEL_NAME = "habitat_cbm_3d_external_splited_data"
METRIC_KEYS = ("auc", "acc", "balanced_acc", "sen", "spe", "ppv", "npv", "f1")


@dataclass(frozen=True)
class PredictionTable:
    domain: str
    split: str
    patient_ids: List[str]
    y_true: np.ndarray
    y_prob: np.ndarray
    y_pred_exported: Optional[np.ndarray]
    source_csv: Path


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"CSV not found: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return [dict(row) for row in reader]


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


def _safe_float(value: str, *, column: str, patient_id: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {column} for patient {patient_id}: {value!r}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"Non-finite {column} for patient {patient_id}: {parsed}")
    return parsed


def _safe_int(value: str, *, column: str, patient_id: str) -> int:
    try:
        parsed = int(float(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {column} for patient {patient_id}: {value!r}") from exc
    if parsed not in (0, 1):
        raise ValueError(f"{column} must be 0/1 for patient {patient_id}, got {parsed}")
    return parsed


def _load_json_if_exists(path: Optional[Path]) -> Dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON must be an object: {path}")
    return payload


def _infer_run_id(run_dir: Optional[Path], run_id: Optional[str]) -> str:
    if run_id:
        return run_id
    if run_dir is not None:
        return run_dir.name
    raise ValueError("Pass --run-id or --run-dir.")


def _first_existing(patterns: Sequence[Path]) -> Optional[Path]:
    for pattern in patterns:
        if any(char in str(pattern) for char in "*?[]"):
            matches = sorted(pattern.parent.glob(pattern.name))
            if matches:
                return matches[0]
        elif pattern.is_file():
            return pattern
    return None


def _resolve_default_paths(args: argparse.Namespace, run_id: str) -> Dict[str, Optional[Path]]:
    run_dir = args.run_dir
    internal_predictions = args.internal_predictions_csv
    external_predictions = args.external_predictions_csv
    run_summary = args.run_summary_json
    external_summary = args.external_summary_json

    if run_dir is not None:
        internal_predictions = internal_predictions or _first_existing(
            [
                run_dir / f"patient_predictions_{MODEL_NAME}_{run_id}.csv",
                run_dir / f"patient_predictions_{MODEL_NAME}_*.csv",
            ]
        )
        run_summary = run_summary or _first_existing(
            [
                run_dir / f"run_train_summary_{MODEL_NAME}_{run_id}.json",
                run_dir / f"run_train_summary_{MODEL_NAME}_*.json",
            ]
        )
        external_dir = args.external_dir or run_dir / "external_splited_data"
        external_predictions = external_predictions or _first_existing(
            [
                external_dir / f"patient_predictions_{EXTERNAL_MODEL_NAME}_{run_id}.csv",
                external_dir / f"patient_predictions_{EXTERNAL_MODEL_NAME}_*.csv",
            ]
        )
        external_summary = external_summary or _first_existing(
            [
                external_dir / f"run_summary_{EXTERNAL_MODEL_NAME}_{run_id}.json",
                external_dir / f"run_summary_{EXTERNAL_MODEL_NAME}_*.json",
            ]
        )

    return {
        "internal_predictions": internal_predictions,
        "external_predictions": external_predictions,
        "run_summary": run_summary,
        "external_summary": external_summary,
    }


def _normalize_split_value(value: object) -> str:
    text = str(value).strip().lower()
    return text if text else "unknown"


def load_prediction_tables(path: Path, *, domain: str, include_overall: bool = True) -> List[PredictionTable]:
    rows = _read_csv(path)
    required = ("patient_id", "y_true", "prob_idh_mut")
    if rows:
        missing = [column for column in required if column not in rows[0]]
        if missing:
            raise ValueError(f"Prediction CSV missing columns {missing}: {path}")
    split_values = sorted({_normalize_split_value(row.get("split", "all")) for row in rows})
    tables: List[PredictionTable] = []
    for split in split_values:
        split_rows = [row for row in rows if _normalize_split_value(row.get("split", "all")) == split]
        tables.append(_build_table(path, domain=domain, split=split, rows=split_rows))
    if include_overall and len(split_values) > 1:
        tables.append(_build_table(path, domain=domain, split="overall", rows=rows))
    return tables


def _build_table(path: Path, *, domain: str, split: str, rows: Sequence[Mapping[str, str]]) -> PredictionTable:
    patient_ids: List[str] = []
    y_true: List[int] = []
    y_prob: List[float] = []
    y_pred: List[int] = []
    has_exported_pred = bool(rows and "pred_label" in rows[0])
    seen = set()
    for row in rows:
        patient_id = str(row.get("patient_id", "")).strip()
        if not patient_id:
            raise ValueError(f"Empty patient_id in {path}")
        unique_key = (split, patient_id)
        if unique_key in seen:
            raise ValueError(f"Duplicate patient_id for {domain}/{split}: {patient_id}")
        seen.add(unique_key)
        patient_ids.append(patient_id)
        y_true.append(_safe_int(str(row["y_true"]), column="y_true", patient_id=patient_id))
        y_prob.append(_safe_float(str(row["prob_idh_mut"]), column="prob_idh_mut", patient_id=patient_id))
        if has_exported_pred:
            y_pred.append(_safe_int(str(row["pred_label"]), column="pred_label", patient_id=patient_id))
    return PredictionTable(
        domain=domain,
        split=split,
        patient_ids=patient_ids,
        y_true=np.asarray(y_true, dtype=np.int64),
        y_prob=np.asarray(y_prob, dtype=np.float64),
        y_pred_exported=np.asarray(y_pred, dtype=np.int64) if has_exported_pred else None,
        source_csv=path,
    )


def _safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def compute_metrics_at_threshold(table: PredictionTable, threshold: float) -> Dict[str, Any]:
    y_pred = (table.y_prob >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(table.y_true, y_pred, labels=[0, 1]).ravel()
    n = int(table.y_true.size)
    n_pos = int((table.y_true == 1).sum())
    n_neg = int((table.y_true == 0).sum())
    sen = float(recall_score(table.y_true, y_pred, pos_label=1, zero_division=0))
    spe = float(tn / (tn + fp)) if (tn + fp) else float("nan")
    ppv = float(precision_score(table.y_true, y_pred, zero_division=0))
    npv = float(tn / (tn + fn)) if (tn + fn) else float("nan")
    return {
        "domain": table.domain,
        "split": table.split,
        "threshold": float(threshold),
        "n": n,
        "n_positive": n_pos,
        "n_negative": n_neg,
        "prevalence": float(n_pos / n) if n else float("nan"),
        "pred_positive": int(y_pred.sum()),
        "pred_positive_rate": float(y_pred.mean()) if n else float("nan"),
        "auc": _safe_auc(table.y_true, table.y_prob),
        "acc": float(accuracy_score(table.y_true, y_pred)) if n else float("nan"),
        "balanced_acc": float(balanced_accuracy_score(table.y_true, y_pred)) if n else float("nan"),
        "sen": sen,
        "spe": spe,
        "ppv": ppv,
        "npv": npv,
        "f1": float(f1_score(table.y_true, y_pred, zero_division=0)) if n else float("nan"),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def youden_threshold(table: PredictionTable) -> Dict[str, Any]:
    if np.unique(table.y_true).size < 2:
        return {
            "domain": table.domain,
            "split": table.split,
            "threshold": float("nan"),
            "youden_j": float("nan"),
            "reason": "single_class",
        }
    fpr, tpr, thresholds = roc_curve(table.y_true, table.y_prob)
    valid = np.isfinite(thresholds)
    if not np.any(valid):
        return {
            "domain": table.domain,
            "split": table.split,
            "threshold": float("nan"),
            "youden_j": float("nan"),
            "reason": "no_finite_threshold",
        }
    fpr = fpr[valid]
    tpr = tpr[valid]
    thresholds = thresholds[valid]
    j_values = tpr - fpr
    idx = int(np.argmax(j_values))
    threshold = float(np.clip(thresholds[idx], 1e-6, 1.0 - 1e-6))
    return {
        "domain": table.domain,
        "split": table.split,
        "threshold": threshold,
        "youden_j": float(j_values[idx]),
        "reason": "ok",
    }


def _formal_threshold_for_table(
    table: PredictionTable,
    internal_threshold: float,
    external_threshold: Optional[float],
) -> float:
    if table.domain == "external" and external_threshold is not None:
        return float(external_threshold)
    return float(internal_threshold)


def build_threshold_rows(
    tables: Sequence[PredictionTable],
    *,
    internal_threshold: float,
    external_threshold: Optional[float],
    grid_size: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    sweep_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    base_grid = np.linspace(0.0, 1.0, int(grid_size))
    for table in tables:
        formal_threshold = _formal_threshold_for_table(table, internal_threshold, external_threshold)
        youden = youden_threshold(table)
        thresholds = set(float(v) for v in base_grid)
        thresholds.add(float(formal_threshold))
        thresholds.add(0.5)
        if math.isfinite(float(youden["threshold"])):
            thresholds.add(float(youden["threshold"]))
        for threshold in sorted(thresholds):
            row = compute_metrics_at_threshold(table, threshold)
            row["threshold_role"] = "grid"
            if abs(float(threshold) - formal_threshold) <= 1e-12:
                row["threshold_role"] = "formal"
            if math.isfinite(float(youden["threshold"])) and abs(float(threshold) - float(youden["threshold"])) <= 1e-12:
                row["threshold_role"] = (
                    "formal+oracle_youden" if row["threshold_role"] == "formal" else "oracle_youden"
                )
            sweep_rows.append(row)

        formal_metrics = compute_metrics_at_threshold(table, formal_threshold)
        oracle_metrics = (
            compute_metrics_at_threshold(table, float(youden["threshold"]))
            if math.isfinite(float(youden["threshold"]))
            else {}
        )
        summary_rows.append(
            {
                "domain": table.domain,
                "split": table.split,
                "source_csv": str(table.source_csv),
                "n": int(table.y_true.size),
                "n_positive": int((table.y_true == 1).sum()),
                "n_negative": int((table.y_true == 0).sum()),
                "prevalence": float(np.mean(table.y_true)) if table.y_true.size else float("nan"),
                "prob_mean": float(np.mean(table.y_prob)) if table.y_prob.size else float("nan"),
                "prob_std": float(np.std(table.y_prob)) if table.y_prob.size else float("nan"),
                "prob_min": float(np.min(table.y_prob)) if table.y_prob.size else float("nan"),
                "prob_p25": float(np.percentile(table.y_prob, 25)) if table.y_prob.size else float("nan"),
                "prob_p50": float(np.percentile(table.y_prob, 50)) if table.y_prob.size else float("nan"),
                "prob_p75": float(np.percentile(table.y_prob, 75)) if table.y_prob.size else float("nan"),
                "prob_max": float(np.max(table.y_prob)) if table.y_prob.size else float("nan"),
                "formal_threshold": formal_threshold,
                "formal_auc": formal_metrics["auc"],
                "formal_acc": formal_metrics["acc"],
                "formal_sen": formal_metrics["sen"],
                "formal_spe": formal_metrics["spe"],
                "formal_f1": formal_metrics["f1"],
                "formal_tn": formal_metrics["tn"],
                "formal_fp": formal_metrics["fp"],
                "formal_fn": formal_metrics["fn"],
                "formal_tp": formal_metrics["tp"],
                "formal_pred_positive_rate": formal_metrics["pred_positive_rate"],
                "oracle_youden_threshold": youden["threshold"],
                "oracle_youden_j": youden["youden_j"],
                "oracle_auc": oracle_metrics.get("auc", float("nan")),
                "oracle_acc": oracle_metrics.get("acc", float("nan")),
                "oracle_sen": oracle_metrics.get("sen", float("nan")),
                "oracle_spe": oracle_metrics.get("spe", float("nan")),
                "oracle_f1": oracle_metrics.get("f1", float("nan")),
                "oracle_tn": oracle_metrics.get("tn", ""),
                "oracle_fp": oracle_metrics.get("fp", ""),
                "oracle_fn": oracle_metrics.get("fn", ""),
                "oracle_tp": oracle_metrics.get("tp", ""),
                "oracle_pred_positive_rate": oracle_metrics.get("pred_positive_rate", float("nan")),
            }
        )
    return sweep_rows, summary_rows


def build_calibration_rows(tables: Sequence[PredictionTable], *, n_bins: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    for table in tables:
        for idx in range(int(n_bins)):
            low = float(edges[idx])
            high = float(edges[idx + 1])
            if idx == int(n_bins) - 1:
                mask = (table.y_prob >= low) & (table.y_prob <= high)
            else:
                mask = (table.y_prob >= low) & (table.y_prob < high)
            count = int(mask.sum())
            positives = int(table.y_true[mask].sum()) if count else 0
            rows.append(
                {
                    "domain": table.domain,
                    "split": table.split,
                    "bin_idx": idx,
                    "prob_low": low,
                    "prob_high": high,
                    "count": count,
                    "n_positive": positives,
                    "n_negative": count - positives,
                    "mean_pred_prob": float(np.mean(table.y_prob[mask])) if count else float("nan"),
                    "fraction_positive": float(positives / count) if count else float("nan"),
                }
            )
        if np.unique(table.y_true).size >= 2:
            prob_true, prob_pred = calibration_curve(table.y_true, table.y_prob, n_bins=int(n_bins), strategy="uniform")
            for idx, (x_value, y_value) in enumerate(zip(prob_pred, prob_true)):
                rows.append(
                    {
                        "domain": table.domain,
                        "split": table.split,
                        "bin_idx": f"sklearn_{idx}",
                        "prob_low": "",
                        "prob_high": "",
                        "count": "",
                        "n_positive": "",
                        "n_negative": "",
                        "mean_pred_prob": float(x_value),
                        "fraction_positive": float(y_value),
                    }
                )
    return rows


def build_probability_bin_rows(tables: Sequence[PredictionTable], *, n_bins: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    for table in tables:
        for idx in range(int(n_bins)):
            low = float(edges[idx])
            high = float(edges[idx + 1])
            if idx == int(n_bins) - 1:
                bin_mask = (table.y_prob >= low) & (table.y_prob <= high)
            else:
                bin_mask = (table.y_prob >= low) & (table.y_prob < high)
            for label in (0, 1):
                mask = bin_mask & (table.y_true == label)
                rows.append(
                    {
                        "domain": table.domain,
                        "split": table.split,
                        "bin_idx": idx,
                        "prob_low": low,
                        "prob_high": high,
                        "y_true": label,
                        "count": int(mask.sum()),
                    }
                )
    return rows


def build_error_rows(
    tables: Sequence[PredictionTable],
    *,
    internal_threshold: float,
    external_threshold: Optional[float],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for table in tables:
        threshold = _formal_threshold_for_table(table, internal_threshold, external_threshold)
        y_pred = (table.y_prob >= threshold).astype(np.int64)
        for patient_id, y_true, y_prob, pred in zip(table.patient_ids, table.y_true, table.y_prob, y_pred):
            if int(y_true) == 0 and int(pred) == 0:
                error_type = "TN"
            elif int(y_true) == 0 and int(pred) == 1:
                error_type = "FP"
            elif int(y_true) == 1 and int(pred) == 0:
                error_type = "FN"
            else:
                error_type = "TP"
            rows.append(
                {
                    "domain": table.domain,
                    "split": table.split,
                    "patient_id": patient_id,
                    "y_true": int(y_true),
                    "prob_idh_mut": float(y_prob),
                    "threshold": threshold,
                    "pred_label": int(pred),
                    "error_type": error_type,
                    "is_error": int(error_type in {"FP", "FN"}),
                    "signed_margin_to_threshold": float(y_prob - threshold),
                    "abs_margin_to_threshold": float(abs(y_prob - threshold)),
                }
            )
    rows.sort(key=lambda row: (str(row["domain"]), str(row["split"]), int(row["is_error"]) * -1, float(row["abs_margin_to_threshold"])))
    return rows


def collect_concept_metrics(run_dir: Optional[Path]) -> List[Dict[str, Any]]:
    if run_dir is None:
        return []
    root = run_dir / "concept_validation"
    if not root.is_dir():
        return []
    rows: List[Dict[str, Any]] = []
    for metrics_path in sorted(root.glob("*/*/concept_metrics_by_concept.csv")):
        parts = metrics_path.relative_to(root).parts
        split = parts[0] if len(parts) >= 3 else ""
        scale = parts[1] if len(parts) >= 3 else ""
        for row in _read_csv(metrics_path):
            rows.append(
                {
                    "split": split,
                    "scale": scale,
                    "concept_id": row.get("concept_id", ""),
                    "n_patients": row.get("n_patients", ""),
                    "mae": row.get("mae", ""),
                    "rmse": row.get("rmse", ""),
                    "r2": row.get("r2", ""),
                    "pearson_r": row.get("pearson_r", ""),
                    "source_csv": str(metrics_path),
                }
            )
    return rows


def _threshold_from_run_summary(payload: Mapping[str, Any]) -> Optional[float]:
    threshold_payload = payload.get("threshold", {})
    if isinstance(threshold_payload, Mapping):
        for key in ("eval_threshold_used", "youden_threshold", "fixed_threshold"):
            value = threshold_payload.get(key)
            if value is not None:
                return float(value)
    return None


def _threshold_from_external_summary(payload: Mapping[str, Any]) -> Optional[float]:
    value = payload.get("threshold")
    return float(value) if value is not None else None


def _format_metric(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not math.isfinite(numeric):
        return "NA"
    return f"{numeric:.4f}"


def write_markdown_report(
    path: Path,
    *,
    run_id: str,
    threshold_summary_rows: Sequence[Mapping[str, Any]],
    concept_rows: Sequence[Mapping[str, Any]],
    run_summary: Mapping[str, Any],
    external_summary: Mapping[str, Any],
    output_files: Mapping[str, str],
) -> None:
    by_key = {(row["domain"], row["split"]): row for row in threshold_summary_rows}
    internal_test = by_key.get(("internal", "test"))
    external_overall = by_key.get(("external", "overall"))
    external_test = by_key.get(("external", "test"))

    lines = [
        "# Habitat-CBM 3D Result Diagnostics",
        "",
        f"- Run ID: `{run_id}`",
        f"- Internal threshold: `{_format_metric(_threshold_from_run_summary(run_summary))}`",
        f"- External threshold: `{_format_metric(_threshold_from_external_summary(external_summary))}`",
        "",
        "## Key Metrics",
        "",
        "| Domain | Split | N | Pos | AUC | ACC | SEN | SPE | F1 | Threshold | Oracle Youden |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in threshold_summary_rows:
        if row["split"] not in {"train", "val", "test", "overall"}:
            continue
        lines.append(
            "| {domain} | {split} | {n} | {n_positive} | {auc} | {acc} | {sen} | {spe} | {f1} | {thr} | {oracle} |".format(
                domain=row["domain"],
                split=row["split"],
                n=row["n"],
                n_positive=row["n_positive"],
                auc=_format_metric(row["formal_auc"]),
                acc=_format_metric(row["formal_acc"]),
                sen=_format_metric(row["formal_sen"]),
                spe=_format_metric(row["formal_spe"]),
                f1=_format_metric(row["formal_f1"]),
                thr=_format_metric(row["formal_threshold"]),
                oracle=_format_metric(row["oracle_youden_threshold"]),
            )
        )

    lines.extend(["", "## Interpretation", ""])
    if internal_test is not None:
        lines.append(
            "- Internal test keeps useful ranking signal "
            f"(AUC={_format_metric(internal_test['formal_auc'])}), but the formal threshold "
            f"{_format_metric(internal_test['formal_threshold'])} produces "
            f"FP={internal_test['formal_fp']} and SPE={_format_metric(internal_test['formal_spe'])}."
        )
        lines.append(
            "- The internal test oracle Youden threshold is diagnostic only: "
            f"{_format_metric(internal_test['oracle_youden_threshold'])}, "
            f"ACC={_format_metric(internal_test['oracle_acc'])}, SPE={_format_metric(internal_test['oracle_spe'])}."
        )
    if external_overall is not None:
        lines.append(
            "- External overall ranking is weak "
            f"(AUC={_format_metric(external_overall['formal_auc'])}); "
            f"probability mean={_format_metric(external_overall['prob_mean'])}, "
            f"formal SEN={_format_metric(external_overall['formal_sen'])}, "
            f"SPE={_format_metric(external_overall['formal_spe'])}."
        )
        lines.append(
            "- The external oracle threshold is also diagnostic only: "
            f"{_format_metric(external_overall['oracle_youden_threshold'])}. "
            "It can change sensitivity/specificity tradeoff, but it cannot repair the low AUC."
        )
    if external_test is not None:
        lines.append(
            "- External test should be watched separately because it is small: "
            f"N={external_test['n']}, AUC={_format_metric(external_test['formal_auc'])}."
        )

    weak_concepts = []
    for row in concept_rows:
        if row.get("split") == "test" and row.get("scale") == "std":
            try:
                r2 = float(row.get("r2", "nan"))
            except ValueError:
                r2 = float("nan")
            if not math.isfinite(r2) or r2 < 0.25:
                weak_concepts.append(f"{row.get('concept_id')} (R2={_format_metric(r2)})")
    if weak_concepts:
        lines.append("- Weak test/std concept proxies: " + ", ".join(weak_concepts) + ".")

    lines.extend(["", "## Output Files", ""])
    for label, output_path in output_files.items():
        lines.append(f"- {label}: `{output_path}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_figures(
    output_dir: Path,
    *,
    tables: Sequence[PredictionTable],
    threshold_rows: Sequence[Mapping[str, Any]],
    calibration_rows: Sequence[Mapping[str, Any]],
    dpi: int,
) -> Dict[str, str]:
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

    threshold_map: Dict[Tuple[str, str], List[Mapping[str, Any]]] = {}
    for row in threshold_rows:
        threshold_map.setdefault((str(row["domain"]), str(row["split"])), []).append(row)
    calibration_map: Dict[Tuple[str, str], List[Mapping[str, Any]]] = {}
    for row in calibration_rows:
        if str(row["bin_idx"]).startswith("sklearn_"):
            calibration_map.setdefault((str(row["domain"]), str(row["split"])), []).append(row)

    for table in tables:
        key = (table.domain, table.split)
        safe_name = f"{table.domain}_{table.split}".replace("/", "_")

        hist_path = figure_dir / f"probability_hist_{safe_name}.png"
        fig, ax = plt.subplots(figsize=(6.0, 4.2), dpi=dpi)
        bins = np.linspace(0.0, 1.0, 21)
        ax.hist(table.y_prob[table.y_true == 0], bins=bins, alpha=0.65, label="IDH-wildtype", color="#4c78a8")
        ax.hist(table.y_prob[table.y_true == 1], bins=bins, alpha=0.65, label="IDH-mutant", color="#f58518")
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel("Predicted probability of IDH-mutant")
        ax.set_ylabel("Patients")
        ax.set_title(f"Probability distribution ({table.domain}/{table.split})")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(hist_path)
        plt.close(fig)
        generated[f"probability_hist_{safe_name}"] = str(hist_path)

        curve_rows = sorted(threshold_map.get(key, []), key=lambda row: float(row["threshold"]))
        if curve_rows:
            curve_path = figure_dir / f"threshold_tradeoff_{safe_name}.png"
            fig, ax = plt.subplots(figsize=(6.0, 4.2), dpi=dpi)
            ax.plot([float(row["threshold"]) for row in curve_rows], [float(row["sen"]) for row in curve_rows], label="SEN")
            ax.plot([float(row["threshold"]) for row in curve_rows], [float(row["spe"]) for row in curve_rows], label="SPE")
            ax.plot([float(row["threshold"]) for row in curve_rows], [float(row["f1"]) for row in curve_rows], label="F1")
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.0)
            ax.set_xlabel("Threshold")
            ax.set_ylabel("Metric")
            ax.set_title(f"Threshold tradeoff ({table.domain}/{table.split})")
            ax.legend(frameon=False)
            ax.grid(alpha=0.2)
            fig.tight_layout()
            fig.savefig(curve_path)
            plt.close(fig)
            generated[f"threshold_tradeoff_{safe_name}"] = str(curve_path)

        cal_rows = calibration_map.get(key, [])
        if cal_rows:
            cal_path = figure_dir / f"calibration_{safe_name}.png"
            fig, ax = plt.subplots(figsize=(5.2, 5.0), dpi=dpi)
            ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="0.55", label="Perfect")
            ax.plot(
                [float(row["mean_pred_prob"]) for row in cal_rows],
                [float(row["fraction_positive"]) for row in cal_rows],
                marker="o",
                label="Model",
            )
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.0)
            ax.set_xlabel("Mean predicted probability")
            ax.set_ylabel("Fraction positive")
            ax.set_title(f"Calibration ({table.domain}/{table.split})")
            ax.legend(frameon=False)
            ax.grid(alpha=0.2)
            fig.tight_layout()
            fig.savefig(cal_path)
            plt.close(fig)
            generated[f"calibration_{safe_name}"] = str(cal_path)

    return generated


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze existing Habitat-CBM 3D prediction outputs.")
    parser.add_argument("--run-dir", type=Path, default=None, help="Run directory under results/habitat_CBM_3D.")
    parser.add_argument("--run-id", type=str, default=None, help="Run id. Defaults to basename of --run-dir.")
    parser.add_argument("--internal-predictions-csv", type=Path, default=None)
    parser.add_argument("--external-predictions-csv", type=Path, default=None)
    parser.add_argument("--run-summary-json", type=Path, default=None)
    parser.add_argument("--external-summary-json", type=Path, default=None)
    parser.add_argument("--external-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--internal-threshold", type=float, default=None)
    parser.add_argument("--external-threshold", type=float, default=None)
    parser.add_argument("--grid-size", type=int, default=101)
    parser.add_argument("--calibration-bins", type=int, default=10)
    parser.add_argument("--export-plots", action="store_true")
    parser.add_argument("--dpi", type=int, default=300)
    return parser


def main() -> None:
    args = _build_argparser().parse_args()
    run_id = _infer_run_id(args.run_dir, args.run_id)
    paths = _resolve_default_paths(args, run_id)

    internal_predictions = paths["internal_predictions"]
    external_predictions = paths["external_predictions"]
    if internal_predictions is None and external_predictions is None:
        raise ValueError("No prediction CSV found. Pass --run-dir or explicit prediction CSV paths.")

    run_summary = _load_json_if_exists(paths["run_summary"])
    external_summary = _load_json_if_exists(paths["external_summary"])
    internal_threshold = (
        float(args.internal_threshold)
        if args.internal_threshold is not None
        else (_threshold_from_run_summary(run_summary) or 0.5)
    )
    external_threshold = (
        float(args.external_threshold)
        if args.external_threshold is not None
        else _threshold_from_external_summary(external_summary)
    )
    if external_threshold is None:
        external_threshold = internal_threshold

    output_dir = args.output_dir
    if output_dir is None:
        if args.run_dir is None:
            output_dir = Path("habitat_cbm_3d_diagnostics") / run_id
        else:
            output_dir = args.run_dir / "diagnostics" / "e1_baseline"
    output_dir.mkdir(parents=True, exist_ok=True)

    tables: List[PredictionTable] = []
    if internal_predictions is not None:
        tables.extend(load_prediction_tables(internal_predictions, domain="internal", include_overall=True))
    if external_predictions is not None:
        tables.extend(load_prediction_tables(external_predictions, domain="external", include_overall=True))

    threshold_rows, threshold_summary_rows = build_threshold_rows(
        tables,
        internal_threshold=internal_threshold,
        external_threshold=external_threshold,
        grid_size=args.grid_size,
    )
    calibration_rows = build_calibration_rows(tables, n_bins=args.calibration_bins)
    probability_bin_rows = build_probability_bin_rows(tables, n_bins=args.calibration_bins)
    error_rows = build_error_rows(tables, internal_threshold=internal_threshold, external_threshold=external_threshold)
    concept_rows = collect_concept_metrics(args.run_dir)

    threshold_sweep_csv = output_dir / "threshold_sweep.csv"
    threshold_summary_csv = output_dir / "threshold_summary.csv"
    calibration_csv = output_dir / "calibration_curve.csv"
    probability_bins_csv = output_dir / "probability_bins.csv"
    error_cases_csv = output_dir / "error_cases.csv"
    concept_metrics_csv = output_dir / "concept_metrics_summary.csv"
    summary_json = output_dir / "analysis_summary.json"
    summary_md = output_dir / "analysis_summary.md"

    threshold_fields = [
        "domain",
        "split",
        "threshold",
        "threshold_role",
        "n",
        "n_positive",
        "n_negative",
        "prevalence",
        "pred_positive",
        "pred_positive_rate",
        *METRIC_KEYS,
        "tn",
        "fp",
        "fn",
        "tp",
    ]
    summary_fields = list(threshold_summary_rows[0].keys()) if threshold_summary_rows else []
    _write_csv(threshold_sweep_csv, threshold_fields, threshold_rows)
    _write_csv(threshold_summary_csv, summary_fields, threshold_summary_rows)
    _write_csv(
        calibration_csv,
        [
            "domain",
            "split",
            "bin_idx",
            "prob_low",
            "prob_high",
            "count",
            "n_positive",
            "n_negative",
            "mean_pred_prob",
            "fraction_positive",
        ],
        calibration_rows,
    )
    _write_csv(
        probability_bins_csv,
        ["domain", "split", "bin_idx", "prob_low", "prob_high", "y_true", "count"],
        probability_bin_rows,
    )
    _write_csv(
        error_cases_csv,
        [
            "domain",
            "split",
            "patient_id",
            "y_true",
            "prob_idh_mut",
            "threshold",
            "pred_label",
            "error_type",
            "is_error",
            "signed_margin_to_threshold",
            "abs_margin_to_threshold",
        ],
        error_rows,
    )
    if concept_rows:
        _write_csv(
            concept_metrics_csv,
            ["split", "scale", "concept_id", "n_patients", "mae", "rmse", "r2", "pearson_r", "source_csv"],
            concept_rows,
        )

    output_files = {
        "threshold_sweep": str(threshold_sweep_csv),
        "threshold_summary": str(threshold_summary_csv),
        "calibration_curve": str(calibration_csv),
        "probability_bins": str(probability_bins_csv),
        "error_cases": str(error_cases_csv),
        "analysis_summary_json": str(summary_json),
        "analysis_summary_markdown": str(summary_md),
    }
    if concept_rows:
        output_files["concept_metrics_summary"] = str(concept_metrics_csv)

    write_markdown_report(
        summary_md,
        run_id=run_id,
        threshold_summary_rows=threshold_summary_rows,
        concept_rows=concept_rows,
        run_summary=run_summary,
        external_summary=external_summary,
        output_files=output_files,
    )

    summary = {
        "run_id": run_id,
        "output_dir": str(output_dir),
        "internal_predictions_csv": str(internal_predictions) if internal_predictions is not None else None,
        "external_predictions_csv": str(external_predictions) if external_predictions is not None else None,
        "run_summary_json": str(paths["run_summary"]) if paths["run_summary"] is not None else None,
        "external_summary_json": str(paths["external_summary"]) if paths["external_summary"] is not None else None,
        "internal_threshold": internal_threshold,
        "external_threshold": external_threshold,
        "tables": [
            {
                "domain": table.domain,
                "split": table.split,
                "n": int(table.y_true.size),
                "n_positive": int((table.y_true == 1).sum()),
                "source_csv": str(table.source_csv),
            }
            for table in tables
        ],
        "threshold_summary": threshold_summary_rows,
        "concept_metrics_rows": len(concept_rows),
        "files": output_files,
        "plots_requested": bool(args.export_plots),
    }
    _write_json(summary_json, summary)

    generated_figures: Dict[str, str] = {}
    if args.export_plots:
        generated_figures = export_figures(
            output_dir,
            tables=tables,
            threshold_rows=threshold_rows,
            calibration_rows=calibration_rows,
            dpi=args.dpi,
        )
        if generated_figures:
            summary["files"].update({f"figure_{key}": value for key, value in generated_figures.items()})
            summary["generated_figures"] = generated_figures
            _write_json(summary_json, summary)

    print(f"[OK] Habitat-CBM 3D diagnostics exported to {output_dir}")
    print(f"[OK] Threshold summary: {threshold_summary_csv}")
    print(f"[OK] Analysis report: {summary_md}")
    if generated_figures:
        print(f"[OK] Figures: {len(generated_figures)}")


if __name__ == "__main__":
    main()
