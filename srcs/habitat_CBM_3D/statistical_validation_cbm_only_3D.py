#!/usr/bin/env python3
"""CBM-only bootstrap validation for Habitat-CBM 3D prediction CSVs.

This script intentionally does not compare against other models. It estimates
patient-level confidence intervals for one Habitat-CBM prediction file, writes
all tabular/JSON summaries first, and only then exports optional figures.
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

try:
    from sklearn.metrics import accuracy_score as _sk_accuracy_score
    from sklearn.metrics import confusion_matrix as _sk_confusion_matrix
    from sklearn.metrics import f1_score as _sk_f1_score
    from sklearn.metrics import roc_auc_score as _sk_roc_auc_score
    from sklearn.metrics import roc_curve as _sk_roc_curve

    SKLEARN_METRICS_AVAILABLE = True
except Exception:  # pragma: no cover - local fallback only
    _sk_accuracy_score = None
    _sk_confusion_matrix = None
    _sk_f1_score = None
    _sk_roc_auc_score = None
    _sk_roc_curve = None
    SKLEARN_METRICS_AVAILABLE = False


REQUIRED_COLUMNS = ("patient_id", "y_true", "prob_idh_mut", "pred_label")
METRIC_ORDER = ("auc", "acc", "sen", "spe", "f1")
METRICS_BACKEND = "sklearn" if SKLEARN_METRICS_AVAILABLE else "numpy_fallback"


@dataclass(frozen=True)
class PredictionTable:
    patient_ids: List[str]
    y_true: np.ndarray
    y_prob: np.ndarray
    y_pred: np.ndarray
    split: str
    source_rows: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute CBM-only bootstrap confidence intervals from a 3D Habitat-CBM prediction CSV."
    )
    parser.add_argument("--predictions-csv", type=Path, required=True, help="CSV with patient_id,y_true,prob_idh_mut,pred_label.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for statistical validation outputs.")
    parser.add_argument(
        "--split",
        default="test",
        help="Split to evaluate. If set to all, all rows are used. If a split column exists, other values filter rows.",
    )
    parser.add_argument("--model-name", default="Habitat-CBM", help="Model name written to output tables.")
    parser.add_argument("--n-bootstrap", type=int, default=2000, help="Number of patient-level bootstrap samples.")
    parser.add_argument("--alpha", type=float, default=0.05, help="Two-sided CI alpha; 0.05 gives 95%% CI.")
    parser.add_argument("--seed", type=int, default=2026, help="Bootstrap random seed.")
    parser.add_argument("--export-plots", action="store_true", help="Write PNG figures after CSV/JSON outputs are saved.")
    parser.add_argument("--dpi", type=int, default=300, help="Figure DPI.")
    return parser.parse_args()


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Prediction CSV not found: {path}")
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Prediction CSV has no header: {path}")
        missing = [col for col in REQUIRED_COLUMNS if col not in reader.fieldnames]
        if missing:
            raise ValueError(f"Prediction CSV missing required columns {missing}: {path}")
        return [dict(row) for row in reader]


def _as_int(value: str, column: str, patient_id: str) -> int:
    try:
        parsed = int(float(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid integer value for {column} in patient {patient_id}: {value!r}") from exc
    if parsed not in (0, 1):
        raise ValueError(f"{column} must be 0 or 1 in patient {patient_id}, got {parsed}")
    return parsed


def _as_float(value: str, column: str, patient_id: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid float value for {column} in patient {patient_id}: {value!r}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{column} must be finite in patient {patient_id}, got {parsed}")
    return parsed


def load_predictions(path: Path, split: str) -> PredictionTable:
    rows = _read_csv_rows(path)
    split_normalized = str(split).strip()
    if split_normalized.lower() != "all" and rows and "split" in rows[0]:
        rows = [row for row in rows if row.get("split") == split_normalized]
    if not rows:
        raise ValueError(f"No prediction rows available for split={split_normalized!r} in {path}")

    patient_ids: List[str] = []
    y_true: List[int] = []
    y_prob: List[float] = []
    y_pred: List[int] = []
    seen = set()
    for row in rows:
        patient_id = str(row["patient_id"]).strip()
        if not patient_id:
            raise ValueError("Encountered an empty patient_id")
        if patient_id in seen:
            raise ValueError(f"Duplicate patient_id after split filtering: {patient_id}")
        seen.add(patient_id)
        patient_ids.append(patient_id)
        y_true.append(_as_int(row["y_true"], "y_true", patient_id))
        y_prob.append(_as_float(row["prob_idh_mut"], "prob_idh_mut", patient_id))
        y_pred.append(_as_int(row["pred_label"], "pred_label", patient_id))

    return PredictionTable(
        patient_ids=patient_ids,
        y_true=np.asarray(y_true, dtype=np.int64),
        y_prob=np.asarray(y_prob, dtype=np.float64),
        y_pred=np.asarray(y_pred, dtype=np.int64),
        split=split_normalized,
        source_rows=len(rows),
    )


def _safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    if SKLEARN_METRICS_AVAILABLE:
        assert _sk_roc_auc_score is not None
        return float(_sk_roc_auc_score(y_true, y_prob))
    ranks = _average_ranks(y_prob)
    sum_pos_ranks = float(ranks[y_true == 1].sum())
    u_stat = sum_pos_ranks - n_pos * (n_pos + 1) / 2.0
    return float(u_stat / (n_pos * n_neg))


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.shape[0], dtype=np.float64)
    start = 0
    n = values.shape[0]
    while start < n:
        end = start + 1
        while end < n and sorted_values[end] == sorted_values[start]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    if SKLEARN_METRICS_AVAILABLE:
        assert _sk_accuracy_score is not None
        assert _sk_confusion_matrix is not None
        assert _sk_f1_score is not None
        tn, fp, fn, tp = _sk_confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        tn = int(tn)
        fp = int(fp)
        fn = int(fn)
        tp = int(tp)
        acc = float(_sk_accuracy_score(y_true, y_pred))
        f1 = float(_sk_f1_score(y_true, y_pred, zero_division=0))
    else:
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        n = int(y_true.shape[0])
        acc = float((tp + tn) / n) if n else float("nan")
        f1_denominator = 2 * tp + fp + fn
        f1 = float((2 * tp) / f1_denominator) if f1_denominator else 0.0
    sen = float(tp / (tp + fn)) if (tp + fn) else float("nan")
    spe = float(tn / (tn + fp)) if (tn + fp) else float("nan")
    return {
        "auc": _safe_auc(y_true, y_prob),
        "acc": acc,
        "sen": sen,
        "spe": spe,
        "f1": f1,
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
    }


def bootstrap_metrics(
    table: PredictionTable,
    *,
    n_bootstrap: int,
    alpha: float,
    seed: int,
) -> Tuple[List[Dict[str, float]], Dict[str, Dict[str, float]]]:
    if n_bootstrap <= 0:
        raise ValueError(f"--n-bootstrap must be positive, got {n_bootstrap}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"--alpha must be between 0 and 1, got {alpha}")

    rng = np.random.default_rng(seed)
    n = table.y_true.shape[0]
    samples: List[Dict[str, float]] = []
    for bootstrap_idx in range(n_bootstrap):
        indices = rng.integers(0, n, size=n)
        metrics = compute_metrics(table.y_true[indices], table.y_prob[indices], table.y_pred[indices])
        samples.append(
            {
                "bootstrap_idx": float(bootstrap_idx),
                "auc": metrics["auc"],
                "acc": metrics["acc"],
                "sen": metrics["sen"],
                "spe": metrics["spe"],
                "f1": metrics["f1"],
                "valid_auc": float(math.isfinite(metrics["auc"])),
            }
        )

    ci: Dict[str, Dict[str, float]] = {}
    low_pct = 100.0 * alpha / 2.0
    high_pct = 100.0 * (1.0 - alpha / 2.0)
    for metric in METRIC_ORDER:
        values = np.asarray([row[metric] for row in samples], dtype=np.float64)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            ci[metric] = {"low": float("nan"), "high": float("nan"), "valid_bootstraps": 0.0}
        else:
            ci[metric] = {
                "low": float(np.percentile(finite, low_pct)),
                "high": float(np.percentile(finite, high_pct)),
                "valid_bootstraps": float(finite.size),
            }
    return samples, ci


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _json_value(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _fmt(value: float, digits: int = 4) -> str:
    if value is None or not math.isfinite(float(value)):
        return "NA"
    return f"{float(value):.{digits}f}"


def _metric_ci_text(point: float, low: float, high: float) -> str:
    return f"{_fmt(point)} ({_fmt(low)}-{_fmt(high)})"


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def build_metric_row(
    *,
    model_name: str,
    table: PredictionTable,
    metrics: Mapping[str, float],
    ci: Mapping[str, Mapping[str, float]],
    alpha: float,
) -> Dict[str, Any]:
    ci_level = 100.0 * (1.0 - alpha)
    row: Dict[str, Any] = {
        "model": model_name,
        "split": table.split,
        "metrics_backend": METRICS_BACKEND,
        "n": int(table.y_true.shape[0]),
        "n_negative": int((table.y_true == 0).sum()),
        "n_positive": int((table.y_true == 1).sum()),
        "tn": int(metrics["tn"]),
        "fp": int(metrics["fp"]),
        "fn": int(metrics["fn"]),
        "tp": int(metrics["tp"]),
        "ci_level": ci_level,
    }
    for metric in METRIC_ORDER:
        low = float(ci[metric]["low"])
        high = float(ci[metric]["high"])
        point = float(metrics[metric])
        row[metric] = point
        row[f"{metric}_ci_low"] = low
        row[f"{metric}_ci_high"] = high
        row[f"{metric}_ci"] = _metric_ci_text(point, low, high)
        row[f"{metric}_valid_bootstraps"] = int(ci[metric]["valid_bootstraps"])
    return row


def build_roc_rows(table: PredictionTable) -> List[Dict[str, float]]:
    if np.unique(table.y_true).size < 2:
        return []
    if SKLEARN_METRICS_AVAILABLE:
        assert _sk_roc_curve is not None
        fpr, tpr, thresholds = _sk_roc_curve(table.y_true, table.y_prob)
        return [
            {
                "point_idx": float(idx),
                "fpr": float(fpr_value),
                "tpr": float(tpr_value),
                "threshold": float(threshold),
            }
            for idx, (fpr_value, tpr_value, threshold) in enumerate(zip(fpr, tpr, thresholds))
        ]
    thresholds = np.concatenate(([float("inf")], np.sort(np.unique(table.y_prob))[::-1]))
    n_pos = int((table.y_true == 1).sum())
    n_neg = int((table.y_true == 0).sum())
    rows: List[Dict[str, float]] = []
    for idx, threshold in enumerate(thresholds):
        predicted_positive = table.y_prob >= threshold
        tp = int(np.sum((table.y_true == 1) & predicted_positive))
        fp = int(np.sum((table.y_true == 0) & predicted_positive))
        rows.append(
            {
                "point_idx": float(idx),
                "fpr": float(fp / n_neg) if n_neg else float("nan"),
                "tpr": float(tp / n_pos) if n_pos else float("nan"),
                "threshold": float(threshold),
            }
        )
    return rows


def write_summary_markdown(
    path: Path,
    *,
    predictions_csv: Path,
    metric_row: Mapping[str, Any],
    output_files: Mapping[str, str],
) -> None:
    lines = [
        "# CBM-only statistical validation",
        "",
        f"- Predictions CSV: `{predictions_csv}`",
        f"- Model: {metric_row['model']}",
        f"- Split: {metric_row['split']}",
        f"- Metrics backend: {metric_row['metrics_backend']}",
        f"- N: {metric_row['n']} ({metric_row['n_negative']} negative, {metric_row['n_positive']} positive)",
        f"- Confusion matrix: TN={metric_row['tn']}, FP={metric_row['fp']}, FN={metric_row['fn']}, TP={metric_row['tp']}",
        "",
        "| Metric | Point estimate with CI |",
        "| --- | --- |",
    ]
    for metric in METRIC_ORDER:
        lines.append(f"| {metric.upper()} | {metric_row[f'{metric}_ci']} |")
    lines.extend(
        [
            "",
            "## Output files",
            "",
        ]
    )
    for label, file_path in output_files.items():
        lines.append(f"- {label}: `{file_path}`")
    path.write_text("\n".join(lines) + "\n")


def export_plots(
    output_dir: Path,
    *,
    table: PredictionTable,
    metric_row: Mapping[str, Any],
    roc_rows: Sequence[Mapping[str, float]],
    dpi: int,
) -> Dict[str, str]:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - depends on runtime environment
        print(f"[WARN] matplotlib unavailable; skipping plots: {exc}")
        return {}

    plot_paths: Dict[str, str] = {}

    if roc_rows:
        roc_path = output_dir / "roc_curve.png"
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot([row["fpr"] for row in roc_rows], [row["tpr"] for row in roc_rows], lw=2, label=f"AUC={_fmt(metric_row['auc'])}")
        ax.plot([0, 1], [0, 1], linestyle="--", color="0.6", lw=1)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title(f"{metric_row['model']} ROC ({metric_row['split']})")
        ax.legend(loc="lower right", frameon=False)
        fig.tight_layout()
        fig.savefig(roc_path, dpi=dpi)
        plt.close(fig)
        plot_paths["roc_curve"] = str(roc_path)

    prob_path = output_dir / "probability_distribution.png"
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(table.y_prob[table.y_true == 0], bins=20, alpha=0.65, label="IDH-wildtype", color="#4c78a8")
    ax.hist(table.y_prob[table.y_true == 1], bins=20, alpha=0.65, label="IDH-mutant", color="#f58518")
    ax.set_xlabel("Predicted probability of IDH-mutant")
    ax.set_ylabel("Patients")
    ax.set_title(f"Prediction distribution ({metric_row['split']})")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(prob_path, dpi=dpi)
    plt.close(fig)
    plot_paths["probability_distribution"] = str(prob_path)

    forest_path = output_dir / "metric_bootstrap_ci.png"
    labels = [metric.upper() for metric in METRIC_ORDER]
    points = np.asarray([float(metric_row[metric]) for metric in METRIC_ORDER], dtype=np.float64)
    lows = np.asarray([float(metric_row[f"{metric}_ci_low"]) for metric in METRIC_ORDER], dtype=np.float64)
    highs = np.asarray([float(metric_row[f"{metric}_ci_high"]) for metric in METRIC_ORDER], dtype=np.float64)
    y_pos = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6, 4))
    valid = np.isfinite(points) & np.isfinite(lows) & np.isfinite(highs)
    xerr = np.vstack([points[valid] - lows[valid], highs[valid] - points[valid]])
    ax.errorbar(points[valid], y_pos[valid], xerr=xerr, fmt="o", color="#2f4b7c", ecolor="#2f4b7c", capsize=4)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Metric value")
    ax.set_title("Bootstrap confidence intervals")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(forest_path, dpi=dpi)
    plt.close(fig)
    plot_paths["metric_bootstrap_ci"] = str(forest_path)

    return plot_paths


def main() -> None:
    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    table = load_predictions(args.predictions_csv, args.split)
    metrics = compute_metrics(table.y_true, table.y_prob, table.y_pred)
    bootstrap_rows, ci = bootstrap_metrics(
        table,
        n_bootstrap=args.n_bootstrap,
        alpha=args.alpha,
        seed=args.seed,
    )
    metric_row = build_metric_row(
        model_name=args.model_name,
        table=table,
        metrics=metrics,
        ci=ci,
        alpha=args.alpha,
    )
    roc_rows = build_roc_rows(table)
    cm_rows = [
        {"actual": 0, "predicted": 0, "count": int(metrics["tn"])},
        {"actual": 0, "predicted": 1, "count": int(metrics["fp"])},
        {"actual": 1, "predicted": 0, "count": int(metrics["fn"])},
        {"actual": 1, "predicted": 1, "count": int(metrics["tp"])},
    ]

    metrics_csv = output_dir / "metrics_with_bootstrap_ci.csv"
    bootstrap_csv = output_dir / "bootstrap_samples.csv"
    roc_csv = output_dir / "roc_points.csv"
    cm_csv = output_dir / "confusion_matrix.csv"
    summary_md = output_dir / "statistical_validation_summary.md"
    summary_json = output_dir / "statistical_validation_summary.json"

    metric_fields = [
        "model",
        "split",
        "metrics_backend",
        "n",
        "n_negative",
        "n_positive",
        "tn",
        "fp",
        "fn",
        "tp",
        "ci_level",
    ]
    for metric in METRIC_ORDER:
        metric_fields.extend([metric, f"{metric}_ci_low", f"{metric}_ci_high", f"{metric}_ci", f"{metric}_valid_bootstraps"])

    _write_csv(metrics_csv, metric_fields, [metric_row])
    _write_csv(bootstrap_csv, ["bootstrap_idx", "auc", "acc", "sen", "spe", "f1", "valid_auc"], bootstrap_rows)
    _write_csv(roc_csv, ["point_idx", "fpr", "tpr", "threshold"], roc_rows)
    _write_csv(cm_csv, ["actual", "predicted", "count"], cm_rows)

    output_files = {
        "metrics": str(metrics_csv),
        "bootstrap_samples": str(bootstrap_csv),
        "roc_points": str(roc_csv),
        "confusion_matrix": str(cm_csv),
        "summary_json": str(summary_json),
    }
    planned_plots = {}
    if args.export_plots:
        planned_plots = {
            "roc_curve": str(output_dir / "roc_curve.png"),
            "probability_distribution": str(output_dir / "probability_distribution.png"),
            "metric_bootstrap_ci": str(output_dir / "metric_bootstrap_ci.png"),
        }
        output_files.update({f"plot_{name}": file_path for name, file_path in planned_plots.items()})

    write_summary_markdown(
        summary_md,
        predictions_csv=args.predictions_csv,
        metric_row=metric_row,
        output_files={**output_files, "summary_markdown": str(summary_md)},
    )

    summary = {
        "predictions_csv": str(args.predictions_csv),
        "output_dir": str(output_dir),
        "model_name": args.model_name,
        "metrics_backend": METRICS_BACKEND,
        "split": table.split,
        "source_rows": table.source_rows,
        "n_bootstrap": args.n_bootstrap,
        "alpha": args.alpha,
        "seed": args.seed,
        "metrics": metric_row,
        "files": {**output_files, "summary_markdown": str(summary_md)},
        "plots_requested": bool(args.export_plots),
        "planned_plots": planned_plots,
    }
    summary_json.write_text(json.dumps(_json_value(summary), indent=2) + "\n")

    generated_plots: Dict[str, str] = {}
    if args.export_plots:
        generated_plots = export_plots(output_dir, table=table, metric_row=metric_row, roc_rows=roc_rows, dpi=args.dpi)

    print(f"[OK] Wrote CBM-only statistics to {output_dir}")
    print(f"[OK] Metrics: {metrics_csv}")
    if generated_plots:
        print(f"[OK] Figures: {', '.join(generated_plots.values())}")


if __name__ == "__main__":
    main()
