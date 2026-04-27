#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Statistical validation for patient-level IDH prediction results.

This script performs post-hoc statistical analysis only. It reads existing
patient-level test predictions and does not load MRI images, checkpoints, or
model code.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import binomtest, fisher_exact
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1]
LABS_ROOT = SCRIPT_PATH.parents[2]
RESULTS_ROOT = LABS_ROOT / "results"

DEFAULT_HABITAT = (
    RESULTS_ROOT
    / "habitat_CBM"
    / "20260422_145024"
    / "patient_predictions_habitat_cbm_20260422_145024.csv"
)
DEFAULT_RESNET = (
    RESULTS_ROOT
    / "baseline_ResNet18"
    / "20260423_113830"
    / "patient_predictions_resnet18_test_20260423_113830.csv"
)
DEFAULT_RADIOMICS = (
    RESULTS_ROOT
    / "baseline_RadiomicsLR"
    / "20260414_164612"
    / "patient_predictions_radiomics_lr_test_20260414_164612.csv"
)
DEFAULT_OUTPUT_DIR = RESULTS_ROOT / "statistical_validation"

MODEL_SPECS = (
    ("habitat", "Habitat-CBM", "Habitat-CBM"),
    ("resnet", "ResNet-18", "ResNet-18"),
    ("radiomics", "Radiomics+LR", "Radiomics+LR"),
)
METRICS = ("auc", "acc", "sen", "spe", "f1")
BOOTSTRAP_SEED_OFFSETS = {
    "habitat": 11,
    "resnet": 23,
    "radiomics": 37,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute bootstrap confidence intervals, DeLong AUC tests, "
            "McNemar tests, and Fisher exact tests for IDH prediction results."
        )
    )
    parser.add_argument("--habitat-predictions", type=Path, default=DEFAULT_HABITAT)
    parser.add_argument("--resnet-predictions", type=Path, default=DEFAULT_RESNET)
    parser.add_argument("--radiomics-predictions", type=Path, default=DEFAULT_RADIOMICS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split", default="test", help="Split to analyze when a split column exists.")
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--alpha", type=float, default=0.05)
    return parser.parse_args()


def read_predictions(path: Path, split: str, model_key: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing prediction file for {model_key}: {path}")

    df = pd.read_csv(path, dtype={"patient_id": str})
    required = {"patient_id", "y_true", "prob_idh_mut", "pred_label"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    if "split" in df.columns:
        df = df[df["split"].astype(str).str.lower() == split.lower()].copy()
    if df.empty:
        raise ValueError(f"No rows found for split={split!r} in {path}")
    if df["patient_id"].duplicated().any():
        duplicated = sorted(df.loc[df["patient_id"].duplicated(), "patient_id"].unique())
        raise ValueError(f"{path} has duplicated patient_id values: {duplicated[:10]}")

    df["patient_id"] = df["patient_id"].astype(str)
    df["y_true"] = df["y_true"].astype(int)
    df["prob_idh_mut"] = df["prob_idh_mut"].astype(float)
    df["pred_label"] = df["pred_label"].astype(int)
    return df[["patient_id", "y_true", "prob_idh_mut", "pred_label"]].sort_values("patient_id")


def build_merged_predictions(
    habitat: pd.DataFrame,
    resnet: pd.DataFrame,
    radiomics: pd.DataFrame,
) -> pd.DataFrame:
    frames = {
        "habitat": habitat,
        "resnet": resnet,
        "radiomics": radiomics,
    }
    id_sets = {key: set(df["patient_id"]) for key, df in frames.items()}
    reference_ids = id_sets["habitat"]
    for key, ids in id_sets.items():
        if ids != reference_ids:
            missing = sorted(reference_ids - ids)
            extra = sorted(ids - reference_ids)
            raise ValueError(
                f"Patient IDs differ between habitat and {key}. "
                f"missing={missing[:10]}, extra={extra[:10]}"
            )

    merged = habitat.rename(
        columns={"prob_idh_mut": "prob_habitat", "pred_label": "pred_habitat"}
    )
    for key, df in (("resnet", resnet), ("radiomics", radiomics)):
        renamed = df.rename(
            columns={
                "y_true": f"y_true_{key}",
                "prob_idh_mut": f"prob_{key}",
                "pred_label": f"pred_{key}",
            }
        )
        merged = merged.merge(renamed, on="patient_id", how="inner")
        mismatch = merged["y_true"] != merged[f"y_true_{key}"]
        if mismatch.any():
            bad_ids = merged.loc[mismatch, "patient_id"].tolist()
            raise ValueError(f"y_true mismatch for {key}: {bad_ids[:10]}")
        merged = merged.drop(columns=[f"y_true_{key}"])

    return merged.sort_values("patient_id").reset_index(drop=True)


def safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    auc = safe_auc(y_true, y_prob)
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sen = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")
    spe = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")
    return {
        "auc": auc,
        "acc": acc,
        "sen": sen,
        "spe": spe,
        "f1": f1,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def bootstrap_metric_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> Tuple[Dict[str, Tuple[float, float]], Dict[str, int]]:
    rng = np.random.default_rng(seed)
    values: Dict[str, List[float]] = {metric: [] for metric in METRICS}

    for _ in range(n_bootstrap):
        indices = rng.integers(0, len(y_true), len(y_true))
        metrics = compute_metrics(y_true[indices], y_prob[indices], y_pred[indices])
        for metric in METRICS:
            values[metric].append(metrics[metric])

    intervals: Dict[str, Tuple[float, float]] = {}
    valid_counts: Dict[str, int] = {}
    for metric, metric_values in values.items():
        arr = np.asarray(metric_values, dtype=float)
        valid = arr[~np.isnan(arr)]
        valid_counts[metric] = int(valid.size)
        if valid.size == 0:
            intervals[metric] = (float("nan"), float("nan"))
        else:
            low, high = np.percentile(valid, [2.5, 97.5])
            intervals[metric] = (float(low), float(high))
    return intervals, valid_counts


def compute_midrank(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    sorted_values = values[order]
    midranks = np.zeros(len(values), dtype=float)

    i = 0
    while i < len(values):
        j = i
        while j < len(values) and sorted_values[j] == sorted_values[i]:
            j += 1
        midranks[i:j] = 0.5 * (i + j - 1) + 1.0
        i = j

    out = np.empty(len(values), dtype=float)
    out[order] = midranks
    return out


def fast_delong(predictions_sorted: np.ndarray, n_positive: int) -> Tuple[np.ndarray, np.ndarray]:
    n_models = predictions_sorted.shape[0]
    n_total = predictions_sorted.shape[1]
    n_negative = n_total - n_positive
    if n_positive <= 0 or n_negative <= 0:
        raise ValueError("DeLong test requires at least one positive and one negative sample.")

    positive_predictions = predictions_sorted[:, :n_positive]
    negative_predictions = predictions_sorted[:, n_positive:]

    tx = np.empty((n_models, n_positive), dtype=float)
    ty = np.empty((n_models, n_negative), dtype=float)
    tz = np.empty((n_models, n_total), dtype=float)

    for model_idx in range(n_models):
        tx[model_idx, :] = compute_midrank(positive_predictions[model_idx, :])
        ty[model_idx, :] = compute_midrank(negative_predictions[model_idx, :])
        tz[model_idx, :] = compute_midrank(predictions_sorted[model_idx, :])

    aucs = tz[:, :n_positive].sum(axis=1) / n_positive / n_negative
    aucs -= (n_positive + 1.0) / (2.0 * n_negative)

    v01 = (tz[:, :n_positive] - tx) / n_negative
    v10 = 1.0 - (tz[:, n_positive:] - ty) / n_positive
    sx = np.atleast_2d(np.cov(v01))
    sy = np.atleast_2d(np.cov(v10))
    delong_cov = sx / n_positive + sy / n_negative
    return aucs, delong_cov


def delong_roc_test(
    y_true: np.ndarray,
    reference_prob: np.ndarray,
    comparison_prob: np.ndarray,
) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    order = np.argsort(-y_true)
    n_positive = int(np.sum(y_true == 1))
    predictions_sorted = np.vstack([reference_prob, comparison_prob])[:, order]
    aucs, covariance = fast_delong(predictions_sorted, n_positive)

    auc_diff = float(aucs[0] - aucs[1])
    variance = float(covariance[0, 0] + covariance[1, 1] - 2.0 * covariance[0, 1])
    if variance <= 0 or math.isnan(variance):
        z_score = float("nan")
        p_value = float("nan")
    else:
        z_score = float(abs(auc_diff) / math.sqrt(variance))
        p_value = float(2.0 * stats.norm.sf(z_score))

    return {
        "reference_auc": float(aucs[0]),
        "comparison_auc": float(aucs[1]),
        "auc_diff": auc_diff,
        "variance": variance,
        "z": z_score,
        "p_value": p_value,
    }


def mcnemar_exact_test(
    y_true: np.ndarray,
    reference_pred: np.ndarray,
    comparison_pred: np.ndarray,
) -> Dict[str, float]:
    reference_correct = reference_pred == y_true
    comparison_correct = comparison_pred == y_true

    both_correct = int(np.sum(reference_correct & comparison_correct))
    reference_correct_comparison_wrong = int(np.sum(reference_correct & ~comparison_correct))
    reference_wrong_comparison_correct = int(np.sum(~reference_correct & comparison_correct))
    both_wrong = int(np.sum(~reference_correct & ~comparison_correct))

    discordant = reference_correct_comparison_wrong + reference_wrong_comparison_correct
    if discordant == 0:
        p_value = 1.0
    else:
        p_value = float(
            binomtest(
                reference_correct_comparison_wrong,
                n=discordant,
                p=0.5,
                alternative="two-sided",
            ).pvalue
        )

    return {
        "both_correct": both_correct,
        "reference_correct_comparison_wrong": reference_correct_comparison_wrong,
        "reference_wrong_comparison_correct": reference_wrong_comparison_correct,
        "both_wrong": both_wrong,
        "discordant": discordant,
        "p_value": p_value,
    }


def fisher_class_error_test(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    negative = y_true == 0
    positive = y_true == 1
    correct = y_pred == y_true

    negative_correct = int(np.sum(negative & correct))
    negative_wrong = int(np.sum(negative & ~correct))
    positive_correct = int(np.sum(positive & correct))
    positive_wrong = int(np.sum(positive & ~correct))
    table = [[negative_correct, negative_wrong], [positive_correct, positive_wrong]]
    odds_ratio, p_value = fisher_exact(table, alternative="two-sided")

    return {
        "negative_correct": negative_correct,
        "negative_wrong": negative_wrong,
        "positive_correct": positive_correct,
        "positive_wrong": positive_wrong,
        "odds_ratio": float(odds_ratio),
        "p_value": float(p_value),
    }


def format_ci(point: float, low: float, high: float) -> str:
    if np.isnan(point):
        return "NA"
    if np.isnan(low) or np.isnan(high):
        return f"{point:.4f} (NA)"
    return f"{point:.4f} ({low:.4f}-{high:.4f})"


def p_text(p_value: float) -> str:
    if np.isnan(p_value):
        return "NA"
    if p_value < 0.001:
        return "<0.001"
    return f"{p_value:.4f}"


def significance_label(p_value: float, alpha: float) -> str:
    if np.isnan(p_value):
        return "not_computable"
    return "significant" if p_value < alpha else "not_significant"


def write_markdown_summary(
    path: Path,
    metrics_rows: Sequence[Mapping[str, object]],
    delong_rows: Sequence[Mapping[str, object]],
    mcnemar_rows: Sequence[Mapping[str, object]],
    fisher_rows: Sequence[Mapping[str, object]],
    n_patients: int,
    class_counts: Mapping[int, int],
    n_bootstrap: int,
    alpha: float,
) -> None:
    lines: List[str] = []
    lines.append("# Statistical Validation Summary")
    lines.append("")
    lines.append(f"- Test patients: {n_patients}")
    lines.append(f"- Class distribution: y=0: {class_counts.get(0, 0)}, y=1: {class_counts.get(1, 0)}")
    lines.append(f"- Bootstrap resampling: patient-level, n={n_bootstrap}")
    lines.append(f"- Significance threshold: alpha={alpha}")
    lines.append("")

    lines.append("## Metrics With Bootstrap 95% CI")
    lines.append("")
    lines.append("| Model | AUC (95% CI) | ACC (95% CI) | SEN (95% CI) | SPE (95% CI) | F1 (95% CI) | TN | FP | FN | TP |")
    lines.append("| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |")
    for row in metrics_rows:
        lines.append(
            "| {model} | {auc} | {acc} | {sen} | {spe} | {f1} | {tn} | {fp} | {fn} | {tp} |".format(
                model=row["model"],
                auc=row["auc_with_ci"],
                acc=row["acc_with_ci"],
                sen=row["sen_with_ci"],
                spe=row["spe_with_ci"],
                f1=row["f1_with_ci"],
                tn=row["tn"],
                fp=row["fp"],
                fn=row["fn"],
                tp=row["tp"],
            )
        )
    lines.append("")

    lines.append("## DeLong AUC Tests")
    lines.append("")
    lines.append("| Comparison | AUC diff | z | p value | Interpretation |")
    lines.append("| --- | ---: | ---: | ---: | --- |")
    for row in delong_rows:
        if row["significance"] == "significant":
            interpretation = "AUC difference reached statistical significance."
        elif row["significance"] == "not_significant":
            interpretation = "AUC difference did not reach statistical significance."
        else:
            interpretation = "AUC difference could not be tested."
        lines.append(
            f"| {row['comparison']} | {row['auc_diff']:.4f} | "
            f"{row['z']:.4f} | {p_text(row['p_value'])} | {interpretation} |"
        )
    lines.append("")

    lines.append("## McNemar Tests")
    lines.append("")
    lines.append("| Comparison | Ref correct / Comp wrong | Ref wrong / Comp correct | p value | Interpretation |")
    lines.append("| --- | ---: | ---: | ---: | --- |")
    for row in mcnemar_rows:
        if row["significance"] == "significant":
            interpretation = "Fixed-threshold error pattern differed significantly."
        else:
            interpretation = "Fixed-threshold error pattern did not differ significantly."
        lines.append(
            "| {comparison} | {b} | {c} | {p} | {interpretation} |".format(
                comparison=row["comparison"],
                b=row["reference_correct_comparison_wrong"],
                c=row["reference_wrong_comparison_correct"],
                p=p_text(row["p_value"]),
                interpretation=interpretation,
            )
        )
    lines.append("")

    lines.append("## Fisher Exact Tests")
    lines.append("")
    lines.append("| Model | Negative correct/wrong | Positive correct/wrong | p value | Interpretation |")
    lines.append("| --- | --- | --- | ---: | --- |")
    for row in fisher_rows:
        if row["significance"] == "significant":
            interpretation = "Correct/error distribution differed by true class."
        else:
            interpretation = "No significant class-wise correct/error imbalance was detected."
        lines.append(
            "| {model} | {nc}/{nw} | {pc}/{pw} | {p} | {interpretation} |".format(
                model=row["model"],
                nc=row["negative_correct"],
                nw=row["negative_wrong"],
                pc=row["positive_correct"],
                pw=row["positive_wrong"],
                p=p_text(row["p_value"]),
                interpretation=interpretation,
            )
        )
    lines.append("")

    lines.append("## Suggested Wording")
    lines.append("")
    lines.append(
        "Habitat-CBM achieved the highest AUC point estimate on the current test split. "
        "Because the test set is small, the result should be reported together with "
        "patient-level bootstrap confidence intervals and DeLong p values. If the "
        "DeLong comparisons are not significant, describe the result as a higher point "
        "estimate or a reference advantage under the current split, not as statistically "
        "significant superiority."
    )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def to_jsonable(value: object) -> object:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [to_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    pd.DataFrame(rows).to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    predictions = {
        "habitat": read_predictions(args.habitat_predictions, args.split, "habitat"),
        "resnet": read_predictions(args.resnet_predictions, args.split, "resnet"),
        "radiomics": read_predictions(args.radiomics_predictions, args.split, "radiomics"),
    }
    merged = build_merged_predictions(**predictions)
    merged.to_csv(args.output_dir / "test_predictions_merged.csv", index=False)

    y_true = merged["y_true"].to_numpy(dtype=int)
    class_counts = {int(k): int(v) for k, v in pd.Series(y_true).value_counts().sort_index().items()}

    metrics_rows: List[Dict[str, object]] = []
    delong_rows: List[Dict[str, object]] = []
    mcnemar_rows: List[Dict[str, object]] = []
    fisher_rows: List[Dict[str, object]] = []

    for key, model_name, _ in MODEL_SPECS:
        y_prob = merged[f"prob_{key}"].to_numpy(dtype=float)
        y_pred = merged[f"pred_{key}"].to_numpy(dtype=int)
        point = compute_metrics(y_true, y_prob, y_pred)
        intervals, valid_counts = bootstrap_metric_ci(
            y_true=y_true,
            y_prob=y_prob,
            y_pred=y_pred,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed + BOOTSTRAP_SEED_OFFSETS[key],
        )

        row: Dict[str, object] = {
            "model": model_name,
            "n_patients": int(len(y_true)),
            "tn": point["tn"],
            "fp": point["fp"],
            "fn": point["fn"],
            "tp": point["tp"],
        }
        for metric in METRICS:
            low, high = intervals[metric]
            row[metric] = point[metric]
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
            row[f"{metric}_bootstrap_valid"] = valid_counts[metric]
            row[f"{metric}_with_ci"] = format_ci(point[metric], low, high)
        metrics_rows.append(row)

        fisher = fisher_class_error_test(y_true, y_pred)
        fisher_rows.append(
            {
                "model": model_name,
                **fisher,
                "significance": significance_label(fisher["p_value"], args.alpha),
            }
        )

    reference_prob = merged["prob_habitat"].to_numpy(dtype=float)
    reference_pred = merged["pred_habitat"].to_numpy(dtype=int)
    for key, model_name, _ in MODEL_SPECS:
        if key == "habitat":
            continue
        comparison_prob = merged[f"prob_{key}"].to_numpy(dtype=float)
        delong = delong_roc_test(y_true, reference_prob, comparison_prob)
        delong_rows.append(
            {
                "reference_model": "Habitat-CBM",
                "comparison_model": model_name,
                "comparison": f"Habitat-CBM vs {model_name}",
                **delong,
                "significance": significance_label(delong["p_value"], args.alpha),
            }
        )

        comparison_pred = merged[f"pred_{key}"].to_numpy(dtype=int)
        mcnemar = mcnemar_exact_test(y_true, reference_pred, comparison_pred)
        mcnemar_rows.append(
            {
                "reference_model": "Habitat-CBM",
                "comparison_model": model_name,
                "comparison": f"Habitat-CBM vs {model_name}",
                **mcnemar,
                "significance": significance_label(mcnemar["p_value"], args.alpha),
            }
        )

    write_csv(args.output_dir / "metrics_with_bootstrap_ci.csv", metrics_rows)
    write_csv(args.output_dir / "delong_tests.csv", delong_rows)
    write_csv(args.output_dir / "mcnemar_tests.csv", mcnemar_rows)
    write_csv(args.output_dir / "fisher_tests.csv", fisher_rows)

    write_markdown_summary(
        path=args.output_dir / "statistical_validation_summary.md",
        metrics_rows=metrics_rows,
        delong_rows=delong_rows,
        mcnemar_rows=mcnemar_rows,
        fisher_rows=fisher_rows,
        n_patients=int(len(y_true)),
        class_counts=class_counts,
        n_bootstrap=args.n_bootstrap,
        alpha=args.alpha,
    )

    summary = {
        "inputs": {
            "habitat": str(args.habitat_predictions),
            "resnet": str(args.resnet_predictions),
            "radiomics": str(args.radiomics_predictions),
        },
        "output_dir": str(args.output_dir),
        "split": args.split,
        "n_patients": int(len(y_true)),
        "class_counts": class_counts,
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
        "alpha": args.alpha,
        "metrics": metrics_rows,
        "delong_tests": delong_rows,
        "mcnemar_tests": mcnemar_rows,
        "fisher_tests": fisher_rows,
    }
    (args.output_dir / "statistical_validation_summary.json").write_text(
        json.dumps(to_jsonable(summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Wrote statistical validation outputs to: {args.output_dir}")
    print("Main table: metrics_with_bootstrap_ci.csv")
    print("Summary: statistical_validation_summary.md")


if __name__ == "__main__":
    main()
