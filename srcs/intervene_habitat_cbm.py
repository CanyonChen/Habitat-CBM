#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Habitat-CBM 患者级概念干预脚本。

输入：
- stage3 checkpoint
- patient_predictions_habitat_cbm_runXX.csv
- patient_concepts_habitat_cbm_runXX.csv
- concept_scaler_stats.json（用于校验/回退转换）

输出：
- intervention_case_list.csv
- intervention_per_case.csv
- intervention_metrics_by_budget.csv
- intervention_correction_summary.csv
- intervention_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent

import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.habitat_CBM import HabitatCBM
from srcs.data_loader_habitat_CBM import load_concept_scaler, resolve_concept_names


def _extract_concept_ids_from_row(row: Mapping[str, str]) -> List[str]:
    matched: List[Tuple[int, str]] = []
    for key in row.keys():
        m = re.fullmatch(r"c(\d+)_abs_error_std", str(key))
        if m is None:
            continue
        matched.append((int(m.group(1)), f"c{int(m.group(1))}"))
    matched.sort(key=lambda item: item[0])
    return [item[1] for item in matched]


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


def _safe_float(value: str, *, field: str, patient_id: str) -> float:
    text = str(value).strip()
    if text == "":
        raise ValueError(f"Empty value for {field} (patient={patient_id})")
    try:
        parsed = float(text)
    except ValueError as exc:
        raise ValueError(
            f"Invalid numeric value for {field} (patient={patient_id}): {text}"
        ) from exc
    if not np.isfinite(parsed):
        raise ValueError(
            f"Non-finite numeric value for {field} (patient={patient_id}): {parsed}"
        )
    return parsed


def _safe_int(value: str, *, field: str, patient_id: str) -> int:
    parsed = int(_safe_float(value, field=field, patient_id=patient_id))
    return parsed


def _safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def _compute_metrics_from_prob(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(np.int64)
    auc = _safe_auc(y_true, y_prob)
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sen = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    spe = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    return {
        "auc": auc,
        "acc": acc,
        "f1": f1,
        "sen": sen,
        "spe": spe,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def _parse_budgets(text: str, n_concepts: int) -> List[int]:
    values: List[int] = []
    for token in [item.strip().lower() for item in text.split(",") if item.strip()]:
        if token == "all":
            values.append(n_concepts)
        else:
            val = int(token)
            if val <= 0:
                raise ValueError(f"Budget must be positive, got {val}")
            values.append(min(val, n_concepts))
    if not values:
        raise ValueError("No valid intervention budgets found.")
    return sorted(set(values))


def _build_patient_maps(
    prediction_rows: Sequence[Mapping[str, str]],
    concept_rows: Sequence[Mapping[str, str]],
    split: str,
    threshold: float,
    scaler_json: Path,
) -> Tuple[Dict[str, Dict[str, object]], List[str], List[str]]:
    pred_map: Dict[str, Dict[str, object]] = {}
    for row in prediction_rows:
        row_split = str(row.get("split", "")).strip().lower()
        if split != "all" and row_split != split:
            continue
        patient_id = str(row.get("patient_id", "")).strip()
        if not patient_id:
            continue
        y_true = _safe_int(str(row.get("y_true", "")), field="y_true", patient_id=patient_id)
        prob = _safe_float(
            str(row.get("prob_idh_mut", "")),
            field="prob_idh_mut",
            patient_id=patient_id,
        )
        pred_label = int(prob >= threshold)
        pred_map[patient_id] = {
            "patient_id": patient_id,
            "split": row_split,
            "y_true": y_true,
            "prob_before": prob,
            "pred_before": pred_label,
            "run_id": str(row.get("run_id", "")).strip(),
            "checkpoint_name": str(row.get("checkpoint_name", "")).strip(),
        }

    if not pred_map:
        raise ValueError(f"No prediction rows available for split={split}.")

    concept_ids = _extract_concept_ids_from_row(concept_rows[0]) if concept_rows else []
    scaler = load_concept_scaler(
        scaler_json,
        concept_names=concept_ids if concept_ids else None,
    )
    concept_ids = list(scaler.concept_names)
    n_concepts = len(concept_ids)

    concept_map: Dict[str, Dict[str, np.ndarray]] = {}
    for row in concept_rows:
        row_split = str(row.get("split", "")).strip().lower()
        if split != "all" and row_split != split:
            continue
        patient_id = str(row.get("patient_id", "")).strip()
        if patient_id not in pred_map:
            continue

        true_std_cols = [f"{concept_id}_true_std" for concept_id in concept_ids]
        pred_std_cols = [f"{concept_id}_pred_std" for concept_id in concept_ids]
        true_raw_cols = [f"{concept_id}_true_raw" for concept_id in concept_ids]
        pred_raw_cols = [f"{concept_id}_pred_raw" for concept_id in concept_ids]

        has_std = all(col in row and str(row[col]).strip() != "" for col in true_std_cols + pred_std_cols)
        has_raw = all(col in row and str(row[col]).strip() != "" for col in true_raw_cols + pred_raw_cols)

        if has_std:
            c_true_std = np.asarray(
                [_safe_float(row[col], field=col, patient_id=patient_id) for col in true_std_cols],
                dtype=np.float32,
            )
            c_pred_std = np.asarray(
                [_safe_float(row[col], field=col, patient_id=patient_id) for col in pred_std_cols],
                dtype=np.float32,
            )
        elif has_raw:
            c_true_raw = np.asarray(
                [_safe_float(row[col], field=col, patient_id=patient_id) for col in true_raw_cols],
                dtype=np.float32,
            )
            c_pred_raw = np.asarray(
                [_safe_float(row[col], field=col, patient_id=patient_id) for col in pred_raw_cols],
                dtype=np.float32,
            )
            c_true_std = scaler.standardize(c_true_raw)
            c_pred_std = scaler.standardize(c_pred_raw)
        else:
            raise ValueError(
                "Concept table must include either std columns (c*_true_std/c*_pred_std) "
                "or raw columns (c*_true_raw/c*_pred_raw)."
            )

        if c_true_std.shape != (n_concepts,) or c_pred_std.shape != (n_concepts,):
            raise ValueError(f"Concept shape mismatch for patient={patient_id}")

        concept_map[patient_id] = {
            "c_true_std": c_true_std,
            "c_pred_std": c_pred_std,
            "abs_error_std": np.abs(c_pred_std - c_true_std),
        }

    missing = sorted(set(pred_map.keys()) - set(concept_map.keys()))
    if missing:
        raise ValueError(
            f"Missing concept rows for {len(missing)} patients, examples: {missing[:5]}"
        )

    for patient_id, item in concept_map.items():
        pred_map[patient_id].update(item)

    patient_ids = sorted(pred_map.keys())
    return pred_map, patient_ids, concept_ids


def _load_model_from_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
    fallback_in_channels: int,
    fallback_n_concepts: int,
) -> HabitatCBM:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model_cfg = payload.get("model_config", {}) if isinstance(payload, dict) else {}
    if not isinstance(model_cfg, Mapping):
        model_cfg = {}
    shared_dropout = model_cfg.get("dropout_p", None)
    if shared_dropout is not None:
        shared = float(shared_dropout)
        concept_dropout_p = float(model_cfg.get("concept_dropout_p", shared))
        label_dropout_p = float(model_cfg.get("label_dropout_p", shared))
    else:
        concept_dropout_p = float(model_cfg.get("concept_dropout_p", 0.3))
        label_dropout_p = float(model_cfg.get("label_dropout_p", 0.1))
    checkpoint_selected_concepts = model_cfg.get("selected_concepts")
    if "n_concepts" in model_cfg:
        checkpoint_n_concepts = int(model_cfg["n_concepts"])
    elif checkpoint_selected_concepts is not None:
        checkpoint_n_concepts = len(resolve_concept_names(checkpoint_selected_concepts))
    else:
        checkpoint_n_concepts = fallback_n_concepts

    model = HabitatCBM(
        in_channels=int(model_cfg.get("in_channels", fallback_in_channels)),
        n_concepts=checkpoint_n_concepts,
        concept_hidden_dim=int(model_cfg.get("concept_hidden_dim", 256)),
        label_hidden_dim=int(model_cfg.get("label_hidden_dim", 32)),
        concept_dropout_p=concept_dropout_p,
        label_dropout_p=label_dropout_p,
        pretrained=False,
    ).to(device)

    if isinstance(payload, dict) and "model_state_dict" in payload:
        state_dict = payload["model_state_dict"]
    elif isinstance(payload, dict):
        state_dict = payload
    else:
        raise ValueError("Unsupported checkpoint format.")

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def run_intervention(
    checkpoint_path: Path,
    patient_predictions_csv: Path,
    patient_concepts_csv: Path,
    concept_scaler_json: Path,
    output_dir: Path,
    split: str,
    budgets_text: str,
    threshold: float,
    low_conf_margin: float,
    device_name: str,
    in_channels: int,
    n_concepts: int,
) -> None:
    pred_rows = _read_csv(patient_predictions_csv)
    concept_rows = _read_csv(patient_concepts_csv)

    patient_map, patient_ids, concept_ids = _build_patient_maps(
        prediction_rows=pred_rows,
        concept_rows=concept_rows,
        split=split,
        threshold=threshold,
        scaler_json=concept_scaler_json,
    )
    n_concepts_from_data = len(concept_ids)
    budgets = _parse_budgets(budgets_text, n_concepts=n_concepts_from_data)

    device = torch.device(device_name)
    model = _load_model_from_checkpoint(
        checkpoint_path=checkpoint_path,
        device=device,
        fallback_in_channels=in_channels,
        fallback_n_concepts=n_concepts,
    )

    # 候选样本 = 误判 OR 低置信度
    candidate_rows: List[Dict[str, object]] = []
    for patient_id in patient_ids:
        item = patient_map[patient_id]
        is_wrong = int(item["pred_before"] != item["y_true"])
        is_low_conf = int(abs(float(item["prob_before"]) - threshold) < low_conf_margin)
        if is_wrong or is_low_conf:
            candidate_rows.append(
                {
                    "patient_id": patient_id,
                    "split": item["split"],
                    "y_true": item["y_true"],
                    "prob_before": item["prob_before"],
                    "pred_before": item["pred_before"],
                    "is_wrong": is_wrong,
                    "is_low_confidence": is_low_conf,
                    "run_id": item["run_id"],
                    "checkpoint_name": item["checkpoint_name"],
                }
            )

    # per-case
    per_case_rows: List[Dict[str, object]] = []
    budget_metrics_rows: List[Dict[str, object]] = []

    y_true_all = np.asarray([int(patient_map[pid]["y_true"]) for pid in patient_ids], dtype=np.int64)
    prob_before_all = np.asarray([float(patient_map[pid]["prob_before"]) for pid in patient_ids], dtype=np.float64)

    candidate_ids = [row["patient_id"] for row in candidate_rows]
    y_true_candidate = np.asarray([int(patient_map[pid]["y_true"]) for pid in candidate_ids], dtype=np.int64)
    prob_before_candidate = np.asarray(
        [float(patient_map[pid]["prob_before"]) for pid in candidate_ids], dtype=np.float64
    )

    correction_rows: List[Dict[str, object]] = []

    for budget_k in budgets:
        after_prob_all: List[float] = []
        after_prob_candidate: List[float] = []

        for patient_id in patient_ids:
            item = patient_map[patient_id]
            c_true_std = np.asarray(item["c_true_std"], dtype=np.float32)
            c_pred_std = np.asarray(item["c_pred_std"], dtype=np.float32)
            order = np.argsort(np.abs(c_pred_std - c_true_std))[::-1]

            k_eff = min(budget_k, c_true_std.shape[0])
            replace_idx = order[:k_eff]
            c_after = c_pred_std.copy()
            c_after[replace_idx] = c_true_std[replace_idx]

            with torch.no_grad():
                c_after_tensor = torch.as_tensor(c_after, dtype=torch.float32, device=device).unsqueeze(0)
                y_logit_after = model.forward_c_to_y(c_after_tensor)
                prob_after = float(torch.sigmoid(y_logit_after).item())

            pred_before = int(item["pred_before"])
            pred_after = int(prob_after >= threshold)
            corrected = int(pred_before != int(item["y_true"]) and pred_after == int(item["y_true"]))

            row = {
                "patient_id": patient_id,
                "split": item["split"],
                "y_true": int(item["y_true"]),
                "budget_k": int(budget_k),
                "concepts_replaced": "|".join(concept_ids[int(idx)] for idx in replace_idx.tolist()),
                "prob_before": float(item["prob_before"]),
                "prob_after": prob_after,
                "pred_before": pred_before,
                "pred_after": pred_after,
                "corrected_flag": corrected,
                "run_id": item["run_id"],
                "checkpoint_name": item["checkpoint_name"],
            }
            per_case_rows.append(row)
            after_prob_all.append(prob_after)

            if patient_id in candidate_ids:
                after_prob_candidate.append(prob_after)

        # 全测试口径
        after_prob_all_np = np.asarray(after_prob_all, dtype=np.float64)
        metrics_before_all = _compute_metrics_from_prob(y_true_all, prob_before_all, threshold)
        metrics_after_all = _compute_metrics_from_prob(y_true_all, after_prob_all_np, threshold)

        budget_metrics_rows.append(
            {
                "scope": "all",
                "budget_k": int(budget_k),
                "n_patients": int(y_true_all.size),
                "auc_before": metrics_before_all["auc"],
                "auc_after": metrics_after_all["auc"],
                "delta_auc": float(metrics_after_all["auc"] - metrics_before_all["auc"])
                if not (np.isnan(metrics_after_all["auc"]) or np.isnan(metrics_before_all["auc"]))
                else float("nan"),
                "acc_before": metrics_before_all["acc"],
                "acc_after": metrics_after_all["acc"],
                "delta_acc": float(metrics_after_all["acc"] - metrics_before_all["acc"]),
                "f1_before": metrics_before_all["f1"],
                "f1_after": metrics_after_all["f1"],
                "delta_f1": float(metrics_after_all["f1"] - metrics_before_all["f1"]),
            }
        )

        # 候选口径
        if candidate_ids:
            after_prob_candidate_np = np.asarray(after_prob_candidate, dtype=np.float64)
            metrics_before_candidate = _compute_metrics_from_prob(
                y_true_candidate,
                prob_before_candidate,
                threshold,
            )
            metrics_after_candidate = _compute_metrics_from_prob(
                y_true_candidate,
                after_prob_candidate_np,
                threshold,
            )
            budget_metrics_rows.append(
                {
                    "scope": "candidate",
                    "budget_k": int(budget_k),
                    "n_patients": int(y_true_candidate.size),
                    "auc_before": metrics_before_candidate["auc"],
                    "auc_after": metrics_after_candidate["auc"],
                    "delta_auc": float(metrics_after_candidate["auc"] - metrics_before_candidate["auc"])
                    if not (
                        np.isnan(metrics_after_candidate["auc"])
                        or np.isnan(metrics_before_candidate["auc"])
                    )
                    else float("nan"),
                    "acc_before": metrics_before_candidate["acc"],
                    "acc_after": metrics_after_candidate["acc"],
                    "delta_acc": float(metrics_after_candidate["acc"] - metrics_before_candidate["acc"]),
                    "f1_before": metrics_before_candidate["f1"],
                    "f1_after": metrics_after_candidate["f1"],
                    "delta_f1": float(metrics_after_candidate["f1"] - metrics_before_candidate["f1"]),
                }
            )

        # 纠正率统计（基于候选集中误判病例）
        wrong_candidate_rows = [
            row
            for row in per_case_rows
            if int(row["budget_k"]) == int(budget_k)
            and str(row["patient_id"]) in set(candidate_ids)
            and int(row["pred_before"]) != int(row["y_true"])
        ]
        wrong_count = len(wrong_candidate_rows)
        corrected_count = sum(int(row["corrected_flag"]) for row in wrong_candidate_rows)
        correction_rate = float(corrected_count / wrong_count) if wrong_count > 0 else float("nan")
        correction_rows.append(
            {
                "budget_k": int(budget_k),
                "n_wrong_cases": int(wrong_count),
                "n_corrected_cases": int(corrected_count),
                "correction_rate": correction_rate,
                "avg_intervened_concepts": float(min(budget_k, n_concepts_from_data)),
            }
        )

    # summary
    summary_row = {
        "split": split,
        "n_patients_all": int(len(patient_ids)),
        "n_candidates": int(len(candidate_ids)),
        "budgets": "|".join(str(k) for k in budgets),
        "threshold": float(threshold),
        "low_conf_margin": float(low_conf_margin),
        "checkpoint": str(checkpoint_path),
        "patient_predictions_csv": str(patient_predictions_csv),
        "patient_concepts_csv": str(patient_concepts_csv),
    }

    # 输出
    _write_csv(
        output_dir / "intervention_case_list.csv",
        [
            "patient_id",
            "split",
            "y_true",
            "prob_before",
            "pred_before",
            "is_wrong",
            "is_low_confidence",
            "run_id",
            "checkpoint_name",
        ],
        candidate_rows,
    )
    _write_csv(
        output_dir / "intervention_per_case.csv",
        [
            "patient_id",
            "split",
            "y_true",
            "budget_k",
            "concepts_replaced",
            "prob_before",
            "prob_after",
            "pred_before",
            "pred_after",
            "corrected_flag",
            "run_id",
            "checkpoint_name",
        ],
        per_case_rows,
    )
    _write_csv(
        output_dir / "intervention_metrics_by_budget.csv",
        [
            "scope",
            "budget_k",
            "n_patients",
            "auc_before",
            "auc_after",
            "delta_auc",
            "acc_before",
            "acc_after",
            "delta_acc",
            "f1_before",
            "f1_after",
            "delta_f1",
        ],
        budget_metrics_rows,
    )
    _write_csv(
        output_dir / "intervention_correction_summary.csv",
        [
            "budget_k",
            "n_wrong_cases",
            "n_corrected_cases",
            "correction_rate",
            "avg_intervened_concepts",
        ],
        correction_rows,
    )
    _write_csv(
        output_dir / "intervention_summary.csv",
        list(summary_row.keys()),
        [summary_row],
    )


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run patient-level concept intervention for Habitat-CBM.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--patient-predictions-csv", type=Path, required=True)
    parser.add_argument("--patient-concepts-csv", type=Path, required=True)
    parser.add_argument("--concept-scaler-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", type=str, default="test", choices=("train", "val", "test", "all"))
    parser.add_argument("--budgets", type=str, default="1,2,4,all")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--low-confidence-margin", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--in-channels", type=int, default=35)
    parser.add_argument("--n-concepts", type=int, default=8)
    return parser


def main() -> None:
    args = _build_argparser().parse_args()
    run_intervention(
        checkpoint_path=args.checkpoint,
        patient_predictions_csv=args.patient_predictions_csv,
        patient_concepts_csv=args.patient_concepts_csv,
        concept_scaler_json=args.concept_scaler_json,
        output_dir=args.output_dir,
        split=args.split,
        budgets_text=args.budgets,
        threshold=args.threshold,
        low_conf_margin=args.low_confidence_margin,
        device_name=args.device,
        in_channels=args.in_channels,
        n_concepts=args.n_concepts,
    )
    print("Intervention outputs written to:")
    print(f"  {args.output_dir}")


if __name__ == "__main__":
    main()
