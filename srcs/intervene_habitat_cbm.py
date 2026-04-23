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

命令行参数
python habitat_CBM/repo/srcs/intervene_habitat_cbm.py \
  --checkpoint habitat_CBM/results/habitat_CBM/20260422_145024/checkpoints/stage3_best.pt \
  --patient-predictions-csv habitat_CBM/results/habitat_CBM/20260422_145024/patient_predictions_habitat_cbm_20260422_145024.csv \
  --patient-concepts-csv habitat_CBM/results/habitat_CBM/20260422_145024/patient_concepts_habitat_cbm_20260422_145024.csv \
  --concept-scaler-json habitat_CBM/dataset/concept_label/concept_scaler_stats.json \
  --output-dir habitat_CBM/results/05_intervention/20260422_145024 \
  --split test \
  --budgets 1,2,3,4,all \
  --threshold 0.4877892766605344 \
  --low-confidence-margin 0.1 \
  --intervene-scope candidates_only \
  --ranking logit_effect \
  --early-stop cross_threshold \
  --device cuda:0


"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Set, Tuple

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


_CONCEPT_COLUMN_PATTERN = re.compile(
    r"[cC](\d+)_(true_std|pred_std|abs_error_std|true_raw|pred_raw|abs_error_raw)"
)
_METRICS_SCOPE_SELECTED_SPLIT_ALL = "selected_split_all"
_METRICS_SCOPE_SELECTED_SPLIT_CANDIDATE = "selected_split_candidate"
_INTERVENE_SCOPE_ALL_PATIENTS = "all_patients"
_INTERVENE_SCOPE_CANDIDATES_ONLY = "candidates_only"
_RANKING_ABS_ERROR = "abs_error"
_RANKING_LOGIT_EFFECT = "logit_effect"
_EARLY_STOP_NONE = "none"
_EARLY_STOP_CROSS_THRESHOLD = "cross_threshold"


def _extract_concept_ids_from_row(row: Mapping[str, str]) -> List[str]:
    matched: Dict[int, str] = {}
    for key in row.keys():
        m = _CONCEPT_COLUMN_PATTERN.fullmatch(str(key))
        if m is None:
            continue
        concept_idx = int(m.group(1))
        matched[concept_idx] = f"c{concept_idx}"
    if not matched:
        raise ValueError(
            "Failed to infer concept ids from concept CSV header. Expected columns like "
            "c1_true_std/c1_pred_std or c1_true_raw/c1_pred_raw."
        )
    return [matched[idx] for idx in sorted(matched.keys())]


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


def _parse_concept_whitelist(text: str, available_concept_ids: Sequence[str]) -> List[str]:
    raw = str(text).strip()
    if raw == "" or raw.lower() in {"all", "*"}:
        return list(available_concept_ids)

    available_map = {concept_id.lower(): concept_id for concept_id in available_concept_ids}
    selected: List[str] = []
    for token in [item.strip().lower() for item in raw.split(",") if item.strip()]:
        if token not in available_map:
            raise ValueError(
                f"Unknown concept id in --concept-whitelist: {token}. "
                f"Available concept ids: {list(available_concept_ids)}"
            )
        concept_id = available_map[token]
        if concept_id not in selected:
            selected.append(concept_id)

    if not selected:
        raise ValueError("No valid concepts found in --concept-whitelist.")
    return selected


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


def _extract_label_head_weights(model: HabitatCBM, expected_n_concepts: int) -> np.ndarray:
    linear_layer = None
    for module in reversed(list(model.label_head)):
        if isinstance(module, torch.nn.Linear):
            linear_layer = module
            break
    if linear_layer is None:
        raise ValueError("Failed to locate the final linear layer in model.label_head.")

    weight = linear_layer.weight.detach().cpu().numpy().reshape(-1)
    if weight.shape != (expected_n_concepts,):
        raise ValueError(
            f"Label-head weight shape mismatch: expected {(expected_n_concepts,)}, got {weight.shape}"
        )
    return weight.astype(np.float32, copy=False)


def _rank_concept_indices(
    c_true_std: np.ndarray,
    c_pred_std: np.ndarray,
    allowed_indices: Sequence[int],
    ranking_policy: str,
    label_head_weights: np.ndarray | None,
) -> List[int]:
    delta = np.asarray(c_true_std - c_pred_std, dtype=np.float32)
    if ranking_policy == _RANKING_ABS_ERROR:
        scores = np.abs(delta)
    elif ranking_policy == _RANKING_LOGIT_EFFECT:
        if label_head_weights is None:
            raise ValueError("label_head_weights is required when ranking_policy=logit_effect.")
        scores = np.abs(delta * label_head_weights)
    else:
        raise ValueError(f"Unsupported ranking_policy: {ranking_policy}")

    score_masked = np.full_like(scores, fill_value=-np.inf, dtype=np.float32)
    for idx in allowed_indices:
        score_masked[int(idx)] = float(scores[int(idx)])

    order = np.argsort(score_masked)[::-1]
    return [int(idx) for idx in order.tolist() if np.isfinite(score_masked[int(idx)])]


def _forward_c_to_prob(model: HabitatCBM, device: torch.device, c_std: np.ndarray) -> float:
    with torch.no_grad():
        c_tensor = torch.as_tensor(c_std, dtype=torch.float32, device=device).unsqueeze(0)
        y_logit = model.forward_c_to_y(c_tensor)
        return float(torch.sigmoid(y_logit).item())


def _intervene_single_patient(
    *,
    patient_id: str,
    item: Mapping[str, object],
    concept_ids: Sequence[str],
    allowed_concept_ids: Sequence[str],
    candidate_id_set: Set[str],
    intervene_scope: str,
    ranking_policy: str,
    label_head_weights: np.ndarray | None,
    early_stop_mode: str,
    budget_k: int,
    threshold: float,
    model: HabitatCBM,
    device: torch.device,
) -> Dict[str, object]:
    y_true = int(item["y_true"])
    pred_before = int(item["pred_before"])
    prob_before = float(item["prob_before"])

    should_intervene = (
        intervene_scope == _INTERVENE_SCOPE_ALL_PATIENTS
        or patient_id in candidate_id_set
    )
    if not should_intervene:
        return {
            "prob_after": prob_before,
            "pred_after": pred_before,
            "concepts_replaced": "",
            "n_concepts_replaced": 0,
            "corrected_flag": 0,
            "intervention_applied": 0,
            "stop_reason": "not_selected_by_scope",
        }

    allowed_concept_id_set = set(allowed_concept_ids)
    allowed_indices = [
        idx for idx, concept_id in enumerate(concept_ids) if concept_id in allowed_concept_id_set
    ]
    if not allowed_indices:
        return {
            "prob_after": prob_before,
            "pred_after": pred_before,
            "concepts_replaced": "",
            "n_concepts_replaced": 0,
            "corrected_flag": 0,
            "intervention_applied": 0,
            "stop_reason": "no_allowed_concepts",
        }

    if early_stop_mode == _EARLY_STOP_CROSS_THRESHOLD and pred_before == y_true:
        return {
            "prob_after": prob_before,
            "pred_after": pred_before,
            "concepts_replaced": "",
            "n_concepts_replaced": 0,
            "corrected_flag": 0,
            "intervention_applied": 0,
            "stop_reason": "already_correct_before",
        }

    c_true_std = np.asarray(item["c_true_std"], dtype=np.float32)
    c_pred_std = np.asarray(item["c_pred_std"], dtype=np.float32)
    ranked_indices = _rank_concept_indices(
        c_true_std=c_true_std,
        c_pred_std=c_pred_std,
        allowed_indices=allowed_indices,
        ranking_policy=ranking_policy,
        label_head_weights=label_head_weights,
    )
    if not ranked_indices:
        return {
            "prob_after": prob_before,
            "pred_after": pred_before,
            "concepts_replaced": "",
            "n_concepts_replaced": 0,
            "corrected_flag": 0,
            "intervention_applied": 0,
            "stop_reason": "no_ranked_concepts",
        }

    max_steps = min(int(budget_k), len(ranked_indices))
    c_after = c_pred_std.copy()
    replaced_indices: List[int] = []
    prob_after = prob_before
    pred_after = pred_before
    stop_reason = "budget_exhausted"

    for concept_idx in ranked_indices[:max_steps]:
        c_after[int(concept_idx)] = c_true_std[int(concept_idx)]
        prob_after = _forward_c_to_prob(model=model, device=device, c_std=c_after)
        pred_after = int(prob_after >= threshold)
        replaced_indices.append(int(concept_idx))

        if early_stop_mode == _EARLY_STOP_CROSS_THRESHOLD and pred_after == y_true:
            stop_reason = "crossed_to_correct_side"
            break

    corrected_flag = int(pred_before != y_true and pred_after == y_true)
    if not replaced_indices:
        stop_reason = "no_change_applied"

    return {
        "prob_after": prob_after,
        "pred_after": pred_after,
        "concepts_replaced": "|".join(concept_ids[idx] for idx in replaced_indices),
        "n_concepts_replaced": len(replaced_indices),
        "corrected_flag": corrected_flag,
        "intervention_applied": int(bool(replaced_indices)),
        "stop_reason": stop_reason,
    }


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
    intervene_scope: str,
    ranking_policy: str,
    concept_whitelist_text: str,
    early_stop_mode: str,
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
    label_head_weights = _extract_label_head_weights(model, expected_n_concepts=n_concepts_from_data)
    allowed_concept_ids = _parse_concept_whitelist(
        concept_whitelist_text,
        available_concept_ids=concept_ids,
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
    candidate_id_set = set(candidate_ids)
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
            patient_result = _intervene_single_patient(
                patient_id=patient_id,
                item=item,
                concept_ids=concept_ids,
                allowed_concept_ids=allowed_concept_ids,
                candidate_id_set=candidate_id_set,
                intervene_scope=intervene_scope,
                ranking_policy=ranking_policy,
                label_head_weights=label_head_weights,
                early_stop_mode=early_stop_mode,
                budget_k=budget_k,
                threshold=threshold,
                model=model,
                device=device,
            )

            pred_before = int(item["pred_before"])
            pred_after = int(patient_result["pred_after"])
            prob_after = float(patient_result["prob_after"])

            row = {
                "patient_id": patient_id,
                "split": item["split"],
                "y_true": int(item["y_true"]),
                "budget_k": int(budget_k),
                "concepts_replaced": str(patient_result["concepts_replaced"]),
                "n_concepts_replaced": int(patient_result["n_concepts_replaced"]),
                "prob_before": float(item["prob_before"]),
                "prob_after": prob_after,
                "pred_before": pred_before,
                "pred_after": pred_after,
                "corrected_flag": int(patient_result["corrected_flag"]),
                "intervention_applied": int(patient_result["intervention_applied"]),
                "stop_reason": str(patient_result["stop_reason"]),
                "run_id": item["run_id"],
                "checkpoint_name": item["checkpoint_name"],
            }
            per_case_rows.append(row)
            after_prob_all.append(prob_after)

            if patient_id in candidate_ids:
                after_prob_candidate.append(prob_after)

        # 当前选中 split 内的全部患者口径
        after_prob_all_np = np.asarray(after_prob_all, dtype=np.float64)
        metrics_before_all = _compute_metrics_from_prob(y_true_all, prob_before_all, threshold)
        metrics_after_all = _compute_metrics_from_prob(y_true_all, after_prob_all_np, threshold)

        budget_metrics_rows.append(
            {
                "selected_split": split,
                "scope": _METRICS_SCOPE_SELECTED_SPLIT_ALL,
                "intervene_scope": intervene_scope,
                "ranking_policy": ranking_policy,
                "early_stop_mode": early_stop_mode,
                "concept_whitelist": "|".join(allowed_concept_ids),
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

        # 当前选中 split 内的候选病例口径
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
                    "selected_split": split,
                    "scope": _METRICS_SCOPE_SELECTED_SPLIT_CANDIDATE,
                    "intervene_scope": intervene_scope,
                    "ranking_policy": ranking_policy,
                    "early_stop_mode": early_stop_mode,
                    "concept_whitelist": "|".join(allowed_concept_ids),
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
            and str(row["patient_id"]) in candidate_id_set
            and int(row["pred_before"]) != int(row["y_true"])
        ]
        wrong_count = len(wrong_candidate_rows)
        corrected_count = sum(int(row["corrected_flag"]) for row in wrong_candidate_rows)
        correction_rate = float(corrected_count / wrong_count) if wrong_count > 0 else float("nan")
        avg_actual_intervened_wrong = (
            float(np.mean([int(row["n_concepts_replaced"]) for row in wrong_candidate_rows]))
            if wrong_candidate_rows
            else float("nan")
        )
        avg_actual_intervened_candidates = (
            float(
                np.mean(
                    [
                        int(row["n_concepts_replaced"])
                        for row in per_case_rows
                        if int(row["budget_k"]) == int(budget_k)
                        and str(row["patient_id"]) in candidate_id_set
                    ]
                )
            )
            if candidate_id_set
            else float("nan")
        )
        correction_rows.append(
            {
                "selected_split": split,
                "intervene_scope": intervene_scope,
                "ranking_policy": ranking_policy,
                "early_stop_mode": early_stop_mode,
                "concept_whitelist": "|".join(allowed_concept_ids),
                "budget_k": int(budget_k),
                "n_wrong_cases": int(wrong_count),
                "n_corrected_cases": int(corrected_count),
                "correction_rate": correction_rate,
                "avg_planned_budget_concepts": float(min(budget_k, n_concepts_from_data)),
                "avg_actual_intervened_concepts_wrong_cases": avg_actual_intervened_wrong,
                "avg_actual_intervened_concepts_all_candidates": avg_actual_intervened_candidates,
            }
        )

    # summary
    summary_row = {
        "selected_split": split,
        "n_patients_all": int(len(patient_ids)),
        "n_candidates": int(len(candidate_ids)),
        "budgets": "|".join(str(k) for k in budgets),
        "threshold": float(threshold),
        "low_conf_margin": float(low_conf_margin),
        "intervene_scope": intervene_scope,
        "ranking_policy": ranking_policy,
        "early_stop_mode": early_stop_mode,
        "concept_whitelist": "|".join(allowed_concept_ids),
        "n_allowed_concepts": int(len(allowed_concept_ids)),
        "metrics_scope_selected_split_all": _METRICS_SCOPE_SELECTED_SPLIT_ALL,
        "metrics_scope_selected_split_candidate": _METRICS_SCOPE_SELECTED_SPLIT_CANDIDATE,
        "metrics_scope_note": (
            "selected_split_all means all patients within the chosen --split; "
            "use --split all to aggregate train+val+test."
        ),
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
            "n_concepts_replaced",
            "prob_before",
            "prob_after",
            "pred_before",
            "pred_after",
            "corrected_flag",
            "intervention_applied",
            "stop_reason",
            "run_id",
            "checkpoint_name",
        ],
        per_case_rows,
    )
    _write_csv(
        output_dir / "intervention_metrics_by_budget.csv",
        [
            "selected_split",
            "scope",
            "intervene_scope",
            "ranking_policy",
            "early_stop_mode",
            "concept_whitelist",
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
            "selected_split",
            "intervene_scope",
            "ranking_policy",
            "early_stop_mode",
            "concept_whitelist",
            "budget_k",
            "n_wrong_cases",
            "n_corrected_cases",
            "correction_rate",
            "avg_planned_budget_concepts",
            "avg_actual_intervened_concepts_wrong_cases",
            "avg_actual_intervened_concepts_all_candidates",
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
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=("train", "val", "test", "all"),
        help="Which split to intervene on. Use all only when you explicitly want train+val+test combined.",
    )
    parser.add_argument("--budgets", type=str, default="1,2,4,all")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.4877892766605344,
        help="Probability threshold used to recompute before/after labels. Keep it aligned with the formal evaluation threshold.",
    )
    parser.add_argument("--low-confidence-margin", type=float, default=0.1)
    parser.add_argument(
        "--intervene-scope",
        type=str,
        default=_INTERVENE_SCOPE_CANDIDATES_ONLY,
        choices=(_INTERVENE_SCOPE_ALL_PATIENTS, _INTERVENE_SCOPE_CANDIDATES_ONLY),
        help="Who will actually be modified. candidates_only is safer because non-candidate patients stay unchanged.",
    )
    parser.add_argument(
        "--ranking",
        type=str,
        default=_RANKING_LOGIT_EFFECT,
        choices=(_RANKING_ABS_ERROR, _RANKING_LOGIT_EFFECT),
        help="How to choose which concept to fix first. logit_effect prefers concepts that matter more to the final prediction.",
    )
    parser.add_argument(
        "--concept-whitelist",
        type=str,
        default="",
        help="Optional comma-separated concept ids to allow intervention on, e.g. c3 or c3,c6. Empty means all available concepts.",
    )
    parser.add_argument(
        "--early-stop",
        type=str,
        default=_EARLY_STOP_CROSS_THRESHOLD,
        choices=(_EARLY_STOP_NONE, _EARLY_STOP_CROSS_THRESHOLD),
        help="Whether to stop once the patient reaches the correct side of the threshold. cross_threshold also leaves already-correct patients unchanged.",
    )
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
        intervene_scope=args.intervene_scope,
        ranking_policy=args.ranking,
        concept_whitelist_text=args.concept_whitelist,
        early_stop_mode=args.early_stop,
        device_name=args.device,
        in_channels=args.in_channels,
        n_concepts=args.n_concepts,
    )
    print("Intervention outputs written to:")
    print(f"  {args.output_dir}")


if __name__ == "__main__":
    main()
