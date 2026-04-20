#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Habitat-CBM 推理/验证脚本。

能力：
1. 兼容旧版工具函数（predict_batch / intervention / 聚合）；
2. 提供 checkpoint 级正式评估：患者级主任务输出 + 患者级概念输出 + CSV/JSON 导出。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except Exception:  # pragma: no cover - matplotlib 缺失时不影响 CSV/JSON 导出
    HAS_MATPLOTLIB = False

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.habitat_CBM import HabitatCBM
from srcs.data_loader_habitat_CBM import (
    ConceptScaler,
    build_habitat_cbm_dataloaders,
    build_habitat_cbm_datasets,
    concept_names_to_columns,
    load_concept_scaler,
    resolve_concept_names,
)
from srcs.monai_augmentation import MonaiAugmentConfig, build_monai_block_transforms

MODEL_NAME = "habitat_cbm"


def _resolve_model_dropouts(model_cfg: Mapping[str, object]) -> Tuple[float, float]:
    shared_dropout = model_cfg.get("dropout_p", None)
    if shared_dropout is not None:
        shared = float(shared_dropout)
        concept_dropout = float(model_cfg.get("concept_dropout_p", shared))
        label_dropout = float(model_cfg.get("label_dropout_p", shared))
    else:
        concept_dropout = float(model_cfg.get("concept_dropout_p", 0.3))
        label_dropout = float(model_cfg.get("label_dropout_p", 0.1))
    return concept_dropout, label_dropout


def _validate_concept_tensor(
    c: torch.Tensor,
    n_concepts: int,
    tensor_name: str,
) -> None:
    if c.ndim != 2:
        raise ValueError(
            f"{tensor_name} must be 2D [B, {n_concepts}], got shape {tuple(c.shape)}."
        )
    if c.shape[1] != n_concepts:
        raise ValueError(
            f"{tensor_name} second dim must be {n_concepts}, got {c.shape[1]}."
        )


@torch.no_grad()
def predict_batch(model: HabitatCBM, x: torch.Tensor) -> dict[str, torch.Tensor]:
    model.eval()
    out = model.forward_x_to_cy(x)
    y_prob = torch.sigmoid(out["y_logit"])
    return {
        "z": out["z"],
        "c_hat": out["c_hat"],
        "y_logit": out["y_logit"],
        "y_prob": y_prob,
    }


def prepare_intervention_order(
    c_pred: torch.Tensor,
    c_true_std: torch.Tensor,
    order: torch.Tensor | None = None,
) -> torch.Tensor:
    _validate_concept_tensor(c_pred, c_pred.shape[1], "c_pred")
    _validate_concept_tensor(c_true_std, c_pred.shape[1], "c_true_std")
    if c_pred.shape != c_true_std.shape:
        raise ValueError(
            "Shape mismatch between c_pred and c_true_std: "
            f"{tuple(c_pred.shape)} vs {tuple(c_true_std.shape)}."
        )

    batch_size = c_pred.shape[0]
    n_concepts = c_pred.shape[1]
    device = c_pred.device

    if order is None:
        abs_error = (c_pred - c_true_std).abs()
        return torch.argsort(abs_error, dim=1, descending=True)

    order = order.to(device=device, dtype=torch.long)
    if order.ndim == 1:
        if order.numel() != n_concepts:
            raise ValueError(
                f"1D order must have length {n_concepts}, got {order.numel()}."
            )
        order = order.unsqueeze(0).expand(batch_size, -1)
    elif order.ndim == 2:
        if order.shape[1] != n_concepts:
            raise ValueError(
                "2D order must have shape [B, n_concepts]. "
                f"Expected second dim {n_concepts}, got {order.shape[1]}."
            )
        if order.shape[0] == 1 and batch_size > 1:
            order = order.expand(batch_size, -1)
        elif order.shape[0] != batch_size:
            raise ValueError(
                "2D order batch size mismatch. "
                f"Expected {batch_size}, got {order.shape[0]}."
            )
    else:
        raise ValueError(f"order must be 1D or 2D, got shape {tuple(order.shape)}.")

    if torch.any(order < 0) or torch.any(order >= n_concepts):
        raise ValueError("order contains out-of-range concept indices.")

    expected = torch.arange(n_concepts, device=device).unsqueeze(0).expand_as(order)
    if not torch.equal(torch.sort(order, dim=1).values, expected):
        raise ValueError("Each row in order must be a permutation of [0, n_concepts-1].")

    return order


@torch.no_grad()
def forward_with_intervention(
    model: HabitatCBM,
    x: torch.Tensor,
    c_true_std: torch.Tensor,
    k: int,
    order: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    if not isinstance(k, int):
        raise TypeError(f"k must be int, got {type(k).__name__}.")
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}.")
    k = min(k, model.n_concepts)

    model.eval()
    out = model.forward_x_to_cy(x)
    c_pred = out["c_hat"]
    y_logit_before = out["y_logit"]

    _validate_concept_tensor(c_true_std, model.n_concepts, "c_true_std")
    c_true_std = c_true_std.to(device=c_pred.device, dtype=c_pred.dtype)
    if c_true_std.shape[0] != c_pred.shape[0]:
        raise ValueError(
            "Batch size mismatch between x and c_true_std: "
            f"{c_pred.shape[0]} vs {c_true_std.shape[0]}."
        )

    final_order = prepare_intervention_order(c_pred=c_pred, c_true_std=c_true_std, order=order)
    selected_indices = final_order[:, :k] if k > 0 else final_order[:, :0]

    intervention_mask = torch.zeros_like(c_pred, dtype=torch.bool)
    if k > 0:
        intervention_mask.scatter_(1, selected_indices, True)

    c_after = torch.where(intervention_mask, c_true_std, c_pred)
    y_logit_after = model.forward_c_to_y(c_after)

    return {
        "c_pred": c_pred,
        "c_after": c_after,
        "y_logit_before": y_logit_before,
        "y_logit_after": y_logit_after,
        "y_prob_before": torch.sigmoid(y_logit_before),
        "y_prob_after": torch.sigmoid(y_logit_after),
        "intervention_mask": intervention_mask,
        "intervention_indices": selected_indices,
        "order": final_order,
    }


def aggregate_patient_probabilities(
    patient_ids: list[str],
    probs: list[float],
    topk: int = 0,
) -> dict[str, float]:
    if len(patient_ids) != len(probs):
        raise ValueError(
            "Length mismatch between patient_ids and probs: "
            f"{len(patient_ids)} vs {len(probs)}."
        )
    if topk < 0:
        raise ValueError(f"topk must be >= 0, got {topk}.")

    buckets: dict[str, list[float]] = {}
    for pid, prob in zip(patient_ids, probs):
        buckets.setdefault(pid, []).append(float(prob))

    aggregated: dict[str, float] = {}
    for pid, values in buckets.items():
        if not values:
            raise RuntimeError(f"Patient {pid} has no probability values.")
        if topk > 0 and len(values) > topk:
            values = sorted(values, key=lambda p: abs(p - 0.5), reverse=True)[:topk]
        aggregated[pid] = float(sum(values) / len(values))
    return aggregated


def aggregate_patient_concepts(
    patient_ids: list[str],
    concepts: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if concepts.ndim != 2:
        raise ValueError(f"concepts must be 2D [N, K], got shape {tuple(concepts.shape)}.")
    if len(patient_ids) != concepts.shape[0]:
        raise ValueError(
            "Length mismatch between patient_ids and concepts rows: "
            f"{len(patient_ids)} vs {concepts.shape[0]}"
        )

    sums: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = {}
    for idx, pid in enumerate(patient_ids):
        row = concepts[idx]
        if pid not in sums:
            sums[pid] = row.clone()
            counts[pid] = 1
        else:
            sums[pid] = sums[pid] + row
            counts[pid] += 1

    means: dict[str, torch.Tensor] = {}
    for pid, total in sums.items():
        means[pid] = total / float(counts[pid])
    return means


def _safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def compute_patient_metrics(patient_rows: Sequence[Mapping[str, object]]) -> Dict[str, float]:
    y_true = np.asarray([int(row["y_true"]) for row in patient_rows], dtype=np.int64)
    y_prob = np.asarray([float(row["prob_idh_mut"]) for row in patient_rows], dtype=np.float64)
    y_pred = np.asarray([int(row["pred_label"]) for row in patient_rows], dtype=np.int64)

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


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _save_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _aggregate_patient_rows(
    block_rows: Sequence[Mapping[str, object]],
    split: str,
    run_id: str,
    checkpoint_name: str,
    threshold: float,
    topk_pool: int,
) -> List[Dict[str, object]]:
    patient_ids = [str(row["patient_id"]) for row in block_rows]
    probs = [float(row["prob_idh_mut"]) for row in block_rows]
    prob_map = aggregate_patient_probabilities(patient_ids=patient_ids, probs=probs, topk=topk_pool)

    y_true_map: Dict[str, int] = {}
    for row in block_rows:
        pid = str(row["patient_id"])
        y_true = int(row["y_true"])
        if pid not in y_true_map:
            y_true_map[pid] = y_true
        elif y_true_map[pid] != y_true:
            raise ValueError(f"Inconsistent y_true across blocks for patient {pid}")

    patient_rows: List[Dict[str, object]] = []
    for pid in sorted(prob_map.keys()):
        prob = float(prob_map[pid])
        patient_rows.append(
            {
                "patient_id": pid,
                "split": split,
                "y_true": y_true_map[pid],
                "prob_idh_mut": prob,
                "pred_label": int(prob >= threshold),
                "run_id": run_id,
                "checkpoint_name": checkpoint_name,
            }
        )
    return patient_rows


def _aggregate_patient_concept_rows(
    block_rows: Sequence[Mapping[str, object]],
    scaler: ConceptScaler,
    split: str,
    run_id: str,
    checkpoint_name: str,
) -> List[Dict[str, object]]:
    patient_ids = [str(row["patient_id"]) for row in block_rows]
    concept_pred_std = torch.as_tensor(
        np.asarray([row["c_pred_std"] for row in block_rows], dtype=np.float32), dtype=torch.float32
    )
    concept_true_std = torch.as_tensor(
        np.asarray([row["c_true_std"] for row in block_rows], dtype=np.float32), dtype=torch.float32
    )
    concept_true_raw = torch.as_tensor(
        np.asarray([row["c_true_raw"] for row in block_rows], dtype=np.float32), dtype=torch.float32
    )

    pred_std_map = aggregate_patient_concepts(patient_ids=patient_ids, concepts=concept_pred_std)
    true_std_map = aggregate_patient_concepts(patient_ids=patient_ids, concepts=concept_true_std)
    true_raw_map = aggregate_patient_concepts(patient_ids=patient_ids, concepts=concept_true_raw)

    y_true_map: Dict[str, int] = {}
    for row in block_rows:
        pid = str(row["patient_id"])
        y_true = int(row["y_true"])
        if pid not in y_true_map:
            y_true_map[pid] = y_true
        elif y_true_map[pid] != y_true:
            raise ValueError(f"Inconsistent y_true across blocks for patient {pid}")

    patient_rows: List[Dict[str, object]] = []
    for pid in sorted(pred_std_map.keys()):
        c_pred_std = pred_std_map[pid].detach().cpu().numpy().astype(np.float32)
        c_true_std = true_std_map[pid].detach().cpu().numpy().astype(np.float32)
        c_true_raw = true_raw_map[pid].detach().cpu().numpy().astype(np.float32)
        c_pred_raw = scaler.destandardize(c_pred_std)

        row: Dict[str, object] = {
            "patient_id": pid,
            "split": split,
            "y_true": y_true_map[pid],
            "run_id": run_id,
            "checkpoint_name": checkpoint_name,
        }
        for idx, concept_name in enumerate(scaler.concept_names):
            row[f"{concept_name}_true_std"] = float(c_true_std[idx])
            row[f"{concept_name}_pred_std"] = float(c_pred_std[idx])
            row[f"{concept_name}_abs_error_std"] = float(abs(c_pred_std[idx] - c_true_std[idx]))
            row[f"{concept_name}_true_raw"] = float(c_true_raw[idx])
            row[f"{concept_name}_pred_raw"] = float(c_pred_raw[idx])
            row[f"{concept_name}_abs_error_raw"] = float(abs(c_pred_raw[idx] - c_true_raw[idx]))
        patient_rows.append(row)

    return patient_rows


@torch.no_grad()
def evaluate_split(
    model: HabitatCBM,
    dataloader,
    scaler: ConceptScaler,
    split: str,
    run_id: str,
    checkpoint_name: str,
    threshold: float,
    topk_pool: int,
    device: torch.device,
) -> Dict[str, object]:
    model.eval()

    block_rows: List[Dict[str, object]] = []
    total_bce = 0.0
    total_samples = 0

    for batch in dataloader:
        images = batch["image"].to(device=device, dtype=torch.float32, non_blocking=True)
        y_true = batch["label"].to(device=device, dtype=torch.float32, non_blocking=True)
        c_true_std = batch["concept_true_std"].to(device=device, dtype=torch.float32, non_blocking=True)

        out = model.forward_x_to_cy(images)
        y_logit = out["y_logit"].squeeze(1)
        y_prob = torch.sigmoid(y_logit)
        c_pred_std = out["c_hat"]

        # 评估损失仅用于诊断，不参与模型选择。
        bce = F.binary_cross_entropy_with_logits(y_logit, y_true)
        batch_size = int(y_true.shape[0])
        total_bce += float(bce.item()) * batch_size
        total_samples += batch_size

        c_true_raw = batch["concept_true_raw"].to(dtype=torch.float32)

        patient_ids = batch["patient_id"]
        slice_indices = batch["slice_index"]
        for idx in range(batch_size):
            block_rows.append(
                {
                    "patient_id": str(patient_ids[idx]),
                    "slice_index": int(slice_indices[idx]),
                    "y_true": int(y_true[idx].item()),
                    "prob_idh_mut": float(y_prob[idx].item()),
                    "c_pred_std": c_pred_std[idx].detach().cpu().numpy().astype(np.float32),
                    "c_true_std": c_true_std[idx].detach().cpu().numpy().astype(np.float32),
                    "c_true_raw": c_true_raw[idx].detach().cpu().numpy().astype(np.float32),
                }
            )

    patient_pred_rows = _aggregate_patient_rows(
        block_rows=block_rows,
        split=split,
        run_id=run_id,
        checkpoint_name=checkpoint_name,
        threshold=threshold,
        topk_pool=topk_pool,
    )
    patient_concept_rows = _aggregate_patient_concept_rows(
        block_rows=block_rows,
        scaler=scaler,
        split=split,
        run_id=run_id,
        checkpoint_name=checkpoint_name,
    )

    metrics = compute_patient_metrics(patient_pred_rows)
    return {
        "split": split,
        "loss_bce_block": total_bce / max(total_samples, 1),
        "num_blocks": int(total_samples),
        "num_patients": int(len(patient_pred_rows)),
        "patient_prediction_rows": patient_pred_rows,
        "patient_concept_rows": patient_concept_rows,
        "metrics": metrics,
    }


def _extract_binary_arrays(patient_rows: Sequence[Mapping[str, object]]) -> Tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray([int(row["y_true"]) for row in patient_rows], dtype=np.int64)
    y_prob = np.asarray([float(row["prob_idh_mut"]) for row in patient_rows], dtype=np.float64)
    return y_true, y_prob


def _plot_roc_curve(
    patient_rows: Sequence[Mapping[str, object]],
    path: Path,
    title: str,
    dpi: int,
) -> bool:
    y_true, y_prob = _extract_binary_arrays(patient_rows)
    if len(np.unique(y_true)) < 2:
        return False
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_value = _safe_auc(y_true, y_prob)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 5.0), dpi=dpi)
    ax.plot(fpr, tpr, color="#1f77b4", linewidth=2.0, label=f"AUC = {auc_value:.3f}")
    ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="#999999", linewidth=1.2)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.2, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_pr_curve(
    patient_rows: Sequence[Mapping[str, object]],
    path: Path,
    title: str,
    dpi: int,
) -> bool:
    y_true, y_prob = _extract_binary_arrays(patient_rows)
    if len(np.unique(y_true)) < 2:
        return False
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = float(average_precision_score(y_true, y_prob))

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 5.0), dpi=dpi)
    ax.plot(recall, precision, color="#ff7f0e", linewidth=2.0, label=f"AP = {ap:.3f}")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="lower left")
    ax.grid(alpha=0.2, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_confusion_matrix(
    patient_rows: Sequence[Mapping[str, object]],
    path: Path,
    title: str,
    dpi: int,
) -> None:
    y_true = np.asarray([int(row["y_true"]) for row in patient_rows], dtype=np.int64)
    y_pred = np.asarray([int(row["pred_label"]) for row in patient_rows], dtype=np.int64)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    matrix = np.asarray([[tn, fp], [fn, tp]], dtype=np.int64)
    labels = (("TN", "FP"), ("FN", "TP"))

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.0, 4.5), dpi=dpi)
    image = ax.imshow(matrix, cmap="Blues")
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks([0, 1], labels=["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1], labels=["True 0", "True 1"])
    ax.set_title(title)
    max_value = max(int(matrix.max()), 1)
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = int(matrix[row_idx, col_idx])
            text_color = "white" if value > max_value / 2 else "black"
            ax.text(
                col_idx,
                row_idx,
                f"{labels[row_idx][col_idx]}\n{value}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=10,
            )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _plot_calibration_curve(
    patient_rows: Sequence[Mapping[str, object]],
    path: Path,
    title: str,
    dpi: int,
) -> bool:
    y_true, y_prob = _extract_binary_arrays(patient_rows)
    if len(np.unique(y_true)) < 2:
        return False
    n_bins = max(4, min(10, int(len(y_true) / 2)))
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    if len(prob_true) == 0:
        return False

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 5.0), dpi=dpi)
    ax.plot(prob_pred, prob_true, marker="o", linewidth=1.8, color="#2ca02c", label="Model")
    ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="#999999", linewidth=1.2, label="Perfect")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.grid(alpha=0.2, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_probability_distribution(
    patient_rows: Sequence[Mapping[str, object]],
    path: Path,
    title: str,
    dpi: int,
) -> bool:
    y_true, y_prob = _extract_binary_arrays(patient_rows)
    pos = y_prob[y_true == 1]
    neg = y_prob[y_true == 0]
    if pos.size == 0 and neg.size == 0:
        return False

    bins = np.linspace(0.0, 1.0, 21)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.0, 4.5), dpi=dpi)
    if neg.size > 0:
        ax.hist(neg, bins=bins, alpha=0.55, color="#1f77b4", label=f"True 0 (n={neg.size})", density=False)
    if pos.size > 0:
        ax.hist(pos, bins=bins, alpha=0.55, color="#d62728", label=f"True 1 (n={pos.size})", density=False)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Predicted Probability (IDH mutant)")
    ax.set_ylabel("Patient Count")
    ax.set_title(title)
    ax.legend(loc="upper center")
    ax.grid(axis="y", alpha=0.2, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return True


def _extract_concept_ids(patient_concept_rows: Sequence[Mapping[str, object]]) -> List[str]:
    if not patient_concept_rows:
        return []
    matched: List[Tuple[int, str]] = []
    for key in patient_concept_rows[0].keys():
        m = re.fullmatch(r"c(\d+)_abs_error_std", str(key))
        if m is None:
            continue
        matched.append((int(m.group(1)), f"c{int(m.group(1))}"))
    matched.sort(key=lambda item: item[0])
    return [item[1] for item in matched]


def _plot_concept_abs_error_boxplot(
    patient_concept_rows: Sequence[Mapping[str, object]],
    path: Path,
    title: str,
    dpi: int,
) -> bool:
    concept_ids = _extract_concept_ids(patient_concept_rows)
    if not concept_ids:
        return False
    data: List[np.ndarray] = []
    for concept_id in concept_ids:
        data.append(
            np.asarray(
                [float(row[f"{concept_id}_abs_error_std"]) for row in patient_concept_rows],
                dtype=np.float64,
            )
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.0, 4.8), dpi=dpi)
    ax.boxplot(data, labels=concept_ids, showmeans=True)
    ax.set_xlabel("Concept")
    ax.set_ylabel("Absolute Error (std scale)")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.2, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_concept_mae_ranking(
    patient_concept_rows: Sequence[Mapping[str, object]],
    path: Path,
    title: str,
    dpi: int,
) -> bool:
    concept_ids = _extract_concept_ids(patient_concept_rows)
    if not concept_ids:
        return False

    rows: List[Tuple[str, float]] = []
    for concept_id in concept_ids:
        mae = float(
            np.mean([float(row[f"{concept_id}_abs_error_std"]) for row in patient_concept_rows], dtype=np.float64)
        )
        rows.append((concept_id, mae))
    rows.sort(key=lambda item: item[1], reverse=True)

    labels = [item[0] for item in rows]
    values = [item[1] for item in rows]

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.0, 4.8), dpi=dpi)
    bars = ax.bar(labels, values, color="#ff7f0e", alpha=0.85)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2.0, value, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("Concept")
    ax.set_ylabel("MAE (std scale)")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.2, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_concept_true_vs_pred_grid(
    patient_concept_rows: Sequence[Mapping[str, object]],
    path: Path,
    title: str,
    dpi: int,
) -> bool:
    concept_ids = _extract_concept_ids(patient_concept_rows)
    if not concept_ids:
        return False

    n_concepts = len(concept_ids)
    cols = min(4, n_concepts)
    rows = int(math.ceil(n_concepts / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.6 * rows), dpi=dpi)
    if rows == 1 and cols == 1:
        axes_list = [axes]
    elif rows == 1:
        axes_list = list(axes)
    elif cols == 1:
        axes_list = list(axes)
    else:
        axes_list = [ax for row_axes in axes for ax in row_axes]

    for idx, concept_id in enumerate(concept_ids):
        ax = axes_list[idx]
        x_true = np.asarray([float(row[f"{concept_id}_true_std"]) for row in patient_concept_rows], dtype=np.float64)
        y_pred = np.asarray([float(row[f"{concept_id}_pred_std"]) for row in patient_concept_rows], dtype=np.float64)
        ax.scatter(x_true, y_pred, s=18, alpha=0.75, color="#1f77b4")
        lim_min = float(min(float(np.min(x_true)), float(np.min(y_pred))))
        lim_max = float(max(float(np.max(x_true)), float(np.max(y_pred))))
        if lim_min == lim_max:
            lim_min -= 1.0
            lim_max += 1.0
        ax.plot([lim_min, lim_max], [lim_min, lim_max], linestyle="--", color="#999999", linewidth=1.0)
        ax.set_title(concept_id)
        ax.set_xlabel("True (std)")
        ax.set_ylabel("Pred (std)")
        ax.grid(alpha=0.2, linewidth=0.5)

    for idx in range(n_concepts, len(axes_list)):
        axes_list[idx].axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return True


def export_evaluation_figures(
    split_prediction_rows: Mapping[str, Sequence[Mapping[str, object]]],
    split_concept_rows: Mapping[str, Sequence[Mapping[str, object]]],
    output_dir: Path,
    run_id: str,
    include_splits: Sequence[str],
    dpi: int,
) -> Dict[str, Dict[str, str]]:
    if not HAS_MATPLOTLIB:
        return {}

    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    records: Dict[str, Dict[str, str]] = {}

    for split in include_splits:
        pred_rows = list(split_prediction_rows.get(split, []))
        if not pred_rows:
            continue
        concept_rows = list(split_concept_rows.get(split, []))
        split_records: Dict[str, str] = {}

        roc_path = figure_dir / f"roc_curve_{MODEL_NAME}_{split}_{run_id}.png"
        if _plot_roc_curve(pred_rows, roc_path, f"{MODEL_NAME.upper()} ROC ({split})", dpi=dpi):
            split_records["roc_curve"] = str(roc_path)

        pr_path = figure_dir / f"pr_curve_{MODEL_NAME}_{split}_{run_id}.png"
        if _plot_pr_curve(pred_rows, pr_path, f"{MODEL_NAME.upper()} PR ({split})", dpi=dpi):
            split_records["pr_curve"] = str(pr_path)

        cm_path = figure_dir / f"confusion_matrix_{MODEL_NAME}_{split}_{run_id}.png"
        _plot_confusion_matrix(pred_rows, cm_path, f"{MODEL_NAME.upper()} Confusion Matrix ({split})", dpi=dpi)
        split_records["confusion_matrix"] = str(cm_path)

        cal_path = figure_dir / f"calibration_curve_{MODEL_NAME}_{split}_{run_id}.png"
        if _plot_calibration_curve(pred_rows, cal_path, f"{MODEL_NAME.upper()} Calibration ({split})", dpi=dpi):
            split_records["calibration_curve"] = str(cal_path)

        prob_hist_path = figure_dir / f"probability_distribution_{MODEL_NAME}_{split}_{run_id}.png"
        if _plot_probability_distribution(
            pred_rows,
            prob_hist_path,
            f"{MODEL_NAME.upper()} Probability Distribution ({split})",
            dpi=dpi,
        ):
            split_records["probability_distribution"] = str(prob_hist_path)

        if concept_rows:
            concept_box_path = figure_dir / f"concept_abs_error_boxplot_{MODEL_NAME}_{split}_{run_id}.png"
            if _plot_concept_abs_error_boxplot(
                concept_rows,
                concept_box_path,
                f"{MODEL_NAME.upper()} Concept Abs Error Boxplot ({split})",
                dpi=dpi,
            ):
                split_records["concept_abs_error_boxplot"] = str(concept_box_path)

            concept_rank_path = figure_dir / f"concept_mae_ranking_{MODEL_NAME}_{split}_{run_id}.png"
            if _plot_concept_mae_ranking(
                concept_rows,
                concept_rank_path,
                f"{MODEL_NAME.upper()} Concept MAE Ranking ({split})",
                dpi=dpi,
            ):
                split_records["concept_mae_ranking"] = str(concept_rank_path)

            concept_scatter_path = figure_dir / f"concept_true_vs_pred_{MODEL_NAME}_{split}_{run_id}.png"
            if _plot_concept_true_vs_pred_grid(
                concept_rows,
                concept_scatter_path,
                f"{MODEL_NAME.upper()} Concept True vs Pred ({split})",
                dpi=dpi,
            ):
                split_records["concept_true_vs_pred"] = str(concept_scatter_path)

        if split_records:
            records[split] = split_records

    return records


def export_evaluation_outputs(
    split_outputs: Mapping[str, Mapping[str, object]],
    output_dir: Path,
    run_id: str,
    checkpoint_name: str,
    threshold: float,
    config_path: str,
    export_png: bool = False,
    figure_include_splits: Optional[Sequence[str]] = None,
    figure_dpi: int = 150,
) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    patient_pred_rows: List[Dict[str, object]] = []
    patient_concept_rows: List[Dict[str, object]] = []
    metric_rows: List[Dict[str, object]] = []
    roc_rows: List[Dict[str, object]] = []
    cm_rows: List[Dict[str, object]] = []
    wrong_rows: List[Dict[str, object]] = []
    split_prediction_rows: Dict[str, List[Dict[str, object]]] = {}
    split_concept_rows: Dict[str, List[Dict[str, object]]] = {}

    for split in ("train", "val", "test"):
        if split not in split_outputs:
            continue
        output = split_outputs[split]
        pred_rows = list(output["patient_prediction_rows"])
        concept_rows = list(output["patient_concept_rows"])
        metrics = dict(output["metrics"])
        split_prediction_rows[split] = pred_rows
        split_concept_rows[split] = concept_rows

        patient_pred_rows.extend(pred_rows)
        patient_concept_rows.extend(concept_rows)

        metric_rows.append(
            {
                "split": split,
                "loss_bce_block": output["loss_bce_block"],
                "auc": metrics["auc"],
                "acc": metrics["acc"],
                "sen": metrics["sen"],
                "spe": metrics["spe"],
                "f1": metrics["f1"],
                "num_patients": output["num_patients"],
                "num_blocks": output["num_blocks"],
                "run_id": run_id,
                "checkpoint_name": checkpoint_name,
                "threshold": threshold,
            }
        )
        cm_rows.append(
            {
                "split": split,
                "tn": metrics["tn"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
                "tp": metrics["tp"],
                "run_id": run_id,
                "checkpoint_name": checkpoint_name,
            }
        )

        y_true = np.asarray([int(row["y_true"]) for row in pred_rows], dtype=np.int64)
        y_prob = np.asarray([float(row["prob_idh_mut"]) for row in pred_rows], dtype=np.float64)
        if len(np.unique(y_true)) >= 2:
            fpr, tpr, thresholds = roc_curve(y_true, y_prob)
            for fpr_item, tpr_item, thr_item in zip(fpr, tpr, thresholds):
                roc_rows.append(
                    {
                        "split": split,
                        "fpr": float(fpr_item),
                        "tpr": float(tpr_item),
                        "threshold": float(thr_item),
                        "run_id": run_id,
                        "checkpoint_name": checkpoint_name,
                    }
                )

        for row in pred_rows:
            y_true_i = int(row["y_true"])
            pred_i = int(row["pred_label"])
            if y_true_i == pred_i:
                continue
            prob = float(row["prob_idh_mut"])
            wrong_rows.append(
                {
                    **row,
                    "error_type": "FP" if y_true_i == 0 and pred_i == 1 else "FN",
                    "prob_margin_to_threshold": float(abs(prob - threshold)),
                    "is_low_confidence": int(abs(prob - threshold) < 0.1),
                }
            )

    pred_path = output_dir / f"patient_predictions_{MODEL_NAME}_{run_id}.csv"
    concept_path = output_dir / f"patient_concepts_{MODEL_NAME}_{run_id}.csv"
    metrics_path = output_dir / f"metrics_{MODEL_NAME}_{run_id}.csv"
    roc_path = output_dir / f"roc_points_{MODEL_NAME}_{run_id}.csv"
    cm_path = output_dir / f"confusion_matrix_{MODEL_NAME}_{run_id}.csv"
    wrong_path = output_dir / f"wrong_cases_{MODEL_NAME}_{run_id}.csv"
    summary_path = output_dir / f"run_summary_{MODEL_NAME}_{run_id}.json"

    _write_csv(
        pred_path,
        [
            "patient_id",
            "split",
            "y_true",
            "prob_idh_mut",
            "pred_label",
            "run_id",
            "checkpoint_name",
        ],
        patient_pred_rows,
    )

    concept_fields = ["patient_id", "split", "y_true", "run_id", "checkpoint_name"]
    # 动态拼概念字段（按首行）
    if patient_concept_rows:
        dynamic_cols = [
            key
            for key in patient_concept_rows[0].keys()
            if key not in set(concept_fields)
        ]
        concept_fields.extend(dynamic_cols)
    _write_csv(concept_path, concept_fields, patient_concept_rows)

    _write_csv(
        metrics_path,
        [
            "split",
            "loss_bce_block",
            "auc",
            "acc",
            "sen",
            "spe",
            "f1",
            "num_patients",
            "num_blocks",
            "run_id",
            "checkpoint_name",
            "threshold",
        ],
        metric_rows,
    )
    _write_csv(
        roc_path,
        ["split", "fpr", "tpr", "threshold", "run_id", "checkpoint_name"],
        roc_rows,
    )
    _write_csv(
        cm_path,
        ["split", "tn", "fp", "fn", "tp", "run_id", "checkpoint_name"],
        cm_rows,
    )
    _write_csv(
        wrong_path,
        [
            "patient_id",
            "split",
            "y_true",
            "prob_idh_mut",
            "pred_label",
            "run_id",
            "checkpoint_name",
            "error_type",
            "prob_margin_to_threshold",
            "is_low_confidence",
        ],
        wrong_rows,
    )

    figure_records: Dict[str, Dict[str, str]] = {}
    if export_png:
        include = tuple(figure_include_splits or tuple(split_prediction_rows.keys()))
        figure_records = export_evaluation_figures(
            split_prediction_rows=split_prediction_rows,
            split_concept_rows=split_concept_rows,
            output_dir=output_dir,
            run_id=run_id,
            include_splits=include,
            dpi=figure_dpi,
        )

    summary = {
        "model": MODEL_NAME,
        "run_id": run_id,
        "checkpoint_name": checkpoint_name,
        "threshold": threshold,
        "config_path": config_path,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "metrics": {row["split"]: row for row in metric_rows},
        "files": {
            "patient_predictions": str(pred_path),
            "patient_concepts": str(concept_path),
            "metrics": str(metrics_path),
            "roc_points": str(roc_path),
            "confusion_matrix": str(cm_path),
            "wrong_cases": str(wrong_path),
            "figures": figure_records,
        },
    }
    _save_json(summary_path, summary)

    exported: Dict[str, Path] = {
        "patient_predictions": pred_path,
        "patient_concepts": concept_path,
        "metrics": metrics_path,
        "roc_points": roc_path,
        "confusion_matrix": cm_path,
        "wrong_cases": wrong_path,
        "run_summary": summary_path,
    }
    if export_png and HAS_MATPLOTLIB:
        exported["figures"] = output_dir / "figures"
    return exported


def build_eval_datasets_and_loaders(
    split_base_root: Path,
    concept_label_csv: Path,
    concept_scaler_json: Path,
    data_cfg: Mapping[str, object],
    batch_size: int,
    num_workers: int,
    concept_columns: Sequence[str] | None = None,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    has_mask = bool(data_cfg.get("require_voi", True)) or bool(data_cfg.get("append_voi_mask", True)) or bool(
        data_cfg.get("mask_background_with_voi", False)
    )
    # eval 默认不做随机增强。
    transform_map = build_monai_block_transforms(
        config=MonaiAugmentConfig(enabled=False),
        has_mask=has_mask,
        spatial_size=(int(data_cfg.get("resize_height", 224)), int(data_cfg.get("resize_width", 224))),
    )

    datasets = build_habitat_cbm_datasets(
        split_base_root=split_base_root,
        modalities=tuple(data_cfg.get("modalities", ("t1", "t1ce", "t2", "t2flair", "adc", "cbf"))),
        require_voi=bool(data_cfg.get("require_voi", True)),
        append_voi_mask=bool(data_cfg.get("append_voi_mask", True)),
        mask_background_with_voi=bool(data_cfg.get("mask_background_with_voi", False)),
        block_depth=int(data_cfg.get("block_depth", 5)),
        slice_axis=int(data_cfg.get("slice_axis", 2)),
        intensity_norm=str(data_cfg.get("intensity_norm", "zscore")),
        min_nonzero_voxels=int(data_cfg.get("min_nonzero_voxels", 16)),
        cache_volumes=bool(data_cfg.get("cache_volumes", True)),
        concept_label_csv=concept_label_csv,
        concept_scaler_json=concept_scaler_json,
        concept_columns=concept_columns,
        transform_map=transform_map,
    )

    loaders = build_habitat_cbm_dataloaders(
        datasets=datasets,
        batch_size=batch_size,
        num_workers=num_workers,
        train_shuffle=False,
    )
    return datasets, loaders


def run_full_evaluation(
    model: HabitatCBM,
    dataloaders: Mapping[str, object],
    scaler: ConceptScaler,
    output_dir: Path,
    run_id: str,
    checkpoint_name: str,
    threshold: float,
    topk_pool: int,
    device: torch.device,
    include_splits: Sequence[str],
    config_path: str,
    export_png: bool = False,
    figure_include_splits: Optional[Sequence[str]] = None,
    figure_dpi: int = 150,
) -> Dict[str, Path]:
    split_outputs: Dict[str, Dict[str, object]] = {}
    for split in include_splits:
        if split not in dataloaders:
            continue
        split_outputs[split] = evaluate_split(
            model=model,
            dataloader=dataloaders[split],
            scaler=scaler,
            split=split,
            run_id=run_id,
            checkpoint_name=checkpoint_name,
            threshold=threshold,
            topk_pool=topk_pool,
            device=device,
        )

    return export_evaluation_outputs(
        split_outputs=split_outputs,
        output_dir=output_dir,
        run_id=run_id,
        checkpoint_name=checkpoint_name,
        threshold=threshold,
        config_path=config_path,
        export_png=export_png,
        figure_include_splits=figure_include_splits,
        figure_dpi=figure_dpi,
    )


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Habitat-CBM full evaluation script.")
    parser.add_argument("--config", type=Path, default=CURRENT_DIR / "args_train_habitat_CBM.json")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--topk-pool", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument(
        "--splits",
        type=str,
        default="val,test",
        help="Comma-separated splits for evaluation, e.g. val,test or train,val,test",
    )
    return parser


def _load_json(path: Path) -> Dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"Config JSON not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Config JSON must be an object.")
    return payload


def _resolve_run_id(run_id: str | None) -> str:
    if run_id:
        return run_id
    return time.strftime("%Y%m%d_%H%M%S")


def _load_model_config_from_checkpoint(checkpoint_path: Path) -> Dict[str, object]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("Unsupported checkpoint format.")
    model_cfg = payload.get("model_config", {})
    if not isinstance(model_cfg, Mapping):
        raise ValueError("Checkpoint missing model_config.")
    return dict(model_cfg)


def _load_model_from_checkpoint(checkpoint_path: Path, device: torch.device) -> HabitatCBM:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("Unsupported checkpoint format.")

    model_cfg = payload.get("model_config", {})
    if not isinstance(model_cfg, Mapping):
        raise ValueError("Checkpoint missing model_config.")
    concept_dropout_p, label_dropout_p = _resolve_model_dropouts(model_cfg)
    checkpoint_selected_concepts = model_cfg.get("selected_concepts")
    if "n_concepts" in model_cfg:
        checkpoint_n_concepts = int(model_cfg["n_concepts"])
    elif checkpoint_selected_concepts is not None:
        checkpoint_n_concepts = len(resolve_concept_names(checkpoint_selected_concepts))
    else:
        checkpoint_n_concepts = 8

    model = HabitatCBM(
        in_channels=int(model_cfg["in_channels"]),
        n_concepts=checkpoint_n_concepts,
        concept_hidden_dim=int(model_cfg.get("concept_hidden_dim", 256)),
        label_hidden_dim=int(model_cfg.get("label_hidden_dim", 32)),
        concept_dropout_p=concept_dropout_p,
        label_dropout_p=label_dropout_p,
        pretrained=False,
    ).to(device)

    state_dict = payload.get("model_state_dict")
    if state_dict is None:
        raise ValueError("Checkpoint missing model_state_dict.")
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def main() -> None:
    args = _build_argparser().parse_args()
    cfg = _load_json(args.config)

    paths_cfg = cfg.get("paths", {})
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    eval_cfg = cfg.get("eval", {})
    train_cfg = cfg.get("train", {})
    if not isinstance(model_cfg, Mapping):
        model_cfg = {}

    split_base_root = Path(paths_cfg.get("split_base_root", REPO_ROOT.parent / "dataset" / "splited_data"))
    concept_label_csv = Path(paths_cfg.get("concept_label_csv", REPO_ROOT.parent / "results" / "02_habitat" / "concept_labels.csv"))
    concept_scaler_json = Path(paths_cfg.get("concept_scaler_json", REPO_ROOT.parent / "results" / "03_habitat_cbm" / "concept_scaler_stats.json"))
    checkpoint_model_cfg = _load_model_config_from_checkpoint(args.checkpoint)
    selected_concept_names = resolve_concept_names(
        model_cfg.get("selected_concepts", checkpoint_model_cfg.get("selected_concepts"))
    )
    concept_columns = concept_names_to_columns(selected_concept_names)

    run_id = _resolve_run_id(args.run_id)
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path(paths_cfg.get("results_root", REPO_ROOT.parent / "results" / "03_habitat_cbm")) / run_id

    threshold = float(args.threshold) if args.threshold is not None else float(eval_cfg.get("threshold", 0.5))
    topk_pool = int(args.topk_pool) if args.topk_pool is not None else int(eval_cfg.get("topk_pool", 0))
    export_png = bool(eval_cfg.get("export_png", False))
    figure_dpi = int(eval_cfg.get("figure_dpi", 150))
    batch_size = int(args.batch_size) if args.batch_size is not None else int(train_cfg.get("batch_size", 8))
    num_workers = int(args.num_workers) if args.num_workers is not None else int(train_cfg.get("num_workers", 4))

    include_splits = [item.strip() for item in args.splits.split(",") if item.strip()]
    if not include_splits:
        raise ValueError("No splits specified for evaluation.")
    figure_include_splits = tuple(eval_cfg.get("figure_include_splits", include_splits))
    if not figure_include_splits:
        figure_include_splits = tuple(include_splits)

    device = torch.device(args.device)
    scaler = load_concept_scaler(
        concept_scaler_json,
        concept_names=selected_concept_names,
    )

    _, dataloaders = build_eval_datasets_and_loaders(
        split_base_root=split_base_root,
        concept_label_csv=concept_label_csv,
        concept_scaler_json=concept_scaler_json,
        data_cfg=data_cfg,
        batch_size=batch_size,
        num_workers=num_workers,
        concept_columns=concept_columns,
    )

    model = _load_model_from_checkpoint(args.checkpoint, device=device)
    if model.n_concepts != len(scaler.concept_names):
        raise ValueError(
            "Checkpoint n_concepts does not match selected concepts/scaler dimension: "
            f"{model.n_concepts} vs {len(scaler.concept_names)}"
        )

    exported = run_full_evaluation(
        model=model,
        dataloaders=dataloaders,
        scaler=scaler,
        output_dir=output_dir,
        run_id=run_id,
        checkpoint_name=args.checkpoint.name,
        threshold=threshold,
        topk_pool=topk_pool,
        device=device,
        include_splits=include_splits,
        config_path=str(args.config),
        export_png=export_png,
        figure_include_splits=figure_include_splits,
        figure_dpi=figure_dpi,
    )

    print("Evaluation completed. Exported files:")
    for key, value in exported.items():
        print(f"  - {key}: {value}")


if __name__ == "__main__":
    main()
