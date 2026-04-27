from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpecFromSubplotSpec

ROOT = Path("/Users/yankeeschen/Documents/Research/bachelor_thesis/idh/paper")
RESULTS = ROOT / "labs" / "results"
FIG_DIR = ROOT / "figs"

MODEL_COLORS = {
    "Habitat-CBM": "#1F9D8A",
    "ResNet-18": "#E76F51",
    "Radiomics + LR": "#4C5B8C",
}
CONCEPT_COLORS = {
    "C1": "#4C78A8",
    "C2": "#F58518",
    "C3": "#54A24B",
    "C4": "#E45756",
    "C5": "#B279A2",
}
NEUTRAL = "#8F98A3"
GRID = "#DCE2EA"
CMAP_MAIN = sns.color_palette("blend:#EEF7F5,#1F8A70", as_cmap=True)
CMAP_BINARY = sns.light_palette("#1F9D8A", as_cmap=True)

sns.set_theme(style="ticks", context="paper")
plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "figure.dpi": 160,
        "savefig.dpi": 320,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelsize": 11,
        "axes.titlesize": 12.5,
        "axes.titleweight": "bold",
        "legend.fontsize": 9,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "font.family": "DejaVu Sans",
    }
)


def add_panel_label(ax, label: str) -> None:
    ax.text(
        -0.12,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=15,
        fontweight="bold",
        ha="left",
        va="top",
    )


def soften_grid(ax, axis: str = "both") -> None:
    ax.grid(True, axis=axis, color=GRID, linewidth=0.8, alpha=0.65)


def save_figure(fig: plt.Figure, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def smooth(arr: pd.Series | np.ndarray, window: int = 7) -> np.ndarray:
    return pd.Series(arr).rolling(window=window, min_periods=1, center=False).mean().to_numpy()


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def load_metrics() -> pd.DataFrame:
    rad = pd.read_csv(RESULTS / "baseline_RadiomicsLR" / "20260414_164612" / "metrics_radiomics_lr_20260414_164612.csv")
    res = pd.read_csv(RESULTS / "baseline_ResNet18" / "20260423_113830" / "metrics_resnet18_20260423_113830.csv")
    hab = pd.read_csv(RESULTS / "habitat_CBM" / "20260422_145024" / "metrics_habitat_cbm_20260422_145024.csv")

    rows = []
    for model, df in [("Radiomics + LR", rad), ("ResNet-18", res), ("Habitat-CBM", hab)]:
        rec = {"model": model}
        rec.update(df.loc[df["split"] == "test", ["auc", "acc", "sen", "spe", "f1"]].iloc[0].to_dict())
        rows.append(rec)
    return pd.DataFrame(rows)


def load_roc() -> dict[str, pd.DataFrame]:
    rad = pd.read_csv(RESULTS / "baseline_RadiomicsLR" / "20260414_164612" / "roc_points_radiomics_lr_test_20260414_164612.csv")
    res = pd.read_csv(RESULTS / "baseline_ResNet18" / "20260423_113830" / "roc_points_resnet18_test_20260423_113830.csv")
    hab = pd.read_csv(RESULTS / "habitat_CBM" / "20260422_145024" / "roc_points_habitat_cbm_20260422_145024.csv")
    return {
        "Radiomics + LR": rad,
        "ResNet-18": res,
        "Habitat-CBM": hab.loc[hab["split"] == "test"].copy(),
    }


def load_confusions() -> dict[str, np.ndarray]:
    rad = pd.read_csv(RESULTS / "baseline_RadiomicsLR" / "20260414_164612" / "confusion_matrix_radiomics_lr_test_20260414_164612.csv").iloc[0]
    res = pd.read_csv(RESULTS / "baseline_ResNet18" / "20260423_113830" / "confusion_matrix_resnet18_test_20260423_113830.csv").iloc[0]
    hab = pd.read_csv(RESULTS / "habitat_CBM" / "20260422_145024" / "confusion_matrix_habitat_cbm_20260422_145024.csv")
    hab = hab.loc[hab["split"] == "test"].iloc[0]
    return {
        "Radiomics + LR": np.array([[rad["tn"], rad["fp"]], [rad["fn"], rad["tp"]]], dtype=float),
        "ResNet-18": np.array([[res["tn"], res["fp"]], [res["fn"], res["tp"]]], dtype=float),
        "Habitat-CBM": np.array([[hab["tn"], hab["fp"]], [hab["fn"], hab["tp"]]], dtype=float),
    }


def plot_fig_a1() -> None:
    metrics = load_metrics()
    rocs = load_roc()
    confusions = load_confusions()

    fig = plt.figure(figsize=(16, 9.6), constrained_layout=True)
    gs = fig.add_gridspec(2, 6, height_ratios=[1.0, 1.0], hspace=0.12, wspace=0.38)

    ax_roc = fig.add_subplot(gs[0, :3])
    add_panel_label(ax_roc, "A")
    for model, df in rocs.items():
        auc = metrics.loc[metrics["model"] == model, "auc"].iloc[0]
        ax_roc.plot(df["fpr"], df["tpr"], lw=2.4, color=MODEL_COLORS[model], label=f"{model}  ({auc:.3f})")
    ax_roc.plot([0, 1], [0, 1], ls="--", lw=1.0, color="#B0B7C3")
    ax_roc.set_title("Test-set ROC comparison")
    ax_roc.set_xlabel("False positive rate")
    ax_roc.set_ylabel("True positive rate")
    ax_roc.set_xlim(0, 1)
    ax_roc.set_ylim(0, 1.02)
    soften_grid(ax_roc)
    ax_roc.legend(frameon=False, loc="lower right", title="Model")

    ax_metric = fig.add_subplot(gs[0, 3:])
    add_panel_label(ax_metric, "B")
    metric_order = ["auc", "acc", "sen", "spe", "f1"]
    x = np.arange(len(metric_order))
    width = 0.22
    model_order = ["Radiomics + LR", "ResNet-18", "Habitat-CBM"]
    for idx, model in enumerate(model_order):
        values = metrics.loc[metrics["model"] == model, metric_order].iloc[0].to_numpy(dtype=float)
        xpos = x + (idx - 1) * width
        ax_metric.bar(xpos, values, width=width, color=MODEL_COLORS[model], edgecolor="none", alpha=0.94, label=model)
    ax_metric.set_title("Patient-level metrics on the test split")
    ax_metric.set_xticks(x)
    ax_metric.set_xticklabels([m.upper() for m in metric_order])
    ax_metric.set_ylim(0.6, 1.0)
    ax_metric.set_ylabel("Score")
    soften_grid(ax_metric, axis="y")
    ax_metric.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.14))

    cm_spec = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[1, :], wspace=0.18)
    for idx, model in enumerate(model_order):
        ax = fig.add_subplot(cm_spec[0, idx])
        add_panel_label(ax, chr(ord("C") + idx))
        cm = confusions[model]
        row_sum = cm.sum(axis=1, keepdims=True)
        annot = np.empty_like(cm, dtype=object)
        for i in range(2):
            for j in range(2):
                pct = 100.0 * cm[i, j] / row_sum[i, 0] if row_sum[i, 0] > 0 else 0.0
                annot[i, j] = f"{int(cm[i, j])}\n{pct:.1f}%"
        sns.heatmap(
            cm,
            annot=annot,
            fmt="",
            cmap=CMAP_MAIN,
            cbar=False,
            square=True,
            linewidths=1.1,
            linecolor="white",
            ax=ax,
            annot_kws={"fontsize": 11.5},
            vmin=0,
            vmax=max(v.max() for v in confusions.values()),
        )
        ax.set_title(model)
        ax.set_xticklabels(["WT", "Mut"], rotation=0)
        ax.set_yticklabels(["WT", "Mut"], rotation=0)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label" if idx == 0 else "")

    fig.suptitle("Supplementary model-level comparison under the fixed test split", fontsize=16, fontweight="bold")
    save_figure(fig, "figA1")


def plot_training_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    title: str,
    left_specs: list[tuple[str, str, str]],
    right_spec: tuple[str, str, str] | None,
    best_epoch: int,
    panel_label: str,
) -> None:
    add_panel_label(ax, panel_label)
    for col, label, color in left_specs:
        raw = df[col].to_numpy(dtype=float)
        ax.plot(df["epoch"], raw, color=color, lw=1.0, alpha=0.18)
        ax.plot(df["epoch"], smooth(raw, 5), color=color, lw=2.0, label=label)
    ax.axvline(best_epoch, color=NEUTRAL, ls="--", lw=1.1)
    ax.text(
        0.03,
        0.95,
        f"best epoch = {best_epoch}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9.2,
        color="#374151",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#E5E7EB", alpha=0.95),
    )
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    soften_grid(ax)

    if right_spec is not None:
        col, label, color = right_spec
        ax2 = ax.twinx()
        auc = df[col].to_numpy(dtype=float)
        ax2.plot(df["epoch"], auc, color=color, lw=2.0, label=label)
        ax2.set_ylabel("AUC", color=color)
        ax2.tick_params(axis="y", colors=color)
        ax2.set_ylim(0.0, max(0.85, float(np.nanmax(auc)) + 0.06))
        handles1, labels1 = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(handles1 + handles2, labels1 + labels2, frameon=False, loc="lower right")
    else:
        ax.legend(frameon=False, loc="upper right")


def plot_fig_a2() -> None:
    stage1 = pd.read_csv(RESULTS / "habitat_CBM" / "20260422_145024" / "stage1_concept_log.csv")
    stage2 = pd.read_csv(RESULTS / "habitat_CBM" / "20260422_145024" / "stage2_label_head_log.csv")
    stage3a = pd.read_csv(RESULTS / "habitat_CBM" / "20260422_145024" / "stage3a_label_adapt_log.csv")
    stage3b = pd.read_csv(RESULTS / "habitat_CBM" / "20260422_145024" / "stage3_joint_log.csv")

    def last_best_epoch(df: pd.DataFrame) -> int:
        best_rows = df.loc[df["is_best"] == 1, "epoch"]
        return int(best_rows.iloc[-1]) if not best_rows.empty else int(df["epoch"].iloc[-1])

    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.0), constrained_layout=True)
    plot_training_panel(
        axes[0, 0],
        stage1,
        "Stage 1: concept pretraining",
        [("train_concept_loss", "Train concept loss", "#4C5B8C"), ("val_concept_loss", "Val concept loss", "#8597B3")],
        None,
        last_best_epoch(stage1),
        "A",
    )
    plot_training_panel(
        axes[0, 1],
        stage2,
        "Stage 2: oracle concept-to-label calibration",
        [("train_label_loss", "Train label loss", "#4C5B8C"), ("val_label_loss", "Val label loss", "#8597B3")],
        ("val_auc", "Val AUC", "#1F9D8A"),
        last_best_epoch(stage2),
        "B",
    )
    plot_training_panel(
        axes[1, 0],
        stage3a,
        "Stage 3A: predicted-concept label adaptation",
        [("train_label_loss", "Train label loss", "#4C5B8C"), ("val_label_loss", "Val label loss", "#8597B3")],
        ("val_auc", "Val AUC", "#1F9D8A"),
        last_best_epoch(stage3a),
        "C",
    )
    plot_training_panel(
        axes[1, 1],
        stage3b,
        "Stage 3B: joint fine-tuning",
        [
            ("train_label_loss", "Train label loss", "#4C5B8C"),
            ("val_label_loss", "Val label loss", "#8597B3"),
            ("val_concept_loss", "Val concept loss", "#C06C84"),
        ],
        ("val_auc", "Val AUC", "#1F9D8A"),
        last_best_epoch(stage3b),
        "D",
    )
    fig.suptitle("Supplementary training dynamics of the Habitat-CBM optimization pipeline", fontsize=16, fontweight="bold")
    save_figure(fig, "figA2")


def plot_fig_a3() -> None:
    df = pd.read_csv(RESULTS / "habitat_CBM" / "20260422_145024" / "patient_concepts_habitat_cbm_20260422_145024.csv")
    df = df.loc[df["split"] == "test"].copy()
    concepts = [
        ("c1", "C1", "Enhancement burden"),
        ("c2", "C2", "Non-enhancing component"),
        ("c3", "C3", "Boundary clarity"),
        ("c4", "C4", "FLAIR abnormality burden"),
        ("c6", "C5", "Hyperperfusion / angiogenesis"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10.6), constrained_layout=True)
    summary_rows = []
    for idx, (prefix, cid, title) in enumerate(concepts):
        ax = axes.flat[idx]
        add_panel_label(ax, chr(ord("A") + idx))
        x = df[f"{prefix}_true_std"].to_numpy(dtype=float)
        y = df[f"{prefix}_pred_std"].to_numpy(dtype=float)
        mae = float(np.mean(np.abs(y - x)))
        r = float(np.corrcoef(x, y)[0, 1])
        r2 = float(compute_r2(x, y))
        summary_rows.append({"concept": cid, "pearson": r, "r2": r2, "mae": mae})

        lo = min(x.min(), y.min()) - 0.25
        hi = max(x.max(), y.max()) + 0.25
        ax.plot([lo, hi], [lo, hi], ls="--", lw=1.0, color="#B7C0CC")
        ax.scatter(x, y, s=34, color=CONCEPT_COLORS[cid], alpha=0.88, edgecolor="white", linewidth=0.7)
        coeff = np.polyfit(x, y, deg=1)
        xx = np.linspace(lo, hi, 120)
        ax.plot(xx, coeff[0] * xx + coeff[1], color=CONCEPT_COLORS[cid], lw=1.7)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_title(f"{cid}: {title}")
        if idx >= 3:
            ax.set_xlabel("True standardized concept")
        else:
            ax.set_xlabel("")
        if idx in [0, 3]:
            ax.set_ylabel("Predicted standardized concept")
        else:
            ax.set_ylabel("")
        soften_grid(ax)
        ax.text(
            0.03,
            0.96,
            f"r = {r:.3f}\nR² = {r2:.3f}\nMAE = {mae:.3f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8.9,
            bbox=dict(boxstyle="round,pad=0.23", facecolor="white", edgecolor="#E5E7EB", alpha=0.95),
        )

    ax_sum = axes.flat[5]
    add_panel_label(ax_sum, "F")
    summary = pd.DataFrame(summary_rows).sort_values("pearson", ascending=True).reset_index(drop=True)
    ypos = np.arange(len(summary))
    pearson_colors = [CONCEPT_COLORS[c] for c in summary["concept"]]
    ax_sum.axvline(0, color="#C7CED9", lw=1.0)
    ax_sum.barh(ypos + 0.15, summary["pearson"], height=0.28, color=pearson_colors, edgecolor="none", label="Pearson r")
    ax_sum.barh(ypos - 0.15, summary["r2"], height=0.22, color="#CBD5E1", edgecolor="none", label="R²")
    for i, row in summary.iterrows():
        text_x = max(row["pearson"], row["r2"], 0) + 0.03
        ax_sum.text(text_x, i + 0.14, f"MAE={row['mae']:.2f}", fontsize=8.8, va="center", color="#374151")
    ax_sum.set_yticks(ypos)
    ax_sum.set_yticklabels(summary["concept"])
    ax_sum.set_xlabel("Agreement score")
    ax_sum.set_title("Recoverability summary")
    ax_sum.set_xlim(-0.12, 0.9)
    soften_grid(ax_sum, axis="x")
    ax_sum.legend(frameon=False, loc="lower right")

    fig.suptitle("Supplementary concept-level recoverability on the test split", fontsize=16, fontweight="bold")
    save_figure(fig, "figA3")


def plot_fig_a4() -> None:
    metrics = pd.read_csv(RESULTS / "intervention" / "20260422_145024_v2" / "intervention_metrics_by_budget.csv")
    correction = pd.read_csv(RESULTS / "intervention" / "20260422_145024_v2" / "intervention_correction_summary.csv")
    case_list = pd.read_csv(RESULTS / "intervention" / "20260422_145024_v2" / "intervention_case_list.csv")
    per_case = pd.read_csv(RESULTS / "intervention" / "20260422_145024_v2" / "intervention_per_case.csv")
    summary = pd.read_csv(RESULTS / "intervention" / "20260422_145024_v2" / "intervention_summary.csv")
    threshold = float(summary.loc[0, "threshold"])

    metrics_all = metrics.loc[metrics["scope"] == "selected_split_all"].copy()
    metrics_all["budget_k"] = metrics_all["budget_k"].astype(int)
    correction["budget_k"] = correction["budget_k"].astype(int)

    fig = plt.figure(figsize=(15.8, 10.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, hspace=0.16, wspace=0.26)

    ax1 = fig.add_subplot(gs[0, 0])
    add_panel_label(ax1, "A")
    for delta_col, label, color in [("delta_auc", "ΔAUC", "#1F9D8A"), ("delta_acc", "ΔACC", "#4C78A8"), ("delta_f1", "ΔF1", "#E76F51")]:
        ax1.plot(metrics_all["budget_k"], metrics_all[delta_col], marker="o", ms=5.5, lw=2.0, color=color, label=label)
    ax1.axhline(0, color="#B7C0CC", lw=1.0)
    ax1.set_title("Performance gain under selective intervention")
    ax1.set_xlabel("Maximum budget k")
    ax1.set_ylabel("Improvement over baseline")
    ax1.set_xticks(metrics_all["budget_k"])
    ax1.set_ylim(0.0, 0.19)
    soften_grid(ax1)
    ax1.legend(frameon=False, loc="upper left")

    ax2 = fig.add_subplot(gs[0, 1])
    add_panel_label(ax2, "B")
    ax2.plot(correction["budget_k"], correction["correction_rate"], marker="o", ms=5.5, lw=2.0, color="#1F9D8A", label="Correction rate")
    ax2.set_title("Correction efficiency versus intervention budget")
    ax2.set_xlabel("Maximum budget k")
    ax2.set_ylabel("Correction rate", color="#1F9D8A")
    ax2.tick_params(axis="y", colors="#1F9D8A")
    ax2.set_xticks(correction["budget_k"])
    ax2.set_ylim(0.0, 1.02)
    soften_grid(ax2)
    ax2b = ax2.twinx()
    ax2b.bar(
        correction["budget_k"],
        correction["avg_actual_intervened_concepts_all_candidates"],
        width=0.55,
        color="#EADFC1",
        edgecolor="none",
        alpha=0.95,
        label="Avg. concepts / candidate",
    )
    ax2b.plot(
        correction["budget_k"],
        correction["avg_actual_intervened_concepts_wrong_cases"],
        marker="s",
        ms=5,
        lw=1.8,
        color="#B56576",
        label="Avg. concepts / wrong case",
    )
    ax2b.set_ylabel("Average intervened concepts")
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2b.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=1)

    ax3 = fig.add_subplot(gs[1, 0])
    add_panel_label(ax3, "C")
    per_case_k1 = per_case.loc[per_case["budget_k"] == 1, ["patient_id", "prob_after", "corrected_flag"]].copy()
    merged = case_list.merge(per_case_k1, on="patient_id", how="left")

    # First draw low-confidence but originally correct cases in gray.
    background = merged.loc[merged["is_wrong"] == 0].copy().sort_values("prob_before")
    for _, row in background.iterrows():
        ax3.plot([0, 1], [row["prob_before"], row["prob_after"]], color="#C2C8D0", lw=1.1, alpha=0.85, zorder=1)
        ax3.scatter([0, 1], [row["prob_before"], row["prob_after"]], color="#A6ADB8", s=24, zorder=2)

    wrong = merged.loc[merged["is_wrong"] == 1].copy().sort_values("prob_after")
    offsets = np.linspace(-0.018, 0.018, len(wrong)) if len(wrong) > 1 else np.array([0.0])
    for (_, row), dy in zip(wrong.iterrows(), offsets):
        corrected = int(row["corrected_flag"]) == 1
        color = "#1F9D8A" if corrected else "#E76F51"
        ax3.plot([0, 1], [row["prob_before"], row["prob_after"]], color=color, lw=2.1, alpha=0.95, zorder=3)
        ax3.scatter([0, 1], [row["prob_before"], row["prob_after"]], color=color, s=48, zorder=4, edgecolor="white", linewidth=0.8)
        ax3.text(1.04, row["prob_after"] + dy, str(int(row["patient_id"])), va="center", ha="left", fontsize=9, color=color)

    ax3.axhline(threshold, linestyle="--", color="#6B7280", lw=1.0)
    ax3.text(0.02, threshold + 0.004, f"threshold = {threshold:.3f}", color="#6B7280", fontsize=8.8)
    ax3.set_xlim(-0.1, 1.22)
    ax3.set_xticks([0, 1])
    ax3.set_xticklabels(["Before", "After (k=1)"])
    ax3.set_ylabel("Predicted IDH-mutant probability")
    ax3.set_title("Candidate-case probability migration")
    soften_grid(ax3, axis="y")
    legend_handles = [
        Line2D([0], [0], color="#1F9D8A", lw=2.0, marker="o", label="Wrong → corrected"),
        Line2D([0], [0], color="#E76F51", lw=2.0, marker="o", label="Wrong → unresolved"),
        Line2D([0], [0], color="#A6ADB8", lw=1.2, marker="o", label="Low-confidence correct"),
    ]
    ax3.legend(handles=legend_handles, frameon=False, loc="upper left")

    ax4 = fig.add_subplot(gs[1, 1])
    add_panel_label(ax4, "D")
    last_budget = int(per_case["budget_k"].max())
    wrong_last = per_case.loc[(per_case["budget_k"] == last_budget) & (per_case["corrected_flag"].notna())].copy()
    wrong_ids = case_list.loc[case_list["is_wrong"] == 1, "patient_id"].astype(int).tolist()
    wrong_last = wrong_last.loc[wrong_last["patient_id"].astype(int).isin(wrong_ids)].copy()
    corrected_map = (
        per_case.loc[(per_case["budget_k"] == last_budget) & (per_case["patient_id"].astype(int).isin(wrong_ids)), ["patient_id", "corrected_flag"]]
        .drop_duplicates()
        .set_index("patient_id")["corrected_flag"]
        .astype(int)
        .to_dict()
    )

    concept_order = ["c1", "c2", "c3", "c4", "c6"]
    label_map = {"c1": "C1", "c2": "C2", "c3": "C3", "c4": "C4", "c6": "C5"}
    matrix = pd.DataFrame(0, index=[str(pid) for pid in wrong_ids], columns=concept_order, dtype=int)
    for _, row in per_case.loc[(per_case["budget_k"] == last_budget) & (per_case["patient_id"].astype(str).isin(matrix.index))].iterrows():
        if isinstance(row["concepts_replaced"], str) and row["concepts_replaced"].strip():
            for concept in row["concepts_replaced"].split("|"):
                if concept in matrix.columns:
                    matrix.loc[str(row["patient_id"]), concept] = 1
    ordered_ids = sorted(matrix.index, key=lambda pid: (1 - corrected_map.get(int(pid), 0), int(pid)))
    matrix = matrix.loc[ordered_ids]
    annot = matrix.replace({0: "", 1: "✓"}).rename(columns=label_map)
    row_labels = [f"{pid} ✓" if corrected_map.get(int(pid), 0) == 1 else f"{pid} ✗" for pid in matrix.index]
    sns.heatmap(
        matrix.rename(columns=label_map),
        cmap=CMAP_BINARY,
        cbar=False,
        linewidths=1.0,
        linecolor="white",
        square=False,
        ax=ax4,
        annot=annot,
        fmt="",
        annot_kws={"fontsize": 12, "fontweight": "bold"},
        vmin=0,
        vmax=1,
    )
    ax4.set_title(f"Replaced concepts in originally wrong cases (k={last_budget})")
    ax4.set_xlabel("Concept dimension")
    ax4.set_ylabel("Patient ID")
    ax4.set_yticklabels(row_labels, rotation=0)

    fig.suptitle("Supplementary selective-concept intervention analysis", fontsize=16, fontweight="bold")
    save_figure(fig, "figA4")


def main() -> None:
    plot_fig_a1()
    plot_fig_a2()
    plot_fig_a3()
    plot_fig_a4()
    print("Saved figures: figA1.png, figA2.png, figA3.png, figA4.png")


if __name__ == "__main__":
    main()
