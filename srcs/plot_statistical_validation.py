#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Publication-style visualization for statistical validation outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


SCRIPT_PATH = Path(__file__).resolve()
LABS_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_RESULTS_DIR = LABS_ROOT / "results" / "statistical_validation"
DEFAULT_DPI = 800

MODEL_ORDER = ["Habitat-CBM", "ResNet-18", "Radiomics+LR"]
MODEL_COLORS = {
    "Habitat-CBM": "#1B9E77",   # teal green
    "ResNet-18": "#3B5BDB",     # scientific blue
    "Radiomics+LR": "#D95F02",  # vermillion
}
METRIC_LABELS = {
    "auc": "AUC",
    "acc": "Accuracy",
    "sen": "Sensitivity",
    "spe": "Specificity",
    "f1": "F1 score",
}
METRIC_SHORT_LABELS = {
    "auc": "AUC",
    "acc": "ACC",
    "sen": "SEN",
    "spe": "SPE",
    "f1": "F1",
}
OUTCOME_COLORS = {
    "TN": "#4C78A8",
    "TP": "#59A14F",
    "FP": "#E15759",
    "FN": "#B07AA1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot statistical validation results.")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    parser.add_argument("--theme", choices=["light", "paper"], default="paper")
    return parser.parse_args()


def configure_style(theme: str) -> None:
    sns.set_theme(style="whitegrid", context="paper")
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.titlesize": 13,
            "figure.dpi": 150,
            "savefig.dpi": DEFAULT_DPI,
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.45,
            "grid.alpha": 0.28,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    if theme == "paper":
        mpl.rcParams.update(
            {
                "figure.facecolor": "#FBFBF8",
                "axes.facecolor": "#FBFBF8",
                "savefig.facecolor": "#FBFBF8",
            }
        )


def load_results(results_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics = pd.read_csv(results_dir / "metrics_with_bootstrap_ci.csv")
    delong = pd.read_csv(results_dir / "delong_tests.csv")
    mcnemar = pd.read_csv(results_dir / "mcnemar_tests.csv")
    fisher = pd.read_csv(results_dir / "fisher_tests.csv")
    metrics["model"] = pd.Categorical(metrics["model"], categories=MODEL_ORDER, ordered=True)
    metrics = metrics.sort_values("model").reset_index(drop=True)
    return metrics, delong, mcnemar, fisher


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.08,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
        ha="left",
    )


def format_p_value(value: float) -> str:
    if pd.isna(value):
        return "NA"
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"


def plot_auc_forest(ax: plt.Axes, metrics: pd.DataFrame) -> None:
    y_positions = np.arange(len(metrics))[::-1]
    for y, row in zip(y_positions, metrics.itertuples(index=False)):
        model = str(row.model)
        color = MODEL_COLORS[model]
        ax.hlines(y, row.auc_ci_low, row.auc_ci_high, color=color, lw=4, alpha=0.28)
        ax.hlines(y, row.auc_ci_low, row.auc_ci_high, color=color, lw=1.35)
        ax.scatter(row.auc, y, s=95, color=color, edgecolor="white", linewidth=1.2, zorder=3)
        ax.text(
            1.035,
            y,
            f"{row.auc:.3f}\n[{row.auc_ci_low:.3f}, {row.auc_ci_high:.3f}]",
            va="center",
            ha="left",
            fontsize=7.6,
            color="#222222",
            linespacing=1.15,
        )
    ax.axvline(0.5, color="#B8B8B8", lw=0.9, ls="--", zorder=0)
    ax.set_yticks(y_positions, metrics["model"].astype(str))
    ax.set_xlim(0.48, 1.12)
    ax.set_xlabel("AUC with patient-level bootstrap 95% CI")
    ax.set_title("AUC point estimate is highest, but uncertainty is wide", loc="left", pad=8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    add_panel_label(ax, "A")


def plot_metric_matrix(ax: plt.Axes, metrics: pd.DataFrame) -> None:
    metric_cols = ["auc", "acc", "sen", "spe", "f1"]
    matrix = metrics.set_index("model")[metric_cols].loc[MODEL_ORDER]
    im = ax.imshow(matrix.values, vmin=0.64, vmax=1.0, cmap=sns.color_palette("mako", as_cmap=True), aspect="auto")

    ax.set_xticks(np.arange(len(metric_cols)), [METRIC_LABELS[m] for m in metric_cols], rotation=25, ha="right")
    ax.set_yticks(np.arange(len(MODEL_ORDER)), MODEL_ORDER)
    ax.set_title("Metric profile at the fixed decision threshold", loc="left", pad=8)
    ax.tick_params(length=0)
    for i, model in enumerate(MODEL_ORDER):
        for j, metric in enumerate(metric_cols):
            value = matrix.loc[model, metric]
            text_color = "white" if value < 0.82 else "#101010"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8, color=text_color)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.025)
    cbar.outline.set_visible(False)
    cbar.set_label("Score", rotation=270, labelpad=10)
    ax.spines[:].set_visible(False)
    add_panel_label(ax, "B")


def plot_statistical_evidence(ax: plt.Axes, delong: pd.DataFrame, mcnemar: pd.DataFrame) -> None:
    rows = []
    for row in delong.itertuples(index=False):
        rows.append((row.comparison.replace("Habitat-CBM vs ", "vs "), "DeLong AUC", row.p_value))
    for row in mcnemar.itertuples(index=False):
        rows.append((row.comparison.replace("Habitat-CBM vs ", "vs "), "McNemar", row.p_value))

    y_positions = np.arange(len(rows))[::-1]
    ax.axvspan(0.0, 0.05, color="#FDECEC", alpha=0.95, zorder=0)
    ax.axvline(0.05, color="#E15759", lw=1.1, ls="--")
    for y, (comparison, test_name, p_value) in zip(y_positions, rows):
        color = "#E15759" if p_value < 0.05 else "#6C757D"
        ax.plot([0.0, min(p_value, 1.0)], [y, y], color=color, lw=1.4, alpha=0.85)
        ax.scatter(p_value, y, s=65, color=color, edgecolor="white", linewidth=0.9, zorder=3)
        ax.text(1.055, y, f"p={format_p_value(p_value)}", va="center", ha="left", fontsize=8.2, color="#222222")

    ax.set_yticks(y_positions, [f"{test}\n{comparison}" for comparison, test, _ in rows])
    ax.set_xlim(0, 1.18)
    ax.set_xlabel("p value")
    ax.set_title("No pairwise comparison crosses the significance boundary", loc="left", pad=8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    add_panel_label(ax, "C")


def draw_outcome_tiles(ax: plt.Axes, metrics: pd.DataFrame) -> None:
    ax.set_title("Patient-level outcomes as compact tiles", loc="left", pad=8)
    tile_size = 0.72
    label_x = -2.25
    n_cols = 14
    y_gap = 2.15

    legend_handles = []
    for label, color in OUTCOME_COLORS.items():
        legend_handles.append(mpl.patches.Patch(facecolor=color, edgecolor="none", label=label))

    for row_idx, row in enumerate(metrics.itertuples(index=False)):
        model = str(row.model)
        y_base = (len(metrics) - 1 - row_idx) * y_gap
        sequence = (
            ["TN"] * int(row.tn)
            + ["TP"] * int(row.tp)
            + ["FP"] * int(row.fp)
            + ["FN"] * int(row.fn)
        )
        ax.text(label_x, y_base + 0.43, model, ha="right", va="center", fontsize=8.5, fontweight="bold")
        for idx, outcome in enumerate(sequence):
            x = idx % n_cols
            y = y_base - idx // n_cols
            rect = mpl.patches.FancyBboxPatch(
                (x, y),
                tile_size,
                tile_size,
                boxstyle="round,pad=0.02,rounding_size=0.06",
                linewidth=0,
                facecolor=OUTCOME_COLORS[outcome],
                alpha=0.94,
            )
            ax.add_patch(rect)

        ax.text(
            n_cols + 0.35,
            y_base + 0.42,
            f"TN {row.tn}  FP {row.fp}\nFN {row.fn}  TP {row.tp}",
            ha="left",
            va="center",
            fontsize=7.5,
            color="#303030",
        )

    ax.set_xlim(label_x - 0.2, n_cols + 3.4)
    ax.set_ylim(-1.1, (len(metrics) - 1) * y_gap + 1.25)
    ax.axis("off")
    ax.legend(handles=legend_handles, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.14), frameon=False)
    add_panel_label(ax, "D")


def plot_dashboard(results_dir: Path, metrics: pd.DataFrame, delong: pd.DataFrame, mcnemar: pd.DataFrame, dpi: int) -> Path:
    fig = plt.figure(figsize=(12.2, 9.0), constrained_layout=False)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05], width_ratios=[1.1, 1.0], hspace=0.46, wspace=0.36)

    ax_auc = fig.add_subplot(gs[0, 0])
    ax_matrix = fig.add_subplot(gs[0, 1])
    ax_stats = fig.add_subplot(gs[1, 0])
    ax_tiles = fig.add_subplot(gs[1, 1])

    plot_auc_forest(ax_auc, metrics)
    plot_metric_matrix(ax_matrix, metrics)
    plot_statistical_evidence(ax_stats, delong, mcnemar)
    draw_outcome_tiles(ax_tiles, metrics)

    fig.suptitle("Statistical validation of IDH prediction on the held-out test split", x=0.02, ha="left", fontweight="bold")
    fig.text(
        0.02,
        0.94,
        "n=28 patients. Error bars are patient-level bootstrap 95% confidence intervals. "
        "p values summarize pairwise tests against Habitat-CBM.",
        ha="left",
        fontsize=9,
        color="#444444",
    )
    fig.subplots_adjust(top=0.88, bottom=0.08, left=0.09, right=0.97)
    out = results_dir / "statistical_validation_dashboard_800ppi.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_metric_forest(results_dir: Path, metrics: pd.DataFrame, dpi: int) -> Path:
    metric_cols = ["auc", "acc", "sen", "spe", "f1"]
    fig, axes = plt.subplots(1, len(metric_cols), figsize=(14.8, 4.7), sharey=True)
    y_positions = np.arange(len(metrics))[::-1]

    for ax, metric in zip(axes, metric_cols):
        for y, row in zip(y_positions, metrics.itertuples(index=False)):
            model = str(row.model)
            point = getattr(row, metric)
            low = getattr(row, f"{metric}_ci_low")
            high = getattr(row, f"{metric}_ci_high")
            color = MODEL_COLORS[model]
            ax.hlines(y, low, high, color=color, lw=3.2, alpha=0.28)
            ax.hlines(y, low, high, color=color, lw=1.05)
            ax.scatter(point, y, s=55, color=color, edgecolor="white", linewidth=0.85, zorder=3)
            ax.text(
                min(point + 0.025, 1.005),
                y,
                f"{point:.2f}",
                ha="left",
                va="center",
                fontsize=7.2,
                bbox={"boxstyle": "round,pad=0.12", "facecolor": "#FBFBF8", "edgecolor": "none", "alpha": 0.88},
            )

        ax.set_xlim(0.45, 1.03)
        ax.set_ylim(-0.28, len(metrics) - 0.72)
        ax.set_title(METRIC_SHORT_LABELS[metric], pad=8, fontweight="bold")
        ax.axvline(0.5, color="#B8B8B8", lw=0.75, ls="--", zorder=0)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel("Score")

    axes[0].set_yticks(y_positions, metrics["model"].astype(str))
    fig.suptitle("Bootstrap uncertainty across all reported test metrics", x=0.02, ha="left", fontweight="bold")
    fig.text(0.02, 0.90, "Each marker is the point estimate; each line is the 95% patient-level bootstrap interval.", ha="left", fontsize=9)
    fig.subplots_adjust(top=0.78, bottom=0.16, left=0.11, right=0.985, wspace=0.18)
    out = results_dir / "metric_ci_forest_800ppi.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    args = parse_args()
    configure_style(args.theme)
    metrics, delong, mcnemar, _ = load_results(args.results_dir)
    outputs = [
        plot_dashboard(args.results_dir, metrics, delong, mcnemar, args.dpi),
        plot_metric_forest(args.results_dir, metrics, args.dpi),
    ]
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
