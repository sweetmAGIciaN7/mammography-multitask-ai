"""Figures for the README (static PNGs)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Reference palette (categorical slots 1-3 validate all-pairs for CVD) + text tokens.
SERIES = {"random": "#2a78d6", "image": "#eb6834", "patient": "#1baf7a"}
LABELS = {"random": "Paper protocol\n(split after augmentation)",
          "image": "Grouped by\noriginal image", "patient": "Grouped by\npatient"}
TEXT, TEXT2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

PAPER_REPORTED = {"file_auc": 0.962, "file_accuracy": 0.936, "density_accuracy": 0.889}  # Esen et al. Tables 2-3 (B3)
CHANCE = {"file_auc": ("chance", None, 0.5), "file_accuracy": ("always 'malignant'", "pathology_majority_accuracy", None),
          "density_accuracy": ("always most common class", "density_majority_accuracy", None)}


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=TEXT2, length=0)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def _ref_line(ax, y, dashes):
    ax.axhline(y, color=TEXT, lw=1.2, ls=(0, dashes), zorder=4)


def plot_leakage(folds: pd.DataFrame, path) -> None:
    protocols = [p for p in SERIES if p in set(folds["protocol"])]
    metrics = [("file_auc", "Malignancy AUC"), ("file_accuracy", "Malignancy accuracy"),
               ("density_accuracy", "Density accuracy (4 classes)")]
    metrics = [m for m in metrics if m[0] in folds]
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.2 * len(metrics), 4.9), facecolor=SURFACE, sharey=True)
    n = len(protocols)
    for ax, (col, title) in zip(np.atleast_1d(axes), metrics):
        _style(ax)
        for i, p in enumerate(protocols):
            vals = folds.loc[folds["protocol"] == p, col].to_numpy()
            ax.bar(i, vals.mean(), width=0.64, color=SERIES[p], edgecolor=SURFACE, linewidth=2, zorder=2)
            ax.scatter(i + np.linspace(-0.14, 0.14, len(vals)), vals, s=20, color=TEXT, zorder=3, linewidths=0)
            ax.text(i, 0.04, f"{vals.mean():.2f}", ha="center", va="bottom", color="white",
                    fontsize=13, fontweight="bold", zorder=4)
        _ref_line(ax, PAPER_REPORTED[col], (5, 3))
        key = f"- - -  paper reports {PAPER_REPORTED[col]:.3f}"
        label, base_col, base_val = CHANCE[col]
        if base_col and base_col in folds:
            base_val = folds.loc[folds["protocol"] == "patient", base_col].mean() if "patient" in protocols \
                else folds[base_col].mean()
        if base_val is not None:
            _ref_line(ax, base_val, (1, 2))
            key += f"      \u00b7\u00b7\u00b7\u00b7  {label}: {base_val:.2f}"
        ax.text(0, 1.015, key, transform=ax.transAxes, ha="left", va="bottom", color=TEXT2, fontsize=8.5)
        ax.set_xticks(range(n), [LABELS[p] for p in protocols], color=TEXT, fontsize=9)
        ax.set_ylim(0, 1.06)
        ax.set_title(title, loc="left", color=TEXT, fontsize=12, fontweight="bold", pad=22)
    np.atleast_1d(axes)[0].set_ylabel("5-fold cross-validation (dots = folds)", color=TEXT2)
    fig.suptitle("Same model, same images - only the train/test split changes", x=0.01, ha="left",
                 color=TEXT, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
