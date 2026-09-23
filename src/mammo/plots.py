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

PAPER_REPORTED = {"file_auc": 0.962, "file_accuracy": 0.936}  # Esen et al. 2025, Table 2 (EfficientNet-B3)


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=TEXT2, length=0)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def plot_leakage(folds: pd.DataFrame, path) -> None:
    protocols = [p for p in SERIES if p in set(folds["protocol"])]
    metrics = [("file_auc", "Malignancy AUC"), ("file_accuracy", "Malignancy accuracy")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), facecolor=SURFACE, sharey=True)
    for ax, (col, title) in zip(axes, metrics):
        _style(ax)
        for i, p in enumerate(protocols):
            vals = folds.loc[folds["protocol"] == p, col].to_numpy()
            ax.bar(i, vals.mean(), width=0.62, color=SERIES[p], edgecolor=SURFACE, linewidth=2, zorder=2)
            jitter = np.linspace(-0.12, 0.12, len(vals))
            ax.scatter(i + jitter, vals, s=22, color=TEXT, zorder=3, linewidths=0)
            ax.text(i, vals.mean() + 0.03, f"{vals.mean():.2f}", ha="center", va="bottom",
                    color=TEXT, fontsize=12, fontweight="bold")
        ref = PAPER_REPORTED[col]
        ax.axhline(ref, color=TEXT2, lw=1.2, ls=(0, (4, 3)), zorder=1)
        ax.text(len(protocols) - 0.5, ref + 0.015, f"paper reports {ref:.3f}", ha="right", va="bottom",
                color=TEXT2, fontsize=9)
        ax.axhline(0.5, color=GRID, lw=1, zorder=1)
        ax.set_xticks(range(len(protocols)), [LABELS[p] for p in protocols], color=TEXT, fontsize=9.5)
        ax.set_ylim(0, 1.08)
        ax.set_title(title, loc="left", color=TEXT, fontsize=12, fontweight="bold")
    axes[0].set_ylabel("5-fold cross-validation (dots = folds)", color=TEXT2)
    fig.suptitle("Same model, same images - only the train/test split changes", x=0.01, ha="left",
                 color=TEXT, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
