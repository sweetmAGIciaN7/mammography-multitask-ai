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


# Phase 3 ablation. Colour follows the model variant in every panel (categorical slots 1-4, validated).
VARIANT_COLORS = {"mt_cbam": "#2a78d6", "mt_plain": "#eb6834", "st_path": "#1baf7a", "st_dens": "#eda100"}
VARIANT_LABELS = {"mt_cbam": "Multi-task\n+ CBAM\n(paper design)", "mt_plain": "Multi-task\nno attention",
                  "st_path": "Malignancy\nonly + CBAM", "st_dens": "Density\nonly + CBAM"}


def plot_ablation(table: dict, comparisons: list, path) -> None:
    """Two panels: malignancy AUC and density accuracy per variant, pooled out-of-fold with 95% CI."""
    panels = [("malignancy", "auc_pooled", "auc_ci95", "Malignancy AUC (benign vs malignant)",
               [("paper reports", PAPER_REPORTED["file_auc"], (5, 3)), ("chance", 0.5, (1, 2))]),
              ("density", "accuracy", "accuracy_ci95", "Density accuracy (4 BI-RADS grades)",
               [("paper reports", PAPER_REPORTED["density_accuracy"], (5, 3))])]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.0), facecolor=SURFACE)
    for ax, (task, key, ci_key, title, refs) in zip(axes, panels):
        _style(ax)
        names = [n for n in VARIANT_COLORS if n in table and task in table[n]]
        refs = list(refs)
        if task == "density" and names:
            refs.append(("always most common grade", table[names[0]][task]["majority_baseline_accuracy"], (1, 2)))
        for i, n in enumerate(names):
            r = table[n][task]
            v, (lo, hi) = r[key], r[ci_key]
            ax.bar(i, v, width=0.62, color=VARIANT_COLORS[n], edgecolor=SURFACE, linewidth=2, zorder=2)
            ax.errorbar(i, v, yerr=[[v - lo], [hi - v]], color=TEXT, lw=1.4, capsize=5, zorder=3)
            ax.text(i, hi + 0.02, f"{v:.3f}", ha="center", va="bottom", color=TEXT, fontsize=11,
                    fontweight="bold", zorder=4)
        key_txt = []
        for label, y, dashes in refs:
            _ref_line(ax, y, dashes)
            key_txt.append(("- - -" if dashes[0] > 1 else "····") + f"  {label} {y:.2f}")
        ax.text(0, 1.015, "      ".join(key_txt), transform=ax.transAxes, ha="left", va="bottom", color=TEXT2,
                fontsize=8.5)
        ax.set_xticks(range(len(names)), [VARIANT_LABELS[n] for n in names], color=TEXT, fontsize=9)
        ax.set_ylim(0, 1.06)
        ax.set_title(title, loc="left", color=TEXT, fontsize=12, fontweight="bold", pad=22)
    axes[0].set_ylabel("Patient-level 5-fold CV, pooled (bars = 95% CI)", color=TEXT2)
    fig.suptitle("CBIS-DDSM, leakage-free: does multi-task learning or attention help?", x=0.01, ha="left",
                 color=TEXT, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


# Phase 4 reliability diagrams. Colour follows the calibration method (categorical slots 1-3, validated).
CALIB_COLORS = {"uncalibrated": "#eb6834", "temperature": "#2a78d6", "platt": "#1baf7a"}
CALIB_LABELS = {"uncalibrated": "As trained", "temperature": "Temperature scaling", "platt": "Platt scaling"}


def plot_reliability(curves: dict, stats: dict, path) -> None:
    """Reliability diagrams: predicted probability (x) vs how often it was right (y). Diagonal = perfect."""
    panels = [(t, title, xl, yl) for t, title, xl, yl in [
        ("malignancy", "Malignancy", "Predicted P(malignant)", "Observed share malignant"),
        ("density", "Density grade (top label)", "Confidence in the predicted grade", "Observed share correct")]
        if t in curves]
    fig, axes = plt.subplots(1, len(panels), figsize=(5.4 * len(panels), 5.2), facecolor=SURFACE)
    for ax, (task, title, xl, yl) in zip(np.atleast_1d(axes), panels):
        ax.set_facecolor(SURFACE)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=TEXT2, length=0)
        ax.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.plot([0, 1], [0, 1], color=TEXT, lw=1.2, ls=(0, (5, 3)), zorder=1)
        ax.text(0.98, 0.03, "- - -  perfectly calibrated", color=TEXT2, fontsize=8.5, ha="right", va="bottom")
        for name, c in curves[task].items():
            ece = stats[task][name]["ece"]
            ax.plot(c["mean_predicted"], c["observed"], color=CALIB_COLORS[name], lw=2, zorder=3,
                    label=f"{CALIB_LABELS[name]}  (ECE {ece:.3f})")
            ax.scatter(c["mean_predicted"], c["observed"], s=[12 + 60 * n / max(c["count"]) for n in c["count"]],
                       color=CALIB_COLORS[name], edgecolors=SURFACE, linewidths=1.5, zorder=4)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel(xl, color=TEXT2); ax.set_ylabel(yl, color=TEXT2)
        ax.set_title(title, loc="left", color=TEXT, fontsize=12, fontweight="bold")
        leg = ax.legend(loc="upper left", frameon=False, fontsize=9)
        for t in leg.get_texts():
            t.set_color(TEXT)
    fig.suptitle("Can the probabilities be taken literally? (CBIS-DDSM, out-of-fold; dot size = images in bin)",
                 x=0.01, ha="left", color=TEXT, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


# Phase 4 external validation. Colour follows the test set (categorical slots 1-2, validated).
DOMAIN_COLORS = {"cbis_official_density": "#2a78d6", "inbreast_density": "#eb6834"}
DOMAIN_LABELS = {"cbis_official_density": "CBIS-DDSM official test (film, same source as training)",
                 "inbreast_density": "INbreast (digital, never seen)"}
SHORT_LABELS = {"mt_cbam": "Multi-task\n+ CBAM\n(paper design)", "mt_plain": "Multi-task\nno attention",
                "st_dens": "Density\nonly + CBAM"}


def plot_external(results: dict, path) -> None:
    """Left: density QWK internal vs external per variant. Right: INbreast confusion matrix of the main model."""
    names = [n for n in SHORT_LABELS if n in results and "inbreast_density" in results[n]]
    if not names:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), facecolor=SURFACE, gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    _style(ax)
    w = 0.36
    for j, dom in enumerate(DOMAIN_COLORS):
        for i, n in enumerate(names):
            r = results[n].get(dom)
            if r is None:
                continue
            x = i + (j - 0.5) * (w + 0.03)
            v, (lo, hi) = r["qwk"], r["qwk_ci95"]
            ax.bar(x, v, width=w, color=DOMAIN_COLORS[dom], edgecolor=SURFACE, linewidth=2, zorder=2,
                   label=DOMAIN_LABELS[dom] if i == 0 else None)
            if not np.isfinite(v):
                continue
            if np.isfinite(lo) and np.isfinite(hi):
                ax.errorbar(x, v, yerr=[[max(v - lo, 0)], [max(hi - v, 0)]], color=TEXT, lw=1.3, capsize=4, zorder=3)
            else:
                hi = v
            ax.text(x, max(hi, v) + 0.02, f"{v:.2f}", ha="center", va="bottom", color=TEXT, fontsize=10,
                    fontweight="bold")
    ax.set_xticks(range(len(names)), [SHORT_LABELS[n] for n in names], color=TEXT, fontsize=9)
    ax.set_ylim(0, 1.06)
    ax.set_ylabel("Density QWK (0 = chance, 1 = perfect)", color=TEXT2)
    ax.set_title("Density grading: internal vs external test", loc="left", color=TEXT, fontsize=12,
                 fontweight="bold", pad=10)
    leg = ax.legend(loc="upper right", frameon=False, fontsize=8.5)
    for t in leg.get_texts():
        t.set_color(TEXT)

    ax = axes[1]
    main = "mt_cbam" if "mt_cbam" in names else names[0]
    cm = np.asarray(results[main]["inbreast_density"]["confusion"], float)
    rows = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    ax.imshow(rows, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("seq", ["#f4f8fd", "#2a78d6", "#0b3a75"]),
              vmin=0, vmax=1)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{int(cm[i, j])}", ha="center", va="center", fontsize=11,
                    color="white" if rows[i, j] > 0.5 else TEXT)
    ax.set_xticks(range(4), list("ABCD"), color=TEXT); ax.set_yticks(range(4), list("ABCD"), color=TEXT)
    ax.set_xlabel("Predicted grade", color=TEXT2); ax.set_ylabel("True ACR grade (INbreast)", color=TEXT2)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("Paper design on INbreast\n(counts; shade = share of each true grade)",
                 loc="left", color=TEXT, fontsize=11, fontweight="bold")
    fig.suptitle("Trained on scanned film (CBIS-DDSM), tested on digital mammograms (INbreast)", x=0.01, ha="left",
                 color=TEXT, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
