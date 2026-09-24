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


# Phase 5: attention vs lesion masks. Colour follows the lesion subset (categorical slots 1-3, validated).
LESION_COLORS = {"all": "#2a78d6", "mass only": "#eb6834", "calcification only": "#1baf7a"}
LESION_OUTLINE = "#eb6834"
MAP_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list("seq_blue", ["#f4f8fd", "#2a78d6", "#0b3a75"])


def mask_preview(images, lesion, breast, titles, path, ncols: int = 8) -> None:
    """Preprocessed test images with the mapped ROI outline (orange) and the breast region used for scoring (grey)."""
    n = len(images)
    nrows = max(1, int(np.ceil(n / ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.6, nrows * 2.8), facecolor=SURFACE)
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, a, les, b, t in zip(np.atleast_1d(axes).ravel(), images, lesion, breast, titles):
        ax.imshow(a, cmap="gray", vmin=0, vmax=255)
        ax.contour(b, levels=[0.5], colors=["#9a9994"], linewidths=0.6)
        if les.any():
            ax.contour(les, levels=[0.5], colors=[LESION_OUTLINE], linewidths=1.0)
        ax.set_title(t, fontsize=6.5, color=TEXT)
    fig.suptitle("ROI masks mapped into model space (orange) and breast region used for scoring (grey); random test images",
                 x=0.01, ha="left", color=TEXT, fontsize=10, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=110, facecolor=SURFACE)
    plt.close(fig)


def attention_gallery(images, lesion, breast, maps: dict, titles, path, per_row: int = 2) -> None:
    """For each case: the image with the lesion outline, then each map (upsampled bilinearly, min-max scaled within
    the breast for display) with the same outline and a marker at the map's maximum inside the breast."""
    from mammo.localization import upsample
    n, k = len(images), 1 + len(maps)
    rows = int(np.ceil(n / per_row))
    fig, axes = plt.subplots(rows, per_row * k, figsize=(per_row * k * 1.45, rows * 2.75 + 1.1), facecolor=SURFACE)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=1 - 1.1 / (rows * 2.75 + 1.1), wspace=0.06, hspace=0.2)
    axes = np.atleast_2d(axes)
    for ax in axes.ravel():
        ax.axis("off")
    H, W = images.shape[1:]
    for i in range(n):
        r, c0 = divmod(i, per_row)
        c0 *= k
        ax = axes[r, c0]
        ax.imshow(images[i], cmap="gray", vmin=0, vmax=255)
        if lesion[i].any():
            ax.contour(lesion[i], levels=[0.5], colors=[LESION_OUTLINE], linewidths=0.9)
        ax.set_title(titles[i], fontsize=6, color=TEXT, loc="left")
        for j, (lab, m) in enumerate(maps.items()):
            ax = axes[r, c0 + 1 + j]
            raw = m[i].astype(float)
            up = upsample(raw, H, W)
            inside = up[breast[i]]
            lo, hi = inside.min(), inside.max()
            disp = np.where(breast[i], (up - lo) / (hi - lo) if hi > lo else 0.0, np.nan)
            ax.imshow(images[i], cmap="gray", vmin=0, vmax=255, alpha=0.55)
            ax.imshow(disp, cmap=MAP_CMAP, vmin=0, vmax=1, alpha=0.6)
            if lesion[i].any():
                ax.contour(lesion[i], levels=[0.5], colors=[LESION_OUTLINE], linewidths=0.9)
            yx = np.unravel_index(np.nanargmax(np.where(breast[i], up, -np.inf)), up.shape)
            hit = bool(lesion[i][yx])
            ax.scatter([yx[1]], [yx[0]], marker="x", s=38, c="white" if hit else TEXT, linewidths=1.8, zorder=5)
            ax.scatter([yx[1]], [yx[0]], marker="x", s=38, c=TEXT if hit else "white", linewidths=0.6, zorder=6)
            ax.set_title(f"{lab}\nraw {raw.min():.2f}–{raw.max():.2f} · {'HIT' if hit else 'miss'}",
                         fontsize=6.5, color=TEXT, loc="left")
    fig.suptitle("Randomly drawn test cases (seed 0, not selected). Orange = radiologist ROI, x = map maximum inside the "
                 "breast (HIT = on the lesion).\nEach map is scaled to its own range inside the breast; 'raw' = its actual "
                 "values.", x=0.01, ha="left", color=TEXT, fontsize=9, fontweight="bold")
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def plot_localisation(s: dict, path) -> None:
    """Dot plot: one row per method, pointing game (±1 cell) and energy-in-lesion with 95% CIs, per lesion subset."""
    from mammo.experiments.attention import method_label
    loc = s["localisation"]
    methods = list(loc)  # already in display order (models, controls, ceiling, baselines)
    panels = [("pg_px_tol", "Pointing game (±1 cell)\nshare of images whose map maximum is on the lesion"),
              ("energy_px", "Energy in lesion\nshare of the map's mass inside the lesion"),
              ("auc_px", "Pixel AUC\nlesion vs other breast pixels (0.5 = no information)")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 0.42 * len(methods) + 2.2), facecolor=SURFACE, sharey=True)
    ypos = np.arange(len(methods))[::-1]
    offs = {"all": 0.22, "mass only": 0.0, "calcification only": -0.22}
    for ax, (key, title) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=TEXT2, length=0)
        ax.xaxis.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        nb = sum(m.startswith(("base", "ref")) for m in methods)
        ax.axhspan(-0.5, nb - 0.5, color="#f1f0ec", zorder=0)
        for y, m in zip(ypos, methods):
            for sub, col in LESION_COLORS.items():
                e = loc[m]["subsets"].get(sub, {}).get(key)
                if not e or not np.isfinite(e["mean"]):
                    continue
                yy = y + offs[sub]
                ax.plot(e["ci95"], [yy, yy], color=col, lw=2, solid_capstyle="round", zorder=2)
                ax.scatter([e["mean"]], [yy], s=40, color=col, edgecolors=SURFACE, linewidths=1.5, zorder=3,
                           label=sub if (y == ypos[0]) else None)
        if key == "auc_px":
            ax.axvline(0.5, color=TEXT, lw=1.1, ls=(0, (5, 3)))
        ax.set_title(title, loc="left", color=TEXT, fontsize=10, fontweight="bold")
    axes[0].set_yticks(ypos, [method_label(m) for m in methods], color=TEXT, fontsize=8.5)
    h, l = axes[0].get_legend_handles_labels()
    leg = fig.legend(h, l, loc="upper right", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.995, 0.985),
                     title="lesion subset (lines = 95% CI)")
    leg.get_title().set_color(TEXT2); leg.get_title().set_fontsize(8.5)
    for t in leg.get_texts():
        t.set_color(TEXT)
    fig.suptitle(f"Do the maps point at the lesion? CBIS-DDSM official test, {s['n_images']} images with ROI masks\n"
                 f"shaded rows: maps that know nothing about lesions, and the best a 20x12 map could do",
                 x=0.01, ha="left", color=TEXT, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
