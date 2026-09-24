"""Phase 6: one-page overview of the whole study, built only from the committed results (no GPU, ~10 s).

    PYTHONPATH=src python -m mammo.experiments.overview

Writes ``results/overview/``:
* ``summary.json``: every headline number used in the README, model card and report, each traced to its source file
  (plus patient-bootstrap CIs for the official CBIS-DDSM test split, which Phase 3/4 reported without CIs);
* ``overview_chart.png``: malignancy AUC under each evaluation protocol, from the paper's claim to honest numbers;
* ``model_diagram.png``: the architecture that was replicated.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mammo.metrics import auc_fn, bootstrap_ci, density_qwk_fn

ROOT = Path(__file__).resolve().parents[3]
PAPER = {"auc": 0.962, "accuracy": 0.936}  # Esen et al. 2025, Table 2 (EfficientNet-B3, hold-out)


def _j(rel: str) -> dict:
    return json.load(open(ROOT / "results" / rel))


def collect(nboot: int = 2000) -> dict:
    leak = _j("leakage/summary.json")
    cbis = _j("cbis/summary.json")
    ext = _j("external/summary.json")
    att = _j("attention/summary.json")
    cal = _j("calibration/summary.json")
    off = _j("external/mt_cbam/cbis_official_metrics.json")

    pred = pd.read_csv(ROOT / "results/external/mt_cbam/cbis_official_predictions.csv")
    dens = pred[pred["density"] >= 0]
    prob = dens[[f"p_density_{k}" for k in range(4)]].to_numpy()
    off_auc_ci = bootstrap_ci(pred["y"].to_numpy(), pred["p_malignant"].to_numpy(), auc_fn, nboot, 0,
                              pred["patient"].to_numpy())
    off_qwk_ci = bootstrap_ci(dens["density"].to_numpy(), prob, density_qwk_fn, nboot, 0, dens["patient"].to_numpy())

    cv = cbis["results"]["mt_cbam"]
    ib = ext["results"]["mt_cbam"]
    loc = att["localisation"]

    def L(key):
        m = loc[key]["subsets"]["all"]
        return {k: {"mean": m[k]["mean"], "ci95": m[k]["ci95"]} for k in ("pg_px_tol", "energy_px", "auc_px")}

    s = {"paper": PAPER,
         "leakage": {p: {"auc_fold_mean": leak["summary"][p]["file_auc_mean"],
                         "auc_fold_sd": leak["summary"][p]["file_auc_std"],
                         "auc_pooled": leak["pooled"][p]["pooled_file_auc"],
                         "auc_pooled_ci95": leak["pooled"][p]["ci95_clustered"],
                         "density_accuracy": leak["summary"][p]["density_accuracy_mean"],
                         "test_patient_in_train": leak["summary"][p]["test_files_whose_patient_is_in_train_mean"]}
                     for p in ("random", "image", "patient")},
         "cbis_cv_mt_cbam": {"auc": cv["malignancy"]["auc_pooled"], "auc_ci95": cv["malignancy"]["auc_ci95"],
                             "qwk": cv["density"]["qwk"], "qwk_ci95": cv["density"]["qwk_ci95"],
                             "n_images": cv["n_images"], "n_patients": cv["n_patients"]},
         "cbis_official": {"auc": off["path_auc"], "auc_ci95": list(off_auc_ci), "qwk": off["dens_qwk"],
                           "qwk_ci95": list(off_qwk_ci), "n_images": off["n_test"],
                           "n_patients": off["test_patients"], "sensitivity": off["path_sensitivity"],
                           "specificity": off["path_specificity"], "density_accuracy": off["dens_accuracy"]},
         "inbreast_mt_cbam": {"density_qwk": ib["inbreast_density"]["qwk"],
                              "density_qwk_ci95": ib["inbreast_density"]["qwk_ci95"],
                              "within_one_grade": ib["inbreast_density"]["within_one_grade"],
                              "proxy_auc": ib["inbreast_malignancy_proxy"]["auc"],
                              "proxy_auc_ci95": ib["inbreast_malignancy_proxy"]["auc_ci95"]},
         "calibration_official_transfer": {"platt": cal["official_transfer"]["platt"],
                                           "ece_uncalibrated": cal["official_transfer"]["malignancy"]["uncalibrated"]["ece"],
                                           "ece_platt": cal["official_transfer"]["malignancy"]["platt"]["ece"]},
         "localisation": {k: L(k) for k in ("cbam:mt_cbam", "gradcam_mal:mt_cbam", "base:brightness", "base:uniform", "ref:oracle")
                          if k in loc},
         "sources": {"leakage": "results/leakage/summary.json", "cbis_cv": "results/cbis/summary.json",
                     "cbis_official": "results/external/mt_cbam/cbis_official_{metrics.json,predictions.csv}",
                     "inbreast": "results/external/summary.json", "localisation": "results/attention/summary.json",
                     "calibration": "results/calibration/summary.json"}}
    return s


def auc_rows(s: dict) -> list[dict]:
    """Rows of the overview chart, top to bottom."""
    L = s["leakage"]
    cv = {"auc": s["cbis_cv_mt_cbam"]["auc"], "ci": s["cbis_cv_mt_cbam"]["auc_ci95"]}
    return [
        {"label": "Paper, as reported\n(INbreast, split after augmentation)", "auc": s["paper"]["auc"], "ci": None,
         "group": "leaky", "note": "reported"},
        {"label": "Our replication of the paper's protocol\n(same data, split after augmentation)",
         "auc": L["random"]["auc_pooled"], "ci": L["random"]["auc_pooled_ci95"], "group": "leaky",
         "note": "test copies' originals all in training"},
        {"label": "Same data, grouped by original image", "auc": L["image"]["auc_pooled"],
         "ci": L["image"]["auc_pooled_ci95"], "group": "partial",
         "note": f"{L['image']['test_patient_in_train']:.0%} of test images: same patient in training"},
        {"label": "Same data, grouped by patient\n(106 images, 50 patients)", "auc": L["patient"]["auc_pooled"],
         "ci": L["patient"]["auc_pooled_ci95"], "group": "honest", "note": "no patient overlap"},
        {"label": "CBIS-DDSM, patient-level 5-fold CV\n(2,802 images, 1,460 patients)", "auc": cv["auc"],
         "ci": cv["ci"], "group": "honest", "note": "biopsy-confirmed labels"},
        {"label": "CBIS-DDSM, official test split\n(368 images, 212 unseen patients)",
         "auc": s["cbis_official"]["auc"], "ci": s["cbis_official"]["auc_ci95"], "group": "honest",
         "note": "trained once on the official training split"},
    ]


GROUP = {"leaky": ("#2a78d6", "Leaky: copies of test images in training"),
         "partial": ("#eb6834", "Partly leaky: same patient in training"),
         "honest": ("#1baf7a", "Leakage-free: patient-level split")}


def plot_overview(s: dict, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mammo.plots import GRID, SURFACE, TEXT, TEXT2

    rows = auc_rows(s)
    n = len(rows)
    fig, ax = plt.subplots(figsize=(10.5, 5.6), facecolor=SURFACE)
    fig.subplots_adjust(left=0.36, right=0.97, top=0.80, bottom=0.12)
    ax.set_facecolor(SURFACE)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=TEXT2, length=0)
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ys = np.arange(n)[::-1]
    for y, r in zip(ys, rows):
        col = GROUP[r["group"]][0]
        if r["ci"] is not None and r["ci"][1] - r["ci"][0] > 1e-6:
            ax.plot(r["ci"], [y, y], color=col, lw=2.2, solid_capstyle="round", zorder=2)
        hollow = r["note"] == "reported"
        ax.scatter([r["auc"]], [y], s=90, zorder=3, color=SURFACE if hollow else col, edgecolors=col,
                   linewidths=2.2 if hollow else 1.5)
        ax.text(r["auc"] + (0.012 if r["auc"] < 0.97 else -0.012), y + 0.28, f"{r['auc']:.3f}",
                ha="left" if r["auc"] < 0.97 else "right", va="center", fontsize=10, fontweight="bold", color=TEXT)
    ax.axvline(0.5, color=TEXT, lw=1.1, ls=(0, (1, 2)))
    ax.text(0.505, -0.75, "chance", color=TEXT2, fontsize=8.5, va="center")
    ax.set_yticks(ys)
    ax.set_yticklabels([r["label"] for r in rows], fontsize=9, color=TEXT)
    ax.set_ylim(-1, n - 0.4)
    ax.set_xlim(0.45, 1.03)
    ax.set_xlabel("Malignancy AUC (dot) with 95% CI (line; bootstrap over patients or original images)",
                  color=TEXT2, fontsize=8.5)
    fig.text(0.01, 0.955, "Same architecture, different evaluation: the published number measures leakage",
             fontsize=13, fontweight="bold", color=TEXT)
    fig.text(0.01, 0.905, "EfficientNet + CBAM, two heads (malignancy, density). Hollow dot: value reported in the "
             "paper (no CI given).", fontsize=9, color=TEXT2)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", ls="", ms=8, color=GROUP[k][0], label=GROUP[k][1])
               for k in ("leaky", "partial", "honest")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.005, 0.89), ncol=3, frameon=False,
               fontsize=8.8, handletextpad=0.3, columnspacing=1.6, labelcolor=TEXT)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_model(path) -> None:
    """Block diagram of the replicated model (matplotlib boxes, no external tools)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    from mammo.plots import SURFACE, TEXT, TEXT2

    fig, ax = plt.subplots(figsize=(11, 3.3), facecolor=SURFACE)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 3.3)
    ax.axis("off")
    boxes = [
        (0.1, 1.05, 1.45, 1.2, "Mammogram\n640 × 384\ncropped, chest\nwall left", "#e4e3df"),
        (1.9, 1.05, 1.6, 1.2, "EfficientNet-B0\n(ImageNet-\npretrained)", "#cfe0f5"),
        (3.85, 1.05, 1.1, 1.2, "features\n20 × 12\n× 1280", "#f4f8fd"),
        (5.3, 1.05, 1.45, 1.2, "CBAM\nchannel gate\n→ 7×7 spatial\ngate (sigmoid)", "#cfe0f5"),
        (7.1, 1.05, 1.35, 1.2, "avg pool →\nshared MLP\n512 → 256", "#cfe0f5"),
    ]
    for x, y, w, h, t, c in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=c, ec="#8a8984",
                                    lw=0.8))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=8.6, color=TEXT)
    heads = [(8.85, 2.0, "Malignancy head\n128 → 1 (sigmoid)\nloss weight 2.0"),
             (8.85, 0.1, "Density head\n128 → 4 (softmax, A–D)\nloss weight 0.8")]
    for x, y, t in heads:
        ax.add_patch(FancyBboxPatch((x, y), 2.0, 1.05, boxstyle="round,pad=0.02,rounding_size=0.08", fc="#d4f0e4",
                                    ec="#8a8984", lw=0.8))
        ax.text(x + 1.0, y + 0.525, t, ha="center", va="center", fontsize=8.6, color=TEXT)
    arrows = [((1.55, 1.65), (1.9, 1.65)), ((3.5, 1.65), (3.85, 1.65)), ((4.95, 1.65), (5.3, 1.65)),
              ((6.75, 1.65), (7.1, 1.65)), ((8.45, 1.85), (8.85, 2.45)), ((8.45, 1.45), (8.85, 0.7))]
    for a, b in arrows:
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=10, color=TEXT2, lw=1.0))
    ax.text(4.4, 0.62, "Grad-CAM\ntaken here", ha="center", fontsize=7.6,
            color=TEXT2, style="italic")
    ax.text(6.02, 0.62, "the paper's\n'attention map'", ha="center", fontsize=7.6, color=TEXT2,
            style="italic")
    ax.text(0.1, 3.05, "Replicated model (paper Sec. IV-B): one network, two outputs", fontsize=11,
            fontweight="bold", color=TEXT)
    ax.text(0.1, 2.72, "Ablations switch off CBAM (mt_plain) or one head (st_path, st_dens); everything else identical.",
            fontsize=8.5, color=TEXT2)
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "results" / "overview"))
    ap.add_argument("--nboot", type=int, default=2000)
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    s = collect(nboot=args.nboot)
    s["overview_rows"] = auc_rows(s)
    json.dump(s, open(out / "summary.json", "w"), indent=1, default=float)
    plot_overview(s, out / "overview_chart.png")
    plot_model(out / "model_diagram.png")
    for r in s["overview_rows"]:
        ci = f"({r['ci'][0]:.3f}-{r['ci'][1]:.3f})" if r["ci"] else ""
        print(f"{r['auc']:.3f} {ci:16s} {r['label'].replace(chr(10), ' ')}")
    o = s["cbis_official"]
    print(f"official test: AUC {o['auc']:.3f} {o['auc_ci95']}, QWK {o['qwk']:.3f} {o['qwk_ci95']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
