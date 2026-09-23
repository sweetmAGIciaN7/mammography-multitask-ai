"""Phase 2 - the leakage experiment.

Question: does the paper's ~93% accuracy come from the model, or from the way
the data was split?

We train the *same* multi-task attention model on the *same* pre-augmented
INbreast mass images three times, changing only the cross-validation split:

    random   -> copies of one mammogram in train and test (the paper's protocol)
    image    -> all copies of one mammogram in the same fold
    patient  -> all mammograms of one patient in the same fold

Usage (Kaggle):
    python -m mammo.experiments.leakage --smoke        # 2-minute check
    python -m mammo.experiments.leakage                # full run, ~30-45 min on a T4
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from mammo.data import build_masses_table, describe, locate_kaggle_inputs
from mammo.metrics import binary_metrics, bootstrap_ci, multiclass_metrics, per_original
from mammo.splits import PROTOCOLS, leakage_report, make_folds


def _smoke_subset(df: pd.DataFrame, n_per_class: int = 6, copies: int = 6, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    firsts = df.groupby("image_id")["pathology"].first()
    keep = []
    for label in (0, 1):
        ids = firsts[firsts == label].index.to_numpy()
        keep += list(rng.choice(ids, size=min(n_per_class, len(ids)), replace=False))
    sub = df[df["image_id"].isin(keep)]
    return sub.groupby("image_id").head(copies).reset_index(drop=True)


def summarise(folds: pd.DataFrame) -> pd.DataFrame:
    cols = ["file_auc", "file_accuracy", "original_auc", "original_accuracy", "density_accuracy",
            "pathology_majority_accuracy", "density_majority_accuracy",
            "test_files_whose_original_is_in_train", "test_files_whose_patient_is_in_train"]
    agg = folds.groupby("protocol")[cols].agg(["mean", "std"])
    agg.columns = [f"{c}_{s}" for c, s in agg.columns]
    return agg.reindex([p for p in PROTOCOLS if p in agg.index])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default="/kaggle/input")
    ap.add_argument("--out", default="/kaggle/working/results/leakage")
    ap.add_argument("--protocols", nargs="+", default=list(PROTOCOLS), choices=list(PROTOCOLS))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--backbone", default="efficientnet_b0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true", help="tiny subset, 1 epoch, 2 folds - checks everything runs")
    args = ap.parse_args(argv)

    import torch  # imported late so --help works without torch
    from mammo.train import TrainConfig, build_model, load_images, predict, set_seed, train_model

    out = Path(args.out + ("_smoke" if args.smoke else ""))
    out.mkdir(parents=True, exist_ok=True)
    log_file = open(out / "log.txt", "w")

    def log(msg=""):
        print(msg, flush=True)
        log_file.write(msg + "\n"); log_file.flush()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not args.smoke:
        log("ERROR: no GPU. In Kaggle: Settings -> Accelerator -> GPU T4 x2 (or P100), then Run All again.")
        return 1

    paths = locate_kaggle_inputs(args.data_root)
    df = build_masses_table(paths["masses_inbreast"], paths["inbreast_release"])
    log("=== Data ===")
    log(describe(df))
    if args.smoke:
        df = _smoke_subset(df)
        args.folds, args.epochs = 2, 1
        log(f"[smoke] using {len(df)} files from {df['image_id'].nunique()} originals")
    df.to_csv(out / "index.csv", index=False)

    cfg = TrainConfig(backbone=args.backbone, epochs=args.epochs, seed=args.seed)
    log(f"\n=== Config ===\n{json.dumps(cfg.to_dict(), indent=1)}\ndevice: {device}")

    t0 = time.time()
    X = torch.from_numpy(load_images(df["path"], cfg.image_size))
    log(f"loaded {tuple(X.shape)} images in {time.time() - t0:.0f}s")
    y_path = df["pathology"].to_numpy()
    y_dens = df["density"].to_numpy()

    rows, preds = [], []
    for protocol in args.protocols:
        log(f"\n=== Protocol: {protocol} - {PROTOCOLS[protocol]} ===")
        for k, (tr, te) in enumerate(make_folds(df, protocol, args.folds, args.seed)):
            leak = leakage_report(df, tr, te)
            log(f"  fold {k + 1}/{args.folds}: train {leak['n_train']} / test {leak['n_test']} files "
                f"({leak['n_test_originals']} originals); test files with their original in train: "
                f"{leak['test_files_whose_original_is_in_train']:.0%}")
            set_seed(args.seed + k)
            model = build_model(cfg).to(device)
            train_model(model, X, y_path, y_dens, tr, cfg, device, log=log)
            p = predict(model, X, te, device)

            m_file = binary_metrics(y_path[te], p["pathology"])
            yo, po = per_original(df["image_id"].to_numpy()[te], y_path[te], p["pathology"])
            m_orig = binary_metrics(yo, po)
            m_dens = multiclass_metrics(y_dens[te], p["density"])
            # "always predict the most common training class" baselines
            path_major = int(np.bincount(y_path[tr]).argmax())
            dens_tr = y_dens[tr][y_dens[tr] >= 0]
            dens_major = int(np.bincount(dens_tr).argmax()) if len(dens_tr) else -1
            dens_te = y_dens[te][y_dens[te] >= 0]
            row = {"protocol": protocol, "fold": k + 1, **leak,
                   "pathology_majority_accuracy": float((y_path[te] == path_major).mean()),
                   "density_majority_accuracy": float((dens_te == dens_major).mean()) if len(dens_te) else float("nan"),
                   **{f"file_{n}": v for n, v in m_file.items()},
                   **{f"original_{n}": v for n, v in m_orig.items()},
                   **{f"density_{n}": v for n, v in m_dens.items()}}
            rows.append(row)
            log(f"    -> file AUC {m_file['auc']:.3f}  acc {m_file['accuracy']:.3f} | per-original AUC "
                f"{m_orig['auc']:.3f} | density acc {m_dens['accuracy']:.3f}")
            preds.append(pd.DataFrame({"protocol": protocol, "fold": k + 1, "row": te,
                                       "image_id": df["image_id"].to_numpy()[te], "y": y_path[te],
                                       "p_malignant": p["pathology"], "density": y_dens[te],
                                       "density_pred": p["density"].argmax(1)}))
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    folds = pd.DataFrame(rows)
    folds.to_csv(out / "fold_metrics.csv", index=False)
    allp = pd.concat(preds)
    allp.to_csv(out / "predictions.csv", index=False)
    summary = summarise(folds)
    summary.to_csv(out / "summary.csv")

    # Pooled out-of-fold AUC with a cluster bootstrap CI (resampling whole original images).
    pooled = {}
    for protocol, g in allp.groupby("protocol"):
        lo, hi = bootstrap_ci(g["y"].to_numpy(), g["p_malignant"].to_numpy(), groups=g["image_id"].to_numpy(),
                              n=200 if args.smoke else 1000)
        pooled[protocol] = {"pooled_file_auc": binary_metrics(g["y"], g["p_malignant"])["auc"],
                            "ci95_clustered": [lo, hi]}
    json.dump({"config": cfg.to_dict(), "args": vars(args), "pooled": pooled,
               "summary": json.loads(summary.to_json(orient="index"))}, open(out / "summary.json", "w"), indent=1)

    from mammo.plots import plot_leakage
    plot_leakage(folds, out / "leakage_chart.png")

    log("\n=== RESULT (mean over folds) ===")
    log(summary[["file_auc_mean", "file_accuracy_mean", "original_auc_mean", "density_accuracy_mean"]]
        .round(3).to_string())
    log(f"\nPooled AUC with 95% CI (bootstrap over original images): {json.dumps(pooled, indent=1)}")
    log(f"\nTotal time {(time.time() - t0) / 60:.1f} min. Outputs in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
