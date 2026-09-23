"""Phase 3 - a leakage-free multi-task model on CBIS-DDSM, with ablations.

Question: once the split is honest, does the paper's design (one network, two
task heads, CBAM attention) actually help?

Four variants, trained on the *same* patient-level folds so they can be compared
image by image:

    mt_cbam   multi-task (malignancy + density) with CBAM  <- the paper's design
    mt_plain  multi-task, no attention                     <- does attention help?
    st_path   malignancy only, with CBAM                   <- does density help malignancy?
    st_dens   density only, with CBAM                      <- does malignancy help density?

Protocol: 5-fold cross-validation grouped by patient (no patient is ever in train
and test), fixed number of epochs (no early stopping on the test fold), 95% CIs
from a bootstrap over patients, paired bootstrap for the A-vs-B comparisons.
The main model is also trained on the official CBIS-DDSM training split and tested
on the official test split (patients present in both are removed from the test set).

Usage (Kaggle; see notebooks/03_cbis_multitask.ipynb):
    python -m mammo.experiments.cbis prepare                 # index + preprocess once (~5 min)
    python -m mammo.experiments.cbis run mt_cbam st_dens --official mt_cbam --gpu 0
    python -m mammo.experiments.cbis summarise               # tables + chart
    add --smoke to any of them for a 3-minute end-to-end check
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from mammo.cbis import describe_cbis, index_cbis, locate_cbis
from mammo.metrics import (auc_fn, binary_metrics, bootstrap_ci, density_acc_fn, density_qwk_fn,
                           multiclass_metrics, paired_bootstrap)
from mammo.preprocess import cached_images, preview_grid
from mammo.splits import make_folds

CONFIGS = {
    "mt_cbam": {"tasks": ("pathology", "density"), "attention": True, "label": "Multi-task + CBAM (paper design)"},
    "mt_plain": {"tasks": ("pathology", "density"), "attention": False, "label": "Multi-task, no attention"},
    "st_path": {"tasks": ("pathology",), "attention": True, "label": "Malignancy only + CBAM"},
    "st_dens": {"tasks": ("density",), "attention": True, "label": "Density only + CBAM"},
}
# (A, B, task, what the comparison answers)
COMPARISONS = [
    ("mt_cbam", "st_path", "pathology", "Does adding the density task help malignancy?"),
    ("mt_cbam", "st_dens", "density", "Does adding the malignancy task help density?"),
    ("mt_cbam", "mt_plain", "pathology", "Does CBAM attention help malignancy?"),
    ("mt_cbam", "mt_plain", "density", "Does CBAM attention help density?"),
]


class Logger:
    def __init__(self, path: Path | None = None):
        self.f = open(path, "a") if path else None

    def __call__(self, msg: str = ""):
        print(msg, flush=True)
        if self.f:
            self.f.write(msg + "\n"); self.f.flush()


def _smoke_subset(df: pd.DataFrame, n_patients: int = 40, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pat = df.groupby("patient")["pathology"].max()
    keep = []
    for label in (0, 1):
        ids = pat[pat == label].index.to_numpy()
        keep += list(rng.choice(ids, size=min(n_patients // 2, len(ids)), replace=False))
    return df[df["patient"].isin(keep)].reset_index(drop=True)


def load_index(args, log) -> tuple[pd.DataFrame, dict]:
    out = Path(args.out)
    idx_file = out / "index.csv"
    if idx_file.exists():
        return pd.read_csv(idx_file), json.load(open(out / "data_stats.json"))
    loc = locate_cbis(args.data_root)
    log(f"CBIS-DDSM csv : {loc['csv']}\nCBIS-DDSM jpeg: {loc['jpeg']}")
    df, stats = index_cbis(loc["csv"], loc["jpeg"])
    if args.smoke:
        df = _smoke_subset(df)
        stats["smoke_subset_images"] = int(len(df))
    df = df.sort_values("image_id").reset_index(drop=True)
    # Folds are fixed once, here, and shared by every variant (paired comparisons need that).
    df["fold"] = -1
    for k, (_, te) in enumerate(make_folds(df, "patient", n_splits=args.folds, seed=args.seed)):
        df.loc[te, "fold"] = k + 1
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(idx_file, index=False)
    json.dump(stats, open(out / "data_stats.json", "w"), indent=1)
    return df, stats


def cmd_prepare(args) -> int:
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    log = Logger(out / "log_prepare.txt")
    t0 = time.time()
    df, stats = load_index(args, log)
    log("=== CBIS-DDSM ===")
    log(describe_cbis(stats))
    folds = df.groupby("fold").agg(images=("image_id", "size"), patients=("patient", "nunique"),
                                   malignant=("pathology", "mean"))
    log(f"\n=== Patient-level folds ===\n{folds.round(3).to_string()}")
    X = cached_images(df["path"], args.height, args.width, args.cache, workers=args.workers, log=log)
    log(f"images ready: {X.shape} uint8 in {(time.time() - t0) / 60:.1f} min")
    pick = df.groupby(["pathology", "view"]).head(4).index[:16]
    titles = [f"{r.image_id}\n{'malignant' if r.pathology else 'benign'} | "
              f"{'?' if r.density < 0 else 'ABCD'[r.density]}" for r in df.loc[pick].itertuples()]
    preview_grid(X[pick], titles, out / "preprocessing_preview.png")
    log(f"preview written to {out / 'preprocessing_preview.png'}")
    return 0


def _train_eval(cfg, X, df, tr, te, device, log, tasks):
    from mammo.train import build_model, predict, set_seed, train_model
    y_path = df["pathology"].to_numpy()
    y_dens = df["density"].to_numpy()
    if tasks == ("density",):
        tr = tr[y_dens[tr] >= 0]
    set_seed(cfg.seed)
    model = build_model(cfg).to(device)
    train_model(model, X, y_path, y_dens, tr, cfg, device, log=log)
    p = predict(model, X, te, device, batch_size=32)
    return model, p


def _pred_frame(df, te, p, config, fold):
    d = pd.DataFrame({"config": config, "fold": fold, "row": te, "image_id": df["image_id"].to_numpy()[te],
                      "patient": df["patient"].to_numpy()[te], "y": df["pathology"].to_numpy()[te],
                      "density": df["density"].to_numpy()[te]})
    if "pathology" in p:
        d["p_malignant"], d["logit_malignant"] = p["pathology"], p["pathology_logit"]
    if "density" in p:
        for k in range(p["density"].shape[1]):
            d[f"p_density_{k}"] = p["density"][:, k]
            d[f"logit_density_{k}"] = p["density_logits"][:, k]
    return d


def _metric_row(df, tr, te, p):
    y_path, y_dens = df["pathology"].to_numpy(), df["density"].to_numpy()
    row = {"n_train": int(len(tr)), "n_test": int(len(te)), "test_patients": int(df["patient"].iloc[te].nunique()),
           "patients_in_both": int(len(set(df["patient"].iloc[tr]) & set(df["patient"].iloc[te])))}
    if "pathology" in p:
        row.update({f"path_{k}": v for k, v in binary_metrics(y_path[te], p["pathology"]).items()})
        row["path_majority_accuracy"] = float((y_path[te] == int(np.bincount(y_path[tr]).argmax())).mean())
    if "density" in p:
        row.update({f"dens_{k}": v for k, v in multiclass_metrics(y_dens[te], p["density"]).items()})
        known_tr = y_dens[tr][y_dens[tr] >= 0]
        known_te = y_dens[te][y_dens[te] >= 0]
        row["dens_majority_accuracy"] = float((known_te == np.bincount(known_tr).argmax()).mean())
    return row


def cmd_run(args) -> int:
    import torch
    from mammo.train import TrainConfig

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    log = Logger(out / f"log_run_gpu{args.gpu}.txt")
    df, _ = load_index(args, log)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not args.smoke:
        log("ERROR: no GPU. In Kaggle: Settings -> Accelerator -> GPU T4 x2, then run again.")
        return 1
    if device.type == "cuda":
        torch.cuda.set_device(device)
    X = torch.from_numpy(cached_images(df["path"], args.height, args.width, args.cache, workers=args.workers, log=log))
    folds = sorted(df["fold"].unique())
    if args.max_folds:
        folds = folds[:args.max_folds]
    ckpt_dir = Path(args.checkpoints); ckpt_dir.mkdir(parents=True, exist_ok=True)

    for name in args.configs:
        spec = CONFIGS[name]
        cfg = TrainConfig(backbone=args.backbone, attention=spec["attention"], tasks=spec["tasks"],
                          epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, seed=args.seed,
                          augment="mammo", image_size=args.height, pretrained=not args.no_pretrained)
        cdir = out / name; cdir.mkdir(parents=True, exist_ok=True)
        json.dump({**cfg.to_dict(), "height": args.height, "width": args.width, "label": spec["label"]},
                  open(cdir / "config.json", "w"), indent=1)
        log(f"\n=== {name}: {spec['label']} on {device} ===")
        rows, preds = [], []
        for k in folds:
            t0 = time.time()
            tr = np.flatnonzero(df["fold"].to_numpy() != k)
            te = np.flatnonzero(df["fold"].to_numpy() == k)
            log(f"  fold {k}/{len(folds)}: train {len(tr)} images / test {len(te)} images "
                f"({df['patient'].iloc[te].nunique()} patients)")
            model, p = _train_eval(cfg, X, df, tr, te, device, log, spec["tasks"])
            row = {"config": name, "fold": int(k), **_metric_row(df, tr, te, p)}
            assert row["patients_in_both"] == 0, "patient leakage between train and test"
            rows.append(row)
            preds.append(_pred_frame(df, te, p, name, int(k)))
            msg = [f"    -> fold {k}:"]
            if "path_auc" in row:
                msg.append(f"malignancy AUC {row['path_auc']:.3f} acc {row['path_accuracy']:.3f}")
            if "dens_accuracy" in row:
                msg.append(f"density acc {row['dens_accuracy']:.3f} QWK {row['dens_qwk']:.3f}")
            log("  ".join(msg) + f"  ({(time.time() - t0) / 60:.1f} min)")
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        pd.DataFrame(rows).to_csv(cdir / "fold_metrics.csv", index=False)
        pd.concat(preds).to_csv(cdir / "oof_predictions.csv", index=False)

        if args.official is not None and (not args.official or name in args.official):
            log("  official split: train on CBIS-DDSM train patients, test on unseen test patients")
            tr = np.flatnonzero(df["official_split"].to_numpy() == "train")
            te = np.flatnonzero(df["official_split"].to_numpy() == "test")
            model, p = _train_eval(cfg, X, df, tr, te, device, log, spec["tasks"])
            row = {"config": name, **_metric_row(df, tr, te, p)}
            json.dump(row, open(cdir / "official_metrics.json", "w"), indent=1)
            _pred_frame(df, te, p, name, 0).to_csv(cdir / "official_predictions.csv", index=False)
            torch.save({"state_dict": model.state_dict(), "config": cfg.to_dict(),
                        "height": args.height, "width": args.width}, ckpt_dir / f"cbis_{name}_official.pt")
            log("    -> official test: " + ", ".join(f"{k} {v:.3f}" for k, v in row.items()
                                                    if k in ("path_auc", "path_accuracy", "dens_accuracy", "dens_qwk")))
            del model
    log("done.")
    return 0


# ----------------------------------------------------------------------------------------------- summary
def _dens_cols(d):
    return [c for c in d.columns if c.startswith("p_density_")]


def cmd_summarise(args) -> int:
    out = Path(args.out)
    log = Logger(out / "log_summary.txt")
    nboot = 200 if args.smoke else 2000
    oof, folds, official = {}, [], {}
    for name in CONFIGS:
        if (out / name / "oof_predictions.csv").exists():
            oof[name] = pd.read_csv(out / name / "oof_predictions.csv", dtype={"patient": str})
            folds.append(pd.read_csv(out / name / "fold_metrics.csv"))
        if (out / name / "official_metrics.json").exists():
            official[name] = json.load(open(out / name / "official_metrics.json"))
    if not oof:
        log("nothing to summarise yet"); return 1
    folds = pd.concat(folds, ignore_index=True)
    folds.to_csv(out / "fold_metrics_all.csv", index=False)

    table = {}
    for name, d in oof.items():
        r = {"label": CONFIGS[name]["label"], "n_images": int(len(d)), "n_patients": int(d["patient"].nunique())}
        f = folds[folds["config"] == name]
        if "p_malignant" in d:
            y, p, g = d["y"].to_numpy(), d["p_malignant"].to_numpy(), d["patient"].to_numpy()
            m = binary_metrics(y, p)
            r["malignancy"] = {
                "auc_pooled": m["auc"], "auc_ci95": bootstrap_ci(y, p, n=nboot, groups=g),
                "auc_fold_mean": float(f["path_auc"].mean()), "auc_fold_std": float(f["path_auc"].std()),
                "accuracy": m["accuracy"], "sensitivity": m["sensitivity"], "specificity": m["specificity"],
                "balanced_accuracy": m["balanced_accuracy"], "brier": m["brier"], "ece": m["ece"],
                "majority_baseline_accuracy": float(f["path_majority_accuracy"].mean()),
            }
        if _dens_cols(d):
            k = d["density"].to_numpy() >= 0
            y, prob, g = d["density"].to_numpy()[k], d[_dens_cols(d)].to_numpy()[k], d["patient"].to_numpy()[k]
            m = multiclass_metrics(y, prob)
            r["density"] = {
                "accuracy": m["accuracy"], "accuracy_ci95": bootstrap_ci(y, prob, fn=density_acc_fn, n=nboot, groups=g),
                "qwk": m["qwk"], "qwk_ci95": bootstrap_ci(y, prob, fn=density_qwk_fn, n=nboot, groups=g),
                "macro_f1": m["macro_f1"],
                "accuracy_fold_mean": float(f["dens_accuracy"].mean()), "accuracy_fold_std": float(f["dens_accuracy"].std()),
                "majority_baseline_accuracy": float(f["dens_majority_accuracy"].mean()),
                "confusion": pd.crosstab(pd.Series(y, name="true"), pd.Series(prob.argmax(1), name="pred"))
                .reindex(index=range(4), columns=range(4), fill_value=0).values.tolist(),
            }
        table[name] = r

    comps = []
    for a, b, task, question in COMPARISONS:
        if a not in oof or b not in oof:
            continue
        m = oof[a].merge(oof[b], on=["image_id", "patient", "y", "density"], suffixes=("_a", "_b"))
        if task == "pathology":
            if "p_malignant_a" not in m or "p_malignant_b" not in m:
                continue
            res = paired_bootstrap(m["y"], m["p_malignant_a"], m["p_malignant_b"], m["patient"], fn=auc_fn, n=nboot)
            metric = "AUC"
        else:
            m = m[m["density"] >= 0]
            ca = [c for c in m.columns if c.startswith("p_density_") and c.endswith("_a")]
            cb = [c for c in m.columns if c.startswith("p_density_") and c.endswith("_b")]
            if not ca or not cb:
                continue
            res = paired_bootstrap(m["density"].to_numpy(), m[ca].to_numpy(), m[cb].to_numpy(), m["patient"],
                                   fn=density_qwk_fn, n=nboot)
            metric = "QWK"
        comps.append({"question": question, "a": a, "b": b, "metric": metric, **res})

    summary = {"results": table, "comparisons": comps, "official_split": official,
               "data": json.load(open(out / "data_stats.json"))}
    json.dump(summary, open(out / "summary.json", "w"), indent=1)
    (out / "summary.md").write_text(render_markdown(summary))
    from mammo.plots import plot_ablation
    plot_ablation(table, comps, out / "ablation_chart.png")
    log(render_markdown(summary))
    return 0


def _ci(v):
    return f"{v[0]:.3f}–{v[1]:.3f}"


def render_markdown(s: dict) -> str:
    t = s["results"]
    lines = ["| Model | Malignancy AUC (95% CI) | Malignancy acc. | Density acc. | Density QWK (95% CI) |",
             "|---|---:|---:|---:|---:|"]
    for name, r in t.items():
        mal, den = r.get("malignancy"), r.get("density")
        lines.append(f"| {r['label']} | "
                     + (f"{mal['auc_pooled']:.3f} ({_ci(mal['auc_ci95'])})" if mal else "–") + " | "
                     + (f"{mal['accuracy']:.3f}" if mal else "–") + " | "
                     + (f"{den['accuracy']:.3f}" if den else "–") + " | "
                     + (f"{den['qwk']:.3f} ({_ci(den['qwk_ci95'])})" if den else "–") + " |")
    mal_base = next((r["malignancy"]["majority_baseline_accuracy"] for r in t.values() if "malignancy" in r), None)
    den_base = next((r["density"]["majority_baseline_accuracy"] for r in t.values() if "density" in r), None)
    lines.append(f"| *Always predict the most common class* | *0.500* | "
                 + (f"*{mal_base:.3f}*" if mal_base is not None else "–") + " | "
                 + (f"*{den_base:.3f}*" if den_base is not None else "–") + " | *0.000* |")
    if s["comparisons"]:
        lines += ["", "| Question | Difference (A − B) | 95% CI | P(A not better) |", "|---|---:|---:|---:|"]
        for c in s["comparisons"]:
            lines.append(f"| {c['question']} ({c['metric']}) | {c['diff']:+.3f} | "
                         f"{c['ci95'][0]:+.3f} to {c['ci95'][1]:+.3f} | {c['p_not_better']:.2f} |")
    for name, r in s["official_split"].items():
        lines.append(f"\nOfficial CBIS-DDSM test split ({CONFIGS[name]['label']}, {r['n_test']} images, "
                     f"{r['test_patients']} unseen patients): "
                     + ", ".join(f"{lbl} {r[k]:.3f}" for k, lbl in [("path_auc", "malignancy AUC"),
                                                                    ("path_accuracy", "accuracy"),
                                                                    ("dens_accuracy", "density acc."),
                                                                    ("dens_qwk", "density QWK")] if k in r))
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["prepare", "run", "summarise"])
    ap.add_argument("configs", nargs="*", default=list(CONFIGS), help=f"variants for 'run': {list(CONFIGS)}")
    ap.add_argument("--data-root", default="/kaggle/input")
    ap.add_argument("--out", default="/kaggle/working/results/cbis")
    ap.add_argument("--cache", default="/tmp/mammo_cache")
    ap.add_argument("--checkpoints", default="/kaggle/working/checkpoints")
    ap.add_argument("--height", type=int, default=640)
    ap.add_argument("--width", type=int, default=384)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--max-folds", type=int, default=0, help="only run the first N folds (0 = all)")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--backbone", default="efficientnet_b0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--official", nargs="*", default=None, metavar="VARIANT",
                    help="also train on the official train split and test on the official test split; "
                         "give variant names to limit it to those (no names = every variant being run)")
    ap.add_argument("--no-pretrained", action="store_true", help="random init (offline tests only)")
    ap.add_argument("--smoke", action="store_true", help="40 patients, small images, 1 epoch, 2 folds")
    args = ap.parse_args(argv)
    bad = set(args.configs) - set(CONFIGS)
    if bad:
        ap.error(f"unknown variant(s) {bad}; choose from {list(CONFIGS)}")
    if args.smoke:
        args.out += "_smoke"
        args.height, args.width, args.epochs, args.folds, args.batch_size = 160, 96, 1, 2, 8
    return {"prepare": cmd_prepare, "run": cmd_run, "summarise": cmd_summarise}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
