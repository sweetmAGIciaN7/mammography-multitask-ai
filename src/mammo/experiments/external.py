"""Phase 4b - external validation: train on CBIS-DDSM (scanned film), test on INbreast (digital).

Nothing from INbreast is used for training, tuning or calibration. The only question is how much of what the
model learned on CBIS-DDSM survives a change of country, decade and imaging technology.

Steps (Kaggle, see notebooks/04_external_inbreast.ipynb):

    python -m mammo.experiments.cbis prepare                            # CBIS-DDSM image cache (same as Phase 3)
    python -m mammo.experiments.external train mt_cbam st_dens --gpu 0  # official-split models -> checkpoints
    python -m mammo.experiments.external prepare                        # INbreast DICOMs -> image cache
    python -m mammo.experiments.external predict                        # every checkpoint on INbreast
    python -m mammo.experiments.external summarise                      # tables + chart (no GPU needed)

``train`` retrains each variant on the official CBIS-DDSM training patients (Phase 3 only kept the main
model's predictions, not its weights) and re-predicts the official test split, which also checks that
Phase 3's official-split numbers reproduce.

Reported on INbreast:

* **Density** (main result): accuracy, quadratic-weighted kappa, confusion matrix, calibration (with the
  temperature fitted on CBIS-DDSM), compared with the same model's CBIS-DDSM official test result.
* **Malignancy proxy** (exploratory): AUC for BI-RADS 4-6 vs 1-3. INbreast has no biopsy result for most
  images and includes normal mammograms, which CBIS-DDSM never shows the model.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax

from mammo.calibration import calibration_summary_binary, calibration_summary_multiclass, fit_platt, fit_temperature
from mammo.metrics import (auc_fn, bootstrap_ci, density_acc_fn, density_qwk_fn, multiclass_metrics,
                           paired_bootstrap)

VARIANTS = {"mt_cbam": "Multi-task + CBAM (paper design)", "mt_plain": "Multi-task, no attention",
            "st_path": "Malignancy only + CBAM", "st_dens": "Density only + CBAM"}
COMPARISONS = [("mt_cbam", "st_dens", "density", "Does multi-task training help density transfer?"),
               ("mt_cbam", "mt_plain", "density", "Does CBAM help density transfer?"),
               ("mt_cbam", "st_path", "pathology", "Does multi-task training help the malignancy proxy?"),
               ("mt_cbam", "mt_plain", "pathology", "Does CBAM help the malignancy proxy?")]


class Logger:
    def __init__(self, path: Path | None = None):
        self.f = open(path, "a") if path else None

    def __call__(self, msg: str = ""):
        print(msg, flush=True)
        if self.f:
            self.f.write(msg + "\n"); self.f.flush()


def _pred_columns(p: dict) -> dict:
    cols = {}
    if "pathology" in p:
        cols["p_malignant"], cols["logit_malignant"] = p["pathology"], p["pathology_logit"]
    if "density" in p:
        for k in range(p["density"].shape[1]):
            cols[f"p_density_{k}"] = p["density"][:, k]
            cols[f"logit_density_{k}"] = p["density_logits"][:, k]
    return cols


# ----------------------------------------------------------------------------------------------- train
def cmd_train(args) -> int:
    import torch
    from mammo.experiments import cbis as C
    from mammo.preprocess import cached_images
    from mammo.train import TrainConfig

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    log = Logger(out / f"log_train_gpu{args.gpu}.txt")
    cargs = argparse.Namespace(out=args.cbis_out, data_root=args.data_root, smoke=args.smoke, folds=5, seed=args.seed)
    df, _ = C.load_index(cargs, log)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not args.smoke:
        log("ERROR: no GPU. In Kaggle: Settings -> Accelerator -> GPU T4 x2, then run again."); return 1
    if device.type == "cuda":
        torch.cuda.set_device(device)
    X = torch.from_numpy(cached_images(df["path"], args.height, args.width, args.cache, workers=args.workers, log=log))
    tr = np.flatnonzero(df["official_split"].to_numpy() == "train")
    te = np.flatnonzero(df["official_split"].to_numpy() == "test")
    ckpt = Path(args.checkpoints); ckpt.mkdir(parents=True, exist_ok=True)
    for name in args.variants:
        spec = C.CONFIGS[name]
        cfg = TrainConfig(backbone=args.backbone, attention=spec["attention"], tasks=spec["tasks"], epochs=args.epochs,
                          batch_size=args.batch_size, lr=args.lr, seed=args.seed, augment="mammo",
                          image_size=args.height, pretrained=not args.no_pretrained)
        t0 = time.time()
        log(f"\n=== {name}: train on {len(tr)} official-train images, test on {len(te)} official-test images ===")
        model, p = C._train_eval(cfg, X, df, tr, te, device, log, spec["tasks"])
        row = C._metric_row(df, tr, te, p)
        assert row["patients_in_both"] == 0
        vdir = out / name; vdir.mkdir(parents=True, exist_ok=True)
        C._pred_frame(df, te, p, name, 0).to_csv(vdir / "cbis_official_predictions.csv", index=False)
        json.dump(row, open(vdir / "cbis_official_metrics.json", "w"), indent=1)
        torch.save({"state_dict": model.state_dict(), "config": cfg.to_dict(), "height": args.height,
                    "width": args.width}, ckpt / f"cbis_{name}_official.pt")
        log(f"    -> CBIS-DDSM official test: " + ", ".join(f"{k} {row[k]:.3f}" for k in
            ("path_auc", "path_accuracy", "dens_accuracy", "dens_qwk") if k in row)
            + f"  ({(time.time() - t0) / 60:.1f} min)")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return 0


# ----------------------------------------------------------------------------------------------- INbreast
def _inbreast_index(args, log) -> pd.DataFrame:
    from mammo.inbreast import index_inbreast_full, locate_inbreast
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    f = out / "inbreast_index.csv"
    if f.exists():
        return pd.read_csv(f, dtype={"image_id": str, "patient": str})
    release = locate_inbreast(args.data_root)
    log(f"INbreast: {release}")
    df = index_inbreast_full(release)
    if args.smoke:
        df = df.head(24).reset_index(drop=True)
    df.to_csv(f, index=False)
    return df


def cmd_prepare(args) -> int:
    from mammo.inbreast import cached_inbreast, describe_inbreast
    from mammo.preprocess import preview_grid
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    log = Logger(out / "log_prepare.txt")
    df = _inbreast_index(args, log)
    log("=== INbreast ===\n" + describe_inbreast(df))
    X = cached_inbreast(df["path"], args.height, args.width, args.cache, workers=args.workers, log=log)
    pick = df.groupby(["density", "view"]).head(2).index[:16]
    titles = [f"{r.image_id} {r.side}-{r.view}\nACR {'?' if r.density < 0 else r.density + 1} | BI-RADS {r.birads}"
              for r in df.loc[pick].itertuples()]
    preview_grid(X[pick], titles, out / "inbreast_preview.png")
    log(f"images ready: {X.shape}; preview written to {out / 'inbreast_preview.png'}")
    return 0


def cmd_predict(args) -> int:
    import torch
    from mammo.inbreast import cached_inbreast
    from mammo.train import TrainConfig, build_model, predict
    out = Path(args.out)
    log = Logger(out / "log_predict.txt")
    df = _inbreast_index(args, log)
    X = torch.from_numpy(cached_inbreast(df["path"], args.height, args.width, args.cache, workers=args.workers, log=log))
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    found = 0
    for name in args.variants:
        f = Path(args.checkpoints) / f"cbis_{name}_official.pt"
        if not f.exists():
            log(f"  {name}: no checkpoint at {f}, skipped"); continue
        ck = torch.load(f, map_location="cpu", weights_only=False)
        cfg = TrainConfig(**{k: (tuple(v) if k == "tasks" else v) for k, v in ck["config"].items()})
        cfg.pretrained = False
        model = build_model(cfg).to(device)
        model.load_state_dict(ck["state_dict"])
        p = predict(model, X, np.arange(len(df)), device, batch_size=32)
        pred = df[["image_id", "patient", "side", "view", "density", "birads", "birads_num", "suspicious"]].copy()
        pred.insert(0, "config", name)
        for k, v in _pred_columns(p).items():
            pred[k] = v
        (out / name).mkdir(parents=True, exist_ok=True)
        pred.to_csv(out / name / "inbreast_predictions.csv", index=False)
        found += 1
        msg = f"  {name}: predicted {len(pred)} INbreast images"
        if "p_density_0" in pred:
            m = multiclass_metrics(pred["density"].to_numpy(), pred[[f"p_density_{k}" for k in range(4)]].to_numpy())
            msg += f"; density acc {m['accuracy']:.3f} QWK {m['qwk']:.3f}"
        log(msg)
    return 0 if found else 1


# ----------------------------------------------------------------------------------------------- summary
def _density_arrays(d: pd.DataFrame):
    k = d["density"].to_numpy() >= 0
    cols_p = [f"p_density_{i}" for i in range(4)]
    cols_l = [f"logit_density_{i}" for i in range(4)]
    return (d["density"].to_numpy()[k], d[cols_p].to_numpy()[k], d[cols_l].to_numpy()[k],
            d["patient"].astype(str).to_numpy()[k])


def density_report(d: pd.DataFrame, nboot: int, T: float | None = None, train_majority: int | None = None) -> dict:
    y, prob, logits, g = _density_arrays(d)
    m = multiclass_metrics(y, prob)
    r = {"n_images": int(len(y)), "n_patients": int(len(np.unique(g))),
         "accuracy": m["accuracy"], "accuracy_ci95": bootstrap_ci(y, prob, fn=density_acc_fn, n=nboot, groups=g),
         "qwk": m["qwk"], "qwk_ci95": bootstrap_ci(y, prob, fn=density_qwk_fn, n=nboot, groups=g),
         "macro_f1": m["macro_f1"],
         "within_one_grade": float((np.abs(prob.argmax(1) - y) <= 1).mean()),
         "mean_predicted_grade": float(prob.argmax(1).mean()), "mean_true_grade": float(y.mean()),
         "true_counts": np.bincount(y, minlength=4).tolist(),
         "predicted_counts": np.bincount(prob.argmax(1), minlength=4).tolist(),
         "confusion": pd.crosstab(pd.Series(y, name="true"), pd.Series(prob.argmax(1), name="pred"))
         .reindex(index=range(4), columns=range(4), fill_value=0).values.tolist(),
         "majority_baseline_accuracy": float((y == np.bincount(y).argmax()).mean())}
    if train_majority is not None:
        r["train_majority_baseline_accuracy"] = float((y == train_majority).mean())
    r["calibration"] = {"uncalibrated": calibration_summary_multiclass(y, logits)}
    if T is not None:
        r["calibration"]["cbis_temperature"] = {"T": T, **calibration_summary_multiclass(y, logits, T)}
        r["calibration"]["oracle_temperature_fitted_on_inbreast"] = {
            "T": fit_temperature(y, logits), **calibration_summary_multiclass(y, logits, fit_temperature(y, logits))}
    return r


def proxy_report(d: pd.DataFrame, nboot: int, platt: tuple | None = None) -> dict:
    k = d["suspicious"].to_numpy() >= 0
    y, p, z, g = (d["suspicious"].to_numpy()[k], d["p_malignant"].to_numpy()[k], d["logit_malignant"].to_numpy()[k],
                  d["patient"].astype(str).to_numpy()[k])
    if len(np.unique(y)) < 2:
        return {"definition": "BI-RADS 4-6 (suspicious) vs 1-3", "n_images": int(len(y)), "auc": float("nan"),
                "auc_ci95": [float("nan"), float("nan")], "note": "only one class present"}
    r = {"definition": "BI-RADS 4-6 (suspicious) vs 1-3", "n_images": int(len(y)), "n_positive": int(y.sum()),
         "auc": auc_fn(y, p), "auc_ci95": bootstrap_ci(y, p, n=nboot, groups=g),
         "mean_p_malignant_birads_1_3": float(p[y == 0].mean()), "mean_p_malignant_birads_4_6": float(p[y == 1].mean())}
    bn = d["birads_num"].to_numpy()[k]
    strict = np.isin(bn, [1, 2, 5, 6])
    if strict.sum() > 10 and len(np.unique(y[strict])) == 2:
        r["auc_birads_5_6_vs_1_2"] = auc_fn(y[strict], p[strict])
        r["auc_birads_5_6_vs_1_2_ci95"] = bootstrap_ci(y[strict], p[strict], n=nboot, groups=g[strict])
        r["n_birads_5_6_vs_1_2"] = int(strict.sum())
    r["by_birads"] = {str(int(b)): {"n": int((bn == b).sum()), "mean_p_malignant": float(p[bn == b].mean())}
                      for b in sorted(np.unique(bn[~np.isnan(bn)]))}
    if platt is not None:
        r["calibration_note"] = "not a calibration test: BI-RADS is not pathology"
    return r


def cmd_summarise(args) -> int:
    out = Path(args.out)
    log = Logger(out / "log_summary.txt")
    nboot = 200 if args.smoke else args.nboot
    cbis_results = Path(args.cbis_results)
    res, preds = {}, {}
    for name in VARIANTS:
        f = out / name / "inbreast_predictions.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f, dtype={"patient": str, "image_id": str})
        preds[name] = d
        r = {"label": VARIANTS[name]}
        oof = cbis_results / name / "oof_predictions.csv"
        oof = pd.read_csv(oof) if oof.exists() else None
        off = out / name / "cbis_official_predictions.csv"
        off = pd.read_csv(off, dtype={"patient": str}) if off.exists() else None
        if "p_density_0" in d:
            T, maj = None, None
            if oof is not None and "logit_density_0" in oof:
                k = oof["density"] >= 0
                T = fit_temperature(oof.loc[k, "density"].to_numpy(),
                                    oof.loc[k, [f"logit_density_{i}" for i in range(4)]].to_numpy())
                maj = int(np.bincount(oof.loc[k, "density"]).argmax())
            r["inbreast_density"] = density_report(d, nboot, T, maj)
            if off is not None and "p_density_0" in off:
                r["cbis_official_density"] = density_report(off, nboot)
        if "p_malignant" in d:
            platt = fit_platt(oof["y"], oof["logit_malignant"]) if oof is not None and "logit_malignant" in oof else None
            r["inbreast_malignancy_proxy"] = proxy_report(d, nboot, platt)
            if off is not None and "p_malignant" in off:
                r["cbis_official_malignancy"] = {"auc": auc_fn(off["y"], off["p_malignant"]),
                                                 "auc_ci95": bootstrap_ci(off["y"].to_numpy(), off["p_malignant"].to_numpy(),
                                                                          n=nboot, groups=off["patient"].to_numpy())}
        res[name] = r
    if not res:
        log("no INbreast predictions yet"); return 1

    comps = []
    for a, b, task, q in COMPARISONS:
        if a not in preds or b not in preds:
            continue
        m = preds[a].merge(preds[b], on=["image_id", "patient", "density", "suspicious"], suffixes=("_a", "_b"))
        if task == "density":
            if "p_density_0_a" not in m or "p_density_0_b" not in m:
                continue
            m = m[m["density"] >= 0]
            res_ = paired_bootstrap(m["density"].to_numpy(), m[[f"p_density_{i}_a" for i in range(4)]].to_numpy(),
                                    m[[f"p_density_{i}_b" for i in range(4)]].to_numpy(), m["patient"].to_numpy(),
                                    fn=density_qwk_fn, n=nboot)
            metric = "QWK"
        else:
            if "p_malignant_a" not in m or "p_malignant_b" not in m:
                continue
            m = m[m["suspicious"] >= 0]
            res_ = paired_bootstrap(m["suspicious"].to_numpy(), m["p_malignant_a"].to_numpy(), m["p_malignant_b"].to_numpy(),
                                    m["patient"].to_numpy(), fn=auc_fn, n=nboot)
            metric = "proxy AUC"
        comps.append({"question": q, "a": a, "b": b, "metric": metric, **res_})

    phase3 = cbis_results / "mt_cbam" / "official_metrics.json"
    reproduced = None
    if phase3.exists() and (out / "mt_cbam" / "cbis_official_metrics.json").exists():
        p3, now = json.load(open(phase3)), json.load(open(out / "mt_cbam" / "cbis_official_metrics.json"))
        reproduced = {k: {"phase3": p3.get(k), "phase4": now.get(k)} for k in
                      ("path_auc", "path_accuracy", "dens_accuracy", "dens_qwk")}
    idx = out / "inbreast_index.csv"
    data = None
    if idx.exists():
        i = pd.read_csv(idx)
        data = {"images": int(len(i)), "patients": int(i["patient"].nunique()),
                "density_known": int((i["density"] >= 0).sum()),
                "density_counts": np.bincount(i.loc[i["density"] >= 0, "density"], minlength=4).tolist(),
                "suspicious": int((i["suspicious"] == 1).sum()), "not_suspicious": int((i["suspicious"] == 0).sum())}
    summary = {"results": res, "comparisons": comps, "official_reproduced": reproduced, "inbreast": data}
    json.dump(summary, open(out / "summary.json", "w"), indent=1, default=float)
    (out / "summary.md").write_text(render_markdown(summary))
    from mammo.plots import plot_external
    plot_external(res, out / "external_chart.png")
    log(render_markdown(summary))
    return 0


def _ci(c):
    return f"{c[0]:.3f}–{c[1]:.3f}"


def render_markdown(s: dict) -> str:
    r = s["results"]
    L = ["### Density: CBIS-DDSM official test (internal) vs INbreast (external)", "",
         "| Model | CBIS-DDSM acc. | CBIS-DDSM QWK | INbreast acc. (95% CI) | INbreast QWK (95% CI) | within ±1 grade |",
         "|---|---:|---:|---:|---:|---:|"]
    for name, x in r.items():
        if "inbreast_density" not in x:
            continue
        e, c = x["inbreast_density"], x.get("cbis_official_density")
        L.append(f"| {x['label']} | " + (f"{c['accuracy']:.3f} | {c['qwk']:.3f}" if c else "– | –")
                 + f" | {e['accuracy']:.3f} ({_ci(e['accuracy_ci95'])}) | {e['qwk']:.3f} ({_ci(e['qwk_ci95'])}) | "
                 f"{e['within_one_grade']:.3f} |")
    first = next((x["inbreast_density"] for x in r.values() if "inbreast_density" in x), None)
    if first:
        L.append(f"| *Always predict INbreast's most common grade* | | | *{first['majority_baseline_accuracy']:.3f}* | "
                 f"*0.000* | |")
        if "train_majority_baseline_accuracy" in first:
            L.append(f"| *Always predict CBIS-DDSM's most common grade* | | | "
                     f"*{first['train_majority_baseline_accuracy']:.3f}* | *0.000* | |")
        L += ["", f"INbreast density: true grade counts A/B/C/D = {first['true_counts']}; the paper-design model "
              f"predicted {r.get('mt_cbam', {}).get('inbreast_density', first)['predicted_counts']}."]
        cal = r.get("mt_cbam", {}).get("inbreast_density", first)["calibration"]
        L += ["", "Density calibration on INbreast (paper design, top-label ECE): "
              + ", ".join(f"{k.replace('_', ' ')} {v['ece']:.3f}" + (f" (T={v['T']:.2f})" if "T" in v else "")
                          for k, v in cal.items())]
        cm = r.get("mt_cbam", {}).get("inbreast_density", first)["confusion"]
        L += ["", "Confusion matrix (paper design, rows = true ACR grade, columns = predicted):", "",
              "| true \\ pred | A | B | C | D |", "|---|---:|---:|---:|---:|"]
        L += [f"| {'ABCD'[i]} | " + " | ".join(str(v) for v in row) + " |" for i, row in enumerate(cm)]
    L += ["", "### Malignancy proxy on INbreast (exploratory: BI-RADS 4–6 vs 1–3, not pathology)", "",
          "| Model | CBIS-DDSM official AUC | INbreast proxy AUC (95% CI) | BI-RADS 5–6 vs 1–2 AUC |", "|---|---:|---:|---:|"]
    for name, x in r.items():
        if "inbreast_malignancy_proxy" not in x:
            continue
        e, c = x["inbreast_malignancy_proxy"], x.get("cbis_official_malignancy")
        L.append(f"| {x['label']} | " + (f"{c['auc']:.3f}" if c else "–") + f" | {e['auc']:.3f} ({_ci(e['auc_ci95'])}) | "
                 + (f"{e['auc_birads_5_6_vs_1_2']:.3f} ({_ci(e['auc_birads_5_6_vs_1_2_ci95'])})"
                    if "auc_birads_5_6_vs_1_2" in e else "–") + " |")
    if s["comparisons"]:
        L += ["", "| Question (INbreast) | Difference (A − B) | 95% CI | P(A not better) |", "|---|---:|---:|---:|"]
        for c in s["comparisons"]:
            L.append(f"| {c['question']} ({c['metric']}) | {c['diff']:+.3f} | {c['ci95'][0]:+.3f} to "
                     f"{c['ci95'][1]:+.3f} | {c['p_not_better']:.2f} |")
    if s.get("official_reproduced"):
        L += ["", "Reproducibility of the Phase 3 official-split result (paper design, retrained with the same seed): "
              + ", ".join(f"{k} {v['phase3']:.3f} → {v['phase4']:.3f}" for k, v in s["official_reproduced"].items()
                          if v["phase3"] is not None and v["phase4"] is not None)]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["train", "prepare", "predict", "summarise"])
    ap.add_argument("variants", nargs="*", default=list(VARIANTS))
    ap.add_argument("--data-root", default="/kaggle/input")
    ap.add_argument("--out", default="/kaggle/working/results/external")
    ap.add_argument("--cbis-out", default="/kaggle/working/results/cbis", help="CBIS-DDSM index from 'cbis prepare'")
    ap.add_argument("--cbis-results", default="results/cbis", help="Phase 3 results (OOF predictions for calibration)")
    ap.add_argument("--cache", default="/tmp/mammo_cache")
    ap.add_argument("--checkpoints", default="/kaggle/working/checkpoints")
    ap.add_argument("--height", type=int, default=640)
    ap.add_argument("--width", type=int, default=384)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--backbone", default="efficientnet_b0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--nboot", type=int, default=2000)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="tiny subset, small images, 1 epoch")
    args = ap.parse_args(argv)
    bad = set(args.variants) - set(VARIANTS)
    if bad:
        ap.error(f"unknown variant(s) {bad}; choose from {list(VARIANTS)}")
    if args.smoke:
        args.out += "_smoke"; args.cbis_out += "_smoke"; args.checkpoints += "_smoke"
        args.height, args.width, args.epochs, args.batch_size = 160, 96, 1, 8
    return {"train": cmd_train, "prepare": cmd_prepare, "predict": cmd_predict, "summarise": cmd_summarise}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
