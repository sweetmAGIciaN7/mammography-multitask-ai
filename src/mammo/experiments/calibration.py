"""Phase 4a - can the model's probabilities be taken literally?

Runs on the out-of-fold predictions saved by Phase 3 (``results/cbis/<variant>/oof_predictions.csv``),
so no GPU and no retraining are needed:

    PYTHONPATH=src python -m mammo.experiments.calibration            # ~1 min on a laptop

For every variant and task it reports, before and after calibration:

* **ECE** (expected calibration error): average gap between "predicted 70%" and "really 70%".
* **Brier score**: mean squared error of the probabilities (lower is better).
* **NLL** (log loss): what training minimises; punishes confident mistakes hardest.

Calibration methods (all *cross-fitted*: fold k is calibrated with parameters fitted on the other four folds):

* **Temperature scaling**: logit / T. One number. Fixes over- or under-confidence, cannot fix bias.
* **Platt scaling** (malignancy only): a * logit + b. Also fixes a model that says "malignant" too often.

It also picks an **operating point**: the threshold that reaches 90% sensitivity on the other folds, applied to the
held-out fold, to show what a clinically-motivated threshold costs in specificity and whether the promised
sensitivity holds on unseen patients.

The main model's official-split predictions are calibrated with parameters fitted on *all* CV out-of-fold
predictions (a different training run) to check whether a calibration transfers to a newly trained model.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, softmax

from mammo.calibration import (calibration_summary_binary, calibration_summary_multiclass, crossfit_operating_point,
                               crossfit_platt, crossfit_temperature, fit_platt, fit_temperature, reliability_curve,
                               top_label_ece)
from mammo.metrics import bootstrap_ci, expected_calibration_error

VARIANTS = {"mt_cbam": "Multi-task + CBAM (paper design)", "mt_plain": "Multi-task, no attention",
            "st_path": "Malignancy only + CBAM", "st_dens": "Density only + CBAM"}
PAPER = {"brier": 0.010, "ece_range": [0.365, 0.419]}  # Esen et al., calibration paragraph


def _ece_fn(y, p):
    return expected_calibration_error(y, p)


def _top_ece_fn(y, prob):
    return top_label_ece(y, prob)


def malignancy_block(d: pd.DataFrame, nboot: int, targets=(0.90, 0.95)) -> tuple[dict, dict]:
    y, z, f, g = d["y"].to_numpy(), d["logit_malignant"].to_numpy(), d["fold"].to_numpy(), d["patient"].to_numpy()
    z_t, temps = crossfit_temperature(y, z, f)
    z_p, platt = crossfit_platt(y, z, f)
    res, curves = {}, {}
    for name, zz in [("uncalibrated", z), ("temperature", z_t), ("platt", z_p)]:
        s = calibration_summary_binary(y, zz)
        s["ece_ci95"] = bootstrap_ci(y, expit(zz), fn=_ece_fn, n=nboot, groups=g)
        res[name] = s
        curves[name] = reliability_curve(y, expit(zz), n_bins=10)
    res["temperature_per_fold"] = temps
    res["temperature_all_folds"] = fit_temperature(y, z)
    res["platt_per_fold"] = {k: {"a": a, "b": b} for k, (a, b) in platt.items()}
    res["brier_ge_ece_squared"] = bool(res["uncalibrated"]["brier"] >= res["uncalibrated"]["ece"] ** 2)
    ops = {}
    for t in targets:
        op = crossfit_operating_point(y, expit(z_p), f, t)
        op.pop("pred")
        ops[f"sens_{int(t * 100)}"] = op
    ops["default_0.5_uncalibrated"] = _confusion(y, (expit(z) >= 0.5).astype(int))
    ops["default_0.5_platt"] = _confusion(y, (expit(z_p) >= 0.5).astype(int))
    res["operating_points"] = ops
    return res, curves


def _confusion(y, pred) -> dict:
    tp = int(((pred == 1) & (y == 1)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    return {"sensitivity": tp / max(tp + fn, 1), "specificity": tn / max(tn + fp, 1), "tp": tp, "fn": fn, "tn": tn, "fp": fp}


def density_block(d: pd.DataFrame, nboot: int) -> tuple[dict, dict]:
    cols = sorted(c for c in d.columns if c.startswith("logit_density_"))
    d = d[d["density"] >= 0]
    y, z, f, g = d["density"].to_numpy(), d[cols].to_numpy(), d["fold"].to_numpy(), d["patient"].to_numpy()
    z_t, temps = crossfit_temperature(y, z, f)
    res, curves = {}, {}
    for name, zz in [("uncalibrated", z), ("temperature", z_t)]:
        p = softmax(zz, axis=1)
        s = calibration_summary_multiclass(y, zz)
        s["ece_ci95"] = bootstrap_ci(y, p, fn=_top_ece_fn, n=nboot, groups=g)
        res[name] = s
        curves[name] = reliability_curve((p.argmax(1) == y).astype(float), p.max(1), n_bins=10)
    res["temperature_per_fold"] = temps
    res["temperature_all_folds"] = fit_temperature(y, z)
    return res, curves


def official_transfer(cbis_dir: Path) -> dict | None:
    """Calibrate the official-split model with parameters fitted on the CV predictions of the same variant."""
    fo, fc = cbis_dir / "mt_cbam" / "official_predictions.csv", cbis_dir / "mt_cbam" / "oof_predictions.csv"
    if not fo.exists():
        return None
    o, c = pd.read_csv(fo), pd.read_csv(fc)
    T = fit_temperature(c["y"], c["logit_malignant"])
    a, b = fit_platt(c["y"], c["logit_malignant"])
    z = o["logit_malignant"].to_numpy()
    out = {"n": int(len(o)), "T": T, "platt": {"a": a, "b": b}, "malignancy": {
        "uncalibrated": calibration_summary_binary(o["y"], z),
        "temperature": calibration_summary_binary(o["y"], z, T),
        "platt": calibration_summary_binary(o["y"], a * z + b)}}
    cols = sorted(x for x in o.columns if x.startswith("logit_density_"))
    if cols:
        k = o["density"] >= 0
        Td = fit_temperature(c.loc[c["density"] >= 0, "density"], c.loc[c["density"] >= 0, cols].to_numpy())
        out["density_T"] = Td
        out["density"] = {"uncalibrated": calibration_summary_multiclass(o.loc[k, "density"], o.loc[k, cols].to_numpy()),
                          "temperature": calibration_summary_multiclass(o.loc[k, "density"], o.loc[k, cols].to_numpy(), Td)}
    return out


def render_markdown(s: dict) -> str:
    L = ["### Malignancy (patient-level 5-fold CV, pooled out-of-fold)", "",
         "| Model | Calibration | ECE (95% CI) | Brier | NLL | Mean predicted P(malignant) |",
         "|---|---|---:|---:|---:|---:|"]
    for v, r in s["variants"].items():
        if "malignancy" not in r:
            continue
        m = r["malignancy"]
        for name, lbl in [("uncalibrated", "none"), ("temperature", f"temperature (T≈{m['temperature_all_folds']:.2f})"),
                          ("platt", "Platt")]:
            x = m[name]
            L.append(f"| {VARIANTS[v] if name == 'uncalibrated' else ''} | {lbl} | {x['ece']:.3f} "
                     f"({x['ece_ci95'][0]:.3f}–{x['ece_ci95'][1]:.3f}) | {x['brier']:.3f} | {x['nll']:.3f} | "
                     f"{x['mean_predicted']:.3f} |")
    obs = next(r["malignancy"]["uncalibrated"]["observed_rate"] for r in s["variants"].values() if "malignancy" in r)
    L += [f"\nObserved share of malignant images: {obs:.3f}.", "",
          "### Density (top-label calibration: when the model is X% sure of a grade, is it right X% of the time?)", "",
          "| Model | Calibration | ECE (95% CI) | Brier (4-class) | NLL | Mean confidence | Accuracy |",
          "|---|---|---:|---:|---:|---:|---:|"]
    for v, r in s["variants"].items():
        if "density" not in r:
            continue
        m = r["density"]
        for name, lbl in [("uncalibrated", "none"), ("temperature", f"temperature (T≈{m['temperature_all_folds']:.2f})")]:
            x = m[name]
            L.append(f"| {VARIANTS[v] if name == 'uncalibrated' else ''} | {lbl} | {x['ece']:.3f} "
                     f"({x['ece_ci95'][0]:.3f}–{x['ece_ci95'][1]:.3f}) | {x['brier']:.3f} | {x['nll']:.3f} | "
                     f"{x['mean_confidence']:.3f} | {x['accuracy']:.3f} |")
    m = s["variants"].get("mt_cbam", {}).get("malignancy")
    if m:
        L += ["", "### Operating points (paper design, Platt-calibrated, threshold chosen on the other folds)", "",
              "| Threshold rule | Sensitivity | Specificity | PPV | NPV | Missed cancers | False alarms |",
              "|---|---:|---:|---:|---:|---:|---:|"]
        for key, lbl in [("default_0.5_uncalibrated", "0.5 on raw probabilities"),
                         ("default_0.5_platt", "0.5 on calibrated probabilities"),
                         ("sens_90", "aim for 90% sensitivity"), ("sens_95", "aim for 95% sensitivity")]:
            o = m["operating_points"][key]
            ppv = o.get("ppv", o["tp"] / max(o["tp"] + o["fp"], 1))
            npv = o.get("npv", o["tn"] / max(o["tn"] + o["fn"], 1))
            L.append(f"| {lbl} | {o['sensitivity']:.3f} | {o['specificity']:.3f} | {ppv:.3f} | {npv:.3f} | "
                     f"{o['fn']} | {o['fp']} |")
    t = s.get("official_transfer")
    if t:
        L += ["", f"### Does a calibration transfer to a newly trained model? (official test split, {t['n']} images)", "",
              "| Calibration (fitted on the CV predictions) | ECE | Brier | Mean predicted P(malignant) | Observed |",
              "|---|---:|---:|---:|---:|"]
        for name in ("uncalibrated", "temperature", "platt"):
            x = t["malignancy"][name]
            L.append(f"| {name} | {x['ece']:.3f} | {x['brier']:.3f} | {x['mean_predicted']:.3f} | {x['observed_rate']:.3f} |")
    L += ["", f"Sanity check against the paper: Brier ≥ ECE² holds for every model here "
          f"({'yes' if s['brier_ge_ece_squared_all'] else 'NO'}). The paper reports Brier {PAPER['brier']} with ECE "
          f"{PAPER['ece_range'][0]}–{PAPER['ece_range'][1]}, which would need Brier ≥ {PAPER['ece_range'][0] ** 2:.3f}."]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cbis", default="results/cbis", help="Phase 3 results folder")
    ap.add_argument("--out", default="results/calibration")
    ap.add_argument("--nboot", type=int, default=1000)
    args = ap.parse_args(argv)
    cbis, out = Path(args.cbis), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    variants, curves = {}, {}
    for v in VARIANTS:
        f = cbis / v / "oof_predictions.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f, dtype={"patient": str})
        r, c = {}, {}
        if "logit_malignant" in d:
            r["malignancy"], c["malignancy"] = malignancy_block(d, args.nboot)
        if any(x.startswith("logit_density_") for x in d.columns):
            r["density"], c["density"] = density_block(d, args.nboot)
        variants[v], curves[v] = r, c
    if not variants:
        print(f"no Phase 3 predictions found in {cbis}")
        return 1
    ok = all(r[t]["uncalibrated"]["brier"] >= r[t]["uncalibrated"]["ece"] ** 2
             for r in variants.values() for t in ("malignancy",) if t in r)
    summary = {"variants": variants, "official_transfer": official_transfer(cbis), "paper": PAPER,
               "brier_ge_ece_squared_all": bool(ok)}
    json.dump(summary, open(out / "summary.json", "w"), indent=1, default=float)
    json.dump(curves, open(out / "reliability_curves.json", "w"), indent=1)
    (out / "summary.md").write_text(render_markdown(summary))
    from mammo.plots import plot_reliability
    main_v = "mt_cbam" if "mt_cbam" in curves else next(iter(curves))
    plot_reliability(curves[main_v], summary["variants"][main_v], out / "reliability_diagram.png")
    print(render_markdown(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
