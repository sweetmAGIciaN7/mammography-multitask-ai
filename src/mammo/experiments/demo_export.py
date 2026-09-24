"""Phase 6: build the files the online demo needs (Streamlit app: ``demo/streamlit_app.py``).

    python -m mammo.experiments.demo_export build   [--out /kaggle/working/demo]

Needs the Phase 4 checkpoint ``cbis_mt_cbam_official.pt`` (attach the Phase 4 notebook's Output) and the CBIS-DDSM
dataset (for the example images). It
1. exports a slim checkpoint ``model.pt`` with the Phase 4a "official transfer" Platt parameters and provenance;
2. draws 6 **random** official-test images (seed 0; 3 malignant, 3 benign; never selected by hand) as examples,
   saved losslessly exactly as the model decodes them, so the demo shows the same predictions as Phase 5;
3. checks that the demo code reproduces the Phase 5 predictions, renders a preview, and zips ``model.pt`` +
   ``examples/`` into ``demo_bundle.zip``, whose contents go into the repository's ``demo/`` folder.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DENS = {0: "A", 1: "B", 2: "C", 3: "D"}  # CBIS-DDSM density 1-4 is stored as 0-3 (-1 missing)


def _log(msg=""):
    print(msg, flush=True)


def _git_commit() -> str:
    r = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    return r.stdout.strip() or "unknown"


def pick_examples(test: pd.DataFrame, n_per_class: int = 3, seed: int = 0) -> pd.DataFrame:
    """Stratified random draw (n malignant + n benign), then shuffled. Deterministic for a given seed."""
    rng = np.random.default_rng(seed)
    parts = []
    for label in (1, 0):
        pool = test.index[test["pathology"] == label].to_numpy()
        parts.append(rng.choice(pool, size=min(n_per_class, len(pool)), replace=False))
    idx = np.concatenate(parts)
    rng.shuffle(idx)
    return test.loc[idx].reset_index(drop=True)


def save_example(src: str, dst: Path, height: int, width: int) -> None:
    """Lossless PNG of the grayscale array ``load_mammogram`` decodes from ``src`` (JPEG draft mode included), so the
    example gives the model exactly the same pixels as the original file."""
    from PIL import Image
    with Image.open(src) as im:
        if im.format == "JPEG":
            im.draft("L", (width * 2, height * 2))
        a = np.asarray(im.convert("L"))
    Image.fromarray(a).save(dst, optimize=True)


def cmd_build(args) -> int:
    from mammo.cbis import _jpeg_relative, locate_cbis
    from mammo.demo import export_demo_checkpoint, load_demo_model, predict_file, render_panels
    from mammo.experiments.attention import find_checkpoints

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report_dir = Path(args.report)
    report_dir.mkdir(parents=True, exist_ok=True)

    ck = find_checkpoints(args.ckpt).get("mt_cbam")
    if ck is None:
        _log("ERROR: cbis_mt_cbam_official.pt not found. Add the Phase 4 notebook (notebookb86c6df37c) as an input.")
        return 1
    _log(f"checkpoint: {ck}")
    cal = json.load(open(ROOT / "results" / "calibration" / "summary.json"))["official_transfer"]
    official = json.load(open(ROOT / "results" / "external" / "mt_cbam" / "cbis_official_metrics.json"))
    meta = {"model": "mt_cbam (EfficientNet-B0 + CBAM, malignancy + density heads)",
            "trained_on": "CBIS-DDSM official training split (patients also in the test files counted as training)",
            "source_checkpoint": ck.name, "code_commit": _git_commit(),
            "official_test_metrics": official}
    calibration = {"platt": cal["platt"], "note": "Platt parameters fitted on Phase 3 cross-validation predictions, "
                   "applied unchanged (official split ECE 0.206 -> 0.090)"}
    info = export_demo_checkpoint(ck, out / "model.pt", calibration, meta)
    _log(f"model.pt: {(out / 'model.pt').stat().st_size / 1e6:.1f} MB, input {info['height']}x{info['width']}")

    # ---- examples: random official-test images, checked against the Phase 5 predictions
    test = pd.read_csv(args.test_images)
    pred = pd.read_csv(args.predictions).set_index("image_id")
    jpeg = locate_cbis(args.cbis_root)["jpeg"]
    test["file"] = [str(jpeg / _jpeg_relative(p)) for p in test["path"]]
    test = test[[os.path.exists(f) for f in test["file"]]].reset_index(drop=True)
    _log(f"official test images found on disk: {len(test)}")
    ex = pick_examples(test, args.n_per_class, args.seed)

    model, ckd = load_demo_model(out / "model.pt")
    if (out / "examples").exists():
        shutil.rmtree(out / "examples")
    (out / "examples").mkdir()
    rows, panels = [], []
    for k, r in enumerate(ex.itertuples(), 1):
        res_orig = predict_file(model, ckd, r.file)
        name = f"example_{k}.png"
        save_example(r.file, out / "examples" / name, info["height"], info["width"])
        res = predict_file(model, ckd, out / "examples" / name)
        phase5 = float(pred.loc[r.image_id, "p_malignant_mt_cbam"])
        rows.append({"file": name, "image_id": r.image_id, "pathology": "malignant" if r.pathology else "benign",
                     "lesion_type": r.lesion_type, "density": DENS.get(int(r.density), "?"),
                     "phase5_p_malignant": phase5, "demo_p_raw_original_file": res_orig["p_malignant_raw"],
                     "demo_p_raw_example_file": res["p_malignant_raw"], "demo_p_calibrated": res["p_malignant"],
                     "demo_density": "ABCD"[int(np.argmax(res["p_density"]))]})
        panels.append((name, rows[-1], render_panels(res, dpi=80)))
        _log(f"  {name}: {r.image_id:22s} {rows[-1]['pathology']:9s} phase5 {phase5:.4f}  demo(original) "
             f"{res_orig['p_malignant_raw']:.4f}  demo(example png) {res['p_malignant_raw']:.4f}  "
             f"calibrated {res['p_malignant']:.3f}")
    table = pd.DataFrame(rows)
    table[["file", "image_id", "pathology", "lesion_type", "density"]].to_csv(out / "examples" / "examples.csv",
                                                                                index=False)
    table.to_csv(report_dir / "examples_check.csv", index=False)
    diff = float(np.abs(table["phase5_p_malignant"] - table["demo_p_raw_original_file"]).max())
    diff_ex = float(np.abs(table["demo_p_raw_original_file"] - table["demo_p_raw_example_file"]).max())
    files = sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file())
    check = {"max_abs_diff_vs_phase5": diff, "max_abs_diff_example_vs_original": diff_ex, "n_examples": len(table),
             "files": files, "model_mb": (out / "model.pt").stat().st_size / 1e6,
             "examples_mb": sum((out / f).stat().st_size for f in files if f.startswith("examples")) / 1e6}
    json.dump(check, open(report_dir / "demo_check.json", "w"), indent=1)
    _preview(panels, report_dir / "demo_preview.png")
    _log(f"max |demo - Phase 5| on the original files: {diff:.2e}; example PNG vs original file: {diff_ex:.2e}")
    if diff > 0.01 or diff_ex > 1e-4:
        _log("ERROR: the demo does not reproduce the Phase 5 predictions. Send Claude the output of this cell.")
        return 1
    zf = Path(args.zip)
    if zf.exists():
        zf.unlink()
    shutil.make_archive(str(zf.with_suffix("")), "zip", root_dir=out)
    _log(f"demo files ready: {out} ({len(files)} files, model {check['model_mb']:.1f} MB, examples "
         f"{check['examples_mb']:.1f} MB) -> {zf} ({zf.stat().st_size / 1e6:.1f} MB)")
    return 0


def _preview(panels, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(panels), 1, figsize=(9, 4.6 * len(panels)))
    for ax, (name, row, img) in zip(np.atleast_1d(axes), panels):
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(f"{name}: {row['pathology']} {row['lesion_type']}, density {row['density']}  |  model: "
                     f"{row['demo_p_calibrated']:.0%} malignant (calibrated), density {row['demo_density']}",
                     fontsize=9, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", default="/kaggle/working/demo")
    b.add_argument("--report", default="/kaggle/working/results/demo")
    b.add_argument("--zip", default="/kaggle/working/demo_bundle.zip")
    b.add_argument("--ckpt", default="auto")
    b.add_argument("--cbis-root", default="/kaggle/input")
    b.add_argument("--n-per-class", type=int, default=3)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--test-images", default=str(ROOT / "results" / "attention" / "test_images.csv"))
    b.add_argument("--predictions", default=str(ROOT / "results" / "attention" / "predictions.csv"))
    args = ap.parse_args(argv)
    return cmd_build(args)


if __name__ == "__main__":
    sys.exit(main())
