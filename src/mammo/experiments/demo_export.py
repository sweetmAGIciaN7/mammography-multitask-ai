"""Phase 6: build (and optionally publish) the Hugging Face Space for the demo.

    python -m mammo.experiments.demo_export build   [--out /kaggle/working/space]
    python -m mammo.experiments.demo_export upload  [--space mammography-multitask-demo]

``build`` needs the Phase 4 checkpoint ``cbis_mt_cbam_official.pt`` (attach the Phase 4 notebook's Output) and the
CBIS-DDSM dataset (for the example images). It
1. exports a slim checkpoint with the Phase 4a "official transfer" Platt parameters and provenance;
2. copies ``src/mammo`` + ``demo/app.py`` into a self-contained Space folder;
3. draws 6 **random** official-test images (seed 0; 3 malignant, 3 benign; never selected by hand) as examples;
4. checks that the demo code reproduces the Phase 5 predictions for those images (same model, same pixels), and
   renders a preview of what the Space will show.

``upload`` pushes the folder to ``https://huggingface.co/spaces/<your user>/<space>`` with a write token taken from
the ``HF_TOKEN`` environment variable or the Kaggle secret of the same name.
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
DENS = {1: "A", 2: "B", 3: "C", 4: "D"}


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


def space_files(out: Path) -> None:
    """Code part of the Space: the mammo package, the app, requirements and the Space card."""
    if (out / "mammo").exists():
        shutil.rmtree(out / "mammo")
    shutil.copytree(ROOT / "src" / "mammo", out / "mammo", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy(ROOT / "demo" / "app.py", out / "app.py")
    shutil.copy(ROOT / "demo" / "requirements.txt", out / "requirements.txt")
    shutil.copy(ROOT / "demo" / "README_space.md", out / "README.md")


def save_example(src: str, dst: Path, max_side: int = 2400) -> None:
    from PIL import Image
    with Image.open(src) as im:
        im = im.convert("L")
        s = max_side / max(im.size)
        if s < 1:
            im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
        im.save(dst, quality=92)


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
    space_files(out)

    # ---- examples: random official-test images, checked against the Phase 5 predictions
    test = pd.read_csv(args.test_images)
    pred = pd.read_csv(args.predictions).set_index("image_id")
    jpeg = locate_cbis(args.cbis_root)["jpeg"]
    test["file"] = [str(jpeg / _jpeg_relative(p)) for p in test["path"]]
    test = test[[os.path.exists(f) for f in test["file"]]].reset_index(drop=True)
    _log(f"official test images found on disk: {len(test)}")
    ex = pick_examples(test, args.n_per_class, args.seed)

    model, ckd = load_demo_model(out / "model.pt")
    (out / "examples").mkdir(exist_ok=True)
    rows, panels = [], []
    for k, r in enumerate(ex.itertuples(), 1):
        res_orig = predict_file(model, ckd, r.file)
        name = f"example_{k}.jpg"
        save_example(r.file, out / "examples" / name)
        res = predict_file(model, ckd, out / "examples" / name)
        phase5 = float(pred.loc[r.image_id, "p_malignant_mt_cbam"])
        rows.append({"file": name, "image_id": r.image_id, "pathology": "malignant" if r.pathology else "benign",
                     "lesion_type": r.lesion_type, "density": DENS.get(int(r.density), "?"),
                     "phase5_p_malignant": phase5, "demo_p_raw_original_file": res_orig["p_malignant_raw"],
                     "demo_p_raw_example_file": res["p_malignant_raw"], "demo_p_calibrated": res["p_malignant"],
                     "demo_density": "ABCD"[int(np.argmax(res["p_density"]))]})
        panels.append((name, rows[-1], render_panels(res, dpi=80)))
        _log(f"  {name}: {r.image_id:22s} {rows[-1]['pathology']:9s} phase5 {phase5:.4f}  demo(original) "
             f"{res_orig['p_malignant_raw']:.4f}  demo(example jpg) {res['p_malignant_raw']:.4f}  "
             f"calibrated {res['p_malignant']:.3f}")
    table = pd.DataFrame(rows)
    table[["file", "image_id", "pathology", "lesion_type", "density"]].to_csv(out / "examples" / "examples.csv",
                                                                                index=False)
    table.to_csv(report_dir / "examples_check.csv", index=False)
    diff = float(np.abs(table["phase5_p_malignant"] - table["demo_p_raw_original_file"]).max())
    diff_jpg = float(np.abs(table["demo_p_raw_original_file"] - table["demo_p_raw_example_file"]).max())
    check = {"max_abs_diff_vs_phase5": diff, "max_abs_diff_resaved_jpeg": diff_jpg, "n_examples": len(table),
             "space_files": sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file()
                                   and "__pycache__" not in str(p))}
    json.dump(check, open(report_dir / "demo_check.json", "w"), indent=1)
    _preview(panels, report_dir / "demo_preview.png")
    _log(f"max |demo - Phase 5| on the original files: {diff:.2e}; effect of re-saving as example JPEG: {diff_jpg:.3f}")
    if diff > 0.01:
        _log("ERROR: the demo does not reproduce the Phase 5 predictions. Send Claude the output of this cell.")
        return 1
    _log(f"Space folder ready: {out}  ({len(check['space_files'])} files)")
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


def _token() -> str | None:
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        return None


def cmd_upload(args) -> int:
    from huggingface_hub import HfApi
    token = _token()
    if not token:
        _log("ERROR: no Hugging Face token. In Kaggle: Add-ons -> Secrets -> add HF_TOKEN and tick the checkbox "
             "next to it, then run this cell again.")
        return 1
    api = HfApi(token=token)
    user = api.whoami()["name"]
    repo_id = f"{user}/{args.space}"
    api.create_repo(repo_id, repo_type="space", space_sdk="gradio", exist_ok=True)
    api.upload_folder(folder_path=args.out, repo_id=repo_id, repo_type="space",
                      commit_message=f"Demo from mammography-multitask-ai@{_git_commit()}",
                      ignore_patterns=["**/__pycache__/**", "*.pyc"])
    url = f"https://huggingface.co/spaces/{repo_id}"
    _log(f"uploaded -> {url}\nThe Space needs ~3-5 minutes to build the first time.")
    Path(args.report).mkdir(parents=True, exist_ok=True)
    json.dump({"space_url": url}, open(Path(args.report) / "space_url.json", "w"))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("build", "upload"):
        p = sub.add_parser(name)
        p.add_argument("--out", default="/kaggle/working/space")
        p.add_argument("--report", default="/kaggle/working/results/demo")
    b = sub.choices["build"]
    b.add_argument("--ckpt", default="auto")
    b.add_argument("--cbis-root", default="/kaggle/input")
    b.add_argument("--n-per-class", type=int, default=3)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--test-images", default=str(ROOT / "results" / "attention" / "test_images.csv"))
    b.add_argument("--predictions", default=str(ROOT / "results" / "attention" / "predictions.csv"))
    sub.choices["upload"].add_argument("--space", default="mammography-multitask-demo")
    args = ap.parse_args(argv)
    return {"build": cmd_build, "upload": cmd_upload}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
