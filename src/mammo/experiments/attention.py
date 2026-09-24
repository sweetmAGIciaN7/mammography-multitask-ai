"""Phase 5 - do the attention maps actually point at lesions?

Esen et al. (Sec. VI-C, Fig. 4) say the CBAM attention "strongly concentrates on irregular mass margins,
architectural distortions and suspicious microcalcifications", and report attention statistics (Gini, entropy,
peak intensity, 80%-mass area) without defining them or checking them against lesion outlines. Here every map is
scored against the radiologist ROI masks of CBIS-DDSM, next to maps that know nothing about lesions.

Data: the **official CBIS-DDSM test split** (patients never seen by the official-split models of Phase 4).
Models: the four Phase 4 checkpoints (``cbis_<variant>_official.pt``), not retrained.

Steps (Kaggle, see notebooks/05_attention.ipynb):

    python -m mammo.experiments.attention prepare     # test images + ROI masks in model space, mask preview
    python -m mammo.experiments.attention run         # maps for every model + baselines, per-image metrics, gallery
    python -m mammo.experiments.attention summarise   # bootstrap CIs, paired comparisons, claims table, chart

``summarise`` needs no GPU and no images: it only reads ``metrics.csv`` / ``test_images.csv``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

VARIANTS = ("mt_cbam", "mt_plain", "st_path", "st_dens")
VARIANT_LABEL = {"mt_cbam": "multi-task + CBAM", "mt_plain": "multi-task, no attention",
                 "st_path": "malignancy-only + CBAM", "st_dens": "density-only + CBAM"}
BASELINES = ("uniform", "random", "brightness", "centre")
METRICS = ["pg_px", "pg_px_tol", "energy_px", "auc_px", "iou_top10", "dice_top10", "iou_top5", "iou_top20",
           "pg_cell", "pg_cell_tol", "energy_cell", "auc_cell"]
MAIN_METRICS = ["pg_px", "pg_px_tol", "energy_px", "auc_px", "iou_top10"]
METRIC_LABEL = {"pg_px": "Pointing game (strict)", "pg_px_tol": "Pointing game (±1 cell)",
                "energy_px": "Energy in lesion", "auc_px": "Pixel AUC", "iou_top10": "IoU, top 10%",
                "dice_top10": "Dice, top 10%", "pg_cell": "Pointing, cell level", "pg_cell_tol": "Pointing, cell ±1",
                "energy_cell": "Energy, cell level", "auc_cell": "Cell AUC", "iou_top5": "IoU, top 5%",
                "iou_top20": "IoU, top 20%"}
DIST_STATS = ["gini", "entropy", "entropy_norm", "area80", "peak"]
PAPER = {  # Esen et al., Sec. VI-C 6)-8): mean ± sd
    "gini": {"malignant": (0.68, 0.12), "benign": (0.43, 0.15)},
    "entropy": {"malignant": (2.31, 0.38), "benign": (3.74, 0.51)},
    "area80": {"malignant": (0.124, 0.032), "benign": (0.287, 0.058)},
    "peak": {"dense": (0.82, 0.09), "non_dense": (0.61, 0.12), "pearson_r": 0.61},
}


class Logger:
    def __init__(self, path: Path | None = None):
        self.f = open(path, "a") if path else None

    def __call__(self, msg: str = ""):
        print(msg, flush=True)
        if self.f:
            self.f.write(msg + "\n"); self.f.flush()


def method_label(m: str) -> str:
    kind, _, who = m.partition(":")
    if kind == "base":
        return {"uniform": "Baseline: uniform", "random": "Baseline: random", "brightness": "Baseline: brightest tissue",
                "centre": "Baseline: breast centre"}.get(who, m)
    if kind == "ref":
        return "Ceiling: ROI mask at map resolution"
    if kind == "control":
        return {"imagenet_cbam": "Control: CBAM, ImageNet-only net", "imagenet_gradcam": "Control: Grad-CAM, ImageNet-only net"}[who]
    name = {"cbam": "CBAM attention", "gradcam_mal": "Grad-CAM malignancy", "gradcam_dens": "Grad-CAM density"}[kind]
    return f"{name} ({who})"


def method_order(m: str):
    kind, _, who = m.partition(":")
    kinds = ["cbam", "gradcam_mal", "gradcam_dens", "control", "ref", "base"]
    return (kinds.index(kind) if kind in kinds else 9, VARIANTS.index(who) if who in VARIANTS else 0, who)


def _data_file(args) -> Path:
    return Path(args.cache) / f"attention_{args.height}x{args.width}{'_smoke' if args.smoke else ''}.npz"


# ----------------------------------------------------------------------------------------------- prepare
def cmd_prepare(args) -> int:
    from mammo.cbis import index_cbis, locate_cbis
    from mammo.localization import (breast_region, candidate_files, pick_mask, read_abnormalities, roi_file_lookup)
    from mammo.preprocess import apply_transform, load_mammogram_with_transform

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    log = Logger(out / "log_prepare.txt")
    t0 = time.time()
    loc = locate_cbis(args.data_root)
    log(f"CBIS-DDSM csv : {loc['csv']}\nCBIS-DDSM jpeg: {loc['jpeg']}")
    df, _ = index_cbis(loc["csv"], loc["jpeg"])
    test = df[df["official_split"] == "test"].sort_values("image_id").reset_index(drop=True)
    if args.smoke:
        test = test.groupby("pathology").head(8).reset_index(drop=True)
    abn = read_abnormalities(loc["csv"])
    abn = abn[abn["image_id"].isin(set(test["image_id"]))].reset_index(drop=True)
    by_key, label = roi_file_lookup(pd.read_csv(Path(loc["csv"]) / "dicom_info.csv"), loc["jpeg"])
    log(f"official test split: {len(test)} images, {test['patient'].nunique()} patients, {len(abn)} abnormalities")

    groups = {k: g for k, g in abn.groupby("image_id")}

    def work(i):
        r = test.iloc[i]
        img, tf = load_mammogram_with_transform(r["path"], args.height, args.width)
        rows, masks = [], []
        for _, a in groups.get(r["image_id"], pd.DataFrame()).iterrows():
            files = candidate_files(a, loc["jpeg"], by_key)
            m, info = pick_mask(files, (tf.orig_w, tf.orig_h))
            row = {"image_id": r["image_id"], "patient": r["patient"], "abn_type": a["abn_type"],
                   "abn_id": int(a["abn_id"]), "pathology_raw": a["pathology_raw"], "malignant": int(a["malignant"]),
                   "assessment": a["assessment"], "subtlety": a["subtlety"], "shape_or_type": a["shape_or_type"],
                   "margins_or_distribution": a["margins_or_distribution"], **info,
                   "mask_label": label.get(info["mask_file"], "") if info["mask_file"] else ""}
            if m is not None:
                frac = apply_transform(m, tf)
                les = frac >= 0.5
                row["tiny_kept_peak"] = bool(not les.any() and frac.max() > 0)
                if row["tiny_kept_peak"]:          # lesion smaller than half a model pixel: keep its best pixel
                    les = frac >= frac.max() - 1e-6
                row.update(area_orig_px=int(m.sum()), area_model_px=int(les.sum()),
                           eq_diameter_model_px=float(2 * np.sqrt(les.sum() / np.pi)))
                if not les.any():
                    row["status"] = "outside_crop"
                    les = None
            else:
                les = None
            rows.append(row)
            masks.append(les)
        return img, tf, rows, masks

    N = len(test)
    images = np.zeros((N, args.height, args.width), np.uint8)
    lesion = np.zeros((N, args.height, args.width), bool)
    breast = np.zeros((N, args.height, args.width), bool)
    lesion_rows, lesion_masks, tfs = [], [], []
    with ThreadPoolExecutor(args.workers) as ex:
        for i, (img, tf, rows, masks) in enumerate(ex.map(work, range(N))):
            images[i] = img
            tfs.append(tf.to_dict())
            for row, m in zip(rows, masks):
                row["row"] = i
                row["lesion_index"] = len(lesion_masks) if m is not None else -1
                if m is not None:
                    lesion_masks.append(m)
                    lesion[i] |= m
                lesion_rows.append(row)
            b = breast_region(img)
            breast[i] = b | lesion[i]
            test.loc[i, "lesion_outside_breast_frac"] = float((lesion[i] & ~b).sum() / max(lesion[i].sum(), 1))
            if (i + 1) % 50 == 0 or i + 1 == N:
                log(f"  images + masks {i + 1}/{N}")
    les = pd.DataFrame(lesion_rows)
    tfd = pd.DataFrame(tfs).add_prefix("tf_")
    test = pd.concat([test, tfd], axis=1)
    usable = les[les["lesion_index"] >= 0]
    kinds = usable.groupby("row")["abn_type"].agg(lambda s: "both" if s.nunique() > 1 else s.iloc[0])
    test["lesion_type"] = test.index.map(kinds).fillna("no_mask")
    test["n_abnormalities"] = test.index.map(les.groupby("row").size()).fillna(0).astype(int)
    test["n_usable_masks"] = test.index.map(usable.groupby("row").size()).fillna(0).astype(int)
    test["lesion_area_px"] = lesion.reshape(N, -1).sum(1)
    test["breast_area_px"] = breast.reshape(N, -1).sum(1)
    test["area_fraction"] = test["lesion_area_px"] / test["breast_area_px"].clip(lower=1)
    cell_area = 32 * 32
    test["size_bin"] = pd.cut(test["lesion_area_px"] / cell_area, [-1, 0, 1, 4, np.inf],
                              labels=["none", "< 1 cell", "1-4 cells", "> 4 cells"]).astype(str)
    test["usable"] = test["n_usable_masks"] > 0
    test = test.drop(columns=["fold"], errors="ignore")
    test.to_csv(out / "test_images.csv", index=False)
    les.to_csv(out / "lesions.csv", index=False)
    lm = np.stack(lesion_masks) if lesion_masks else np.zeros((0, args.height, args.width), bool)
    f = _data_file(args); f.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, images=images, lesion=lesion, breast=breast, lesion_masks=lm)

    st = les["status"].value_counts().to_dict()
    ok = les["status"].isin(["ok", "ok_resized"])
    stats = {
        "test_images": int(N), "test_patients": int(test["patient"].nunique()), "abnormalities": int(len(les)),
        "mask_status": {k: int(v) for k, v in st.items()},
        "usable_masks": int(ok.sum()),
        "mask_file_labelled_as": {k: int(v) for k, v in les.loc[ok, "mask_label"].value_counts().items()},
        "images_with_usable_mask": int(test["usable"].sum()),
        "patients_with_usable_mask": int(test.loc[test["usable"], "patient"].nunique()),
        "images_by_lesion_type": {k: int(v) for k, v in test.loc[test["usable"], "lesion_type"].value_counts().items()},
        "images_by_size_bin": {k: int(v) for k, v in test.loc[test["usable"], "size_bin"].value_counts().items()},
        "malignant_images_with_mask": int(test.loc[test["usable"], "pathology"].sum()),
        "tiny_lesions_kept_as_peak_pixel": int(les.get("tiny_kept_peak", pd.Series(dtype=bool)).fillna(False).sum()),
        "median_lesion_area_fraction_of_breast": float(test.loc[test["usable"], "area_fraction"].median()),
        "median_lesion_eq_diameter_px": {k: float(v) for k, v in
                                         les[ok].groupby("abn_type")["eq_diameter_model_px"].median().items()},
        "lesion_pixels_outside_breast_region_mean": float(test.loc[test["usable"], "lesion_outside_breast_frac"].mean()),
        "image_size": [args.height, args.width], "map_cell_px": 32,
    }
    json.dump(stats, open(out / "data_stats.json", "w"), indent=1)
    log(json.dumps(stats, indent=1))
    rng = np.random.default_rng(0)
    pool = np.flatnonzero(test["usable"].to_numpy())
    pick = np.sort(rng.choice(pool, size=min(16, len(pool)), replace=False)) if len(pool) else []
    from mammo.plots import mask_preview
    titles = [f"{r.image_id}\n{'malignant' if r.pathology else 'benign'} | {r.lesion_type}" for r in test.loc[pick].itertuples()]
    mask_preview(images[pick], lesion[pick], breast[pick], titles, out / "mask_preview.png")
    log(f"mask preview (16 random test images, seed 0) -> {out / 'mask_preview.png'}  ({(time.time() - t0) / 60:.1f} min)")
    if not ok.any():
        log("ERROR: no usable ROI mask found. Send Claude log_prepare.txt."); return 1
    return 0


# ----------------------------------------------------------------------------------------------- run
def find_checkpoints(spec: str) -> dict:
    roots = [Path("/kaggle/input"), Path("/kaggle/working/checkpoints")] if spec == "auto" else [Path(spec)]
    found = {}
    for root in roots:
        if not root.exists():
            continue
        for f in sorted(root.rglob("cbis_*_official.pt")):
            name = f.name[len("cbis_"):-len("_official.pt")]
            if name in VARIANTS:
                found.setdefault(name, f)
    return found


def _load_model(f, device):
    import torch
    from mammo.train import TrainConfig, build_model
    ck = torch.load(f, map_location="cpu", weights_only=False)
    cfg = TrainConfig(**{k: (tuple(v) if k == "tasks" else v) for k, v in ck["config"].items()})
    cfg.pretrained = False
    model = build_model(cfg).to(device)
    model.load_state_dict(ck["state_dict"])
    return model.eval(), ck


def _untrained(variant, device, seed=0):
    """Never trained on mammograms: ImageNet-pretrained backbone, randomly initialised CBAM and heads. (A fully random
    EfficientNet is useless as a control: in eval mode its activations collapse to ~0 and every map is constant.)"""
    import torch
    from mammo.experiments.cbis import CONFIGS
    from mammo.model import MultiTaskNet
    torch.manual_seed(seed)
    spec = CONFIGS[variant]
    return MultiTaskNet("efficientnet_b0", pretrained=True, attention=spec["attention"], tasks=spec["tasks"]).to(device).eval()


def _maps_for(model, images, device, batch=16):
    import torch
    from mammo.explain import explain_batch
    from mammo.train import _Prep
    prep = _Prep(device)
    parts = {}
    for s in range(0, len(images), batch):
        x = prep(torch.from_numpy(images[s:s + batch]), train=False)
        for k, v in explain_batch(model, x).items():
            parts.setdefault(k, []).append(v)
    return {k: np.concatenate(v) for k, v in parts.items()}


def cmd_run(args) -> int:
    import torch
    from mammo.metrics import auc_fn

    out = Path(args.out)
    log = Logger(out / "log_run.txt")
    t0 = time.time()
    test = pd.read_csv(out / "test_images.csv", dtype={"image_id": str, "patient": str})
    data = np.load(_data_file(args))
    images, lesion, breast = data["images"], data["lesion"], data["breast"]
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not args.smoke:
        log("ERROR: no GPU. In Kaggle: Settings -> Accelerator -> GPU T4 x2, then run again."); return 1
    ckpts = find_checkpoints(args.checkpoints)
    log("checkpoints: " + (", ".join(f"{k} <- {v}" for k, v in ckpts.items()) or "none found"))
    missing = [v for v in VARIANTS if v not in ckpts]
    if missing and not args.smoke:
        log(f"ERROR: no checkpoint for {missing}. Attach the Output of the Phase 4 notebook (Add Input -> Your Work -> "
            "Notebooks), or train them with: python -m mammo.experiments.external train " + " ".join(missing)); return 1

    maps, preds, info = {}, pd.DataFrame({"image_id": test["image_id"]}), {"checkpoints": {}, "cross_check": {}}
    for v in VARIANTS:
        if v in ckpts:
            model, ck = _load_model(ckpts[v], device)
            info["checkpoints"][v] = str(ckpts[v])
            if (ck["height"], ck["width"]) != (args.height, args.width):
                log(f"ERROR: {v} was trained at {ck['height']}x{ck['width']}, images are {args.height}x{args.width}"); return 1
        else:
            model = _untrained(v, device, seed=1)
            info["checkpoints"][v] = "UNTRAINED (smoke test only)"
        m = _maps_for(model, images, device)
        if "cbam" in m:
            maps[f"cbam:{v}"] = m["cbam"]
        if "gradcam_mal" in m:
            maps[f"gradcam_mal:{v}"] = m["gradcam_mal"]
            preds[f"p_malignant_{v}"] = m["p_malignant"]
        if "gradcam_dens" in m:
            maps[f"gradcam_dens:{v}"] = m["gradcam_dens"]
            preds[f"pred_density_{v}"] = m["p_density"].argmax(1)
        ph4 = Path(args.external_results) / v / "cbis_official_predictions.csv"
        o = pd.read_csv(ph4, dtype={"image_id": str}).merge(preds[["image_id", f"p_malignant_{v}"]], on="image_id") \
            if ph4.exists() and f"p_malignant_{v}" in preds else pd.DataFrame()
        if len(o) and o["y"].nunique() == 2:
            info["cross_check"][v] = {
                "n": int(len(o)), "max_abs_diff_p_malignant": float((o["p_malignant"] - o[f"p_malignant_{v}"]).abs().max()),
                "auc_phase4": auc_fn(o["y"], o["p_malignant"]), "auc_now": auc_fn(o["y"], o[f"p_malignant_{v}"])}
            log(f"  {v}: same test images as Phase 4? n={len(o)}, AUC {info['cross_check'][v]['auc_phase4']:.3f} -> "
                f"{info['cross_check'][v]['auc_now']:.3f}, max |dP| {info['cross_check'][v]['max_abs_diff_p_malignant']:.4f}")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    ctrl = _untrained("mt_cbam", device, seed=0)
    m = _maps_for(ctrl, images, device)
    maps["control:imagenet_cbam"], maps["control:imagenet_gradcam"] = m["cbam"], m["gradcam_mal"]
    del ctrl
    log(f"maps: {len(maps)} methods ({(time.time() - t0) / 60:.1f} min)")
    return evaluate_maps(maps, preds, info, test, images, lesion, breast, args, log, t0)


def evaluate_maps(maps: dict, preds: pd.DataFrame, info: dict, test: pd.DataFrame, images, lesion, breast, args, log,
                  t0: float | None = None) -> int:
    """Everything after the forward passes (no torch): CBAM range, correctness, per-image metrics, gallery."""
    from mammo.localization import Geometry, adaptive_pool, baseline_maps, distribution_stats, localisation_metrics
    out = Path(args.out)
    t0 = t0 or time.time()
    map_shape = maps["gradcam_mal:mt_cbam"].shape[1:]

    # CBAM is a sigmoid gate: is it flat, saturated, or does it actually vary?
    rng_info = {}
    for k, a in maps.items():
        if not k.startswith(("cbam", "control:imagenet_cbam")):
            continue
        per = a.reshape(len(a), -1)
        rng_info[k] = {"min": float(per.min()), "max": float(per.max()), "mean": float(per.mean()),
                       "median_within_image_range": float(np.median(per.max(1) - per.min(1))),
                       "median_within_image_sd": float(np.median(per.std(1))),
                       "share_cells_above_0.99": float((per > 0.99).mean()), "share_cells_below_0.01": float((per < 0.01).mean())}
    info["cbam_dynamic_range"] = rng_info
    log("CBAM dynamic range: " + json.dumps(rng_info, indent=1))

    # correctness of each model on each image
    y = test["pathology"].to_numpy(); dens = test["density"].to_numpy()
    for v in VARIANTS:
        if f"p_malignant_{v}" in preds:
            preds[f"correct_mal_{v}"] = ((preds[f"p_malignant_{v}"].to_numpy() >= 0.5).astype(int) == y).astype(int)
        if f"pred_density_{v}" in preds:
            preds[f"correct_dens_{v}"] = np.where(dens >= 0, (preds[f"pred_density_{v}"].to_numpy() == dens).astype(int), -1)
    preds.to_csv(out / "predictions.csv", index=False)

    rows = []
    rng = np.random.default_rng(args.seed)
    usable = np.flatnonzero(test["usable"].to_numpy())
    for n_done, i in enumerate(usable):
        g = Geometry(lesion[i], breast[i], map_shape)
        cell_breast = g.cell_breast
        base = baseline_maps(images[i], breast[i], map_shape, rng, n_random=args.n_random)
        items = [(k, a[i]) for k, a in maps.items()] + [(f"base:{k}", a) for k, a in base.items()]
        items.append(("ref:oracle", g.cell_lesion_frac))   # the best any 20x12 map could do: the mask, pooled
        for k, a in items:
            r = {"row": int(i), "method": k, **localisation_metrics(a, g)}
            if not k.startswith(("base:", "ref:")):
                r.update(distribution_stats(a, prefix="full_"))
                r.update(distribution_stats(a, keep=cell_breast, prefix="breast_"))
                r.update(distribution_stats(adaptive_pool(a, 7, 7), prefix="p7_"))
            rows.append(r)
        if (n_done + 1) % 50 == 0 or n_done + 1 == len(usable):
            log(f"  metrics {n_done + 1}/{len(usable)} images ({(time.time() - t0) / 60:.1f} min)")
    met = pd.DataFrame(rows)
    rnd = met["method"].str.startswith("base:random_")
    avg = met[rnd].groupby("row", as_index=False)[[c for c in met.columns if c not in ("row", "method")]].mean(numeric_only=True)
    avg["method"] = "base:random"
    met = pd.concat([met[~rnd], avg], ignore_index=True)
    meta = test[["image_id", "patient", "pathology", "density", "lesion_type", "size_bin", "lesion_area_px", "view"]]
    met = met.merge(meta, left_on="row", right_index=True).merge(preds, on="image_id")
    met["variant"] = met["method"].str.split(":").str[1].where(met["method"].str.split(":").str[0].isin(
        ["cbam", "gradcam_mal", "gradcam_dens"]), "mt_cbam")
    corr = []
    for mth, v in zip(met["method"], met["variant"]):
        corr.append(f"correct_dens_{v}" if mth.startswith("gradcam_dens") or v == "st_dens" else f"correct_mal_{v}")
    met["correct"] = [row[c] if c in met.columns else -1 for row, c in zip(met.to_dict("records"), corr)]
    met.to_csv(out / "metrics.csv", index=False)
    np.savez_compressed(out / "maps.npz", image_id=test["image_id"].to_numpy(), **{k: a.astype(np.float16) for k, a in maps.items()})
    json.dump(info, open(out / "run_info.json", "w"), indent=1, default=float)

    from mammo.plots import attention_gallery
    grng = np.random.default_rng(0)
    pick = np.sort(grng.choice(usable, size=min(args.gallery_n, len(usable)), replace=False))
    panels = [("cbam:mt_cbam", "CBAM (mt_cbam)"), ("gradcam_mal:mt_cbam", "Grad-CAM mal. (mt_cbam)"),
              ("gradcam_mal:mt_plain", "Grad-CAM mal. (mt_plain)")]
    panels = [p for p in panels if p[0] in maps]
    titles = [f"{r.image_id}\n{'MALIGNANT' if r.pathology else 'benign'} {r.lesion_type}, "
              f"P(mal) {preds.loc[k, 'p_malignant_mt_cbam']:.2f}" for k, r in zip(pick, test.loc[pick].itertuples())]
    attention_gallery(images[pick], lesion[pick], breast[pick], {lab: maps[k][pick] for k, lab in panels}, titles,
                      out / "gallery.png")
    log(f"gallery: {len(pick)} test images drawn at random (seed 0) -> {out / 'gallery.png'}")
    log(f"done in {(time.time() - t0) / 60:.1f} min")
    return 0


# ----------------------------------------------------------------------------------------------- summarise
def _subsets(met_m: pd.DataFrame) -> dict:
    d = met_m
    s = {"all": np.ones(len(d), bool),
         "mass only": (d["lesion_type"] == "mass").to_numpy(), "calcification only": (d["lesion_type"] == "calc").to_numpy(),
         "mass + calcification": (d["lesion_type"] == "both").to_numpy(),
         "benign": (d["pathology"] == 0).to_numpy(), "malignant": (d["pathology"] == 1).to_numpy(),
         "model correct": (d["correct"] == 1).to_numpy(), "model wrong": (d["correct"] == 0).to_numpy()}
    for b in ("< 1 cell", "1-4 cells", "> 4 cells"):
        s[f"lesion {b}"] = (d["size_bin"] == b).to_numpy()
    return s


def summarise_frame(met: pd.DataFrame, test: pd.DataFrame, nboot: int, run_info: dict | None = None,
                    data_stats: dict | None = None) -> dict:
    from mammo.localization import ClusterBootstrap, cohens_d, correlation_ci
    from scipy.stats import ttest_ind

    met = met.sort_values(["method", "row"]).reset_index(drop=True)
    rows = np.sort(met["row"].unique())
    base = met[met["method"] == "base:uniform"].set_index("row").loc[rows]
    bs = ClusterBootstrap(base["patient"].to_numpy(), n=nboot, seed=0)
    methods = sorted(met["method"].unique(), key=method_order)

    def frame(m):
        return met[met["method"] == m].set_index("row").reindex(rows)

    loc = {}
    for m in methods:
        d = frame(m)
        loc[m] = {"label": method_label(m), "subsets": {}}
        for sname, mask in _subsets(d).items():
            if mask.sum() == 0:
                continue
            loc[m]["subsets"][sname] = {k: bs.mean(d[k].to_numpy(), mask) for k in METRICS if k in d}
    chance = {s: bs.mean(frame("base:uniform")["area_fraction"].to_numpy(), mk)
              for s, mk in _subsets(frame("base:uniform")).items() if mk.sum()}

    pairs = []
    model_methods = [m for m in methods if not m.startswith(("base", "control", "ref"))]
    for m in model_methods + ["control:imagenet_gradcam", "control:imagenet_cbam"]:
        for b in ("base:uniform", "base:brightness", "base:centre", "base:random"):
            pairs.append((m, b))
    extra = [("cbam:mt_cbam", "gradcam_mal:mt_cbam", "Attention map vs Grad-CAM of the same model"),
             ("cbam:mt_cbam", "cbam:st_dens", "Malignancy-trained vs density-only attention"),
             ("cbam:st_path", "cbam:st_dens", "Malignancy-only vs density-only attention"),
             ("gradcam_mal:mt_cbam", "gradcam_mal:mt_plain", "Does CBAM make Grad-CAM more lesion-focused?"),
             ("gradcam_mal:mt_cbam", "gradcam_dens:mt_cbam", "Malignancy head vs density head, same model"),
             ("gradcam_mal:mt_cbam", "control:imagenet_gradcam", "Mammography-trained vs ImageNet-only network (Grad-CAM)"),
             ("cbam:mt_cbam", "control:imagenet_cbam", "Mammography-trained vs ImageNet-only network (CBAM)")]
    comps = []
    for a, b, *q in [(p[0], p[1]) for p in pairs] + extra:
        if a not in methods or b not in methods:
            continue
        da, db = frame(a), frame(b)
        for k in ("pg_px_tol", "energy_px", "auc_px"):
            for sname in ("all", "mass only", "calcification only"):
                mk = _subsets(da)[sname]
                if mk.sum() < 5:
                    continue
                comps.append({"a": a, "b": b, "question": q[0] if q else "", "metric": k, "subset": sname,
                              **bs.paired(da[k].to_numpy(), db[k].to_numpy(), mk)})

    # the paper's attention statistics
    paper = {}
    for m in [x for x in methods if x.startswith(("cbam", "gradcam_mal:mt_cbam", "control:imagenet_cbam"))]:
        d = frame(m)
        mal, ben = (d["pathology"] == 1).to_numpy(), (d["pathology"] == 0).to_numpy()
        dense, nondense = d["density"].isin([2, 3]).to_numpy(), d["density"].isin([0, 1]).to_numpy()
        res = {}
        for grid in ("full", "breast", "p7"):
            for st in DIST_STATS:
                c = f"{grid}_{st}"
                if c not in d:
                    continue
                v = d[c].to_numpy(float)
                entry = {}
                for gname, ga, gb in (("pathology", mal, ben), ("density", dense, nondense)):
                    a_, b_ = v[ga & ~np.isnan(v)], v[gb & ~np.isnan(v)]
                    if len(a_) < 2 or len(b_) < 2:
                        continue
                    entry[gname] = {"a": {"mean": float(a_.mean()), "sd": float(a_.std(ddof=1)), "n": int(len(a_))},
                                    "b": {"mean": float(b_.mean()), "sd": float(b_.std(ddof=1)), "n": int(len(b_))},
                                    **bs.group_diff(v, ga, gb), "cohens_d": cohens_d(a_, b_),
                                    "welch_p": float(ttest_ind(a_, b_, equal_var=False).pvalue)}
                entry["n_cells"] = int(np.nanmedian(d[f"{grid}_n_cells"]))
                res[c] = entry
            k = d["density"].to_numpy() >= 0
            if f"{grid}_peak" in d and k.sum() > 10:
                res[f"{grid}_peak_vs_density"] = correlation_ci(d[f"{grid}_peak"].to_numpy()[k], d["density"].to_numpy()[k] + 1,
                                                                d["patient"].to_numpy()[k], n=min(nboot, 1000))
        paper[m] = res
    t = test[test["usable"]] if "usable" in test else test
    return {"n_images": int(len(rows)), "n_patients": int(base["patient"].nunique()),
            "subset_sizes": {s: int(mk.sum()) for s, mk in _subsets(frame("base:uniform")).items()},
            "chance_area_fraction": chance, "localisation": loc, "comparisons": comps, "paper_statistics": paper,
            "run_info": run_info or {}, "data": data_stats or {}, "n_test_images_total": int(len(test)),
            "n_usable": int(len(t))}


def cmd_summarise(args) -> int:
    out = Path(args.out)
    log = Logger(out / "log_summary.txt")
    met = pd.read_csv(out / "metrics.csv", dtype={"image_id": str, "patient": str})
    test = pd.read_csv(out / "test_images.csv", dtype={"image_id": str, "patient": str})
    ri = json.load(open(out / "run_info.json")) if (out / "run_info.json").exists() else None
    ds = json.load(open(out / "data_stats.json")) if (out / "data_stats.json").exists() else None
    s = summarise_frame(met, test, 200 if args.smoke else args.nboot, ri, ds)
    json.dump(s, open(out / "summary.json", "w"), indent=1, default=float)
    (out / "summary.md").write_text(render_markdown(s))
    from mammo.plots import plot_localisation
    plot_localisation(s, out / "localisation_chart.png")
    log(render_markdown(s))
    return 0


def _ci(c, fmt=".3f"):
    return f"{c[0]:{fmt}}–{c[1]:{fmt}}"


def render_markdown(s: dict) -> str:
    L = [f"Official CBIS-DDSM test split: {s['n_usable']} of {s['n_test_images_total']} images have a usable ROI mask "
         f"({s['n_patients']} patients). Subsets: " + ", ".join(f"{k} {v}" for k, v in s["subset_sizes"].items()) + ".", "",
         "### Localisation (inside the breast; 95% CI, patient bootstrap)", "",
         "| Method | " + " | ".join(METRIC_LABEL[k] for k in MAIN_METRICS) + " |", "|---|" + "---:|" * len(MAIN_METRICS)]
    for m, x in s["localisation"].items():
        a = x["subsets"]["all"]
        L.append(f"| {x['label']} | " + " | ".join(f"{a[k]['mean']:.3f} ({_ci(a[k]['ci95'], '.2f')})" for k in MAIN_METRICS) + " |")
    ch = s["chance_area_fraction"]["all"]
    L += ["", f"The uniform baseline is chance level: a map with no information scores the lesion's share of the breast "
          f"on strict pointing and energy (here {ch['mean']:.3f}, {_ci(ch['ci95'])}), the share of the breast within one "
          f"cell of the lesion on ±1-cell pointing, and 0.5 AUC. The ceiling row is the ROI mask itself pooled to the "
          f"map grid: the best any map of this resolution could score.", ""]
    for sub in ("mass only", "calcification only", "benign", "malignant", "model correct", "model wrong",
                "lesion < 1 cell", "lesion 1-4 cells", "lesion > 4 cells"):
        L += [f"#### {sub}", "", "| Method | Pointing ±1 cell | Energy in lesion | Pixel AUC |", "|---|---:|---:|---:|"]
        for m, x in s["localisation"].items():
            a = x["subsets"].get(sub)
            if a:
                L.append(f"| {x['label']} | " + " | ".join(f"{a[k]['mean']:.3f} ({_ci(a[k]['ci95'], '.2f')})"
                                                           for k in ("pg_px_tol", "energy_px", "auc_px")) + f" |")
        L.append("")
    L += ["### Paired comparisons (A − B on the same images)", "",
          "| A | B | Metric | Subset | A − B | 95% CI | P(A not better) |", "|---|---|---|---|---:|---:|---:|"]
    for c in s["comparisons"]:
        if c["subset"] == "all" or c["question"]:
            L.append(f"| {method_label(c['a'])} | {method_label(c['b'])} | {METRIC_LABEL[c['metric']]} | {c['subset']} | "
                     f"{c['diff']:+.3f} | {c['ci95'][0]:+.3f} to {c['ci95'][1]:+.3f} | {c['p_not_better']:.3f} |")
    L += ["", "### The paper's attention statistics, recomputed", "",
          "Definitions: Gini of the raw map values; Shannon entropy of the map normalised to sum 1, in nats (max = ln n); "
          "normalised entropy = entropy / ln n; area80 = smallest share of cells holding 80% of the map's mass; peak = max "
          "value. `full` = all 20×12 cells, `breast` = breast cells only, `p7` = map average-pooled to 7×7 (the paper's size).", "",
          "| Map | Statistic | Paper (malignant vs benign) | Ours malignant | Ours benign | Difference (95% CI) | Cohen's d | Welch p |",
          "|---|---|---|---:|---:|---:|---:|---:|"]
    for m, res in s["paper_statistics"].items():
        for grid in ("full", "p7", "breast"):
            for st in ("gini", "entropy", "entropy_norm", "area80", "peak"):
                e = res.get(f"{grid}_{st}", {}).get("pathology")
                if not e:
                    continue
                pp = PAPER.get(st, {})
                paper = (f"{pp['malignant'][0]} ± {pp['malignant'][1]} vs {pp['benign'][0]} ± {pp['benign'][1]}"
                         if "malignant" in pp else "–")
                L.append(f"| {method_label(m)} | {grid} {st} | {paper} | {e['a']['mean']:.3f} ± {e['a']['sd']:.3f} | "
                         f"{e['b']['mean']:.3f} ± {e['b']['sd']:.3f} | {e['diff']:+.3f} ({e['ci95'][0]:+.3f} to "
                         f"{e['ci95'][1]:+.3f}) | {e['cohens_d']:+.2f} | {e['welch_p']:.3g} |")
    L += ["", "| Map | Peak (grid) | Paper dense vs non-dense | Ours dense (C/D) | Ours non-dense (A/B) | Difference (95% CI) | "
          "Cohen's d | Pearson r with grade (95% CI) | Spearman |", "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for m, res in s["paper_statistics"].items():
        for grid in ("full", "p7", "breast"):
            e = res.get(f"{grid}_peak", {}).get("density")
            r = res.get(f"{grid}_peak_vs_density")
            if not e or not r:
                continue
            L.append(f"| {method_label(m)} | {grid} | 0.82 ± 0.09 vs 0.61 ± 0.12, r = 0.61 | {e['a']['mean']:.3f} ± {e['a']['sd']:.3f} | "
                     f"{e['b']['mean']:.3f} ± {e['b']['sd']:.3f} | {e['diff']:+.3f} ({e['ci95'][0]:+.3f} to {e['ci95'][1]:+.3f}) | "
                     f"{e['cohens_d']:+.2f} | {r['pearson']:+.3f} ({_ci(r['pearson_ci95'], '+.2f')}) | {r['spearman']:+.3f} |")
    ri = s.get("run_info", {})
    if ri.get("cbam_dynamic_range"):
        L += ["", "### CBAM dynamic range (sigmoid gate values)", "", "| Map | min | max | mean | median range within an image | "
              "median SD within an image |", "|---|---:|---:|---:|---:|---:|"]
        for k, v in ri["cbam_dynamic_range"].items():
            L.append(f"| {method_label(k)} | {v['min']:.3f} | {v['max']:.3f} | {v['mean']:.3f} | "
                     f"{v['median_within_image_range']:.3f} | {v['median_within_image_sd']:.3f} |")
    if ri.get("cross_check"):
        L += ["", "Checkpoint check (same test images and predictions as Phase 4): " + "; ".join(
            f"{k}: AUC {v['auc_phase4']:.3f} → {v['auc_now']:.3f}, max |ΔP| {v['max_abs_diff_p_malignant']:.4f}"
            for k, v in ri["cross_check"].items())]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["prepare", "run", "summarise"])
    ap.add_argument("--data-root", default="/kaggle/input")
    ap.add_argument("--out", default="/kaggle/working/results/attention")
    ap.add_argument("--cache", default="/tmp/mammo_cache")
    ap.add_argument("--checkpoints", default="auto", help="'auto' searches /kaggle/input and /kaggle/working/checkpoints")
    ap.add_argument("--external-results", default="results/external", help="Phase 4 predictions, for the checkpoint check")
    ap.add_argument("--height", type=int, default=640)
    ap.add_argument("--width", type=int, default=384)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--nboot", type=int, default=2000)
    ap.add_argument("--n-random", type=int, default=5)
    ap.add_argument("--gallery-n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="16 test images, small images, untrained models if needed")
    args = ap.parse_args(argv)
    if args.smoke:
        args.out += "_smoke"
        args.height, args.width, args.checkpoints = 160, 96, "/nonexistent" if args.checkpoints == "auto" else args.checkpoints
        args.n_random, args.gallery_n = 2, 6
    return {"prepare": cmd_prepare, "run": cmd_run, "summarise": cmd_summarise}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
