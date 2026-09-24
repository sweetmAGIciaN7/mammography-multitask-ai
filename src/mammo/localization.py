"""Phase 5: do attention / Grad-CAM maps point at the lesion? (numpy only, no torch)

Three parts:

1. **Lesion masks** for CBIS-DDSM. Every abnormality has a radiologist-drawn ROI mask. The Kaggle JPEG release
   stores the mask and the lesion crop in the same series folder, and the labels are known to be swapped for
   some cases, so a mask is recognised by its *content*: a (near) black-and-white image with the same size as
   the full mammogram. Masks are then sent through the image's own crop/flip/resize (``preprocess.apply_transform``).
2. **Localisation metrics**, all computed *inside the breast only* (padding and film background are trivially
   "not lesion" and would flatter every method). See ``localisation_metrics``.
3. **Distribution statistics** the paper reports (Gini, entropy, peak, 80%-mass area), with exact definitions,
   plus baselines and the patient-clustered bootstrap used for every confidence interval.

Map conventions: a map is a non-negative ``h x w`` array at feature-map resolution (20 x 12 for 640 x 384 input,
one cell = 32 x 32 pixels). Upsampling to pixel resolution is bilinear with ``align_corners=False`` (identical
to ``torch.nn.functional.interpolate``) for every method and baseline alike.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
from scipy.stats import rankdata

from mammo.cbis import CASE_FILES, _UID, _jpeg_relative, _norm_cols

# ------------------------------------------------------------------------------------------------ masks
MASK_STATUS = ("ok", "ok_resized", "no_file_of_image_size", "not_binary", "empty", "no_candidate_files")


def read_abnormalities(csv_dir: str | Path) -> pd.DataFrame:
    """One row per abnormality (all four case files) with what Phase 5 needs: image id, lesion type, pathology,
    and every series UID mentioned in the ROI-mask / cropped-image columns."""
    frames = []
    for (abn, split), fname in CASE_FILES.items():
        d = _norm_cols(pd.read_csv(Path(csv_dir) / fname))
        d["abn_type"], d["official_split"] = abn, split
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)

    def col(name, default=""):
        return df[name] if name in df else pd.Series(default, index=df.index)

    patient = df["patient_id"].astype(str).str.strip()
    side = df["left_or_right_breast"].astype(str).str.strip().str.upper()
    view = df["image_view"].astype(str).str.strip().str.upper()
    full = df["image_file_path"].astype(str).str.strip().str.replace("\\", "/", regex=False)
    roi = col("roi_mask_file_path").fillna("").astype(str).str.strip().str.replace("\\", "/", regex=False)
    crop = col("cropped_image_file_path").fillna("").astype(str).str.strip().str.replace("\\", "/", regex=False)
    abn_id = pd.to_numeric(col("abnormality_id", 1), errors="coerce").fillna(1).astype(int)

    def uids(*paths):
        out = []
        for p in paths:
            for part in str(p).split("/")[:-1]:
                if _UID.match(part) and part not in out:
                    out.append(part)
        return out

    out = pd.DataFrame({
        "image_id": patient + "_" + side + "_" + view,
        "patient": patient, "abn_type": df["abn_type"], "abn_id": abn_id,
        "official_split": df["official_split"],
        "pathology_raw": df["pathology"].astype(str).str.strip().str.upper(),
        "case_key": full.str.split("/").str[0],
        "roi_key": roi.str.split("/").str[0],
        "roi_uids": [uids(a, b) for a, b in zip(roi, crop)],
        "assessment": pd.to_numeric(col("assessment", np.nan), errors="coerce"),
        "subtlety": pd.to_numeric(col("subtlety", np.nan), errors="coerce"),
        "shape_or_type": col("mass_shape").fillna("").astype(str).where(df["abn_type"] == "mass",
                                                                          col("calc_type").fillna("").astype(str)),
        "margins_or_distribution": col("mass_margins").fillna("").astype(str).where(
            df["abn_type"] == "mass", col("calc_distribution").fillna("").astype(str)),
    })
    out["malignant"] = (out["pathology_raw"] == "MALIGNANT").astype(int)
    out.loc[out["roi_key"] == "", "roi_key"] = out["case_key"] + "_" + out["abn_id"].astype(str)
    return out


def roi_file_lookup(dicom_info: pd.DataFrame, jpeg_dir: str | Path) -> tuple[dict, dict]:
    """(DICOM PatientID -> JPEG files of every non-full-mammogram series, JPEG path -> its SeriesDescription)."""
    di = _norm_cols(dicom_info)
    desc = di.get("seriesdescription", pd.Series("", index=di.index)).fillna("").astype(str).str.strip().str.lower()
    by_key, label = {}, {}
    jpeg_dir = Path(jpeg_dir)
    for (_, r), d in zip(di.iterrows(), desc):
        rel = _jpeg_relative(r.get("image_path", ""))
        if not rel:
            continue
        path = str(jpeg_dir / rel)
        label[path] = d
        if "full mammogram" in d:
            continue
        key = str(r.get("patientid", "") or "").strip()
        if key:
            by_key.setdefault(key, []).append(path)
    return by_key, label


def candidate_files(row, jpeg_dir: str | Path, by_key: dict) -> list[str]:
    """Every JPEG that could be this abnormality's ROI mask: the files in the series folders named in the case
    file, plus the dicom_info entries of its ROI PatientID (e.g. 'Mass-Test_P_00016_LEFT_CC_1')."""
    files = []
    for uid in row["roi_uids"]:
        folder = Path(jpeg_dir) / uid
        if folder.is_dir():
            files += sorted(str(p) for p in folder.glob("*.jp*g"))
    files += by_key.get(row["roi_key"], [])
    return list(dict.fromkeys(files))


def looks_binary(a: np.ndarray, tol: float = 0.98) -> bool:
    """A JPEG'd 0/255 mask: almost every pixel is near black or near white (JPEG blurs the edges a little)."""
    return bool(((a < 32) | (a > 223)).mean() >= tol)


def pick_mask(files: list[str], full_size: tuple[int, int], size_tol: float = 0.02) -> tuple[np.ndarray | None, dict]:
    """Choose the ROI mask among ``files`` by content. ``full_size`` = (width, height) of the full mammogram.

    Returns (bool mask at full size or None, info). Files of exactly the image size are tried first; a file within
    ``size_tol`` of it in both dimensions is accepted and resized ('ok_resized')."""
    W, H = full_size
    info = {"status": "no_candidate_files", "mask_file": None, "n_candidates": len(files)}
    if not files:
        return None, info
    sized = []
    for f in files:
        try:
            with Image.open(f) as im:
                w, h = im.size
        except Exception:
            continue
        if (w, h) == (W, H):
            sized.append((0, f))
        elif abs(w - W) <= size_tol * W and abs(h - H) <= size_tol * H:
            sized.append((1, f))
    if not sized:
        info["status"] = "no_file_of_image_size"
        return None, info
    info["status"] = "not_binary"
    for rank, f in sorted(sized):
        with Image.open(f) as im:
            a = np.asarray(im.convert("L"))
        if not looks_binary(a):
            continue
        m = a > 127
        if not m.any():
            info.update(status="empty", mask_file=f)
            continue
        if rank == 1:
            m = np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize((W, H), Image.NEAREST)) > 127
        info.update(status="ok" if rank == 0 else "ok_resized", mask_file=f)
        return m, info
    return None, info


def breast_region(img: np.ndarray, threshold: int = 20) -> np.ndarray:
    """Breast pixels of a preprocessed (model-space) image: brighter than film background, largest connected
    region, holes filled. The lesion masks are OR-ed in afterwards by the caller."""
    m = ndimage.binary_opening(img > threshold, iterations=2)
    lab, n = ndimage.label(m)
    if n == 0:
        return img > 0
    sizes = ndimage.sum(m, lab, range(1, n + 1))
    return ndimage.binary_fill_holes(lab == int(np.argmax(sizes)) + 1)


# ------------------------------------------------------------------------------------------------ map geometry
def _interp_matrix(n_in: int, n_out: int) -> np.ndarray:
    x = np.clip((np.arange(n_out) + 0.5) * n_in / n_out - 0.5, 0, n_in - 1)
    i0 = np.floor(x).astype(int)
    i1 = np.minimum(i0 + 1, n_in - 1)
    w = x - i0
    M = np.zeros((n_out, n_in))
    np.add.at(M, (np.arange(n_out), i0), 1 - w)
    np.add.at(M, (np.arange(n_out), i1), w)
    return M


def upsample(m: np.ndarray, H: int, W: int) -> np.ndarray:
    """Bilinear, align_corners=False (same numbers as torch.nn.functional.interpolate)."""
    return _interp_matrix(m.shape[0], H) @ np.asarray(m, float) @ _interp_matrix(m.shape[1], W).T


def pool_cells(a: np.ndarray, h: int, w: int) -> np.ndarray:
    """Average over the ``h x w`` grid of equal cells (H and W must be multiples of h and w)."""
    H, W = a.shape
    return np.asarray(a, float).reshape(h, H // h, w, W // w).mean((1, 3))


def adaptive_pool(a: np.ndarray, oh: int, ow: int) -> np.ndarray:
    """torch.nn.AdaptiveAvgPool2d on a 2-D array (used to compare with the paper's 7 x 7 map)."""
    H, W = a.shape
    out = np.zeros((oh, ow))
    for i in range(oh):
        r0, r1 = (i * H) // oh, math.ceil((i + 1) * H / oh)
        for j in range(ow):
            c0, c1 = (j * W) // ow, math.ceil((j + 1) * W / ow)
            out[i, j] = a[r0:r1, c0:c1].mean()
    return out


class Geometry:
    """Everything about one test image that the metrics need, precomputed once and shared by all methods."""

    def __init__(self, lesion: np.ndarray, breast: np.ndarray, map_shape: tuple[int, int]):
        self.lesion = lesion.astype(bool)
        self.breast = breast.astype(bool) | self.lesion
        self.H, self.W = lesion.shape
        self.h, self.w = map_shape
        self.cell = self.H // self.h
        dist = ndimage.distance_transform_edt(~self.lesion) if self.lesion.any() else np.full(lesion.shape, np.inf)
        self.near = dist <= self.cell                         # within one cell of the lesion outline
        self.b_idx = np.flatnonzero(self.breast.ravel())
        self.les_b = self.lesion.ravel()[self.b_idx]
        self.near_b = self.near.ravel()[self.b_idx]
        self.area_fraction = float(self.les_b.mean()) if len(self.b_idx) else float("nan")
        cb = pool_cells(self.breast, self.h, self.w)
        self.cell_breast = cb >= 0.5
        self.cell_lesion_frac = pool_cells(self.lesion, self.h, self.w)
        self.cell_lesion = self.cell_lesion_frac > 0
        self.cell_near = ndimage.binary_dilation(self.cell_lesion, np.ones((3, 3), bool))
        self.cell_breast |= self.cell_lesion


def _pointing(values: np.ndarray, hit: np.ndarray) -> float:
    """Share of the maximum's location that falls on ``hit``. Ties are split evenly (expected score of a random
    tie-break), so a constant map scores exactly the hit-area fraction."""
    if len(values) == 0:
        return float("nan")
    mx = values.max()
    top = values >= mx - 1e-9 * max(abs(mx), 1.0)
    return float(hit[top].mean())


def _auc(values: np.ndarray, pos: np.ndarray) -> float:
    n1 = int(pos.sum()); n0 = len(pos) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(values)
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def localisation_metrics(cam: np.ndarray, g: Geometry, topk=(0.05, 0.10, 0.20)) -> dict:
    """Does a (non-negative) map point at the lesion? Everything is measured inside the breast.

    Pixel level (map upsampled bilinearly to the image):
      pg_px       pointing game: the map's maximum lies inside the lesion
      pg_px_tol   ... or within one cell (32 px) of it
      energy_px   share of the map's total mass (inside the breast) that falls on the lesion
      auc_px      AUC of map values, lesion pixels vs other breast pixels (0.5 = no information)
      iou_top{k}, dice_top{k}: the top k% of breast pixels by map value vs the lesion
    Cell level (the map's native grid, lesion = any lesion pixel in the cell):
      pg_cell, pg_cell_tol (adjacent cell counts), energy_cell (map-weighted lesion fraction), auc_cell
    Reference: area_fraction = lesion / breast pixels = what a map with no information scores on pg_px and energy_px.
    """
    cam = np.clip(np.asarray(cam, float), 0, None)
    up = upsample(cam, g.H, g.W).ravel()[g.b_idx]
    r = {"area_fraction": g.area_fraction,
         "pg_px": _pointing(up, g.les_b), "pg_px_tol": _pointing(up, g.near_b),
         "auc_px": _auc(up, g.les_b)}
    tot = up.sum()
    r["energy_px"] = float(up[g.les_b].sum() / tot) if tot > 0 else g.area_fraction
    n = len(up)
    order = np.argsort(-up, kind="stable")
    for k in topk:
        kk = max(1, int(round(k * n)))
        thr = up[order[kk - 1]]
        pred = up >= thr                                      # ties at the threshold are all included
        inter = float((pred & g.les_b).sum())
        union = float((pred | g.les_b).sum())
        tag = f"{int(round(k * 100))}"
        r[f"iou_top{tag}"] = inter / union if union else float("nan")
        r[f"dice_top{tag}"] = 2 * inter / (pred.sum() + g.les_b.sum()) if (pred.sum() + g.les_b.sum()) else float("nan")
    cb = g.cell_breast
    v = cam[cb]
    r["pg_cell"] = _pointing(v, g.cell_lesion[cb])
    r["pg_cell_tol"] = _pointing(v, g.cell_near[cb])
    r["auc_cell"] = _auc(v, g.cell_lesion[cb])
    r["energy_cell"] = float((v * g.cell_lesion_frac[cb]).sum() / v.sum()) if v.sum() > 0 \
        else float(g.cell_lesion_frac[cb].mean())
    return r


# ------------------------------------------------------------------------------------------------ paper statistics
def distribution_stats(a: np.ndarray, keep: np.ndarray | None = None, prefix: str = "") -> dict:
    """The paper's attention statistics, with the definitions it leaves out (computed on the raw map values v):

      gini        Gini coefficient of v (0 = perfectly even, (n-1)/n = all mass in one cell)
      entropy     Shannon entropy of p = v / sum(v), natural log (nats); maximum = ln(n)
      entropy_norm entropy / ln(n), in [0, 1], comparable across map sizes
      area80      smallest share of cells that together hold >= 80% of sum(v)
      peak        max(v) (meaningful for CBAM, whose values are sigmoid gates in [0, 1])
      n_cells     n
    ``keep`` restricts the statistics to some cells (e.g. the breast)."""
    v = np.asarray(a, float).ravel()
    if keep is not None:
        v = v[np.asarray(keep, bool).ravel()]
    v = np.clip(v, 0, None)
    n = len(v)
    out = {f"{prefix}n_cells": n, f"{prefix}peak": float(v.max()) if n else float("nan")}
    s = v.sum()
    if n < 2 or s <= 0:
        out.update({f"{prefix}{k}": float("nan") for k in ("gini", "entropy", "entropy_norm", "area80")})
        return out
    vs = np.sort(v)
    out[f"{prefix}gini"] = float((2 * np.sum(np.arange(1, n + 1) * vs)) / (n * s) - (n + 1) / n)
    p = v / s
    nz = p[p > 0]
    H = float(-(nz * np.log(nz)).sum())
    out[f"{prefix}entropy"] = H
    out[f"{prefix}entropy_norm"] = H / math.log(n)
    c = np.cumsum(np.sort(p)[::-1])
    out[f"{prefix}area80"] = float((np.searchsorted(c, 0.8 - 1e-12) + 1) / n)
    return out


# ------------------------------------------------------------------------------------------------ baselines
def baseline_maps(img: np.ndarray, breast: np.ndarray, map_shape: tuple[int, int], rng: np.random.Generator,
                  n_random: int = 5) -> dict:
    """Maps that know nothing about lesions, at the same resolution as the attention maps.

      uniform     constant (every breast pixel ties -> scores the lesion's area fraction)
      random_i    i.i.d. uniform noise per cell (n_random draws; metrics are averaged over draws)
      brightness  mean image intensity per cell ("look at the brightest tissue")
      centre      Gaussian centred on the breast's centre of mass, sd = 1/4 of the breast's height and width
    """
    h, w = map_shape
    out = {"uniform": np.ones(map_shape)}
    for i in range(n_random):
        out[f"random_{i}"] = rng.random(map_shape)
    out["brightness"] = pool_cells(img.astype(float) / 255.0, h, w)
    ys, xs = np.nonzero(breast)
    H, W = img.shape
    if len(ys):
        cy, cx = ys.mean(), xs.mean()
        sy, sx = max(np.ptp(ys), 1) / 4, max(np.ptp(xs), 1) / 4
    else:
        cy, cx, sy, sx = H / 2, W / 2, H / 4, W / 4
    yy, xx = np.mgrid[0:H, 0:W]
    out["centre"] = pool_cells(np.exp(-0.5 * (((yy - cy) / sy) ** 2 + ((xx - cx) / sx) ** 2)), h, w)
    return out


# ------------------------------------------------------------------------------------------------ statistics
class ClusterBootstrap:
    """Patient-clustered bootstrap with one fixed set of resamples, reused for every metric so that all CIs and
    paired differences are computed on the same resampled patient sets."""

    def __init__(self, groups, n: int = 2000, seed: int = 0):
        self.keys, self.inv = np.unique(np.asarray(groups).astype(str), return_inverse=True)
        G = len(self.keys)
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, G, (n, G))
        self.W = np.zeros((n, G))
        np.add.at(self.W, (np.repeat(np.arange(n), G), idx.ravel()), 1.0)

    def _boot_means(self, v, mask):
        ok = mask & ~np.isnan(v)
        sums = np.bincount(self.inv[ok], weights=v[ok], minlength=len(self.keys))
        cnts = np.bincount(self.inv[ok], minlength=len(self.keys)).astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            return (self.W @ sums) / (self.W @ cnts), (float(v[ok].mean()) if ok.any() else float("nan")), int(ok.sum())

    def mean(self, v, mask=None) -> dict:
        v = np.asarray(v, float)
        mask = np.ones(len(v), bool) if mask is None else np.asarray(mask, bool)
        b, m, n = self._boot_means(v, mask)
        lo, hi = np.nanpercentile(b, [2.5, 97.5]) if n else (float("nan"),) * 2
        return {"mean": m, "ci95": [float(lo), float(hi)], "n": n}

    def paired(self, a, b, mask=None) -> dict:
        """Mean of a - b over the same images; p = share of resamples with difference <= 0."""
        a, b = np.asarray(a, float), np.asarray(b, float)
        d = a - b
        mask = np.ones(len(d), bool) if mask is None else np.asarray(mask, bool)
        boot, m, n = self._boot_means(d, mask)
        lo, hi = np.nanpercentile(boot, [2.5, 97.5]) if n else (float("nan"),) * 2
        return {"diff": m, "ci95": [float(lo), float(hi)], "p_not_better": float(np.nanmean(boot <= 0)), "n": n}

    def group_diff(self, v, in_a, in_b) -> dict:
        """Mean(v | A) - mean(v | B) for two disjoint image groups (e.g. malignant vs benign), patient-clustered."""
        v = np.asarray(v, float)
        ba, ma, na = self._boot_means(v, np.asarray(in_a, bool))
        bb, mb, nb = self._boot_means(v, np.asarray(in_b, bool))
        lo, hi = np.nanpercentile(ba - bb, [2.5, 97.5])
        return {"diff": ma - mb, "ci95": [float(lo), float(hi)], "n_a": na, "n_b": nb}


def cohens_d(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    sp = math.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / sp) if sp > 0 else float("nan")


def correlation_ci(x, y, groups, n: int = 2000, seed: int = 0) -> dict:
    """Pearson and Spearman correlation with patient-clustered bootstrap CIs."""
    from scipy.stats import pearsonr, spearmanr
    x, y, g = np.asarray(x, float), np.asarray(y, float), np.asarray(groups).astype(str)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y, g = x[ok], y[ok], g[ok]
    units = [np.flatnonzero(g == u) for u in np.unique(g)]
    rng = np.random.default_rng(seed)
    bp, bs = [], []
    for _ in range(n):
        pick = np.concatenate([units[i] for i in rng.integers(0, len(units), len(units))])
        if np.std(x[pick]) == 0 or np.std(y[pick]) == 0:
            continue
        bp.append(pearsonr(x[pick], y[pick])[0]); bs.append(spearmanr(x[pick], y[pick])[0])
    return {"pearson": float(pearsonr(x, y)[0]), "pearson_ci95": [float(v) for v in np.percentile(bp, [2.5, 97.5])],
            "pearson_p": float(pearsonr(x, y)[1]),
            "spearman": float(spearmanr(x, y)[0]), "spearman_ci95": [float(v) for v in np.percentile(bs, [2.5, 97.5])],
            "n": int(len(x))}
