"""INbreast as an *external* test set (Phase 4b).

INbreast (Moreira et al., Acad. Radiol. 2012) is a different world from CBIS-DDSM:

=====================  ==============================  ===============================
                       CBIS-DDSM (training)            INbreast (external test)
=====================  ==============================  ===============================
Acquisition            scanned **film** (1990s, USA)   **full-field digital** (Portugal)
Images                 2,802 with a finding            410, incl. normal mammograms
Density label          BI-RADS density 1-4             ACR density 1-4
Malignancy label       biopsy-confirmed                BI-RADS assessment only
=====================  ==============================  ===============================

A model that scores well on CBIS-DDSM and holds up on INbreast has learned something about breasts, not
about one scanner. Density is the main external test because both datasets label it on the same 4-grade scale.
Malignancy can only be checked against a *proxy* (BI-RADS 4-6 = suspicious vs 1-3), because INbreast
has no pathology for most images.

The DICOMs are 14-bit digital images. They are mapped to 8 bits with a percentile window inside the breast
and then go through exactly the same crop / orient / resize steps as the CBIS-DDSM films
(:mod:`mammo.preprocess`).
"""
from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .data import find_dir, index_inbreast
from .preprocess import breast_bbox, fit_pad, orient_left


def birads_number(s) -> float:
    """'4a' -> 4, '6' -> 6, '' -> nan."""
    s = str(s).strip()
    return float(s[0]) if s[:1].isdigit() else float("nan")


def index_inbreast_full(release_dir: str | Path) -> pd.DataFrame:
    """One row per INbreast image: image_id, patient, side, view, path, density (0-3/-1), birads, birads_num,
    suspicious (1 = BI-RADS 4-6, 0 = BI-RADS 1-3, -1 = unknown)."""
    release = Path(release_dir)
    df = index_inbreast(release)
    names = {f.split("_")[0]: f for f in os.listdir(release / "AllDICOMs") if f.lower().endswith(".dcm")}
    df["path"] = [str(release / "AllDICOMs" / names[i]) for i in df["image_id"]]
    df["birads"] = df["birads"].fillna("").astype(str)
    df["birads_num"] = df["birads"].map(birads_number)
    df["suspicious"] = np.where(df["birads_num"] >= 4, 1, np.where(df["birads_num"] >= 1, 0, -1)).astype(int)
    df["view"] = df["view"].str.upper()
    return df.sort_values("image_id").reset_index(drop=True)


def locate_inbreast(root: str | Path = "/kaggle/input") -> Path:
    return find_dir(root, "INbreast Release 1.0", must_contain="AllDICOMs")


def to_uint8(a: np.ndarray, invert: bool = False, lo_pct: float = 0.5, hi_pct: float = 99.5) -> np.ndarray:
    """Map a 12-16 bit mammogram to 0-255: window from the background level to a high percentile of the tissue.

    The window is computed on the pixels above the background (the breast), so a large black background
    doesn't compress the tissue contrast."""
    a = a.astype(np.float32)
    if invert:                                      # MONOCHROME1: high values are dark
        a = a.max() - a
    bg = np.percentile(a, 1)
    tissue = a[a > bg + 0.02 * (a.max() - bg)]
    if tissue.size < 100:
        tissue = a.ravel()
    lo, hi = np.percentile(tissue, [lo_pct, hi_pct])
    lo = min(lo, bg + 0.5 * (lo - bg))              # keep the skin line, which is darker than most tissue
    out = np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)
    out[a <= bg] = 0
    return (out * 255).round().astype(np.uint8)


def read_dicom(path: str | Path) -> np.ndarray:
    import pydicom
    ds = pydicom.dcmread(str(path))
    a = ds.pixel_array
    return to_uint8(a, invert=str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1")


def load_inbreast_image(path: str | Path, height: int = 640, width: int = 384) -> np.ndarray:
    a = read_dicom(path)
    t, b, l, r = breast_bbox(a)
    a = orient_left(a[t:b, l:r])
    return fit_pad(np.ascontiguousarray(a), height, width)


def cached_inbreast(paths, height: int, width: int, cache_dir: str | Path, workers: int = 4, log=print) -> np.ndarray:
    paths = [str(p) for p in paths]
    h = hashlib.sha1(f"inbreast|{height}x{width}|v1|".encode())
    for p in paths:
        h.update(p.encode()); h.update(b"\0")
    cache_dir = Path(cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)
    f = cache_dir / f"inbreast_{height}x{width}_{h.hexdigest()[:12]}.npy"
    if f.exists():
        arr = np.load(f)
        if arr.shape == (len(paths), height, width):
            log(f"using cached images {f.name}")
            return arr
    arr = np.zeros((len(paths), height, width), np.uint8)

    def work(i):
        arr[i] = load_inbreast_image(paths[i], height, width)

    with ThreadPoolExecutor(workers) as ex:
        for n, _ in enumerate(ex.map(work, range(len(paths))), 1):
            if n % 100 == 0 or n == len(paths):
                log(f"  preprocessed {n}/{len(paths)}")
    np.save(f, arr)
    json.dump({"n": len(paths), "height": height, "width": width}, open(f.with_suffix(".json"), "w"))
    return arr


def describe_inbreast(df: pd.DataFrame) -> str:
    dens = df.loc[df["density"] >= 0, "density"].value_counts().sort_index()
    return "\n".join([
        f"images          : {len(df)}  (CC {int((df['view'] == 'CC').sum())}, MLO {int((df['view'] == 'MLO').sum())})",
        f"patients        : {df['patient'].nunique()}",
        f"density known   : {int((df['density'] >= 0).sum())}  " + str({f'ACR {k + 1}': int(v) for k, v in dens.items()}),
        f"BI-RADS         : " + str(df["birads_num"].value_counts(dropna=False).sort_index().to_dict()),
        f"suspicious (4-6): {int((df['suspicious'] == 1).sum())} vs 1-3: {int((df['suspicious'] == 0).sum())}",
    ])
