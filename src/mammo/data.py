"""Dataset indexing.

Two sources are used:

* **Huang & Lin (2020) "Dataset of breast mammography images with masses"** -
  106 INbreast mass images, each pre-augmented into ~72 copies (7,632 PNGs).
  Files are named ``<INbreast image id> (<copy number>).png``, e.g.
  ``20586934 (17).png``, so every copy can be traced back to its original.
  This is (very likely) the data behind Esen et al. (IEEE Access, 2025).

* **INbreast Release 1.0** - original DICOMs named
  ``<image id>_<patient hash>_MG_<side>_<view>_ANON.dcm`` plus ``INbreast.csv``
  with ACR breast-density labels. We use it to attach a *patient* identifier
  and a *density* label to every augmented copy.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd

# ``20586934 (17).png`` -> id=20586934, copy=17 ; ``20586934.png`` -> copy=0
_MASS_FILE = re.compile(r"^(?P<image_id>\d+)\s*(?:\((?P<copy>\d+)\))?\.(?:png|jpe?g)$", re.I)
_DICOM_FILE = re.compile(r"^(?P<image_id>\d+)_(?P<patient>[0-9a-f]+)_MG_(?P<side>[LR])_(?P<view>\w+?)_ANON\.dcm$", re.I)

PATHOLOGY_DIRS = {"Benign Masses": 0, "Malignant Masses": 1}
DENSITY_NAMES = ["ACR 1 (fatty)", "ACR 2 (scattered)", "ACR 3 (heterogeneous)", "ACR 4 (extremely dense)"]


def find_dir(root: str | Path, name: str, must_contain: str | None = None) -> Path:
    """Return the first directory called ``name`` under ``root`` (breadth-first)."""
    root = Path(root)
    for dirpath, dirnames, _ in os.walk(root):
        dirnames.sort()
        if Path(dirpath).name == name and (must_contain is None or (Path(dirpath) / must_contain).exists()):
            return Path(dirpath)
    raise FileNotFoundError(f"Could not find a folder named {name!r} under {root}. Is the dataset attached?")


def parse_mass_filename(fname: str) -> tuple[str, int] | None:
    m = _MASS_FILE.match(fname.strip())
    if not m:
        return None
    return m["image_id"], int(m["copy"] or 0)


def parse_dicom_filename(fname: str) -> dict | None:
    m = _DICOM_FILE.match(fname.strip())
    return m.groupdict() if m else None


def index_masses(masses_inbreast_dir: str | Path) -> pd.DataFrame:
    """One row per augmented PNG: path, image_id, copy, pathology (0 benign / 1 malignant)."""
    rows = []
    for sub, label in PATHOLOGY_DIRS.items():
        folder = Path(masses_inbreast_dir) / sub
        if not folder.is_dir():
            raise FileNotFoundError(f"Expected sub-folder {folder}")
        for f in sorted(os.listdir(folder)):
            parsed = parse_mass_filename(f)
            if parsed is None:
                continue
            rows.append({"path": str(folder / f), "image_id": parsed[0], "copy": parsed[1], "pathology": label})
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No mass images found in {masses_inbreast_dir}")
    # An original image must never carry two different labels.
    conflicts = df.groupby("image_id")["pathology"].nunique()
    if (conflicts > 1).any():
        raise RuntimeError(f"Images with conflicting labels: {list(conflicts[conflicts > 1].index)[:5]}")
    return df


def index_inbreast(inbreast_release_dir: str | Path) -> pd.DataFrame:
    """One row per original INbreast image: image_id, patient, side, view, density (0-3 or -1), birads."""
    release = Path(inbreast_release_dir)
    dicoms = []
    for f in os.listdir(release / "AllDICOMs"):
        d = parse_dicom_filename(f)
        if d:
            dicoms.append(d)
    dcm = pd.DataFrame(dicoms)

    meta = pd.read_csv(release / "INbreast.csv", sep=";")
    meta.columns = [c.strip() for c in meta.columns]
    meta = meta.rename(columns={"File Name": "image_id", "ACR": "acr", "Bi-Rads": "birads"})
    meta["image_id"] = meta["image_id"].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
    acr = pd.to_numeric(meta["acr"], errors="coerce")
    meta["density"] = acr.where(acr.between(1, 4)).sub(1).fillna(-1).astype(int)
    meta["birads"] = meta["birads"].astype(str).str.strip()

    out = dcm.merge(meta[["image_id", "density", "birads"]], on="image_id", how="left")
    out["density"] = out["density"].fillna(-1).astype(int)
    return out


def build_masses_table(masses_inbreast_dir: str | Path, inbreast_release_dir: str | Path) -> pd.DataFrame:
    """Augmented mass images joined with patient id and density label."""
    masses = index_masses(masses_inbreast_dir)
    inb = index_inbreast(inbreast_release_dir)
    df = masses.merge(inb[["image_id", "patient", "density", "birads"]], on="image_id", how="left")
    missing = df["patient"].isna()
    if missing.any():
        # Keep the image but make it its own "patient" so it can never leak across folds silently.
        df.loc[missing, "patient"] = "unknown_" + df.loc[missing, "image_id"]
        df.loc[missing, "density"] = -1
    df["density"] = df["density"].astype(int)
    return df.reset_index(drop=True)


def locate_kaggle_inputs(root: str = "/kaggle/input") -> dict[str, Path]:
    """Find the two dataset folders wherever Kaggle mounted them."""
    masses_inbreast = find_dir(root, "INbreast Dataset", must_contain="Benign Masses")
    release = find_dir(root, "INbreast Release 1.0", must_contain="AllDICOMs")
    return {"masses_inbreast": masses_inbreast, "inbreast_release": release}


def describe(df: pd.DataFrame) -> str:
    n_img = df["image_id"].nunique()
    n_pat = df["patient"].nunique()
    per_img = df.groupby("image_id").size()
    lines = [
        f"augmented files       : {len(df)}",
        f"original images       : {n_img}  (benign {df.groupby('image_id')['pathology'].first().eq(0).sum()}, "
        f"malignant {df.groupby('image_id')['pathology'].first().eq(1).sum()})",
        f"patients              : {n_pat}",
        f"copies per original   : min {per_img.min()}, median {int(per_img.median())}, max {per_img.max()}",
        f"density known (files) : {(df['density'] >= 0).mean():.1%}",
    ]
    return "\n".join(lines)
