"""CBIS-DDSM indexing (Phase 3).

CBIS-DDSM (Lee et al., Sci. Data 2017) is the curated DDSM subset with
**biopsy-confirmed pathology**, a BI-RADS breast-density grade and a patient id
for every abnormality. We use the JPEG release on Kaggle
(``awsaf49/cbis-ddsm-breast-cancer-image-dataset``), which looks like::

    csv/mass_case_description_{train,test}_set.csv   one row per abnormality
    csv/calc_case_description_{train,test}_set.csv
    csv/dicom_info.csv                               DICOM -> JPEG mapping
    jpeg/<SeriesInstanceUID>/1-1.jpg

How a *case row* is turned into a *training example*
---------------------------------------------------
* The unit is one **breast image** = (patient, side, view). CBIS-DDSM stores the
  same mammogram twice when it has both a mass and a calcification (once under
  ``Mass-...`` and once under ``Calc-...``). Those copies are merged, so no
  mammogram can sit in train and test at the same time.
* **Label** = malignant if *any* abnormality on the image is ``MALIGNANT``;
  ``BENIGN`` and ``BENIGN_WITHOUT_CALLBACK`` count as benign.
* **Density** = BI-RADS density 1-4 stored as 0-3 (-1 if missing). The column
  is ``breast_density`` in the mass files and ``breast density`` in the calc files.
* Every image in CBIS-DDSM contains an abnormality, so the task is
  "is this *suspicious* finding malignant?", not screening (no normal mammograms).
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

CASE_FILES = {
    ("mass", "train"): "mass_case_description_train_set.csv",
    ("mass", "test"): "mass_case_description_test_set.csv",
    ("calc", "train"): "calc_case_description_train_set.csv",
    ("calc", "test"): "calc_case_description_test_set.csv",
}
DENSITY_NAMES = ["BI-RADS A (fatty)", "BI-RADS B (scattered)", "BI-RADS C (heterogeneous)", "BI-RADS D (extremely dense)"]
_UID = re.compile(r"^\d+(\.\d+){4,}$")  # DICOM UID such as 1.3.6.1.4.1.9590.100.1.2.123...


def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [re.sub(r"[^0-9a-z]+", "_", str(c).strip().lower()).strip("_") for c in df.columns]
    return df


def locate_cbis(root: str | Path = "/kaggle/input") -> dict[str, Path]:
    """Find the ``csv`` folder (with dicom_info.csv + the 4 case files) and the ``jpeg`` folder."""
    root = Path(root)
    csv_dir = jpeg_dir = None
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        here = Path(dirpath)
        if csv_dir is None and "dicom_info.csv" in filenames and all(f in filenames for f in CASE_FILES.values()):
            csv_dir = here
        if jpeg_dir is None and here.name == "jpeg" and dirnames:
            jpeg_dir = here
        if csv_dir is not None and jpeg_dir is not None:
            break
    if csv_dir is None:
        raise FileNotFoundError(
            f"Could not find dicom_info.csv + the four *_case_description_*_set.csv files under {root}. "
            "Attach the Kaggle dataset 'CBIS-DDSM: Breast Cancer Image Dataset' (awsaf49).")
    if jpeg_dir is None:
        raise FileNotFoundError(f"Could not find the 'jpeg' image folder under {root}.")
    return {"csv": csv_dir, "jpeg": jpeg_dir}


def _jpeg_relative(image_path: str) -> str | None:
    """'CBIS-DDSM/jpeg/<uid>/1-1.jpg' -> '<uid>/1-1.jpg'."""
    parts = str(image_path).strip().replace("\\", "/").split("/")
    if "jpeg" in parts:
        rest = parts[parts.index("jpeg") + 1:]
        return "/".join(rest) if rest else None
    return "/".join(parts[-2:]) if len(parts) >= 2 else None


def full_mammogram_lookup(dicom_info: pd.DataFrame, jpeg_dir: str | Path) -> tuple[dict, dict]:
    """Two maps to the JPEG of each *full mammogram*: by SeriesInstanceUID and by DICOM PatientID
    (e.g. 'Mass-Training_P_00001_LEFT_CC')."""
    di = _norm_cols(dicom_info)
    desc = di.get("seriesdescription", pd.Series("", index=di.index)).fillna("").astype(str).str.lower()
    full = di[desc.str.contains("full mammogram")]
    by_series, by_case = {}, {}
    jpeg_dir = Path(jpeg_dir)
    for _, r in full.iterrows():
        rel = _jpeg_relative(r.get("image_path", ""))
        if not rel:
            continue
        path = str(jpeg_dir / rel)
        series = str(r.get("seriesinstanceuid", "") or "").strip() or rel.split("/")[0]
        by_series.setdefault(series, path)
        by_series.setdefault(rel.split("/")[0], path)
        case = str(r.get("patientid", "") or "").strip()
        if case:
            by_case.setdefault(case, path)
    return by_series, by_case


def read_cases(csv_dir: str | Path) -> pd.DataFrame:
    """All four case-description files stacked, one row per abnormality, with harmonised columns."""
    frames = []
    for (abn, split), fname in CASE_FILES.items():
        d = _norm_cols(pd.read_csv(Path(csv_dir) / fname))
        if "breast_density" not in d and "breast_density_1" in d:  # defensive: odd header variants
            d = d.rename(columns={"breast_density_1": "breast_density"})
        d["abn_type"], d["official_split"] = abn, split
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    need = ["patient_id", "left_or_right_breast", "image_view", "pathology", "breast_density", "image_file_path"]
    missing = [c for c in need if c not in df]
    if missing:
        raise KeyError(f"CBIS-DDSM case files lack columns {missing}; found {list(df.columns)}")
    out = pd.DataFrame({
        "patient": df["patient_id"].astype(str).str.strip(),
        "side": df["left_or_right_breast"].astype(str).str.strip().str.upper(),
        "view": df["image_view"].astype(str).str.strip().str.upper(),
        "pathology_raw": df["pathology"].astype(str).str.strip().str.upper(),
        "density_raw": pd.to_numeric(df["breast_density"], errors="coerce"),
        "image_file_path": df["image_file_path"].astype(str).str.strip().str.replace("\\", "/", regex=False),
        "abn_type": df["abn_type"],
        "official_split": df["official_split"],
    })
    known = {"MALIGNANT", "BENIGN", "BENIGN_WITHOUT_CALLBACK"}
    bad = set(out["pathology_raw"]) - known
    if bad:
        raise ValueError(f"Unexpected pathology values {bad}")
    out["malignant"] = (out["pathology_raw"] == "MALIGNANT").astype(int)
    parts = out["image_file_path"].str.split("/")
    out["case_key"] = parts.str[0]
    out["series_uid"] = parts.apply(lambda p: next((x for x in reversed(p[:-1]) if _UID.match(x)), None)
                                    if len(p) >= 3 else None)
    return out


def _mode_density(d: pd.Series) -> int:
    d = d[(d >= 1) & (d <= 4)]
    return int(d.mode().min()) - 1 if len(d) else -1


def index_cbis(csv_dir: str | Path, jpeg_dir: str | Path) -> tuple[pd.DataFrame, dict]:
    """One row per breast image: image_id, patient, side, view, path, pathology (0/1), density (0-3/-1),
    n_mass, n_calc, official_split ('train' / 'test' at *patient* level). Also returns a stats dict."""
    cases = read_cases(csv_dir)
    by_series, by_case = full_mammogram_lookup(pd.read_csv(Path(csv_dir) / "dicom_info.csv"), jpeg_dir)
    cases["path"] = [by_series.get(s) or by_case.get(c) for s, c in zip(cases["series_uid"], cases["case_key"])]

    stats = {"abnormalities": int(len(cases)),
             "abnormalities_without_jpeg": int(cases["path"].isna().sum())}
    cases = cases[cases["path"].notna()]
    if cases.empty:
        raise RuntimeError("No case row could be matched to a full-mammogram JPEG. The dicom_info.csv / image "
                           "path layout differs from what mammo.cbis expects.")

    cases = cases.assign(image_id=cases["patient"] + "_" + cases["side"] + "_" + cases["view"])
    # Prefer the mass copy of a mammogram that exists twice (mass + calc); both are the same film.
    cases = cases.sort_values(["image_id", "abn_type", "path"], ascending=[True, False, True])
    g = cases.groupby("image_id", sort=True)
    img = pd.DataFrame({
        "patient": g["patient"].first(),
        "side": g["side"].first(),
        "view": g["view"].first(),
        "path": g["path"].first(),
        "pathology": g["malignant"].max(),
        "density": g["density_raw"].apply(_mode_density),
        "n_mass": g["abn_type"].apply(lambda s: int((s == "mass").sum())),
        "n_calc": g["abn_type"].apply(lambda s: int((s == "calc").sum())),
        "n_files": g["path"].nunique(),
    }).reset_index()
    density_conflict = g["density_raw"].apply(lambda d: d[(d >= 1) & (d <= 4)].nunique() > 1)

    # Official split, decided per *patient*: anyone who appears in a training file is a training patient.
    train_patients = set(cases.loc[cases["official_split"] == "train", "patient"])
    test_patients = set(cases.loc[cases["official_split"] == "test", "patient"])
    img["official_split"] = np.where(img["patient"].isin(train_patients), "train", "test")

    stats.update({
        "images": int(len(img)),
        "patients": int(img["patient"].nunique()),
        "malignant_images": int(img["pathology"].sum()),
        "density_known": int((img["density"] >= 0).sum()),
        "density_counts": {DENSITY_NAMES[k]: int(v) for k, v in img.loc[img["density"] >= 0, "density"]
                           .value_counts().sort_index().items()},
        "density_conflicts": int(density_conflict.sum()),
        "images_stored_twice_mass_and_calc": int((img["n_files"] > 1).sum()),
        "patients_in_official_train_and_test": int(len(train_patients & test_patients)),
        "official_test_images_after_patient_dedup": int((img["official_split"] == "test").sum()),
    })
    return img.drop(columns="n_files"), stats


def describe_cbis(stats: dict) -> str:
    s = stats
    return "\n".join([
        f"abnormalities (case rows) : {s['abnormalities']}  (no JPEG found: {s['abnormalities_without_jpeg']})",
        f"breast images             : {s['images']}  (malignant {s['malignant_images']}, "
        f"{s['malignant_images'] / max(s['images'], 1):.1%})",
        f"patients                  : {s['patients']}",
        f"density known             : {s['density_known']}  {s['density_counts']}",
        f"density conflicts         : {s['density_conflicts']} images",
        f"stored twice (mass+calc)  : {s['images_stored_twice_mass_and_calc']} images, merged",
        f"patients in both official train and test: {s['patients_in_official_train_and_test']} "
        f"-> official test keeps {s['official_test_images_after_patient_dedup']} images",
    ])
