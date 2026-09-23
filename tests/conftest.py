import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def make_fake_kaggle(root: Path, n_patients: int = 6, images_per_patient: int = 2, copies: int = 4) -> Path:
    """Mimic the Kaggle layout of the two datasets with tiny random PNGs."""
    masses = (root / "datasets/tommyngx/breastcancermasses/Dataset of Mammography with Benign Malignant Breast Masses"
              / "Dataset of Mammography with Benign Malignant Breast Masses/INbreast Dataset")
    release = root / "datasets/ramanathansp20/inbreast-dataset/INbreast Release 1.0"
    (release / "AllDICOMs").mkdir(parents=True)
    for sub in ("Benign Masses", "Malignant Masses"):
        (masses / sub).mkdir(parents=True)
    rng = np.random.default_rng(0)
    csv = ["Patient ID;Patient age;Laterality;View;Acquisition date;File Name;ACR;Bi-Rads"]
    img_id = 20000000
    for p in range(n_patients):
        patient = f"{p:016x}"
        label = p % 2  # patient-level label keeps every group single-class
        for i in range(images_per_patient):
            img_id += 1
            side, view = ("L", "CC") if i % 2 else ("R", "MLO")
            (release / "AllDICOMs" / f"{img_id}_{patient}_MG_{side}_{view}_ANON.dcm").write_bytes(b"")
            acr = "" if (p == 0 and i == 0) else str(1 + (p + i) % 4)  # one missing density
            csv.append(f"removed;removed;{side};{view};201001;{img_id};{acr};{4 if label else 2}")
            folder = masses / ("Malignant Masses" if label else "Benign Masses")
            for c in range(1, copies + 1):
                arr = rng.integers(0, 255, (32, 32), dtype=np.uint8)
                Image.fromarray(arr).save(folder / f"{img_id} ({c}).png")
    (release / "INbreast.csv").write_text("\n".join(csv))
    (masses / "Benign Masses" / "Thumbs.db").write_bytes(b"")  # junk file must be ignored
    return root


@pytest.fixture
def fake_kaggle(tmp_path):
    return make_fake_kaggle(tmp_path)


def _fake_mammogram(breast_on_right: bool, h: int = 500, w: int = 300, seed: int = 0) -> np.ndarray:
    """Black film, a bright half-disc 'breast' against one edge, a white scanner strip on top."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    cx = w - 1 if breast_on_right else 0
    r = ((yy - h / 2) / (0.42 * h)) ** 2 + ((xx - cx) / (0.8 * w)) ** 2
    img = np.where(r < 1, 120 + 100 * (1 - r), 0) + rng.normal(0, 4, (h, w))
    img[:6, :] = 255  # scanner border strip, disconnected from the breast
    return np.clip(img, 0, 255).astype(np.uint8)


def make_fake_cbis(root: Path) -> Path:
    """Mimic the awsaf49 Kaggle layout: csv/ (4 case files + dicom_info.csv) and jpeg/<series uid>/1-1.jpg."""
    base = root / "datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset"
    csv_dir, jpeg_dir = base / "csv", base / "jpeg"
    csv_dir.mkdir(parents=True); jpeg_dir.mkdir(parents=True)
    uid = iter(range(1000, 9999))
    dicom_rows, cases = [], {k: [] for k in ["mass_train", "mass_test", "calc_train", "calc_test"]}

    def add(kind, split, patient, side, view, pathology, density, series_in_csv=None, jpeg_dir_name=None):
        prefix = {"mass": "Mass", "calc": "Calc"}[kind] + ("-Training" if split == "train" else "-Test")
        case_key = f"{prefix}_{patient}_{side}_{view}"
        study, series = f"1.3.6.1.4.1.9590.100.1.2.{next(uid)}", f"1.3.6.1.4.1.9590.100.1.2.{next(uid)}"
        series_in_csv = series_in_csv or series
        folder = jpeg_dir_name or series
        (jpeg_dir / folder).mkdir(exist_ok=True)
        Image.fromarray(_fake_mammogram(side == "RIGHT", seed=len(dicom_rows))).save(jpeg_dir / folder / "1-1.jpg")
        dicom_rows.append({"file_path": f"CBIS-DDSM/dicom/{folder}/1-1.dcm", "image_path": f"CBIS-DDSM/jpeg/{folder}/1-1.jpg",
                           "PatientID": case_key, "SeriesDescription": "full mammogram images",
                           "SeriesInstanceUID": folder})
        # a cropped-lesion image of the same case, which must be ignored
        crop = f"1.3.6.1.4.1.9590.100.1.2.{next(uid)}"
        (jpeg_dir / crop).mkdir()
        Image.fromarray(np.full((50, 50), 200, np.uint8)).save(jpeg_dir / crop / "1-1.jpg")
        dicom_rows.append({"file_path": "x", "image_path": f"CBIS-DDSM/jpeg/{crop}/1-1.jpg",
                           "PatientID": case_key + "_1", "SeriesDescription": "cropped images",
                           "SeriesInstanceUID": crop})
        cases[f"{kind}_{split}"].append({
            "patient_id": patient, "density": density, "left or right breast": side, "image view": view,
            "abnormality id": 1, "abnormality type": kind, "pathology": pathology,
            "image file path": f"{case_key}/{study}/{series_in_csv}/000000.dcm",
            "cropped image file path": f"{case_key}_1/{study}/{crop}/000000.dcm",
            "ROI mask file path": f"{case_key}_1/{study}/{crop}/000001.dcm\n"})

    for p in range(1, 13):
        pid = f"P_{p:05d}"
        split = "test" if p > 10 else "train"
        path = "MALIGNANT" if p % 2 else ("BENIGN_WITHOUT_CALLBACK" if p % 4 == 0 else "BENIGN")
        add("mass", split, pid, "LEFT", "CC", path, 1 + p % 4)
        add("calc", split, pid, "RIGHT", "MLO", "BENIGN", 1 + p % 4)
    # P_00002 LEFT CC also stored as a calcification case that is malignant -> merged image is malignant
    add("calc", "train", "P_00002", "LEFT", "CC", "MALIGNANT", 3)
    # P_00003 appears in the official test set too -> counted as a training patient
    add("mass", "test", "P_00003", "RIGHT", "CC", "BENIGN", 0)  # density 0 = invalid -> missing
    # a case whose JPEG folder is not its series UID: must be found through its DICOM PatientID
    add("mass", "train", "P_00004", "RIGHT", "CC", "MALIGNANT", 2, jpeg_dir_name="1.3.6.1.4.1.9590.100.1.2.777777")

    for key, rows in cases.items():
        d = pd.DataFrame(rows)
        # the real files disagree on this header: 'breast_density' (mass) vs 'breast density' (calc)
        d = d.rename(columns={"density": "breast_density" if key.startswith("mass") else "breast density"})
        kind, split = key.split("_")
        d.to_csv(csv_dir / f"{kind}_case_description_{split}_set.csv", index=False)
    pd.DataFrame(dicom_rows).to_csv(csv_dir / "dicom_info.csv", index=False)
    return root


@pytest.fixture
def fake_cbis(tmp_path):
    return make_fake_cbis(tmp_path / "input")
