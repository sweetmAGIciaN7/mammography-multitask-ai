import sys
from pathlib import Path

import numpy as np
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
