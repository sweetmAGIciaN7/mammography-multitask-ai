"""Cross-validation splits.

The *only* thing that differs between the three protocols of the leakage
experiment is how files are assigned to folds:

* ``random``  - the paper's protocol: augmented files are split independently,
                so rotated/flipped copies of one mammogram land in train AND test.
* ``image``   - all copies of one original image stay in the same fold.
* ``patient`` - all images of one patient (both breasts, both views) stay together.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

PROTOCOLS = {
    "random": "Paper protocol: split after augmentation",
    "image": "Grouped by original image",
    "patient": "Grouped by patient",
}


def make_folds(df: pd.DataFrame, protocol: str, n_splits: int = 5, seed: int = 42,
               label_col: str = "pathology") -> list[tuple[np.ndarray, np.ndarray]]:
    y = df[label_col].to_numpy()
    idx = np.arange(len(df))
    if protocol == "random":
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return [(tr, te) for tr, te in cv.split(idx, y)]
    group_col = {"image": "image_id", "patient": "patient"}.get(protocol)
    if group_col is None:
        raise ValueError(f"Unknown protocol {protocol!r}; choose from {list(PROTOCOLS)}")
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [(tr, te) for tr, te in cv.split(idx, y, groups=df[group_col].to_numpy())]


def leakage_report(df: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray) -> dict:
    """How much of the test set has a 'sibling' in the training set?"""
    tr, te = df.iloc[train_idx], df.iloc[test_idx]
    img_leak = te["image_id"].isin(set(tr["image_id"])).mean()
    pat_leak = te["patient"].isin(set(tr["patient"])).mean()
    return {
        "test_files_whose_original_is_in_train": float(img_leak),
        "test_files_whose_patient_is_in_train": float(pat_leak),
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
        "n_test_originals": int(te["image_id"].nunique()),
    }
