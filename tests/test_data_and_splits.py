import numpy as np
import pytest

from mammo.data import build_masses_table, locate_kaggle_inputs, parse_dicom_filename, parse_mass_filename
from mammo.splits import leakage_report, make_folds


def test_filename_parsers():
    assert parse_mass_filename("20586934 (17).png") == ("20586934", 17)
    assert parse_mass_filename("20586934.png") == ("20586934", 0)
    assert parse_mass_filename("Thumbs.db") is None
    d = parse_dicom_filename("20586908_6c613a14b80a8591_MG_R_CC_ANON.dcm")
    assert d == {"image_id": "20586908", "patient": "6c613a14b80a8591", "side": "R", "view": "CC"}
    assert parse_dicom_filename("20586960_6c613a14b80a8591_MG_R_ML_ANON.dcm")["view"] == "ML"


def test_table_joins_patient_and_density(fake_kaggle):
    paths = locate_kaggle_inputs(str(fake_kaggle))
    df = build_masses_table(paths["masses_inbreast"], paths["inbreast_release"])
    assert len(df) == 6 * 2 * 4
    assert df["image_id"].nunique() == 12
    assert df["patient"].nunique() == 6
    assert not df["patient"].str.startswith("unknown").any()
    assert (df["density"] == -1).sum() == 4          # the one image with missing ACR, all its copies
    assert set(df["density"]) <= {-1, 0, 1, 2, 3}


@pytest.mark.parametrize("protocol,group", [("image", "image_id"), ("patient", "patient")])
def test_grouped_protocols_never_leak(fake_kaggle, protocol, group):
    paths = locate_kaggle_inputs(str(fake_kaggle))
    df = build_masses_table(paths["masses_inbreast"], paths["inbreast_release"])
    folds = make_folds(df, protocol, n_splits=3, seed=0)
    seen_test = np.concatenate([te for _, te in folds])
    assert sorted(seen_test) == list(range(len(df)))  # every file tested exactly once
    for tr, te in folds:
        assert not set(df.iloc[tr][group]) & set(df.iloc[te][group])
        rep = leakage_report(df, tr, te)
        assert rep["test_files_whose_original_is_in_train"] == 0
        if protocol == "patient":
            assert rep["test_files_whose_patient_is_in_train"] == 0


def test_random_protocol_does_leak(fake_kaggle):
    paths = locate_kaggle_inputs(str(fake_kaggle))
    df = build_masses_table(paths["masses_inbreast"], paths["inbreast_release"])
    rep = [leakage_report(df, tr, te)["test_files_whose_original_is_in_train"]
           for tr, te in make_folds(df, "random", n_splits=3, seed=0)]
    assert np.mean(rep) > 0.9  # with 4 copies per image, nearly every test file has a sibling in train
