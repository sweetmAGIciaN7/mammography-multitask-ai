"""Phase 6: demo helpers (numpy parts locally, torch parts on Kaggle)."""
import json
import math

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from mammo import demo as D
from mammo.experiments import demo_export as E


def test_calibration_maths():
    assert D.calibrated_probability(0.3, None) == 0.3
    assert D.calibrated_probability(0.3, {"a": 1.0, "b": 0.0}) == pytest.approx(0.3)
    # a < 1 pulls probabilities towards 0.5, b < 0 lowers them (the Phase 4a direction)
    p = D.calibrated_probability(0.9, {"a": 0.62, "b": -0.48})
    assert p == pytest.approx(1 / (1 + math.exp(-(0.62 * math.log(9) - 0.48))))
    assert 0.5 < p < 0.9
    assert 0 < D.calibrated_probability(1.0, {"a": 0.62, "b": -0.48}) < 1  # no overflow at the extremes


def test_density_labels_and_summary():
    lab = D.density_labels([0.1, 0.6, 0.2, 0.1])
    assert list(lab) == list(D.DENSITY_NAMES) and lab[D.DENSITY_NAMES[1]] == 0.6
    txt = D.summary_markdown({"p_malignant": 0.42, "p_malignant_raw": 0.55, "p_density": [0.1, 0.6, 0.2, 0.1]})
    assert "42%" in txt and "55%" in txt and D.DENSITY_NAMES[1] in txt and D.DISCLAIMER in txt


def test_render_panels_fake_result():
    from conftest import _fake_mammogram
    from mammo.preprocess import fit_pad
    img = fit_pad(_fake_mammogram(False, lesion=True), 640, 384)
    rng = np.random.default_rng(0)
    res = {"image": img, "gradcam": rng.random((20, 12)), "cbam": 0.8 + 0.2 * rng.random((20, 12))}
    out = D.render_panels(res, dpi=40)
    assert out.ndim == 3 and out.shape[2] == 3 and out.dtype == np.uint8
    res["cbam"] = None  # model without attention: two panels, narrower
    assert D.render_panels(res, dpi=40).shape[1] < out.shape[1]
    res["gradcam"] = np.zeros((20, 12))  # all-zero Grad-CAM must not crash
    D.render_panels(res, dpi=40)


def test_pick_examples_is_stratified_and_deterministic():
    test = pd.DataFrame({"image_id": [f"i{k}" for k in range(40)], "pathology": [k % 3 == 0 for k in range(40)]})
    test["pathology"] = test["pathology"].astype(int)
    a, b = E.pick_examples(test, 3, 0), E.pick_examples(test, 3, 0)
    assert list(a["image_id"]) == list(b["image_id"])
    assert a["pathology"].sum() == 3 and len(a) == 6 and a["image_id"].is_unique
    assert list(E.pick_examples(test, 3, 1)["image_id"]) != list(a["image_id"])


def test_density_letters_follow_cbis_coding():
    """CBIS-DDSM density 1-4 is stored as 0-3 (see mammo.cbis); the example table must use the same mapping."""
    assert [E.DENS[k] for k in range(4)] == ["A", "B", "C", "D"]


def test_save_example_is_lossless(tmp_path):
    from conftest import _fake_mammogram
    from mammo.preprocess import load_mammogram
    src = tmp_path / "m.jpg"
    Image.fromarray(_fake_mammogram(False, h=2000, w=1200, lesion=True)).save(src, quality=90)
    E.save_example(str(src), tmp_path / "m.png", 640, 384)
    assert np.array_equal(load_mammogram(src), load_mammogram(tmp_path / "m.png"))


def test_streamlit_app_files():
    demo = E.ROOT / "demo"
    app = (demo / "streamlit_app.py").read_text()
    assert "load_demo_model" in app and "st.cache_resource" in app and "DISCLAIMER" in app
    req = (demo / "requirements.txt").read_text()
    assert "streamlit" in req and "download.pytorch.org/whl/cpu" in req


def test_repo_inputs_exist():
    """build reads these from the repo; the Phase 4a transfer parameters must be there."""
    root = E.ROOT
    cal = json.load(open(root / "results/calibration/summary.json"))["official_transfer"]["platt"]
    assert set(cal) == {"a", "b"}
    t = pd.read_csv(root / "results/attention/test_images.csv")
    p = pd.read_csv(root / "results/attention/predictions.csv")
    assert {"path", "pathology", "density", "lesion_type", "image_id"} <= set(t.columns)
    assert "p_malignant_mt_cbam" in p.columns and set(t["image_id"]) >= set(p["image_id"])
    assert set(t["pathology"]) == {0, 1}


# ------------------------------------------------------------------ torch parts (run on Kaggle)
def _fake_checkpoint(path, h=160, w=96):
    torch = pytest.importorskip("torch")
    from mammo.train import TrainConfig, build_model
    torch.manual_seed(0)
    cfg = TrainConfig(pretrained=False, tasks=("pathology", "density"))
    m = build_model(cfg).eval()
    torch.save({"state_dict": m.state_dict(), "config": cfg.to_dict(), "height": h, "width": w}, path)
    return m


def test_export_load_predict_roundtrip(tmp_path):
    torch = pytest.importorskip("torch")
    from conftest import _fake_mammogram
    from mammo.preprocess import load_mammogram
    from mammo.train import _Prep
    ref = _fake_checkpoint(tmp_path / "cbis_mt_cbam_official.pt")
    info = D.export_demo_checkpoint(tmp_path / "cbis_mt_cbam_official.pt", tmp_path / "model.pt",
                                    {"platt": {"a": 0.6, "b": -0.5}}, {"x": 1})
    assert info["height"] == 160 and info["calibration"]["platt"]["a"] == 0.6
    model, ck = D.load_demo_model(tmp_path / "model.pt")
    f = tmp_path / "m.jpg"
    Image.fromarray(_fake_mammogram(True, lesion=True)).save(f, quality=95)
    res = D.predict_file(model, ck, f)
    assert res["image"].shape == (160, 96) and res["gradcam"].shape == (5, 3) and res["cbam"].shape == (5, 3)
    with torch.no_grad():
        x = _Prep("cpu")(torch.from_numpy(load_mammogram(f, 160, 96)[None]), train=False)
        p = torch.sigmoid(ref(x)["pathology"]).item()
    assert res["p_malignant_raw"] == pytest.approx(p, abs=1e-5)
    assert res["p_malignant"] == pytest.approx(D.calibrated_probability(p, {"a": 0.6, "b": -0.5}), abs=1e-5)
    assert sum(res["p_density"]) == pytest.approx(1, abs=1e-5)
    assert D.render_panels(res, dpi=40).ndim == 3


def test_build_end_to_end(fake_cbis, tmp_path):
    pytest.importorskip("torch")
    from mammo.cbis import locate_cbis
    ck_dir = tmp_path / "ck"
    ck_dir.mkdir()
    _fake_checkpoint(ck_dir / "cbis_mt_cbam_official.pt")
    jpeg = locate_cbis(fake_cbis)["jpeg"]
    files = sorted(jpeg.glob("*/1-1.jpg"))[:8]
    files = [f for f in files if Image.open(f).size[0] > 100][:6]
    test = pd.DataFrame({"image_id": [f"P_{k}" for k in range(len(files))],
                         "path": [f"CBIS-DDSM/jpeg/{f.parent.name}/{f.name}" for f in files],
                         "pathology": [k % 2 for k in range(len(files))], "density": [k % 4 for k in range(len(files))],
                         "lesion_type": "mass"})
    test.to_csv(tmp_path / "t.csv", index=False)
    # "Phase 5" predictions from the same untrained model
    model, ck = D.load_demo_model(ck_dir / "cbis_mt_cbam_official.pt")
    ck.setdefault("calibration", {})
    preds = [D.predict_file(model, ck, jpeg / p.split("jpeg/")[1])["p_malignant_raw"] for p in test["path"]]
    pd.DataFrame({"image_id": test["image_id"], "p_malignant_mt_cbam": preds}).to_csv(tmp_path / "p.csv", index=False)
    out, rep = tmp_path / "demo", tmp_path / "rep"
    rc = E.main(["build", "--out", str(out), "--report", str(rep), "--zip", str(tmp_path / "demo_bundle.zip"),
                 "--ckpt", str(ck_dir), "--cbis-root", str(fake_cbis),
                 "--test-images", str(tmp_path / "t.csv"), "--predictions", str(tmp_path / "p.csv"), "--n-per-class", "2"])
    assert rc == 0
    ex = pd.read_csv(out / "examples/examples.csv")
    assert len(ex) == 4 and set(ex["pathology"]) == {"malignant", "benign"}
    assert (out / "model.pt").exists() and (rep / "demo_preview.png").exists()
    chk = json.load(open(rep / "demo_check.json"))
    assert chk["max_abs_diff_vs_phase5"] < 1e-4 and chk["max_abs_diff_example_vs_original"] < 1e-6
    assert set(ex["density"]) <= set("ABCD")
    import zipfile
    names = zipfile.ZipFile(tmp_path / "demo_bundle.zip").namelist()
    assert "model.pt" in names and "examples/examples.csv" in names and "examples/example_1.png" in names
