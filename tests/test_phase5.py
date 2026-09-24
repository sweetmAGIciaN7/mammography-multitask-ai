"""Phase 5: mask geometry, localisation metrics, baselines, paper statistics, and the end-to-end pipeline."""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from conftest import _fake_mammogram, fake_lesion
from mammo.localization import (ClusterBootstrap, Geometry, adaptive_pool, baseline_maps, breast_region,
                                distribution_stats, localisation_metrics, looks_binary, pick_mask, pool_cells, upsample)
from mammo.preprocess import (_key, apply_transform, breast_bbox, fit_pad, load_mammogram,
                              load_mammogram_with_transform, orient_left)


def _load_mammogram_v1(path, height=640, width=384):
    """Frozen copy of load_mammogram as of Phase 3/4 (commit 656faeb)."""
    with Image.open(path) as im:
        if im.format == "JPEG":
            im.draft("L", (width * 2, height * 2))
        a = np.asarray(im.convert("L"))
    t, b, l, r = breast_bbox(a)
    a = orient_left(a[t:b, l:r])
    return fit_pad(np.ascontiguousarray(a), height, width)


def _save(tmp_path, arr, name):
    f = tmp_path / name
    Image.fromarray(arr).save(f, quality=95) if name.endswith(".jpg") else Image.fromarray(arr).save(f)
    return f


@pytest.mark.parametrize("right", [False, True])
@pytest.mark.parametrize("ext", [".jpg", ".png"])
@pytest.mark.parametrize("size", [(640, 384), (160, 96)])
def test_pixels_identical_to_phase3_loader(tmp_path, right, ext, size):
    f = _save(tmp_path, _fake_mammogram(right, h=1400, w=840, seed=3, lesion=True), "m" + ext)
    new, tf = load_mammogram_with_transform(f, *size)
    assert np.array_equal(new, _load_mammogram_v1(f, *size))
    assert np.array_equal(load_mammogram(f, *size), new)
    assert tf.flip == right


def test_cache_key_unchanged():
    assert _key(["/kaggle/input/a/1-1.jpg", "/kaggle/input/b/1-2.jpg"], 640, 384) == "8308f8df5ce3"


@pytest.mark.parametrize("right", [False, True])
@pytest.mark.parametrize("h,w", [(1400, 840), (500, 300)])
def test_mask_follows_crop_flip_resize(tmp_path, right, h, w):
    """The lesion is painted white into the image; after preprocessing, the mapped mask must sit on those pixels."""
    f = _save(tmp_path, _fake_mammogram(right, h=h, w=w, seed=1, lesion=True), "m.jpg")
    img, tf = load_mammogram_with_transform(f, 640, 384)
    m = apply_transform(fake_lesion(right, h, w), tf) >= 0.5
    bright = img >= 245
    assert m.sum() > 20
    assert (m & bright).sum() / (m | bright).sum() > 0.6
    if right:  # breast was on the right, so the lesion must now be in the left half (chest wall on the left)
        assert np.nonzero(m)[1].mean() < 384 / 2


def test_apply_transform_reproduces_the_image(tmp_path):
    arr = _fake_mammogram(True, h=500, w=300, seed=2)
    f = _save(tmp_path, arr, "m.png")
    img, tf = load_mammogram_with_transform(f, 640, 384)
    assert np.abs(apply_transform(arr, tf) - img.astype(float)).mean() < 1.0
    with pytest.raises(ValueError):
        apply_transform(np.zeros((10, 10)), tf)


# ------------------------------------------------------------------ metric sanity
def _geom(lesion_cells=((8, 4), (9, 4), (8, 5), (9, 5)), H=640, W=384, h=20, w=12):
    lesion = np.zeros((H, W), bool)
    for r, c in lesion_cells:
        lesion[r * 32:(r + 1) * 32, c * 32:(c + 1) * 32] = True
    breast = np.zeros((H, W), bool)
    breast[:, :300] = True
    return Geometry(lesion, breast, (h, w))


def test_perfect_map_scores_one():
    g = _geom()
    m = localisation_metrics(g.cell_lesion.astype(float), g)
    assert m["pg_px"] == 1 and m["pg_px_tol"] == 1 and m["pg_cell"] == 1
    assert m["energy_cell"] == pytest.approx(1) and m["auc_cell"] == pytest.approx(1)
    assert m["auc_px"] > 0.97 and m["energy_px"] > 0.6


def test_uniform_map_scores_area_fraction():
    g = _geom()
    m = localisation_metrics(np.ones((20, 12)), g)
    assert m["pg_px"] == pytest.approx(g.area_fraction)
    assert m["energy_px"] == pytest.approx(g.area_fraction)
    assert m["auc_px"] == pytest.approx(0.5) and m["auc_cell"] == pytest.approx(0.5)
    assert m["iou_top10"] == pytest.approx(g.area_fraction)  # ties -> whole breast selected


def test_wrong_map_scores_zero_and_zero_map_is_uninformative():
    g = _geom()
    far = np.zeros((20, 12)); far[1, 1] = 1.0
    m = localisation_metrics(far, g)
    assert m["pg_px"] == 0 and m["pg_px_tol"] == 0 and m["pg_cell"] == 0 and m["energy_px"] < 0.01
    z = localisation_metrics(np.zeros((20, 12)), g)            # an all-zero Grad-CAM carries no information
    assert z["pg_px"] == pytest.approx(g.area_fraction) and z["energy_px"] == pytest.approx(g.area_fraction)


def test_tolerance_counts_adjacent_cells():
    g = _geom(lesion_cells=((8, 4),))
    m = np.zeros((20, 12)); m[8, 5] = 1.0                      # right next to the lesion cell
    r = localisation_metrics(m, g)
    assert r["pg_cell"] == 0 and r["pg_cell_tol"] == 1 and r["pg_px_tol"] == 1


def test_upsample_and_pooling():
    c = np.full((20, 12), 0.3)
    assert np.allclose(upsample(c, 640, 384), 0.3)
    a = np.random.default_rng(0).random((20, 12))
    assert np.allclose(pool_cells(upsample(a, 640, 384), 20, 12).mean(), a.mean(), atol=0.02)
    assert np.allclose(adaptive_pool(np.full((20, 12), 2.0), 7, 7), 2.0)
    p = adaptive_pool(np.arange(240, dtype=float).reshape(20, 12), 7, 7)
    assert p.shape == (7, 7) and p[0, 0] < p[-1, -1]


def test_upsample_matches_torch():
    torch = pytest.importorskip("torch")
    a = np.random.default_rng(1).random((20, 12))
    t = torch.nn.functional.interpolate(torch.tensor(a)[None, None], size=(640, 384), mode="bilinear", align_corners=False)
    assert np.allclose(upsample(a, 640, 384), t[0, 0].numpy(), atol=1e-6)


def test_distribution_stats_extremes():
    n = 240
    u = distribution_stats(np.ones((20, 12)))
    assert u["gini"] == pytest.approx(0, abs=1e-9) and u["entropy"] == pytest.approx(math.log(n))
    assert u["entropy_norm"] == pytest.approx(1) and u["area80"] == pytest.approx(math.ceil(0.8 * n) / n)
    one = np.zeros((20, 12)); one[3, 3] = 0.9
    o = distribution_stats(one)
    assert o["gini"] == pytest.approx((n - 1) / n) and o["entropy"] == pytest.approx(0) and o["area80"] == pytest.approx(1 / n)
    assert o["peak"] == pytest.approx(0.9)
    p7 = distribution_stats(adaptive_pool(np.ones((20, 12)), 7, 7))
    assert p7["n_cells"] == 49 and p7["entropy"] == pytest.approx(math.log(49))   # the paper's ceiling: ln 49 = 3.89
    assert np.isnan(distribution_stats(np.zeros((20, 12)))["gini"])


def test_baselines():
    img = _fake_mammogram(False, h=640, w=384, seed=0)
    b = breast_region(img)
    base = baseline_maps(img, b, (20, 12), np.random.default_rng(0), n_random=3)
    assert set(base) == {"uniform", "random_0", "random_1", "random_2", "brightness", "centre"}
    assert all(v.shape == (20, 12) for v in base.values())
    assert np.all(base["uniform"] == 1)
    assert np.allclose(base["brightness"], pool_cells(img / 255.0, 20, 12))
    ys, xs = np.nonzero(b)
    r, c = np.unravel_index(base["centre"].argmax(), (20, 12))
    assert abs(r * 32 + 16 - ys.mean()) <= 32 and abs(c * 32 + 16 - xs.mean()) <= 32
    assert b[:6].sum() == 0                                     # the scanner strip is not breast


def test_cluster_bootstrap():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(50), 2)
    v = rng.normal(1.0, 0.5, 100)
    bs = ClusterBootstrap(groups, n=500)
    m = bs.mean(v)
    assert m["mean"] == pytest.approx(v.mean()) and m["ci95"][0] < v.mean() < m["ci95"][1]
    p = bs.paired(v, v)
    assert p["diff"] == 0 and p["p_not_better"] == 1
    d = bs.group_diff(v + (groups < 25), groups < 25, groups >= 25)
    assert d["ci95"][0] > 0


def test_pick_mask_by_content(tmp_path):
    H, W = 500, 300
    crop = _save(tmp_path, np.full((50, 50), 200, np.uint8), "a.jpg")
    mask = _save(tmp_path, fake_lesion(False, H, W).astype(np.uint8) * 255, "b.jpg")
    gray = _save(tmp_path, _fake_mammogram(False, H, W), "c.jpg")                # same size, not binary
    m, info = pick_mask([str(crop), str(gray), str(mask)], (W, H))
    assert info["status"] == "ok" and info["mask_file"] == str(mask)
    assert (m ^ fake_lesion(False, H, W)).sum() < 30
    near = _save(tmp_path, fake_lesion(False, H + 4, W).astype(np.uint8) * 255, "d.jpg")   # 0.8% taller
    m2, info2 = pick_mask([str(near)], (W, H))
    assert info2["status"] == "ok_resized" and m2.shape == (H, W)
    assert pick_mask([str(crop)], (W, H))[1]["status"] == "no_file_of_image_size"
    assert pick_mask([str(gray)], (W, H))[1]["status"] == "not_binary"
    assert pick_mask([], (W, H))[1]["status"] == "no_candidate_files"
    assert looks_binary(np.array([[0, 255], [3, 250]], np.uint8)) and not looks_binary(np.full((4, 4), 128, np.uint8))


# ------------------------------------------------------------------ pipeline on the fake CBIS-DDSM
def _prepare(fake_cbis, tmp_path):
    from mammo.experiments import attention as A
    out, cache = tmp_path / "res", tmp_path / "cache"
    common = ["--data-root", str(fake_cbis), "--out", str(out), "--cache", str(cache), "--workers", "2", "--smoke"]
    assert A.main(["prepare", *common]) == 0
    return A, common, Path(str(out) + "_smoke"), cache


def test_prepare_finds_masks_by_content(fake_cbis, tmp_path):
    _, _, out, cache = _prepare(fake_cbis, tmp_path)
    st = json.load(open(out / "data_stats.json"))
    test = pd.read_csv(out / "test_images.csv")
    les = pd.read_csv(out / "lesions.csv")
    # official test patients P_00011, P_00012 -> 4 images, 4 abnormalities; P_00012 RIGHT MLO has no mask file
    assert st["test_images"] == 4 and st["abnormalities"] == 4
    assert st["usable_masks"] == 3 and st["mask_status"]["no_file_of_image_size"] == 1
    assert st["mask_file_labelled_as"].get("cropped images") == 1          # the swapped-label case is still found
    assert set(test["lesion_type"]) == {"mass", "calc", "no_mask"}
    assert (out / "mask_preview.png").exists()
    d = np.load(cache / "attention_160x96_smoke.npz")
    for i in np.flatnonzero(test["usable"]):
        les_px = d["lesion"][i]
        assert les_px.any() and (les_px & d["breast"][i]).sum() == les_px.sum()
        # the painted lesion (brightest pixels) and the mapped mask coincide
        bright = d["images"][i] >= 240
        assert (les_px & bright).sum() / max(les_px.sum(), 1) > 0.5
    assert les.loc[les["status"] == "ok", "area_model_px"].min() > 0


def test_scoring_and_summary_without_torch(fake_cbis, tmp_path):
    """Everything after the forward passes, with a perfect 'model' map and random ones, then summarise + chart."""
    import argparse
    A, common, out, cache = _prepare(fake_cbis, tmp_path)
    test = pd.read_csv(out / "test_images.csv", dtype={"image_id": str, "patient": str})
    d = np.load(cache / "attention_160x96_smoke.npz")
    N, rng = len(test), np.random.default_rng(0)
    perfect = np.stack([pool_cells(les, 5, 3) > 0 for les in d["lesion"]]).astype(float)
    maps = {"cbam:mt_cbam": 0.3 + 0.5 * perfect, "gradcam_mal:mt_cbam": perfect, "gradcam_mal:mt_plain": rng.random((N, 5, 3)),
            "gradcam_dens:st_dens": rng.random((N, 5, 3)), "cbam:st_dens": rng.random((N, 5, 3)),
            "control:untrained_cbam": rng.random((N, 5, 3)), "control:untrained_gradcam": rng.random((N, 5, 3))}
    preds = pd.DataFrame({"image_id": test["image_id"], "p_malignant_mt_cbam": 0.6, "p_malignant_mt_plain": 0.4,
                          "pred_density_st_dens": 1})
    args = argparse.Namespace(out=str(out), seed=0, n_random=2, gallery_n=3)
    assert A.evaluate_maps(maps, preds, {}, test, d["images"], d["lesion"], d["breast"], args, print) == 0
    met = pd.read_csv(out / "metrics.csv")
    assert set(met["method"]) >= {"base:random", "base:uniform", "ref:oracle", "cbam:mt_cbam"}
    assert not met["method"].str.startswith("base:random_").any()
    assert (met.groupby("method").size() == int(test["usable"].sum())).all()
    assert (out / "gallery.png").exists() and (out / "maps.npz").exists()
    assert A.main(["summarise", *common]) == 0
    s = json.load(open(out / "summary.json"))
    # a perfect map at 5x3 resolution: always the right cell (these lesions are smaller than a cell, so the strict
    # pixel-level pointing game can still miss; the ceiling row shows exactly that)
    assert s["localisation"]["gradcam_mal:mt_cbam"]["subsets"]["all"]["pg_cell"]["mean"] == 1
    assert s["localisation"]["ref:oracle"]["subsets"]["all"]["pg_cell"]["mean"] == 1
    assert s["localisation"]["base:uniform"]["subsets"]["all"]["auc_px"]["mean"] == pytest.approx(0.5)
    assert "cbam:mt_cbam" in s["paper_statistics"] and s["run_info"]["cbam_dynamic_range"]
    assert (out / "localisation_chart.png").exists() and "Paired comparisons" in (out / "summary.md").read_text()


# ------------------------------------------------------------------ torch parts (run on Kaggle)
def test_explain_matches_forward_and_shapes():
    torch = pytest.importorskip("torch")
    from mammo.explain import explain_batch
    from mammo.model import MultiTaskNet
    torch.manual_seed(0)
    for attention, tasks in ((True, ("pathology", "density")), (False, ("pathology", "density")), (True, ("density",))):
        m = MultiTaskNet("efficientnet_b0", pretrained=False, attention=attention, tasks=tasks).eval()
        x = torch.randn(3, 3, 160, 96)
        e = explain_batch(m, x)
        with torch.no_grad():
            ref = m(x, return_attention=True)
        if "pathology" in tasks:
            assert np.allclose(e["p_malignant"], torch.sigmoid(ref["pathology"]).numpy(), atol=1e-5)
            assert e["gradcam_mal"].shape == (3, 5, 3) and (e["gradcam_mal"] >= 0).all()
        else:
            assert "gradcam_mal" not in e
        assert e["gradcam_dens"].shape == (3, 5, 3)
        assert np.allclose(e["p_density"], torch.softmax(ref["density"], 1).numpy(), atol=1e-5)
        if attention:
            assert np.allclose(e["cbam"], ref["attention"][:, 0].numpy(), atol=1e-6)
        else:
            assert "cbam" not in e


def test_gradcam_is_per_sample():
    """A sample's Grad-CAM must not depend on the other images in the batch."""
    torch = pytest.importorskip("torch")
    from mammo.explain import explain_batch
    from mammo.model import MultiTaskNet
    torch.manual_seed(0)
    m = MultiTaskNet("efficientnet_b0", pretrained=False).eval()
    x = torch.randn(4, 3, 160, 96)
    both = explain_batch(m, x)["gradcam_mal"]
    alone = explain_batch(m, x[:1])["gradcam_mal"]
    assert np.allclose(both[0], alone[0], atol=1e-5)


def test_run_end_to_end_untrained(fake_cbis, tmp_path):
    pytest.importorskip("torch")
    A, common, out, _ = _prepare(fake_cbis, tmp_path)
    assert A.main(["run", *common]) == 0
    assert A.main(["summarise", *common]) == 0
    met = pd.read_csv(out / "metrics.csv")
    assert {"cbam:mt_cbam", "gradcam_mal:mt_plain", "gradcam_dens:st_dens", "base:random", "control:untrained_gradcam"} \
        <= set(met["method"])
    assert (out / "gallery.png").exists() and (out / "localisation_chart.png").exists()
