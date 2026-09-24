"""Mammogram preprocessing: crop to the breast, orient, resize with padding, cache.

Whole CBIS-DDSM mammograms are ~3000x5000 px scanned films, so decoding them is
the slow part. Every image is processed **once** into a fixed-size uint8 array
(``H x W``, grayscale) and stored in a ``.npy`` cache that later runs reuse.

Steps per image
1. Decode at reduced resolution (JPEG draft mode, several times faster).
2. Find the breast: Otsu threshold on a small copy, keep the largest connected
   region, crop to its bounding box. This removes film borders and most
   scanner labels.
3. Orient: flip so the chest wall (the brighter half) is on the left. Every
   breast then "points" the same way, whatever side it came from.
4. Resize keeping the aspect ratio, pad the right/bottom with black.
"""
from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage


def otsu_threshold(a: np.ndarray) -> float:
    hist = np.bincount(a.ravel(), minlength=256).astype(float)
    p = hist / hist.sum()
    omega = np.cumsum(p)
    mu = np.cumsum(p * np.arange(256))
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (mu[-1] * omega - mu) ** 2 / (omega * (1 - omega))
    return float(np.nanargmax(between))


def breast_bbox(a: np.ndarray, small: int = 256) -> tuple[int, int, int, int]:
    """(top, bottom, left, right) of the largest bright region, in ``a``'s coordinates."""
    h, w = a.shape
    s = max(h, w) / small
    sm = np.asarray(Image.fromarray(a).resize((max(1, round(w / s)), max(1, round(h / s))), Image.BILINEAR))
    mask = sm > max(otsu_threshold(sm), 10)
    mask = ndimage.binary_opening(mask, iterations=2)
    lab, n = ndimage.label(mask)
    if n == 0:
        return 0, h, 0, w
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    rows, cols = np.nonzero(lab == (int(np.argmax(sizes)) + 1))
    t, b = int(rows.min() * s), int(np.ceil((rows.max() + 1) * s))
    l, r = int(cols.min() * s), int(np.ceil((cols.max() + 1) * s))
    if (b - t) < 0.2 * h or (r - l) < 0.1 * w:  # implausibly small -> don't trust it
        return 0, h, 0, w
    return max(t, 0), min(b, h), max(l, 0), min(r, w)


def orient_left(a: np.ndarray) -> np.ndarray:
    """Flip horizontally if the right half is brighter (chest wall on the right)."""
    half = a.shape[1] // 2
    return a[:, ::-1] if a[:, half:].mean() > a[:, :half].mean() else a


def fit_pad(a: np.ndarray, height: int, width: int) -> np.ndarray:
    h, w = a.shape
    scale = min(height / h, width / w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    out = np.zeros((height, width), np.uint8)
    out[:nh, :nw] = np.asarray(Image.fromarray(a).resize((nw, nh), Image.BILINEAR))
    return out


def load_mammogram(path: str | Path, height: int = 640, width: int = 384) -> np.ndarray:
    return load_mammogram_with_transform(path, height, width)[0]


@dataclass(frozen=True)
class Transform:
    """The geometry ``load_mammogram`` applied to one image, so annotations can follow it (Phase 5).

    ``orig_*``: size of the file on disk. ``dec_*``: size actually decoded (JPEG draft mode shrinks by 2/4/8).
    ``top/bottom/left/right``: breast crop, in decoded pixels. ``flip``: mirrored left-right after cropping.
    ``nh/nw``: size of the resized crop; the rest of the ``height x width`` output is zero padding."""
    orig_h: int
    orig_w: int
    dec_h: int
    dec_w: int
    top: int
    bottom: int
    left: int
    right: int
    flip: bool
    nh: int
    nw: int
    height: int
    width: int

    def to_dict(self) -> dict:
        return asdict(self)


def load_mammogram_with_transform(path: str | Path, height: int = 640, width: int = 384) -> tuple[np.ndarray, Transform]:
    """Same pixels as ``load_mammogram`` (it calls this), plus the geometric ``Transform`` that produced them."""
    with Image.open(path) as im:
        orig_w, orig_h = im.size
        if im.format == "JPEG":
            im.draft("L", (width * 2, height * 2))  # decode at >= 2x the target size, much faster
        a = np.asarray(im.convert("L"))
    t, b, l, r = breast_bbox(a)
    c = a[t:b, l:r]
    half = c.shape[1] // 2
    flip = bool(c[:, half:].mean() > c[:, :half].mean())  # identical rule to orient_left
    c = c[:, ::-1] if flip else c
    h, w = c.shape
    scale = min(height / h, width / w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    tf = Transform(orig_h, orig_w, a.shape[0], a.shape[1], t, b, l, r, flip, nh, nw, height, width)
    return fit_pad(np.ascontiguousarray(c), height, width), tf


def _resize_float(a: np.ndarray, h: int, w: int) -> np.ndarray:
    if a.shape == (h, w):
        return a.astype(np.float32, copy=False)
    return np.asarray(Image.fromarray(a.astype(np.float32)).resize((w, h), Image.BILINEAR))


def apply_transform(a: np.ndarray, tf: Transform) -> np.ndarray:
    """Send an array that is pixel-aligned with the original file (e.g. a lesion mask) through the same
    decode-size / crop / flip / resize / pad steps. Returns float32 ``height x width``; a 0/1 mask comes back as
    the fraction of each output pixel covered by the mask."""
    if a.shape not in ((tf.orig_h, tf.orig_w), (tf.dec_h, tf.dec_w)):
        raise ValueError(f"array {a.shape} matches neither the original {(tf.orig_h, tf.orig_w)} "
                         f"nor the decoded {(tf.dec_h, tf.dec_w)} image size")
    a = _resize_float(np.asarray(a, np.float32), tf.dec_h, tf.dec_w)
    c = a[tf.top:tf.bottom, tf.left:tf.right]
    if tf.flip:
        c = c[:, ::-1]
    out = np.zeros((tf.height, tf.width), np.float32)
    out[:tf.nh, :tf.nw] = _resize_float(np.ascontiguousarray(c), tf.nh, tf.nw)
    return out


def _key(paths, height, width) -> str:
    h = hashlib.sha1(f"{height}x{width}|v1|".encode())
    for p in paths:
        h.update(str(p).encode()); h.update(b"\0")
    return h.hexdigest()[:12]


def cached_images(paths, height: int, width: int, cache_dir: str | Path, workers: int = 4, log=print) -> np.ndarray:
    """Preprocess ``paths`` once into ``cache_dir/cbis_<H>x<W>_<hash>.npy`` and return the array (N, H, W) uint8."""
    paths = [str(p) for p in paths]
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    f = cache_dir / f"cbis_{height}x{width}_{_key(paths, height, width)}.npy"
    if f.exists():
        arr = np.load(f)
        if arr.shape == (len(paths), height, width):
            log(f"using cached images {f.name}")
            return arr
    tmp = f.with_suffix(".partial.npy")
    arr = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=(len(paths), height, width))
    done = 0

    def work(i):
        arr[i] = load_mammogram(paths[i], height, width)

    with ThreadPoolExecutor(workers) as ex:
        for _ in ex.map(work, range(len(paths))):
            done += 1
            if done % 250 == 0 or done == len(paths):
                log(f"  preprocessed {done}/{len(paths)}")
    arr.flush()
    del arr
    tmp.replace(f)
    json.dump({"n": len(paths), "height": height, "width": width}, open(f.with_suffix(".json"), "w"))
    return np.load(f)


def preview_grid(arr: np.ndarray, titles, path, ncols: int = 8) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(arr)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.6, nrows * 2.7), facecolor="#fcfcfb")
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, a, t in zip(np.atleast_1d(axes).ravel(), arr, titles):
        ax.imshow(a, cmap="gray", vmin=0, vmax=255)
        ax.set_title(t, fontsize=7, color="#0b0b0b")
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor="#fcfcfb")
    plt.close(fig)
