"""Training / inference loops.

The whole dataset is pre-loaded into one uint8 tensor (RGB ``N,H,W,3`` for the
small Phase 2 images, grayscale ``N,H,W`` for CBIS-DDSM mammograms) and batches
are sliced from it: no DataLoader workers, nothing to misconfigure on Kaggle.
Augmentation runs on the GPU.
"""
from __future__ import annotations

import math
import random
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .model import MultiTaskNet

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class TrainConfig:
    backbone: str = "efficientnet_b0"
    attention: bool = True
    pretrained: bool = True
    image_size: int = 224
    epochs: int = 8
    batch_size: int = 64
    lr: float = 2e-4
    weight_decay: float = 1e-2
    w_pathology: float = 2.0      # paper: lambda_1
    w_density: float = 0.8        # paper: lambda_2
    label_smoothing: float = 0.05  # paper: density CE smoothing
    seed: int = 42
    amp: bool = True
    augment: str = "flip"          # "flip" (Phase 2) or "mammo" (Phase 3, see _Prep)
    tasks: tuple = ("pathology", "density")

    def to_dict(self):
        return asdict(self)


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def load_images(paths, size: int = 224, workers: int = 8) -> np.ndarray:
    def _load(p):
        with Image.open(p) as im:
            return np.asarray(im.convert("RGB").resize((size, size), Image.BILINEAR), dtype=np.uint8)
    with ThreadPoolExecutor(workers) as ex:
        return np.stack(list(ex.map(_load, list(paths))))


class _Prep:
    """uint8 batch -> normalised float NCHW on the device, with optional GPU augmentation.

    augment="flip":  random horizontal flip (Phase 2 images).
    augment="mammo": for whole mammograms that were oriented chest-wall-left during preprocessing, so a
                     horizontal flip would break that convention. Instead: random vertical flip, small
                     rotation (+-10 deg), zoom (0.9-1.15), shift (+-5%) and brightness/contrast jitter.
    """

    def __init__(self, device, augment: str = "flip"):
        self.device = device
        self.augment = augment
        self.mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    def _mammo_aug(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        dev = x.device
        vflip = torch.rand(n, device=dev) < 0.5
        x = torch.where(vflip.view(-1, 1, 1, 1), x.flip(2), x)
        ang = (torch.rand(n, device=dev) * 2 - 1) * math.radians(10)
        scale = 1 / (0.9 + torch.rand(n, device=dev) * 0.25)
        shift = (torch.rand(n, 2, device=dev) * 2 - 1) * 0.1  # grid units: 0.1 = 5% of the image
        cos, sin = torch.cos(ang) * scale, torch.sin(ang) * scale
        h, w = x.shape[2], x.shape[3]
        theta = torch.stack([torch.stack([cos, -sin * h / w, shift[:, 0]], 1),
                             torch.stack([sin * w / h, cos, shift[:, 1]], 1)], 1)
        grid = F.affine_grid(theta, list(x.shape), align_corners=False)
        x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        contrast = 1 + (torch.rand(n, 1, 1, 1, device=dev) * 2 - 1) * 0.15
        bright = (torch.rand(n, 1, 1, 1, device=dev) * 2 - 1) * 0.08
        return ((x - 0.5) * contrast + 0.5 + bright).clamp_(0, 1)

    def __call__(self, xb: torch.Tensor, train: bool) -> torch.Tensor:
        x = xb.to(self.device, non_blocking=True)
        if x.ndim == 3:                                   # grayscale N,H,W -> N,1,H,W
            x = x.unsqueeze(1).float().div_(255)
        else:                                             # RGB N,H,W,3 -> N,3,H,W
            x = x.permute(0, 3, 1, 2).float().div_(255)
        if train and self.augment == "mammo":
            x = self._mammo_aug(x)
        elif train and self.augment == "flip":
            flip = torch.rand(x.shape[0], device=self.device) < 0.5
            x = torch.where(flip.view(-1, 1, 1, 1), x.flip(3), x)
        if x.shape[1] == 1:
            x = x.expand(-1, 3, -1, -1)
        return (x - self.mean) / self.std


def multitask_loss(out: dict, y_path: torch.Tensor, y_dens: torch.Tensor, cfg: TrainConfig) -> torch.Tensor:
    loss = torch.zeros((), device=y_path.device)
    if "pathology" in out:
        loss = loss + cfg.w_pathology * F.binary_cross_entropy_with_logits(out["pathology"].float(), y_path.float())
    if "density" in out:
        known = y_dens >= 0  # missing density labels are masked out, never guessed
        if known.any():
            loss = loss + cfg.w_density * F.cross_entropy(out["density"][known].float(), y_dens[known],
                                                          label_smoothing=cfg.label_smoothing)
    return loss


def build_model(cfg: TrainConfig, tasks=None) -> MultiTaskNet:
    return MultiTaskNet(cfg.backbone, pretrained=cfg.pretrained, attention=cfg.attention,
                        tasks=tuple(tasks or cfg.tasks))


def train_model(model, X: torch.Tensor, y_path: np.ndarray, y_dens: np.ndarray, train_idx: np.ndarray,
                cfg: TrainConfig, device, log=print) -> None:
    prep = _Prep(device, cfg.augment)
    yp = torch.as_tensor(y_path, dtype=torch.long)
    yd = torch.as_tensor(y_dens, dtype=torch.long)
    use_amp = cfg.amp and device.type == "cuda"
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = max(1, math.ceil(len(train_idx) / cfg.batch_size))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=cfg.epochs * steps_per_epoch,
                                                pct_start=0.15)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    rng = np.random.default_rng(cfg.seed)
    for epoch in range(cfg.epochs):
        model.train()
        t0, total, seen = time.time(), 0.0, 0
        order = rng.permutation(train_idx)
        for s in range(0, len(order), cfg.batch_size):
            b = torch.as_tensor(order[s:s + cfg.batch_size])
            if len(b) < 2:  # BatchNorm needs >= 2 samples
                continue
            x = prep(X[b], train=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                out = model(x)
            loss = multitask_loss(out, yp[b].to(device), yd[b].to(device), cfg)
            if not loss.requires_grad:  # e.g. a density-only batch with no known density label
                continue
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            if sched.last_epoch < sched.total_steps - 1:
                sched.step()
            total += loss.item() * len(b); seen += len(b)
        log(f"    epoch {epoch + 1}/{cfg.epochs}  loss {total / max(seen, 1):.4f}  ({time.time() - t0:.0f}s)")


@torch.no_grad()
def predict(model, X: torch.Tensor, idx: np.ndarray, device, batch_size: int = 128, tta: bool = False) -> dict:
    model.eval()
    prep = _Prep(device)  # no augmentation at test time
    p_path, p_dens, l_path, l_dens = [], [], [], []
    for s in range(0, len(idx), batch_size):
        b = torch.as_tensor(idx[s:s + batch_size])
        x = prep(X[b], train=False)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            out = model(x)
            if tta:
                out_f = model(x.flip(3))
                out = {k: (out[k] + out_f[k]) / 2 for k in out}
        if "pathology" in out:
            l_path.append(out["pathology"].float().cpu())
            p_path.append(torch.sigmoid(out["pathology"].float()).cpu())
        if "density" in out:
            l_dens.append(out["density"].float().cpu())
            p_dens.append(torch.softmax(out["density"].float(), 1).cpu())
    res = {}
    if p_path:  # probabilities, plus raw logits for temperature scaling in Phase 4
        res["pathology"] = torch.cat(p_path).numpy()
        res["pathology_logit"] = torch.cat(l_path).numpy()
    if p_dens:
        res["density"] = torch.cat(p_dens).numpy()
        res["density_logits"] = torch.cat(l_dens).numpy()
    return res
