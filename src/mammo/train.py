"""Training / inference loops.

Images are small (227x227), so the whole dataset is pre-loaded into one uint8
tensor and batches are sliced from it - no DataLoader workers, nothing to
misconfigure on Kaggle.
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
    """uint8 NHWC batch -> normalised float NCHW on device, optional random h-flip."""

    def __init__(self, device):
        self.device = device
        self.mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    def __call__(self, xb: torch.Tensor, train: bool) -> torch.Tensor:
        x = xb.to(self.device, non_blocking=True).permute(0, 3, 1, 2).float().div_(255)
        if train:
            flip = torch.rand(x.shape[0], device=self.device) < 0.5
            x = torch.where(flip.view(-1, 1, 1, 1), x.flip(3), x)
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


def build_model(cfg: TrainConfig, tasks=("pathology", "density")) -> MultiTaskNet:
    return MultiTaskNet(cfg.backbone, pretrained=cfg.pretrained, attention=cfg.attention, tasks=tuple(tasks))


def train_model(model, X: torch.Tensor, y_path: np.ndarray, y_dens: np.ndarray, train_idx: np.ndarray,
                cfg: TrainConfig, device, log=print) -> None:
    prep = _Prep(device)
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
    prep = _Prep(device)
    p_path, p_dens = [], []
    for s in range(0, len(idx), batch_size):
        b = torch.as_tensor(idx[s:s + batch_size])
        x = prep(X[b], train=False)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            out = model(x)
            if tta:
                out_f = model(x.flip(3))
                out = {k: (out[k] + out_f[k]) / 2 for k in out}
        if "pathology" in out:
            p_path.append(torch.sigmoid(out["pathology"].float()).cpu())
        if "density" in out:
            p_dens.append(torch.softmax(out["density"].float(), 1).cpu())
    res = {}
    if p_path:
        res["pathology"] = torch.cat(p_path).numpy()
    if p_dens:
        res["density"] = torch.cat(p_dens).numpy()
    return res
