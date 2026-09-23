"""Multi-task CNN: pretrained backbone -> CBAM attention -> shared MLP -> two heads.

Follows Section IV-B of Esen et al. (2025): channel attention (avg+max pooled
MLP, reduction 16) then 7x7 spatial attention on the last feature map, a shared
Dense(512)-BN-Dropout(0.5)-Dense(256)-BN-Dropout(0.3) trunk, and
Dense(128)-Dropout(0.25) heads for pathology (1 logit) and density (4 logits).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )

    def forward(self, x):
        w = torch.sigmoid(self.mlp(x.mean((2, 3), keepdim=True)) + self.mlp(x.amax((2, 3), keepdim=True)))
        return x * w


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        a = torch.sigmoid(self.conv(torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], 1)))
        return x * a, a


class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel = ChannelAttention(channels, reduction)
        self.spatial = SpatialAttention(kernel_size)

    def forward(self, x):
        return self.spatial(self.channel(x))


def _backbone(name: str, pretrained: bool) -> tuple[nn.Module, int]:
    if name == "efficientnet_b0":
        net = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        return net.features, 1280
    if name == "efficientnet_b3":
        net = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None)
        return net.features, 1536
    if name == "mobilenet_v3_large":
        net = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None)
        return net.features, 960
    if name == "resnet50":
        net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        return nn.Sequential(*list(net.children())[:-2]), 2048
    raise ValueError(f"Unknown backbone {name!r}")


class MultiTaskNet(nn.Module):
    def __init__(self, backbone: str = "efficientnet_b0", pretrained: bool = True, attention: bool = True,
                 n_density: int = 4, tasks: tuple[str, ...] = ("pathology", "density")):
        super().__init__()
        self.features, c = _backbone(backbone, pretrained)
        self.attention = CBAM(c) if attention else None
        self.tasks = tasks
        self.shared = nn.Sequential(
            nn.Linear(c, 512), nn.ReLU(inplace=True), nn.BatchNorm1d(512), nn.Dropout(0.5),
            nn.Linear(512, 256), nn.ReLU(inplace=True), nn.BatchNorm1d(256), nn.Dropout(0.3),
        )

        def head(n_out):
            return nn.Sequential(nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Dropout(0.25), nn.Linear(128, n_out))

        self.pathology_head = head(1) if "pathology" in tasks else None
        self.density_head = head(n_density) if "density" in tasks else None

    def forward(self, x, return_attention: bool = False):
        f = self.features(x)
        attn = None
        if self.attention is not None:
            f, attn = self.attention(f)
        z = self.shared(f.mean((2, 3)))
        out = {}
        if self.pathology_head is not None:
            out["pathology"] = self.pathology_head(z).squeeze(1)
        if self.density_head is not None:
            out["density"] = self.density_head(z)
        if return_attention:
            out["attention"] = attn
        return out
