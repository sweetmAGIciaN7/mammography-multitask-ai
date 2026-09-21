import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()

        hidden = max(channels // reduction, 1)

        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))

        attention = self.sigmoid(avg_out + max_out)

        return x * attention


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()

        padding = kernel_size // 2

        self.conv = nn.Conv2d(
            2,
            1,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(
            x,
            dim=1,
            keepdim=True,
        )

        max_out, _ = torch.max(
            x,
            dim=1,
            keepdim=True,
        )

        combined = torch.cat(
            [avg_out, max_out],
            dim=1,
        )

        attention = self.sigmoid(
            self.conv(combined)
        )

        return x * attention


class CBAMBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.channel_attention = ChannelAttention(
            channels
        )

        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)

        return x


class AttentionMultiTaskResNet18(nn.Module):
    """
    ResNet18 backbone + CBAM-style attention
    + two task heads:

    BI-RADS:
        5 classes

    Density:
        4 classes
    """

    def __init__(
        self,
        num_birads_classes=5,
        num_density_classes=4,
        pretrained=True,
    ):
        super().__init__()

        weights = (
            ResNet18_Weights.DEFAULT
            if pretrained
            else None
        )

        backbone = resnet18(
            weights=weights
        )

        self.feature_extractor = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )

        feature_channels = 512

        self.attention = CBAMBlock(
            feature_channels
        )

        self.pool = nn.AdaptiveAvgPool2d(
            (1, 1)
        )

        self.birads_head = nn.Linear(
            feature_channels,
            num_birads_classes,
        )

        self.density_head = nn.Linear(
            feature_channels,
            num_density_classes,
        )

    def forward(self, x):
        features = self.feature_extractor(x)

        features = self.attention(
            features
        )

        pooled = self.pool(
            features
        )

        pooled = torch.flatten(
            pooled,
            1,
        )

        birads_logits = self.birads_head(
            pooled
        )

        density_logits = self.density_head(
            pooled
        )

        return {
            "birads": birads_logits,
            "density": density_logits,
        }