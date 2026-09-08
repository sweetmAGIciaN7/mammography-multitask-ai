"""Attention-guided multi-task model for BI-RADS assessment + density."""

from __future__ import annotations

from torch import nn
from torchvision.models import (
    EfficientNet_B0_Weights,
    efficientnet_b0,
)

from src.models.attention import ChannelSpatialAttention


class MultiTaskAttentionEfficientNetB0(nn.Module):
    """
    EfficientNet-B0 with channel + spatial attention.

    Pipeline:
        image
        -> EfficientNet convolutional features
        -> channel + spatial attention
        -> global average pooling
        -> shared fully connected representation
        -> assessment head (5 classes)
        -> density head (4 classes)
    """

    def __init__(
        self,
        pretrained: bool = True,
    ) -> None:
        super().__init__()

        weights = (
            EfficientNet_B0_Weights.DEFAULT
            if pretrained
            else None
        )

        backbone = efficientnet_b0(
            weights=weights
        )

        # EfficientNet-B0 feature extractor.
        self.features = backbone.features

        # Final EfficientNet-B0 feature channels.
        feature_channels = 1280

        # Attention is applied before global pooling.
        self.attention = ChannelSpatialAttention(
            channels=feature_channels,
            reduction=16,
            spatial_kernel_size=7,
        )

        self.pool = nn.AdaptiveAvgPool2d(
            output_size=1
        )

        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                feature_channels,
                256,
            ),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.assessment_head = nn.Linear(
            256,
            5,
        )

        self.density_head = nn.Linear(
            256,
            4,
        )

    def freeze_backbone(self):
        """
        Freeze EfficientNet convolutional feature extractor.

        Attention and task heads remain trainable.
        """

        for parameter in self.features.parameters():
            parameter.requires_grad = False

    def unfreeze_backbone(self):
        """
        Unfreeze EfficientNet feature extractor.
        """

        for parameter in self.features.parameters():
            parameter.requires_grad = True

    def forward(self, x):
        features = self.features(x)

        attended_features = self.attention(
            features
        )

        pooled = self.pool(
            attended_features
        )

        shared_features = self.shared(
            pooled
        )

        assessment_logits = self.assessment_head(
            shared_features
        )

        density_logits = self.density_head(
            shared_features
        )

        return {
            "assessment_logits": assessment_logits,
            "density_logits": density_logits,
        }