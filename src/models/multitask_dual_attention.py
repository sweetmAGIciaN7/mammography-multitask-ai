"""Dual-attention multi-task model for BI-RADS assessment + density."""

from __future__ import annotations

from torch import nn
from torchvision.models import (
    EfficientNet_B0_Weights,
    efficientnet_b0,
)

from src.models.attention import ChannelSpatialAttention


class MultiTaskDualAttentionEfficientNetB0(nn.Module):
    """
    EfficientNet-B0 with task-specific attention branches.

    Pipeline:
        image
        -> shared EfficientNet convolutional features
        -> assessment-specific attention
        -> density-specific attention
        -> separate pooling + shared-style FC blocks
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

        self.features = backbone.features

        feature_channels = 1280

        # Task-specific attention modules.
        self.assessment_attention = ChannelSpatialAttention(
            channels=feature_channels,
            reduction=16,
            spatial_kernel_size=7,
        )

        self.density_attention = ChannelSpatialAttention(
            channels=feature_channels,
            reduction=16,
            spatial_kernel_size=7,
        )

        self.assessment_pool = nn.AdaptiveAvgPool2d(
            output_size=1
        )

        self.density_pool = nn.AdaptiveAvgPool2d(
            output_size=1
        )

        self.assessment_shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                feature_channels,
                256,
            ),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.density_shared = nn.Sequential(
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
        Freeze only the shared EfficientNet feature extractor.

        Both attention branches and both task heads remain trainable.
        """

        for parameter in self.features.parameters():
            parameter.requires_grad = False

    def unfreeze_backbone(self):
        """
        Unfreeze the shared EfficientNet feature extractor.
        """

        for parameter in self.features.parameters():
            parameter.requires_grad = True

    def forward(self, x):
        features = self.features(x)

        # ------------------------------------------------------
        # Assessment branch
        # ------------------------------------------------------

        assessment_features = self.assessment_attention(
            features
        )

        assessment_pooled = self.assessment_pool(
            assessment_features
        )

        assessment_shared = self.assessment_shared(
            assessment_pooled
        )

        assessment_logits = self.assessment_head(
            assessment_shared
        )

        # ------------------------------------------------------
        # Density branch
        # ------------------------------------------------------

        density_features = self.density_attention(
            features
        )

        density_pooled = self.density_pool(
            density_features
        )

        density_shared = self.density_shared(
            density_pooled
        )

        density_logits = self.density_head(
            density_shared
        )

        return {
            "assessment_logits": assessment_logits,
            "density_logits": density_logits,
        }