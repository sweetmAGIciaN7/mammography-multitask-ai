"""Ordinal multi-task model for BI-RADS assessment + breast density."""

from __future__ import annotations

from torch import nn
from torchvision.models import (
    EfficientNet_B0_Weights,
    efficientnet_b0,
)


class MultiTaskOrdinalEfficientNetB0(nn.Module):
    """
    Multi-task EfficientNet-B0.

    Outputs:
    - assessment_logits: 4 ordinal threshold logits
    - density_logits: 4-class density logits
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()

        weights = (
            EfficientNet_B0_Weights.DEFAULT
            if pretrained
            else None
        )

        backbone = efficientnet_b0(weights=weights)

        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()

        self.backbone = backbone

        self.shared = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # Four thresholds for five ordered assessment classes.
        self.assessment_head = nn.Linear(256, 4)

        # Four breast-density classes.
        self.density_head = nn.Linear(256, 4)

    def freeze_backbone(self):
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

    def unfreeze_backbone(self):
        for parameter in self.backbone.parameters():
            parameter.requires_grad = True

    def forward(self, x):
        features = self.backbone(x)
        shared_features = self.shared(features)

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