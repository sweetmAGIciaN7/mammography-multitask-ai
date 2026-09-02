from __future__ import annotations

import torch
from torch import nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class MultiTaskEfficientNetB0(nn.Module):
    """
    Multi-task baseline with a shared EfficientNet-B0 backbone.

    Outputs:
    - diagnostic_logits: binary diagnostic prediction
    - density_logits: 4-class breast density prediction
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

        self.assessment_head = nn.Linear(256, 5)
        self.density_head = nn.Linear(256, 4)
    

    def freeze_backbone(self):
        """Freeze all EfficientNet backbone parameters."""
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze all EfficientNet backbone parameters."""
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