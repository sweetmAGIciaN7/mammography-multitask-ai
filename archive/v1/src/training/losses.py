"""Loss functions for multi-task INbreast training."""

from __future__ import annotations

import torch
from torch import nn


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss for:

    - BI-RADS assessment: 5 classes
    - Breast density: 4 classes

    Density target == -1 is ignored.
    """

    def __init__(
        self,
        assessment_weight: float = 1.0,
        density_weight: float = 1.0,
        assessment_label_smoothing: float = 0.05,
        density_label_smoothing: float = 0.05,
        assessment_class_weights=None,
    ) -> None:
        super().__init__()

        self.assessment_weight = assessment_weight
        self.density_weight = density_weight

        if assessment_class_weights is not None:
            assessment_class_weights = torch.tensor(
                assessment_class_weights,
                dtype=torch.float32,
            )

        self.register_buffer(
            "assessment_class_weights",
            assessment_class_weights,
        )

        self.assessment_label_smoothing = (
            assessment_label_smoothing
        )

        self.density_loss_fn = nn.CrossEntropyLoss(
            label_smoothing=density_label_smoothing
        )

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        assessment_target: torch.Tensor,
        density_target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:

        assessment_logits = outputs["assessment_logits"]
        density_logits = outputs["density_logits"]

        assessment_loss = nn.functional.cross_entropy(
            assessment_logits,
            assessment_target.long(),
            weight=self.assessment_class_weights,
            label_smoothing=self.assessment_label_smoothing,
        )

        density_mask = density_target != -1

        if density_mask.any():
            density_loss = self.density_loss_fn(
                density_logits[density_mask],
                density_target[density_mask].long(),
            )
        else:
            density_loss = density_logits.sum() * 0.0

        total_loss = (
            self.assessment_weight * assessment_loss
            + self.density_weight * density_loss
        )

        return {
            "loss": total_loss,
            "assessment_loss": assessment_loss,
            "density_loss": density_loss,
        }