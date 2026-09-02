"""Ordinal multi-task loss for BI-RADS assessment + breast density."""

from __future__ import annotations

import torch
from torch import nn


class OrdinalMultiTaskLoss(nn.Module):
    """
    Weighted ordinal loss for 5 ordered BI-RADS assessment classes.

    Threshold targets:
        class 0 -> [0, 0, 0, 0]
        class 1 -> [1, 0, 0, 0]
        class 2 -> [1, 1, 0, 0]
        class 3 -> [1, 1, 1, 0]
        class 4 -> [1, 1, 1, 1]

    Density remains standard 4-class classification.
    """

    def __init__(
        self,
        assessment_weight: float = 1.0,
        density_weight: float = 1.0,
        density_label_smoothing: float = 0.05,
        threshold_pos_weights=None,
    ) -> None:
        super().__init__()

        self.assessment_weight = assessment_weight
        self.density_weight = density_weight

        if threshold_pos_weights is None:
            threshold_pos_weights = [
                0.1963,
                2.3590,
                3.1587,
                6.2778,
            ]

        self.register_buffer(
            "threshold_pos_weights",
            torch.tensor(
                threshold_pos_weights,
                dtype=torch.float32,
            ),
        )

        self.density_loss_fn = nn.CrossEntropyLoss(
            label_smoothing=density_label_smoothing
        )

    @staticmethod
    def make_ordinal_targets(
        assessment_target: torch.Tensor,
    ) -> torch.Tensor:

        thresholds = torch.arange(
            4,
            device=assessment_target.device,
        )

        return (
            assessment_target.unsqueeze(1) > thresholds
        ).float()

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        assessment_target: torch.Tensor,
        density_target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:

        assessment_logits = outputs["assessment_logits"]

        density_logits = outputs["density_logits"]

        ordinal_targets = self.make_ordinal_targets(
            assessment_target
        )

        assessment_loss = nn.functional.binary_cross_entropy_with_logits(
            assessment_logits,
            ordinal_targets,
            pos_weight=self.threshold_pos_weights,
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