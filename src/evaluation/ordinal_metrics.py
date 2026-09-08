"""Evaluation metrics for cumulative-threshold BI-RADS assessment + density."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_recall_fscore_support,
)


@torch.no_grad()
def evaluate_ordinal_model(model, loader, device):
    model.eval()

    assessment_targets = []
    assessment_predictions = []

    density_targets = []
    density_predictions = []

    non_monotonic_count = 0
    total_ordinal_predictions = 0

    for batch in loader:
        images = batch["image"].to(device)

        outputs = model(images)

        # ------------------------------------------------------
        # Ordinal assessment
        # ------------------------------------------------------

        assessment_probs = torch.sigmoid(
            outputs["assessment_logits"]
        )

        threshold_passes = assessment_probs > 0.5

        non_monotonic = (
            threshold_passes[:, 1:].int()
            > threshold_passes[:, :-1].int()
        ).any(dim=1)

        non_monotonic_count += int(non_monotonic.sum().item())
        total_ordinal_predictions += int(non_monotonic.numel())

        passed_thresholds = (
            assessment_probs > 0.5
        ).sum(dim=1)

        assessment_targets.extend(
            batch["assessment_target"].numpy().tolist()
        )

        assessment_predictions.extend(
            passed_thresholds.cpu().numpy().tolist()
        )

        # ------------------------------------------------------
        # Density
        # ------------------------------------------------------

        density_target = batch["density_target"]
        density_mask = density_target != -1

        if density_mask.any():
            density_pred = (
                outputs["density_logits"]
                .cpu()
                .argmax(dim=1)
            )

            density_targets.extend(
                density_target[density_mask].numpy().tolist()
            )

            density_predictions.extend(
                density_pred[density_mask].numpy().tolist()
            )

    assessment_targets = np.asarray(
        assessment_targets
    )

    assessment_predictions = np.asarray(
        assessment_predictions
    )

    density_targets = np.asarray(
        density_targets
    )

    density_predictions = np.asarray(
        density_predictions
    )

    assessment_precision, assessment_recall, assessment_f1, _ = (
        precision_recall_fscore_support(
            assessment_targets,
            assessment_predictions,
            labels=[0, 1, 2, 3, 4],
            zero_division=0,
        )
    )

    assessment_cm = confusion_matrix(
        assessment_targets,
        assessment_predictions,
        labels=[0, 1, 2, 3, 4],
    )

    density_precision, density_recall, density_f1, _ = (
        precision_recall_fscore_support(
            density_targets,
            density_predictions,
            labels=[0, 1, 2, 3],
            zero_division=0,
        )
    )

    density_cm = confusion_matrix(
        density_targets,
        density_predictions,
        labels=[0, 1, 2, 3],
    )

    return {
        "assessment_non_monotonic_count": non_monotonic_count,
        "assessment_non_monotonic_rate": (
            non_monotonic_count / total_ordinal_predictions
            if total_ordinal_predictions > 0
            else 0.0
        ),    
        "assessment_accuracy": accuracy_score(
            assessment_targets,
            assessment_predictions,
        ),
        "assessment_balanced_accuracy": balanced_accuracy_score(
            assessment_targets,
            assessment_predictions,
        ),
        "assessment_macro_f1": f1_score(
            assessment_targets,
            assessment_predictions,
            average="macro",
            zero_division=0,
        ),
        "assessment_mae": mean_absolute_error(
            assessment_targets,
            assessment_predictions,
        ),
        "assessment_precision_per_class": assessment_precision,
        "assessment_recall_per_class": assessment_recall,
        "assessment_f1_per_class": assessment_f1,
        "assessment_confusion_matrix": assessment_cm,

        "density_accuracy": accuracy_score(
            density_targets,
            density_predictions,
        ),
        "density_balanced_accuracy": balanced_accuracy_score(
            density_targets,
            density_predictions,
        ),
        "density_macro_f1": f1_score(
            density_targets,
            density_predictions,
            average="macro",
            zero_division=0,
        ),
        "density_precision_per_class": density_precision,
        "density_recall_per_class": density_recall,
        "density_f1_per_class": density_f1,
        "density_confusion_matrix": density_cm,
    }