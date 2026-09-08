"""Compare saved multi-task models on the same test split."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from src.data.loaders import make_dataloaders
from src.evaluation.metrics import evaluate_model
from src.evaluation.ordinal_metrics import evaluate_ordinal_model
from src.models.multitask_attention import MultiTaskAttentionEfficientNetB0
from src.models.multitask_baseline import MultiTaskEfficientNetB0
from src.models.multitask_dual_attention import (
    MultiTaskDualAttentionEfficientNetB0,
)
from src.models.multitask_ordinal import MultiTaskOrdinalEfficientNetB0


def evaluate_flat_model(model_class, checkpoint_path, device, test_loader):
    model = model_class(pretrained=False).to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    metrics = evaluate_model(model, test_loader, device)

    return {
        "assessment_accuracy": metrics["assessment_accuracy"],
        "assessment_balanced_accuracy": metrics["assessment_balanced_accuracy"],
        "assessment_macro_f1": metrics["assessment_macro_f1"],
        "assessment_mae": metrics["assessment_mae"],
        "density_accuracy": metrics["density_accuracy"],
        "density_balanced_accuracy": metrics["density_balanced_accuracy"],
        "density_macro_f1": metrics["density_macro_f1"],
    }


def evaluate_ordinal(checkpoint_path, device, test_loader):
    model = MultiTaskOrdinalEfficientNetB0(pretrained=False).to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    metrics = evaluate_ordinal_model(model, test_loader, device)

    return {
        "assessment_accuracy": metrics["assessment_accuracy"],
        "assessment_balanced_accuracy": metrics["assessment_balanced_accuracy"],
        "assessment_macro_f1": metrics["assessment_macro_f1"],
        "assessment_mae": metrics["assessment_mae"],
        "assessment_non_monotonic_count": metrics[
            "assessment_non_monotonic_count"
        ],
        "assessment_non_monotonic_rate": metrics[
            "assessment_non_monotonic_rate"
        ],
        "density_accuracy": metrics["density_accuracy"],
        "density_balanced_accuracy": metrics["density_balanced_accuracy"],
        "density_macro_f1": metrics["density_macro_f1"],
    } 


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    _, _, test_loader = make_dataloaders(
        batch_size=16,
        image_size=224,
        seed=42,
    )

    results = []

    experiments = [
        (
            "weighted_flat_baseline",
            "results/checkpoints/baseline_weighted_multitask.pt",
            "flat",
            MultiTaskEfficientNetB0,
        ),
        (
            "weighted_ordinal",
            "results/checkpoints/ordinal_weighted_thresholds.pt",
            "ordinal",
            None,
        ),
        (
            "shared_attention",
            "results/checkpoints/attention_shared_weighted.pt",
            "flat",
            MultiTaskAttentionEfficientNetB0,
        ),
        (
            "dual_attention",
            "results/checkpoints/dual_attention_weighted.pt",
            "flat",
            MultiTaskDualAttentionEfficientNetB0,
        ),
    ]

    for name, checkpoint_path, model_type, model_class in experiments:
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.is_file():
            print(f"Skipping {name}: checkpoint not found -> {checkpoint_path}")
            continue

        print(f"Evaluating: {name}")

        if model_type == "ordinal":
            metrics = evaluate_ordinal(
                checkpoint_path,
                device,
                test_loader,
            )
        else:
            metrics = evaluate_flat_model(
                model_class,
                checkpoint_path,
                device,
                test_loader,
            )

        row = {"model": name}
        row.update(metrics)
        results.append(row)

    df = pd.DataFrame(results)

    output_dir = Path("results/tables")
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "model_comparison.csv"
    df.to_csv(csv_path, index=False)

    print()
    print("Model comparison:")
    print(df.to_string(index=False))

    print()
    print("Saved:", csv_path)


if __name__ == "__main__":
    main()