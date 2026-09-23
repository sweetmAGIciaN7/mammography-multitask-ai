"""Two-stage training for task-specific dual-attention multi-task model."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.optim import AdamW

from src.data.loaders import make_dataloaders
from src.evaluation.metrics import evaluate_model
from src.models.multitask_dual_attention import (
    MultiTaskDualAttentionEfficientNetB0,
)
from src.training.losses import MultiTaskLoss


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
):
    model.train()
    running_loss = 0.0

    for batch in loader:
        images = batch["image"].to(device)
        assessment_target = batch["assessment_target"].to(device)
        density_target = batch["density_target"].to(device)

        optimizer.zero_grad()

        outputs = model(images)

        losses = criterion(
            outputs,
            assessment_target,
            density_target,
        )

        loss = losses["loss"]
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        running_loss += loss.detach().item()

    return running_loss / len(loader)


@torch.no_grad()
def validate_loss(
    model,
    loader,
    criterion,
    device,
):
    model.eval()
    running_loss = 0.0

    for batch in loader:
        images = batch["image"].to(device)
        assessment_target = batch["assessment_target"].to(device)
        density_target = batch["density_target"].to(device)

        outputs = model(images)

        losses = criterion(
            outputs,
            assessment_target,
            density_target,
        )

        running_loss += losses["loss"].detach().item()

    return running_loss / len(loader)


def run_stage(
    stage_name,
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    epochs,
    checkpoint_path,
    best_val_score,
):
    print()
    print("=" * 60)
    print(stage_name)
    print("=" * 60)

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        val_loss = validate_loss(
            model,
            val_loader,
            criterion,
            device,
        )

        val_metrics = evaluate_model(
            model,
            val_loader,
            device,
        )

        print()
        print(f"Epoch {epoch}/{epochs}")
        print(f"  train_loss: {train_loss:.4f}")
        print(f"  val_loss:   {val_loss:.4f}")

        scalar_metrics = {
            key: value
            for key, value in val_metrics.items()
            if not hasattr(value, "__len__")
        }

        for name, value in scalar_metrics.items():
            print(f"  {name}: {value:.4f}")

        current_score = (
            val_metrics["assessment_macro_f1"]
            + val_metrics["density_macro_f1"]
        ) / 2.0

        print(
            f"  combined_macro_f1: {current_score:.4f}"
        )

        if current_score > best_val_score:
            best_val_score = current_score

            torch.save(
                {
                    "stage": stage_name,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metrics": val_metrics,
                    "combined_macro_f1": current_score,
                },
                checkpoint_path,
            )

            print(
                f"  Saved best checkpoint: "
                f"{checkpoint_path}"
            )

    return best_val_score


def main():
    torch.manual_seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    train_loader, val_loader, test_loader = make_dataloaders(
        batch_size=16,
        image_size=224,
        seed=42,
    )

    model = MultiTaskDualAttentionEfficientNetB0(
        pretrained=True
    ).to(device)

    criterion = MultiTaskLoss(
        assessment_weight=1.0,
        density_weight=1.0,
        assessment_label_smoothing=0.05,
        density_label_smoothing=0.05,
        assessment_class_weights=[
            262 / (5 * 43),
            262 / (5 * 141),
            262 / (5 * 15),
            262 / (5 * 27),
            262 / (5 * 36),
        ],
    ).to(device)

    checkpoint_dir = Path(
        "results/checkpoints"
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = (
        checkpoint_dir
        / "best_multitask_dual_attention.pt"
    )

    best_val_score = -1.0

    # Stage 1: freeze shared EfficientNet backbone.
    model.freeze_backbone()

    optimizer = AdamW(
        filter(
            lambda p: p.requires_grad,
            model.parameters(),
        ),
        lr=1e-4,
        weight_decay=1e-2,
    )

    best_val_score = run_stage(
        stage_name="Stage 1 - Frozen backbone",
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=4,
        checkpoint_path=checkpoint_path,
        best_val_score=best_val_score,
    )

    # Stage 2: fine-tune full network.
    model.unfreeze_backbone()

    optimizer = AdamW(
        model.parameters(),
        lr=5e-5,
        weight_decay=5e-3,
    )

    best_val_score = run_stage(
        stage_name="Stage 2 - Fine-tuning",
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=3,
        checkpoint_path=checkpoint_path,
        best_val_score=best_val_score,
    )

    print()
    print("Training complete.")
    print(
        f"Best validation combined macro F1: "
        f"{best_val_score:.4f}"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    test_metrics = evaluate_model(
        model,
        test_loader,
        device,
    )

    print()
    print("Test metrics:")

    for name, value in test_metrics.items():
        if hasattr(value, "__len__"):
            continue

        print(f"  {name}: {value:.4f}")

    print()
    print(
        "Assessment F1 per class:",
        test_metrics["assessment_f1_per_class"],
    )

    print("Assessment confusion matrix:")
    print(
        test_metrics["assessment_confusion_matrix"]
    )

    print()
    print(
        "Density F1 per class:",
        test_metrics["density_f1_per_class"],
    )

    print("Density confusion matrix:")
    print(
        test_metrics["density_confusion_matrix"]
    )


if __name__ == "__main__":
    main()