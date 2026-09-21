from pathlib import Path
import sys
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    f1_score,
)

from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch.utils.data import (
    DataLoader,
    WeightedRandomSampler,
)

from torchvision import transforms


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import InbreastMultiTaskDataset

from src.models.attention_multitask_resnet import (
    AttentionMultiTaskResNet18
)


METADATA_PATH = (
    ROOT / "data" / "metadata_split.csv"
)

RESULTS_DIR = (
    ROOT / "results" / "attention"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CHECKPOINT_PATH = (
    RESULTS_DIR / "best_model.pt"
)

METRICS_PATH = (
    RESULTS_DIR / "metrics.json"
)


IMAGE_SIZE = 384
BATCH_SIZE = 8
MAX_EPOCHS = 30

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

DENSITY_LOSS_WEIGHT = 1.0

EARLY_STOPPING_PATIENCE = 7

NUM_WORKERS = 0
RANDOM_SEED = 42


torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(
        RANDOM_SEED
    )


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("\n=== DEVICE ===")
print(device)


df = pd.read_csv(
    METADATA_PATH
)

train_df = df[
    df["split"] == "train"
].reset_index(drop=True)

val_df = df[
    df["split"] == "val"
].reset_index(drop=True)

test_df = df[
    df["split"] == "test"
].reset_index(drop=True)


print("\n=== SPLITS ===")
print("Train:", len(train_df))
print("Val:", len(val_df))
print("Test:", len(test_df))


train_transform = transforms.Compose(
    [
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),

        transforms.RandomHorizontalFlip(
            p=0.5
        ),

        transforms.RandomRotation(
            degrees=7
        ),

        transforms.RandomAffine(
            degrees=0,
            translate=(0.03, 0.03),
            scale=(0.97, 1.03),
        ),

        transforms.Grayscale(
            num_output_channels=3
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406,
            ],
            std=[
                0.229,
                0.224,
                0.225,
            ],
        ),
    ]
)


eval_transform = transforms.Compose(
    [
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),

        transforms.Grayscale(
            num_output_channels=3
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406,
            ],
            std=[
                0.229,
                0.224,
                0.225,
            ],
        ),
    ]
)


train_dataset = InbreastMultiTaskDataset(
    metadata_csv=METADATA_PATH,
    dataframe=train_df,
    transform=train_transform,
)

val_dataset = InbreastMultiTaskDataset(
    metadata_csv=METADATA_PATH,
    dataframe=val_df,
    transform=eval_transform,
)

test_dataset = InbreastMultiTaskDataset(
    metadata_csv=METADATA_PATH,
    dataframe=test_df,
    transform=eval_transform,
)


class_counts = (
    train_df["birads_class"]
    .value_counts()
    .to_dict()
)

sample_weights = [
    1.0 / class_counts[label]
    for label in train_df["birads_class"]
]

sample_weights = torch.DoubleTensor(
    sample_weights
)

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True,
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    num_workers=NUM_WORKERS,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
)


def calculate_class_weights(
    series,
    num_classes,
):
    counts = (
        series
        .value_counts()
        .sort_index()
    )

    total = counts.sum()

    weights = []

    for label in range(
        1,
        num_classes + 1,
    ):
        count = counts.get(
            label,
            0,
        )

        if count == 0:
            weight = 0.0
        else:
            weight = (
                total
                / (
                    num_classes
                    * count
                )
            )

        weights.append(weight)

    return torch.tensor(
        weights,
        dtype=torch.float32,
    )


birads_weights = calculate_class_weights(
    train_df["birads_class"],
    5,
).to(device)

density_weights = calculate_class_weights(
    train_df["density_acr"].dropna(),
    4,
).to(device)


print("\n=== CLASS WEIGHTS ===")

print(
    "BI-RADS:",
    birads_weights.cpu().numpy()
)

print(
    "Density:",
    density_weights.cpu().numpy()
)


model = AttentionMultiTaskResNet18(
    pretrained=True
).to(device)


birads_criterion = nn.CrossEntropyLoss(
    weight=birads_weights
)

density_criterion = nn.CrossEntropyLoss(
    weight=density_weights,
    reduction="none",
)


optimizer = AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

scheduler = ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2,
)


def compute_loss(
    outputs,
    batch,
):
    birads_target = (
        batch["birads"]
        .to(device)
    )

    density_target = (
        batch["density"]
        .to(device)
    )

    density_mask = (
        batch["density_mask"]
        .to(device)
    )


    birads_loss = birads_criterion(
        outputs["birads"],
        birads_target,
    )


    density_loss_raw = density_criterion(
        outputs["density"],
        density_target,
    )

    valid_count = (
        density_mask.sum()
    )

    if valid_count > 0:
        density_loss = (
            density_loss_raw
            * density_mask
        ).sum() / valid_count
    else:
        density_loss = torch.tensor(
            0.0,
            device=device,
        )


    total_loss = (
        birads_loss
        +
        DENSITY_LOSS_WEIGHT
        * density_loss
    )

    return total_loss


def train_one_epoch():
    model.train()

    total_loss_sum = 0.0

    for batch in train_loader:
        images = (
            batch["image"]
            .to(device)
        )

        optimizer.zero_grad()

        outputs = model(
            images
        )

        loss = compute_loss(
            outputs,
            batch,
        )

        loss.backward()

        optimizer.step()

        total_loss_sum += (
            loss.item()
            * images.size(0)
        )

    return (
        total_loss_sum
        / len(train_loader.dataset)
    )


def evaluate(loader):
    model.eval()

    total_loss = 0.0

    birads_true = []
    birads_pred = []

    density_true = []
    density_pred = []


    with torch.no_grad():

        for batch in loader:

            images = (
                batch["image"]
                .to(device)
            )

            outputs = model(
                images
            )

            loss = compute_loss(
                outputs,
                batch,
            )

            total_loss += (
                loss.item()
                * images.size(0)
            )


            birads_predictions = (
                outputs["birads"]
                .argmax(dim=1)
                .cpu()
                .numpy()
            )

            birads_targets = (
                batch["birads"]
                .numpy()
            )

            birads_true.extend(
                birads_targets.tolist()
            )

            birads_pred.extend(
                birads_predictions.tolist()
            )


            density_predictions = (
                outputs["density"]
                .argmax(dim=1)
                .cpu()
                .numpy()
            )

            density_targets = (
                batch["density"]
                .numpy()
            )

            masks = (
                batch["density_mask"]
                .numpy()
            )


            for target, pred, mask in zip(
                density_targets,
                density_predictions,
                masks,
            ):
                if mask == 1:
                    density_true.append(
                        int(target)
                    )

                    density_pred.append(
                        int(pred)
                    )


    return {
        "loss":
            total_loss
            / len(loader.dataset),

        "birads_accuracy":
            accuracy_score(
                birads_true,
                birads_pred,
            ),

        "birads_macro_f1":
            f1_score(
                birads_true,
                birads_pred,
                average="macro",
                zero_division=0,
            ),

        "density_accuracy":
            accuracy_score(
                density_true,
                density_pred,
            ),

        "density_macro_f1":
            f1_score(
                density_true,
                density_pred,
                average="macro",
                zero_division=0,
            ),
    }


best_score = -1.0

epochs_without_improvement = 0

history = []


print("\n=== TRAINING ATTENTION MODEL ===")


for epoch in range(
    1,
    MAX_EPOCHS + 1,
):

    train_loss = train_one_epoch()

    val_metrics = evaluate(
        val_loader
    )


    validation_score = (
        val_metrics["birads_macro_f1"]
        +
        val_metrics["density_macro_f1"]
    ) / 2


    scheduler.step(
        validation_score
    )


    current_lr = (
        optimizer.param_groups[0]["lr"]
    )


    print(
        f"\nEpoch {epoch:02d}/{MAX_EPOCHS}"
    )

    print(
        f"Train loss: {train_loss:.4f}"
    )

    print(
        f"Val loss: {val_metrics['loss']:.4f}"
    )

    print(
        f"BI-RADS accuracy: "
        f"{val_metrics['birads_accuracy']:.4f}"
    )

    print(
        f"BI-RADS macro F1: "
        f"{val_metrics['birads_macro_f1']:.4f}"
    )

    print(
        f"Density accuracy: "
        f"{val_metrics['density_accuracy']:.4f}"
    )

    print(
        f"Density macro F1: "
        f"{val_metrics['density_macro_f1']:.4f}"
    )

    print(
        f"Learning rate: "
        f"{current_lr:.8f}"
    )


    history.append(
        {
            "epoch": epoch,
            "train_loss": train_loss,
            "learning_rate": current_lr,

            **{
                f"val_{key}": value
                for key, value
                in val_metrics.items()
            },
        }
    )


    if validation_score > best_score:
        best_score = validation_score

        epochs_without_improvement = 0

        torch.save(
            {
                "epoch": epoch,

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "validation_score":
                    validation_score,

                "validation_metrics":
                    val_metrics,
            },

            CHECKPOINT_PATH,
        )

        print(
            "Saved new best model."
        )

    else:
        epochs_without_improvement += 1


    if (
        epochs_without_improvement
        >= EARLY_STOPPING_PATIENCE
    ):
        print(
            "\nEarly stopping triggered."
        )

        break


checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)


print("\n=== BEST MODEL ===")
print(
    "Best epoch:",
    checkpoint["epoch"]
)


test_metrics = evaluate(
    test_loader
)


print("\n=== TEST RESULTS ===")

for key, value in test_metrics.items():
    print(
        f"{key}: {value:.4f}"
    )


results = {
    "model":
        "AttentionMultiTaskResNet18",

    "config": {
        "image_size":
            IMAGE_SIZE,

        "batch_size":
            BATCH_SIZE,

        "max_epochs":
            MAX_EPOCHS,

        "learning_rate":
            LEARNING_RATE,

        "weight_decay":
            WEIGHT_DECAY,

        "density_loss_weight":
            DENSITY_LOSS_WEIGHT,

        "early_stopping_patience":
            EARLY_STOPPING_PATIENCE,

        "random_seed":
            RANDOM_SEED,
    },

    "best_epoch":
        checkpoint["epoch"],

    "best_validation_metrics":
        checkpoint["validation_metrics"],

    "test_metrics":
        test_metrics,

    "history":
        history,
}


with open(
    METRICS_PATH,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        results,
        file,
        indent=2,
    )


print("\nSaved checkpoint:")
print(CHECKPOINT_PATH)

print("\nSaved metrics:")
print(METRICS_PATH)

print("\nSUCCESS")