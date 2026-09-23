from __future__ import annotations

from torch.utils.data import DataLoader, Subset

from src.data.dataset import InbreastDataset
from src.data.preprocessing import get_train_transform, get_eval_transform
from src.data.splits import make_stratified_splits


def make_dataloaders(
    root_dir="data/Inbreast",
    metadata_csv="data/metadata/INbreast.csv",
    batch_size=16,
    image_size=224,
    seed=42,
):
    """
    Build train, validation, and test DataLoaders.

    Training receives augmentation.
    Validation and test use deterministic preprocessing.
    """

    # Base dataset only used to create reproducible split indices.
    base_dataset = InbreastDataset(
        root_dir=root_dir,
        metadata_csv=metadata_csv,
    )

    train_idx, val_idx, test_idx = make_stratified_splits(
        base_dataset.samples,
        seed=seed,
    )

    # Separate dataset objects are necessary because train/eval transforms differ.
    train_dataset = InbreastDataset(
        root_dir=root_dir,
        metadata_csv=metadata_csv,
        transform=get_train_transform(image_size),
    )

    eval_dataset = InbreastDataset(
        root_dir=root_dir,
        metadata_csv=metadata_csv,
        transform=get_eval_transform(image_size),
    )

    train_subset = Subset(train_dataset, train_idx)
    val_subset = Subset(eval_dataset, val_idx)
    test_subset = Subset(eval_dataset, test_idx)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, val_loader, test_loader