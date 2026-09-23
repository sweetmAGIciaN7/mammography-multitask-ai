from __future__ import annotations

from collections import Counter
from typing import Sequence

from sklearn.model_selection import train_test_split


def make_stratified_splits(
    samples: Sequence[dict],
    seed: int = 42,
):
    """
    Create reproducible image-level train/val/test splits.

    Stratification is based on the 5-class BI-RADS assessment target.

    Split proportions:
        train: 64%
        val:   16%
        test:  20%
    """

    indices = list(range(len(samples)))

    assessment_labels = [
        sample["assessment_target"]
        for sample in samples
    ]

    train_idx, temp_idx = train_test_split(
        indices,
        test_size=0.36,
        random_state=seed,
        stratify=assessment_labels,
    )

    temp_labels = [
        samples[i]["assessment_target"]
        for i in temp_idx
    ]

    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=20 / 36,
        random_state=seed,
        stratify=temp_labels,
    )

    return train_idx, val_idx, test_idx


def summarize_split(
    name: str,
    indices: Sequence[int],
    samples: Sequence[dict],
):
    assessment = Counter(
        samples[i]["assessment_target"]
        for i in indices
    )

    density = Counter(
        samples[i]["density_target"]
        for i in indices
    )

    print(f"{name}: {len(indices)}")
    print(f"  Assessment: {assessment}")
    print(f"  Density:    {density}")