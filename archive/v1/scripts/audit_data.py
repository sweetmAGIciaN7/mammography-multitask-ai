from collections import Counter
from pathlib import Path

import pandas as pd

from src.data.dataset import InbreastDataset
from src.data.splits import make_stratified_splits


ROOT = Path("data/Inbreast")
METADATA = Path("data/metadata/INbreast.csv")


dataset = InbreastDataset(
    root_dir=ROOT,
    metadata_csv=METADATA,
)

samples = dataset.samples

print("=" * 60)
print("DATASET INTEGRITY AUDIT")
print("=" * 60)

# --------------------------------------------------
# 1. Basic dataset size
# --------------------------------------------------

print("\n[1] DATASET SIZE")

print("Samples:", len(samples))

image_ids = [sample["image_id"] for sample in samples]
image_paths = [str(sample["image_path"].resolve()) for sample in samples]

print("Unique image IDs:", len(set(image_ids)))
print("Unique image paths:", len(set(image_paths)))


# --------------------------------------------------
# 2. Duplicate checks
# --------------------------------------------------

print("\n[2] DUPLICATE CHECK")

duplicate_ids = [
    image_id
    for image_id, count in Counter(image_ids).items()
    if count > 1
]

duplicate_paths = [
    path
    for path, count in Counter(image_paths).items()
    if count > 1
]

print("Duplicate image IDs:", duplicate_ids)
print("Duplicate image paths:", duplicate_paths)


# --------------------------------------------------
# 3. Metadata ↔ PNG correspondence
# --------------------------------------------------

print("\n[3] METADATA / IMAGE-ID JOIN")

metadata = pd.read_csv(
    METADATA,
    sep=";",
    dtype=str,
)

metadata_ids = set(
    metadata["File Name"]
    .fillna("")
    .str.strip()
)

png_ids = set(image_ids)

missing_metadata = sorted(png_ids - metadata_ids)
metadata_without_png = sorted(metadata_ids - png_ids)

print("PNG IDs:", len(png_ids))
print("Metadata IDs:", len(metadata_ids))

print("PNG IDs without metadata:", missing_metadata)
print("Metadata IDs without PNG:", metadata_without_png)


# --------------------------------------------------
# 4. Target validity
# --------------------------------------------------

print("\n[4] TARGET VALIDITY")

assessment_targets = [
    sample["assessment_target"]
    for sample in samples
]

density_targets = [
    sample["density_target"]
    for sample in samples
]

print(
    "Assessment classes:",
    sorted(set(assessment_targets)),
)

print(
    "Density classes:",
    sorted(set(density_targets)),
)

invalid_assessment = sorted(
    set(assessment_targets) - {0, 1, 2, 3, 4}
)

invalid_density = sorted(
    set(density_targets) - {-1, 0, 1, 2, 3}
)

print("Invalid assessment targets:", invalid_assessment)
print("Invalid density targets:", invalid_density)

print(
    "Missing density labels (-1):",
    density_targets.count(-1),
)


# --------------------------------------------------
# 5. Split
# --------------------------------------------------

print("\n[5] SPLIT")

train_idx, val_idx, test_idx = make_stratified_splits(
    samples,
    seed=42,
)

print("Train:", len(train_idx))
print("Val:", len(val_idx))
print("Test:", len(test_idx))
print(
    "Total:",
    len(train_idx) + len(val_idx) + len(test_idx),
)


# --------------------------------------------------
# 6. Overlap
# --------------------------------------------------

print("\n[6] SPLIT OVERLAP")

train_set = set(train_idx)
val_set = set(val_idx)
test_set = set(test_idx)

print("Train ∩ Val:", len(train_set & val_set))
print("Train ∩ Test:", len(train_set & test_set))
print("Val ∩ Test:", len(val_set & test_set))

all_split_indices = train_set | val_set | test_set

print(
    "Unique indices across all splits:",
    len(all_split_indices),
)

missing_indices = (
    set(range(len(samples)))
    - all_split_indices
)

print("Missing dataset indices:", sorted(missing_indices))


# --------------------------------------------------
# 7. Class distributions
# --------------------------------------------------

print("\n[7] CLASS DISTRIBUTIONS")


def print_distribution(name, indices):
    assessment = Counter(
        samples[i]["assessment_target"]
        for i in indices
    )

    density = Counter(
        samples[i]["density_target"]
        for i in indices
    )

    print(f"\n{name}")
    print("Assessment:", dict(sorted(assessment.items())))
    print("Density:", dict(sorted(density.items())))


print_distribution("TRAIN", train_idx)
print_distribution("VAL", val_idx)
print_distribution("TEST", test_idx)


# --------------------------------------------------
# Final assertions
# --------------------------------------------------

print("\n[8] HARD ASSERTIONS")

assert len(samples) == 410
assert len(set(image_ids)) == 410
assert len(set(image_paths)) == 410

assert not duplicate_ids
assert not duplicate_paths

assert not missing_metadata

assert set(assessment_targets) <= {0, 1, 2, 3, 4}
assert set(density_targets) <= {-1, 0, 1, 2, 3}

assert len(train_idx) == 262
assert len(val_idx) == 65
assert len(test_idx) == 83

assert not (train_set & val_set)
assert not (train_set & test_set)
assert not (val_set & test_set)

assert len(all_split_indices) == 410

print("\nALL HARD DATA/SPLIT CHECKS PASSED.")