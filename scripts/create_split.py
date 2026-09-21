from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]

METADATA_PATH = ROOT / "data" / "metadata.csv"
OUTPUT_PATH = ROOT / "data" / "metadata_split.csv"

RANDOM_SEED = 42


# ============================================================
# LOAD METADATA
# ============================================================

df = pd.read_csv(METADATA_PATH)

print("\n=== ORIGINAL DATASET ===")
print("Images:", len(df))

print("\nBI-RADS distribution:")
print(
    df["birads_class"]
    .value_counts()
    .sort_index()
)


# ============================================================
# TRAIN / TEMP SPLIT
#
# 70% train
# 30% temporary
# ============================================================

train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=RANDOM_SEED,
    stratify=df["birads_class"],
)


# ============================================================
# VALIDATION / TEST SPLIT
#
# Temporary set is split equally:
#
# 15% validation
# 15% test
# ============================================================

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=RANDOM_SEED,
    stratify=temp_df["birads_class"],
)


# ============================================================
# ADD SPLIT LABELS
# ============================================================

train_df = train_df.copy()
val_df = val_df.copy()
test_df = test_df.copy()

train_df["split"] = "train"
val_df["split"] = "val"
test_df["split"] = "test"


# ============================================================
# COMBINE
# ============================================================

split_df = pd.concat(
    [
        train_df,
        val_df,
        test_df,
    ],
    ignore_index=True,
)

# Restore deterministic ordering
split_df = split_df.sort_values(
    "image_id"
).reset_index(drop=True)


# ============================================================
# VALIDATION
# ============================================================

assert len(split_df) == len(df)

assert split_df["image_id"].nunique() == len(df)

assert set(split_df["split"]) == {
    "train",
    "val",
    "test",
}


# ============================================================
# SAVE
# ============================================================

split_df.to_csv(
    OUTPUT_PATH,
    index=False,
)


# ============================================================
# REPORT
# ============================================================

print("\n=== SPLIT CREATED ===")

for split_name in [
    "train",
    "val",
    "test",
]:

    subset = split_df[
        split_df["split"] == split_name
    ]

    print(
        f"\n--- {split_name.upper()} ---"
    )

    print(
        "Images:",
        len(subset),
    )

    print(
        "\nBI-RADS:"
    )

    print(
        subset["birads_class"]
        .value_counts()
        .sort_index()
    )

    print(
        "\nDensity:"
    )

    print(
        subset["density_acr"]
        .value_counts(dropna=False)
        .sort_index()
    )


print("\nOutput:")
print(OUTPUT_PATH)

print("\nSUCCESS")