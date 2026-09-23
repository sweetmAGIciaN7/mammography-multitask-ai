from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

XLS_PATH = ROOT / "data" / "INbreast.xls"
IMAGE_ROOT = ROOT / "data" / "Inbreast"
OUTPUT_PATH = ROOT / "data" / "metadata.csv"


def normalize_birads(value):
    """
    Convert original INbreast BI-RADS labels into the 5-class
    representation used by the local PNG dataset.

    Original:
        1, 2, 3, 4a, 4b, 4c, 5, 6

    Training classes:
        1, 2, 3, 4, 5

    Mapping:
        4a/4b/4c -> 4
        6 -> 5
    """

    if pd.isna(value):
        return pd.NA

    value = str(value).strip().lower()

    if value in {"1", "1.0"}:
        return 1

    if value in {"2", "2.0"}:
        return 2

    if value in {"3", "3.0"}:
        return 3

    if value.startswith("4"):
        return 4

    if value in {"5", "5.0"}:
        return 5

    if value in {"6", "6.0"}:
        return 5

    return pd.NA


def clean_acr(value):
    """
    Convert ACR breast density to integer 1-4.

    Invalid or missing values remain missing.
    """

    if pd.isna(value):
        return pd.NA

    value = str(value).strip()

    if value in {"1", "2", "3", "4"}:
        return int(value)

    if value in {"1.0", "2.0", "3.0", "4.0"}:
        return int(float(value))

    return pd.NA


# ============================================================
# READ ORIGINAL INBREAST METADATA
# ============================================================

df = pd.read_excel(XLS_PATH)

df["image_id"] = (
    pd.to_numeric(
        df["File Name"],
        errors="coerce",
    )
    .astype("Int64")
    .astype("string")
)

# Ignore the two spreadsheet rows that do not correspond
# to mammogram image files.
df = df[df["image_id"].notna()].copy()

if len(df) != 410:
    raise RuntimeError(
        f"Expected 410 metadata rows with File Name, found {len(df)}."
    )

if df["image_id"].duplicated().any():
    raise RuntimeError(
        "Duplicate image IDs found in INbreast metadata."
    )


# ============================================================
# FIND LOCAL PNG IMAGES
# ============================================================

records = []

for image_path in sorted(IMAGE_ROOT.rglob("*.png")):

    folder = image_path.parent.name.lower()

    if not folder.startswith("birads"):
        continue

    birads_folder = int(
        folder.replace("birads", "")
    )

    records.append(
        {
            "image_id": image_path.stem,

            # POSIX path makes metadata portable across
            # Windows/Linux/macOS.
            "image_path": (
                image_path
                .relative_to(ROOT)
                .as_posix()
            ),

            "birads_folder": birads_folder,
        }
    )


images = pd.DataFrame(records)

if len(images) != 410:
    raise RuntimeError(
        f"Expected 410 PNGs, found {len(images)}."
    )

if images["image_id"].duplicated().any():
    raise RuntimeError(
        "Duplicate PNG image IDs found."
    )


# ============================================================
# PREPARE ORIGINAL METADATA
# ============================================================

metadata = df[
    [
        "image_id",
        "Laterality",
        "View",
        "Acquisition date",
        "ACR",
        "Bi-Rads",
    ]
].copy()

metadata = metadata.rename(
    columns={
        "Laterality": "laterality",
        "View": "view",
        "Acquisition date": "acquisition_date",
        "ACR": "acr_raw",
        "Bi-Rads": "birads_original",
    }
)


# ============================================================
# MERGE LOCAL FILES WITH ORIGINAL METADATA
# ============================================================

merged = images.merge(
    metadata,
    on="image_id",
    how="left",
    validate="one_to_one",
)


# ============================================================
# CLEAN LABELS
# ============================================================

merged["birads_class"] = (
    merged["birads_original"]
    .apply(normalize_birads)
    .astype("Int64")
)

merged["density_acr"] = (
    merged["acr_raw"]
    .apply(clean_acr)
    .astype("Int64")
)


# ============================================================
# VERIFY BI-RADS AGAINST KAGGLE FOLDER LABELS
# ============================================================

birads_mismatch = merged[
    merged["birads_class"]
    != merged["birads_folder"]
]

if len(birads_mismatch):
    raise RuntimeError(
        "BI-RADS mismatch detected between "
        "folder labels and original metadata:\n"
        + birads_mismatch[
            [
                "image_id",
                "birads_folder",
                "birads_original",
                "birads_class",
            ]
        ].to_string(index=False)
    )


# ============================================================
# FINAL TABLE
# ============================================================

final = merged[
    [
        "image_id",
        "image_path",
        "birads_original",
        "birads_class",
        "density_acr",
        "laterality",
        "view",
        "acquisition_date",
    ]
].copy()


# ============================================================
# SAVE
# ============================================================

final.to_csv(
    OUTPUT_PATH,
    index=False,
)


# ============================================================
# SUMMARY
# ============================================================

print("\n=== METADATA BUILD COMPLETE ===")

print("Output:", OUTPUT_PATH)

print("\nImages:")
print(len(final))

print("\nUnique image IDs:")
print(final["image_id"].nunique())

print("\nBI-RADS usable:")
print(final["birads_class"].notna().sum())

print("\nDensity usable:")
print(final["density_acr"].notna().sum())

print("\nMissing density:")
print(final["density_acr"].isna().sum())

print("\nBI-RADS training distribution:")
print(
    final["birads_class"]
    .value_counts()
    .sort_index()
)

print("\nOriginal BI-RADS distribution:")
print(
    final["birads_original"]
    .value_counts()
)

print("\nDensity distribution:")
print(
    final["density_acr"]
    .value_counts(dropna=False)
    .sort_index()
)

print("\nView distribution:")
print(
    final["view"]
    .value_counts(dropna=False)
)

print("\nLaterality distribution:")
print(
    final["laterality"]
    .value_counts(dropna=False)
)

print("\nFirst 10 rows:")
print(
    final.head(10)
    .to_string(index=False)
)

print("\nSUCCESS")