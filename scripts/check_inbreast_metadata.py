from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parents[1]
xls_path = root / "data" / "INbreast.xls"
image_root = root / "data" / "Inbreast"

# =========================
# READ METADATA
# =========================

df = pd.read_excel(xls_path)

df["image_id"] = (
    pd.to_numeric(df["File Name"], errors="coerce")
    .astype("Int64")
    .astype("string")
)

print("\n=== METADATA ===")
print("Rows:", len(df))
print("Non-null File Names:", df["image_id"].notna().sum())
print("Unique File Names:", df["image_id"].nunique())

# =========================
# LOCAL PNG DATASET
# =========================

records = []

for p in sorted(image_root.rglob("*.png")):
    folder = p.parent.name.lower()

    if not folder.startswith("birads"):
        continue

    birads_folder = int(folder.replace("birads", ""))

    records.append(
        {
            "image_id": p.stem,
            "image_path": str(p.relative_to(root)),
            "birads_folder": birads_folder,
        }
    )

png_df = pd.DataFrame(records)

print("\n=== PNG DATASET ===")
print("PNG files:", len(png_df))
print("Unique PNG IDs:", png_df["image_id"].nunique())

# =========================
# ID MATCHING
# =========================

png_ids = set(png_df["image_id"])
metadata_ids = set(df["image_id"].dropna())

matched = png_ids & metadata_ids
png_only = png_ids - metadata_ids
metadata_only = metadata_ids - png_ids

print("\n=== ID MATCHING ===")
print("Matched:", len(matched))
print("PNG IDs missing from metadata:", len(png_only))
print("Metadata IDs missing from PNG dataset:", len(metadata_only))

if png_only:
    print("\nPNG-only IDs:")
    print(sorted(png_only))

if metadata_only:
    print("\nMetadata-only IDs:")
    print(sorted(metadata_only))

# =========================
# DUPLICATES
# =========================

valid_metadata = df[df["image_id"].notna()].copy()

duplicates = valid_metadata[
    valid_metadata["image_id"].duplicated(keep=False)
]

print("\n=== DUPLICATE METADATA IDs ===")
print("Duplicate rows:", len(duplicates))

if len(duplicates):
    print(
        duplicates[
            [
                "image_id",
                "Laterality",
                "View",
                "ACR",
                "Bi-Rads",
            ]
        ].to_string(index=False)
    )

# =========================
# RAW VALUES
# =========================

print("\n=== RAW BI-RADS VALUES ===")
print(df["Bi-Rads"].value_counts(dropna=False))

print("\n=== RAW ACR VALUES ===")
print(df["ACR"].value_counts(dropna=False))

# =========================
# MERGE
# =========================

meta_columns = [
    "image_id",
    "Patient ID",
    "Laterality",
    "View",
    "Acquisition date",
    "ACR",
    "Bi-Rads",
]

metadata_clean = valid_metadata[meta_columns].copy()

merged = png_df.merge(
    metadata_clean,
    on="image_id",
    how="left",
    validate="one_to_one",
)

# =========================
# NORMALIZE BI-RADS
# =========================

def normalize_birads(value):
    if pd.isna(value):
        return pd.NA

    s = str(value).strip().lower()

    if s in {"1", "1.0"}:
        return 1
    if s in {"2", "2.0"}:
        return 2
    if s in {"3", "3.0"}:
        return 3
    if s in {"5", "5.0"}:
        return 5

    if s.startswith("4"):
        return 4

    if s in {"6", "6.0"}:
        return 5

    return pd.NA


merged["birads_metadata_raw"] = merged["Bi-Rads"]

merged["birads_metadata_normalized"] = merged[
    "Bi-Rads"
].apply(normalize_birads)

# =========================
# BI-RADS CHECK
# =========================

missing_birads = merged[
    merged["birads_metadata_normalized"].isna()
]

disagreements = merged[
    merged["birads_metadata_normalized"].notna()
    & (
        merged["birads_folder"]
        != merged["birads_metadata_normalized"]
    )
]

print("\n=== BI-RADS CHECK ===")
print("Images checked:", len(merged))
print("Normalized BI-RADS disagreements:", len(disagreements))
print("Missing/unrecognized BI-RADS:", len(missing_birads))

if len(disagreements):
    print("\nNormalized disagreements:")
    print(
        disagreements[
            [
                "image_id",
                "image_path",
                "birads_folder",
                "birads_metadata_raw",
                "birads_metadata_normalized",
            ]
        ].to_string(index=False)
    )

if len(missing_birads):
    print("\nMissing/unrecognized BI-RADS:")
    print(
        missing_birads[
            [
                "image_id",
                "image_path",
                "birads_metadata_raw",
            ]
        ].to_string(index=False)
    )

# =========================
# SHOW BI-RADS 4 SUBCATEGORIES
# =========================

print("\n=== BI-RADS 4 RAW VALUES ===")

birads4_rows = merged[
    merged["birads_folder"] == 4
][
    [
        "image_id",
        "birads_folder",
        "birads_metadata_raw",
    ]
]

print(
    birads4_rows["birads_metadata_raw"]
    .value_counts(dropna=False)
)

# =========================
# SHOW ORIGINAL BI-RADS 6
# =========================

birads6_rows = merged[
    merged["birads_metadata_raw"]
    .astype(str)
    .str.strip()
    .isin(["6", "6.0"])
]

print("\n=== ORIGINAL BI-RADS 6 ===")
print("Count:", len(birads6_rows))

if len(birads6_rows):
    print(
        birads6_rows[
            [
                "image_id",
                "image_path",
                "birads_folder",
                "birads_metadata_raw",
            ]
        ].to_string(index=False)
    )

# =========================
# ACR / DENSITY
# =========================

print("\n=== ACR / BREAST DENSITY ===")
print(merged["ACR"].value_counts(dropna=False))

# =========================
# VIEW / LATERALITY
# =========================

print("\n=== VIEWS ===")
print(merged["View"].value_counts(dropna=False))

print("\n=== LATERALITY ===")
print(merged["Laterality"].value_counts(dropna=False))

# =========================
# PATIENT ID
# =========================

print("\n=== PATIENT ID ===")
print(merged["Patient ID"].value_counts(dropna=False).head(20))

# =========================
# ANOMALOUS ACR VALUES
# =========================

print("\n=== ANOMALOUS ACR VALUES ===")

acr_numeric = pd.to_numeric(merged["ACR"], errors="coerce")

acr_anomalies = merged[
    ~acr_numeric.isin([1, 2, 3, 4])
]

if len(acr_anomalies) == 0:
    print("None")
else:
    print(
        acr_anomalies[
            [
                "image_id",
                "image_path",
                "ACR",
                "Bi-Rads",
                "Laterality",
                "View",
            ]
        ].to_string(index=False)
    )

# =========================
# ANOMALOUS VIEW VALUES
# =========================

print("\n=== ANOMALOUS VIEW VALUES ===")

view_anomalies = merged[
    ~merged["View"].isin(["CC", "MLO"])
]

if len(view_anomalies) == 0:
    print("None")
else:
    print(
        view_anomalies[
            [
                "image_id",
                "image_path",
                "ACR",
                "Bi-Rads",
                "Laterality",
                "View",
            ]
        ].to_string(index=False)
    )