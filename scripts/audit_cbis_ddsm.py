from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

CBIS_ROOT = ROOT / "data" / "external" / "cbis_ddsm"

MASS_CSV = CBIS_ROOT / "mass_case_description_test_set.csv"
CALC_CSV = CBIS_ROOT / "calc_case_description_test_set.csv"

MASS_DIR = CBIS_ROOT / "mass_test"
CALC_DIR = CBIS_ROOT / "calc_test"


def inspect_csv(path, name):
    print(f"\n=== {name} CSV ===")
    print("Path:", path)

    if not path.exists():
        print("MISSING")
        return None

    df = pd.read_csv(path)

    print("Rows:", len(df))
    print("Columns:", len(df.columns))

    print("\nColumns:")
    for i, col in enumerate(df.columns):
        print(f"{i}: {col}")

    print("\nFirst 5 rows:")
    print(df.head().to_string())

    print("\nNull counts for key-looking columns:")

    for col in df.columns:
        col_lower = col.lower()

        if any(
            key in col_lower
            for key in [
                "patient",
                "density",
                "assessment",
                "view",
                "laterality",
                "image",
                "path",
                "file",
            ]
        ):
            print(
                f"{col}:",
                df[col].isna().sum(),
            )

    return df


def inspect_dicom_dir(path, name):
    print(f"\n=== {name} DICOM DIRECTORY ===")
    print("Path:", path)

    if not path.exists():
        print("MISSING")
        return []

    all_files = [
        p
        for p in path.rglob("*")
        if p.is_file()
    ]

    dcm_files = [
        p
        for p in all_files
        if p.suffix.lower() == ".dcm"
    ]

    print("All files:", len(all_files))
    print("Files ending in .dcm:", len(dcm_files))

    print("\nFirst 20 files:")

    for p in all_files[:20]:
        print(
            p.relative_to(CBIS_ROOT)
        )

    print("\nTop-level directories:")

    for p in sorted(path.iterdir())[:30]:
        print(
            p.name,
            "[DIR]" if p.is_dir() else "[FILE]",
        )

    return dcm_files


mass_df = inspect_csv(
    MASS_CSV,
    "MASS TEST",
)

calc_df = inspect_csv(
    CALC_CSV,
    "CALC TEST",
)

mass_dcm = inspect_dicom_dir(
    MASS_DIR,
    "MASS TEST",
)

calc_dcm = inspect_dicom_dir(
    CALC_DIR,
    "CALC TEST",
)


print("\n=== SUMMARY ===")

print(
    "Mass CSV rows:",
    len(mass_df)
    if mass_df is not None
    else 0,
)

print(
    "Calc CSV rows:",
    len(calc_df)
    if calc_df is not None
    else 0,
)

print(
    "Mass .dcm files:",
    len(mass_dcm),
)

print(
    "Calc .dcm files:",
    len(calc_dcm),
)

print("\nSUCCESS")