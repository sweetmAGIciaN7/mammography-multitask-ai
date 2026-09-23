from pathlib import Path
import argparse

import pandas as pd
import pydicom


ROOT = Path(__file__).resolve().parents[1]

CSV_ROOT = (
    ROOT
    / "data"
    / "external"
    / "cbis_ddsm"
)

MASS_CSV = CSV_ROOT / "mass_case_description_test_set.csv"
CALC_CSV = CSV_ROOT / "calc_case_description_test_set.csv"


def load_csv(path, name):
    print(f"\n=== {name} ===")
    print("Path:", path)

    if not path.exists():
        raise FileNotFoundError(
            f"CSV not found: {path}"
        )

    df = pd.read_csv(path)

    print("Rows:", len(df))
    print("Columns:", len(df.columns))

    print("\nColumns:")

    for i, col in enumerate(df.columns):
        print(f"{i}: {col}")

    print("\nFirst 5 rows:")
    print(df.head().to_string())

    return df


def normalize_patient_id(value):
    value = str(value).strip()

    value = value.replace(
        "Mass-Test_",
        "",
    )

    value = value.replace(
        "Calc-Test_",
        "",
    )

    value = value.replace(
        ".dcm",
        "",
    )

    return value


def parse_dicom_root(dicom_root):
    records = []

    files = [
        p
        for p in dicom_root.rglob("*")
        if p.is_file()
    ]

    print(
        "\nReading DICOM headers:",
        len(files),
    )

    for i, path in enumerate(
        files,
        start=1,
    ):
        ds = pydicom.dcmread(
            path,
            stop_before_pixels=True,
            force=True,
        )

        patient_id_raw = str(
            getattr(
                ds,
                "PatientID",
                "",
            )
        ).strip()

        series_uid = str(
            getattr(
                ds,
                "SeriesInstanceUID",
                "",
            )
        ).strip()

        records.append(
            {
                "dicom_path": str(path),
                "dicom_patient_id_raw":
                    patient_id_raw,
                "dicom_patient_id":
                    normalize_patient_id(
                        patient_id_raw
                    ),
                "series_uid":
                    series_uid,
            }
        )

        if (
            i % 100 == 0
            or i == len(files)
        ):
            print(
                f"Processed {i}/{len(files)}"
            )

    return pd.DataFrame(records)


def inspect_values(df, name):
    print(
        f"\n=== {name} VALUE COUNTS ==="
    )

    wanted = [
        "patient_id",
        "breast_density",
        "left or right breast",
        "image view",
        "assessment",
        "pathology",
        "image file path",
    ]

    lookup = {
        col.lower(): col
        for col in df.columns
    }

    for key in wanted:
        if key not in lookup:
            continue

        col = lookup[key]

        print(
            f"\n--- {col} ---"
        )

        if "path" in key:
            print(
                df[col]
                .head(10)
                .to_string(index=False)
            )
        else:
            print(
                df[col]
                .value_counts(
                    dropna=False
                )
            )


def inspect_matching(
    csv_df,
    dicom_df,
    dataset_name,
):
    lookup = {
        col.lower(): col
        for col in csv_df.columns
    }

    required = [
        "patient_id",
        "left or right breast",
        "image view",
    ]

    missing = [
        col
        for col in required
        if col not in lookup
    ]

    if missing:
        print(
            "\nCannot test matching."
        )

        print(
            "Missing CSV columns:",
            missing,
        )

        return

    patient_col = lookup["patient_id"]

    side_col = lookup[
        "left or right breast"
    ]

    view_col = lookup[
        "image view"
    ]

    work = csv_df.copy()

    work["match_id"] = (
        work[patient_col]
        .astype(str)
        .str.strip()
        + "_"
        + work[side_col]
        .astype(str)
        .str.strip()
        + "_"
        + work[view_col]
        .astype(str)
        .str.strip()
    )

    csv_ids = set(
        work["match_id"]
    )

    dicom_ids = set(
        dicom_df[
            "dicom_patient_id"
        ]
    )

    matched = (
        csv_ids
        & dicom_ids
    )

    missing_in_dicom = (
        csv_ids
        - dicom_ids
    )

    extra_in_dicom = (
        dicom_ids
        - csv_ids
    )

    print(
        f"\n=== {dataset_name} MATCHING ==="
    )

    print(
        "Unique CSV match IDs:",
        len(csv_ids),
    )

    print(
        "Unique DICOM IDs:",
        len(dicom_ids),
    )

    print(
        "Matched IDs:",
        len(matched),
    )

    print(
        "CSV IDs missing in DICOM:",
        len(missing_in_dicom),
    )

    print(
        "DICOM IDs not in CSV:",
        len(extra_in_dicom),
    )

    if missing_in_dicom:
        print(
            "\nExamples missing in DICOM:"
        )

        for value in sorted(
            missing_in_dicom
        )[:20]:
            print(value)

    if extra_in_dicom:
        print(
            "\nExamples extra DICOM IDs:"
        )

        for value in sorted(
            extra_in_dicom
        )[:20]:
            print(value)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dicom-root",
        required=True,
    )

    args = parser.parse_args()

    dicom_root = Path(
        args.dicom_root
    )

    mass_df = load_csv(
        MASS_CSV,
        "MASS TEST CSV",
    )

    calc_df = load_csv(
        CALC_CSV,
        "CALC TEST CSV",
    )

    inspect_values(
        mass_df,
        "MASS TEST",
    )

    inspect_values(
        calc_df,
        "CALC TEST",
    )

    dicom_df = parse_dicom_root(
        dicom_root
    )

    print(
        "\n=== DICOM SUMMARY ==="
    )

    print(
        "Rows:",
        len(dicom_df),
    )

    print(
        "Unique patient IDs:",
        dicom_df[
            "dicom_patient_id"
        ].nunique(),
    )

    print(
        "Unique series UIDs:",
        dicom_df[
            "series_uid"
        ].nunique(),
    )

    mass_dicoms = dicom_df[
        dicom_df[
            "dicom_patient_id_raw"
        ].str.startswith(
            "Mass-Test_",
            na=False,
        )
    ]

    calc_dicoms = dicom_df[
        ~dicom_df.index.isin(
            mass_dicoms.index
        )
    ]

    inspect_matching(
        mass_df,
        mass_dicoms,
        "MASS TEST",
    )

    inspect_matching(
        calc_df,
        calc_dicoms,
        "CALC TEST",
    )

    print("\nSUCCESS")


if __name__ == "__main__":
    main()