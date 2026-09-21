from pathlib import Path
import argparse
import re

import pandas as pd
import pydicom


ROOT = Path(__file__).resolve().parents[1]

CSV_ROOT = ROOT / "data" / "external" / "cbis_ddsm"

MASS_CSV = CSV_ROOT / "mass_case_description_test_set.csv"
CALC_CSV = CSV_ROOT / "calc_case_description_test_set.csv"

OUTPUT_CSV = CSV_ROOT / "cbis_metadata.csv"


def read_manifest_series(path):
    lines = Path(path).read_text(
        encoding="utf-8",
        errors="ignore",
    ).splitlines()

    try:
        start = next(
            i
            for i, line in enumerate(lines)
            if line.strip().lower()
            == "listofseriestodownload="
        )
    except StopIteration:
        raise RuntimeError(
            f"Could not find ListOfSeriesToDownload in {path}"
        )

    series = set()

    for line in lines[start + 1:]:
        value = line.strip()

        if not value:
            continue

        if "=" in value:
            break

        if re.fullmatch(
            r"\d+(?:\.\d+)+",
            value,
        ):
            series.add(value)

    return series


def normalize_patient_view_id(value):
    value = str(value).strip()

    value = value.replace("Mass-Test_", "")
    value = value.replace("Calc-Test_", "")
    value = value.replace(".dcm", "")

    return value


def build_csv_match_id(df):
    return (
        df["patient_id"]
        .astype(str)
        .str.strip()
        + "_"
        + df["left or right breast"]
        .astype(str)
        .str.strip()
        + "_"
        + df["image view"]
        .astype(str)
        .str.strip()
    )


def unique_values(series):
    values = (
        series
        .dropna()
        .astype(str)
        .str.strip()
    )

    values = [
        value
        for value in values
        if value
        and value.lower() != "nan"
    ]

    return sorted(set(values))


def aggregate_dataset(
    df,
    dataset_name,
    density_column,
):
    df = df.copy()

    df["match_id"] = build_csv_match_id(df)

    records = []

    for match_id, group in df.groupby("match_id"):

        assessments = unique_values(
            group["assessment"]
        )

        densities = unique_values(
            group[density_column]
        )

        pathologies = unique_values(
            group["pathology"]
        )

        patients = unique_values(
            group["patient_id"]
        )

        sides = unique_values(
            group["left or right breast"]
        )

        views = unique_values(
            group["image view"]
        )

        if len(patients) != 1:
            raise RuntimeError(
                f"Patient conflict: {dataset_name}, "
                f"{match_id}, {patients}"
            )

        if len(sides) != 1:
            raise RuntimeError(
                f"Laterality conflict: {dataset_name}, "
                f"{match_id}, {sides}"
            )

        if len(views) != 1:
            raise RuntimeError(
                f"View conflict: {dataset_name}, "
                f"{match_id}, {views}"
            )

        assessment_conflict = len(assessments) > 1
        density_conflict = len(densities) > 1

        assessment = None

        if len(assessments) == 1:
            assessment = int(
                float(assessments[0])
            )

        density = None

        if len(densities) == 1:
            density = int(
                float(densities[0])
            )

        records.append(
            {
                "dataset": dataset_name,
                "match_id": match_id,

                "patient_id": patients[0],
                "laterality": sides[0],
                "view": views[0],

                "assessment": assessment,

                "assessment_values": "|".join(
                    assessments
                ),

                "assessment_conflict":
                    assessment_conflict,

                "breast_density": density,

                "density_values": "|".join(
                    densities
                ),

                "density_conflict":
                    density_conflict,

                "pathology_values": "|".join(
                    pathologies
                ),

                "lesion_rows": len(group),
            }
        )

    return pd.DataFrame(records)


def read_dicom_table(
    dicom_root,
    mass_series,
    calc_series,
):
    files = [
        path
        for path in dicom_root.rglob("*")
        if path.is_file()
    ]

    print(
        f"\nReading {len(files)} DICOM files..."
    )

    records = []

    unknown = 0

    for i, path in enumerate(
        files,
        start=1,
    ):
        ds = pydicom.dcmread(
            path,
            stop_before_pixels=True,
            force=True,
        )

        series_uid = str(
            getattr(
                ds,
                "SeriesInstanceUID",
                "",
            )
        ).strip()

        raw_patient_id = str(
            getattr(
                ds,
                "PatientID",
                "",
            )
        ).strip()

        if series_uid in mass_series:
            dataset = "mass"

        elif series_uid in calc_series:
            dataset = "calc"

        else:
            dataset = None
            unknown += 1

        match_id = normalize_patient_view_id(
            raw_patient_id
        )

        records.append(
            {
                "dataset": dataset,
                "match_id": match_id,
                "dicom_path": str(path),
                "dicom_patient_id":
                    raw_patient_id,
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

    dicom_df = pd.DataFrame(records)

    print(
        "Unknown DICOM dataset labels:",
        unknown,
    )

    if unknown != 0:
        raise RuntimeError(
            "Some DICOM series were not present "
            "in either manifest."
        )

    if dicom_df["series_uid"].duplicated().any():
        raise RuntimeError(
            "Duplicate SeriesInstanceUID found."
        )

    duplicate_keys = dicom_df.duplicated(
        subset=[
            "dataset",
            "match_id",
        ],
        keep=False,
    )

    if duplicate_keys.any():
        print(
            "\nDuplicate DICOM composite keys:"
        )

        print(
            dicom_df.loc[
                duplicate_keys,
                [
                    "dataset",
                    "match_id",
                    "series_uid",
                ],
            ].to_string(
                index=False
            )
        )

        raise RuntimeError(
            "dataset + match_id is not unique "
            "in DICOM table."
        )

    return dicom_df


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dicom-root",
        required=True,
    )

    parser.add_argument(
        "--mass-manifest",
        required=True,
    )

    parser.add_argument(
        "--calc-manifest",
        required=True,
    )

    args = parser.parse_args()

    dicom_root = Path(
        args.dicom_root
    )

    mass_series = read_manifest_series(
        args.mass_manifest
    )

    calc_series = read_manifest_series(
        args.calc_manifest
    )

    print(
        "Mass manifest series:",
        len(mass_series),
    )

    print(
        "Calc manifest series:",
        len(calc_series),
    )

    overlap = mass_series & calc_series

    print(
        "Manifest overlap:",
        len(overlap),
    )

    if overlap:
        raise RuntimeError(
            "Mass and Calc manifests overlap."
        )

    mass = pd.read_csv(
        MASS_CSV
    )

    calc = pd.read_csv(
        CALC_CSV
    )

    print(
        "\nMass raw rows:",
        len(mass),
    )

    print(
        "Calc raw rows:",
        len(calc),
    )

    mass_images = aggregate_dataset(
        mass,
        "mass",
        "breast_density",
    )

    calc_images = aggregate_dataset(
        calc,
        "calc",
        "breast density",
    )

    metadata = pd.concat(
        [
            mass_images,
            calc_images,
        ],
        ignore_index=True,
    )

    print(
        "\nImage-level metadata rows:",
        len(metadata),
    )

    print(
        "Mass image rows:",
        len(mass_images),
    )

    print(
        "Calc image rows:",
        len(calc_images),
    )

    print(
        "\nAssessment conflicts:",
        int(
            metadata[
                "assessment_conflict"
            ].sum()
        ),
    )

    print(
        "Density conflicts:",
        int(
            metadata[
                "density_conflict"
            ].sum()
        ),
    )

    print(
        "Images with >1 lesion row:",
        int(
            (
                metadata["lesion_rows"] > 1
            ).sum()
        ),
    )

    print(
        "\nAssessment distribution:"
    )

    print(
        metadata[
            "assessment"
        ]
        .value_counts(
            dropna=False
        )
        .sort_index()
    )

    print(
        "\nDensity distribution:"
    )

    print(
        metadata[
            "breast_density"
        ]
        .value_counts(
            dropna=False
        )
        .sort_index()
    )

    dicoms = read_dicom_table(
        dicom_root,
        mass_series,
        calc_series,
    )

    print(
        "\nDICOM rows:",
        len(dicoms),
    )

    print(
        "Mass DICOM rows:",
        int(
            (
                dicoms["dataset"]
                == "mass"
            ).sum()
        ),
    )

    print(
        "Calc DICOM rows:",
        int(
            (
                dicoms["dataset"]
                == "calc"
            ).sum()
        ),
    )

    merged = metadata.merge(
        dicoms,
        on=[
            "dataset",
            "match_id",
        ],
        how="left",
        validate="one_to_one",
    )

    missing_dicom = int(
        merged[
            "dicom_path"
        ]
        .isna()
        .sum()
    )

    print(
        "\nMissing DICOM after merge:",
        missing_dicom,
    )

    if missing_dicom != 0:

        print(
            merged[
                merged[
                    "dicom_path"
                ].isna()
            ][
                [
                    "dataset",
                    "match_id",
                    "patient_id",
                    "laterality",
                    "view",
                ]
            ].to_string(
                index=False
            )
        )

        raise RuntimeError(
            "Some metadata rows did not "
            "match a DICOM file."
        )

    # Our INbreast model predicts
    # BI-RADS classes 1-5.
    #
    # CBIS-DDSM also includes assessment 0.
    # We do NOT remap 0 into another class.

    merged["birads_usable"] = (
        merged[
            "assessment"
        ].isin(
            [1, 2, 3, 4, 5]
        )
        &
        ~merged[
            "assessment_conflict"
        ]
    )

    # INbreast density task is ACR 1-4.
    # CBIS-DDSM contains two density=0 rows.
    # They are excluded, not remapped.

    merged["density_usable"] = (
        merged[
            "breast_density"
        ].isin(
            [1, 2, 3, 4]
        )
        &
        ~merged[
            "density_conflict"
        ]
    )

    print(
        "\nBI-RADS usable images:",
        int(
            merged[
                "birads_usable"
            ].sum()
        ),
    )

    print(
        "BI-RADS excluded images:",
        int(
            (
                ~merged[
                    "birads_usable"
                ]
            ).sum()
        ),
    )

    print(
        "Density usable images:",
        int(
            merged[
                "density_usable"
            ].sum()
        ),
    )

    print(
        "Density excluded images:",
        int(
            (
                ~merged[
                    "density_usable"
                ]
            ).sum()
        ),
    )

    merged.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    print(
        "\nSaved:",
        OUTPUT_CSV,
    )

    print("\nSUCCESS")


if __name__ == "__main__":
    main()