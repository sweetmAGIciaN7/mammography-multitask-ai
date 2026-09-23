from pathlib import Path
import re

import pydicom


DICOM_ROOT = Path(
    r"C:\Users\Akpatsha\Downloads"
    r"\Mass-Test_full_mammogram_images-doiJNLP-6ccCrb8t"
    r"\cbis_ddsm"
)

MASS_MANIFEST = Path(
    r"C:\Users\Akpatsha\Downloads"
    r"\Mass-Test_full_mammogram_images-doiJNLP-6ccCrb8t.tcia"
)

CALC_MANIFEST = Path(
    r"C:\Users\Akpatsha\Downloads"
    r"\Calc-Test_full_mammogram_images-doiJNLP-SiXj6kpS.tcia"
)


def read_manifest_series(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {path}"
        )

    lines = path.read_text(
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

        # stop if another key=value section begins
        if "=" in value:
            break

        if re.fullmatch(
            r"\d+(?:\.\d+)+",
            value,
        ):
            series.add(value)

    return series


def get_dicom_info(path):
    try:
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

        sop_uid = str(
            getattr(
                ds,
                "SOPInstanceUID",
                "",
            )
        ).strip()

        patient_id = str(
            getattr(
                ds,
                "PatientID",
                "",
            )
        ).strip()

        series_description = str(
            getattr(
                ds,
                "SeriesDescription",
                "",
            )
        ).strip()

        return {
            "series_uid": series_uid,
            "sop_uid": sop_uid,
            "patient_id": patient_id,
            "series_description": series_description,
        }

    except Exception as exc:
        return {
            "error": str(exc),
        }


mass_series = read_manifest_series(
    MASS_MANIFEST
)

calc_series = read_manifest_series(
    CALC_MANIFEST
)


print("=== MANIFESTS ===")

print(
    "Mass manifest series:",
    len(mass_series),
)

print(
    "Calc manifest series:",
    len(calc_series),
)

overlap = (
    mass_series
    & calc_series
)

print(
    "Series present in BOTH manifests:",
    len(overlap),
)


files = [
    path
    for path in DICOM_ROOT.rglob("*")
    if path.is_file()
]

print("\n=== DOWNLOADED FILES ===")
print("Files:", len(files))


mass_files = []
calc_files = []
unknown_files = []
failed_files = []

seen_series = set()


for index, path in enumerate(
    files,
    start=1,
):
    info = get_dicom_info(path)

    if "error" in info:
        failed_files.append(
            (path, info["error"])
        )
        continue

    series_uid = info["series_uid"]

    if series_uid:
        seen_series.add(
            series_uid
        )

    if series_uid in mass_series:
        mass_files.append(
            (path, info)
        )

    elif series_uid in calc_series:
        calc_files.append(
            (path, info)
        )

    else:
        unknown_files.append(
            (path, info)
        )

    if (
        index % 100 == 0
        or index == len(files)
    ):
        print(
            f"Processed {index}/{len(files)}"
        )


print("\n=== CLASSIFICATION ===")

print(
    "Mass DICOM files:",
    len(mass_files),
)

print(
    "Calc DICOM files:",
    len(calc_files),
)

print(
    "Unknown files:",
    len(unknown_files),
)

print(
    "Unreadable files:",
    len(failed_files),
)


mass_seen = (
    seen_series
    & mass_series
)

calc_seen = (
    seen_series
    & calc_series
)


print("\n=== SERIES CHECK ===")

print(
    "Downloaded Mass series:",
    len(mass_seen),
    "/",
    len(mass_series),
)

print(
    "Downloaded Calc series:",
    len(calc_seen),
    "/",
    len(calc_series),
)


missing_mass = (
    mass_series
    - seen_series
)

missing_calc = (
    calc_series
    - seen_series
)


print(
    "Missing Mass series:",
    len(missing_mass),
)

print(
    "Missing Calc series:",
    len(missing_calc),
)


print("\n=== EXAMPLE MASS FILES ===")

for path, info in mass_files[:5]:
    print(
        path.relative_to(
            DICOM_ROOT
        )
    )
    print(
        "  PatientID:",
        info["patient_id"],
    )
    print(
        "  Series:",
        info["series_uid"],
    )


print("\n=== EXAMPLE CALC FILES ===")

for path, info in calc_files[:5]:
    print(
        path.relative_to(
            DICOM_ROOT
        )
    )
    print(
        "  PatientID:",
        info["patient_id"],
    )
    print(
        "  Series:",
        info["series_uid"],
    )


if unknown_files:
    print(
        "\n=== EXAMPLE UNKNOWN FILES ==="
    )

    for path, info in unknown_files[:10]:
        print(
            path.relative_to(
                DICOM_ROOT
            )
        )

        print(
            "  Series:",
            info.get(
                "series_uid",
                "",
            )
        )


if failed_files:
    print(
        "\n=== UNREADABLE FILES ==="
    )

    for path, error in failed_files[:10]:
        print(
            path.relative_to(
                DICOM_ROOT
            )
        )

        print(
            "  ERROR:",
            error,
        )


print("\n=== RESULT ===")

if (
    len(missing_mass) == 0
    and len(missing_calc) == 0
):
    print(
        "SUCCESS: all manifest series "
        "were found in the downloaded data."
    )
else:
    print(
        "WARNING: some manifest series "
        "were not found."
    )