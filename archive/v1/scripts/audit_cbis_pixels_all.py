from pathlib import Path
from collections import Counter

import pandas as pd
import pydicom


ROOT = Path(__file__).resolve().parents[1]

METADATA = (
    ROOT
    / "data"
    / "external"
    / "cbis_ddsm"
    / "cbis_metadata.csv"
)


def main():
    df = pd.read_csv(METADATA)

    print("Metadata rows:", len(df))

    photometric = Counter()
    transfer_syntax = Counter()
    bits_stored = Counter()
    pixel_representation = Counter()

    missing_pixel_data = []
    errors = []

    for i, row in df.iterrows():
        path = Path(row["dicom_path"])

        try:
            ds = pydicom.dcmread(
                path,
                stop_before_pixels=True,
                force=True,
            )

            photometric[
                str(
                    getattr(
                        ds,
                        "PhotometricInterpretation",
                        "MISSING",
                    )
                )
            ] += 1

            transfer_syntax[
                str(
                    ds.file_meta.TransferSyntaxUID
                )
            ] += 1

            bits_stored[
                str(
                    getattr(
                        ds,
                        "BitsStored",
                        "MISSING",
                    )
                )
            ] += 1

            pixel_representation[
                str(
                    getattr(
                        ds,
                        "PixelRepresentation",
                        "MISSING",
                    )
                )
            ] += 1

            if "PixelData" not in ds:
                # stop_before_pixels prevents PixelData from
                # being loaded, so this is not useful here.
                pass

        except Exception as exc:
            errors.append(
                (
                    row["dataset"],
                    row["match_id"],
                    str(path),
                    repr(exc),
                )
            )

        if (
            (i + 1) % 100 == 0
            or i + 1 == len(df)
        ):
            print(
                f"Processed {i + 1}/{len(df)}"
            )

    print("\n=== DATASET ===")
    print(
        df["dataset"].value_counts()
    )

    print(
        "\n=== PHOTOMETRIC INTERPRETATION ==="
    )

    for key, value in photometric.items():
        print(key, value)

    print(
        "\n=== TRANSFER SYNTAX ==="
    )

    for key, value in transfer_syntax.items():
        print(key, value)

    print(
        "\n=== BITS STORED ==="
    )

    for key, value in bits_stored.items():
        print(key, value)

    print(
        "\n=== PIXEL REPRESENTATION ==="
    )

    for key, value in pixel_representation.items():
        print(key, value)

    print(
        "\nHeader errors:",
        len(errors),
    )

    if errors:
        print("\nFirst errors:")

        for error in errors[:10]:
            print(error)

    # Explicitly decode samples from BOTH subsets.
    print(
        "\n=== PIXEL DECODE BY SUBSET ==="
    )

    for dataset_name in [
        "mass",
        "calc",
    ]:
        subset = (
            df[
                df["dataset"]
                == dataset_name
            ]
            .head(5)
        )

        success = 0

        print(
            f"\n{dataset_name.upper()}:"
        )

        for _, row in subset.iterrows():
            path = Path(
                row["dicom_path"]
            )

            try:
                ds = pydicom.dcmread(
                    path
                )

                arr = ds.pixel_array

                print(
                    row["match_id"],
                    arr.shape,
                    arr.dtype,
                    int(arr.min()),
                    int(arr.max()),
                )

                success += 1

            except Exception as exc:
                print(
                    row["match_id"],
                    "ERROR:",
                    repr(exc),
                )

        print(
            f"Decoded: {success}/{len(subset)}"
        )

    print("\nSUCCESS")


if __name__ == "__main__":
    main()