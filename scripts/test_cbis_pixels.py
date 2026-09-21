from pathlib import Path

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

    # Берём несколько разных изображений.
    samples = df.head(10)

    success = 0
    failed = 0

    for i, row in samples.iterrows():
        path = Path(row["dicom_path"])

        print("\n" + "=" * 70)
        print("ROW:", i)
        print("Dataset:", row["dataset"])
        print("Match ID:", row["match_id"])
        print("Path:", path)

        ds = pydicom.dcmread(path)

        print(
            "TransferSyntaxUID:",
            ds.file_meta.TransferSyntaxUID,
        )

        print(
            "PhotometricInterpretation:",
            getattr(
                ds,
                "PhotometricInterpretation",
                None,
            ),
        )

        print(
            "Rows:",
            getattr(ds, "Rows", None),
        )

        print(
            "Columns:",
            getattr(ds, "Columns", None),
        )

        print(
            "BitsStored:",
            getattr(ds, "BitsStored", None),
        )

        print(
            "PixelRepresentation:",
            getattr(
                ds,
                "PixelRepresentation",
                None,
            ),
        )

        try:
            arr = ds.pixel_array

            print(
                "pixel_array shape:",
                arr.shape,
            )

            print(
                "pixel_array dtype:",
                arr.dtype,
            )

            print(
                "pixel min:",
                arr.min(),
            )

            print(
                "pixel max:",
                arr.max(),
            )

            success += 1

        except Exception as exc:
            print(
                "PIXEL DECODE ERROR:",
                repr(exc),
            )

            failed += 1

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("Successfully decoded:", success)
    print("Failed:", failed)

    if failed == 0:
        print(
            "\nSUCCESS: sampled CBIS-DDSM "
            "pixel data can be decoded."
        )
    else:
        print(
            "\nWARNING: additional DICOM "
            "decoder packages may be required."
        )


if __name__ == "__main__":
    main()