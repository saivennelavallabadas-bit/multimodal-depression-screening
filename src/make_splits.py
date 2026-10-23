
import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser(
        description="Create stratified LMVD train/validation/test splits."
    )
    parser.add_argument(
        "--manifest",
        default="data/lmvd_manifest.csv",
    )
    parser.add_argument(
        "--out-dir",
        default="data/splits",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    df = pd.read_csv(manifest_path)

    required = {"sample_id", "audio_path", "video_path", "label"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing manifest columns: {sorted(missing)}")

    if df["sample_id"].duplicated().any():
        raise ValueError("Duplicate sample IDs found.")

    if df[["audio_path", "video_path", "label"]].isnull().any().any():
        raise ValueError("Manifest contains missing paths or labels.")

    df["label"] = df["label"].astype(int)

    if not set(df["label"].unique()).issubset({0, 1}):
        raise ValueError("Labels must be 0 or 1.")

    if df["label"].nunique() != 2:
        raise ValueError("Both classes must be present.")

    # 70% training, 15% validation, 15% testing.
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=args.seed,
        stratify=df["label"],
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=args.seed,
        stratify=temp_df["label"],
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = {
        "train": train_df,
        "val": val_df,
        "test": test_df,
    }

    for name, split_df in splits.items():
        split_df = split_df.sort_values("sample_id")
        split_df.to_csv(out_dir / f"{name}.csv", index=False)

        counts = split_df["label"].value_counts().sort_index().to_dict()
        print(f"{name}: {len(split_df)} samples; class counts {counts}")

    total = sum(len(split_df) for split_df in splits.values())
    assert total == len(df)

    print(f"\nTotal samples: {total}")
    print(f"Split files saved in: {out_dir.resolve()}")
    print(
        "\nIMPORTANT: These are sample-level splits. "
        "Participant IDs are not available in this manifest, "
        "so participant-independent evaluation is not established."
    )


if __name__ == "__main__":
    main()
