
from pathlib import Path
import numpy as np
import pandas as pd

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = PROJECT_ROOT / "data" / "lmvd_manifest.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed_lmvd.npz"

EXCLUDED_COLUMNS = {
    "frame",
    "face_id",
    "timestamp",
    "confidence",
    "success",
}


def load_visual_csv(path):
    """Read visual features and convert all usable measurements to numbers."""
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()

    # Remove metadata columns, not facial measurement columns.
    df = df.drop(
        columns=[
            col for col in df.columns
            if col.strip().lower() in EXCLUDED_COLUMNS
        ],
        errors="ignore",
    )

    # Convert non-numeric values to missing values.
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)

    # Remove columns that contain no numeric values in this file.
    df = df.dropna(axis=1, how="all")

    return df


def summarize_audio(path):
    """Create a fixed-length vector from a time-by-feature audio array."""
    features = np.load(path, allow_pickle=False)
    features = np.asarray(features, dtype=np.float32)

    if features.ndim != 2 or features.shape[0] == 0:
        raise ValueError(
            f"Unexpected audio feature shape {features.shape} in {path}"
        )

    features = np.nan_to_num(
        features, nan=0.0, posinf=0.0, neginf=0.0
    )

    return np.concatenate([
        features.mean(axis=0),
        features.std(axis=0),
    ]).astype(np.float32)


def main():
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST_PATH}"
        )

    manifest = pd.read_csv(MANIFEST_PATH)

    required = {"sample_id", "audio_path", "video_path", "label"}
    missing = required - set(manifest.columns)

    if missing:
        raise ValueError(
            f"Manifest is missing required columns: {sorted(missing)}"
        )

    if manifest.empty:
        raise ValueError("The manifest contains no samples.")

    print(f"Manifest samples: {len(manifest)}")
    print("Checking visual feature columns across all samples...")

    # First pass: find columns that have numeric data in EVERY sample.
    common_columns = None
    visual_columns_by_sample = []

    for index, row in manifest.iterrows():
        video_path = Path(str(row["video_path"]))

        if not video_path.is_absolute():
            video_path = PROJECT_ROOT / video_path

        if not video_path.exists():
            raise FileNotFoundError(
                f"Visual file not found for sample "
                f"{row['sample_id']}: {video_path}"
            )

        visual_df = load_visual_csv(video_path)
        usable_columns = set(visual_df.columns)

        if common_columns is None:
            common_columns = usable_columns
        else:
            common_columns &= usable_columns

        visual_columns_by_sample.append(
            (video_path, visual_df)
        )

        if (index + 1) % 200 == 0:
            print(f"Checked {index + 1}/{len(manifest)} visual files")

    common_columns = sorted(common_columns or [])

    if not common_columns:
        raise ValueError(
            "No common numeric visual columns were found. "
            "Inspect the CSV files and their column formats."
        )

    print(f"Common visual columns: {len(common_columns)}")

    # Second pass: aggregate the same columns for every sample.
    audio_vectors = []
    visual_vectors = []
    labels = []
    sample_ids = []

    for index, (_, row) in enumerate(manifest.iterrows()):
        audio_path = Path(str(row["audio_path"]))

        if not audio_path.is_absolute():
            audio_path = PROJECT_ROOT / audio_path

        if not audio_path.exists():
            raise FileNotFoundError(
                f"Audio file not found for sample "
                f"{row['sample_id']}: {audio_path}"
            )

        audio_vector = summarize_audio(audio_path)

        video_path, visual_df = visual_columns_by_sample[index]
        visual_df = visual_df.reindex(columns=common_columns)

        # Fill missing measurements using each column's median
        # within this sample; use zero if a median is unavailable.
        medians = visual_df.median(axis=0)
        visual_df = visual_df.fillna(medians).fillna(0.0)

        visual_values = visual_df.to_numpy(dtype=np.float32)

        if visual_values.shape[0] == 0:
            raise ValueError(
                f"No visual frames found for sample {row['sample_id']}"
            )

        visual_vector = np.concatenate([
            visual_values.mean(axis=0),
            visual_values.std(axis=0),
        ]).astype(np.float32)

        audio_vectors.append(audio_vector)
        visual_vectors.append(visual_vector)
        labels.append(int(row["label"]))
        sample_ids.append(str(row["sample_id"]))

        if (index + 1) % 200 == 0:
            print(f"Processed {index + 1}/{len(manifest)} samples")

    # All samples must have identical feature dimensions.
    audio_array = np.stack(audio_vectors)
    visual_array = np.stack(visual_vectors)
    label_array = np.asarray(labels, dtype=np.int64)
    sample_id_array = np.asarray(sample_ids, dtype=str)

    if not (
        len(audio_array)
        == len(visual_array)
        == len(label_array)
        == len(sample_id_array)
    ):
        raise ValueError("The processed arrays have different sample counts.")

    if not np.isfinite(audio_array).all():
        raise ValueError("Audio features contain NaN or infinite values.")

    if not np.isfinite(visual_array).all():
        raise ValueError("Visual features contain NaN or infinite values.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        OUTPUT_PATH,
        audio=audio_array,
        visual=visual_array,
        labels=label_array,
        sample_ids=sample_id_array,
        visual_columns=np.asarray(common_columns, dtype=str),
    )

    print("\nPreprocessing completed successfully.")
    print(f"Samples: {len(label_array)}")
    print(f"Audio feature shape: {audio_array.shape}")
    print(f"Visual feature shape: {visual_array.shape}")
    print(f"Labels shape: {label_array.shape}")
    print("Label counts:")
    print(pd.Series(label_array).value_counts().sort_index())
    print(f"Saved dataset: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
