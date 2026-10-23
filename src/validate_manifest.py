import argparse
from pathlib import Path
import pandas as pd

REQUIRED = ["participant_id", "audio_path", "video_path", "transcript_path", "label"]

def resolve_path(value, base: Path):
    p = Path(str(value))
    return p if p.is_absolute() else (base / p).resolve()

def validate(manifest: Path):
    df = pd.read_csv(manifest, dtype={"participant_id": str})
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if df.empty:
        raise ValueError("Manifest has no rows.")
    if df["participant_id"].isna().any() or (df["participant_id"].str.strip() == "").any():
        raise ValueError("participant_id contains empty values.")
    duplicates = df[df["participant_id"].duplicated(keep=False)]["participant_id"].tolist()
    if duplicates:
        raise ValueError(f"Expected one row per participant; duplicate IDs: {sorted(set(duplicates))[:20]}")
    labels = set(pd.to_numeric(df["label"], errors="coerce").dropna().astype(int).tolist())
    if df["label"].isna().any() or not set(df["label"].astype(str)).issubset({"0", "1", "0.0", "1.0"}):
        raise ValueError("label must be binary 0/1 and non-missing. Follow the dataset's documented label protocol.")
    if labels != {0, 1}:
        print("WARNING: manifest does not contain both classes; stratified split/training may fail.")
    base = manifest.resolve().parent.parent
    errors = []
    for _, row in df.iterrows():
        for col in ["audio_path", "video_path", "transcript_path"]:
            val = row[col]
            if pd.isna(val) or str(val).strip() == "":
                errors.append(f"{row['participant_id']}: empty {col}")
            elif not resolve_path(val, base).is_file():
                errors.append(f"{row['participant_id']}: missing {col}: {resolve_path(val, base)}")
    if errors:
        raise FileNotFoundError("Manifest validation failed:\n" + "\n".join(errors[:50]))
    print(f"Manifest OK: {len(df)} unique participants; all three files exist.")
    print("IMPORTANT: file existence does not prove participant alignment. Verify mappings and labels manually against the authorized dataset documentation.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifest.csv")
    args = ap.parse_args()
    validate(Path(args.manifest))

if __name__ == "__main__":
    main()
