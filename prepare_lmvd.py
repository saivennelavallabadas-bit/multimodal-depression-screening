
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
AUDIO_DIR = PROJECT_ROOT / "Audio_feature"
VIDEO_DIR = PROJECT_ROOT / "Video_feature"
LABEL_DIR = PROJECT_ROOT / "label"
OUTPUT_FILE = PROJECT_ROOT / "data" / "lmvd_manifest.csv"

rows = []
missing = []

for audio_path in sorted(AUDIO_DIR.glob("*.npy")):
    sample_id = audio_path.stem
    video_path = VIDEO_DIR / f"{sample_id}.csv"
    label_path = LABEL_DIR / f"{sample_id}_Depression.csv"

    if not video_path.exists() or not label_path.exists():
        missing.append(sample_id)
        continue

    try:
        label_text = label_path.read_text(
            encoding="utf-8-sig"
        ).strip()
        label = int(label_text)
    except (ValueError, OSError):
        print(f"Skipping invalid label: {label_path.name}")
        continue

    if label not in (0, 1):
        print(f"Skipping unexpected label: {label_path.name}")
        continue

    rows.append({
        "sample_id": sample_id,
        "audio_path": str(audio_path),
        "video_path": str(video_path),
        "label": label,
    })

df = pd.DataFrame(rows)

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False)

print("Manifest created:", OUTPUT_FILE)
print("Matched samples:", len(df))
print("Missing files:", len(missing))
print("\nLabel counts:")
print(df["label"].value_counts().sort_index().rename(
    index={0: "Normal", 1: "Depression"}
))
