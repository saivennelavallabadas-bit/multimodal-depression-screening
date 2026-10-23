
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
AUDIO = ROOT / "Audio_feature"
VIDEO = ROOT / "Video_feature"

# Labels from the supplied LMVD Readme:
# 1 = depression, 0 = normal
def get_label(number):
    if 1 <= number <= 601 or 1117 <= number <= 1423:
        return 1
    if 602 <= number <= 1116 or 1425 <= number <= 1824:
        return 0
    return None

audio_ids = {
    int(p.stem) for p in AUDIO.glob("*.npy") if p.stem.isdigit()
}
video_ids = {
    int(p.stem) for p in VIDEO.glob("*.csv") if p.stem.isdigit()
}

print("Audio files:", len(audio_ids))
print("Video files:", len(video_ids))
print("IDs present in both:", len(audio_ids & video_ids))
print("Audio IDs without video:", len(audio_ids - video_ids))
print("Video IDs without audio:", len(video_ids - audio_ids))

labels = {i: get_label(i) for i in audio_ids & video_ids}
print("\nLabel counts among matched files:")
print("Depression:", sum(v == 1 for v in labels.values()))
print("Normal:", sum(v == 0 for v in labels.values()))
print("Unassigned:", sum(v is None for v in labels.values()))

# Check only the first video file for missing values
sample = VIDEO / "001.csv"
if sample.exists():
    df = pd.read_csv(sample)
    print("\n001.csv rows:", len(df))
    print("001.csv columns:", len(df.columns))
    print("Duplicate column names:", int(df.columns.duplicated().sum()))
    print("Missing values in first 10 columns:")
    print(df.iloc[:, :10].isna().sum().to_dict())

print("\nSummary complete. No files were modified.")
