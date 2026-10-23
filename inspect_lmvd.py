
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
AUDIO_DIR = ROOT / "Audio_feature"
VIDEO_DIR = ROOT / "Video_feature"

print("Audio folder exists:", AUDIO_DIR.exists())
print("Video folder exists:", VIDEO_DIR.exists())

audio_files = sorted(AUDIO_DIR.glob("*.npy"))
video_files = sorted(VIDEO_DIR.glob("*.csv"))

print("\nAudio files:", len(audio_files))
print("Video files:", len(video_files))

if audio_files:
    x = np.load(audio_files[0], allow_pickle=False)
    print("\nFirst audio file:", audio_files[0].name)
    print("Audio shape:", x.shape)
    print("Audio dtype:", x.dtype)
    print("First audio values:", x.flatten()[:10])

if video_files:
    print("\nFirst video file:", video_files[0].name)
    df = pd.read_csv(video_files[0], nrows=5)
    print("Video columns:", list(df.columns))
    print("First five rows:")
    print(df.to_string(index=False))

print("\nInspection completed. No dataset files were modified.")
