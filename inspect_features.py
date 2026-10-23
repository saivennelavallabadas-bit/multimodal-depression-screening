
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent

manifest = pd.read_csv(ROOT / "data" / "lmvd_manifest.csv")

print("Manifest rows:", len(manifest))
print("\nAudio feature shapes:")

for sample_id in ["001", "002", "602"]:
    path = ROOT / "Audio_feature" / f"{sample_id}.npy"
    if path.exists():
        arr = np.load(path, mmap_mode="r")
        print(sample_id, arr.shape, arr.dtype)

print("\nVisual feature shapes:")

for sample_id in ["001", "002", "602"]:
    path = ROOT / "Video_feature" / f"{sample_id}.csv"
    if path.exists():
        df = pd.read_csv(path)
        print(sample_id, df.shape)
        print("First 10 columns:", df.columns.tolist()[:10])

print("\nLabels:")
print(manifest["label"].value_counts().sort_index())
