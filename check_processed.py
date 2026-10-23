
import numpy as np
from pathlib import Path

path = Path("data/processed_lmvd.npz")

with np.load(path, allow_pickle=False) as data:
    for key in data.files:
        arr = data[key]
        print(f"{key}: shape={arr.shape}, dtype={arr.dtype}")

    labels = data["labels"]
    print("\nClass counts:")
    print("Normal (0):", int((labels == 0).sum()))
    print("Depression (1):", int((labels == 1).sum()))

    assert data["audio"].shape[0] == len(labels)
    assert data["visual"].shape[0] == len(labels)
    assert np.isfinite(data["audio"]).all()
    assert np.isfinite(data["visual"]).all()

print("\nDataset verification passed.")
