"""Create tiny synthetic files to smoke-test the pipeline; not research data."""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import soundfile as sf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="data/demo")
    parser.add_argument("--participants", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.participants < 12 or args.participants % 2:
        raise ValueError("Use an even participant count of at least 12 for stratified demo splits")
    root = Path(args.out_dir)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    sample_rate = 16000
    frame_size = (160, 120)
    rows = []
    for i in range(args.participants):
        participant = f"DEMO{i + 1:03d}"
        label = i % 2
        # Deterministic class-correlated toy signals only to exercise code paths.
        seconds = 2
        t = np.arange(sample_rate * seconds) / sample_rate
        tone = 170 + 60 * label
        audio = (0.12 * np.sin(2 * np.pi * tone * t) + rng.normal(0, .01, len(t))).astype("float32")
        audio_rel = f"{participant}.wav"
        sf.write(root / audio_rel, audio, sample_rate)
        video_rel = f"{participant}.avi"
        writer = cv2.VideoWriter(str(root / video_rel), cv2.VideoWriter_fourcc(*"MJPG"), 8, frame_size)
        if not writer.isOpened():
            raise RuntimeError("OpenCV could not create an MJPG AVI; check the OpenCV video codec installation")
        for frame_i in range(16):
            frame = np.full((frame_size[1], frame_size[0], 3), 80 + 30 * label, dtype=np.uint8)
            cv2.circle(frame, (80, 60), 28, (170, 190, 210), -1)
            cv2.circle(frame, (70, 53), 3, (15, 15, 15), -1)
            cv2.circle(frame, (90, 53), 3, (15, 15, 15), -1)
            cv2.ellipse(frame, (80, 69), (10, 4 + 3 * label), 0, 0, 180, (20, 20, 20), 2)
            writer.write(frame)
        writer.release()
        text_rel = f"{participant}.txt"
        (root / text_rel).write_text(
            "This is a synthetic sample transcript for pipeline testing. " * (2 + label), encoding="utf-8"
        )
        rows.append({"participant_id": participant, "audio_path": (Path(args.out_dir) / audio_rel).as_posix(),
                     "video_path": (Path(args.out_dir) / video_rel).as_posix(),
                     "transcript_path": (Path(args.out_dir) / text_rel).as_posix(), "label": label})
    manifest = Path("data/demo_manifest.csv")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(manifest, index=False)
    print(f"Created {len(rows)} SYNTHETIC participants and {manifest}. Never report these metrics as research results.")


if __name__ == "__main__":
    main()
