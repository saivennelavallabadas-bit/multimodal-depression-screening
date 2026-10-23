from pathlib import Path
import warnings
import cv2
import librosa
import numpy as np

def resolve_path(value, project_root: Path) -> Path:
    p = Path(str(value))
    return p if p.is_absolute() else (project_root / p).resolve()

def audio_features(path: Path, sr: int = 16000) -> np.ndarray:
    """Fixed, non-learned summary features. Not clinical biomarkers."""
    y, sr = librosa.load(str(path), sr=sr, mono=True)
    if y.size == 0:
        raise ValueError(f"Empty audio: {path}")
    if len(y) > sr * 300:
        y = y[:sr * 300]  # bounded runtime; document truncation policy
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)
    features = []
    for matrix in [mfcc, centroid, rolloff, zcr, rms]:
        features.extend(np.mean(matrix, axis=1).tolist())
        features.extend(np.std(matrix, axis=1).tolist())
    features.extend([float(len(y) / sr), float(np.mean(np.abs(y))), float(np.std(y))])
    arr = np.asarray(features, dtype=np.float32)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

def video_features(path: Path, max_frames: int = 80) -> np.ndarray:
    """
    Face-region HOG summaries sampled uniformly from a video.
    These are low-level visual descriptors, not facial emotion labels.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise ValueError(f"No readable frames in video: {path}")
    positions = np.linspace(0, max(total - 1, 0), num=min(max_frames, total), dtype=int)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    hog = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)
    descriptors = []
    gray_stats = []
    for pos in positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(pos))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40))
        if len(faces):
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            roi = gray[y:y+h, x:x+w]
        else:
            # Keep a consistent descriptor if no face is detected, but track it via the face-detection rate.
            roi = gray
        roi = cv2.resize(roi, (64, 64), interpolation=cv2.INTER_AREA)
        descriptors.append(hog.compute(roi).flatten())
        gray_stats.extend([float(np.mean(roi)), float(np.std(roi))])
    cap.release()
    if not descriptors:
        raise ValueError(f"No usable frames in video: {path}")
    d = np.asarray(descriptors, dtype=np.float32)
    # Aggregate frame descriptors so each participant yields one fixed-length vector.
    feature = np.concatenate([d.mean(axis=0), d.std(axis=0),
                              np.asarray([len(descriptors) / max(len(positions), 1),
                                          np.mean(gray_stats), np.std(gray_stats)], dtype=np.float32)])
    return np.nan_to_num(feature, nan=0.0, posinf=0.0, neginf=0.0)

def read_transcript(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        raise ValueError(f"Empty transcript: {path}")
    return text
