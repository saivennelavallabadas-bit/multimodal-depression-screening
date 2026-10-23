
import io

import librosa
import numpy as np
import soundfile as sf


def extract_audio_features(audio_bytes: bytes) -> dict:
    """Extract basic acoustic features from an uploaded audio file."""

    audio, sample_rate = sf.read(io.BytesIO(audio_bytes))

    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    if audio.size == 0:
        raise ValueError("The uploaded audio file is empty.")

    if not np.isfinite(audio).all():
        raise ValueError("The audio contains invalid numeric values.")

    # Convert to a consistent sample rate.
    target_rate = 16000
    if sample_rate != target_rate:
        audio = librosa.resample(
            audio.astype(np.float32),
            orig_sr=sample_rate,
            target_sr=target_rate,
        )
        sample_rate = target_rate

    audio = audio.astype(np.float32)

    if len(audio) < sample_rate // 2:
        raise ValueError("Please upload at least 0.5 seconds of audio.")

    # Extract acoustic descriptors.
    rms = librosa.feature.rms(y=audio)[0]
    zcr = librosa.feature.zero_crossing_rate(y=audio)[0]
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=13,
    )
    spectral_centroid = librosa.feature.spectral_centroid(
        y=audio,
        sr=sample_rate,
    )[0]

    features = {
        "duration_seconds": round(len(audio) / sample_rate, 2),
        "sample_rate_hz": int(sample_rate),
        "rms_mean": float(np.mean(rms)),
        "zero_crossing_rate_mean": float(np.mean(zcr)),
        "spectral_centroid_mean": float(np.mean(spectral_centroid)),
    }

    for index in range(mfcc.shape[0]):
        features[f"mfcc_{index + 1}_mean"] = float(
            np.mean(mfcc[index])
        )
        features[f"mfcc_{index + 1}_std"] = float(
            np.std(mfcc[index])
        )

    return features
