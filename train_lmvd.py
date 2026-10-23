
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "processed_lmvd.npz"
SPLIT_DIR = ROOT / "data" / "splits"
OUT_DIR = ROOT / "artifacts" / "lmvd_baseline"


def make_model():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        )),
    ])


def calculate_metrics(y_true, probabilities):
    predictions = (probabilities >= 0.5).astype(int)

    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(
            y_true, predictions, zero_division=0
        )),
        "recall": float(recall_score(
            y_true, predictions, zero_division=0
        )),
        "f1": float(f1_score(
            y_true, predictions, zero_division=0
        )),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "confusion_matrix": confusion_matrix(
            y_true, predictions, labels=[0, 1]
        ).tolist(),
    }


def load_split(name, sample_ids, labels):
    path = SPLIT_DIR / f"{name}.csv"
    df = pd.read_csv(path)

    required = {"sample_id", "label"}
    if not required.issubset(df.columns):
        raise ValueError(f"{path} must contain {sorted(required)}")

    ids = df["sample_id"].astype(str).to_numpy()
    y = df["label"].astype(int).to_numpy()

    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicate sample IDs in {name} split")

    if not set(np.unique(y)).issubset({0, 1}):
        raise ValueError(f"Invalid labels in {name} split")

    index_by_id = {sid: i for i, sid in enumerate(sample_ids)}

    missing = set(ids) - set(index_by_id)
    if missing:
        raise ValueError(f"Unknown sample IDs in {name}: {list(missing)[:5]}")

    indices = np.array([index_by_id[sid] for sid in ids])

    if not np.array_equal(y, labels[indices]):
        raise ValueError(f"Manifest labels disagree with NPZ labels in {name}")

    return indices, y


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Processed dataset not found: {DATA_PATH}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with np.load(DATA_PATH, allow_pickle=False) as data:
        audio = data["audio"].astype(np.float32)
        visual = data["visual"].astype(np.float32)
        labels = data["labels"].astype(int)
        sample_ids = data["sample_ids"].astype(str)

    if not (
        len(audio) == len(visual) == len(labels) == len(sample_ids)
    ):
        raise ValueError("Audio, visual, labels, and IDs have different sizes")

    if not np.isfinite(audio).all() or not np.isfinite(visual).all():
        raise ValueError("Features contain NaN or infinite values")

    train_idx, y_train = load_split("train", sample_ids, labels)
    val_idx, y_val = load_split("val", sample_ids, labels)
    test_idx, y_test = load_split("test", sample_ids, labels)

    # Verify that no sample appears in more than one split.
    assert not (set(train_idx) & set(val_idx))
    assert not (set(train_idx) & set(test_idx))
    assert not (set(val_idx) & set(test_idx))

    # Each sample has both audio and visual features.
    features = {
        "audio": audio,
        "visual": visual,
        "fusion": np.concatenate([audio, visual], axis=1),
    }

    results = {
        "dataset": "LMVD pre-extracted audio and visual features",
        "warning": (
            "Research baseline only, not a diagnostic tool. "
            "Sample-level splitting was used; participant-independent "
            "evaluation has not been established."
        ),
        "split_counts": {
            "train": len(train_idx),
            "validation": len(val_idx),
            "test": len(test_idx),
        },
        "models": {},
    }

    models_dir = OUT_DIR / "models"
    models_dir.mkdir(exist_ok=True)

    for name, X in features.items():
        print(f"\nTraining {name} model...")

        X_train = X[train_idx]
        X_val = X[val_idx]
        X_test = X[test_idx]

        # Out-of-fold training probabilities are used only to estimate
        # the training-set baseline. No validation/test samples enter CV.
        cv = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=42,
        )

        oof_probabilities = cross_val_predict(
            make_model(),
            X_train,
            y_train,
            cv=cv,
            method="predict_proba",
        )[:, 1]

        model = make_model()
        model.fit(X_train, y_train)

        val_probabilities = model.predict_proba(X_val)[:, 1]
        test_probabilities = model.predict_proba(X_test)[:, 1]

        val_metrics = calculate_metrics(y_val, val_probabilities)
        test_metrics = calculate_metrics(y_test, test_probabilities)

        joblib.dump(model, models_dir / f"{name}_model.joblib")

        results["models"][name] = {
            "validation": val_metrics,
            "test": test_metrics,
            "training_oof_accuracy": float(
                accuracy_score(y_train, oof_probabilities >= 0.5)
            ),
        }

        print("Validation metrics:")
        print(json.dumps(val_metrics, indent=2))
        print("Test metrics:")
        print(json.dumps(test_metrics, indent=2))

    with open(OUT_DIR / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print("\nTraining completed.")
    print(f"Results saved to: {OUT_DIR / 'metrics.json'}")
    print(f"Saved models to: {models_dir}")


if __name__ == "__main__":
    main()
