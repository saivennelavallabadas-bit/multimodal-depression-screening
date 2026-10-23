
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "processed_lmvd.npz"
SPLIT_DIR = ROOT / "data" / "splits"
OUT_DIR = ROOT / "artifacts" / "audio_comparison"


def make_models():
    # Preprocessing is fitted only on the training data.
    # Feature selection is tested as part of the logistic regression pipeline.
    return {
        "logistic_regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("select", SelectPercentile(score_func=f_classif, percentile=75)),
            ("model", LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                random_state=42,
            )),
        ]),
        "svm_rbf": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", SVC(
                kernel="rbf",
                C=1.0,
                probability=True,
                class_weight="balanced",
                random_state=42,
            )),
        ]),
        "random_forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced",
                min_samples_leaf=3,
                random_state=42,
                n_jobs=-1,
            )),
        ]),
        "extra_trees": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", ExtraTreesClassifier(
                n_estimators=300,
                class_weight="balanced",
                min_samples_leaf=3,
                random_state=42,
                n_jobs=-1,
            )),
        ]),
    }


def metrics(y_true, probabilities):
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


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with np.load(DATA_PATH, allow_pickle=False) as data:
        audio = data["audio"].astype(np.float32)
        labels = data["labels"].astype(int)
        sample_ids = data["sample_ids"].astype(str)

    # Map sample IDs to rows in the processed feature array.
    id_to_index = {sid: i for i, sid in enumerate(sample_ids)}

    def get_split(name):
        df = pd.read_csv(SPLIT_DIR / f"{name}.csv")
        ids = df["sample_id"].astype(str).to_numpy()
        y = df["label"].astype(int).to_numpy()

        if len(set(ids)) != len(ids):
            raise ValueError(f"Duplicate sample IDs in {name} split.")

        missing = set(ids) - set(id_to_index)
        if missing:
            raise ValueError(f"Unknown IDs in {name}: {list(missing)[:5]}")

        indices = np.array([id_to_index[sid] for sid in ids])

        if not np.array_equal(y, labels[indices]):
            raise ValueError(f"Label mismatch in {name} split.")

        return audio[indices], y, set(ids)

    X_train, y_train, train_ids = get_split("train")
    X_val, y_val, val_ids = get_split("val")
    X_test, y_test, test_ids = get_split("test")

    if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
        raise ValueError("Sample overlap detected across splits.")

    if not (
        np.isfinite(X_train).all()
        and np.isfinite(X_val).all()
        and np.isfinite(X_test).all()
    ):
        raise ValueError("Audio features contain non-finite values.")

    print("Audio-only model comparison")
    print(f"Training samples: {len(y_train)}")
    print(f"Validation samples: {len(y_val)}")
    print(f"Test samples: {len(y_test)}")
    print(f"Audio feature dimensions: {X_train.shape[1]}")

    models = make_models()
    validation_results = {}
    fitted_models = {}

    # Compare models on validation ROC-AUC, not test performance.
    for name, model in models.items():
        print(f"\nFitting {name}...")
        model.fit(X_train, y_train)

        probabilities = model.predict_proba(X_val)[:, 1]
        result = metrics(y_val, probabilities)

        validation_results[name] = result
        fitted_models[name] = model

        print(json.dumps(result, indent=2))

    comparison = pd.DataFrame([
        {
            "model": name,
            **result,
        }
        for name, result in validation_results.items()
    ]).sort_values("roc_auc", ascending=False)

    comparison.to_csv(OUT_DIR / "validation_comparison.csv", index=False)

    print("\nValidation comparison:")
    print(comparison[
        ["model", "accuracy", "precision", "recall", "f1", "roc_auc"]
    ].to_string(index=False))

    best_name = comparison.iloc[0]["model"]
    print(f"\nSelected using validation ROC-AUC: {best_name}")

    # After model selection, refit that model on train + validation.
    # The test set has not been used for model selection.
    best_model = make_models()[best_name]
    X_train_val = np.concatenate([X_train, X_val], axis=0)
    y_train_val = np.concatenate([y_train, y_val], axis=0)
    best_model.fit(X_train_val, y_train_val)

    # Final evaluation on the held-out test set.
    test_probabilities = best_model.predict_proba(X_test)[:, 1]
    test_results = metrics(y_test, test_probabilities)
    predictions = (test_probabilities >= 0.5).astype(int)

    joblib.dump(best_model, OUT_DIR / "best_audio_model.joblib")

    results = {
        "note": (
            "Research baseline only; not a diagnostic tool. "
            "The model was selected using validation ROC-AUC. "
            "The held-out test set was evaluated once after selection. "
            "Splits are sample-level; participant-independent evaluation "
            "has not been established."
        ),
        "selected_model": best_name,
        "selection_metric": "validation ROC-AUC",
        "validation_results": validation_results,
        "final_test_results": test_results,
        "test_classification_report": classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Normal", "Depression"],
            zero_division=0,
            output_dict=True,
        ),
    }

    with open(OUT_DIR / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\nFINAL TEST RESULTS — selected model")
    print(json.dumps(test_results, indent=2))
    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Normal", "Depression"],
        zero_division=0,
    ))

    print(f"\nResults saved in: {OUT_DIR}")
    print("Best model: best_audio_model.joblib")


if __name__ == "__main__":
    main()
