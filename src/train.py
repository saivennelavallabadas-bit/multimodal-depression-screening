import argparse
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.impute import SimpleImputer
from src.features import audio_features, video_features, read_transcript, resolve_path

def metrics(y_true, probabilities, threshold=0.5):
    pred = (np.asarray(probabilities) >= threshold).astype(int)
    result = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "confusion_matrix_labels_0_1": confusion_matrix(y_true, pred, labels=[0,1]).tolist(),
    }
    if len(set(y_true)) == 2:
        result["roc_auc"] = float(roc_auc_score(y_true, probabilities))
    else:
        result["roc_auc"] = None
    return result

def load_split(path):
    df = pd.read_csv(path, dtype={"participant_id": str})
    required = {"participant_id", "audio_path", "video_path", "transcript_path", "label"}
    if not required.issubset(df.columns):
        raise ValueError(f"{path} missing columns: {sorted(required - set(df.columns))}")
    if df["participant_id"].duplicated().any():
        raise ValueError(f"Duplicate participant in {path}")
    return df

def feature_matrix(df, root, modality):
    rows = []
    for i, row in df.iterrows():
        try:
            if modality == "audio":
                rows.append(audio_features(resolve_path(row.audio_path, root)))
            elif modality == "video":
                rows.append(video_features(resolve_path(row.video_path, root)))
            else:
                raise ValueError(modality)
        except Exception as e:
            raise RuntimeError(f"{modality} feature extraction failed for participant {row.participant_id}: {e}") from e
    return np.vstack(rows)

def build_model():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42))
    ])

def text_pipeline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1,2), min_df=1, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42))
    ])

def positive_probability(model, X):
    classes = list(model.classes_) if hasattr(model, "classes_") else list(model.named_steps["clf"].classes_)
    pos_idx = classes.index(1)
    return model.predict_proba(X)[:, pos_idx]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--splits-dir", default="data/splits")
    ap.add_argument("--out-dir", default="artifacts/run_001")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    manifest_path = Path(args.manifest).resolve()
    root = manifest_path.parent.parent
    split_dir = Path(args.splits_dir)
    train = load_split(split_dir / "train.csv")
    val = load_split(split_dir / "val.csv")
    test = load_split(split_dir / "test.csv")
    ids = [set(d.participant_id) for d in (train, val, test)]
    if ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2]:
        raise ValueError("Participant leakage detected across splits.")
    for part in (train, val, test):
        if set(part.participant_id) - set(pd.read_csv(manifest_path, dtype={"participant_id": str}).participant_id):
            raise ValueError("Split contains participants not found in manifest.")
    if any(len(set(d.label.astype(int))) != 2 for d in (train, val, test)):
        raise ValueError("Each split must contain both classes for this starter evaluation.")
    out = Path(args.out_dir)
    models_dir = out / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    print("Extracting fixed audio features...")
    Xa_tr = feature_matrix(train, root, "audio")
    Xa_va = feature_matrix(val, root, "audio")
    Xa_te = feature_matrix(test, root, "audio")
    print("Extracting fixed video features...")
    Xv_tr = feature_matrix(train, root, "video")
    Xv_va = feature_matrix(val, root, "video")
    Xv_te = feature_matrix(test, root, "video")
    texts_tr = [read_transcript(resolve_path(p, root)) for p in train.transcript_path]
    texts_va = [read_transcript(resolve_path(p, root)) for p in val.transcript_path]
    texts_te = [read_transcript(resolve_path(p, root)) for p in test.transcript_path]
    ytr, yva, yte = (d.label.astype(int).to_numpy() for d in (train, val, test))

    # OOF predictions are generated only inside the training split for fusion training.
    min_class = int(pd.Series(ytr).value_counts().min())
    n_splits = min(5, min_class)
    if n_splits < 2:
        raise ValueError("Training set needs at least 2 examples in each class for OOF fusion. Use more participants.")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
    oof = {}
    final_models = {}
    val_probs = {}
    test_probs = {}

    for name, Xtr, Xva, Xte in [
        ("audio", Xa_tr, Xa_va, Xa_te),
        ("video", Xv_tr, Xv_va, Xv_te),
    ]:
        print(f"Training {name} baseline...")
        oof_prob = cross_val_predict(build_model(), Xtr, ytr, cv=cv, method="predict_proba")[:, 1]
        model = build_model().fit(Xtr, ytr)
        final_models[name] = model
        oof[name] = oof_prob
        val_probs[name] = positive_probability(model, Xva)
        test_probs[name] = positive_probability(model, Xte)
        joblib.dump(model, models_dir / f"{name}_model.joblib")

    print("Training text baseline...")
    oof_text = cross_val_predict(text_pipeline(), texts_tr, ytr, cv=cv, method="predict_proba")[:, 1]
    text_model = text_pipeline().fit(texts_tr, ytr)
    final_models["text"] = text_model
    oof["text"] = oof_text
    val_probs["text"] = positive_probability(text_model, texts_va)
    test_probs["text"] = positive_probability(text_model, texts_te)
    joblib.dump(text_model, models_dir / "text_model.joblib")

    # Fusion learns from OOF base-model predictions, not in-sample predictions.
    fusion_X_train = np.column_stack([oof["audio"], oof["video"], oof["text"]])
    fusion = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=args.seed)
    fusion.fit(fusion_X_train, ytr)
    fusion_val_X = np.column_stack([val_probs["audio"], val_probs["video"], val_probs["text"]])
    fusion_test_X = np.column_stack([test_probs["audio"], test_probs["video"], test_probs["text"]])
    val_probs["fusion"] = positive_probability(fusion, fusion_val_X)
    test_probs["fusion"] = positive_probability(fusion, fusion_test_X)
    joblib.dump(fusion, models_dir / "fusion_model.joblib")

    results = {"warning": "Research baseline only; not a diagnostic tool. Metrics are valid only if the supplied dataset, labels, and participant mapping are valid.",
               "seed": args.seed, "counts": {"train": len(train), "validation": len(val), "test": len(test)},
               "participant_ids": {"train": train.participant_id.tolist(), "validation": val.participant_id.tolist(), "test": test.participant_id.tolist()},
               "validation_metrics": {}, "test_metrics": {}}
    for name in ["audio", "video", "text", "fusion"]:
        results["validation_metrics"][name] = metrics(yva, val_probs[name])
        results["test_metrics"][name] = metrics(yte, test_probs[name])
    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    metadata = {
        "manifest": str(manifest_path), "splits_dir": str(split_dir.resolve()),
        "seed": args.seed, "n_oof_folds": n_splits,
        "model_types": {"audio": "MFCC/spectral summary + scaled logistic regression",
                        "video": "sampled face-region HOG summary + scaled logistic regression",
                        "text": "TF-IDF + logistic regression",
                        "fusion": "logistic regression on out-of-fold branch probabilities"},
        "test_set_used_for_training": False,
        "threshold": 0.5,
        "notes": ["Validation metrics are for model development; test metrics are held-out estimates.",
                  "This code does not tune thresholds on the test set.",
                  "The video descriptors are not validated emotion or depression measures."]
    }
    with open(out / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print("\nTEST METRICS (actual measurements from the supplied files):")
    print(json.dumps(results["test_metrics"], indent=2))
    print(f"\nSaved models and metrics to: {out.resolve()}")

if __name__ == "__main__":
    main()
