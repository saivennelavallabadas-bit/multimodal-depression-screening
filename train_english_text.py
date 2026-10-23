
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "english_depression.csv"
OUTPUT_DIR = ROOT / "artifacts" / "english_text"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load the downloaded dataset
if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}\n"
        "Place english_depression.csv in the main project folder."
    )

df = pd.read_csv(DATA_PATH)

# 2. Check the required columns
required = {"clean_text", "is_depression"}
if not required.issubset(df.columns):
    raise ValueError(
        f"Expected columns {required}, but found {list(df.columns)}"
    )

# 3. Clean the data
df = df[["clean_text", "is_depression"]].dropna().copy()
df["clean_text"] = df["clean_text"].astype(str).str.strip()
df["is_depression"] = pd.to_numeric(
    df["is_depression"], errors="coerce"
)
df = df.dropna()
df = df[df["clean_text"].str.len() > 0]
df = df[df["is_depression"].isin([0, 1])]
df["is_depression"] = df["is_depression"].astype(int)
df = df.drop_duplicates(subset=["clean_text"])

# Check for exact duplicate text and conflicting labels
print("\nDuplicate check:")
print("Rows after exact-text deduplication:", len(df))
print("Duplicate texts remaining:", df["clean_text"].duplicated().sum())

conflicting = (
    df.groupby("clean_text")["is_depression"]
    .nunique()
    .gt(1)
    .sum()
)
print("Texts with conflicting labels:", int(conflicting))

print("\nChecking for highly similar posts is a separate step.")

if df["is_depression"].nunique() != 2:
    raise ValueError("The dataset must contain both label 0 and label 1.")

print("Dataset rows:", len(df))
print("Label counts:")
print(df["is_depression"].value_counts().sort_index())

# 4. Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"],
    df["is_depression"],
    test_size=0.20,
    random_state=42,
    stratify=df["is_depression"],
)

# 5. Train the English text classifier
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_features=20000,
        sublinear_tf=True,
        stop_words="english",
    )),
    ("classifier", LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
    )),
])

print("\nTraining the English text model...")
model.fit(X_train, y_train)

# 6. Evaluate on the held-out test set
predictions = model.predict(X_test)

metrics = {
    "dataset": "Reddit Depression Cleaned",
    "language": "English",
    "method": "TF-IDF + Logistic Regression",
    "training_samples": int(len(X_train)),
    "test_samples": int(len(X_test)),
    "accuracy": float(accuracy_score(y_test, predictions)),
    "precision": float(precision_score(
        y_test, predictions, zero_division=0
    )),
    "recall": float(recall_score(
        y_test, predictions, zero_division=0
    )),
    "f1": float(f1_score(
        y_test, predictions, zero_division=0
    )),
    "confusion_matrix": confusion_matrix(
        y_test, predictions, labels=[0, 1]
    ).tolist(),
    "label_meaning": (
        "Dataset-provided labels; not clinical diagnoses."
    ),
}

# 7. Save without overwriting the Chinese model
joblib.dump(model, OUTPUT_DIR / "english_text_model.joblib")

with (OUTPUT_DIR / "metrics.json").open(
    "w", encoding="utf-8"
) as file:
    json.dump(metrics, file, indent=2)

print("\nClassification report:")
print(classification_report(y_test, predictions, zero_division=0))

print("Evaluation metrics:")
print(json.dumps(metrics, indent=2))
print(f"\nModel saved to: {OUTPUT_DIR / 'english_text_model.joblib'}")
print(f"Metrics saved to: {OUTPUT_DIR / 'metrics.json'}")
