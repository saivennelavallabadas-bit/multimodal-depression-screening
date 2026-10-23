
from pathlib import Path
import json

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import joblib

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "english_depression.csv"
PAIRS_PATH = ROOT / "verified_duplicate_review.csv"

OUT_DIR = ROOT / "artifacts" / "english_text_grouped"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load and clean the original dataset.
df = pd.read_csv(DATA_PATH)
df = df[["clean_text", "is_depression"]].dropna().copy()
df["clean_text"] = df["clean_text"].astype(str).str.strip()
df["is_depression"] = pd.to_numeric(
    df["is_depression"], errors="coerce"
)
df = df.dropna()
df = df[df["clean_text"].str.len() > 0]
df = df[df["is_depression"].isin([0, 1])]
df["is_depression"] = df["is_depression"].astype(int)
df = df.drop_duplicates(subset=["clean_text"]).reset_index(drop=True)

# 2. Build groups from conservative duplicate candidates.
# Only long posts with high character similarity are grouped.
parent = list(range(len(df)))
text_to_idx = {
    text: i for i, text in enumerate(df["clean_text"])
}

def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x

def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb:
        parent[rb] = ra

pairs = pd.read_csv(PAIRS_PATH)

required = {
    "text_a", "text_b", "char_similarity",
    "length_a", "length_b"
}
if not required.issubset(pairs.columns):
    raise ValueError(
        f"Review CSV must contain columns: {sorted(required)}"
    )

grouped_pairs = 0

for _, row in pairs.iterrows():
    a_text = str(row["text_a"]).strip()
    b_text = str(row["text_b"]).strip()

    # Conservative rules reduce false matches on short posts.
    if float(row["char_similarity"]) < 0.90:
        continue
    if min(len(a_text), len(b_text)) < 80:
        continue
    if a_text not in text_to_idx or b_text not in text_to_idx:
        continue

    union(text_to_idx[a_text], text_to_idx[b_text])
    grouped_pairs += 1

df["group_id"] = [find(i) for i in range(len(df))]

# 3. Split by group so identified duplicates cannot cross splits.
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42,
)

train_idx, test_idx = next(
    splitter.split(
        df["clean_text"],
        df["is_depression"],
        groups=df["group_id"],
    )
)

train_df = df.iloc[train_idx]
test_df = df.iloc[test_idx]

X_train = train_df["clean_text"]
y_train = train_df["is_depression"]
X_test = test_df["clean_text"]
y_test = test_df["is_depression"]

# 4. Train the same model family on the new split.
model = Pipeline([
    ("tfidf", TfidfVectorizer(
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

model.fit(X_train, y_train)
predictions = model.predict(X_test)

metrics = {
    "dataset_rows": int(len(df)),
    "train_rows": int(len(train_df)),
    "test_rows": int(len(test_df)),
    "groups": int(df["group_id"].nunique()),
    "candidate_pairs_grouped": int(grouped_pairs),
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
        y_test, predictions
    ).tolist(),
    "test_label_counts": {
        str(k): int(v)
        for k, v in y_test.value_counts().sort_index().items()
    },
}

joblib.dump(model, OUT_DIR / "english_text_model_grouped.joblib")

with open(OUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

print("\nGrouped-split evaluation")
print(json.dumps(metrics, indent=2))
print("\nClassification report:")
print(classification_report(y_test, predictions, zero_division=0))
print("\nSaved model and metrics to:", OUT_DIR)
print(
    "\nImportant: this groups only candidates found by the earlier "
    "nearest-neighbor review. It does not guarantee that every "
    "near-duplicate in the dataset has been found."
)
