
from pathlib import Path
import json
from difflib import SequenceMatcher

import joblib
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

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "english_depression.csv"
CANDIDATE_PATH = ROOT / "improved_duplicate_candidates.csv"

OUT_DIR = ROOT / "artifacts" / "english_text_grouped_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load and clean the dataset.
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

# 2. Create duplicate groups using the improved audit candidates.
parent = list(range(len(df)))
text_to_idx = {text: i for i, text in enumerate(df["clean_text"])}

def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x

def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb:
        parent[rb] = ra

pairs = pd.read_csv(CANDIDATE_PATH)
required = {
    "train_text", "test_text",
    "char_similarity", "tfidf_similarity",
}
if not required.issubset(pairs.columns):
    raise ValueError(
        f"Candidate file must contain columns: {sorted(required)}"
    )

grouped_pairs = 0
skipped_pairs = 0

for _, row in pairs.iterrows():
    a_text = str(row["train_text"]).strip()
    b_text = str(row["test_text"]).strip()

    if a_text not in text_to_idx or b_text not in text_to_idx:
        skipped_pairs += 1
        continue

    char_sim = float(row["char_similarity"])
    tfidf_sim = float(row["tfidf_similarity"])

    # Conservative rule: group only strong wording matches.
    # Short posts are excluded to reduce false matches.
    if (
        char_sim >= 0.90
        and tfidf_sim >= 0.90
        and min(len(a_text), len(b_text)) >= 80
    ):
        union(text_to_idx[a_text], text_to_idx[b_text])
        grouped_pairs += 1
    else:
        skipped_pairs += 1

df["group_id"] = [find(i) for i in range(len(df))]

# 3. Split by group so identified duplicates stay together.
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

# 4. Train a fresh model.
model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_features=20000,
            sublinear_tf=True,
            stop_words="english",
        ),
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        ),
    ),
])

model.fit(X_train, y_train)
predictions = model.predict(X_test)

# 5. Calculate metrics.
metrics = {
    "dataset_rows": int(len(df)),
    "train_rows": int(len(train_df)),
    "test_rows": int(len(test_df)),
    "groups": int(df["group_id"].nunique()),
    "candidate_pairs_found": int(len(pairs)),
    "candidate_pairs_grouped": int(grouped_pairs),
    "candidate_pairs_skipped": int(skipped_pairs),
    "accuracy": float(accuracy_score(y_test, predictions)),
    "precision": float(
        precision_score(y_test, predictions, zero_division=0)
    ),
    "recall": float(
        recall_score(y_test, predictions, zero_division=0)
    ),
    "f1": float(f1_score(y_test, predictions, zero_division=0)),
    "confusion_matrix": confusion_matrix(
        y_test, predictions, labels=[0, 1]
    ).tolist(),
    "test_label_counts": {
        str(k): int(v)
        for k, v in y_test.value_counts().sort_index().items()
    },
    "limitations": [
        "Only candidate pairs found by the audit are grouped.",
        "This does not guarantee all near-duplicates were detected.",
        "Split is grouped but not stratified by class.",
        "Dataset labels are not clinical diagnoses.",
    ],
}

# 6. Save under a new directory; preserve previous models.
joblib.dump(model, OUT_DIR / "english_text_model_grouped_v2.joblib")

with open(OUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

with open(OUT_DIR / "classification_report.txt", "w", encoding="utf-8") as f:
    f.write(classification_report(
        y_test, predictions, zero_division=0
    ))

print("\nGrouped-split v2 evaluation")
print(json.dumps(metrics, indent=2))
print("\nClassification report:")
print(classification_report(y_test, predictions, zero_division=0))
print(f"\nSaved safely to: {OUT_DIR}")
