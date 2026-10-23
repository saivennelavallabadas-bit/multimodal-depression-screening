
import json
from pathlib import Path

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

# Dataset metadata
DATA_PATH = Path(
    "data/mmpsy-data-main/data/wav_select.json"
)
OUTPUT_DIR = Path("artifacts/mental_perceiver_text")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load participant metadata
with DATA_PATH.open("r", encoding="utf-8") as file:
    participants = json.load(file)

rows = []

for participant in participants:
    phq = participant.get("phq-label", {})
    score = phq.get("score")
    audios = participant.get("audios", {})

    if score is None or not isinstance(audios, dict):
        continue

    # PHQ-9 >= 5: mild or greater symptoms
    # PHQ-9 < 5: minimal symptoms
    try:
        label = int(float(score) >= 5)
    except (TypeError, ValueError):
        continue

    # Combine all transcript segments for each participant
    text = " ".join(
        str(segment).strip()
        for segment in audios.values()
        if isinstance(segment, str) and segment.strip()
    )

    if text:
        rows.append({
            "user_id": participant.get("user_id"),
            "text": text,
            "phq_score": float(score),
            "label": label,
        })

df = pd.DataFrame(rows).drop_duplicates("user_id")

if len(df) < 20 or df["label"].nunique() != 2:
    raise ValueError(
        "Not enough valid participants or only one class found."
    )

print(f"Participants with usable text: {len(df)}")
print("\nLabel counts:")
print(df["label"].value_counts().sort_index())
print("Label 0 = PHQ-9 below 5")
print("Label 1 = PHQ-9 of 5 or above")

# Split participants before fitting the text vectorizer
train_df, test_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["label"],
)

model = Pipeline([
    ("tfidf", TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        min_df=2,
        max_features=20000,
        sublinear_tf=True,
    )),
    ("classifier", LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
    )),
])

model.fit(train_df["text"], train_df["label"])
predictions = model.predict(test_df["text"])

metrics = {
    "test_participants": int(len(test_df)),
    "accuracy": float(accuracy_score(
        test_df["label"], predictions
    )),
    "precision": float(precision_score(
        test_df["label"], predictions, zero_division=0
    )),
    "recall": float(recall_score(
        test_df["label"], predictions, zero_division=0
    )),
    "f1": float(f1_score(
        test_df["label"], predictions, zero_division=0
    )),
}

print("\nTest results:")
for name, value in metrics.items():
    if isinstance(value, float):
        print(f"{name}: {value:.4f}")
    else:
        print(f"{name}: {value}")

print("\nClassification report:")
print(classification_report(
    test_df["label"],
    predictions,
    target_names=["PHQ-9 < 5", "PHQ-9 >= 5"],
    zero_division=0,
))

print("Confusion matrix:")
print(confusion_matrix(test_df["label"], predictions))

# Save reproducible results and the trained model
import joblib

joblib.dump(model, OUTPUT_DIR / "text_model.joblib")

with (OUTPUT_DIR / "metrics.json").open(
    "w", encoding="utf-8"
) as file:
    json.dump(metrics, file, indent=2)

df[["user_id", "phq_score", "label"]].to_csv(
    OUTPUT_DIR / "participant_labels.csv",
    index=False,
)

print(f"\nModel and metrics saved to: {OUTPUT_DIR}")
print("Research baseline only; not a diagnostic tool.")
