
from pathlib import Path
import pandas as pd
from difflib import SequenceMatcher
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "english_depression.csv"
OUT_PATH = ROOT / "improved_duplicate_candidates.csv"

df = pd.read_csv(DATA_PATH)
df = df[["clean_text", "is_depression"]].dropna().copy()
df["clean_text"] = df["clean_text"].astype(str).str.strip()
df["is_depression"] = pd.to_numeric(
    df["is_depression"], errors="coerce"
)
df = df.dropna()
df = df[df["clean_text"].str.len() >= 80]
df = df[df["is_depression"].isin([0, 1])]
df["is_depression"] = df["is_depression"].astype(int)
df = df.drop_duplicates(subset=["clean_text"]).reset_index(drop=True)

# Reproduce the original stratified split for audit purposes.
train_idx, test_idx = train_test_split(
    df.index,
    test_size=0.20,
    random_state=42,
    stratify=df["is_depression"],
)
df["split"] = "train"
df.loc[test_idx, "split"] = "test"

# Find high TF-IDF similarity candidates across the two splits.
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    max_features=20000,
)
matrix = vectorizer.fit_transform(df["clean_text"])

train_positions = df.index[df["split"] == "train"].to_numpy()
test_positions = df.index[df["split"] == "test"].to_numpy()

valid = matrix.getnnz(axis=1) > 0
train_positions = [i for i in train_positions if valid[i]]
test_positions = [i for i in test_positions if valid[i]]

neighbors = NearestNeighbors(
    n_neighbors=3,
    metric="cosine",
    algorithm="brute",
    n_jobs=-1,
)
neighbors.fit(matrix[train_positions])
distances, indices = neighbors.kneighbors(matrix[test_positions])

records = []
seen = set()

for test_pos, row_distances, row_indices in zip(
    test_positions, distances, indices
):
    test_text = df.iloc[test_pos]["clean_text"]

    for distance, neighbor_idx in zip(row_distances, row_indices):
        train_pos = train_positions[int(neighbor_idx)]
        train_text = df.iloc[train_pos]["clean_text"]
        tfidf_similarity = 1 - float(distance)

        # Keep only high-similarity candidates for manual review.
        if tfidf_similarity < 0.85:
            continue

        char_similarity = SequenceMatcher(
            None, train_text, test_text, autojunk=False
        ).ratio()

        # Ignore pairs that are similar in topic but not wording.
        if char_similarity < 0.80:
            continue

        key = tuple(sorted((train_pos, test_pos)))
        if key in seen:
            continue
        seen.add(key)

        records.append({
            "tfidf_similarity": round(tfidf_similarity, 4),
            "char_similarity": round(char_similarity, 4),
            "train_label": int(df.iloc[train_pos]["is_depression"]),
            "test_label": int(df.iloc[test_pos]["is_depression"]),
            "train_length": len(train_text),
            "test_length": len(test_text),
            "train_text": train_text,
            "test_text": test_text,
        })

result = pd.DataFrame(records)

if not result.empty:
    result = result.sort_values(
        ["char_similarity", "tfidf_similarity"],
        ascending=False,
    )

result.to_csv(OUT_PATH, index=False)

print("Posts checked:", len(df))
print("Train posts:", len(train_positions))
print("Test posts:", len(test_positions))
print("Likely near-duplicate candidates:", len(result))
print("Saved report:", OUT_PATH)
print("Review candidates manually; these scores do not prove duplication.")
