
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "english_depression.csv"

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

train_idx, test_idx = train_test_split(
    df.index,
    test_size=0.20,
    random_state=42,
    stratify=df["is_depression"],
)

df["split"] = "train"
df.loc[test_idx, "split"] = "test"

# Find the nearest different text for each post.
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    max_features=20000,
)
matrix = vectorizer.fit_transform(df["clean_text"])

neighbors = NearestNeighbors(
    n_neighbors=2,
    metric="cosine",
    algorithm="brute",
    n_jobs=-1,
)
neighbors.fit(matrix)
distances, indices = neighbors.kneighbors(matrix)

pairs = []
seen_pairs = set()
threshold = 0.90

for i in range(len(df)):
    j = int(indices[i, 1])
    similarity = 1 - float(distances[i, 1])

    if similarity < threshold:
        continue

    # Keep each pair only once, regardless of direction.
    a, b = sorted((i, j))
    if a == b or (a, b) in seen_pairs:
        continue
    seen_pairs.add((a, b))

    pairs.append({
        "similarity": round(similarity, 4),
        "split_a": df.iloc[a]["split"],
        "split_b": df.iloc[b]["split"],
        "label_a": int(df.iloc[a]["is_depression"]),
        "label_b": int(df.iloc[b]["is_depression"]),
        "text_a": df.iloc[a]["clean_text"],
        "text_b": df.iloc[b]["clean_text"],
    })

result = pd.DataFrame(pairs)

if not result.empty:
    result = result.sort_values(
        "similarity", ascending=False
    ).reset_index(drop=True)
    cross_split = result[
        result["split_a"] != result["split_b"]
    ].copy()
else:
    cross_split = result.copy()

result.to_csv(ROOT / "near_duplicate_split_review.csv", index=False)
cross_split.to_csv(ROOT / "near_duplicates_across_splits.csv", index=False)

print("Unique posts:", len(df))
print("Unique candidate pairs:", len(result))
print("Unique candidates crossing train/test:", len(cross_split))
print("Full-text review: near_duplicate_split_review.csv")
print("Cross-split review: near_duplicates_across_splits.csv")
print("Review candidates manually; similarity is not proof of duplication.")
