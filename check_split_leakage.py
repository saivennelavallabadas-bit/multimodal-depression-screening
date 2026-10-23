
from pathlib import Path
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "english_depression.csv"
PAIRS_PATH = ROOT / "verified_duplicate_review.csv"

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

# Reproduce the grouping rules used by the grouped training script.
text_to_idx = {text: i for i, text in enumerate(df["clean_text"])}
parent = list(range(len(df)))

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
for _, row in pairs.iterrows():
    a = str(row["text_a"]).strip()
    b = str(row["text_b"]).strip()

    if float(row["char_similarity"]) < 0.90:
        continue
    if min(len(a), len(b)) < 80:
        continue
    if a in text_to_idx and b in text_to_idx:
        union(text_to_idx[a], text_to_idx[b])

groups = [find(i) for i in range(len(df))]

splitter = GroupShuffleSplit(
    n_splits=1, test_size=0.20, random_state=42
)
train_idx, test_idx = next(
    splitter.split(df["clean_text"], df["is_depression"], groups=groups)
)

train_texts = df.iloc[train_idx]["clean_text"]
test_texts = df.iloc[test_idx]["clean_text"]

# Check exact-text overlap.
train_set = set(train_texts)
test_set = set(test_texts)
exact_overlap = train_set.intersection(test_set)

print("Training rows:", len(train_texts))
print("Test rows:", len(test_texts))
print("Exact text overlap:", len(exact_overlap))

# Screen for remaining high-similarity cross-split pairs.
combined = pd.concat(
    [
        pd.DataFrame({"text": train_texts, "split": "train"}),
        pd.DataFrame({"text": test_texts, "split": "test"}),
    ],
    ignore_index=True,
)

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    max_features=20000,
)
matrix = vectorizer.fit_transform(combined["text"])

train_positions = combined.index[combined["split"] == "train"].to_numpy()
test_positions = combined.index[combined["split"] == "test"].to_numpy()

# Ignore posts that become empty after TF-IDF tokenization.
valid_positions = matrix.getnnz(axis=1) > 0
valid_test_positions = [
    i for i in test_positions if valid_positions[i]
]
valid_train_positions = [
    i for i in train_positions if valid_positions[i]
]

nearest = NearestNeighbors(
    n_neighbors=1, metric="cosine", algorithm="brute", n_jobs=-1
)
nearest.fit(matrix[valid_train_positions])

distances, indices = nearest.kneighbors(matrix[valid_test_positions])

findings = []
for test_pos, distance, neighbor_idx in zip(
    valid_test_positions, distances[:, 0], indices[:, 0]
):
    similarity = 1 - float(distance)
    if similarity >= 0.90:
        train_pos = valid_train_positions[int(neighbor_idx)]
        findings.append({
            "similarity": round(similarity, 4),
            "train_text": combined.iloc[train_pos]["text"],
            "test_text": combined.iloc[test_pos]["text"],
        })

out = ROOT / "remaining_cross_split_candidates.csv"
pd.DataFrame(findings).to_csv(out, index=False)

print("Cross-split nearest-neighbor candidates >= 0.90:", len(findings))
print("Saved candidate report:", out)
print("Review candidates manually; similarity does not prove duplication.")
