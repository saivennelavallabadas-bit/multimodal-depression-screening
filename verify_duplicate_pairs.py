
from pathlib import Path
from difflib import SequenceMatcher
import pandas as pd

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "near_duplicates_across_splits.csv"
OUTPUT = ROOT / "verified_duplicate_review.csv"

df = pd.read_csv(INPUT)

def compare_texts(row):
    a = str(row["text_a"]).strip()
    b = str(row["text_b"]).strip()

    ratio = SequenceMatcher(
        None, a, b, autojunk=False
    ).ratio()

    return pd.Series({
        "char_similarity": round(ratio, 4),
        "length_a": len(a),
        "length_b": len(b),
        "length_ratio": round(
            min(len(a), len(b)) / max(len(a), len(b)), 4
        ) if max(len(a), len(b)) else 1.0,
    })

details = df.apply(compare_texts, axis=1)
result = pd.concat([df, details], axis=1)

result = result.sort_values(
    "char_similarity", ascending=False
)

result.to_csv(OUTPUT, index=False)

print("Candidate pairs reviewed:", len(result))
print("\nTop 15 pairs by character similarity:")
print(result[
    ["char_similarity", "similarity", "length_a", "length_b"]
].head(15).to_string(index=False))

print("\nSaved full review to:", OUTPUT)
print("Character similarity is a screening measure, not proof.")
