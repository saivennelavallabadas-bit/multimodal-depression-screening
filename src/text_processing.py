
import re
from collections import Counter


def extract_text_features(text: str) -> dict:
    """
    Extract basic linguistic features from text.
    These features do not diagnose or predict depression.
    """

    if not isinstance(text, str):
        raise TypeError("Input must be text.")

    cleaned_text = re.sub(r"\s+", " ", text).strip()

    if not cleaned_text:
        raise ValueError("Please enter some text for analysis.")

    words = re.findall(r"\b[\w']+\b", cleaned_text.lower())
    sentences = [
        sentence.strip()
        for sentence in re.split(r"[.!?]+", cleaned_text)
        if sentence.strip()
    ]

    word_count = len(words)
    sentence_count = len(sentences)
    unique_words = len(set(words))

    word_lengths = [len(word) for word in words]
    word_frequency = Counter(words)

    return {
        "character_count": len(cleaned_text),
        "word_count": word_count,
        "sentence_count": sentence_count,
        "unique_word_count": unique_words,
        "average_word_length": round(
            sum(word_lengths) / word_count, 2
        ),
        "average_words_per_sentence": round(
            word_count / sentence_count, 2
        ),
        "top_10_common_words": dict(
            word_frequency.most_common(10)
        ),
    }
