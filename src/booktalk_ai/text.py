from __future__ import annotations

import hashlib
import math
import re

TOKEN_PATTERN = re.compile(r"[가-힣A-Za-z0-9]+")
KOREAN_STOPWORDS = {"은", "는", "이", "가", "을", "를", "에", "의", "와", "과", "좀", "알려줘", "해줘"}


def tokenize(text: str) -> list[str]:
    return [token for token in TOKEN_PATTERN.findall(text.lower()) if token not in KOREAN_STOPWORDS]


def semantic_vector(text: str, dimensions: int = 128) -> list[float]:
    """Dependency-free feature hashing vector used as a deterministic local semantic baseline."""
    compact = " ".join(tokenize(text))
    features = tokenize(text)
    for token in list(features):
        if len(token) >= 3:
            features.extend(token[i : i + 3] for i in range(len(token) - 2))
    vector = [0.0] * dimensions
    for feature in features or [compact]:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest[:4], "big") % dimensions
        sign = 1.0 if digest[4] & 1 else -1.0
        vector[index] += sign
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def first_sentence(text: str, max_length: int = 220) -> str:
    sentence = re.split(r"(?<=[.!?。])\s+|\n+", text.strip())[0]
    return sentence if len(sentence) <= max_length else sentence[: max_length - 1].rstrip() + "…"
