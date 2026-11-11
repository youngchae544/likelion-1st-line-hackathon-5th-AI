from __future__ import annotations

import math
from collections import Counter

from .models import Document, Evidence
from .text import cosine, semantic_vector, tokenize


class HybridRetriever:
    def __init__(self, lexical_weight: float = 0.55, semantic_weight: float = 0.45) -> None:
        self.lexical_weight = lexical_weight
        self.semantic_weight = semantic_weight

    def search(self, query: str, documents: list[Document], limit: int = 8) -> list[Evidence]:
        if not query.strip() or not documents:
            return []
        tokenized = [tokenize(document.title + " " + document.text) for document in documents]
        query_tokens = tokenize(query)
        average_length = sum(map(len, tokenized)) / max(len(tokenized), 1)
        document_frequency = Counter(token for tokens in tokenized for token in set(tokens))
        query_vector = semantic_vector(query)
        scored: list[Evidence] = []
        for document, tokens in zip(documents, tokenized):
            frequencies = Counter(tokens)
            lexical = 0.0
            for token in query_tokens:
                df = document_frequency.get(token, 0)
                idf = math.log(1 + (len(documents) - df + 0.5) / (df + 0.5))
                tf = frequencies.get(token, 0)
                denominator = tf + 1.5 * (1 - 0.75 + 0.75 * len(tokens) / max(average_length, 1))
                lexical += idf * (tf * 2.5 / denominator if denominator else 0)
            lexical_normalized = lexical / (lexical + 1.0)
            semantic = max(0.0, cosine(query_vector, semantic_vector(document.title + " " + document.text)))
            title_tokens = set(tokenize(document.title))
            exact_bonus = 0.15 if any(token in document.text.lower() for token in query_tokens) else 0.0
            title_bonus = 0.25 * len(title_tokens & set(query_tokens)) / max(len(set(query_tokens)), 1)
            canonical_bonus = 0.05 if document.kind.upper() == "CONTENT" else 0.0
            score = min(1.0, self.lexical_weight * lexical_normalized + self.semantic_weight * semantic
                        + exact_bonus + title_bonus + canonical_bonus)
            if score > 0:
                scored.append(Evidence(document.id, document.text, "HYBRID", score, document.metadata))
        return sorted(scored, key=lambda evidence: (-evidence.score, evidence.id))[:limit]
