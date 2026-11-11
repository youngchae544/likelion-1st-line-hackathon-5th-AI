from __future__ import annotations

import re

from .models import Evidence, Verification
from .text import tokenize

CITATION_PATTERN = re.compile(r"\[([^\]]+)]")


class EvidenceVerifier:
    def verify(self, answer: str, evidence: list[Evidence]) -> Verification:
        evidence_by_id = {item.id: item for item in evidence}
        citations = CITATION_PATTERN.findall(answer)
        valid = [citation for citation in citations if citation in evidence_by_id]
        citation_validity = len(valid) / max(len(citations), 1) if citations else 0.0
        claims = [claim.strip() for claim in re.split(r"(?<=[.!?])\s+|\n+", answer) if claim.strip()]
        supported = 0
        evaluated = 0
        unsupported: list[str] = []
        corpus_tokens = set(tokenize(" ".join(item.text for item in evidence)))
        for claim in claims:
            claim_without_citations = CITATION_PATTERN.sub("", claim)
            claim_tokens = set(tokenize(claim_without_citations))
            if not claim_tokens:
                continue
            evaluated += 1
            overlap = len(claim_tokens & corpus_tokens) / max(len(claim_tokens), 1)
            if overlap >= 0.35 or claim_without_citations.startswith("확인된 근거"):
                supported += 1
            else:
                unsupported.append(claim_without_citations[:160])
        coverage = supported / max(evaluated, 1)
        return Verification(bool(evidence) and citation_validity == 1.0 and coverage >= 0.75,
                            citation_validity, coverage, tuple(unsupported))
