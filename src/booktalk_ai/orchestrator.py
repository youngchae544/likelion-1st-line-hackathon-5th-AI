from __future__ import annotations

from dataclasses import asdict
from typing import Protocol

from .graph import GraphIndex
from .models import AnswerResult, Document, Evidence, GraphEdge, GraphNode, Route, Tool, Verification
from .planner import BoundedPlanner, QueryRouter
from .retrieval import HybridRetriever
from .text import first_sentence
from .verification import CITATION_PATTERN, EvidenceVerifier


class AnswerGenerator(Protocol):
    def generate(self, query: str, evidence: list[Evidence]) -> str: ...


class ExtractiveAnswerGenerator:
    """Grounded default generator using verbatim evidence excerpts."""

    def generate(self, query: str, evidence: list[Evidence]) -> str:
        selected = evidence[:4]
        if not selected:
            return "확인할 수 있는 근거가 부족해 답변하지 않았습니다."
        lines = ["확인된 근거를 바탕으로 답변드릴게요."]
        for item in selected:
            lines.append(f"- {first_sentence(item.text)} [{item.id}]")
        return "\n".join(lines)


class RagOrchestrator:
    def __init__(self, max_plan_steps: int = 4, max_evidence: int = 8,
                 minimum_confidence: float = 0.35, generator: AnswerGenerator | None = None) -> None:
        self.router = QueryRouter()
        self.planner = BoundedPlanner(max_plan_steps)
        self.retriever = HybridRetriever()
        self.generator = generator or ExtractiveAnswerGenerator()
        self.verifier = EvidenceVerifier()
        self.max_evidence = max_evidence
        self.minimum_confidence = minimum_confidence

    def answer(self, query: str, documents: list[Document], nodes: list[GraphNode],
               edges: list[GraphEdge], requested_route: Route | None = None) -> AnswerResult:
        route = requested_route or self.router.route(query)
        plan = self.planner.create(query, route, documents)
        graph = GraphIndex(nodes, edges)
        collected: list[Evidence] = []
        for step in plan:
            if step.tool == Tool.HYBRID_SEARCH:
                collected.extend(self.retriever.search(query, documents, self.max_evidence))
            elif step.tool == Tool.GRAPH_LOCAL:
                collected.extend(graph.local_search(query, self.max_evidence))
            elif step.tool == Tool.GRAPH_GLOBAL:
                collected.extend(graph.global_search(query, self.max_evidence))
            elif step.tool == Tool.PROFILE_LOOKUP:
                collected.extend(self._by_kind(documents, "PROFILE"))
            elif step.tool == Tool.HISTORY_LOOKUP:
                collected.extend(self._by_kind(documents, "HISTORY"))
        evidence = self._deduplicate(collected)[: self.max_evidence]
        retrieval_confidence = sum(item.score for item in evidence[:4]) / max(min(len(evidence), 4), 1)
        if not evidence or retrieval_confidence < self.minimum_confidence:
            verification = Verification(False, 0.0, 0.0, ("insufficient_evidence",))
            return AnswerResult("확인할 수 있는 근거가 부족해 답변하지 않았습니다.", route, (),
                                retrieval_confidence, True, tuple(plan), tuple(evidence), verification,
                                {"reason": "INSUFFICIENT_EVIDENCE"})

        answer = self.generator.generate(query, evidence)
        verification = self.verifier.verify(answer, evidence)
        confidence = min(retrieval_confidence, verification.evidence_coverage, verification.citation_validity)
        abstained = not verification.passed
        if abstained:
            answer = "검색된 근거끼리 일치하지 않아 답변을 보류했습니다. 보호자 또는 콘텐츠 검수자의 확인이 필요합니다."
        citations = tuple(dict.fromkeys(CITATION_PATTERN.findall(answer))) if not abstained else ()
        return AnswerResult(answer, route, citations, confidence, abstained, tuple(plan), tuple(evidence),
                            verification, {"evidenceCount": len(evidence), "requestedRoute": requested_route is not None})

    @staticmethod
    def _by_kind(documents: list[Document], kind: str) -> list[Evidence]:
        return [Evidence(document.id, document.text, kind, 0.95, document.metadata)
                for document in documents if document.kind.upper() == kind]

    @staticmethod
    def _deduplicate(evidence: list[Evidence]) -> list[Evidence]:
        best: dict[str, Evidence] = {}
        for item in evidence:
            if item.id not in best or item.score > best[item.id].score:
                best[item.id] = item
        return sorted(best.values(), key=lambda item: (-item.score, item.id))

    def preview_route(self, query: str, documents: list[Document]) -> dict:
        route = self.router.route(query)
        return {"route": route.value, "plan": [asdict(step) for step in self.planner.create(query, route, documents)]}
