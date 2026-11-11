from __future__ import annotations

from .models import Document, PlanStep, Route, Tool
from .text import tokenize


class QueryRouter:
    GLOBAL_MARKERS = ("전체", "주제", "경향", "공통", "전반", "요약")
    RELATION_MARKERS = ("관계", "비슷", "유사", "차이", "연결", "어떤 상황", "어디에서")
    PLAN_MARKERS = ("아이", "연령", "학습 이력", "기록", "추천", "어떻게", "계획", "적합", "단계")

    def route(self, query: str) -> Route:
        lowered = query.lower()
        plan_hits = sum(marker in lowered for marker in self.PLAN_MARKERS)
        if plan_hits >= 2 or (plan_hits >= 1 and len(tokenize(query)) >= 10):
            return Route.PLAN
        if any(marker in lowered for marker in self.GLOBAL_MARKERS):
            return Route.GRAPH_GLOBAL
        if any(marker in lowered for marker in self.RELATION_MARKERS):
            return Route.GRAPH_LOCAL
        return Route.HYBRID


class BoundedPlanner:
    def __init__(self, max_steps: int = 4) -> None:
        if max_steps < 1 or max_steps > 8:
            raise ValueError("max_steps must be between 1 and 8")
        self.max_steps = max_steps

    def create(self, query: str, route: Route, documents: list[Document]) -> list[PlanStep]:
        if route == Route.HYBRID:
            return [PlanStep(1, Tool.HYBRID_SEARCH, "질문과 직접 관련된 검수 문서를 검색")]
        if route == Route.GRAPH_LOCAL:
            return [PlanStep(1, Tool.GRAPH_LOCAL, "표현과 주변 엔티티 관계를 탐색"),
                    PlanStep(2, Tool.HYBRID_SEARCH, "관계 근거가 되는 원문을 확인")][: self.max_steps]
        if route == Route.GRAPH_GLOBAL:
            return [PlanStep(1, Tool.GRAPH_GLOBAL, "전체 지식 군집을 요약"),
                    PlanStep(2, Tool.HYBRID_SEARCH, "주요 군집의 원문 근거를 확인")][: self.max_steps]

        kinds = {document.kind.upper() for document in documents}
        steps: list[PlanStep] = []
        if "PROFILE" in kinds:
            steps.append(PlanStep(len(steps) + 1, Tool.PROFILE_LOOKUP, "연령과 설명 언어 확인"))
        if "HISTORY" in kinds:
            steps.append(PlanStep(len(steps) + 1, Tool.HISTORY_LOOKUP, "이전 학습 행동과 어려움 확인"))
        steps.append(PlanStep(len(steps) + 1, Tool.HYBRID_SEARCH, "질문에 직접 답하는 검수 콘텐츠 검색"))
        graph_tool = Tool.GRAPH_GLOBAL if any(marker in query for marker in QueryRouter.GLOBAL_MARKERS) else Tool.GRAPH_LOCAL
        steps.append(PlanStep(len(steps) + 1, graph_tool, "표현 관계를 교차 확인"))
        return steps[: self.max_steps]
