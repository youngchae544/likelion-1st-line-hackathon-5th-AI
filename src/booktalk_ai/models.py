from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Route(str, Enum):
    HYBRID = "HYBRID"
    GRAPH_LOCAL = "GRAPH_LOCAL"
    GRAPH_GLOBAL = "GRAPH_GLOBAL"
    PLAN = "PLAN"


class Tool(str, Enum):
    PROFILE_LOOKUP = "PROFILE_LOOKUP"
    HISTORY_LOOKUP = "HISTORY_LOOKUP"
    HYBRID_SEARCH = "HYBRID_SEARCH"
    GRAPH_LOCAL = "GRAPH_LOCAL"
    GRAPH_GLOBAL = "GRAPH_GLOBAL"


@dataclass(frozen=True)
class Document:
    id: str
    text: str
    title: str = ""
    kind: str = "CONTENT"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GraphNode:
    id: str
    label: str
    type: str
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GraphEdge:
    source: str
    target: str
    type: str
    weight: float = 1.0


@dataclass(frozen=True)
class Evidence:
    id: str
    text: str
    source: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PlanStep:
    order: int
    tool: Tool
    reason: str


@dataclass(frozen=True)
class Verification:
    passed: bool
    citation_validity: float
    evidence_coverage: float
    unsupported_claims: tuple[str, ...] = ()


@dataclass(frozen=True)
class AnswerResult:
    answer: str
    route: Route
    citations: tuple[str, ...]
    confidence: float
    abstained: bool
    plan: tuple[PlanStep, ...]
    evidence: tuple[Evidence, ...]
    verification: Verification
    diagnostics: dict[str, Any] = field(default_factory=dict)
