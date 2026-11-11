from __future__ import annotations

from collections import defaultdict, deque

from .models import Evidence, GraphEdge, GraphNode
from .text import tokenize


class GraphIndex:
    def __init__(self, nodes: list[GraphNode], edges: list[GraphEdge]) -> None:
        self.nodes = {node.id: node for node in nodes}
        self.edges = [edge for edge in edges if edge.source in self.nodes and edge.target in self.nodes]
        self.adjacency: dict[str, list[tuple[str, GraphEdge]]] = defaultdict(list)
        for edge in self.edges:
            self.adjacency[edge.source].append((edge.target, edge))
            self.adjacency[edge.target].append((edge.source, edge))

    def local_search(self, query: str, limit: int = 8, depth: int = 2) -> list[Evidence]:
        query_tokens = set(tokenize(query))
        seeds = sorted(self.nodes.values(), key=lambda node: -self._node_score(node, query_tokens))
        seeds = [node for node in seeds if self._node_score(node, query_tokens) > 0][:3]
        queue = deque((node.id, 0) for node in seeds)
        visited: set[str] = set()
        evidence: list[Evidence] = []
        while queue and len(evidence) < limit:
            node_id, current_depth = queue.popleft()
            if node_id in visited or current_depth > depth:
                continue
            visited.add(node_id)
            node = self.nodes[node_id]
            neighbors = self.adjacency.get(node_id, [])
            relation_text = ", ".join(f"{edge.type}→{self.nodes[target].label}" for target, edge in neighbors[:5])
            text = f"{node.type} '{node.label}'" + (f" 관계: {relation_text}" if relation_text else "")
            score = min(1.0, 0.45 + self._node_score(node, query_tokens) - current_depth * 0.12)
            evidence.append(Evidence(f"graph:{node.id}", text, "GRAPH_LOCAL", score,
                                     {"nodeId": node.id, "nodeType": node.type, **node.properties}))
            for target, _ in sorted(neighbors, key=lambda pair: -pair[1].weight):
                if target not in visited:
                    queue.append((target, current_depth + 1))
        return evidence

    def global_search(self, query: str, limit: int = 5) -> list[Evidence]:
        query_tokens = set(tokenize(query))
        communities = self._connected_components()
        results: list[Evidence] = []
        for index, component in enumerate(communities):
            nodes = [self.nodes[node_id] for node_id in component]
            labels = ", ".join(node.label for node in nodes[:12])
            types = sorted({node.type for node in nodes})
            score = max((self._node_score(node, query_tokens) for node in nodes), default=0.0)
            if self._is_global_query(query):
                score = max(score, min(0.75, 0.25 + len(nodes) / 20))
            results.append(Evidence(f"community:{index}", f"지식 군집({', '.join(types)}): {labels}",
                                    "GRAPH_GLOBAL", score, {"size": len(nodes)}))
        return sorted(results, key=lambda item: (-item.score, item.id))[:limit]

    def _connected_components(self) -> list[set[str]]:
        remaining = set(self.nodes)
        components: list[set[str]] = []
        while remaining:
            start = min(remaining)
            queue = [start]
            component: set[str] = set()
            while queue:
                current = queue.pop()
                if current in component:
                    continue
                component.add(current)
                queue.extend(target for target, _ in self.adjacency.get(current, []))
            remaining -= component
            components.append(component)
        return components

    def _node_score(self, node: GraphNode, query_tokens: set[str]) -> float:
        tokens = set(tokenize(node.label + " " + " ".join(map(str, node.properties.values()))))
        return len(tokens & query_tokens) / max(len(query_tokens), 1)

    @staticmethod
    def _is_global_query(query: str) -> bool:
        return any(marker in query.lower() for marker in ("전체", "주제", "경향", "요약", "공통", "전반"))
