"""领域模型：检索计划等纯数据结构（不含 IO / 业务逻辑）。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class QueryPlan:
    raw_query: str
    intent: str
    filters: list[dict[str, Any]]
    constraints: dict[str, Any]
    lexical_query: str
    semantic_query: str
    enable_dense: bool
    enable_rerank: bool
    rank_window_size: int

    def to_debug_dict(self) -> dict[str, Any]:
        return {
            "raw_query": self.raw_query,
            "intent": self.intent,
            "lexical_query": self.lexical_query,
            "semantic_query": self.semantic_query,
            "constraints": self.constraints,
            "enable_dense": self.enable_dense,
            "enable_rerank": self.enable_rerank,
            "rank_window_size": self.rank_window_size,
            "filter_count": len(self.filters),
        }
