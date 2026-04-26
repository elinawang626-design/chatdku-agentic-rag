from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class Chunk:
    chunk_id: str
    doc_name: str
    page: int | None
    text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SearchHit:
    tool: str
    score: float
    chunk: Chunk


@dataclass
class AgentAnswer:
    language: str
    answer: str
    sources: list[dict[str, Any]]
    tool_trace: list[str]


@dataclass
class EvalExample:
    question: str
    language: str
    expected_document: str
    expected_answer_keywords: list[str]


@dataclass
class EvalRow:
    label: str
    retrieval_hit_rate: float
    answer_keyword_hit_rate: float | None
    avg_latency_ms: float
