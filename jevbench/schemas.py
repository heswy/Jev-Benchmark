"""Canonical sample / prediction schemas for JEV-Benchmark."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

TaskType = Literal["choice", "score", "noul"]


@dataclass
class Sample:
    id: str
    dataset: str
    split: str
    text: str
    lang: str
    task_type: TaskType
    label: Any
    labels: list[str] = field(default_factory=list)
    label_descriptions: dict[str, str] = field(default_factory=dict)
    levels: list[int] = field(default_factory=list)
    level_descriptions: dict[str, str] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Prediction:
    id: str
    dataset: str
    model: str
    task_type: TaskType
    answer: Any
    probabilities: dict[str, float] = field(default_factory=dict)
    confidence: float | None = None
    format_valid: bool = True
    label_in_set: bool = True
    latency_ms: float | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    raw: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
