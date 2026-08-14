"""Core domain types shared across the router, adapters, and gateway."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any


class WorkloadType(str, enum.Enum):
    """What kind of inference a request represents.

    The workload type drives backend affinity: vLLM owns token-generation
    workloads, Triton owns tensor/ensemble workloads, KServe-hosted
    predictors sit wherever their runtime puts them.
    """

    LLM_CHAT = "llm_chat"
    LLM_COMPLETION = "llm_completion"
    EMBEDDING = "embedding"
    TENSOR = "tensor"
    ENSEMBLE = "ensemble"


class BackendKind(str, enum.Enum):
    VLLM = "vllm"
    TRITON = "triton"
    KSERVE = "kserve"


class Protocol(str, enum.Enum):
    """Wire protocol spoken by a backend."""

    OPENAI = "openai"      # /v1/chat/completions, /v1/completions, /v1/embeddings
    KSERVE_V1 = "v1"       # /v1/models/{name}:predict
    KSERVE_V2 = "v2"       # /v2/models/{name}/infer (also Triton's native HTTP API)


@dataclass(slots=True)
class RouteRequest:
    """A protocol-agnostic inference request flowing through the router."""

    model: str
    workload: WorkloadType
    payload: dict[str, Any]
    slo_class: str = "default"
    stream: bool = False
    request_id: str = ""
    session: str = ""  # affinity key for KV/prefix-cache reuse ("" = none)
    deadline_ms: float | None = None  # total budget incl. failover; None = unbounded
    est_tokens: int | None = None  # rough input-token estimate for LLM workloads


@dataclass(slots=True)
class AdapterResponse:
    """Result of dispatching a request to one backend."""

    status_code: int
    body: Any
    headers: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return 200 <= self.status_code < 300


@dataclass(slots=True)
class ScoredCandidate:
    """A backend candidate with its dispatch utility score (higher is better)."""

    backend_name: str
    score: float
    predicted_latency_ms: float
    breakdown: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class RoutingDecision:
    """Outcome of the routing brain for a single request."""

    primary: str
    ranked: list[ScoredCandidate]
    shadow: str | None = None
    canary: bool = False
    reason: str = "score"
