"""Configuration schema and loader.

TRIDENT is configured with a single YAML file describing SLO classes,
routing weights, backends (and the models each serves), and optional
canary/shadow rules. See config/trident.example.yaml for a full example.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from .model import BackendKind, Protocol, WorkloadType


class SLOClassConfig(BaseModel):
    """A named service-level objective bucket requests can opt into."""

    target_p95_ms: float = Field(gt=0, description="Latency budget for this class")
    priority: int = Field(default=1, ge=0, description="Higher preempts on contention")
    hedge: bool = Field(
        default=False,
        description="Send a backup request to the runner-up backend when the primary "
        "is slow (requires routing.hedge.enabled)",
    )


class ModelConfig(BaseModel):
    """A model served by a backend."""

    name: str
    workloads: list[WorkloadType]
    upstream_name: str | None = Field(
        default=None,
        description="Name the backend knows the model by, if different from `name`",
    )

    def resolved_upstream(self) -> str:
        return self.upstream_name or self.name


class BackendConfig(BaseModel):
    """One serving endpoint: a vLLM server, Triton server, or KServe InferenceService."""

    name: str
    kind: BackendKind
    base_url: str
    protocol: Protocol | None = Field(
        default=None,
        description="Wire protocol; defaults to openai for vllm, v2 for triton/kserve",
    )
    metrics_url: str | None = Field(
        default=None, description="Prometheus metrics endpoint to scrape for load signals"
    )
    health_url: str | None = Field(
        default=None, description="Override the kind-specific default health check URL"
    )
    models: list[ModelConfig]
    max_concurrency: int = Field(default=32, gt=0)
    weight_bias: float = Field(
        default=0.0, description="Static additive score bias (e.g. prefer cheap capacity)"
    )
    affinity: dict[WorkloadType, float] | None = Field(
        default=None, description="Override the default kind->workload affinity"
    )

    @field_validator("base_url", "metrics_url", "health_url")
    @classmethod
    def _strip_trailing_slash(cls, v: str | None) -> str | None:
        return v.rstrip("/") if v else v

    @model_validator(mode="after")
    def _default_protocol(self) -> "BackendConfig":
        if self.protocol is None:
            self.protocol = (
                Protocol.OPENAI if self.kind == BackendKind.VLLM else Protocol.KSERVE_V2
            )
        return self

    def default_health_url(self) -> str:
        if self.health_url:
            return self.health_url
        if self.kind == BackendKind.VLLM:
            return f"{self.base_url}/health"
        return f"{self.base_url}/v2/health/ready"


class RoutingWeights(BaseModel):
    """Relative weights of the utility terms in the dispatch score."""

    latency: float = 0.45
    affinity: float = 0.25
    headroom: float = 0.20
    bias: float = 0.10


class CanaryRule(BaseModel):
    """Split a model's traffic between a stable and a candidate backend."""

    model: str
    stable: str
    candidate: str
    weight: float = Field(ge=0.0, le=1.0, description="Fraction of traffic to candidate")


class ShadowRule(BaseModel):
    """Mirror a sample of a model's traffic to another backend, fire-and-forget."""

    model: str
    target: str
    sample: float = Field(default=1.0, ge=0.0, le=1.0)


class AdmissionConfig(BaseModel):
    """Priority load shedding: when every capable backend is saturated,
    reject low-priority work fast (429) instead of queuing it in front of
    high-priority traffic."""

    enabled: bool = False
    saturation_threshold: float = Field(
        default=0.9, gt=0,
        description="A backend counts as saturated when inflight >= "
        "max_concurrency * threshold",
    )
    shed_below_priority: int = Field(
        default=5, ge=0,
        description="Requests whose SLO priority is below this are shed under saturation",
    )
    retry_after_seconds: int = Field(default=1, ge=0)


class HedgeConfig(BaseModel):
    """Tail-latency hedging: if the primary hasn't answered within a delay
    derived from its own latency profile, race a backup request against it
    on the runner-up backend and take whichever finishes first."""

    enabled: bool = False
    delay_factor: float = Field(
        default=1.5, gt=0,
        description="Hedge delay = factor * predicted primary latency",
    )
    min_delay_ms: float = Field(default=50.0, ge=0)
    max_delay_ms: float = Field(default=2000.0, gt=0)


class SessionAffinityConfig(BaseModel):
    """Sticky sessions for KV/prefix-cache reuse: requests carrying the same
    session key keep landing on the same backend while it stays competitive,
    so vLLM prefix caching and warm KV state actually get hits."""

    enabled: bool = False
    header: str = "x-trident-session"
    ttl_seconds: float = Field(default=300.0, gt=0)
    max_sessions: int = Field(default=10000, gt=0)
    min_score_ratio: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Stickiness holds only while the pinned backend scores at least "
        "this fraction of the current best candidate",
    )


class RoutingConfig(BaseModel):
    weights: RoutingWeights = RoutingWeights()
    max_attempts: int = Field(default=2, ge=1, description="Failover attempts across backends")
    default_latency_ms: float = Field(
        default=250.0, gt=0, description="Latency prior before any observations exist"
    )
    breaker_failure_threshold: int = Field(default=5, ge=1)
    breaker_reset_seconds: float = Field(default=30.0, gt=0)
    admission: AdmissionConfig = AdmissionConfig()
    hedge: HedgeConfig = HedgeConfig()
    session_affinity: SessionAffinityConfig = SessionAffinityConfig()
    canary: list[CanaryRule] = []
    shadow: list[ShadowRule] = []


class TelemetryConfig(BaseModel):
    poll_interval_seconds: float = Field(default=5.0, gt=0)
    request_timeout_seconds: float = Field(default=120.0, gt=0)


class TridentConfig(BaseModel):
    slo_classes: dict[str, SLOClassConfig] = {
        "default": SLOClassConfig(target_p95_ms=2000, priority=1)
    }
    routing: RoutingConfig = RoutingConfig()
    telemetry: TelemetryConfig = TelemetryConfig()
    backends: list[BackendConfig]

    @model_validator(mode="after")
    def _validate_references(self) -> "TridentConfig":
        names = {b.name for b in self.backends}
        if len(names) != len(self.backends):
            raise ValueError("backend names must be unique")
        for rule in self.routing.canary:
            for ref in (rule.stable, rule.candidate):
                if ref not in names:
                    raise ValueError(f"canary rule for {rule.model!r} references unknown backend {ref!r}")
        for rule in self.routing.shadow:
            if rule.target not in names:
                raise ValueError(f"shadow rule for {rule.model!r} references unknown backend {rule.target!r}")
        if "default" not in self.slo_classes:
            self.slo_classes["default"] = SLOClassConfig(target_p95_ms=2000, priority=1)
        return self


def load_config(path: str | Path) -> TridentConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return TridentConfig.model_validate(raw)
