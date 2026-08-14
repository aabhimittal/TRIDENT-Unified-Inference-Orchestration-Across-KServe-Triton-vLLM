"""SLO-aware dispatch scoring.

Every eligible backend gets a utility score in roughly [0, 1+bias]:

    score = w_lat * latency_fit + w_aff * affinity + w_head * headroom + w_bias * bias

- latency_fit: how well the *predicted* latency fits the SLO budget.
  Prediction = EWMA of observed latency, inflated by in-flight pressure
  and scraped queue depth. A backend predicted to blow the budget decays
  toward zero fast (quadratic penalty past the target).
- affinity: how naturally the backend's engine fits the workload
  (vLLM for token generation, Triton for tensor/ensemble graphs).
- headroom: 1 - utilization (KV-cache usage on vLLM, GPU util on Triton),
  so saturated capacity sheds new work before it queues.
- bias: operator-set static preference (e.g. spot capacity discount).
"""

from __future__ import annotations

from .config import BackendConfig, RoutingConfig, SLOClassConfig
from .model import BackendKind, ScoredCandidate, WorkloadType
from .telemetry import BackendStats

# Engine sweet spots. Overridable per backend via `affinity:` in config —
# essential for KServe, where the InferenceService may wrap either runtime.
DEFAULT_AFFINITY: dict[BackendKind, dict[WorkloadType, float]] = {
    BackendKind.VLLM: {
        WorkloadType.LLM_CHAT: 1.0,
        WorkloadType.LLM_COMPLETION: 1.0,
        WorkloadType.EMBEDDING: 0.8,
        WorkloadType.TENSOR: 0.1,
        WorkloadType.ENSEMBLE: 0.0,
    },
    BackendKind.TRITON: {
        WorkloadType.TENSOR: 1.0,
        WorkloadType.ENSEMBLE: 1.0,
        WorkloadType.EMBEDDING: 0.7,
        WorkloadType.LLM_COMPLETION: 0.5,  # TensorRT-LLM under Triton is viable
        WorkloadType.LLM_CHAT: 0.4,
    },
    BackendKind.KSERVE: {
        # Neutral: KServe is a deployment layer, not an engine. Operators
        # should override per-InferenceService based on the actual runtime.
        WorkloadType.TENSOR: 0.7,
        WorkloadType.ENSEMBLE: 0.5,
        WorkloadType.EMBEDDING: 0.7,
        WorkloadType.LLM_COMPLETION: 0.6,
        WorkloadType.LLM_CHAT: 0.6,
    },
}


def affinity_for(cfg: BackendConfig, workload: WorkloadType) -> float:
    if cfg.affinity and workload in cfg.affinity:
        return cfg.affinity[workload]
    return DEFAULT_AFFINITY[cfg.kind].get(workload, 0.3)


def predict_latency_ms(
    cfg: BackendConfig,
    stats: BackendStats,
    default_ms: float,
    *,
    est_tokens: int | None = None,
) -> float:
    """Predict request latency from the EWMA, inflated by live pressure.

    For LLM traffic with a token estimate, the EWMA (which reflects the
    backend's *average* request size) is scaled by this request's size
    relative to that average — a 32k-token prompt should not be predicted
    at the latency of the 500-token chats that trained the EWMA. The ratio
    is clamped so one outlier can't produce absurd predictions.
    """
    base = stats.latency_ms.value if stats.latency_ms.value is not None else default_ms
    if est_tokens is not None and stats.tokens_in.value:
        ratio = est_tokens / stats.tokens_in.value
        base *= min(max(ratio, 0.25), 8.0)
    # In-flight pressure: at max_concurrency the estimate doubles.
    concurrency_factor = 1.0 + (stats.inflight / cfg.max_concurrency)
    # Queued work ahead of us costs roughly one base latency per queue slot,
    # amortized across the backend's concurrency.
    queue_penalty = stats.queue_depth * (base / cfg.max_concurrency)
    return base * concurrency_factor + queue_penalty


def latency_fit(predicted_ms: float, slo: SLOClassConfig) -> float:
    """1.0 when instant, ~0.5 at the SLO budget, quadratic decay past it."""
    ratio = predicted_ms / slo.target_p95_ms
    if ratio <= 1.0:
        return 1.0 / (1.0 + ratio)
    # Past the budget: fall off quadratically so a fitting backend always wins.
    return 1.0 / (1.0 + ratio * ratio)


def score_backend(
    cfg: BackendConfig,
    stats: BackendStats,
    workload: WorkloadType,
    slo: SLOClassConfig,
    routing: RoutingConfig,
    est_tokens: int | None = None,
) -> ScoredCandidate:
    w = routing.weights
    predicted = predict_latency_ms(
        cfg, stats, routing.default_latency_ms, est_tokens=est_tokens
    )
    parts = {
        "latency_fit": latency_fit(predicted, slo),
        "affinity": affinity_for(cfg, workload),
        "headroom": max(0.0, 1.0 - stats.utilization),
        "bias": cfg.weight_bias,
    }
    score = (
        w.latency * parts["latency_fit"]
        + w.affinity * parts["affinity"]
        + w.headroom * parts["headroom"]
        + w.bias * parts["bias"]
    )
    return ScoredCandidate(
        backend_name=cfg.name,
        score=score,
        predicted_latency_ms=predicted,
        breakdown=parts,
    )


def rank_candidates(
    candidates: list[tuple[BackendConfig, BackendStats]],
    workload: WorkloadType,
    slo: SLOClassConfig,
    routing: RoutingConfig,
    est_tokens: int | None = None,
) -> list[ScoredCandidate]:
    """Score eligible backends, best first. Callers pre-filter for capability."""
    scored = [
        score_backend(cfg, stats, workload, slo, routing, est_tokens=est_tokens)
        for cfg, stats in candidates
        if stats.healthy and not stats.draining and stats.breaker.available()
    ]
    scored.sort(key=lambda c: c.score, reverse=True)
    return scored
