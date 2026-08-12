"""Live backend state: latency EWMAs, circuit breakers, and scraped load signals.

The router never asks backends anything on the request path. Instead, a
background poller scrapes each backend's Prometheus endpoint (vLLM exports
queue depth and KV-cache usage; Triton exports pending request counts and
GPU utilization) and health endpoint, and the gateway feeds observed request
latencies back after every dispatch. Scoring reads this state lock-free.
"""

from __future__ import annotations

import enum
import math
import time
from dataclasses import dataclass, field


class EWMA:
    """Exponentially weighted moving average with a lazy prior."""

    __slots__ = ("alpha", "value")

    def __init__(self, alpha: float = 0.2) -> None:
        self.alpha = alpha
        self.value: float | None = None

    def update(self, sample: float) -> float:
        if self.value is None:
            self.value = sample
        else:
            self.value = self.alpha * sample + (1 - self.alpha) * self.value
        return self.value


class BreakerState(str, enum.Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Consecutive-failure breaker with a half-open probe after a cooldown."""

    def __init__(self, failure_threshold: int = 5, reset_seconds: float = 30.0,
                 clock=time.monotonic) -> None:
        self.failure_threshold = failure_threshold
        self.reset_seconds = reset_seconds
        self._clock = clock
        self._failures = 0
        self._opened_at: float | None = None
        self._probing = False

    @property
    def state(self) -> BreakerState:
        if self._opened_at is None:
            return BreakerState.CLOSED
        if self._clock() - self._opened_at >= self.reset_seconds:
            return BreakerState.HALF_OPEN
        return BreakerState.OPEN

    def available(self) -> bool:
        """Non-mutating eligibility check, safe to call during ranking."""
        state = self.state
        return state == BreakerState.CLOSED or (
            state == BreakerState.HALF_OPEN and not self._probing
        )

    def allow(self) -> bool:
        """Whether a request may be sent. In half-open, admit a single probe."""
        state = self.state
        if state == BreakerState.CLOSED:
            return True
        if state == BreakerState.HALF_OPEN and not self._probing:
            self._probing = True
            return True
        return False

    def record_success(self) -> None:
        self._failures = 0
        self._opened_at = None
        self._probing = False

    def record_failure(self) -> None:
        self._failures += 1
        self._probing = False
        if self._failures >= self.failure_threshold:
            self._opened_at = self._clock()


@dataclass
class BackendStats:
    """Mutable runtime state for one backend, read by the scorer."""

    latency_ms: EWMA = field(default_factory=EWMA)
    tokens_in: EWMA = field(default_factory=EWMA)  # avg input tokens per LLM request
    inflight: int = 0
    queue_depth: float = 0.0
    utilization: float = 0.0  # 0..1; KV-cache usage for vLLM, GPU util for Triton
    healthy: bool = True
    draining: bool = False  # operator-initiated: finish inflight, take no new work
    total_requests: int = 0
    total_failures: int = 0
    breaker: CircuitBreaker = field(default_factory=CircuitBreaker)

    def record_result(self, latency_ms: float, ok: bool, tokens: int | None = None) -> None:
        self.total_requests += 1
        if ok:
            self.latency_ms.update(latency_ms)
            if tokens is not None and tokens > 0:
                self.tokens_in.update(tokens)
            self.breaker.record_success()
        else:
            self.total_failures += 1
            self.breaker.record_failure()

    def snapshot(self) -> dict:
        return {
            "ewma_latency_ms": round(self.latency_ms.value, 2) if self.latency_ms.value else None,
            "ewma_tokens_in": round(self.tokens_in.value, 1) if self.tokens_in.value else None,
            "inflight": self.inflight,
            "queue_depth": self.queue_depth,
            "utilization": round(self.utilization, 4),
            "healthy": self.healthy,
            "draining": self.draining,
            "breaker": self.breaker.state.value,
            "total_requests": self.total_requests,
            "total_failures": self.total_failures,
        }


def parse_prometheus(text: str) -> dict[str, list[tuple[dict[str, str], float]]]:
    """Minimal Prometheus text-format parser: family name -> [(labels, value)]."""
    out: dict[str, list[tuple[dict[str, str], float]]] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            # Exposition format: `name[{labels}] value [timestamp]`. Split off
            # the metric part first (label values may contain spaces), then the
            # value is the FIRST remaining token — never the trailing timestamp.
            if "{" in line:
                brace_end = line.index("}", line.index("{"))
                metric_part, rest = line[: brace_end + 1], line[brace_end + 1 :]
            else:
                metric_part, _, rest = line.partition(" ")
            value = float(rest.split()[0])
            # NaN/Inf are legal in the exposition format but would poison the
            # sums and averages we feed into scoring — drop those samples.
            if not math.isfinite(value):
                continue
            if "{" in metric_part:
                name, label_blob = metric_part.split("{", 1)
                label_blob = label_blob.rstrip("}")
                labels: dict[str, str] = {}
                for pair in _split_labels(label_blob):
                    k, _, v = pair.partition("=")
                    labels[k.strip()] = v.strip().strip('"')
            else:
                name, labels = metric_part, {}
            out.setdefault(name.strip(), []).append((labels, value))
        except (ValueError, IndexError):
            continue
    return out


def _split_labels(blob: str) -> list[str]:
    """Split label pairs on commas that are outside quoted values.

    Handles backslash-escaped quotes inside label values (legal in the
    exposition format: model="he said \\"hi\\"").
    """
    parts, buf, in_quotes, escaped = [], [], False, False
    for ch in blob:
        if escaped:
            escaped = False
        elif ch == "\\":
            escaped = True
        elif ch == '"':
            in_quotes = not in_quotes
        if ch == "," and not in_quotes:
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    if buf:
        parts.append("".join(buf))
    return [p for p in (p.strip() for p in parts) if p]


def _sum_family(families: dict, *names: str) -> float | None:
    for name in names:
        if name in families:
            return sum(v for _, v in families[name])
    return None


def _avg_family(families: dict, *names: str) -> float | None:
    for name in names:
        if name in families and families[name]:
            vals = [v for _, v in families[name]]
            return sum(vals) / len(vals)
    return None


def apply_scraped_metrics(kind: str, stats: BackendStats, text: str) -> None:
    """Fold a scraped /metrics payload into a backend's load signals."""
    families = parse_prometheus(text)
    if kind == "vllm":
        waiting = _sum_family(families, "vllm:num_requests_waiting", "vllm_num_requests_waiting")
        if waiting is not None:
            stats.queue_depth = waiting
        cache = _avg_family(families, "vllm:gpu_cache_usage_perc", "vllm_gpu_cache_usage_perc")
        if cache is not None:
            # vLLM reports 0..1 already
            stats.utilization = min(max(cache, 0.0), 1.0)
    else:  # triton, or kserve fronting triton
        pending = _sum_family(families, "nv_inference_pending_request_count")
        if pending is not None:
            stats.queue_depth = pending
        gpu = _avg_family(families, "nv_gpu_utilization")
        if gpu is not None:
            # Triton reports 0..1
            stats.utilization = min(max(gpu, 0.0), 1.0)
