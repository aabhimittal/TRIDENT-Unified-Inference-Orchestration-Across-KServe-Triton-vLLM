"""TRIDENT's own Prometheus metrics (the router as an observable system)."""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

REQUESTS = Counter(
    "trident_requests_total",
    "Requests dispatched, by backend, workload, and outcome",
    ["backend", "workload", "status"],
)

LATENCY = Histogram(
    "trident_request_latency_seconds",
    "End-to-end dispatch latency per backend",
    ["backend", "workload"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
)

DECISIONS = Counter(
    "trident_routing_decisions_total",
    "Routing decisions, by chosen backend and reason (score/canary/failover)",
    ["backend", "reason"],
)

SHADOWS = Counter(
    "trident_shadow_requests_total",
    "Shadow-mirrored requests, by target backend and outcome",
    ["backend", "status"],
)

BACKEND_HEALTH = Gauge(
    "trident_backend_healthy",
    "1 if the backend passed its last health check",
    ["backend"],
)
