from __future__ import annotations

from trident.telemetry import (
    EWMA,
    BackendStats,
    BreakerState,
    CircuitBreaker,
    apply_scraped_metrics,
    parse_prometheus,
)


def test_ewma_converges():
    e = EWMA(alpha=0.5)
    assert e.value is None
    e.update(100)
    assert e.value == 100
    e.update(200)
    assert e.value == 150


def test_breaker_opens_after_threshold():
    clock = [0.0]
    b = CircuitBreaker(failure_threshold=3, reset_seconds=30, clock=lambda: clock[0])
    for _ in range(2):
        b.record_failure()
    assert b.state == BreakerState.CLOSED
    b.record_failure()
    assert b.state == BreakerState.OPEN
    assert not b.allow()


def test_breaker_half_open_probe_then_recovery():
    clock = [0.0]
    b = CircuitBreaker(failure_threshold=1, reset_seconds=30, clock=lambda: clock[0])
    b.record_failure()
    assert b.state == BreakerState.OPEN
    clock[0] = 31.0
    assert b.state == BreakerState.HALF_OPEN
    assert b.allow()          # single probe admitted
    assert not b.allow()      # second concurrent probe denied
    b.record_success()
    assert b.state == BreakerState.CLOSED
    assert b.allow()


def test_breaker_half_open_failure_reopens():
    clock = [0.0]
    b = CircuitBreaker(failure_threshold=1, reset_seconds=30, clock=lambda: clock[0])
    b.record_failure()
    clock[0] = 31.0
    assert b.allow()
    b.record_failure()
    assert b.state == BreakerState.OPEN
    assert not b.allow()


def test_available_does_not_mutate():
    clock = [0.0]
    b = CircuitBreaker(failure_threshold=1, reset_seconds=30, clock=lambda: clock[0])
    b.record_failure()
    clock[0] = 31.0
    assert b.available() and b.available()  # repeated checks stay true
    assert b.allow()                         # probe still available for dispatch
    assert not b.available()


def test_parse_prometheus_labels_and_values():
    text = """
# HELP vllm:num_requests_waiting Number of requests waiting
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{model_name="llama"} 7.0
vllm:gpu_cache_usage_perc{model_name="llama"} 0.42
nv_inference_pending_request_count{model="resnet50",version="1"} 3
plain_metric 5
"""
    fams = parse_prometheus(text)
    assert fams["vllm:num_requests_waiting"][0][1] == 7.0
    assert fams["nv_inference_pending_request_count"][0][0]["model"] == "resnet50"
    assert fams["plain_metric"][0] == ({}, 5.0)


def test_apply_vllm_metrics():
    stats = BackendStats()
    apply_scraped_metrics(
        "vllm",
        stats,
        'vllm:num_requests_waiting{m="x"} 12\nvllm:gpu_cache_usage_perc{m="x"} 0.8\n',
    )
    assert stats.queue_depth == 12
    assert stats.utilization == 0.8


def test_apply_triton_metrics():
    stats = BackendStats()
    apply_scraped_metrics(
        "triton",
        stats,
        'nv_inference_pending_request_count{model="a"} 2\n'
        'nv_inference_pending_request_count{model="b"} 3\n'
        'nv_gpu_utilization{gpu="0"} 0.65\n',
    )
    assert stats.queue_depth == 5
    assert stats.utilization == 0.65
