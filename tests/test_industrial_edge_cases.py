"""Industrial edge cases: the failure modes that show up in production fleets.

Covered scenarios:
- Overload storms: priority load shedding under full saturation
- Tail latency: hedged requests racing a slow primary
- Deadline propagation: budgets exhausted mid-failover
- Session/prefix-cache affinity: stickiness, competitiveness bounds, drain evictions
- Operator drains: maintenance without dropped traffic
- Circuit breaker thundering herd: one probe, not a stampede
- Hostile telemetry: NaN/Inf metrics, escaped labels, garbage exposition lines
- Malformed client payloads that must never 500 the gateway
- Concurrency accounting under parallel load
"""

from __future__ import annotations

import asyncio

import httpx

from trident.gateway import create_app
from trident.telemetry import BackendStats, apply_scraped_metrics, parse_prometheus


def make_app(base_config, handler_or_behavior, **kwargs):
    if callable(handler_or_behavior):
        transport = httpx.MockTransport(handler_or_behavior)
    else:
        behavior = handler_or_behavior

        def handler(request: httpx.Request) -> httpx.Response:
            action = behavior.get(request.url.host, {"default": True})
            if isinstance(action, Exception):
                raise action
            if isinstance(action, int):
                return httpx.Response(action, json={"error": "boom"})
            return httpx.Response(200, json=action)

        transport = httpx.MockTransport(handler)
    return create_app(base_config, transport=transport, start_poller=False, **kwargs)


def client_for(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")


CHAT = {"model": "llama-3-8b", "messages": [{"role": "user", "content": "hi"}]}


# ---------------------------------------------------------------------------
# Overload storms: priority admission control
# ---------------------------------------------------------------------------


def saturate(app, *names):
    for name in names:
        rt = app.state.registry.get(name)
        rt.stats.inflight = rt.cfg.max_concurrency


async def test_overload_sheds_batch_but_serves_interactive(base_config):
    base_config.routing.admission.enabled = True
    app = make_app(base_config, {"vllm-a": {"id": "x"}, "triton-llm": {"id": "y"}})
    saturate(app, "vllm-a", "triton-llm")
    async with client_for(app) as client:
        batch = await client.post("/v1/chat/completions", json=CHAT,
                                  headers={"x-trident-slo": "batch"})
        interactive = await client.post("/v1/chat/completions", json=CHAT,
                                        headers={"x-trident-slo": "interactive"})
    # batch (priority 1) is shed with backpressure signalling...
    assert batch.status_code == 429
    assert batch.headers["retry-after"] == "1"
    assert "saturated" in batch.json()["error"]
    # ...interactive (priority 10) rides through and still gets served.
    assert interactive.status_code == 200


async def test_overload_recovers_when_pressure_drops(base_config):
    base_config.routing.admission.enabled = True
    app = make_app(base_config, {"vllm-a": {"id": "x"}})
    saturate(app, "vllm-a", "triton-llm")
    async with client_for(app) as client:
        shed = await client.post("/v1/chat/completions", json=CHAT,
                                 headers={"x-trident-slo": "batch"})
        app.state.registry.get("vllm-a").stats.inflight = 0
        served = await client.post("/v1/chat/completions", json=CHAT,
                                   headers={"x-trident-slo": "batch"})
    assert shed.status_code == 429
    assert served.status_code == 200


async def test_partial_saturation_does_not_shed(base_config):
    """One saturated backend is a routing problem, not an admission problem."""
    base_config.routing.admission.enabled = True
    app = make_app(base_config, {"vllm-a": {"id": "x"}, "triton-llm": {"id": "y"}})
    saturate(app, "vllm-a")  # triton-llm still has headroom
    async with client_for(app) as client:
        resp = await client.post("/v1/chat/completions", json=CHAT,
                                 headers={"x-trident-slo": "batch"})
    assert resp.status_code == 200


async def test_admission_disabled_never_sheds(base_config):
    app = make_app(base_config, {"vllm-a": {"id": "x"}, "triton-llm": {"id": "y"}})
    saturate(app, "vllm-a", "triton-llm")
    async with client_for(app) as client:
        resp = await client.post("/v1/chat/completions", json=CHAT,
                                 headers={"x-trident-slo": "batch"})
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Tail latency: hedged requests
# ---------------------------------------------------------------------------


def enable_hedging(config, slo="interactive"):
    config.routing.hedge.enabled = True
    config.routing.hedge.min_delay_ms = 30
    config.slo_classes[slo].hedge = True


async def test_hedge_beats_slow_primary(base_config):
    """Primary stalls (200ms); the hedge fires at ~30ms and wins."""
    enable_hedging(base_config)

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "vllm-a":
            await asyncio.sleep(0.2)
            return httpx.Response(200, json={"id": "slow-primary"})
        return httpx.Response(200, json={"id": "fast-hedge"})

    app = make_app(base_config, handler)
    # Teach the router that vllm-a is normally fast, so it stays primary
    # and the hedge delay is computed from a small predicted latency.
    for _ in range(5):
        app.state.registry.get("vllm-a").stats.latency_ms.update(10)
    async with client_for(app) as client:
        resp = await client.post("/v1/chat/completions", json=CHAT,
                                 headers={"x-trident-slo": "interactive"})
    assert resp.status_code == 200
    assert resp.headers["x-trident-backend"] == "triton-llm"
    assert resp.json()["id"] == "fast-hedge"
    # The winner came in far below the primary's 200ms stall.
    assert float(resp.headers["x-trident-latency-ms"]) < 150


async def test_fast_primary_wins_without_hedge_effect(base_config):
    enable_hedging(base_config)
    app = make_app(base_config, {"vllm-a": {"id": "primary"}, "triton-llm": {"id": "hedge"}})
    async with client_for(app) as client:
        resp = await client.post("/v1/chat/completions", json=CHAT,
                                 headers={"x-trident-slo": "interactive"})
    assert resp.headers["x-trident-backend"] == "vllm-a"
    assert resp.json()["id"] == "primary"


async def test_hedge_covers_primary_connection_failure(base_config):
    """Primary dies outright: the hedge (or its own failure handling) still
    produces a response rather than an exception."""
    enable_hedging(base_config)

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "vllm-a":
            await asyncio.sleep(0.05)
            raise httpx.ConnectError("connection reset")
        return httpx.Response(200, json={"id": "hedge-saves"})

    app = make_app(base_config, handler)
    for _ in range(5):
        app.state.registry.get("vllm-a").stats.latency_ms.update(10)
    async with client_for(app) as client:
        resp = await client.post("/v1/chat/completions", json=CHAT,
                                 headers={"x-trident-slo": "interactive"})
    assert resp.status_code == 200
    assert resp.headers["x-trident-backend"] == "triton-llm"


async def test_hedge_all_backends_down_returns_503(base_config):
    enable_hedging(base_config)
    behavior = {
        "vllm-a": httpx.ConnectError("down"),
        "triton-llm": httpx.ConnectError("down"),
    }
    app = make_app(base_config, behavior)
    async with client_for(app) as client:
        resp = await client.post("/v1/chat/completions", json=CHAT,
                                 headers={"x-trident-slo": "interactive"})
    assert resp.status_code == 503


async def test_streaming_never_hedges(base_config):
    """Hedging a stream would interleave two token streams — must not happen."""
    enable_hedging(base_config)
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.host)
        return httpx.Response(200, content=b"data: [DONE]\n\n",
                              headers={"content-type": "text/event-stream"})

    app = make_app(base_config, handler)
    async with client_for(app) as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={**CHAT, "stream": True},
            headers={"x-trident-slo": "interactive"},
        )
    assert resp.status_code == 200
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# Deadline propagation
# ---------------------------------------------------------------------------


async def test_deadline_already_blown_returns_504(base_config):
    app = make_app(base_config, {"vllm-a": {"id": "x"}})
    async with client_for(app) as client:
        resp = await client.post("/v1/chat/completions", json=CHAT,
                                 headers={"x-trident-deadline-ms": "0.000001"})
    assert resp.status_code == 504
    assert "deadline" in resp.json()["error"]


async def test_deadline_exhausted_mid_failover_returns_504(base_config):
    """Primary burns the whole budget then dies; failover must not start."""

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "vllm-a":
            await asyncio.sleep(0.08)
            raise httpx.ConnectError("died after eating the budget")
        return httpx.Response(200, json={"id": "should-not-be-reached"})

    app = make_app(base_config, handler)
    async with client_for(app) as client:
        resp = await client.post("/v1/chat/completions", json=CHAT,
                                 headers={"x-trident-deadline-ms": "50"})
    assert resp.status_code == 504


async def test_generous_deadline_allows_failover(base_config):
    behavior = {
        "vllm-a": httpx.ConnectError("down"),
        "triton-llm": {"id": "failover-ok"},
    }
    app = make_app(base_config, behavior)
    async with client_for(app) as client:
        resp = await client.post("/v1/chat/completions", json=CHAT,
                                 headers={"x-trident-deadline-ms": "5000"})
    assert resp.status_code == 200
    assert resp.headers["x-trident-backend"] == "triton-llm"


async def test_garbage_deadline_header_is_ignored(base_config):
    app = make_app(base_config, {"vllm-a": {"id": "x"}})
    async with client_for(app) as client:
        for bad in ("abc", "-100", "", "NaN,50"):
            resp = await client.post("/v1/chat/completions", json=CHAT,
                                     headers={"x-trident-deadline-ms": bad})
            assert resp.status_code == 200, bad


# ---------------------------------------------------------------------------
# Session / prefix-cache affinity
# ---------------------------------------------------------------------------


async def test_session_sticks_despite_score_shift(base_config):
    """Once pinned, a session survives a moderate score disadvantage —
    that's the whole point (prefix-cache hits beat a slightly better score)."""
    base_config.routing.session_affinity.enabled = True
    app = make_app(base_config, {"vllm-a": {"id": "a"}, "triton-llm": {"id": "b"}})
    registry = app.state.registry
    async with client_for(app) as client:
        first = await client.post("/v1/chat/completions", json=CHAT,
                                  headers={"x-trident-session": "user-42"})
        assert first.headers["x-trident-backend"] == "vllm-a"
        # vLLM slows down: pure scoring would now prefer triton-llm...
        for _ in range(10):
            registry.get("vllm-a").stats.latency_ms.update(2000)
            registry.get("triton-llm").stats.latency_ms.update(100)
        second = await client.post("/v1/chat/completions", json=CHAT,
                                   headers={"x-trident-session": "user-42"})
        # ...but the session stays pinned (still within min_score_ratio).
        assert second.headers["x-trident-backend"] == "vllm-a"
        # A fresh session with no pin follows the scores instead.
        fresh = await client.post("/v1/chat/completions", json=CHAT,
                                  headers={"x-trident-session": "user-99",
                                           "x-trident-slo": "interactive"})
        assert fresh.headers["x-trident-backend"] == "triton-llm"


async def test_session_evicted_when_backend_drains(base_config):
    base_config.routing.session_affinity.enabled = True
    app = make_app(base_config, {"vllm-a": {"id": "a"}, "triton-llm": {"id": "b"}})
    async with client_for(app) as client:
        first = await client.post("/v1/chat/completions", json=CHAT,
                                  headers={"x-trident-session": "user-42"})
        assert first.headers["x-trident-backend"] == "vllm-a"
        await client.post("/admin/backends/vllm-a/drain")
        second = await client.post("/v1/chat/completions", json=CHAT,
                                   headers={"x-trident-session": "user-42"})
        # Pinned backend is draining -> session re-pins to the survivor.
        assert second.headers["x-trident-backend"] == "triton-llm"
        await client.post("/admin/backends/vllm-a/undrain")
        third = await client.post("/v1/chat/completions", json=CHAT,
                                  headers={"x-trident-session": "user-42"})
        # Re-pinned: stickiness now belongs to triton-llm, not the old home.
        assert third.headers["x-trident-backend"] == "triton-llm"


def test_session_store_ttl_and_lru_bounds():
    from trident.router import SessionStore

    clock = [0.0]
    store = SessionStore(ttl_seconds=10, max_sessions=3, clock=lambda: clock[0])
    store.set("a", "b1")
    clock[0] = 11.0
    assert store.get("a") is None  # expired

    for i in range(5):
        store.set(f"s{i}", "b1")
    assert len(store) == 3  # LRU-bounded: memory can't grow with session count
    assert store.get("s0") is None
    assert store.get("s4") == "b1"


# ---------------------------------------------------------------------------
# Operator drains
# ---------------------------------------------------------------------------


async def test_drain_redirects_traffic_and_undrain_restores(base_config):
    app = make_app(base_config, {"vllm-a": {"id": "a"}, "triton-llm": {"id": "b"}})
    async with client_for(app) as client:
        before = await client.post("/v1/chat/completions", json=CHAT)
        assert before.headers["x-trident-backend"] == "vllm-a"

        drain = await client.post("/admin/backends/vllm-a/drain")
        assert drain.json() == {"backend": "vllm-a", "draining": True}
        state = (await client.get("/admin/backends")).json()
        assert state["vllm-a"]["draining"] is True

        during = await client.post("/v1/chat/completions", json=CHAT)
        assert during.headers["x-trident-backend"] == "triton-llm"

        await client.post("/admin/backends/vllm-a/undrain")
        after = await client.post("/v1/chat/completions", json=CHAT)
        assert after.headers["x-trident-backend"] == "vllm-a"


async def test_drain_unknown_backend_404(base_config):
    app = make_app(base_config, {})
    async with client_for(app) as client:
        resp = await client.post("/admin/backends/ghost/drain")
    assert resp.status_code == 404


async def test_draining_all_backends_returns_503_not_500(base_config):
    app = make_app(base_config, {"vllm-a": {"id": "a"}, "triton-llm": {"id": "b"}})
    async with client_for(app) as client:
        await client.post("/admin/backends/vllm-a/drain")
        await client.post("/admin/backends/triton-llm/drain")
        resp = await client.post("/v1/chat/completions", json=CHAT)
    assert resp.status_code == 503


# ---------------------------------------------------------------------------
# Circuit breaker thundering herd
# ---------------------------------------------------------------------------


async def test_half_open_admits_one_probe_not_a_stampede(base_config):
    """After recovery, concurrent requests must not all pile onto the
    half-open backend: exactly one probes it, the rest fail over."""
    base_config.routing.breaker_failure_threshold = 2
    base_config.routing.breaker_reset_seconds = 0.05
    vllm_calls = []

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "vllm-a":
            vllm_calls.append(1)
            if len(vllm_calls) <= 2:
                raise httpx.ConnectError("still down")
            await asyncio.sleep(0.05)  # recovered but slow: probes overlap
            return httpx.Response(200, json={"id": "recovered"})
        return httpx.Response(200, json={"id": "fallback"})

    app = make_app(base_config, handler)
    async with client_for(app) as client:
        # Trip the breaker.
        for _ in range(2):
            await client.post("/v1/chat/completions", json=CHAT)
        state = (await client.get("/admin/backends")).json()
        assert state["vllm-a"]["breaker"] == "open"

        await asyncio.sleep(0.06)  # cooldown elapses -> half-open

        probes_before = len(vllm_calls)
        responses = await asyncio.gather(
            *(client.post("/v1/chat/completions", json=CHAT) for _ in range(4))
        )
    assert all(r.status_code == 200 for r in responses)
    # Exactly one request probed the half-open backend; the other three
    # failed over to triton-llm instead of stampeding.
    assert len(vllm_calls) - probes_before == 1
    backends = sorted(r.headers["x-trident-backend"] for r in responses)
    assert backends.count("triton-llm") == 3
    assert backends.count("vllm-a") == 1


# ---------------------------------------------------------------------------
# Hostile telemetry payloads
# ---------------------------------------------------------------------------


def test_nan_and_inf_metrics_do_not_poison_scoring():
    stats = BackendStats()
    stats.queue_depth = 3.0
    stats.utilization = 0.5
    apply_scraped_metrics(
        "vllm",
        stats,
        "vllm:num_requests_waiting NaN\n"
        "vllm:gpu_cache_usage_perc +Inf\n",
    )
    # Non-finite samples are dropped; the previous good values survive.
    assert stats.queue_depth == 3.0
    assert stats.utilization == 0.5


def test_out_of_range_utilization_is_clamped():
    stats = BackendStats()
    apply_scraped_metrics("vllm", stats, "vllm:gpu_cache_usage_perc 42.5\n")
    assert stats.utilization == 1.0
    apply_scraped_metrics("triton", stats, "nv_gpu_utilization -3\n")
    assert stats.utilization == 0.0


def test_parser_survives_garbage_exposition():
    text = (
        "# HELP something helpful\n"
        "not a metric line at all!!!\n"
        "metric_no_value\n"
        'ok_metric{a="1"} 5 1712345678901\n'   # trailing timestamp
        'escaped{model="he said \\"hi\\", twice"} 7\n'
        "exponent_metric 1.5e3\n"
        "\n"
    )
    fams = parse_prometheus(text)
    assert fams["ok_metric"][0][1] == 5.0
    assert fams["exponent_metric"][0][1] == 1500.0
    # The escaped-quote label value did not break pair splitting.
    labels = fams["escaped"][0][0]
    assert "model" in labels


def test_scrape_with_no_relevant_families_changes_nothing():
    stats = BackendStats()
    stats.queue_depth = 7.0
    apply_scraped_metrics("triton", stats, "some_other_metric 12\n")
    assert stats.queue_depth == 7.0


# ---------------------------------------------------------------------------
# Malformed client payloads: never 500
# ---------------------------------------------------------------------------


async def test_missing_model_field_is_a_client_error(base_config):
    app = make_app(base_config, {"vllm-a": {"id": "x"}})
    async with client_for(app) as client:
        resp = await client.post("/v1/chat/completions", json={"messages": []})
    assert resp.status_code == 503  # unknown model "", not a crash
    assert "no backend" in resp.json()["error"]


async def test_non_string_model_field_does_not_crash(base_config):
    app = make_app(base_config, {"vllm-a": {"id": "x"}})
    async with client_for(app) as client:
        for weird in (123, None, ["llama-3-8b"], {"name": "llama"}):
            resp = await client.post("/v1/chat/completions",
                                     json={"model": weird, "messages": []})
            assert resp.status_code == 503, weird


async def test_bizarre_message_content_shapes_route_fine(base_config):
    """Multimodal-style content parts, null content, and non-dict messages
    must all survive token estimation and reach the backend."""
    app = make_app(base_config, {"vllm-a": {"id": "x"}})
    payloads = [
        {"model": "llama-3-8b",
         "messages": [{"role": "user",
                       "content": [{"type": "text", "text": "hi"},
                                   {"type": "image_url", "image_url": {"url": "..."}}]}]},
        {"model": "llama-3-8b", "messages": [{"role": "user", "content": None}]},
        {"model": "llama-3-8b", "messages": ["not-a-dict", 42]},
        {"model": "llama-3-8b", "messages": "not-a-list"},
    ]
    async with client_for(app) as client:
        for payload in payloads:
            resp = await client.post("/v1/chat/completions", json=payload)
            assert resp.status_code == 200, payload


def test_token_estimation_edge_shapes():
    from trident.model import WorkloadType
    from trident.tokens import estimate_tokens

    huge = {"model": "m", "messages": [{"role": "user", "content": "x" * 400_000}]}
    assert estimate_tokens(huge, WorkloadType.LLM_CHAT) == 100_000
    assert estimate_tokens({"prompt": ["a", "b"]}, WorkloadType.LLM_COMPLETION) is not None
    assert estimate_tokens({"inputs": []}, WorkloadType.TENSOR) is None
    assert estimate_tokens("not-a-dict", WorkloadType.LLM_CHAT) is None
    assert estimate_tokens({}, WorkloadType.LLM_CHAT) is None


def test_long_context_shifts_prediction(base_config):
    """A 32k-token request must be predicted slower than the 500-token
    average that trained the EWMA — and clamped, not extrapolated to infinity."""
    from trident.scoring import predict_latency_ms

    cfg = next(b for b in base_config.backends if b.name == "vllm-a")
    stats = BackendStats()
    for _ in range(5):
        stats.latency_ms.update(1000)
        stats.tokens_in.update(500)

    short = predict_latency_ms(cfg, stats, 250, est_tokens=500)
    long = predict_latency_ms(cfg, stats, 250, est_tokens=32_000)
    absurd = predict_latency_ms(cfg, stats, 250, est_tokens=10_000_000)
    assert long > short * 5
    assert absurd == long  # ratio clamp caps the extrapolation


# ---------------------------------------------------------------------------
# Concurrency accounting
# ---------------------------------------------------------------------------


async def test_inflight_returns_to_zero_after_parallel_burst(base_config):
    async def handler(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.01)
        return httpx.Response(200, json={"id": "x"})

    app = make_app(base_config, handler)
    async with client_for(app) as client:
        responses = await asyncio.gather(
            *(client.post("/v1/chat/completions", json=CHAT) for _ in range(25))
        )
    assert all(r.status_code == 200 for r in responses)
    for name, rt in app.state.registry.backends.items():
        assert rt.stats.inflight == 0, name


async def test_inflight_zero_even_when_backends_error(base_config):
    behavior = {
        "vllm-a": httpx.ConnectError("down"),
        "triton-llm": httpx.ConnectError("down"),
    }
    app = make_app(base_config, behavior)
    async with client_for(app) as client:
        responses = await asyncio.gather(
            *(client.post("/v1/chat/completions", json=CHAT) for _ in range(10))
        )
    assert all(r.status_code == 503 for r in responses)
    for name, rt in app.state.registry.backends.items():
        assert rt.stats.inflight == 0, name
