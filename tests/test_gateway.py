"""End-to-end gateway tests with mocked backend transports."""

from __future__ import annotations

import httpx

from trident.gateway import create_app


def backend_transport(behavior: dict[str, object]) -> httpx.MockTransport:
    """Route mock requests by host. behavior[host] is a JSON body, an int
    status code, or an exception instance to raise (simulating a dead backend)."""

    def handler(request: httpx.Request) -> httpx.Response:
        action = behavior.get(request.url.host, {"default": True})
        if isinstance(action, Exception):
            raise action
        if isinstance(action, int):
            return httpx.Response(action, json={"error": "boom"})
        return httpx.Response(200, json=action)

    return httpx.MockTransport(handler)


def make_client(base_config, behavior) -> httpx.AsyncClient:
    app = create_app(base_config, transport=backend_transport(behavior), start_poller=False)
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://trident"
    )


async def test_chat_routes_to_vllm(base_config):
    behavior = {"vllm-a": {"id": "cmpl-1", "choices": [{"message": {"content": "hi"}}]}}
    async with make_client(base_config, behavior) as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "llama-3-8b", "messages": [{"role": "user", "content": "hello"}]},
        )
    assert resp.status_code == 200
    assert resp.headers["x-trident-backend"] == "vllm-a"
    assert resp.headers["x-trident-workload"] == "llm_chat"
    assert float(resp.headers["x-trident-latency-ms"]) >= 0
    assert resp.json()["id"] == "cmpl-1"


async def test_v2_infer_routes_to_triton(base_config):
    behavior = {"triton-cv": {"model_name": "resnet50", "outputs": []}}
    async with make_client(base_config, behavior) as client:
        resp = await client.post(
            "/v2/models/resnet50/infer",
            json={"inputs": [{"name": "input0", "shape": [1], "datatype": "FP32",
                              "data": [0.5]}]},
        )
    assert resp.status_code == 200
    assert resp.headers["x-trident-backend"] == "triton-cv"
    assert resp.headers["x-trident-workload"] == "tensor"


async def test_ensemble_classified_from_model_declaration(base_config):
    behavior = {"triton-cv": {"model_name": "detection_pipeline", "outputs": []}}
    async with make_client(base_config, behavior) as client:
        resp = await client.post("/v2/models/detection_pipeline/infer", json={"inputs": []})
    assert resp.status_code == 200
    assert resp.headers["x-trident-workload"] == "ensemble"


async def test_failover_to_second_backend(base_config):
    """Primary (vllm-a) connection-errors; the request lands on triton-llm."""
    behavior = {
        "vllm-a": httpx.ConnectError("connection refused"),
        "triton-llm": {"id": "cmpl-2"},
    }
    async with make_client(base_config, behavior) as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "llama-3-8b", "messages": []},
        )
    assert resp.status_code == 200
    assert resp.headers["x-trident-backend"] == "triton-llm"


async def test_unknown_model_returns_503(base_config):
    async with make_client(base_config, {}) as client:
        resp = await client.post("/v1/chat/completions", json={"model": "ghost", "messages": []})
    assert resp.status_code == 503
    assert "no backend" in resp.json()["error"]


async def test_5xx_from_backend_fails_over(base_config):
    behavior = {"vllm-a": 500, "triton-llm": {"id": "ok"}}
    async with make_client(base_config, behavior) as client:
        resp = await client.post(
            "/v1/chat/completions", json={"model": "llama-3-8b", "messages": []}
        )
    assert resp.status_code == 200
    assert resp.headers["x-trident-backend"] == "triton-llm"


async def test_repeated_failures_open_breaker(base_config):
    """After enough failures, vllm-a is circuit-broken and skipped at ranking time."""
    behavior = {
        "vllm-a": httpx.ConnectError("connection refused"),
        "triton-llm": {"id": "ok"},
    }
    app = create_app(base_config, transport=backend_transport(behavior), start_poller=False)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://trident"
    ) as client:
        for _ in range(6):
            await client.post("/v1/chat/completions",
                              json={"model": "llama-3-8b", "messages": []})
        state = (await client.get("/admin/backends")).json()
    assert state["vllm-a"]["breaker"] == "open"
    assert state["vllm-a"]["total_failures"] >= 5


async def test_kserve_v1_predict_endpoint(base_config):
    behavior = {"triton-cv": {"predictions": [[0.9, 0.1]]}}
    async with make_client(base_config, behavior) as client:
        resp = await client.post("/v1/models/resnet50:predict",
                                 json={"instances": [[1, 2, 3]]})
    assert resp.status_code == 200
    assert resp.json() == {"predictions": [[0.9, 0.1]]}


async def test_slo_header_is_honored(base_config):
    """An interactive SLO with a slow primary shifts traffic to the faster backend."""
    behavior = {"vllm-a": {"id": "a"}, "triton-llm": {"id": "b"}}
    app = create_app(base_config, transport=backend_transport(behavior), start_poller=False)
    registry = app.state.registry
    for _ in range(10):
        registry.get("vllm-a").stats.latency_ms.update(2500)
        registry.get("triton-llm").stats.latency_ms.update(200)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://trident"
    ) as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "llama-3-8b", "messages": []},
            headers={"x-trident-slo": "interactive"},
        )
    assert resp.headers["x-trident-backend"] == "triton-llm"


async def test_streaming_proxies_sse(base_config):
    chunks = b'data: {"delta": "he"}\n\ndata: {"delta": "llo"}\n\ndata: [DONE]\n\n'

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=chunks,
                              headers={"content-type": "text/event-stream"})

    app = create_app(base_config, transport=httpx.MockTransport(handler), start_poller=False)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://trident"
    ) as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "llama-3-8b", "messages": [], "stream": True},
        )
    assert resp.status_code == 200
    assert resp.headers["x-trident-backend"] == "vllm-a"
    assert b"[DONE]" in resp.content


async def test_metrics_endpoint(base_config):
    behavior = {"vllm-a": {"id": "x"}}
    async with make_client(base_config, behavior) as client:
        await client.post("/v1/chat/completions", json={"model": "llama-3-8b", "messages": []})
        resp = await client.get("/metrics")
    assert resp.status_code == 200
    assert b"trident_requests_total" in resp.content


async def test_admin_models(base_config):
    async with make_client(base_config, {}) as client:
        resp = await client.get("/admin/models")
    models = resp.json()
    assert set(models["llama-3-8b"]) == {"vllm-a", "triton-llm"}
    assert models["resnet50"] == ["triton-cv"]


async def test_healthz(base_config):
    async with make_client(base_config, {}) as client:
        resp = await client.get("/healthz")
    assert resp.json() == {"status": "ok"}
