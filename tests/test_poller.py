from __future__ import annotations

import httpx

from trident.poller import TelemetryPoller
from trident.registry import Registry


async def test_poll_updates_health_and_metrics(base_config):
    base_config.backends[0].metrics_url = "http://vllm-a:8000/metrics"

    def handler(request: httpx.Request) -> httpx.Response:
        host, path = request.url.host, request.url.path
        if host == "vllm-a" and path == "/health":
            return httpx.Response(200)
        if host == "vllm-a" and path == "/metrics":
            return httpx.Response(
                200,
                text='vllm:num_requests_waiting{m="x"} 4\n'
                     'vllm:gpu_cache_usage_perc{m="x"} 0.55\n',
            )
        if host == "triton-llm":
            return httpx.Response(503)
        raise httpx.ConnectError("down")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    registry = Registry(base_config, client)
    poller = TelemetryPoller(registry)
    await poller.poll_once()

    vllm = registry.get("vllm-a").stats
    assert vllm.healthy
    assert vllm.queue_depth == 4
    assert vllm.utilization == 0.55
    assert not registry.get("triton-llm").stats.healthy   # 503 on readiness
    assert not registry.get("triton-cv").stats.healthy    # connection refused
    await client.aclose()
