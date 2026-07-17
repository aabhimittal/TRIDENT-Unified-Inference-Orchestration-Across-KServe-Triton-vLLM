"""Mock vLLM and Triton servers for a GPU-free local demo.

Runs two fake backends:
- port 8001: "vLLM"   — OpenAI protocol, /health, /metrics with vllm gauges
- port 8002: "Triton" — KServe V2 protocol, /v2/health/ready, /metrics with nv gauges

Usage:  python examples/mock_backends.py
Then:   trident --config examples/demo.yaml --port 8080
"""

from __future__ import annotations

import asyncio
import random

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse


def make_vllm(name: str, mean_latency_ms: float) -> FastAPI:
    app = FastAPI(title=f"mock-vllm-{name}")

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/metrics")
    async def metrics():
        return PlainTextResponse(
            f'vllm:num_requests_waiting{{engine="{name}"}} {random.randint(0, 8)}\n'
            f'vllm:gpu_cache_usage_perc{{engine="{name}"}} {random.uniform(0.1, 0.7):.2f}\n'
        )

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        body = await request.json()
        await asyncio.sleep(random.gauss(mean_latency_ms, mean_latency_ms / 5) / 1000)
        return {
            "id": f"chatcmpl-{random.randint(1000, 9999)}",
            "object": "chat.completion",
            "model": body.get("model"),
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant",
                                     "content": f"(mock reply from {name})"}}],
        }

    @app.post("/v1/completions")
    async def completions(request: Request):
        body = await request.json()
        await asyncio.sleep(random.gauss(mean_latency_ms, mean_latency_ms / 5) / 1000)
        return {
            "id": f"cmpl-{random.randint(1000, 9999)}",
            "object": "text_completion",
            "model": body.get("model"),
            "choices": [{"index": 0, "text": f"(mock completion from {name})",
                         "finish_reason": "stop"}],
        }

    return app


def make_triton(name: str, mean_latency_ms: float) -> FastAPI:
    app = FastAPI(title=f"mock-triton-{name}")

    @app.get("/v2/health/ready")
    async def ready():
        return {"ready": True}

    @app.get("/metrics")
    async def metrics():
        return PlainTextResponse(
            f'nv_inference_pending_request_count{{model="resnet50"}} {random.randint(0, 4)}\n'
            f'nv_gpu_utilization{{gpu="0"}} {random.uniform(0.2, 0.8):.2f}\n'
        )

    @app.post("/v2/models/{model}/infer")
    async def infer(model: str, request: Request):
        await request.json()
        await asyncio.sleep(random.gauss(mean_latency_ms, mean_latency_ms / 5) / 1000)
        return {
            "model_name": model,
            "outputs": [{"name": "output0", "shape": [1, 2], "datatype": "FP32",
                         "data": [0.93, 0.07]}],
        }

    return app


async def main() -> None:
    servers = [
        uvicorn.Server(uvicorn.Config(make_vllm("fast-vllm", 120), port=8001,
                                      log_level="warning")),
        uvicorn.Server(uvicorn.Config(make_triton("cv-triton", 40), port=8002,
                                      log_level="warning")),
    ]
    print("mock vLLM   -> http://localhost:8001  (llama-3-8b)")
    print("mock Triton -> http://localhost:8002  (resnet50)")
    await asyncio.gather(*(s.serve() for s in servers))


if __name__ == "__main__":
    asyncio.run(main())
