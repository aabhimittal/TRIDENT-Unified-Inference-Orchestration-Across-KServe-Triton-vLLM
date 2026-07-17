from __future__ import annotations

import httpx
import pytest

from trident.adapters import KServeAdapter, TritonAdapter, VLLMAdapter, build_adapter
from trident.config import BackendConfig
from trident.model import RouteRequest, WorkloadType


def make_cfg(**kwargs) -> BackendConfig:
    defaults = {
        "name": "b",
        "base_url": "http://backend:8000",
        "models": [
            {"name": "public-name", "upstream_name": "upstream-name",
             "workloads": ["llm_chat", "tensor", "embedding"]},
        ],
    }
    defaults.update(kwargs)
    return BackendConfig.model_validate(defaults)


def capture_client(captured: dict) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["json"] = request.content
        return httpx.Response(200, json={"ok": True})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_vllm_adapter_paths_and_model_rewrite():
    captured: dict = {}
    cfg = make_cfg(kind="vllm")
    adapter = VLLMAdapter(cfg, capture_client(captured))
    req = RouteRequest(
        model="public-name", workload=WorkloadType.LLM_CHAT,
        payload={"model": "public-name", "messages": [{"role": "user", "content": "hi"}]},
    )
    resp = await adapter.infer(req)
    assert resp.ok
    assert captured["url"] == "http://backend:8000/v1/chat/completions"
    assert b'"model": "upstream-name"' in captured["json"] or \
           b'"model":"upstream-name"' in captured["json"]


async def test_vllm_adapter_rejects_tensor_workload():
    cfg = make_cfg(kind="vllm")
    adapter = VLLMAdapter(cfg, capture_client({}))
    req = RouteRequest(model="public-name", workload=WorkloadType.TENSOR, payload={})
    with pytest.raises(ValueError, match="cannot serve"):
        await adapter.infer(req)


async def test_triton_adapter_v2_url():
    captured: dict = {}
    cfg = make_cfg(kind="triton")
    adapter = TritonAdapter(cfg, capture_client(captured))
    req = RouteRequest(
        model="public-name", workload=WorkloadType.TENSOR,
        payload={"inputs": [{"name": "input0", "shape": [1, 3], "datatype": "FP32",
                             "data": [1.0, 2.0, 3.0]}]},
    )
    resp = await adapter.infer(req)
    assert resp.ok
    assert captured["url"] == "http://backend:8000/v2/models/upstream-name/infer"


async def test_kserve_v1_predict_url():
    captured: dict = {}
    cfg = make_cfg(kind="kserve", protocol="v1")
    adapter = KServeAdapter(cfg, capture_client(captured))
    req = RouteRequest(model="public-name", workload=WorkloadType.TENSOR,
                       payload={"instances": [[1, 2, 3]]})
    await adapter.infer(req)
    assert captured["url"] == "http://backend:8000/v1/models/upstream-name:predict"


async def test_kserve_openai_protocol():
    captured: dict = {}
    cfg = make_cfg(kind="kserve", protocol="openai")
    adapter = KServeAdapter(cfg, capture_client(captured))
    req = RouteRequest(model="public-name", workload=WorkloadType.EMBEDDING,
                       payload={"model": "public-name", "input": "hello"})
    await adapter.infer(req)
    assert captured["url"] == "http://backend:8000/v1/embeddings"


def test_build_adapter_dispatch():
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200)))
    assert isinstance(build_adapter(make_cfg(kind="vllm"), client), VLLMAdapter)
    assert isinstance(build_adapter(make_cfg(kind="triton"), client), TritonAdapter)
    assert isinstance(build_adapter(make_cfg(kind="kserve"), client), KServeAdapter)
