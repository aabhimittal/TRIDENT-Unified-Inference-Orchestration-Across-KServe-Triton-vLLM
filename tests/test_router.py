from __future__ import annotations

import random

import httpx
import pytest

from trident.config import TridentConfig
from trident.model import WorkloadType
from trident.registry import Registry
from trident.router import NoBackendAvailable, Router, classify_workload


def make_router(config: TridentConfig, seed: int = 42) -> Router:
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200)))
    registry = Registry(config, client)
    return Router(config, registry, rng=random.Random(seed))


def test_select_returns_ranked_decision(base_config):
    router = make_router(base_config)
    decision = router.select("llama-3-8b", WorkloadType.LLM_CHAT, "interactive")
    assert decision.primary == "vllm-a"
    assert len(decision.ranked) == 2
    assert decision.reason == "score"


def test_unknown_model_raises(base_config):
    router = make_router(base_config)
    with pytest.raises(NoBackendAvailable, match="no backend declares"):
        router.select("ghost-model", WorkloadType.LLM_CHAT)


def test_all_unhealthy_raises(base_config):
    router = make_router(base_config)
    for rt in router.registry.backends.values():
        rt.stats.healthy = False
    with pytest.raises(NoBackendAvailable, match="unhealthy, draining, or circuit-broken"):
        router.select("llama-3-8b", WorkloadType.LLM_CHAT)


def test_canary_split_ratio(base_config):
    from trident.config import CanaryRule

    base_config.routing.canary = [
        CanaryRule(model="llama-3-8b", stable="vllm-a", candidate="triton-llm", weight=0.3)
    ]
    router = make_router(base_config, seed=7)
    picks = [router.select("llama-3-8b", WorkloadType.LLM_CHAT).primary for _ in range(2000)]
    canary_share = picks.count("triton-llm") / len(picks)
    assert 0.25 < canary_share < 0.35
    assert set(picks) == {"vllm-a", "triton-llm"}


def test_canary_skipped_when_candidate_unhealthy(base_config):
    from trident.config import CanaryRule

    base_config.routing.canary = [
        CanaryRule(model="llama-3-8b", stable="vllm-a", candidate="triton-llm", weight=1.0)
    ]
    router = make_router(base_config)
    router.registry.get("triton-llm").stats.healthy = False
    decision = router.select("llama-3-8b", WorkloadType.LLM_CHAT)
    assert decision.primary == "vllm-a"


def test_shadow_sampling(base_config):
    from trident.config import ShadowRule

    base_config.routing.shadow = [
        ShadowRule(model="llama-3-8b", target="triton-llm", sample=1.0)
    ]
    router = make_router(base_config)
    decision = router.select("llama-3-8b", WorkloadType.LLM_CHAT)
    assert decision.primary == "vllm-a"
    assert decision.shadow == "triton-llm"


def test_shadow_never_targets_primary(base_config):
    from trident.config import ShadowRule

    base_config.routing.shadow = [
        ShadowRule(model="llama-3-8b", target="vllm-a", sample=1.0)
    ]
    router = make_router(base_config)
    decision = router.select("llama-3-8b", WorkloadType.LLM_CHAT)
    assert decision.shadow is None


def test_failover_order_puts_primary_first(base_config):
    router = make_router(base_config)
    decision = router.select("llama-3-8b", WorkloadType.LLM_CHAT)
    order = router.failover_order(decision)
    assert order[0] == decision.primary
    assert set(order) == {"vllm-a", "triton-llm"}


def test_classify_workload():
    assert classify_workload("chat", []) == WorkloadType.LLM_CHAT
    assert classify_workload("completion", []) == WorkloadType.LLM_COMPLETION
    assert classify_workload("embedding", []) == WorkloadType.EMBEDDING
    assert classify_workload("infer", [WorkloadType.TENSOR]) == WorkloadType.TENSOR
    assert classify_workload("infer", [WorkloadType.ENSEMBLE]) == WorkloadType.ENSEMBLE
    assert classify_workload("infer", []) == WorkloadType.TENSOR
