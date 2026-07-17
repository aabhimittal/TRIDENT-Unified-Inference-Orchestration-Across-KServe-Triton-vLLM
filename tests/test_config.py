from __future__ import annotations

from pathlib import Path

import pytest

from trident.config import TridentConfig, load_config
from trident.model import Protocol

EXAMPLE = Path(__file__).parent.parent / "config" / "trident.example.yaml"


def test_example_config_loads():
    cfg = load_config(EXAMPLE)
    assert {b.name for b in cfg.backends} >= {"vllm-llama", "triton-cv", "kserve-embed"}
    assert "interactive" in cfg.slo_classes
    assert cfg.routing.canary[0].candidate == "triton-trtllm"


def test_protocol_defaults():
    cfg = load_config(EXAMPLE)
    by_name = {b.name: b for b in cfg.backends}
    assert by_name["vllm-llama"].protocol == Protocol.OPENAI
    assert by_name["triton-cv"].protocol == Protocol.KSERVE_V2
    assert by_name["kserve-embed"].protocol == Protocol.OPENAI


def test_health_url_defaults():
    cfg = load_config(EXAMPLE)
    by_name = {b.name: b for b in cfg.backends}
    assert by_name["vllm-llama"].default_health_url().endswith("/health")
    assert by_name["triton-cv"].default_health_url().endswith("/v2/health/ready")


def test_duplicate_backend_names_rejected():
    with pytest.raises(ValueError, match="unique"):
        TridentConfig.model_validate(
            {
                "backends": [
                    {"name": "a", "kind": "vllm", "base_url": "http://x",
                     "models": [{"name": "m", "workloads": ["llm_chat"]}]},
                    {"name": "a", "kind": "triton", "base_url": "http://y",
                     "models": [{"name": "m", "workloads": ["tensor"]}]},
                ]
            }
        )


def test_canary_referencing_unknown_backend_rejected():
    with pytest.raises(ValueError, match="unknown backend"):
        TridentConfig.model_validate(
            {
                "routing": {"canary": [{"model": "m", "stable": "a", "candidate": "ghost",
                                        "weight": 0.1}]},
                "backends": [
                    {"name": "a", "kind": "vllm", "base_url": "http://x",
                     "models": [{"name": "m", "workloads": ["llm_chat"]}]},
                ],
            }
        )


def test_default_slo_class_always_present(base_config):
    assert "default" in base_config.slo_classes
