from __future__ import annotations

import pytest

from trident.config import TridentConfig


@pytest.fixture
def base_config() -> TridentConfig:
    """Two LLM-capable backends + one CV Triton backend, no canary/shadow."""
    return TridentConfig.model_validate(
        {
            "slo_classes": {
                "interactive": {"target_p95_ms": 1500, "priority": 10},
                "batch": {"target_p95_ms": 30000, "priority": 1},
            },
            "backends": [
                {
                    "name": "vllm-a",
                    "kind": "vllm",
                    "base_url": "http://vllm-a:8000",
                    "max_concurrency": 64,
                    "models": [
                        {
                            "name": "llama-3-8b",
                            "upstream_name": "meta-llama/Meta-Llama-3-8B-Instruct",
                            "workloads": ["llm_chat", "llm_completion"],
                        }
                    ],
                },
                {
                    "name": "triton-llm",
                    "kind": "triton",
                    "base_url": "http://triton-llm:8000",
                    "max_concurrency": 32,
                    "models": [
                        {
                            "name": "llama-3-8b",
                            "upstream_name": "llama3_trtllm",
                            "workloads": ["llm_completion", "llm_chat"],
                        }
                    ],
                },
                {
                    "name": "triton-cv",
                    "kind": "triton",
                    "base_url": "http://triton-cv:8000",
                    "max_concurrency": 48,
                    "models": [
                        {"name": "resnet50", "workloads": ["tensor"]},
                        {"name": "detection_pipeline", "workloads": ["ensemble"]},
                    ],
                },
            ],
        }
    )
