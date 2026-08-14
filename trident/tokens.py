"""Cheap input-token estimation for LLM workloads.

The router can't afford a real tokenizer on the hot path (and doesn't know
which tokenizer each backend uses anyway), so it uses the industry-standard
~4 chars/token heuristic. The estimate only ever scales a latency prediction
relative to the backend's *average* observed request size, so systematic
bias cancels out — only relative size matters.
"""

from __future__ import annotations

import json
from typing import Any

from .model import WorkloadType

_CHARS_PER_TOKEN = 4


def _text_of(value: Any) -> str:
    """Flatten a content field that may be a string, a list of parts, or junk."""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, dict)):
        try:
            return json.dumps(value, ensure_ascii=False)
        except (TypeError, ValueError):
            return ""
    if value is None:
        return ""
    return str(value)


def estimate_tokens(payload: Any, workload: WorkloadType) -> int | None:
    """Rough input-token count for LLM workloads; None for tensor traffic
    (tensor request cost is shape-driven, not token-driven) and for payloads
    too malformed to guess at."""
    if not isinstance(payload, dict):
        return None

    if workload == WorkloadType.LLM_CHAT:
        messages = payload.get("messages")
        if not isinstance(messages, list):
            return None
        text = "".join(
            _text_of(m.get("content")) for m in messages if isinstance(m, dict)
        )
    elif workload == WorkloadType.LLM_COMPLETION:
        text = _text_of(payload.get("prompt"))
    elif workload == WorkloadType.EMBEDDING:
        text = _text_of(payload.get("input"))
    else:
        return None

    if not text:
        return None
    return max(1, len(text) // _CHARS_PER_TOKEN)
