"""vLLM adapter: OpenAI-compatible endpoints (/v1/chat/completions et al)."""

from __future__ import annotations

from ..model import RouteRequest, WorkloadType
from .base import BaseAdapter

_PATHS = {
    WorkloadType.LLM_CHAT: "/v1/chat/completions",
    WorkloadType.LLM_COMPLETION: "/v1/completions",
    WorkloadType.EMBEDDING: "/v1/embeddings",
}


class VLLMAdapter(BaseAdapter):
    def target_url(self, request: RouteRequest) -> str:
        try:
            path = _PATHS[request.workload]
        except KeyError:
            raise ValueError(
                f"vLLM backend {self.cfg.name!r} cannot serve workload {request.workload.value!r}"
            ) from None
        return f"{self.cfg.base_url}{path}"

    def prepare_payload(self, request: RouteRequest) -> dict:
        # Rewrite the model name to whatever this vLLM server was launched with.
        payload = dict(request.payload)
        payload["model"] = self.upstream_model(request.model)
        return payload
