"""KServe adapter: routes to an InferenceService predictor.

KServe is a deployment layer, so the wire protocol depends on the runtime
inside the InferenceService: classic predictors speak V1 (:predict) or V2
(/v2/models/{m}/infer); LLM runtimes (e.g. a vLLM ServingRuntime) speak the
OpenAI protocol. The backend's `protocol` field selects the dialect.
"""

from __future__ import annotations

from ..model import Protocol, RouteRequest, WorkloadType
from .base import BaseAdapter

_OPENAI_PATHS = {
    WorkloadType.LLM_CHAT: "/v1/chat/completions",
    WorkloadType.LLM_COMPLETION: "/v1/completions",
    WorkloadType.EMBEDDING: "/v1/embeddings",
}


class KServeAdapter(BaseAdapter):
    def target_url(self, request: RouteRequest) -> str:
        model = self.upstream_model(request.model)
        if self.cfg.protocol == Protocol.KSERVE_V1:
            return f"{self.cfg.base_url}/v1/models/{model}:predict"
        if self.cfg.protocol == Protocol.OPENAI:
            try:
                return f"{self.cfg.base_url}{_OPENAI_PATHS[request.workload]}"
            except KeyError:
                raise ValueError(
                    f"KServe backend {self.cfg.name!r} (openai protocol) cannot serve "
                    f"workload {request.workload.value!r}"
                ) from None
        return f"{self.cfg.base_url}/v2/models/{model}/infer"

    def prepare_payload(self, request: RouteRequest) -> dict:
        if self.cfg.protocol == Protocol.OPENAI:
            payload = dict(request.payload)
            payload["model"] = self.upstream_model(request.model)
            return payload
        return request.payload
