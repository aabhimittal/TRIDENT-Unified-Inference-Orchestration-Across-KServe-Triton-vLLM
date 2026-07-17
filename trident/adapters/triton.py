"""Triton adapter: KServe V2 inference protocol (Triton's native HTTP API)."""

from __future__ import annotations

from ..model import RouteRequest
from .base import BaseAdapter


class TritonAdapter(BaseAdapter):
    def target_url(self, request: RouteRequest) -> str:
        model = self.upstream_model(request.model)
        return f"{self.cfg.base_url}/v2/models/{model}/infer"
