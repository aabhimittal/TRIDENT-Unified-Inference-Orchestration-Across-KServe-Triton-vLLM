"""Adapter contract: turn a RouteRequest into a backend-native HTTP call."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator

import httpx

from ..config import BackendConfig
from ..model import AdapterResponse, RouteRequest


class BaseAdapter(ABC):
    def __init__(self, cfg: BackendConfig, client: httpx.AsyncClient) -> None:
        self.cfg = cfg
        self.client = client

    def upstream_model(self, model: str) -> str:
        """Map the public model name to whatever the backend calls it."""
        for m in self.cfg.models:
            if m.name == model:
                return m.resolved_upstream()
        return model

    @abstractmethod
    def target_url(self, request: RouteRequest) -> str:
        """The backend URL this request should be POSTed to."""

    def prepare_payload(self, request: RouteRequest) -> dict:
        """Hook for adapters that rewrite the payload (default: passthrough)."""
        return request.payload

    async def infer(
        self, request: RouteRequest, timeout: float | None = None
    ) -> AdapterResponse:
        kwargs = {} if timeout is None else {"timeout": timeout}
        resp = await self.client.post(
            self.target_url(request), json=self.prepare_payload(request), **kwargs
        )
        try:
            body = resp.json()
        except ValueError:
            body = resp.text
        return AdapterResponse(status_code=resp.status_code, body=body)

    async def stream(self, request: RouteRequest) -> AsyncIterator[bytes]:
        """Proxy a streaming response (SSE) as raw bytes."""
        async with self.client.stream(
            "POST", self.target_url(request), json=self.prepare_payload(request)
        ) as resp:
            resp.raise_for_status()
            async for chunk in resp.aiter_bytes():
                yield chunk
