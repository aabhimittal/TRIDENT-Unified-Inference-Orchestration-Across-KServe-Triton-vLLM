"""Background loop keeping backend health and load signals fresh.

Each cycle, for every backend:
- GET its health endpoint (kind-specific default, overridable) -> stats.healthy
- GET its Prometheus /metrics endpoint (if configured) -> queue depth, utilization

Failures mark the backend unhealthy but never crash the loop; the scorer
simply routes around it until the next successful poll.
"""

from __future__ import annotations

import asyncio
import logging

import httpx

from .registry import Registry
from .telemetry import apply_scraped_metrics

log = logging.getLogger("trident.poller")


class TelemetryPoller:
    def __init__(self, registry: Registry, interval_seconds: float = 5.0) -> None:
        self.registry = registry
        self.interval = interval_seconds
        self._task: asyncio.Task | None = None

    async def poll_once(self) -> None:
        await asyncio.gather(
            *(self._poll_backend(name) for name in self.registry.backends),
            return_exceptions=True,
        )

    async def _poll_backend(self, name: str) -> None:
        rt = self.registry.get(name)
        client = self.registry.client
        try:
            resp = await client.get(rt.cfg.default_health_url(), timeout=5.0)
            healthy = resp.status_code == 200
        except httpx.HTTPError:
            healthy = False
        if healthy != rt.stats.healthy:
            log.warning("backend %s health changed: %s -> %s", name, rt.stats.healthy, healthy)
        rt.stats.healthy = healthy

        if healthy and rt.cfg.metrics_url:
            try:
                resp = await client.get(rt.cfg.metrics_url, timeout=5.0)
                if resp.status_code == 200:
                    apply_scraped_metrics(rt.cfg.kind.value, rt.stats, resp.text)
            except httpx.HTTPError as exc:
                log.debug("metrics scrape failed for %s: %s", name, exc)

    async def _run(self) -> None:
        while True:
            try:
                await self.poll_once()
            except Exception:  # never let the poll loop die
                log.exception("telemetry poll cycle failed")
            await asyncio.sleep(self.interval)

    def start(self) -> None:
        self._task = asyncio.get_running_loop().create_task(self._run())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
