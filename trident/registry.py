"""Backend registry: config + live stats + adapter for every serving endpoint."""

from __future__ import annotations

from dataclasses import dataclass

import httpx

from .adapters import BaseAdapter, build_adapter
from .config import BackendConfig, TridentConfig
from .model import WorkloadType
from .telemetry import BackendStats, CircuitBreaker


@dataclass
class BackendRuntime:
    cfg: BackendConfig
    stats: BackendStats
    adapter: BaseAdapter

    def serves(self, model: str, workload: WorkloadType) -> bool:
        return any(m.name == model and workload in m.workloads for m in self.cfg.models)


class Registry:
    """Holds every backend's runtime state; shared by router, poller, and gateway."""

    def __init__(self, config: TridentConfig, client: httpx.AsyncClient) -> None:
        self.config = config
        self.client = client
        self.backends: dict[str, BackendRuntime] = {}
        for cfg in config.backends:
            stats = BackendStats(
                breaker=CircuitBreaker(
                    failure_threshold=config.routing.breaker_failure_threshold,
                    reset_seconds=config.routing.breaker_reset_seconds,
                )
            )
            self.backends[cfg.name] = BackendRuntime(
                cfg=cfg, stats=stats, adapter=build_adapter(cfg, client)
            )

    def get(self, name: str) -> BackendRuntime:
        return self.backends[name]

    def candidates_for(self, model: str, workload: WorkloadType) -> list[BackendRuntime]:
        return [b for b in self.backends.values() if b.serves(model, workload)]

    def known_models(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for b in self.backends.values():
            for m in b.cfg.models:
                out.setdefault(m.name, []).append(b.cfg.name)
        return out

    def snapshot(self) -> dict:
        return {
            name: {
                "kind": rt.cfg.kind.value,
                "base_url": rt.cfg.base_url,
                "models": [m.name for m in rt.cfg.models],
                **rt.stats.snapshot(),
            }
            for name, rt in self.backends.items()
        }
