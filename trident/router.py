"""The routing brain: capability filter -> canary override -> SLO-aware ranking."""

from __future__ import annotations

import random

from .config import SLOClassConfig, TridentConfig
from .model import RoutingDecision, ScoredCandidate, WorkloadType
from .registry import Registry
from .scoring import rank_candidates


class NoBackendAvailable(Exception):
    """No healthy backend can serve this (model, workload) pair right now."""

    def __init__(self, model: str, workload: WorkloadType, detail: str = "") -> None:
        self.model = model
        self.workload = workload
        msg = f"no backend available for model={model!r} workload={workload.value!r}"
        if detail:
            msg = f"{msg} ({detail})"
        super().__init__(msg)


class Router:
    def __init__(self, config: TridentConfig, registry: Registry,
                 rng: random.Random | None = None) -> None:
        self.config = config
        self.registry = registry
        self.rng = rng or random.Random()
        self._canary_by_model = {r.model: r for r in config.routing.canary}
        self._shadow_by_model = {r.model: r for r in config.routing.shadow}

    def slo_for(self, name: str) -> SLOClassConfig:
        return self.config.slo_classes.get(name) or self.config.slo_classes["default"]

    def select(self, model: str, workload: WorkloadType,
               slo_class: str = "default") -> RoutingDecision:
        """Pick a primary backend (plus failover ranking and optional shadow)."""
        candidates = self.registry.candidates_for(model, workload)
        if not candidates:
            raise NoBackendAvailable(model, workload, "no backend declares this model/workload")

        slo = self.slo_for(slo_class)
        ranked = rank_candidates(
            [(b.cfg, b.stats) for b in candidates], workload, slo, self.config.routing
        )
        if not ranked:
            raise NoBackendAvailable(
                model, workload, "all candidates unhealthy or circuit-broken"
            )

        decision = self._apply_canary(model, ranked)
        decision.shadow = self._pick_shadow(model, decision.primary)
        return decision

    def _apply_canary(self, model: str, ranked: list[ScoredCandidate]) -> RoutingDecision:
        rule = self._canary_by_model.get(model)
        eligible = {c.backend_name for c in ranked}
        if rule and rule.stable in eligible:
            # Canary overrides scoring for this model: deterministic traffic
            # split, but only onto backends that passed health/breaker filters.
            if rule.candidate in eligible and self.rng.random() < rule.weight:
                return RoutingDecision(
                    primary=rule.candidate, ranked=ranked, canary=True, reason="canary"
                )
            return RoutingDecision(
                primary=rule.stable, ranked=ranked, canary=False, reason="canary-stable"
            )
        return RoutingDecision(primary=ranked[0].backend_name, ranked=ranked, reason="score")

    def _pick_shadow(self, model: str, primary: str) -> str | None:
        rule = self._shadow_by_model.get(model)
        if not rule or rule.target == primary:
            return None
        target = self.registry.backends.get(rule.target)
        if target is None or not target.stats.healthy:
            return None
        if self.rng.random() < rule.sample:
            return rule.target
        return None

    def failover_order(self, decision: RoutingDecision) -> list[str]:
        """Primary first, then remaining ranked candidates for retry."""
        rest = [c.backend_name for c in decision.ranked if c.backend_name != decision.primary]
        return [decision.primary, *rest]


def classify_workload(endpoint: str, model_workloads: list[WorkloadType]) -> WorkloadType:
    """Map a gateway endpoint (plus the model's declared workloads) to a workload type.

    OpenAI-style endpoints are unambiguous. The V2 infer endpoint serves both
    raw tensor models and Triton ensembles, so we disambiguate from the model's
    declared workloads.
    """
    if endpoint == "chat":
        return WorkloadType.LLM_CHAT
    if endpoint == "completion":
        return WorkloadType.LLM_COMPLETION
    if endpoint == "embedding":
        return WorkloadType.EMBEDDING
    if WorkloadType.ENSEMBLE in model_workloads and WorkloadType.TENSOR not in model_workloads:
        return WorkloadType.ENSEMBLE
    return WorkloadType.TENSOR
