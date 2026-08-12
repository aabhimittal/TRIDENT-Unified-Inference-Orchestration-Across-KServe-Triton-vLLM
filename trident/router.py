"""The routing brain: admission -> capability filter -> canary -> affinity -> ranking."""

from __future__ import annotations

import random
import time
from collections import OrderedDict

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


class Overloaded(Exception):
    """All capable backends are saturated; low-priority work is shed."""

    def __init__(self, model: str, retry_after_seconds: int) -> None:
        self.retry_after_seconds = retry_after_seconds
        super().__init__(
            f"all backends for model={model!r} are saturated; "
            f"low-priority request shed, retry after {retry_after_seconds}s"
        )


class DeadlineExceeded(Exception):
    """The request's total latency budget ran out before a backend answered."""

    def __init__(self, deadline_ms: float) -> None:
        self.deadline_ms = deadline_ms
        super().__init__(f"deadline of {deadline_ms:.0f}ms exhausted before completion")


class SessionStore:
    """TTL + LRU map of session key -> backend name, for cache-affinity routing."""

    def __init__(self, ttl_seconds: float, max_sessions: int, clock=time.monotonic) -> None:
        self.ttl = ttl_seconds
        self.max_sessions = max_sessions
        self._clock = clock
        self._entries: OrderedDict[str, tuple[str, float]] = OrderedDict()

    def get(self, session: str) -> str | None:
        entry = self._entries.get(session)
        if entry is None:
            return None
        backend, expires_at = entry
        if self._clock() >= expires_at:
            del self._entries[session]
            return None
        self._entries.move_to_end(session)
        return backend

    def set(self, session: str, backend: str) -> None:
        self._entries[session] = (backend, self._clock() + self.ttl)
        self._entries.move_to_end(session)
        while len(self._entries) > self.max_sessions:
            self._entries.popitem(last=False)

    def __len__(self) -> int:
        return len(self._entries)


class Router:
    def __init__(self, config: TridentConfig, registry: Registry,
                 rng: random.Random | None = None) -> None:
        self.config = config
        self.registry = registry
        self.rng = rng or random.Random()
        self._canary_by_model = {r.model: r for r in config.routing.canary}
        self._shadow_by_model = {r.model: r for r in config.routing.shadow}
        aff = config.routing.session_affinity
        self.sessions = SessionStore(aff.ttl_seconds, aff.max_sessions)

    def slo_for(self, name: str) -> SLOClassConfig:
        return self.config.slo_classes.get(name) or self.config.slo_classes["default"]

    def select(self, model: str, workload: WorkloadType, slo_class: str = "default",
               session: str = "", est_tokens: int | None = None) -> RoutingDecision:
        """Pick a primary backend (plus failover ranking and optional shadow)."""
        candidates = self.registry.candidates_for(model, workload)
        if not candidates:
            raise NoBackendAvailable(model, workload, "no backend declares this model/workload")

        slo = self.slo_for(slo_class)
        self._check_admission(model, candidates, slo)

        ranked = rank_candidates(
            [(b.cfg, b.stats) for b in candidates], workload, slo,
            self.config.routing, est_tokens=est_tokens,
        )
        if not ranked:
            raise NoBackendAvailable(
                model, workload, "all candidates unhealthy, draining, or circuit-broken"
            )

        decision = self._apply_canary(model, ranked)
        if not decision.canary:
            self._apply_session_affinity(session, decision)
        decision.shadow = self._pick_shadow(model, decision.primary)
        return decision

    def _check_admission(self, model: str, candidates: list, slo: SLOClassConfig) -> None:
        """Shed low-priority work when every usable backend is saturated.

        High-priority classes are never shed here — they proceed and let
        scoring pick the least-bad option. Shedding only kicks in when there
        is at least one usable backend (otherwise NoBackendAvailable is the
        right error, not 429).
        """
        adm = self.config.routing.admission
        if not adm.enabled or slo.priority >= adm.shed_below_priority:
            return
        usable = [b for b in candidates if b.stats.healthy and not b.stats.draining]
        if usable and all(
            b.stats.inflight >= b.cfg.max_concurrency * adm.saturation_threshold
            for b in usable
        ):
            raise Overloaded(model, adm.retry_after_seconds)

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

    def _apply_session_affinity(self, session: str, decision: RoutingDecision) -> None:
        """Keep a session pinned to its backend while that backend stays competitive.

        Pinned backends must still be in the ranked list (healthy, not draining,
        breaker closed) AND score at least `min_score_ratio` of the current
        best — so affinity yields to real degradation instead of riding a
        dying backend for the cache hit.
        """
        aff = self.config.routing.session_affinity
        if not aff.enabled or not session:
            return
        by_name = {c.backend_name: c for c in decision.ranked}
        pinned = self.sessions.get(session)
        if pinned is not None and pinned in by_name:
            best = decision.ranked[0].score
            if best <= 0 or by_name[pinned].score >= aff.min_score_ratio * best:
                decision.primary = pinned
                decision.reason = "session-affinity"
        self.sessions.set(session, decision.primary)

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
