"""The TRIDENT gateway: one front door, three engines behind it.

Exposed surface:
- OpenAI-compatible:  POST /v1/chat/completions, /v1/completions, /v1/embeddings
- KServe V2:          POST /v2/models/{model}/infer  (tensor + ensemble workloads)
- KServe V1:          POST /v1/models/{model}:predict
- Ops:                GET /healthz, /metrics, /admin/backends, /admin/models
                      POST /admin/backends/{name}/drain, .../undrain

Request headers understood by the router:
- x-trident-slo:         SLO class (latency budget + priority + hedging opt-in)
- x-trident-session:     affinity key — sticky routing for KV/prefix-cache reuse
- x-trident-deadline-ms: total latency budget including failover; 504 when exhausted

Every request is classified into a workload, scored against the live state of
all capable backends, dispatched with failover (and optionally a hedged backup
request racing the slow primary), and (optionally) shadowed.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from . import metrics
from .config import TridentConfig
from .model import AdapterResponse, RouteRequest, RoutingDecision, WorkloadType
from .poller import TelemetryPoller
from .registry import Registry
from .router import (
    DeadlineExceeded,
    NoBackendAvailable,
    Overloaded,
    Router,
    classify_workload,
)
from .tokens import estimate_tokens

log = logging.getLogger("trident.gateway")


class BackendAttemptFailed(Exception):
    """One backend attempt failed in a retryable way (transport error / breaker)."""

    def __init__(self, backend: str, detail: str) -> None:
        self.backend = backend
        self.detail = detail
        super().__init__(f"{backend}: {detail}")


class Dispatcher:
    """Executes routing decisions: dispatch, failover, hedging, shadow mirroring."""

    def __init__(self, config: TridentConfig, registry: Registry, router: Router) -> None:
        self.config = config
        self.registry = registry
        self.router = router

    async def dispatch(self, request: RouteRequest) -> tuple[AdapterResponse, str, float]:
        """Route and execute; returns (response, backend_name, latency_ms)."""
        start = time.perf_counter()
        decision = self.router.select(
            request.model, request.workload, request.slo_class,
            session=request.session, est_tokens=request.est_tokens,
        )
        if decision.shadow:
            asyncio.get_running_loop().create_task(self._shadow(decision.shadow, request))

        order = self.router.failover_order(decision)
        slo = self.router.slo_for(request.slo_class)
        if (
            self.config.routing.hedge.enabled
            and slo.hedge
            and not request.stream
            and len(order) >= 2
        ):
            return await self._dispatch_hedged(request, decision, order, start)
        return await self._dispatch_sequential(request, decision, order, start)

    # ---- sequential failover path ---------------------------------------------------

    async def _dispatch_sequential(
        self, request: RouteRequest, decision: RoutingDecision,
        order: list[str], start: float,
    ) -> tuple[AdapterResponse, str, float]:
        attempts = min(self.config.routing.max_attempts, len(order))
        last_exc: Exception | None = None
        for i, name in enumerate(order[:attempts]):
            timeout = self._remaining_budget(request, start)  # raises DeadlineExceeded
            reason = decision.reason if i == 0 else "failover"
            try:
                resp, latency_ms, backend_ok = await self._attempt(
                    name, request, reason, timeout
                )
            except BackendAttemptFailed as exc:
                last_exc = exc
                continue
            # 5xx counts against the backend; retry it if attempts remain,
            # otherwise pass it through (4xx always passes through).
            if not backend_ok and i + 1 < attempts:
                continue
            return resp, name, latency_ms

        raise NoBackendAvailable(
            request.model, request.workload,
            f"all {attempts} attempt(s) failed" + (f"; last error: {last_exc}" if last_exc else ""),
        )

    # ---- hedged path -----------------------------------------------------------------

    async def _dispatch_hedged(
        self, request: RouteRequest, decision: RoutingDecision,
        order: list[str], start: float,
    ) -> tuple[AdapterResponse, str, float]:
        """Race a backup request against a slow primary (tail-latency hedging).

        The primary is dispatched immediately. If it hasn't finished within a
        delay derived from its own latency profile, the runner-up backend gets
        the same request; first successful response wins and the loser is
        cancelled. Worst case cost: one duplicate request on the slowest tail.
        """
        hedge_cfg = self.config.routing.hedge
        primary, backup = order[0], order[1]
        predicted = next(
            (c.predicted_latency_ms for c in decision.ranked if c.backend_name == primary),
            self.config.routing.default_latency_ms,
        )
        delay_s = min(
            max(hedge_cfg.delay_factor * predicted, hedge_cfg.min_delay_ms),
            hedge_cfg.max_delay_ms,
        ) / 1000

        async def run(name: str, reason: str):
            timeout = self._remaining_budget(request, start)
            resp, latency_ms, ok = await self._attempt(name, request, reason, timeout)
            return resp, name, latency_ms, ok

        tasks: dict[asyncio.Task, str] = {}
        primary_task = asyncio.create_task(run(primary, decision.reason))
        tasks[primary_task] = primary

        done, _ = await asyncio.wait({primary_task}, timeout=delay_s)
        if not done:
            metrics.HEDGES.labels(backend=backup, outcome="fired").inc()
            hedge_task = asyncio.create_task(run(backup, "hedge"))
            tasks[hedge_task] = backup

        fallback: tuple[AdapterResponse, str, float] | None = None
        last_exc: Exception | None = None
        pending = set(tasks)
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                try:
                    resp, name, latency_ms, ok = task.result()
                except (BackendAttemptFailed, DeadlineExceeded) as exc:
                    last_exc = exc
                    continue
                if ok:
                    for p in pending:
                        p.cancel()
                    if pending:
                        await asyncio.gather(*pending, return_exceptions=True)
                    if len(tasks) > 1:
                        outcome = "hedge_won" if tasks[task] == backup else "primary_won"
                        metrics.HEDGES.labels(backend=backup, outcome=outcome).inc()
                    return resp, name, latency_ms
                fallback = (resp, name, latency_ms)

        if fallback is not None:  # both answered but non-ok: pass the 5xx through
            return fallback
        raise NoBackendAvailable(
            request.model, request.workload,
            "hedged attempts failed" + (f"; last error: {last_exc}" if last_exc else ""),
        )

    # ---- shared helpers --------------------------------------------------------------

    def _remaining_budget(self, request: RouteRequest, start: float) -> float | None:
        """Seconds left in the request's deadline; raises when exhausted."""
        if request.deadline_ms is None:
            return None
        remaining_ms = request.deadline_ms - (time.perf_counter() - start) * 1000
        if remaining_ms <= 0:
            raise DeadlineExceeded(request.deadline_ms)
        return remaining_ms / 1000

    async def _attempt(
        self, name: str, request: RouteRequest, reason: str, timeout: float | None,
    ) -> tuple[AdapterResponse, float, bool]:
        """One dispatch to one backend, with full stats/metrics bookkeeping."""
        rt = self.registry.get(name)
        if not rt.stats.breaker.allow():
            raise BackendAttemptFailed(name, "circuit breaker denied the request")
        metrics.DECISIONS.labels(backend=name, reason=reason).inc()
        rt.stats.inflight += 1
        start = time.perf_counter()
        try:
            resp = await rt.adapter.infer(request, timeout=timeout)
        except (httpx.HTTPError, ValueError) as exc:
            rt.stats.record_result(0.0, ok=False)
            metrics.REQUESTS.labels(
                backend=name, workload=request.workload.value, status="error"
            ).inc()
            log.warning("dispatch to %s failed: %s", name, exc)
            raise BackendAttemptFailed(name, str(exc)) from exc
        finally:
            rt.stats.inflight -= 1
        latency_ms = (time.perf_counter() - start) * 1000
        backend_ok = resp.status_code < 500
        rt.stats.record_result(
            latency_ms, ok=backend_ok,
            tokens=request.est_tokens if backend_ok else None,
        )
        metrics.REQUESTS.labels(
            backend=name,
            workload=request.workload.value,
            status="ok" if resp.ok else str(resp.status_code),
        ).inc()
        metrics.LATENCY.labels(backend=name, workload=request.workload.value).observe(
            latency_ms / 1000
        )
        return resp, latency_ms, backend_ok

    async def _shadow(self, backend_name: str, request: RouteRequest) -> None:
        rt = self.registry.get(backend_name)
        try:
            resp = await rt.adapter.infer(request)
            metrics.SHADOWS.labels(
                backend=backend_name, status="ok" if resp.ok else str(resp.status_code)
            ).inc()
        except Exception as exc:  # noqa: BLE001 — shadow is fire-and-forget; nothing may propagate
            metrics.SHADOWS.labels(backend=backend_name, status="error").inc()
            log.debug("shadow to %s failed: %s", backend_name, exc)


def create_app(
    config: TridentConfig,
    transport: httpx.AsyncBaseTransport | None = None,
    start_poller: bool = True,
) -> FastAPI:
    client = httpx.AsyncClient(
        transport=transport,
        timeout=httpx.Timeout(config.telemetry.request_timeout_seconds, connect=10.0),
    )
    registry = Registry(config, client)
    router = Router(config, registry)
    dispatcher = Dispatcher(config, registry, router)
    poller = TelemetryPoller(registry, config.telemetry.poll_interval_seconds)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if start_poller:
            poller.start()
        yield
        await poller.stop()
        await client.aclose()

    app = FastAPI(title="TRIDENT", version="0.2.0", lifespan=lifespan)
    app.state.registry = registry
    app.state.router = router
    app.state.dispatcher = dispatcher

    def _parse_deadline(request: Request) -> float | None:
        raw = request.headers.get("x-trident-deadline-ms")
        if not raw:
            return None
        try:
            deadline = float(raw)
        except ValueError:
            return None
        return deadline if deadline > 0 else None

    async def handle(request: Request, endpoint: str, model: str,
                     payload: dict) -> Response:
        slo_class = request.headers.get("x-trident-slo", "default")
        workloads = _declared_workloads(registry, model)
        workload = classify_workload(endpoint, workloads)
        route_req = RouteRequest(
            model=model,
            workload=workload,
            payload=payload,
            slo_class=slo_class,
            stream=bool(payload.get("stream")) if isinstance(payload, dict) else False,
            request_id=request.headers.get("x-request-id", str(uuid.uuid4())),
            session=request.headers.get(config.routing.session_affinity.header, ""),
            deadline_ms=_parse_deadline(request),
            est_tokens=estimate_tokens(payload, workload),
        )
        try:
            if route_req.stream:
                return await _dispatch_stream(route_req)
            resp, backend, latency_ms = await dispatcher.dispatch(route_req)
        except Overloaded as exc:
            metrics.SHED.labels(slo_class=slo_class).inc()
            return JSONResponse(
                status_code=429,
                content={"error": str(exc)},
                headers={"Retry-After": str(exc.retry_after_seconds)},
            )
        except DeadlineExceeded as exc:
            return JSONResponse(status_code=504, content={"error": str(exc)})
        except NoBackendAvailable as exc:
            return JSONResponse(status_code=503, content={"error": str(exc)})
        headers = {
            "x-trident-backend": backend,
            "x-trident-latency-ms": f"{latency_ms:.1f}",
            "x-trident-workload": workload.value,
        }
        if isinstance(resp.body, (dict, list)):
            return JSONResponse(status_code=resp.status_code, content=resp.body, headers=headers)
        return Response(status_code=resp.status_code, content=resp.body, headers=headers)

    async def _dispatch_stream(route_req: RouteRequest) -> Response:
        # Streaming can't failover mid-stream: pick the primary and proxy bytes.
        decision: RoutingDecision = router.select(
            route_req.model, route_req.workload, route_req.slo_class,
            session=route_req.session, est_tokens=route_req.est_tokens,
        )
        rt = registry.get(decision.primary)
        metrics.DECISIONS.labels(backend=decision.primary, reason=decision.reason).inc()
        start = time.perf_counter()

        async def body():
            rt.stats.inflight += 1
            ok = True
            try:
                async for chunk in rt.adapter.stream(route_req):
                    yield chunk
            except httpx.HTTPError as exc:
                ok = False
                log.warning("stream from %s failed: %s", decision.primary, exc)
            finally:
                rt.stats.inflight -= 1
                latency_ms = (time.perf_counter() - start) * 1000
                rt.stats.record_result(latency_ms, ok=ok)
                metrics.REQUESTS.labels(
                    backend=decision.primary,
                    workload=route_req.workload.value,
                    status="ok" if ok else "error",
                ).inc()

        return StreamingResponse(
            body(),
            media_type="text/event-stream",
            headers={"x-trident-backend": decision.primary,
                     "x-trident-workload": route_req.workload.value},
        )

    # ---- OpenAI-compatible surface -------------------------------------------------

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        payload = await request.json()
        return await handle(request, "chat", str(payload.get("model", "")), payload)

    @app.post("/v1/completions")
    async def completions(request: Request):
        payload = await request.json()
        return await handle(request, "completion", str(payload.get("model", "")), payload)

    @app.post("/v1/embeddings")
    async def embeddings(request: Request):
        payload = await request.json()
        return await handle(request, "embedding", str(payload.get("model", "")), payload)

    # ---- KServe V2 / V1 surface ----------------------------------------------------

    @app.post("/v2/models/{model}/infer")
    async def v2_infer(model: str, request: Request):
        payload = await request.json()
        return await handle(request, "infer", model, payload)

    @app.post("/v1/models/{model_action}")
    async def v1_predict(model_action: str, request: Request):
        # KServe V1 uses /v1/models/{name}:predict — FastAPI can't express the
        # colon in a path template, so parse it out of the segment.
        model, _, action = model_action.partition(":")
        if action != "predict":
            return JSONResponse(status_code=404, content={"error": f"unknown action {action!r}"})
        payload = await request.json()
        return await handle(request, "infer", model, payload)

    # ---- Ops surface ---------------------------------------------------------------

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    @app.get("/metrics")
    async def prometheus_metrics():
        for name, rt in registry.backends.items():
            metrics.BACKEND_HEALTH.labels(backend=name).set(1 if rt.stats.healthy else 0)
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/admin/backends")
    async def admin_backends():
        return registry.snapshot()

    @app.get("/admin/models")
    async def admin_models():
        return registry.known_models()

    @app.post("/admin/backends/{name}/drain")
    async def drain_backend(name: str):
        """Stop routing new traffic to a backend; inflight requests finish
        naturally. Use before node maintenance, engine upgrades, or model swaps."""
        if name not in registry.backends:
            return JSONResponse(status_code=404, content={"error": f"unknown backend {name!r}"})
        registry.get(name).stats.draining = True
        log.info("backend %s set to draining", name)
        return {"backend": name, "draining": True}

    @app.post("/admin/backends/{name}/undrain")
    async def undrain_backend(name: str):
        if name not in registry.backends:
            return JSONResponse(status_code=404, content={"error": f"unknown backend {name!r}"})
        registry.get(name).stats.draining = False
        log.info("backend %s draining cleared", name)
        return {"backend": name, "draining": False}

    return app


def _declared_workloads(registry: Registry, model: str) -> list[WorkloadType]:
    out: set[WorkloadType] = set()
    for b in registry.backends.values():
        for m in b.cfg.models:
            if m.name == model:
                out.update(m.workloads)
    return list(out)
