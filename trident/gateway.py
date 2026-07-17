"""The TRIDENT gateway: one front door, three engines behind it.

Exposed surface:
- OpenAI-compatible:  POST /v1/chat/completions, /v1/completions, /v1/embeddings
- KServe V2:          POST /v2/models/{model}/infer  (tensor + ensemble workloads)
- KServe V1:          POST /v1/models/{model}:predict
- Ops:                GET /healthz, /metrics, /admin/backends, /admin/models

Every request is classified into a workload, scored against the live state of
all capable backends, dispatched with failover, and (optionally) shadowed.
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
from .router import NoBackendAvailable, Router, classify_workload

log = logging.getLogger("trident.gateway")


class Dispatcher:
    """Executes routing decisions: primary dispatch, failover, shadow mirroring."""

    def __init__(self, config: TridentConfig, registry: Registry, router: Router) -> None:
        self.config = config
        self.registry = registry
        self.router = router

    async def dispatch(self, request: RouteRequest) -> tuple[AdapterResponse, str, float]:
        """Route and execute; returns (response, backend_name, latency_ms)."""
        decision = self.router.select(request.model, request.workload, request.slo_class)
        if decision.shadow:
            asyncio.get_running_loop().create_task(self._shadow(decision.shadow, request))

        order = self.router.failover_order(decision)
        attempts = min(self.config.routing.max_attempts, len(order))
        last_exc: Exception | None = None
        for i, name in enumerate(order[:attempts]):
            rt = self.registry.get(name)
            if not rt.stats.breaker.allow():
                continue
            reason = decision.reason if i == 0 else "failover"
            metrics.DECISIONS.labels(backend=name, reason=reason).inc()
            rt.stats.inflight += 1
            start = time.perf_counter()
            try:
                resp = await rt.adapter.infer(request)
            except (httpx.HTTPError, ValueError) as exc:
                rt.stats.record_result(0.0, ok=False)
                metrics.REQUESTS.labels(
                    backend=name, workload=request.workload.value, status="error"
                ).inc()
                log.warning("dispatch to %s failed: %s", name, exc)
                last_exc = exc
                continue
            finally:
                rt.stats.inflight -= 1
            latency_ms = (time.perf_counter() - start) * 1000
            # 5xx counts against the backend; 4xx is the caller's problem.
            backend_ok = resp.status_code < 500
            rt.stats.record_result(latency_ms, ok=backend_ok)
            metrics.REQUESTS.labels(
                backend=name,
                workload=request.workload.value,
                status="ok" if resp.ok else str(resp.status_code),
            ).inc()
            metrics.LATENCY.labels(backend=name, workload=request.workload.value).observe(
                latency_ms / 1000
            )
            if not backend_ok and i + 1 < attempts:
                continue
            return resp, name, latency_ms

        raise NoBackendAvailable(
            request.model, request.workload,
            f"all {attempts} attempt(s) failed" + (f"; last error: {last_exc}" if last_exc else ""),
        )

    async def _shadow(self, backend_name: str, request: RouteRequest) -> None:
        rt = self.registry.get(backend_name)
        try:
            resp = await rt.adapter.infer(request)
            metrics.SHADOWS.labels(
                backend=backend_name, status="ok" if resp.ok else str(resp.status_code)
            ).inc()
        except Exception as exc:
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

    app = FastAPI(title="TRIDENT", version="0.1.0", lifespan=lifespan)
    app.state.registry = registry
    app.state.router = router
    app.state.dispatcher = dispatcher

    def error_response(exc: NoBackendAvailable) -> JSONResponse:
        return JSONResponse(status_code=503, content={"error": str(exc)})

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
            stream=bool(payload.get("stream")),
            request_id=request.headers.get("x-request-id", str(uuid.uuid4())),
        )
        try:
            if route_req.stream:
                return await _dispatch_stream(route_req)
            resp, backend, latency_ms = await dispatcher.dispatch(route_req)
        except NoBackendAvailable as exc:
            return error_response(exc)
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
            route_req.model, route_req.workload, route_req.slo_class
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

    return app


def _declared_workloads(registry: Registry, model: str) -> list[WorkloadType]:
    out: set[WorkloadType] = set()
    for b in registry.backends.values():
        for m in b.cfg.models:
            if m.name == model:
                out.update(m.workloads)
    return list(out)
