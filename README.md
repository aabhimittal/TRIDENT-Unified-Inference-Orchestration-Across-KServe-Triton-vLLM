# TRIDENT

**Unified, SLO-aware inference orchestration across KServe, NVIDIA Triton, and vLLM.**

Each serving stack has a distinct sweet spot:

- **Triton** dominates CV, ensemble graphs, and TensorRT workloads.
- **vLLM** owns LLM throughput via PagedAttention and continuous batching.
- **KServe** provides the Kubernetes-native deployment lifecycle — canary rollout, scale-to-zero, revision management.

What none of them provide is a **routing brain that unifies all three under SLO-aware dispatch**. TRIDENT is that brain: a single gateway that classifies every request, scores every capable backend against live telemetry and the request's latency budget, and dispatches to the best engine — with failover, circuit breaking, canary splits, and shadow mirroring built in.

```mermaid
flowchart LR
    C1[OpenAI clients] -->|/v1/chat/completions| G
    C2[CV pipelines] -->|/v2/models/m/infer| G
    C3[Legacy clients] -->|/v1/models/m:predict| G

    subgraph TRIDENT gateway
        G[Classify workload] --> S[SLO-aware scorer]
        S --> D[Dispatcher<br/>failover + breaker]
        P[Telemetry poller<br/>health + /metrics] -.-> S
    end

    D -->|OpenAI protocol| V[vLLM<br/>LLM chat / completions]
    D -->|KServe V2 protocol| T[Triton<br/>tensor / ensembles / TRT-LLM]
    D -->|V1 / V2 / OpenAI| K[KServe InferenceServices<br/>canary, scale-to-zero]
    D -.->|shadow mirror| K
```

## How routing works

Every request is classified into a workload (`llm_chat`, `llm_completion`, `embedding`, `tensor`, `ensemble`), then every healthy, capable backend is scored:

```
score = w_lat · latency_fit + w_aff · affinity + w_head · headroom + w_bias · bias
```

| Term | Signal | Where it comes from |
|---|---|---|
| `latency_fit` | Predicted latency vs. the request's SLO budget | EWMA of observed latencies, inflated by in-flight requests and scraped queue depth; quadratic penalty past the budget |
| `affinity` | Engine ↔ workload fit | vLLM ≈ 1.0 for token generation, Triton ≈ 1.0 for tensor/ensemble; overridable per backend |
| `headroom` | `1 − utilization` | vLLM KV-cache usage (`vllm:gpu_cache_usage_perc`) or Triton GPU utilization (`nv_gpu_utilization`), scraped from `/metrics` |
| `bias` | Operator preference | Static per-backend config (e.g. prefer spot capacity) |

The interplay is the point: with a relaxed `batch` SLO, affinity keeps chat on vLLM even when it's slow; under a tight `interactive` SLO, the same overloaded vLLM loses to a TensorRT-LLM copy on Triton that answers in 300 ms. Clients pick their budget with one header: `x-trident-slo: interactive`.

On top of scoring:

- **Failover** — if the top backend connection-errors or 5xxes, the request retries down the ranking (`routing.max_attempts`).
- **Circuit breakers** — consecutive failures open the breaker; after a cooldown a single half-open probe is admitted.
- **Canary** — deterministic traffic splits per model (`10% of llama-3-8b to the TRT-LLM build`), applied only across backends that passed health/breaker filters.
- **Shadow** — mirror a sample of a model's traffic to another backend, fire-and-forget, for validating a new deployment against production traffic.
- **Telemetry poller** — health checks (`/health`, `/v2/health/ready`) and Prometheus scrapes run off the request path; the scorer reads state lock-free.

## API surface

One gateway, three protocol dialects — clients keep whatever they already speak:

| Endpoint | Protocol | Workloads |
|---|---|---|
| `POST /v1/chat/completions` | OpenAI (streaming supported) | `llm_chat` |
| `POST /v1/completions` | OpenAI | `llm_completion` |
| `POST /v1/embeddings` | OpenAI | `embedding` |
| `POST /v2/models/{m}/infer` | KServe V2 / Triton native | `tensor`, `ensemble` |
| `POST /v1/models/{m}:predict` | KServe V1 | `tensor` |
| `GET /metrics` | Prometheus | router observability |
| `GET /admin/backends` | JSON | live backend state (EWMA, breaker, utilization) |
| `GET /admin/models` | JSON | model → backends map |

Responses carry `x-trident-backend`, `x-trident-workload`, and `x-trident-latency-ms` headers so you can always see where a request landed and why.

## Quickstart (no GPUs needed)

```bash
pip install -e ".[dev]"

# Terminal 1: fake vLLM (:8001) + fake Triton (:8002)
python examples/mock_backends.py

# Terminal 2: the router
trident --config examples/demo.yaml --port 8080

# Chat goes to the vLLM backend...
curl -s localhost:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model": "llama-3-8b", "messages": [{"role": "user", "content": "hi"}]}' -i

# ...tensor inference goes to Triton...
curl -s localhost:8080/v2/models/resnet50/infer \
  -H 'content-type: application/json' \
  -d '{"inputs": [{"name": "input0", "shape": [1,3], "datatype": "FP32", "data": [1,2,3]}]}' -i

# ...and the routing state is inspectable:
curl -s localhost:8080/admin/backends | python -m json.tool
```

Run the test suite with `pytest -q`.

## Configuration

One YAML file declares SLO classes, routing weights, backends, and canary/shadow rules — see [`config/trident.example.yaml`](config/trident.example.yaml) for the annotated full example:

```yaml
slo_classes:
  interactive: { target_p95_ms: 1500, priority: 10 }

routing:
  canary:
    - { model: llama-3-8b, stable: vllm-llama, candidate: triton-trtllm, weight: 0.10 }
  shadow:
    - { model: resnet50, target: kserve-resnet, sample: 0.20 }

backends:
  - name: vllm-llama
    kind: vllm                      # protocol defaults to openai
    base_url: http://vllm-llama.serving:8000
    metrics_url: http://vllm-llama.serving:8000/metrics
    max_concurrency: 64
    models:
      - name: llama-3-8b
        upstream_name: meta-llama/Meta-Llama-3-8B-Instruct
        workloads: [llm_chat, llm_completion]

  - name: kserve-embed
    kind: kserve
    protocol: openai                # vLLM ServingRuntime behind KServe
    base_url: http://embedder.serving.svc.cluster.local
    models:
      - name: bge-large
        workloads: [embedding]
```

KServe backends declare which dialect their predictor speaks (`v1`, `v2`, or `openai`), since KServe is a deployment layer rather than an engine — and their affinity can be overridden to match the runtime actually inside the InferenceService.

## Deploying on Kubernetes

[`deploy/k8s/`](deploy/k8s) contains a complete reference stack:

- [`trident.yaml`](deploy/k8s/trident.yaml) — the router Deployment + ConfigMap + Service
- [`backends/vllm-llama.yaml`](deploy/k8s/backends/vllm-llama.yaml) — raw vLLM deployment
- [`backends/triton-cv.yaml`](deploy/k8s/backends/triton-cv.yaml) — raw Triton deployment
- [`backends/kserve-inferenceservices.yaml`](deploy/k8s/backends/kserve-inferenceservices.yaml) — KServe-managed predictors (Triton runtime + vLLM ServingRuntime)

The intended division of labor: KServe owns the *lifecycle* of predictors that benefit from canary/scale-to-zero; raw vLLM/Triton deployments own steady-state heavy traffic; TRIDENT owns *dispatch* across all of them.

## Project layout

```
trident/
  gateway.py      # FastAPI front door: OpenAI + KServe V1/V2 endpoints, dispatch w/ failover
  router.py       # routing brain: capability filter -> canary -> SLO-aware ranking
  scoring.py      # the utility function (latency fit, affinity, headroom, bias)
  telemetry.py    # EWMAs, circuit breakers, Prometheus text parsing
  poller.py       # background health + metrics scraping loop
  registry.py     # backend runtime state (config + stats + adapter)
  adapters/       # protocol translation: vllm (OpenAI), triton (V2), kserve (V1/V2/OpenAI)
  config.py       # pydantic schema for the YAML config
  metrics.py      # TRIDENT's own Prometheus metrics
deploy/k8s/       # reference Kubernetes manifests
examples/         # GPU-free demo (mock backends + demo config)
tests/            # 52 tests: scoring, routing, breakers, adapters, e2e gateway
```

## Status & roadmap

Working: unified gateway, SLO-aware scoring, telemetry-driven dispatch, failover, circuit breaking, canary, shadow, Prometheus observability, K8s manifests.

Planned:

- gRPC ingress (Triton clients that speak gRPC today)
- Priority-based preemption/queueing when all backends are saturated (the `priority` field is scored but not yet a queue discipline)
- Token-aware LLM cost prediction (prompt length → latency estimate, not just EWMA)
- KServe controller integration: watch InferenceService status instead of polling
- Autoscaling hints: export queue-pressure signals per backend as KEDA/HPA metrics

## License

MIT
