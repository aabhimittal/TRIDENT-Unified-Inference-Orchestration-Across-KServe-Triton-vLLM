from __future__ import annotations

from trident.model import WorkloadType
from trident.scoring import latency_fit, predict_latency_ms, rank_candidates
from trident.telemetry import BackendStats


def _by_name(cfg, name):
    return next(b for b in cfg.backends if b.name == name)


def _rank(config, names, stats_map, workload, slo_name="default"):
    slo = config.slo_classes.get(slo_name) or config.slo_classes["default"]
    candidates = [(_by_name(config, n), stats_map[n]) for n in names]
    return rank_candidates(candidates, workload, slo, config.routing)


def test_vllm_preferred_for_chat(base_config):
    stats = {"vllm-a": BackendStats(), "triton-llm": BackendStats()}
    ranked = _rank(base_config, ["vllm-a", "triton-llm"], stats, WorkloadType.LLM_CHAT)
    assert ranked[0].backend_name == "vllm-a"


def test_triton_preferred_for_tensor(base_config):
    stats = {"vllm-a": BackendStats(), "triton-cv": BackendStats()}
    ranked = _rank(base_config, ["vllm-a", "triton-cv"], stats, WorkloadType.TENSOR)
    assert ranked[0].backend_name == "triton-cv"


def test_slo_pressure_shifts_to_faster_backend(base_config):
    """Under a tight SLO, a slow-but-preferred engine loses to a fast alternative."""
    vllm_stats, triton_stats = BackendStats(), BackendStats()
    for _ in range(10):
        vllm_stats.latency_ms.update(2500)   # vLLM overloaded: 2.5s observed
        triton_stats.latency_ms.update(300)  # TRT-LLM copy answering in 300ms
    ranked = _rank(
        base_config, ["vllm-a", "triton-llm"],
        {"vllm-a": vllm_stats, "triton-llm": triton_stats},
        WorkloadType.LLM_CHAT, slo_name="interactive",
    )
    assert ranked[0].backend_name == "triton-llm"


def test_relaxed_slo_keeps_affinity_winner(base_config):
    """With a huge batch budget, both fit the SLO and affinity dominates."""
    vllm_stats, triton_stats = BackendStats(), BackendStats()
    for _ in range(10):
        vllm_stats.latency_ms.update(2500)
        triton_stats.latency_ms.update(300)
    ranked = _rank(
        base_config, ["vllm-a", "triton-llm"],
        {"vllm-a": vllm_stats, "triton-llm": triton_stats},
        WorkloadType.LLM_CHAT, slo_name="batch",
    )
    assert ranked[0].backend_name == "vllm-a"


def test_unhealthy_backend_excluded(base_config):
    bad = BackendStats(healthy=False)
    ranked = _rank(base_config, ["vllm-a", "triton-llm"],
                   {"vllm-a": bad, "triton-llm": BackendStats()}, WorkloadType.LLM_CHAT)
    assert [c.backend_name for c in ranked] == ["triton-llm"]


def test_open_breaker_excluded(base_config):
    stats = BackendStats()
    for _ in range(5):
        stats.breaker.record_failure()
    ranked = _rank(base_config, ["vllm-a", "triton-llm"],
                   {"vllm-a": stats, "triton-llm": BackendStats()}, WorkloadType.LLM_CHAT)
    assert [c.backend_name for c in ranked] == ["triton-llm"]


def test_predicted_latency_inflates_with_pressure(base_config):
    cfg = _by_name(base_config, "vllm-a")
    idle, busy = BackendStats(), BackendStats()
    idle.latency_ms.update(100)
    busy.latency_ms.update(100)
    busy.inflight = 32
    busy.queue_depth = 20
    assert predict_latency_ms(cfg, busy, 250) > predict_latency_ms(cfg, idle, 250)


def test_headroom_breaks_latency_ties(base_config):
    """Same engine kind, same latency: the less-saturated backend wins."""
    hot, cool = BackendStats(), BackendStats()
    hot.latency_ms.update(200)
    cool.latency_ms.update(200)
    hot.utilization = 0.95
    cool.utilization = 0.10
    ranked = _rank(base_config, ["vllm-a", "triton-llm"],
                   {"vllm-a": hot, "triton-llm": cool}, WorkloadType.LLM_COMPLETION)
    # vLLM has higher affinity for completions, but saturation costs it here:
    # w_head * (0.95-0.10) = 0.17 > w_aff * (1.0-0.5) = 0.125
    assert ranked[0].backend_name == "triton-llm"


def test_latency_fit_shape(base_config):
    slo = base_config.slo_classes["interactive"]  # 1500ms budget
    assert latency_fit(0, slo) == 1.0
    assert latency_fit(1500, slo) == 0.5
    assert latency_fit(3000, slo) < latency_fit(1500, slo) / 1.5  # quadratic falloff
