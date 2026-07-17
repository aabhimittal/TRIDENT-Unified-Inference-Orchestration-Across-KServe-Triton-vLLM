"""Protocol adapters translating TRIDENT's unified request onto each engine's wire API."""

from .base import BaseAdapter
from .kserve import KServeAdapter
from .triton import TritonAdapter
from .vllm import VLLMAdapter

from ..config import BackendConfig
from ..model import BackendKind

__all__ = ["BaseAdapter", "VLLMAdapter", "TritonAdapter", "KServeAdapter", "build_adapter"]


def build_adapter(cfg: BackendConfig, client) -> BaseAdapter:
    if cfg.kind == BackendKind.VLLM:
        return VLLMAdapter(cfg, client)
    if cfg.kind == BackendKind.TRITON:
        return TritonAdapter(cfg, client)
    return KServeAdapter(cfg, client)
