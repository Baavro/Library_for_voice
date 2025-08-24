from __future__ import annotations
from typing import List
from .client import AsyncTTSClient

def prom_text(endpoints: List[AsyncTTSClient]) -> str:
    """
    Returns a small Prometheus-compatible text snippet you can serve from your API.
    """
    lines = []
    for ep in endpoints:
        s = ep.snapshot()
        labels = f'endpoint="{s.base_url}"'
        def emit(name, value):
            if value is None: return
            lines.append(f'orpheus_tts_{name}{{{labels}}} {value}')
        emit("in_flight", s.in_flight)
        emit("max_concurrency", s.max_concurrency)
        emit("ema_ttfb_ms", s.ema_ttfb_ms)
        emit("p50_ms", s.p50_ms)
        emit("p95_ms", s.p95_ms)
        emit("gpu_util", s.gpu_util)
        emit("vram_used_mb", s.vram_used_mb)
        emit("vram_total_mb", s.vram_total_mb)
        emit("streams_active", s.streams_active)
        emit("queue_depth", s.queue_depth)
        emit("breaker_open", 1 if s.breaker_open else 0)
    return "\n".join(lines) + "\n"
