from __future__ import annotations
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional, Dict, Any, List
import time

@dataclass
class TTSRequest:
    text: str
    voice: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)
    request_id: Optional[str] = None
    timeout_s: float = 60.0

@dataclass
class TTSChunk:
    audio: bytes
    t_sent: float  # client receive time (monotonic)
    seq: int

@dataclass
class TTSServerMetrics:
    gpu_util: Optional[float] = None            # 0..1
    vram_used_mb: Optional[float] = None
    vram_total_mb: Optional[float] = None
    streams_active: Optional[int] = None
    queue_depth: Optional[int] = None
    server_concurrency: Optional[int] = None
    extras: Dict[str, Any] = field(default_factory=dict)
    t_observed: float = field(default_factory=time.perf_counter)

@dataclass
class RequestTimings:
    t_start: float
    t_connect: Optional[float] = None
    t_first_byte: Optional[float] = None
    t_end: Optional[float] = None
    bytes_rcv: int = 0
    chunks: int = 0

    def ttfb_ms(self) -> Optional[float]:
        if self.t_first_byte and self.t_start:
            return (self.t_first_byte - self.t_start) * 1000
        return None

    def total_ms(self) -> Optional[float]:
        if self.t_end and self.t_start:
            return (self.t_end - self.t_start) * 1000
        return None

@dataclass
class RollingStats:
    # Light, fast rolling stats for p50/p95 approximations
    last_n_lat_ms: List[float] = field(default_factory=list)
    window: int = 512

    def add(self, ms: float):
        self.last_n_lat_ms.append(ms)
        if len(self.last_n_lat_ms) > self.window:
            self.last_n_lat_ms.pop(0)

    def p(self, q: float) -> Optional[float]:
        if not self.last_n_lat_ms: return None
        arr = sorted(self.last_n_lat_ms)
        idx = max(0, min(len(arr)-1, int(q * (len(arr)-1))))
        return arr[idx]

@dataclass
class EndpointSnapshot:
    base_url: str
    in_flight: int
    max_concurrency: int
    ema_ttfb_ms: Optional[float]
    p50_ms: Optional[float]
    p95_ms: Optional[float]
    gpu_util: Optional[float]
    vram_used_mb: Optional[float]
    vram_total_mb: Optional[float]
    streams_active: Optional[int]
    queue_depth: Optional[int]
    breaker_open: bool
