# stt_sdk.py
# -----------------------------------------------------------------------------
# Minimal, production-ready async client for the STT service.
# - Auth: Authorization: Bearer <api_key>
# - Multipart upload: files={"audio":(...)} + form-encoded params
# - Retries/backoff on transient HTTP errors
# - Rolling latency stats + helper pretty printers
# - Pool client for round-robin across multiple base URLs
# -----------------------------------------------------------------------------
from __future__ import annotations

import asyncio
import io
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Tuple

import httpx

TelemetryHook = Callable[[Dict[str, Any]], None]


# =============================================================================
# Configuration / Models
# =============================================================================

@dataclass
class STTClientConfig:
    """
    Configuration for a single STT service endpoint.

    - base_url: e.g. "http://127.0.0.1:8081"
    - api_key: optional bearer token
    - timeout_s: client timeout
    - max_concurrency: client-side concurrency gate
    - max_retries: number of retries on failure (0 = no retry)
    - backoff_s: fixed sleep between retries (you can customize if needed)
    - collect_server_metrics: include server headers in telemetry events
    - default_params: always merged into request form fields (stringified)
    """
    base_url: str
    api_key: Optional[str] = None
    timeout_s: float = 60.0
    max_concurrency: int = 8
    max_retries: int = 2
    backoff_s: float = 0.25
    collect_server_metrics: bool = True
    default_params: Dict[str, Any] = field(
        default_factory=lambda: {"beam_size": 1, "vad_filter": True, "word_timestamps": False}
    )


@dataclass
class STTResult:
    """
    Canonical result returned by the SDK, normalized from the service response.

    Attributes mirror the service's JSON fields plus some convenience helpers.
    """
    text: str
    segments: List[Dict[str, Any]]
    language: Dict[str, Any]
    timings_ms: Dict[str, float]
    server: Dict[str, Any]
    gpu: Dict[str, Any]
    queue_depth: int
    headers: Dict[str, str]
    raw: Dict[str, Any]
    confidence: Dict[str, Any] = field(default_factory=dict)
    routing: Dict[str, Any] = field(default_factory=dict)
    final_model: Optional[str] = None
    fallback_used: bool = False

    def _hdr(self, name: str) -> Optional[str]:
        lname = name.lower()
        for k, v in self.headers.items():
            if k.lower() == lname:
                return v
        return None

    @property
    def seconds(self) -> Dict[str, float]:
        g = lambda k: float(self.timings_ms.get(k, 0.0)) / 1000.0
        return {
            "total": g("total"),
            "queue": g("queue_wait"),
            "primary_infer": g("primary_infer"),
            "fallback_infer": g("fallback_infer"),
            "model": (g("primary_infer") + g("fallback_infer")),
        }

    def low_confidence(self) -> bool:
        c = self.confidence or {}
        flags = [
            bool(c.get("below_threshold", False)),
            bool(c.get("below_langprob", False)),
            bool(c.get("too_short", False)),
        ]
        # In case a future server adds an explicit header for this:
        hb = self._hdr("X-Conf-Below")
        if hb is not None:
            s = str(hb).strip().lower()
            if s in ("true", "1", "yes", "y", "on"):
                flags.append(True)
        return any(flags)

    def pretty_seconds(self) -> str:
        s = self.seconds
        fm = self.final_model or self.routing.get("final_model", "?")
        return (f"Model:{fm} | total {s['total']:.3f}s "
                f"(primary {s['primary_infer']:.3f}s, fallback {s['fallback_infer']:.3f}s, queue {s['queue']:.3f}s) "
                f"| low_conf={self.low_confidence()}")

    def __repr__(self) -> str:
        return f"<STTResult {self.pretty_seconds()} | text_len={len(self.text)}>"


class STTError(Exception):
    """Base SDK error."""


class STTHTTPError(STTError):
    """Raised when the server returns a non-2xx status code."""

    def __init__(self, status_code: int, message: str, *, url: str, payload: Optional[Dict[str, Any]] = None):
        super().__init__(f"HTTP {status_code} for {url}: {message}")
        self.status_code = status_code
        self.url = url
        self.payload = payload or {}


# =============================================================================
# Rolling Stats
# =============================================================================

class RollingStats:
    """Thread/async-safe rolling latency statistics."""

    def __init__(self, capacity: int = 512):
        self.capacity = capacity
        self.samples_total: List[float] = []
        self.samples_model: List[float] = []
        self._lock = asyncio.Lock()

    async def add(self, *, total_ms: float, model_ms: float):
        async with self._lock:
            self.samples_total.append(total_ms)
            self.samples_model.append(model_ms)
            if len(self.samples_total) > self.capacity:
                self.samples_total = self.samples_total[-self.capacity:]
                self.samples_model = self.samples_model[-self.capacity:]

    @staticmethod
    def _percentile(data: List[float], p: float) -> Optional[float]:
        if not data:
            return None
        k = max(0, min(len(data) - 1, int(round((p / 100.0) * (len(data) - 1)))))
        return sorted(data)[k]

    def summary(self) -> Dict[str, Optional[float]]:
        import statistics
        return {
            "count": len(self.samples_total),
            "p50_total_ms": self._percentile(self.samples_total, 50),
            "p95_total_ms": self._percentile(self.samples_total, 95),
            "p99_total_ms": self._percentile(self.samples_total, 99),
            "avg_total_ms": statistics.fmean(self.samples_total) if self.samples_total else None,
            "avg_model_ms": statistics.fmean(self.samples_model) if self.samples_model else None,
        }


# =============================================================================
# Utilities
# =============================================================================

def _bool_to_str(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return "true" if bool(v) else "false"
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("1", "true", "yes", "y", "on"):
            return "true"
        return "false"
    return "false"


def _merge_params(defaults: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, str]:
    """
    Merge default + override params and stringify for form fields.

    - Skips None values (keeps requests smaller)
    - Booleans become "true"/"false"
    """
    out: Dict[str, str] = {}
    merged = {**defaults, **overrides}
    for k, v in merged.items():
        if v is None:
            continue
        if isinstance(v, bool) or (isinstance(v, (int, float, str)) and str(v).lower() in ("true", "false")):
            out[k] = _bool_to_str(v)
        else:
            out[k] = str(v)
    return out


# =============================================================================
# Core Client
# =============================================================================

class STTClient:
    """
    High-level async client for a single STT endpoint.

    Usage:
        async with STTClient(STTClientConfig(base_url="http://...")) as cli:
            res = await cli.transcribe_file("audio.wav", language="en")
            print(res.text)
    """

    def __init__(self, cfg: STTClientConfig, on_telemetry: Optional[TelemetryHook] = None):
        self.cfg = cfg
        self.sem = asyncio.Semaphore(cfg.max_concurrency)
        self.client = httpx.AsyncClient(
            base_url=cfg.base_url.rstrip("/"),
            timeout=cfg.timeout_s,
            headers={"Authorization": f"Bearer {cfg.api_key}"} if cfg.api_key else {},
        )
        self.stats = RollingStats()
        self.on_telemetry = on_telemetry

    # ---- context management ----
    async def __aenter__(self) -> "STTClient":
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.close()

    async def close(self):
        await self.client.aclose()

    # ---- helpers ----
    def _hdr(self, headers: Dict[str, str], name: str) -> Optional[str]:
        lname = name.lower()
        for k, v in headers.items():
            if k.lower() == lname:
                return v
        return None

    def _parse_timings(self, headers: Dict[str, str], payload: Dict[str, Any]) -> Tuple[float, float, float, float]:
        """
        Extract total/model/queue/fallback timings (ms) from headers or payload.
        Returns: total_ms, primary_ms, fallback_ms, queue_ms
        """
        t_hdr = self._hdr(headers, "X-Total-ms")
        p_hdr = self._hdr(headers, "X-Primary-Infer-ms")
        f_hdr = self._hdr(headers, "X-Fallback-Infer-ms")
        q_hdr = self._hdr(headers, "X-Queue-Wait-ms")

        timings = payload.get("timings_ms", {}) or {}

        total_ms = float(t_hdr or timings.get("total") or 0.0)
        primary_ms = float(p_hdr or timings.get("primary_infer") or 0.0)
        fallback_ms = float(f_hdr or timings.get("fallback_infer") or 0.0)
        queue_ms = float(q_hdr or timings.get("queue_wait") or 0.0)

        # If we didn't get "total" from server, estimate from parts we have
        if total_ms <= 0.0:
            total_ms = primary_ms + fallback_ms + queue_ms

        return total_ms, primary_ms, fallback_ms, queue_ms

    # ---- core request ----
    async def _post_audio(self, audio_bytes: bytes, params: Dict[str, Any]) -> STTResult:
        """
        POST /v1/transcribe with multipart {audio, form fields}.
        Retries on failure up to cfg.max_retries times.
        """
        last_exc: Optional[Exception] = None
        url = "/v1/transcribe"

        for attempt in range(self.cfg.max_retries + 1):
            try:
                files = {"audio": ("audio.wav", io.BytesIO(audio_bytes), "audio/wav")}
                data = _merge_params(self.cfg.default_params, params)

                # keep request size light: skip empty language
                if "language" in data and (data["language"] == "" or data["language"].lower() == "none"):
                    data.pop("language", None)

                t0 = time.perf_counter()
                r = await self.client.post(url, files=files, data=data)
                t1 = time.perf_counter()

                # Raise if not 2xx
                try:
                    r.raise_for_status()
                except httpx.HTTPStatusError as e:
                    # Bubble up with any JSON error info if available
                    msg = ""
                    body_json: Optional[Dict[str, Any]] = None
                    try:
                        body_json = r.json()
                        msg = body_json.get("detail") or body_json.get("message") or r.text
                    except Exception:
                        msg = r.text
                    raise STTHTTPError(r.status_code, msg, url=str(r.url), payload=body_json) from e

                payload = r.json()
                xhdrs = {k: v for k, v in r.headers.items() if k.lower().startswith("x-")}

                total_ms_h, primary_ms_h, fallback_ms_h, queue_ms_h = self._parse_timings(xhdrs, payload)
                wall_ms = (t1 - t0) * 1000.0
                total_ms = total_ms_h or wall_ms
                model_ms = (primary_ms_h + fallback_ms_h) if (primary_ms_h or fallback_ms_h) else float(
                    payload.get("timings_ms", {}).get("infer", 0.0) or 0.0
                )

                conf = payload.get("confidence", {}) or {}
                routing = payload.get("routing", {}) or {}
                fallback_used = routing.get("fallback_used")
                if fallback_used is None:
                    fb_hdr = self._hdr(xhdrs, "X-Fallback-Used")
                    fallback_used = (str(fb_hdr).strip().lower() == "true") if fb_hdr is not None else False
                final_model = routing.get("final_model") or (self._hdr(xhdrs, "X-Final-Model") or None)

                result = STTResult(
                    text=payload.get("text", ""),
                    segments=payload.get("segments", []) or [],
                    language=payload.get("language", {}) or {},
                    timings_ms={
                        "total": float(total_ms),
                        "primary_infer": float(primary_ms_h),
                        "fallback_infer": float(fallback_ms_h),
                        "queue_wait": float(queue_ms_h),
                        # carry forward any extra timings server provided
                        **{
                            k: v for k, v in (payload.get("timings_ms", {}) or {}).items()
                            if k not in ("total", "primary_infer", "fallback_infer", "queue_wait")
                        },
                    },
                    server=payload.get("server", {}) or {},
                    gpu=payload.get("gpu", {}) or {},
                    queue_depth=int(self._hdr(xhdrs, "X-Queue-Depth") or payload.get("queue_depth", 0) or 0),
                    headers=xhdrs,
                    raw=payload,
                    confidence=conf,
                    routing=routing,
                    final_model=final_model,
                    fallback_used=bool(fallback_used),
                )

                # record stats
                await self.stats.add(total_ms=result.timings_ms["total"], model_ms=model_ms)

                # optional telemetry callback
                if self.on_telemetry:
                    telem: Dict[str, Any] = {
                        "type": "stt_request",
                        "ok": True,
                        "attempt": attempt,
                        "headers": xhdrs if self.cfg.collect_server_metrics else {},
                        "timings_ms": result.timings_ms,
                        "gpu": result.gpu,
                        "queue_depth": result.queue_depth,
                        "server": result.server,
                        "routing": {"fallback_used": result.fallback_used, "final_model": result.final_model},
                        "confidence": result.confidence,
                    }
                    try:
                        self.on_telemetry(telem)
                    except Exception:
                        # Never break the main flow on telemetry errors
                        pass

                return result

            except (httpx.TransportError, httpx.ReadTimeout, httpx.WriteTimeout) as e:
                last_exc = e
                if self.on_telemetry:
                    try:
                        self.on_telemetry({"type": "stt_request", "ok": False, "attempt": attempt, "error": repr(e)})
                    except Exception:
                        pass
                if attempt < self.cfg.max_retries:
                    await asyncio.sleep(self.cfg.backoff_s)
                else:
                    raise
            except STTHTTPError as e:
                last_exc = e
                if self.on_telemetry:
                    try:
                        self.on_telemetry({
                            "type": "stt_request",
                            "ok": False,
                            "attempt": attempt,
                            "status_code": e.status_code,
                            "error": str(e),
                            "payload": e.payload,
                        })
                    except Exception:
                        pass
                # Retry only for 5xx on next attempt
                if attempt < self.cfg.max_retries and 500 <= e.status_code < 600:
                    await asyncio.sleep(self.cfg.backoff_s)
                else:
                    raise

        # If we exit loop without returning or raising, surface last exception
        if last_exc:
            raise last_exc
        raise STTError("Unexpected error in _post_audio (no response and no exception).")

    # ---- public API ----
    async def transcribe_bytes(
        self,
        audio_bytes: bytes,
        *,
        language: Optional[str] = None,
        beam_size: int = 1,
        vad_filter: bool = True,
        word_timestamps: bool = False,
        conf_threshold: Optional[float] = None,
        langprob_threshold: Optional[float] = None,
        min_words_for_conf: Optional[int] = None,
    ) -> STTResult:
        """
        Transcribe raw audio bytes.

        language:
            - If None: server will auto-detect.
            - If provided (e.g. "en"): server will run primary+fallback with that language.
        """
        params: Dict[str, Any] = {
            "beam_size": int(beam_size),
            "vad_filter": bool(vad_filter),
            "word_timestamps": bool(word_timestamps),
        }
        # Only include language if provided (keeps request small)
        if language:
            params["language"] = str(language)

        if conf_threshold is not None:
            params["conf_threshold"] = str(conf_threshold)
        if langprob_threshold is not None:
            params["langprob_threshold"] = str(langprob_threshold)
        if min_words_for_conf is not None:
            params["min_words_for_conf"] = str(min_words_for_conf)

        async with self.sem:
            return await self._post_audio(audio_bytes, params)

    async def transcribe_file(self, path: str, **kwargs) -> STTResult:
        with open(path, "rb") as f:
            data = f.read()
        return await self.transcribe_bytes(data, **kwargs)

    # ---- observability helpers ----
    def latency_summary(self) -> Dict[str, Optional[float]]:
        return self.stats.summary()

    async def server_metrics(self) -> Dict[str, Any]:
        r = await self.client.get("/v1/metrics")
        r.raise_for_status()
        return r.json()

    async def server_health(self) -> Dict[str, Any]:
        r = await self.client.get("/v1/health")
        r.raise_for_status()
        return r.json()

    def estimate_throughput(self, target_p95_ms: Optional[float] = None) -> Dict[str, Any]:
        s = self.stats.summary()
        avg_total = s.get("avg_total_ms") or 0
        p95 = s.get("p95_total_ms") or 0
        rec = None
        if target_p95_ms and avg_total > 0:
            # trivial sizing heuristic: parallelism ≈ target_p95 / avg
            rec = max(1, int(target_p95_ms / avg_total))
        return {
            "observed_avg_ms": avg_total,
            "observed_p95_ms": p95,
            "recommended_client_concurrency_for_target_p95": rec,
        }


# =============================================================================
# Pool Client (round-robin across multiple hosts)
# =============================================================================

class STTPool:
    """
    Simple round-robin pool across multiple STT hosts.

    Example:
        pool = STTPool(["http://10.0.0.1:8081", "http://10.0.0.2:8081"], api_key="...")
        res = await pool.transcribe_file("a.wav", language="en")
    """

    def __init__(
        self,
        base_urls: List[str],
        *,
        api_key: Optional[str] = None,
        per_host_concurrency: int = 4,
        timeout_s: float = 60.0,
        max_retries: int = 2,
        backoff_s: float = 0.25,
        default_params: Optional[Dict[str, Any]] = None,
        on_telemetry: Optional[TelemetryHook] = None,
    ):
        cfgs = [
            STTClientConfig(
                base_url=u,
                api_key=api_key,
                timeout_s=timeout_s,
                max_concurrency=per_host_concurrency,
                max_retries=max_retries,
                backoff_s=backoff_s,
                default_params=(default_params or {"beam_size": 1, "vad_filter": True, "word_timestamps": False}),
            )
            for u in base_urls
        ]
        self.clients = [STTClient(cfg, on_telemetry=on_telemetry) for cfg in cfgs]
        self._rr = _RoundRobin(len(self.clients))
        self._lock = asyncio.Lock()

    async def __aenter__(self) -> "STTPool":
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.close()

    async def close(self):
        await asyncio.gather(*[c.close() for c in self.clients])

    async def transcribe_bytes(self, audio: bytes, **kwargs) -> STTResult:
        async with self._lock:
            idx = self._rr.next()
        try:
            return await self.clients[idx].transcribe_bytes(audio, **kwargs)
        except Exception:
            # Simple failover to a different host once
            alt = self._rr.next_not(idx)
            return await self.clients[alt].transcribe_bytes(audio, **kwargs)

    async def transcribe_file(self, path: str, **kwargs) -> STTResult:
        with open(path, "rb") as f:
            data = f.read()
        return await self.transcribe_bytes(data, **kwargs)


class _RoundRobin:
    def __init__(self, n: int):
        from itertools import cycle
        self._idxs = list(range(n))
        self._cyc = cycle(self._idxs)

    def next(self) -> int:
        return next(self._cyc)

    def next_not(self, i: int) -> int:
        # return any index != i
        for _ in range(len(self._idxs)):
            j = next(self._cyc)
            if j != i:
                return j
        return i  # fallback (n=1)
