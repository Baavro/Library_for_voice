# from __future__ import annotations
# import asyncio, base64, json, uuid, time
# from typing import AsyncIterator, Optional, Dict, Any
# import httpx

# from .types import TTSRequest, TTSChunk, TTSServerMetrics, RequestTimings, RollingStats, EndpointSnapshot
# from .metrics import EMA, CircuitBreaker, AIMDConcurrency

# class AsyncTTSClient:
#     """
#     Async client for a *single* TTS endpoint (one GPU box).
#     Streams NDJSON frames and yields raw PCM chunks.
#     Tracks GPU/VRAM, p50/p95, TTFB, in-flight, and adapts concurrency.
#     """
#     def __init__(
#         self,
#         base_url: str,
#         api_key: Optional[str] = None,
#         *,
#         connect_timeout=5.0,
#         read_timeout=60.0,
#         http2=True,
#         init_concurrency=8,
#         max_concurrency=64,
#     ):
#         self.base_url = base_url.rstrip("/")
#         self.api_key = api_key
#         self.client = httpx.AsyncClient(
#     http2=http2,
#     timeout=httpx.Timeout(
#         connect=connect_timeout,
#         read=read_timeout,
#         write=read_timeout,
#         pool=connect_timeout,
#     ),
# )
#         self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
#         self._sem = asyncio.Semaphore(init_concurrency)
#         self._aimd = AIMDConcurrency(init=init_concurrency, max_cap=max_concurrency)
#         self._breaker = CircuitBreaker()
#         self._in_flight = 0

#         # Observability
#         self._ema_ttfb = EMA(0.2)
#         self._lat_stats = RollingStats(window=512)
#         self._last_server = TTSServerMetrics()

#     @property
#     def in_flight(self): return self._in_flight
#     @property
#     def max_concurrency(self): return self._sem._value + self._in_flight  # effective cap

#     async def _acquire(self):
#         # Basic breaker + concurrency gate
#         if not self._breaker.allow():
#             raise RuntimeError("Circuit open for endpoint")
#         await self._sem.acquire()
#         self._in_flight += 1

#     def _release(self):
#         self._in_flight -= 1
#         self._sem.release()

#     async def _bump_capacity(self, healthy: bool):
#         if healthy: 
#             self._aimd.ok_tick()
#         else:
#             self._aimd.bad_tick()
#         # smoothly steer semaphore towards AIMD target
#         delta = self._aimd.current - (self._sem._value + self._in_flight)
#         if delta > 0:
#             for _ in range(delta):
#                 self._sem.release()  # increase capacity
#         elif delta < 0:
#             # can't reduce below in-flight; semaphore will naturally gate next requests
#             pass

#     def _update_server_metrics(self, d: Dict[str, Any]):
#         self._last_server = TTSServerMetrics(
#             gpu_util=d.get("gpu_util"),
#             vram_used_mb=d.get("vram_used_mb"),
#             vram_total_mb=d.get("vram_total_mb"),
#             streams_active=d.get("streams_active"),
#             queue_depth=d.get("queue_depth"),
#             server_concurrency=d.get("server_concurrency"),
#             extras={k:v for k,v in d.items() if k not in {
#                 "gpu_util","vram_used_mb","vram_total_mb","streams_active","queue_depth","server_concurrency"
#             }},
#         )

#     async def generate(self, req: TTSRequest) -> AsyncIterator[TTSChunk]:
#         """
#         Returns an async iterator of TTSChunk(audio=bytes).
#         Records timings + server metrics as the stream progresses.
#         """
#         await self._acquire()
#         timings = RequestTimings(t_start=time.perf_counter())
#         healthy = True
#         url = f"{self.base_url}/v1/tts:stream"
#         headers = {"Accept": "application/x-ndjson", **self.headers}
#         payload = {
#             "text": req.text,
#             "voice": req.voice,
#             "params": req.params or {},
#             "request_id": req.request_id or f"tts-{uuid.uuid4()}",
#         }

#         try:
#             async with self.client.stream("POST", url, headers=headers, json=payload) as r:
#                 r.raise_for_status()
#                 timings.t_connect = time.perf_counter()

#                 seq = 0
#                 async for line in r.aiter_lines():
#                     if not line: 
#                         continue
#                     try:
#                         obj = json.loads(line)
#                     except json.JSONDecodeError:
#                         continue

#                     typ = obj.get("type")
#                     if typ == "chunk":
#                         if timings.t_first_byte is None:
#                             timings.t_first_byte = time.perf_counter()
#                             ttfb = timings.ttfb_ms()
#                             if ttfb is not None:
#                                 self._ema_ttfb.add(ttfb)
#                         b = base64.b64decode(obj["b64"])
#                         timings.bytes_rcv += len(b)
#                         timings.chunks += 1
#                         yield TTSChunk(audio=b, t_sent=time.perf_counter(), seq=seq)
#                         seq += 1
#                     elif typ == "metrics":
#                         self._update_server_metrics(obj)
#                     elif typ == "eos":
#                         break

#                 timings.t_end = time.perf_counter()
#                 total_ms = timings.total_ms()
#                 if total_ms:
#                     self._lat_stats.add(total_ms)

#                 self._breaker.record_success()
#                 await self._bump_capacity(healthy=True)

#         except Exception:
#             healthy = False
#             self._breaker.record_failure()
#             await self._bump_capacity(healthy=False)
#             raise
#         finally:
#             self._release()

#     async def system_snapshot(self) -> TTSServerMetrics:
#         try:
#             r = await self.client.get(f"{self.base_url}/v1/system", headers=self.headers, timeout=5.0)
#             if r.status_code == 200:
#                 self._update_server_metrics(r.json())
#         except Exception:
#             pass
#         return self._last_server

#     def snapshot(self) -> EndpointSnapshot:
#         return EndpointSnapshot(
#             base_url=self.base_url,
#             in_flight=self._in_flight,
#             max_concurrency=self.max_concurrency,
#             ema_ttfb_ms=self._ema_ttfb.value,
#             p50_ms=self._lat_stats.p(0.50),
#             p95_ms=self._lat_stats.p(0.95),
#             gpu_util=self._last_server.gpu_util,
#             vram_used_mb=self._last_server.vram_used_mb,
#             vram_total_mb=self._last_server.vram_total_mb,
#             streams_active=self._last_server.streams_active,
#             queue_depth=self._last_server.queue_depth,
#             breaker_open=not self._breaker.allow(),
#         )

#     async def aclose(self):
#         await self.client.aclose()


from __future__ import annotations
import asyncio, base64, json, uuid, time
from typing import AsyncIterator, Optional, Dict, Any
import httpx

from .types import TTSRequest, TTSChunk, TTSServerMetrics, RequestTimings, RollingStats, EndpointSnapshot
from .metrics import EMA, CircuitBreaker, AIMDConcurrency, CapLimiter



class AsyncTTSClient:
    """
    Async client for a *single* TTS endpoint (one GPU box).
    Streams NDJSON frames and yields raw PCM chunks.
    Tracks GPU/VRAM, p50/p95, TTFB, in-flight, and adapts concurrency.
    """
    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        *,
        connect_timeout=5.0,
        read_timeout=60.0,
        http2=True,
        init_concurrency=8,
        max_concurrency=64,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.client = httpx.AsyncClient(
    http2=http2,
    timeout=httpx.Timeout(
        connect=connect_timeout,
        read=read_timeout,
        write=read_timeout,
        pool=connect_timeout,
    ),
)
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._sem = asyncio.Semaphore(init_concurrency)
        self._cap = CapLimiter(init_concurrency, max_concurrency)
        self._aimd = AIMDConcurrency(init=init_concurrency, max_cap=max_concurrency)
        self._breaker = CircuitBreaker()
        self._in_flight = 0

        # Observability
        self._ema_ttfb = EMA(0.2)
        self._lat_stats = RollingStats(window=512)
        self._last_server = TTSServerMetrics()

    @property
    def in_flight(self): return self._in_flight
    
    @property
    def max_concurrency(self):  return self._cap.target
    

    async def _acquire(self):
        if not self._breaker.allow():
            raise RuntimeError("Circuit open for endpoint")
        await self._sem.acquire()
        self._in_flight += 1

    def _release(self):
        self._in_flight -= 1
        self._sem.release()

    async def _bump_capacity(self, healthy: bool):
        if healthy: self._aimd.ok_tick()
        else: self._aimd.bad_tick()
        self._cap.set_target(self._aimd.current)
        desired = max(self._in_flight, self._cap.target)
        extra = desired - (self._sem._value + self._in_flight)
        if extra > 0:
            for _ in range(extra):
                self._sem.release()

    def _update_server_metrics(self, d: Dict[str, Any]):
        self._last_server = TTSServerMetrics(
            gpu_util=d.get("gpu_util"),
            vram_used_mb=d.get("vram_used_mb"),
            vram_total_mb=d.get("vram_total_mb"),
            streams_active=d.get("streams_active"),
            queue_depth=d.get("queue_depth"),
            server_concurrency=d.get("server_concurrency"),
            extras={k:v for k,v in d.items() if k not in {
                "gpu_util","vram_used_mb","vram_total_mb","streams_active","queue_depth","server_concurrency"
            }},
        )

    async def generate(self, req: TTSRequest) -> AsyncIterator[TTSChunk]:
        """
        Returns an async iterator of TTSChunk(audio=bytes).
        Records timings + server metrics as the stream progresses.
        """
        await self._acquire()
        timings = RequestTimings(t_start=time.perf_counter())
        deadline = timings.t_start + (req.timeout_s or 60.0)
        healthy = True
        url = f"{self.base_url}/v1/tts:stream"
        
        headers = {"Accept": "application/x-ndjson",
            "User-Agent": "orpheus-tts-python/1.0",
            "X-Request-ID": req.request_id,
            **self.headers}
        payload = {
            "text": req.text,
            "voice": req.voice,
            "params": req.params or {},
            "request_id": req.request_id or f"tts-{uuid.uuid4()}",
        }

        try:
            async with self.client.stream("POST", url, headers=headers, json=payload) as r:
                r.raise_for_status()
                timings.t_connect = time.perf_counter()

                seq = 0
                async for line in r.aiter_lines():
                    if not line: 
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    typ = obj.get("type")
                    if typ == "chunk":
                        if timings.t_first_byte is None:
                            timings.t_first_byte = time.perf_counter()
                            ttfb = timings.ttfb_ms()
                            if ttfb is not None:
                                self._ema_ttfb.add(ttfb)
                        b = base64.b64decode(obj["b64"])
                        timings.bytes_rcv += len(b)
                        timings.chunks += 1
                        yield TTSChunk(audio=b, t_sent=time.perf_counter(), seq=seq)
                        seq += 1
                    elif typ == "metrics":
                        self._update_server_metrics(obj)
                    elif typ == "eos":
                        break
                    
                    if time.perf_counter() > deadline:
                        raise asyncio.TimeoutError("tts request deadline exceeded")

                timings.t_end = time.perf_counter()
                total_ms = timings.total_ms()
                if total_ms:
                    self._lat_stats.add(total_ms)

                self._breaker.record_success()
                await self._bump_capacity(healthy=True)

        except (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError) as e:
            # lightweight retry once if no bytes were received
            if timings.chunks == 0:
                await self._bump_capacity(healthy=False)
                # one retry path: (optionally jittered)
                async for item in self.generate(req):
                    yield item
                return
            raise
        finally:
            self._release()

    async def system_snapshot(self) -> TTSServerMetrics:
        try:
            r = await self.client.get(f"{self.base_url}/v1/system", headers=self.headers, timeout=5.0)
            if r.status_code == 200:
                data = r.json()
                if "endpoints" in data:  # gateway style
                    # pick first endpoint or the one matching base_url
                    payload = (data["endpoints"][0] or {}) if data["endpoints"] else {}
                else:  # worker style
                    payload = data
                self._update_server_metrics(payload)
        except Exception:
            pass
        return self._last_server

    def snapshot(self) -> EndpointSnapshot:
        return EndpointSnapshot(
            base_url=self.base_url,
            in_flight=self._in_flight,
            max_concurrency=self.max_concurrency,
            ema_ttfb_ms=self._ema_ttfb.value,
            p50_ms=self._lat_stats.p(0.50),
            p95_ms=self._lat_stats.p(0.95),
            gpu_util=self._last_server.gpu_util,
            vram_used_mb=self._last_server.vram_used_mb,
            vram_total_mb=self._last_server.vram_total_mb,
            streams_active=self._last_server.streams_active,
            queue_depth=self._last_server.queue_depth,
            breaker_open=not self._breaker.allow(),
        )

    async def aclose(self):
        await self.client.aclose()
