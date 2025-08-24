from __future__ import annotations
import asyncio, random
from typing import AsyncIterator, List, Optional
from .types import TTSRequest, TTSChunk, EndpointSnapshot
from .client import AsyncTTSClient

class TTSCluster:
    """
    Multi-endpoint router:
    - Weighted round robin (weights derive from p95 + queue depth + breaker state)
    - Optional hedged requests: start a 2nd request if TTFB busts a percentile
    """
    def __init__(self, endpoints: List[AsyncTTSClient], *, hedge_ttfb_ms: Optional[float] = None):
        self.endpoints = endpoints
        self.hedge_ttfb_ms = hedge_ttfb_ms

    def _score(self, snap: EndpointSnapshot) -> float:
        # Lower is better
        p95 = snap.p95_ms or 500.0
        qd = snap.queue_depth or 0
        pen_breaker = 5_000 if snap.breaker_open else 0
        return p95 + 50 * qd + pen_breaker

    def _pick(self) -> AsyncTTSClient:
        snaps = [e.snapshot() for e in self.endpoints]
        scores = [self._score(s) for s in snaps]
        # softmax-ish inverse scoring
        inv = [1.0 / (s + 1e-6) for s in scores]
        total = sum(inv) or 1.0
        r = random.random() * total
        acc = 0.0
        for ep, w in zip(self.endpoints, inv):
            acc += w
            if r <= acc:
                return ep
        return self.endpoints[0]

    async def generate(self, req: TTSRequest) -> AsyncIterator[TTSChunk]:
        primary, backup = self._pick(), self._pick()
        if not self.hedge_ttfb_ms or primary is backup:
            async for c in primary.generate(req):
                yield c
            return

        q1, q2 = asyncio.Queue(), asyncio.Queue()
        first_chunk_sent = asyncio.Event()

        async def run(ep, out_q):
            try:
                async for c in ep.generate(req):
                    # if another stream already won, stop early
                    if first_chunk_sent.is_set():
                        break
                    await out_q.put(("chunk", c))
                    # mark winner on first chunk
                    first_chunk_sent.set()
                await out_q.put(("done", None))
            except Exception as e:
                await out_q.put(("error", e))

        t1 = asyncio.create_task(run(primary, q1))
        t_backup = None

        async def watchdog():
            await asyncio.sleep(self.hedge_ttfb_ms / 1000.0)
            if not first_chunk_sent.is_set():
                return asyncio.create_task(run(backup, q2))
            return None

        t_backup = await watchdog()

        try:
            done = 0
            while done < (2 if t_backup else 1):
                futs = {asyncio.create_task(q1.get())}
                if t_backup: futs.add(asyncio.create_task(q2.get()))
                done_set, _ = await asyncio.wait(futs, return_when=asyncio.FIRST_COMPLETED)
                for f in done_set:
                    typ, payload = f.result()
                    if typ == "chunk":
                        yield payload
                    else:
                        done += 1
                # if first chunk already sent, cancel the other task
                if first_chunk_sent.is_set() and t_backup:
                    if t1 and not t1.done(): t1.cancel()
                    if t_backup and not t_backup.done(): t_backup.cancel()
                    t_backup = None  # ensure loop exits once primary finishes
        finally:
            for t in [t1, t_backup]:
                if t: t.cancel()