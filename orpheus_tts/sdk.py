from __future__ import annotations
from typing import AsyncIterator, Optional, List
from .client import AsyncTTSClient
from .cluster import TTSCluster
from .types import TTSRequest, TTSChunk

class OrpheusTTS:
    """
    Facade for consuming a *gateway* (or a single worker) as a client.
    Pass one URL for single endpoint or multiple URLs for a cluster-aware client.
    """
    def __init__(self, base_urls: List[str], api_key: Optional[str] = None, hedge_ttfb_ms: Optional[float] = 400):
        self._endpoints = [AsyncTTSClient(u, api_key=api_key, init_concurrency=12, max_concurrency=96) for u in base_urls]
        self._cluster = TTSCluster(self._endpoints, hedge_ttfb_ms=hedge_ttfb_ms)

    async def stream(self, text: str, voice: Optional[str] = None, **params) -> AsyncIterator[bytes]:
        req = TTSRequest(text=text, voice=voice, params=params)
        async for chunk in self._cluster.generate(req):
            yield chunk.audio

    async def close(self):
        for ep in self._endpoints:
            await ep.aclose()
