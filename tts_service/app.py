from __future__ import annotations
import asyncio
from fastapi import FastAPI
from .config import Config
from tts_sdk.client import AsyncTTSClient
from tts_sdk.cluster import TTSCluster
from .routers import tts as tts_router, admin as admin_router

def build_app() -> FastAPI:
    app = FastAPI(title="Orpheus TTS Gateway", version="1.0.0")
    app.state.inbound_sem = asyncio.Semaphore(Config.INBOUND_MAX_CONCURRENCY)
    @app.on_event("startup")
    async def _startup():
        # Build downstream endpoints
        endpoints = [
            AsyncTTSClient(u, api_key=Config.DOWNSTREAM_API_KEY, init_concurrency=12, max_concurrency=96)
            for u in Config.DOWNSTREAM_URLS
        ]
        cluster = TTSCluster(endpoints, hedge_ttfb_ms=Config.HEDGE_TTFB_MS)

        app.state.endpoints = endpoints
        app.state.cluster = cluster

    @app.on_event("shutdown")
    async def _shutdown():
        tasks = [ep.aclose() for ep in app.state.endpoints]
        await asyncio.gather(*tasks, return_exceptions=True)

    app.include_router(tts_router.router)
    app.include_router(admin_router.router)
    return app

app = build_app()
