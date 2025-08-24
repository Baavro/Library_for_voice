# add import
from fastapi import APIRouter, Request
from fastapi.responses import PlainTextResponse, JSONResponse
from tts_service.config import Config
from tts_sdk.prom import prom_text

router = APIRouter()

@router.get("/healthz")
async def health():
    return {"ok": True}

@router.get("/v1/system")
async def system_state(request: Request):
    eps = request.app.state.endpoints
    snaps = [ep.snapshot().__dict__ for ep in eps]
    return {"endpoints": snaps, "config": {
        "inbound_max_concurrency": Config.INBOUND_MAX_CONCURRENCY,
        "hedge_ttfb_ms": Config.HEDGE_TTFB_MS,
        "rate_limit_rpm": Config.RATE_LIMIT_RPM,
    }}

@router.get("/metrics", response_class=PlainTextResponse)
async def metrics(request: Request):
    eps = request.app.state.endpoints
    return prom_text(eps)
