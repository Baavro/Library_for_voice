from __future__ import annotations
import asyncio, base64, json, time
from typing import AsyncIterator
from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.responses import StreamingResponse

from orpheus_tts.types import TTSRequest
from ..auth import require_api_key
from ..config import Config
from ..rate_limiter import TokenBucket

router = APIRouter()
bucket = TokenBucket(Config.RATE_LIMIT_RPM)

def _ndjson(obj: dict) -> bytes:
    return (json.dumps(obj, separators=(",", ":")) + "\n").encode("utf-8")

def _metrics_frame(snapshot) -> dict:
    return {
        "type": "metrics",
        "gpu_util": snapshot.gpu_util,
        "vram_used_mb": snapshot.vram_used_mb,
        "vram_total_mb": snapshot.vram_total_mb,
        "streams_active": snapshot.streams_active,
        "queue_depth": snapshot.queue_depth,
        "server_concurrency": snapshot.max_concurrency,
    }

def make_streamer(app_state, req: TTSRequest) -> AsyncIterator[bytes]:
    """
    Bridges cluster.generate(req) -> NDJSON frames.
    Injects periodic 'metrics' frames from downstream snapshot even if no chunk arrives yet.
    """
    cluster = app_state.cluster
    interval = Config.STREAM_METRICS_INTERVAL_MS / 1000.0

    async def gen():
        if not bucket.allow(req.request_id or "global"):  # coarse limiting; you can key by api key too
            yield _ndjson({"type": "error", "message": "rate_limited"})
            return

        # spawn downstream puller
        q = asyncio.Queue()

        async def pull():
            try:
                async for c in cluster.generate(req):
                    await q.put(("chunk", c))
                await q.put(("done", None))
            except Exception as e:
                await q.put(("error", str(e)))

        task = asyncio.create_task(pull())
        deadline = time.time() + req.timeout_s
        last_metrics = 0.0
        first_sent = False

        # send initial metrics snapshot
        snap = cluster.endpoints[0].snapshot() if cluster.endpoints else None
        if snap:
            yield _ndjson(_metrics_frame(snap))

        try:
            while True:
                if time.time() > deadline:
                    yield _ndjson({"type":"error","message":"deadline_exceeded"})
                    break
                try:
                    item = await asyncio.wait_for(q.get(), timeout=interval)
                except asyncio.TimeoutError:
                    # periodic metrics
                    snap = cluster.endpoints[0].snapshot() if cluster.endpoints else None
                    if snap:
                        yield _ndjson(_metrics_frame(snap))
                    continue

                typ, payload = item
                if typ == "chunk":
                    if not first_sent:
                        first_sent = True
                    b64 = base64.b64encode(payload.audio).decode("ascii")
                    yield _ndjson({"type": "chunk", "b64": b64, "ts": time.time()})
                elif typ == "done":
                    yield _ndjson({"type": "eos"})
                    break
                else:  # error
                    yield _ndjson({"type": "error", "message": payload})
                    break
        finally:
            task.cancel()

    return gen()

# @router.post("/v1/tts:stream")
# async def tts_stream(body: dict, request: Request, api_key=Depends(require_api_key)):
#     text = (body.get("text") or "").strip()
#     if not text:
#         raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="text required")
#     req = TTSRequest(
#         text=text,
#         voice=body.get("voice"),
#         params=body.get("params") or {},
#         request_id=body.get("request_id"),
#         timeout_s=float(body.get("timeout_s") or 60.0),
#     )
#     gen = make_streamer(request.app.state, req)
#     return StreamingResponse(gen, media_type="application/x-ndjson")

@router.post("/v1/tts:stream")
async def tts_stream(body: dict, request: Request, api_key=Depends(require_api_key)):
    sem: asyncio.Semaphore = request.app.state.inbound_sem
    await sem.acquire()
    text = (body.get("text") or "").strip()
    if not text:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="text required")
    try:
        if not bucket.allow(api_key):  # per API key
            return StreamingResponse(iter([_ndjson({"type":"error","message":"rate_limited"})]), media_type="application/x-ndjson")
        
        req = TTSRequest(
            text=text,
            voice=body.get("voice"),
            params=body.get("params") or {},
            request_id=body.get("request_id"),
            timeout_s=float(body.get("timeout_s") or 60.0),
        )
        
        gen = make_streamer(request.app.state, req)
        return StreamingResponse(gen, media_type="application/x-ndjson")
    except: 
        return StreamingResponse(iter([_ndjson({"type":"error","message":"uknown"})]), media_type="application/x-ndjson")
    finally:
        sem.release()