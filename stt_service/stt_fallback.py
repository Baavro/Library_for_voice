# from __future__ import annotations
# import os, io, time
# from typing import Optional
# from pathlib import Path
# from fastapi import FastAPI, File, UploadFile, Form
# from fastapi.responses import JSONResponse
# from faster_whisper import WhisperModel

# CT2_MEDIUM_PATH = os.getenv("CT2_MEDIUM_PATH", "/home/ubuntu/Whisper/model/ct2_model")
# DEVICE          = os.getenv("DEVICE", "cuda")
# DEVICE_INDEX    = int(os.getenv("DEVICE_INDEX", "0"))
# COMPUTE_TYPE    = os.getenv("COMPUTE_TYPE", "int8")
# CPU_THREADS     = int(os.getenv("CPU_THREADS", "4"))

# def _resolve_dir(name: str, p: str) -> str:
#     path = Path(p).expanduser()
#     if not path.is_dir():
#         raise FileNotFoundError(f"{name} path does not exist: {path}")
#     return str(path.resolve())

# CT2_MEDIUM_PATH = _resolve_dir("CT2_MEDIUM_PATH", CT2_MEDIUM_PATH)

# print("—🚀 Loading FasterWhisper (medium)…", CT2_MEDIUM_PATH)
# model = WhisperModel(
#     CT2_MEDIUM_PATH,
#     device=DEVICE, device_index=DEVICE_INDEX,
#     compute_type=COMPUTE_TYPE, cpu_threads=CPU_THREADS
# )
# print("—✅ Medium model loaded.")

# app = FastAPI(title="STT Fallback (medium-only)", version="1.0.0")

# @app.post("/v1/transcribe")
# async def transcribe(
#     audio: UploadFile = File(...),
#     language: Optional[str] = Form(default=None),
#     beam_size: int = Form(default=1),
#     vad_filter: bool = Form(default=True),
#     word_timestamps: bool = Form(default=False),
# ):
#     t0 = time.perf_counter()
#     audio_bytes = await audio.read()
#     segs, info = model.transcribe(
#         audio=io.BytesIO(audio_bytes),
#         beam_size=int(beam_size),
#         language=language,
#         vad_filter=bool(vad_filter),
#         word_timestamps=bool(word_timestamps),
#     )
#     seg_list = list(segs)
#     total_ms = (time.perf_counter() - t0) * 1000.0
#     payload = {
#         "text": " ".join(s.text.strip() for s in seg_list),
#         "segments": [{"start": s.start, "end": s.end, "text": s.text} for s in seg_list],
#         "language": {"code": info.language, "prob": info.language_probability},
#         "timings_ms": {"total": round(total_ms, 2)},
#         "server": {"model": os.path.basename(CT2_MEDIUM_PATH), "device": f"{DEVICE}:{DEVICE_INDEX}"},
#     }
#     return JSONResponse(status_code=200, content=payload,
#                         headers={"X-Total-ms": str(round(total_ms, 2))})

# stt_fallback.py
from __future__ import annotations
import os, io, time, asyncio
from typing import Optional, Dict, Any
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Form, Depends, Header, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from faster_whisper import WhisperModel
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

from .config_loader import load_config

CFG = load_config(
    path=os.getenv("CONFIG_PATH", "stt_fallback.env"),
    schema_defaults={
        "CT2_MEDIUM_PATH": "/home/ubuntu/Server/ct2_model",
        "DEVICE": "cuda",
        "DEVICE_INDEX": 0,
        "COMPUTE_TYPE": "int8",
        "CPU_THREADS": 4,
        "API_KEYS": "",
        "RATE_LIMIT_RPS": 60.0,
        "RATE_LIMIT_BURST": 120,
        "MAX_UPLOAD_MB": 16.0,
    },
    schema_types={
        "CT2_MEDIUM_PATH": "str",
        "DEVICE": "str",
        "DEVICE_INDEX": "int",
        "COMPUTE_TYPE": "str",
        "CPU_THREADS": "int",
        "API_KEYS": "str",
        "RATE_LIMIT_RPS": "float",
        "RATE_LIMIT_BURST": "int",
        "MAX_UPLOAD_MB": "float",
    },
)

CT2_MEDIUM_PATH = CFG["CT2_MEDIUM_PATH"]
DEVICE          = CFG["DEVICE"]
DEVICE_INDEX    = CFG["DEVICE_INDEX"]
COMPUTE_TYPE    = CFG["COMPUTE_TYPE"]
CPU_THREADS     = CFG["CPU_THREADS"]
API_KEYS        = {k.strip() for k in CFG["API_KEYS"].split(",") if k.strip()}
RATE_LIMIT_RPS  = CFG["RATE_LIMIT_RPS"]
RATE_LIMIT_BURST= CFG["RATE_LIMIT_BURST"]
MAX_UPLOAD_MB   = CFG["MAX_UPLOAD_MB"]

def _resolve_dir(name: str, p: str) -> str:
    path = Path(p).expanduser()
    if not path.is_dir():
        raise FileNotFoundError(f"{name} path does not exist: {path}")
    return str(path.resolve())

CT2_MEDIUM_PATH = _resolve_dir("CT2_MEDIUM_PATH", CT2_MEDIUM_PATH)

print("—🚀 Loading FasterWhisper (medium)…", CT2_MEDIUM_PATH)
model = WhisperModel(
    CT2_MEDIUM_PATH,
    device=DEVICE, device_index=DEVICE_INDEX,
    compute_type=COMPUTE_TYPE, cpu_threads=CPU_THREADS
)
print("—✅ Medium model loaded.")

app = FastAPI(title="STT Fallback (medium-only)", version="1.2.0")

REQUESTS_TOTAL = Counter("stt_fb_requests_total", "Total STT fallback requests")
REQ_TOTAL_SEC  = Histogram("stt_fb_request_total_seconds", "Total request time (s)")

async def require_auth(authorization: str = Header(None)) -> Optional[str]:
    if not API_KEYS:
        return None
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = authorization.split(None, 1)[1]
    if token not in API_KEYS:
        raise HTTPException(status_code=403, detail="Invalid token")
    return token

class TokenBucket:
    __slots__ = ("capacity","rate","tokens","last","lock")
    def __init__(self, capacity: int, rate: float):
        self.capacity = capacity
        self.rate = rate
        self.tokens = capacity
        self.last = time.monotonic()
        self.lock = asyncio.Lock()
    async def consume(self, n: int = 1) -> bool:
        async with self.lock:
            now = time.monotonic()
            elapsed = now - self.last
            self.last = now
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            if self.tokens >= n:
                self.tokens -= n
                return True
            return False

_buckets: Dict[str, TokenBucket] = {}
_buckets_lock = asyncio.Lock()

async def admit_request(req: Request, token: Optional[str] = Depends(require_auth)) -> None:
    key = token or (req.client.host if req.client else "anon")
    async with _buckets_lock:
        if key not in _buckets:
            _buckets[key] = TokenBucket(capacity=RATE_LIMIT_BURST, rate=RATE_LIMIT_RPS)
        bucket = _buckets[key]
    if not await bucket.consume(1):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

@app.middleware("http")
async def size_cap_mw(request: Request, call_next):
    if request.method == "POST" and request.url.path == "/v1/transcribe":
        cl = request.headers.get("content-length")
        if cl is not None:
            try:
                if int(cl) > int(MAX_UPLOAD_MB * 1024 * 1024):
                    return JSONResponse(status_code=413, content={"error": {"type":"PayloadTooLarge","message": "File too large"}})
            except Exception:
                pass
    return await call_next(request)

@app.get("/metrics")
def metrics_prom():
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/v1/health")
def health():
    return {"ok": True, "role": "fallback", "device": f"{DEVICE}:{DEVICE_INDEX}", "model": os.path.basename(CT2_MEDIUM_PATH)}

@app.post("/v1/transcribe")
async def transcribe(
    _=Depends(admit_request),
    audio: UploadFile = File(...),
    language: Optional[str] = Form(default=None),
    beam_size: int = Form(default=1),
    vad_filter: bool = Form(default=True),
    word_timestamps: bool = Form(default=False),
):
    REQUESTS_TOTAL.inc()
    t0 = time.perf_counter()
    try:
        audio_bytes = await audio.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid audio: {e}")

    language = language or None  # "" -> autodetect

    segs, info = model.transcribe(
        audio=io.BytesIO(audio_bytes),
        beam_size=int(beam_size),
        language=language,
        vad_filter=bool(vad_filter),
        word_timestamps=bool(word_timestamps),
    )
    seg_list = list(segs)
    total_ms = (time.perf_counter() - t0) * 1000.0
    REQ_TOTAL_SEC.observe(total_ms / 1000.0)

    payload = {
        "text": " ".join(s.text.strip() for s in seg_list),
        "segments": [{"start": s.start, "end": s.end, "text": s.text} for s in seg_list],
        "language": {"code": info.language, "prob": info.language_probability, "source": ("forced" if language else "auto")},
        "timings_ms": {"total": round(total_ms, 2)},
        "server": {"model": os.path.basename(CT2_MEDIUM_PATH), "device": f"{DEVICE}:{DEVICE_INDEX}"},
    }
    return JSONResponse(status_code=200, content=payload,
                        headers={"X-Total-ms": str(round(total_ms, 2))})
