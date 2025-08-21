# from __future__ import annotations
# import os, time, asyncio, uuid, io, math
# from typing import List, Optional, Dict, Any
# from concurrent.futures import ThreadPoolExecutor
# from pathlib import Path

# import httpx
# from fastapi import FastAPI, File, UploadFile, Form
# from fastapi.responses import JSONResponse
# from pydantic import BaseModel
# from faster_whisper import WhisperModel
# import pynvml

# # ── Config ───────────────────────────────────────────────────────────
# CT2_SMALL_PATH   = os.getenv("CT2_SMALL_PATH", "/home/ubuntu/Server/whisper-small-en-ct2-fp16")
# DEVICE           = os.getenv("DEVICE", "cuda")
# DEVICE_INDEX     = int(os.getenv("DEVICE_INDEX", "0"))
# COMPUTE_TYPE     = os.getenv("COMPUTE_TYPE", "float16")
# CPU_THREADS      = int(os.getenv("CPU_THREADS", "4"))

# # concurrency per process
# MAX_CONCURRENCY  = int(os.getenv("MAX_CONCURRENCY", "2"))
# SERVER_COMMIT    = os.getenv("SERVER_COMMIT", "local-dev")

# # confidence routing thresholds
# CONF_LOGPROB_THRESHOLD  = float(os.getenv("CONF_LOGPROB_THRESHOLD", "0.55"))
# CONF_LANGPROB_THRESHOLD = float(os.getenv("CONF_LANGPROB_THRESHOLD", "0.70"))
# MIN_WORDS_FOR_CONF      = int(os.getenv("MIN_WORDS_FOR_CONF", "3"))
# DEFAULT_LANGUAGE        = os.getenv("DEFAULT_LANGUAGE", "en")

# # remote fallback target
# FALLBACK_URL      = os.getenv("FALLBACK_URL", "http://127.0.0.1:8082")
# FALLBACK_API_KEY  = os.getenv("FALLBACK_API_KEY", "")


# def _resolve_dir(name: str, p: str) -> str:
#     path = Path(p).expanduser()
#     if not path.is_dir():
#         raise FileNotFoundError(f"{name} path does not exist: {path}")
#     return str(path.resolve())

# CT2_SMALL_PATH = _resolve_dir("CT2_SMALL_PATH", CT2_SMALL_PATH)

# # ── Init models ──────────────────────────────────────────────────────
# print("—🚀 Loading FasterWhisper (small)…", CT2_SMALL_PATH)
# small_model = WhisperModel(
#     CT2_SMALL_PATH,
#     device=DEVICE,
#     device_index=DEVICE_INDEX,
#     compute_type=COMPUTE_TYPE,
#     cpu_threads=CPU_THREADS,
# )
# print("—✅ Small model loaded.")

# # NVML (safe)
# try:
#     pynvml.nvmlInit()
#     nv_dev = pynvml.nvmlDeviceGetHandleByIndex(DEVICE_INDEX)
#     NVML_OK = True
# except Exception as _e:
#     nv_dev = None
#     NVML_OK = False
#     print(f"—⚠️ NVML unavailable: {_e}")

# # Concurrency
# sem = asyncio.Semaphore(MAX_CONCURRENCY)
# state_lock = asyncio.Lock()
# waiting = 0
# in_flight = 0
# executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENCY)

# app = FastAPI(title="STT Primary (small → remote medium)", version="2.1.0")

# # ── Schemas ──────────────────────────────────────────────────────────
# class SegmentOut(BaseModel):
#     start: float
#     end: float
#     text: str
#     avg_logprob: Optional[float] = None
#     words: Optional[int] = None

# class STTResponse(BaseModel):
#     text: str
#     segments: List[SegmentOut]
#     language: dict
#     timings_ms: dict
#     server: dict
#     gpu: dict
#     queue_depth: int
#     confidence: dict
#     routing: dict

# # ── Helpers ──────────────────────────────────────────────────────────
# def gpu_stats() -> Dict[str, Any]:
#     if not NVML_OK:
#         return {"name": "unknown", "index": DEVICE_INDEX, "memory_used_mb": 0, "memory_total_mb": 0,
#                 "utilization_pct": 0, "memory_util_pct": 0, "temperature_c": 0}
#     try:
#         mem = pynvml.nvmlDeviceGetMemoryInfo(nv_dev)
#         util = pynvml.nvmlDeviceGetUtilizationRates(nv_dev)
#         temp = pynvml.nvmlDeviceGetTemperature(nv_dev, pynvml.NVML_TEMPERATURE_GPU)
#         raw  = pynvml.nvmlDeviceGetName(nv_dev)
#         name = raw.decode() if isinstance(raw, (bytes, bytearray)) else str(raw)
#         return {
#             "name": name, "index": DEVICE_INDEX,
#             "memory_used_mb": round(mem.used / (1024**2)),
#             "memory_total_mb": round(mem.total / (1024**2)),
#             "utilization_pct": int(util.gpu),
#             "memory_util_pct": int(100 * mem.used / mem.total),
#             "temperature_c": int(temp),
#         }
#     except Exception as e:
#         return {"name": "unknown", "index": DEVICE_INDEX, "memory_used_mb": 0, "memory_total_mb": 0,
#                 "utilization_pct": 0, "memory_util_pct": 0, "temperature_c": 0, "error": str(e)}

# def aggregate_text(segments) -> str:
#     return " ".join(s.text.strip() for s in segments)

# def to_bool(v) -> bool:
#     if isinstance(v, bool): return v
#     if isinstance(v, (int, float)): return bool(v)
#     if isinstance(v, str): return v.strip().lower() in ("1","true","yes","y","on")
#     return False

# def _wordish_weight(text: str) -> int:
#     return max(1, len((text or "").strip().split()))

# def _logprob_confidence(seg_list) -> Dict[str, Any]:
#     pairs = []
#     for s in seg_list:
#         lp = getattr(s, "avg_logprob", None)
#         if lp is None: continue
#         w = _wordish_weight(getattr(s, "text", "") or "")
#         pairs.append((float(lp), w))
#     if not pairs:
#         return {"logprob_avg_weighted": None, "logprob_min": None, "logprob_max": None,
#                 "word_weight": 0, "logprob_confidence": 0.0, "method": "exp(weighted_avg_logprob)"}
#     total_w = sum(w for _, w in pairs)
#     avg_lp = sum(lp*w for lp, w in pairs) / max(1, total_w)
#     clamped = max(-5.0, min(0.0, avg_lp))
#     conf = math.exp(clamped)
#     lp_vals = [lp for lp, _ in pairs]
#     return {
#         "logprob_avg_weighted": round(avg_lp, 4),
#         "logprob_min": round(min(lp_vals), 4),
#         "logprob_max": round(max(lp_vals), 4),
#         "word_weight": int(total_w),
#         "logprob_confidence": round(conf, 4),
#         "method": "exp(weighted_avg_logprob)",
#     }

# def _build_segments(seg_list) -> List[SegmentOut]:
#     out: List[SegmentOut] = []
#     for s in seg_list:
#         out.append(SegmentOut(
#             start=s.start, end=s.end, text=s.text,
#             avg_logprob=getattr(s, "avg_logprob", None),
#             words=_wordish_weight(getattr(s, "text", "") or "")
#         ))
#     return out

# def _transcribe_bytes(model: WhisperModel, audio_bytes: bytes, beam_size: int, language: Optional[str],
#                       vad_filter: bool, word_timestamps: bool):
#     audio_io = io.BytesIO(audio_bytes)
#     segs, info = model.transcribe(
#         audio=audio_io, beam_size=beam_size, language=language,
#         vad_filter=vad_filter, word_timestamps=word_timestamps
#     )
#     return list(segs), info

# # ── Endpoints ────────────────────────────────────────────────────────
# @app.get("/v1/health")
# def health():
#     return {
#         "ok": True,
#         "role": "primary",
#         "device": DEVICE, "device_index": DEVICE_INDEX, "compute_type": COMPUTE_TYPE,
#         "small": CT2_SMALL_PATH,
#         "fallback_url": FALLBACK_URL,
#         "max_conc": MAX_CONCURRENCY,
#         "thresholds": {"logprob": CONF_LOGPROB_THRESHOLD, "language_prob": CONF_LANGPROB_THRESHOLD, "min_words": MIN_WORDS_FOR_CONF},
#         "nvml": NVML_OK, "commit": SERVER_COMMIT,
#     }

# @app.get("/v1/metrics")
# def metrics():
#     gs = gpu_stats()
#     return {"gpu": gs, "queue_depth": waiting, "in_flight": in_flight,
#             "max_concurrency": MAX_CONCURRENCY, "server_commit": SERVER_COMMIT}

# @app.post("/v1/transcribe")
# async def transcribe(
#     audio: UploadFile = File(...),
#     language: Optional[str] = Form(default=None),
#     beam_size: int = Form(default=1),
#     vad_filter: bool | str = Form(default=True),
#     word_timestamps: bool | str = Form(default=False),
#     conf_threshold: Optional[float] = Form(default=None),
#     langprob_threshold: Optional[float] = Form(default=None),
#     min_words_for_conf: Optional[int] = Form(default=None),
# ):
#     global waiting, in_flight
#     req_id = str(uuid.uuid4())[:8]
#     t0 = time.perf_counter()

#     # waiting
#     async with state_lock: waiting += 1
#     q_enter = t0

#     # acquire GPU slot for primary only
#     await sem.acquire()
#     async with state_lock:
#         waiting -= 1
#         in_flight += 1

#     # read bytes once
#     audio_bytes = await audio.read()

#     try:
#         q_leave = time.perf_counter()
#         queue_wait_ms = (q_leave - q_enter) * 1000.0

#         # primary (small)
#         loop = asyncio.get_running_loop()
#         t_p0 = time.perf_counter()
#         small_segs, small_info = await loop.run_in_executor(
#             executor, _transcribe_bytes, small_model, audio_bytes,
#             int(beam_size), language or DEFAULT_LANGUAGE or None,
#             to_bool(vad_filter), to_bool(word_timestamps),
#         )
#         t_p1 = time.perf_counter()
#         primary_ms = (t_p1 - t_p0) * 1000.0

#     finally:
#         # release local GPU slot before remote fallback
#         async with state_lock:
#             in_flight = max(0, in_flight - 1)
#         sem.release()

#     # confidence decision
#     conf_meta = _logprob_confidence(small_segs)
#     conf = conf_meta["logprob_confidence"] or 0.0
#     lp = float(small_info.language_probability or 0.0)
#     threshold = float(conf_threshold) if conf_threshold is not None else CONF_LOGPROB_THRESHOLD
#     lang_threshold = float(langprob_threshold) if langprob_threshold is not None else CONF_LANGPROB_THRESHOLD
#     min_words = int(min_words_for_conf) if min_words_for_conf is not None else MIN_WORDS_FOR_CONF
#     too_short = sum(_wordish_weight(getattr(s, "text", "") or "") for s in small_segs) < max(1, min_words)
#     need_fallback = (conf < threshold) or (lp < lang_threshold) or too_short

#     fallback_ms = 0.0
#     final_model = "small"
#     final_segs, final_info = small_segs, small_info
#     fb_err: Optional[str] = None

#     if need_fallback:
#         # remote fallback call
#         t_f0 = time.perf_counter()
#         try:
#             headers = {"Authorization": f"Bearer {FALLBACK_API_KEY}"} if FALLBACK_API_KEY else {}
#             data = {
#                 "language": language or DEFAULT_LANGUAGE or "",
#                 "beam_size": 1,                # keep greedy for speed; tune if you want accuracy
#                 "vad_filter": "true",
#                 "word_timestamps": "false",
#             }
#             files = {"audio": ("audio.wav", io.BytesIO(audio_bytes), "audio/wav")}
#             async with httpx.AsyncClient(base_url=FALLBACK_URL, timeout=60.0, headers=headers) as hc:
#                 # r = await fallback_client.post("/v1/transcribe", files=files, data=data)
#                 r = await hc.post("/v1/transcribe", files=files, data=data)
#                 r.raise_for_status()
#                 fb = r.json()
#                 # prefer fallback text/segments/lang
#                 final_model = "medium"
#                 final_segs = fb.get("segments", [])
#                 final_info = fb.get("language", {})
#                 # backfill segments into SegmentOut schema if needed
#                 # (we’ll treat them as plain dicts; primary segment extras are only for confidence calc)
#                 fallback_ms_hdr = r.headers.get("X-Total-ms") or r.headers.get("X-Infer-ms")
#                 fallback_ms = float(fallback_ms_hdr) if fallback_ms_hdr else float(fb.get("timings_ms", {}).get("total", 0.0))
#         except Exception as e:
#             fb_err = repr(e)[:200]
#         t_f1 = time.perf_counter()
#         if fallback_ms <= 0.0:
#             fallback_ms = (t_f1 - t_f0) * 1000.0  # wall time if server didn't return timing

#     total_ms = (time.perf_counter() - t0) * 1000.0
#     gs = gpu_stats()
#     q_depth_snapshot = waiting

#     # normalize final segments payload
#     def _as_segout_list(segs):
#         if segs and isinstance(segs[0], dict):
#             # coming from fallback server
#             norm = []
#             for d in segs:
#                 norm.append(SegmentOut(
#                     start=float(d.get("start", 0.0)),
#                     end=float(d.get("end", 0.0)),
#                     text=str(d.get("text", "")),
#                     avg_logprob=d.get("avg_logprob"),
#                     words=_wordish_weight(d.get("text", "")),
#                 ))
#             return norm
#         return _build_segments(segs)

#     seg_out = _as_segout_list(final_segs)

#     # language dict normalize
#     if isinstance(final_info, dict):
#         lang_dict = final_info
#     else:
#         lang_dict = {"code": final_info.language, "prob": final_info.language_probability}

#     payload = STTResponse(
#         text=aggregate_text(seg_out),
#         segments=seg_out,
#         language=lang_dict,
#         timings_ms={
#             "queue_wait": round(queue_wait_ms, 2),
#             "primary_infer": round(primary_ms, 2),
#             "fallback_infer": round(fallback_ms, 2) if need_fallback else 0.0,
#             "total": round(total_ms, 2),
#         },
#         server={
#             "role": "primary",
#             "device": f"{DEVICE}:{DEVICE_INDEX}",
#             "compute_type": COMPUTE_TYPE,
#             "commit": SERVER_COMMIT,
#             "request_id": req_id,
#             "models": {"primary": os.path.basename(CT2_SMALL_PATH), "fallback_remote": FALLBACK_URL},
#         },
#         gpu=gs,
#         queue_depth=max(q_depth_snapshot, 0),
#         confidence={
#             **conf_meta,
#             "language_probability_primary": round(lp, 4),
#             "threshold": round(threshold, 3),
#             "lang_threshold": round(lang_threshold, 3),
#             "below_threshold": conf < threshold,
#             "below_langprob": lp < lang_threshold,
#             "too_short": bool(too_short),
#         },
#         routing={
#             "fallback_used": bool(need_fallback and fb_err is None),
#             "fallback_attempted": bool(need_fallback),
#             "fallback_error": fb_err,
#             "final_model": final_model,
#             "reason": ("below_logprob_threshold" if conf < threshold else
#                        ("below_language_probability" if lp < lang_threshold else
#                         ("too_short" if too_short else "ok"))),
#         },
#     )

#     headers = {
#         "X-Req-Id": req_id,
#         "X-Queue-Wait-ms": str(round(queue_wait_ms, 2)),
#         "X-Primary-Infer-ms": str(round(primary_ms, 2)),
#         "X-Fallback-Infer-ms": str(round(fallback_ms, 2) if need_fallback else 0),
#         "X-Total-ms": str(round(total_ms, 2)),
#         "X-GPU-Util-pct": str(gs.get("utilization_pct", 0)),
#         "X-GPU-Mem-Used-MB": str(gs.get("memory_used_mb", 0)),
#         "X-Queue-Depth": str(q_depth_snapshot),
#         "X-InFlight": str(in_flight),
#         "X-Max-Concurrency": str(MAX_CONCURRENCY),
#         "X-Conf-Logprob-Primary": str(conf),
#         "X-Conf-Threshold": str(threshold),
#         "X-LangProb-Primary": str(lp),
#         "X-LangProb-Threshold": str(lang_threshold),
#         "X-Fallback-Used": "true" if (need_fallback and fb_err is None) else "false",
#         "X-Final-Model": final_model,
#         "X-Fallback-Error": fb_err or "",
#     }
#     return JSONResponse(status_code=200, content=payload.model_dump(), headers=headers)


# stt_primary.py
from __future__ import annotations
import os, time, asyncio, uuid, io, math
from typing import List, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from contextlib import suppress

import httpx
from fastapi import FastAPI, File, UploadFile, Form, Depends, Header, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel
from faster_whisper import WhisperModel
import pynvml
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

from .config_loader import load_config

# ── Load config (.env) ───────────────────────────────────────────────
# Pass path via CLI env when launching uvicorn, e.g.:
#   uvicorn stt_primary:app --port 8081 --workers 1
# and ensure stt_primary.env is in the working dir (or pass CONFIG_PATH)

CFG = load_config(
    path=os.getenv("CONFIG_PATH", "stt_primary.env"),
    schema_defaults={
        "CT2_SMALL_PATH": "/home/ubuntu/Server/Library/stt_service/models/whisper-small-en-ct2-fp16",
        "DEVICE": "cuda",
        "DEVICE_INDEX": 0,
        "COMPUTE_TYPE": "float16",
        "CPU_THREADS": 4,
        "MAX_CONCURRENCY": 2,
        "SERVER_COMMIT": "local-dev",
        "CONF_LOGPROB_THRESHOLD": 0.55,
        "CONF_LANGPROB_THRESHOLD": 0.70,
        "MIN_WORDS_FOR_CONF": 3,
        "FALLBACK_URL": "http://127.0.0.1:8082",
        "FALLBACK_API_KEY": "",
        "API_KEYS": "",  # comma-separated
        "RATE_LIMIT_RPS": 15.0,
        "RATE_LIMIT_BURST": 30,
        "MAX_UPLOAD_MB": 16.0,
    },
    schema_types={
        "CT2_SMALL_PATH": "str",
        "DEVICE": "str",
        "DEVICE_INDEX": "int",
        "COMPUTE_TYPE": "str",
        "CPU_THREADS": "int",
        "MAX_CONCURRENCY": "int",
        "SERVER_COMMIT": "str",
        "CONF_LOGPROB_THRESHOLD": "float",
        "CONF_LANGPROB_THRESHOLD": "float",
        "MIN_WORDS_FOR_CONF": "int",
        "FALLBACK_URL": "str",
        "FALLBACK_API_KEY": "str",
        "API_KEYS": "str",
        "RATE_LIMIT_RPS": "float",
        "RATE_LIMIT_BURST": "int",
        "MAX_UPLOAD_MB": "float",
    },
)

CT2_SMALL_PATH   = CFG["CT2_SMALL_PATH"]
DEVICE           = CFG["DEVICE"]
DEVICE_INDEX     = CFG["DEVICE_INDEX"]
COMPUTE_TYPE     = CFG["COMPUTE_TYPE"]
CPU_THREADS      = CFG["CPU_THREADS"]
MAX_CONCURRENCY  = CFG["MAX_CONCURRENCY"]
SERVER_COMMIT    = CFG["SERVER_COMMIT"]
CONF_LOGPROB_THRESHOLD  = CFG["CONF_LOGPROB_THRESHOLD"]
CONF_LANGPROB_THRESHOLD = CFG["CONF_LANGPROB_THRESHOLD"]
MIN_WORDS_FOR_CONF      = CFG["MIN_WORDS_FOR_CONF"]
FALLBACK_URL      = CFG["FALLBACK_URL"]
FALLBACK_API_KEY  = CFG["FALLBACK_API_KEY"]
API_KEYS          = {k.strip() for k in CFG["API_KEYS"].split(",") if k.strip()}
RATE_LIMIT_RPS    = CFG["RATE_LIMIT_RPS"]
RATE_LIMIT_BURST  = CFG["RATE_LIMIT_BURST"]
MAX_UPLOAD_MB     = CFG["MAX_UPLOAD_MB"]

def _resolve_dir(name: str, p: str) -> str:
    path = Path(p).expanduser()
    if not path.is_dir():
        raise FileNotFoundError(f"{name} path does not exist: {path}")
    return str(path.resolve())

CT2_SMALL_PATH = _resolve_dir("CT2_SMALL_PATH", CT2_SMALL_PATH)

print("—🚀 Loading FasterWhisper (small)…", CT2_SMALL_PATH)
small_model = WhisperModel(
    CT2_SMALL_PATH,
    device=DEVICE,
    device_index=DEVICE_INDEX,
    compute_type=COMPUTE_TYPE,
    cpu_threads=CPU_THREADS,
)
print("—✅ Small model loaded.")

# NVML
try:
    pynvml.nvmlInit()
    nv_dev = pynvml.nvmlDeviceGetHandleByIndex(DEVICE_INDEX)
    NVML_OK = True
except Exception as _e:
    nv_dev = None
    NVML_OK = False
    print(f"—⚠️ NVML unavailable: {_e}")

# Concurrency
sem = asyncio.Semaphore(MAX_CONCURRENCY)
state_lock = asyncio.Lock()
waiting = 0
in_flight = 0
executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENCY)

# Fallback client
_fallback_client: Optional[httpx.AsyncClient] = None

app = FastAPI(title="STT Primary (small → remote medium)", version="2.3.0")

class SegmentOut(BaseModel):
    start: float
    end: float
    text: str
    avg_logprob: Optional[float] = None
    words: Optional[int] = None

class STTResponse(BaseModel):
    text: str
    segments: List[SegmentOut]
    language: Dict[str, Any]  # {code, prob, source}
    timings_ms: Dict[str, float]
    server: Dict[str, Any]
    gpu: Dict[str, Any]
    queue_depth: int
    confidence: Dict[str, Any]
    routing: Dict[str, Any]

# Prometheus
REQUESTS_TOTAL = Counter("stt_requests_total", "Total STT requests", ["endpoint"])
REQUESTS_EXC   = Counter("stt_requests_exceptions_total", "Exceptions", ["endpoint", "type"])
FALLBACK_TOTAL = Counter("stt_fallback_total", "Fallback attempts", ["result"])  # used|skipped|failed
REQ_TOTAL_SEC  = Histogram("stt_request_total_seconds", "Total request time (s)")
PRIMARY_SEC    = Histogram("stt_primary_infer_seconds", "Primary inference time (s)")
FALLBACK_SEC   = Histogram("stt_fallback_infer_seconds", "Fallback inference time (s)")
QUEUE_WAIT_SEC = Histogram("stt_queue_wait_seconds", "Queue wait time (s)")
QUEUE_GAUGE    = Gauge("stt_queue_depth", "Queue depth")
INFLIGHT_GAUGE = Gauge("stt_in_flight", "In-flight requests")
GPU_UTIL       = Gauge("stt_gpu_utilization_pct", "GPU utilization percent")
GPU_MEM_USED   = Gauge("stt_gpu_mem_used_mb", "GPU memory used (MB)")
GPU_MEM_TOTAL  = Gauge("stt_gpu_mem_total_mb", "GPU memory total (MB)")

def gpu_stats() -> Dict[str, Any]:
    if not NVML_OK:
        return {"name": "unknown", "index": DEVICE_INDEX, "memory_used_mb": 0, "memory_total_mb": 0,
                "utilization_pct": 0, "memory_util_pct": 0, "temperature_c": 0}
    try:
        mem = pynvml.nvmlDeviceGetMemoryInfo(nv_dev)
        util = pynvml.nvmlDeviceGetUtilizationRates(nv_dev)
        temp = pynvml.nvmlDeviceGetTemperature(nv_dev, pynvml.NVML_TEMPERATURE_GPU)
        raw  = pynvml.nvmlDeviceGetName(nv_dev)
        name = raw.decode() if isinstance(raw, (bytes, bytearray)) else str(raw)
        used_mb = round(mem.used / (1024**2))
        total_mb = round(mem.total / (1024**2))
        util_pct = int(util.gpu)
        GPU_UTIL.set(util_pct)
        GPU_MEM_USED.set(used_mb)
        GPU_MEM_TOTAL.set(total_mb)
        return {
            "name": name, "index": DEVICE_INDEX,
            "memory_used_mb": used_mb, "memory_total_mb": total_mb,
            "utilization_pct": util_pct,
            "memory_util_pct": int(100 * mem.used / mem.total) if mem.total else 0,
            "temperature_c": int(temp),
        }
    except Exception as e:
        return {"name": "unknown", "index": DEVICE_INDEX, "memory_used_mb": 0, "memory_total_mb": 0,
                "utilization_pct": 0, "memory_util_pct": 0, "temperature_c": 0, "error": str(e)}

def aggregate_text(segments) -> str:
    return " ".join(s.text.strip() for s in segments)

def to_bool(v) -> bool:
    if isinstance(v, bool): return v
    if isinstance(v, (int, float)): return bool(v)
    if isinstance(v, str): return v.strip().lower() in ("1","true","yes","y","on")
    return False

def _wordish_weight(text: str) -> int:
    return max(1, len((text or "").strip().split()))

def _logprob_confidence(seg_list) -> Dict[str, Any]:
    pairs = []
    for s in seg_list:
        lp = getattr(s, "avg_logprob", None)
        if lp is None: continue
        w = _wordish_weight(getattr(s, "text", "") or "")
        pairs.append((float(lp), w))
    if not pairs:
        return {"logprob_avg_weighted": None, "logprob_min": None, "logprob_max": None,
                "word_weight": 0, "logprob_confidence": 0.0, "method": "exp(weighted_avg_logprob)"}
    total_w = sum(w for _, w in pairs)
    avg_lp = sum(lp*w for lp, w in pairs) / max(1, total_w)
    clamped = max(-5.0, min(0.0, avg_lp))
    conf = math.exp(clamped)
    lp_vals = [lp for lp, _ in pairs]
    return {
        "logprob_avg_weighted": round(avg_lp, 4),
        "logprob_min": round(min(lp_vals), 4),
        "logprob_max": round(max(lp_vals), 4),
        "word_weight": int(total_w),
        "logprob_confidence": round(conf, 4),
        "method": "exp(weighted_avg_logprob)",
    }

def _build_segments(seg_list) -> List[SegmentOut]:
    return [SegmentOut(
        start=s.start, end=s.end, text=s.text,
        avg_logprob=getattr(s, "avg_logprob", None),
        words=_wordish_weight(getattr(s, "text", "") or "")
    ) for s in seg_list]

def _transcribe_bytes(model: WhisperModel, audio_bytes: bytes, beam_size: int, language: Optional[str],
                      vad_filter: bool, word_timestamps: bool):
    audio_io = io.BytesIO(audio_bytes)
    segs, info = model.transcribe(
        audio=audio_io, beam_size=beam_size, language=language,
        vad_filter=vad_filter, word_timestamps=word_timestamps
    )
    return list(segs), info

# Security: Auth + Rate Limiting
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

# Middleware: size cap
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

# Lifespan
@app.on_event("startup")
async def _startup():
    global _fallback_client
    headers = {"Authorization": f"Bearer {FALLBACK_API_KEY}"} if FALLBACK_API_KEY else {}
    _fallback_client = httpx.AsyncClient(base_url=FALLBACK_URL, timeout=60.0, headers=headers)
    print(f"Ready: device={DEVICE}:{DEVICE_INDEX} commit={SERVER_COMMIT} fallback={FALLBACK_URL}")

@app.on_event("shutdown")
async def _shutdown():
    global _fallback_client
    executor.shutdown(wait=True)
    with suppress(Exception):
        if _fallback_client:
            await _fallback_client.aclose()
    with suppress(Exception):
        if NVML_OK:
            pynvml.nvmlShutdown()

# Health & metrics
@app.get("/v1/health")
def health():
    return {
        "ok": True, "role": "primary",
        "device": DEVICE, "device_index": DEVICE_INDEX, "compute_type": COMPUTE_TYPE,
        "small": CT2_SMALL_PATH, "fallback_url": FALLBACK_URL,
        "max_conc": MAX_CONCURRENCY,
        "thresholds": {"logprob": CONF_LOGPROB_THRESHOLD, "language_prob": CONF_LANGPROB_THRESHOLD, "min_words": MIN_WORDS_FOR_CONF},
        "nvml": NVML_OK, "commit": SERVER_COMMIT,
    }

@app.get("/v1/metrics")
def metrics_json():
    gs = gpu_stats()
    return {"gpu": gs, "queue_depth": waiting, "in_flight": in_flight,
            "max_concurrency": MAX_CONCURRENCY, "server_commit": SERVER_COMMIT}

@app.get("/metrics")
def metrics_prom():
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# Transcribe
@app.post("/v1/transcribe")
async def transcribe(
    request: Request,
    _admit=Depends(admit_request),
    audio: UploadFile = File(...),
    language: Optional[str] = Form(default=None),   # None => autodetect; present => forced
    beam_size: int = Form(default=1),
    vad_filter: bool | str = Form(default=True),
    word_timestamps: bool | str = Form(default=False),
    conf_threshold: Optional[float] = Form(default=None),
    langprob_threshold: Optional[float] = Form(default=None),
    min_words_for_conf: Optional[int] = Form(default=None),
):
    global waiting, in_flight
    REQUESTS_TOTAL.labels(endpoint="/v1/transcribe").inc()

    req_id = str(uuid.uuid4())[:8]
    t_total0 = time.perf_counter()

    async with state_lock:
        waiting += 1
        QUEUE_GAUGE.set(waiting)
    q_enter = t_total0

    await sem.acquire()
    async with state_lock:
        waiting -= 1
        in_flight += 1
        QUEUE_GAUGE.set(waiting)
        INFLIGHT_GAUGE.set(in_flight)

    try:
        audio_bytes = await audio.read()
    except Exception as e:
        REQUESTS_EXC.labels(endpoint="/v1/transcribe", type="read_error").inc()
        async with state_lock:
            in_flight = max(0, in_flight - 1)
            INFLIGHT_GAUGE.set(in_flight)
        sem.release()
        raise HTTPException(status_code=400, detail=f"Invalid audio: {e}")

    try:
        q_leave = time.perf_counter()
        queue_wait_ms = (q_leave - q_enter) * 1000.0
        with QUEUE_WAIT_SEC.time(): pass

        loop = asyncio.get_running_loop()
        t_p0 = time.perf_counter()

        forced_language = (language is not None and str(language).strip() != "")
        effective_language: Optional[str] = language if forced_language else None

        primary_err: Optional[str] = None
        try:
            small_segs, small_info = await loop.run_in_executor(
                executor, _transcribe_bytes, small_model, audio_bytes,
                int(beam_size), effective_language,
                to_bool(vad_filter), to_bool(word_timestamps),
            )
        except Exception as e:
            small_segs, small_info = [], type("Info", (), {"language": None, "language_probability": None})()
            primary_err = repr(e)[:200]

        t_p1 = time.perf_counter()
        primary_ms = (t_p1 - t_p0) * 1000.0
        PRIMARY_SEC.observe(primary_ms / 1000.0)

    finally:
        async with state_lock:
            in_flight = max(0, in_flight - 1)
            INFLIGHT_GAUGE.set(in_flight)
        sem.release()

    # Routing decision
    fallback_ms = 0.0
    final_model = "small"
    fb_err: Optional[str] = None

    conf_meta = _logprob_confidence(small_segs) if not primary_err else {"logprob_confidence": 0.0}
    conf = conf_meta.get("logprob_confidence", 0.0) or 0.0
    lp = float(getattr(small_info, "language_probability", 0.0) or 0.0)

    threshold      = float(conf_threshold) if conf_threshold is not None else CONF_LOGPROB_THRESHOLD
    lang_threshold = float(langprob_threshold) if langprob_threshold is not None else CONF_LANGPROB_THRESHOLD
    min_words      = int(min_words_for_conf) if min_words_for_conf is not None else MIN_WORDS_FOR_CONF

    too_short = sum(_wordish_weight(getattr(s, "text", "") or "") for s in small_segs) < max(1, min_words)
    lang_check_applicable = not forced_language
    lang_below = (lp < lang_threshold) if lang_check_applicable else False

    need_fallback = bool(primary_err) or (conf < threshold) or lang_below or too_short

    def _as_segout_list(segs): return _build_segments(segs)
    final_segs: List[SegmentOut] = _as_segout_list(small_segs) if small_segs else []
    final_lang: Dict[str, Any] = {
        "code": getattr(small_info, "language", None),
        "prob": getattr(small_info, "language_probability", None),
        "source": "forced" if forced_language else "auto",
    }

    if need_fallback:
        t_f0 = time.perf_counter()
        try:
            assert _fallback_client is not None
            data: Dict[str, Any] = {"beam_size": 1, "vad_filter": "true", "word_timestamps": "false"}
            if forced_language:
                data["language"] = language  # same as user provided
            files = {"audio": ("audio.wav", io.BytesIO(audio_bytes), "audio/wav")}
            r = await _fallback_client.post("/v1/transcribe", files=files, data=data)
            r.raise_for_status()
            fb = r.json()
            fallback_ms_hdr = r.headers.get("X-Total-ms") or r.headers.get("X-Infer-ms")
            fallback_ms = float(fallback_ms_hdr) if fallback_ms_hdr else float(fb.get("timings_ms", {}).get("total", 0.0))
            FALLBACK_SEC.observe(fallback_ms / 1000.0)

            segs = fb.get("segments", []) or []
            final_segs = [SegmentOut(
                start=float(d.get("start", 0.0)),
                end=float(d.get("end", 0.0)),
                text=str(d.get("text", "")),
                avg_logprob=d.get("avg_logprob"),
                words=_wordish_weight(d.get("text", "")),
            ) for d in segs]

            lang_dict = fb.get("language", {}) or {}
            final_lang = {
                "code": lang_dict.get("code"),
                "prob": lang_dict.get("prob"),
                "source": "forced" if forced_language else "auto",
            }
            final_model = "medium"
            FALLBACK_TOTAL.labels(result="used").inc()
        except Exception as e:
            fb_err = repr(e)[:200]
            FALLBACK_TOTAL.labels(result="failed").inc()
        t_f1 = time.perf_counter()
        if fallback_ms <= 0.0:
            fallback_ms = (t_f1 - t_f0) * 1000.0

    total_ms = (time.perf_counter() - t_total0) * 1000.0
    REQ_TOTAL_SEC.observe(total_ms / 1000.0)
    gs = gpu_stats()
    q_depth_snapshot = waiting

    if need_fallback and final_model != "medium" and fb_err:
        payload_err = {
            "error": {
                "type": "UpstreamFailure",
                "message": "Primary failed or low-confidence; fallback failed",
                "primary_error": primary_err,
                "fallback_error": fb_err,
                "request_id": req_id,
            }
        }
        headers = {
            "X-Req-Id": req_id,
            "X-Queue-Wait-ms": f"{queue_wait_ms:.2f}",
            "X-Primary-Infer-ms": f"{primary_ms:.2f}",
            "X-Fallback-Infer-ms": f"{fallback_ms:.2f}",
            "X-Total-ms": f"{total_ms:.2f}",
            "X-Queue-Depth": str(q_depth_snapshot),
            "X-InFlight": str(in_flight),
            "X-Max-Concurrency": str(MAX_CONCURRENCY),
            "X-Fallback-Used": "true",
            "X-Final-Model": "error",
            "X-Fallback-Error": fb_err or "",
        }
        return JSONResponse(status_code=502, content=payload_err, headers=headers)

    routing_reason = (
        "primary_error" if primary_err else
        ("below_logprob_threshold" if conf < threshold else
         ("below_language_probability" if lang_below else
          ("too_short" if too_short else "ok")))
    )
    routing = {
        "fallback_used": final_model == "medium" and not fb_err,
        "fallback_attempted": need_fallback,
        "fallback_error": fb_err,
        "final_model": final_model,
        "reason": routing_reason,
    }
    if not need_fallback:
        FALLBACK_TOTAL.labels(result="skipped").inc()

    payload = STTResponse(
        text=aggregate_text(final_segs),
        segments=final_segs,
        language=final_lang,
        timings_ms={
            "queue_wait": round(queue_wait_ms, 2),
            "primary_infer": round(primary_ms, 2),
            "fallback_infer": round(fallback_ms, 2) if need_fallback else 0.0,
            "total": round(total_ms, 2),
        },
        server={
            "role": "primary",
            "device": f"{DEVICE}:{DEVICE_INDEX}",
            "compute_type": COMPUTE_TYPE,
            "commit": SERVER_COMMIT,
            "request_id": req_id,
            "models": {"primary": os.path.basename(CT2_SMALL_PATH), "fallback_remote": FALLBACK_URL},
        },
        gpu=gs,
        queue_depth=max(q_depth_snapshot, 0),
        confidence={
            **(conf_meta if conf_meta else {}),
            "language_probability_primary": round(lp, 4),
            "threshold": round(threshold, 3),
            "lang_threshold": round(lang_threshold, 3),
            "below_threshold": bool(conf < threshold) if not primary_err else True,
            "below_langprob": bool(lang_below) if not primary_err else False,
            "too_short": bool(too_short) if not primary_err else False,
        },
        routing=routing,
    )

    headers = {
        "X-Req-Id": req_id,
        "X-Queue-Wait-ms": f"{queue_wait_ms:.2f}",
        "X-Primary-Infer-ms": f"{primary_ms:.2f}",
        "X-Fallback-Infer-ms": f"{fallback_ms:.2f}" if need_fallback else "0",
        "X-Total-ms": f"{total_ms:.2f}",
        "X-GPU-Util-pct": str(gs.get("utilization_pct", 0)),
        "X-GPU-Mem-Used-MB": str(gs.get("memory_used_mb", 0)),
        "X-Queue-Depth": str(q_depth_snapshot),
        "X-InFlight": str(in_flight),
        "X-Max-Concurrency": str(MAX_CONCURRENCY),
        "X-Conf-Logprob-Primary": str(conf),
        "X-Conf-Threshold": str(threshold),
        "X-Conf-Below": "true" if (primary_err or conf < threshold or too_short) else "false",
        "X-LangProb-Primary": str(lp),
        "X-LangProb-Threshold": str(lang_threshold),
        "X-Lang-Check-Applicable": "true" if lang_check_applicable else "false",
        "X-Fallback-Used": "true" if routing["fallback_used"] else "false",
        "X-Final-Model": final_model,
        "X-Fallback-Error": fb_err or "",
    }
    return JSONResponse(status_code=200, content=payload.model_dump(), headers=headers)
