from __future__ import annotations
import asyncio, base64, json, time, os
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import psutil
import torch
# NVML
try:
    import pynvml as nvml
    NVML_AVAILABLE = True
except Exception:
    NVML_AVAILABLE = False

from .engine_class import OrpheusModel
app = FastAPI()

# --- LLM (AWQ) config via env ---
MODEL_DIR = os.getenv("ORPHEUS_MODEL_PATH", "/home/ubuntu/Server/orpheus-3b-0.1-ft-awq-w4g128-zp")
TOKENIZER_DIR = os.getenv("ORPHEUS_TOKENIZER_PATH", MODEL_DIR)
QUANT = os.getenv("ORPHEUS_QUANTIZATION", "awq")  # "awq" | "gptq" | "bnb" | "none"

llm = OrpheusModel(
    model_path=MODEL_DIR,
    tokenizer_path=TOKENIZER_DIR,
    quantization=(None if QUANT.lower() == "none" else QUANT),
    dtype=torch.float16,
    max_model_len=int(os.getenv("ORPHEUS_MAX_MODEL_LEN", "512")),
    gpu_memory_utilization=float(os.getenv("ORPHEUS_GPU_MEM_UTIL", "0.650")),
    enforce_eager=True,
    max_num_seqs=int(os.getenv("ORPHEUS_MAX_NUM_SEQS", "192")),
    # kv_cache_dtype=os.getenv("VLLM_KV_CACHE_DTYPE") or "fp8",  # e.g. "fp8" if your vLLM supports it
    kv_cache_dtype="fp8",  # e.g. "fp8" if your vLLM supports it
    
)
print(f"[worker] LLM ready: {MODEL_DIR} (quant={QUANT})")


# Config
SR = 24000
GPU_INDEX = int(os.getenv("NVML_GPU_INDEX", os.getenv("TTS_GPU_INDEX", "0")))
MAX_CONCURRENCY = int(os.getenv("WORKER_MAX_CONCURRENCY", "16"))
METRICS_INTERVAL = float(os.getenv("METRICS_INTERVAL_MS", "250")) / 1000.0

# Concurrency tracking
sema = asyncio.Semaphore(MAX_CONCURRENCY)

# NVML init
NVML_HANDLE = None
if NVML_AVAILABLE:
    try:
        nvml.nvmlInit()
        NVML_HANDLE = nvml.nvmlDeviceGetHandleByIndex(GPU_INDEX)
    except Exception:
        NVML_HANDLE = None

def _gpu_metrics():
    gpu_util = None
    vram_used_mb = None
    vram_total_mb = None
    if NVML_HANDLE is not None:
        try:
            util = nvml.nvmlDeviceGetUtilizationRates(NVML_HANDLE)
            mem = nvml.nvmlDeviceGetMemoryInfo(NVML_HANDLE)
            gpu_util = util.gpu / 100.0
            vram_used_mb = round(mem.used / (1024**2), 1)
            vram_total_mb = round(mem.total / (1024**2), 1)
        except Exception:
            pass
    # streams_active derived from semaphore
    streams_active = MAX_CONCURRENCY - sema._value  # noqa: accessing _value is fine here
    return {
        "gpu_util": gpu_util,
        "vram_used_mb": vram_used_mb,
        "vram_total_mb": vram_total_mb,
        "streams_active": streams_active,
        "queue_depth": None,              # if you implement a queue, report it here
        "server_concurrency": MAX_CONCURRENCY,
    }

@app.post("/v1/tts:stream")
async def tts_stream(req: Request):
    body = await req.json()
    text = (body.get("text") or "").strip()
    voice = body.get("voice")
    params = body.get("params") or {}
    if not text:
        return {"error": "text required"}

    await sema.acquire()
    async def gen():
        try:
            # send initial metrics immediately for fast TTFB
            yield (json.dumps({"type":"metrics", **_gpu_metrics()}) + "\n").encode()

            # fan-in audio chunks into a queue
            q = asyncio.Queue()

            async def produce_audio():
                try:
                    async for audio in llm.generate_speech_async(prompt=text, voice=voice, **params):
                        await q.put(audio)
                finally:
                    await q.put(None)

            task = asyncio.create_task(produce_audio())

            while True:
                try:
                    item = await asyncio.wait_for(q.get(), timeout=METRICS_INTERVAL)
                except asyncio.TimeoutError:
                    # periodic metrics frame
                    yield (json.dumps({"type":"metrics", **_gpu_metrics()}) + "\n").encode()
                    continue

                if item is None:
                    break  # done

                b64 = base64.b64encode(item).decode("ascii")
                yield (json.dumps({"type":"chunk","b64":b64,"ts":time.time()}) + "\n").encode()

            # final metrics + eos
            yield (json.dumps({"type":"metrics", **_gpu_metrics()}) + "\n").encode()
            yield (json.dumps({"type":"eos"}) + "\n").encode()
        finally:
            sema.release()

    return StreamingResponse(gen(), media_type="application/x-ndjson")

@app.get("/v1/system")
async def system():
    # one-shot snapshot for the gateway/ops
    proc = psutil.Process(os.getpid())
    return {
        **_gpu_metrics(),
        "proc_rss_mb": round(proc.memory_info().rss / (1024**2), 1),
        "pid": os.getpid(),
        "gpu_index": GPU_INDEX,
    }


@app.get("/healthz")
async def healthz():
    return {"ok": True}