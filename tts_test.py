

#!/usr/bin/env python3
# tts_test.py — Full-spectrum test for orpheus-tts SDK

import argparse, asyncio, wave, os, sys, time
from typing import List, Optional

# If running from source tree without install:
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from Library.tts_sdk.sdk import OrpheusTTS

def write_wav(path: str, pcm_bytes: bytes, sample_rate: int = 24000):
    # Service streams raw PCM 16-bit mono little-endian (typical).
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)      # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)

async def stream_once(
    urls: List[str],
    api_key: Optional[str],
    text: str,
    voice: Optional[str],
    out: Optional[str],
    sample_rate: int,
    hedge_ttfb_ms: Optional[int],
    timeout_s: float,
    verbose: bool,
) -> float:
    pcm = bytearray()
    t0 = time.perf_counter()
    async with OrpheusTTS(base_urls=urls, api_key=api_key, hedge_ttfb_ms=hedge_ttfb_ms) as tts:
        # (optional) pre-snapshot
        if verbose:
            snaps = await tts.snapshots()
            for s in snaps:
                print("[pre] snap:", s)

        try:
            async for chunk in tts.stream(text=text, voice=voice, timeout_s=timeout_s):
                pcm.extend(chunk)
        except Exception as e:
            print(f"[ERROR] stream failed: {e}")
            raise
        finally:
            # (optional) post-snapshot
            if verbose:
                snaps = await tts.snapshots()
                for s in snaps:
                    print("[post] snap:", s)

    dt = time.perf_counter() - t0
    if out:
        write_wav(out, bytes(pcm), sample_rate=sample_rate)
        if verbose:
            print(f"[ok] wrote {len(pcm)} bytes to {out} in {dt*1000:.1f} ms")
    else:
        if verbose:
            print(f"[ok] received {len(pcm)} bytes in {dt*1000:.1f} ms")

    # return seconds per request (latency)
    return dt

async def run_parallel(
    urls: List[str],
    api_key: Optional[str],
    text: str,
    voice: Optional[str],
    out_dir: Optional[str],
    sample_rate: int,
    hedge_ttfb_ms: Optional[int],
    timeout_s: float,
    parallel: int,
    verbose: bool,
):
    os.makedirs(out_dir or ".", exist_ok=True)
    tasks = []
    for i in range(parallel):
        out = f"{out_dir}/tts_{i:03d}.wav" if out_dir else None
        tasks.append(
            stream_once(
                urls, api_key, text, voice, out,
                sample_rate, hedge_ttfb_ms, timeout_s, verbose
            )
        )
    latencies = await asyncio.gather(*tasks, return_exceptions=True)
    ok = [x for x in latencies if not isinstance(x, Exception)]
    errs = [x for x in latencies if isinstance(x, Exception)]
    if verbose:
        print(f"[summary] ok={len(ok)} err={len(errs)}")
        if ok:
            print(f"[summary] mean latency: {sum(ok)/len(ok):.3f}s | min={min(ok):.3f}s max={max(ok):.3f}s")

def parse_args():
    p = argparse.ArgumentParser(description="Orpheus TTS SDK test")
    p.add_argument("--urls", required=False, default="http://localhost:8080",
                   help="Comma-separated gateway/worker URLs (gateway recommended)")
    p.add_argument("--api-key", default="dev_key", help="API key for the gateway (Authorization: Bearer ...)")
    p.add_argument("--text", required=False, default="Hello from Orpheus. This is a streaming T T S check.",
                   help="Text to synthesize")
    p.add_argument("--voice", default=None, help="Voice id/name if your backend supports it")
    p.add_argument("--timeout", type=float, default=60.0, help="Client + server timeout seconds")
    p.add_argument("--sample-rate", type=int, default=24000, help="Sample rate for WAV")
    p.add_argument("--hedge-ttfb-ms", type=int, default=400, help="Hedge backup start if TTFB exceeds this (ms)")
    p.add_argument("--out", default="out.wav", help="WAV path (single request). Use --parallel for many.")
    p.add_argument("--verbose", action="store_true", help="Print snapshots and progress")
    p.add_argument("--parallel", type=int, default=1, help="Run N requests in parallel")
    p.add_argument("--out-dir", default="outs", help="Directory for parallel WAVs")
    return p.parse_args()

async def main():
    args = parse_args()
    urls = [u.strip() for u in args.urls.split(",") if u.strip()]

    if args.parallel > 1:
        await run_parallel(
            urls=urls,
            api_key=args.api_key,
            text=args.text,
            voice=args.voice,
            out_dir=args.out_dir,
            sample_rate=args.sample_rate,
            hedge_ttfb_ms=args.hedge_ttfb_ms,
            timeout_s=args.timeout,
            parallel=args.parallel,
            verbose=args.verbose,
        )
    else:
        await stream_once(
            urls=urls,
            api_key=args.api_key,
            text=args.text,
            voice=args.voice,
            out=args.out,
            sample_rate=args.sample_rate,
            hedge_ttfb_ms=args.hedge_ttfb_ms,
            timeout_s=args.timeout,
            verbose=args.verbose,
        )

if __name__ == "__main__":
    asyncio.run(main())
