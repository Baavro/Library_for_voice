#!/usr/bin/env python3
"""
Minimal transcription client using the STT Python SDK.

Usage:
  python transcribe_file.py --file sample.wav \
      --base-url http://PRIMARY_HOST:8081 \
      --api-key your_api_key \
      --language en

- Omit --language (or set --language none) to enable auto-detect routing.
- Keep beam_size=1 and vad_filter=True for best latency (SDK defaults).

python stt_test.py \
  --file /Users/sankalppatidar/Developers/Voice\ Library/Test_Code/test_wav/F10_02_04.wav \
  --base-url http://127.0.0.1:8081 \
  --api-key primary_key_abc \
  --language en
  
"""
from __future__ import annotations
import argparse, pathlib, asyncio
import httpx

# SDK is a single module per the doc; place stt_sdk.py in your project or install your package.
from stt_sdk.client import STTClient, STTClientConfig  # provided by your repo’s SDK

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--file", required=True, help="Path to a mono 16k/32k WAV (recommended).")
    p.add_argument("--base-url", required=True, help="Primary service URL, e.g. http://host:8081")
    p.add_argument("--api-key", required=True, help="Bearer token for the service")
    p.add_argument("--language", default="en",
                   help="e.g. en, hi. Use 'none' to auto-detect.")
    p.add_argument("--word-timestamps", action="store_true",
                   help="Enable word-level timestamps (adds latency).")
    p.add_argument("--conf-threshold", type=float, default=None,
                   help="Lower => fewer fallbacks (default 0.55 if unset).")
    p.add_argument("--langprob-threshold", type=float, default=None,
                   help="Used only in autodetect mode (default 0.70 if unset).")
    p.add_argument("--min-words-for-conf", type=int, default=None,
                   help="Default 3; use 2 for very short commands.")
    p.add_argument("--timeout", type=float, default=60.0)
    return p.parse_args()

async def main():
    args = parse_args()
    language = None if args.language.lower() == "none" else args.language

    cfg = STTClientConfig(
        base_url=args.base_url,
        api_key=args.api_key,
        max_concurrency=4,      # client-side parallelism
        timeout_s=args.timeout,
    )
    client = STTClient(cfg)

    try:
        audio = pathlib.Path(args.file).read_bytes()

        kwargs = {
            "language": language,           # force language or enable auto-detect
            "beam_size": 1,                 # best latency
            "vad_filter": True,             # recommended
            "word_timestamps": args.word_timestamps,
        }
        # Optional routing knobs
        if args.conf_threshold is not None:
            kwargs["conf_threshold"] = args.conf_threshold
        if args.langprob_threshold is not None:
            kwargs["langprob_threshold"] = args.langprob_threshold
        if args.min_words_for_conf is not None:
            kwargs["min_words_for_conf"] = args.min_words_for_conf

        res = await client.transcribe_bytes(audio, **kwargs)

        # --- Output summary ---
        print(res.pretty_seconds())
        print("Text:", res.text)
        print("Final model:", res.final_model)       # "small" or "medium"
        print("Language:", res.language)             # {'code','prob','source'}
        print("Fallback used:", res.fallback_used)
        print("Segments:")
        for s in res.segments:
            print(f"  [{s['start']:.2f} -> {s['end']:.2f}] {s['text']}")

    except httpx.HTTPStatusError as e:
        r = e.response
        print("HTTP error:", r.status_code)
        # Server-side diagnostics (helpful in logs)
        for h in [
            "X-Req-Id", "X-Error-Type", "X-Error-Message", "X-Fallback-Error",
            "X-Queue-Wait-ms", "X-Primary-Infer-ms", "X-Fallback-Infer-ms", "X-Total-ms",
            "X-Fallback-Used", "X-Final-Model",
        ]:
            if h in r.headers:
                print(f"{h}: {r.headers[h]}")
        try:
            print("Body:", r.json())
        except Exception:
            pass
    except Exception as e:
        print("Fatal client-side error:", repr(e))
    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(main())


