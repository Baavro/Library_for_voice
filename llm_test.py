#!/usr/bin/env python3
"""
Smoke tests for llm_service.client

Usage:
  # 1) Ensure you have the package installed:
  #    pip install llm-service
  #    # or: pip install git+https://github.com/<your-org>/llm_service.git
  #
  # 2) Export your credentials:
  #    export GROQ_API_KEY=sk-xxxx
  #    # (Optional) export GROQ_API_BASE=https://api.groq.com/openai
  #
  # 3) Run the script:
  #    python smoke_tests_llm_service.py \
  #      --chat-model llama-3.3-70b-versatile \
  #      --fast-model llama-3.1-8b-instant
"""

from __future__ import annotations
import os
import argparse
import asyncio
import sys
import time

from llm_sdk.client import (
    GroqLLM,
    AsyncGroqLLM,
    llm_input_for_chat,
    llm_input_for_prompt,
    LLMInput,
    Message,
    # Exceptions (as documented)
    GroqHTTPError,
    GroqSDKError,
)

def require_env(var: str):
    val = os.environ.get(var, "")
    if not val:
        raise RuntimeError(f"Environment variable {var} is required but not set.")
    return val

def test_chat_sync(chat_model: str):
    print(f"\n[SYNC CHAT] model={chat_model}")
    with GroqLLM() as cli:
        t0 = time.perf_counter()
        inp = llm_input_for_chat(
            model=chat_model,
            messages=[Message(role="user", content="Write a haiku about code.")],
        )
        res = cli.chat(inp)
        dt = (time.perf_counter() - t0) * 1000
        print("→ Text:", (res.text or "").strip())
        print(f"→ Finish reason: {res.finish_reason}, time_ms={dt:.1f}")

def test_chat_stream_sync(chat_model: str):
    print(f"\n[SYNC CHAT STREAM] model={chat_model}")
    with GroqLLM() as cli:
        inp = llm_input_for_chat(
            model=chat_model,
            messages=[Message("user", "Tell me a short story, stream it.")],
        )
        print("→ Stream (delta chunks):")
        for chunk in cli.chat_stream(inp):
            if getattr(chunk, "delta", None):
                print(chunk.delta, end="", flush=True)
        print("\n→ (stream ended)")

async def test_response_async(fast_model: str):
    print(f"\n[ASYNC RESPONSE] model={fast_model}")
    async with AsyncGroqLLM() as cli:
        inp = llm_input_for_prompt(
            model=fast_model,
            prompt="Explain recursion in one friendly paragraph.",
        )
        res = await cli.response(inp)
        print("→ Text:", (res.text or "").strip())
        print(f"→ Finish reason: {res.finish_reason}")

async def test_response_stream_async(fast_model: str):
    print(f"\n[ASYNC RESPONSE STREAM] model={fast_model}")
    async with AsyncGroqLLM() as cli:
        inp = llm_input_for_prompt(
            model=fast_model,
            prompt="List 5 startup ideas as bullets.",
            stream=True,
        )
        print("→ Stream (delta chunks):")
        async for chunk in cli.response_stream(inp):
            if getattr(chunk, "delta", None):
                print(chunk.delta, end="", flush=True)
        print("\n→ (stream ended)")

def test_batch_small(chat_model_for_batch: str):
    """
    Minimal batch create + retrieve demonstration.
    Note: depending on vendor semantics, retrieval may show 'processing' immediately after create.
    """
    print(f"\n[BATCH] requests -> /v1/responses via {chat_model_for_batch}")
    with GroqLLM() as cli:
        payload = {
            "requests": [
                {
                    "custom_id": "1",
                    "method": "POST",
                    "url": "/v1/responses",
                    "body": {"model": chat_model_for_batch, "input": "First batch item"},
                },
                {
                    "custom_id": "2",
                    "method": "POST",
                    "url": "/v1/responses",
                    "body": {"model": chat_model_for_batch, "input": "Second batch item"},
                },
            ]
        }
        res = cli.batch_create(LLMInput(batch=payload))
        print("→ Created batch id:", res.id)
        # Try a simple retrieve (optional; may be pending)
        try:
            got = cli.batch_retrieve(res.id)
            print("→ Retrieve status:", getattr(got, "raw", None))
        except Exception as e:
            print("→ Retrieve attempt raised (often OK if not ready):", e)

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chat-model", default="llama-3.3-70b-versatile",
                        help="Model for chat / streaming chat tests")
    parser.add_argument("--fast-model", default="llama-3.1-8b-instant",
                        help="Model for async response tests / batch")
    args = parser.parse_args()

    # Ensure required env present (per docs)
    require_env("GROQ_API_KEY")

    try:
        test_chat_sync(args.chat_model)
        test_chat_stream_sync(args.chat_model)
        await test_response_async(args.fast_model)
        await test_response_stream_async(args.fast_model)
        test_batch_small(args.fast_model)
        print("\n✅ All smoke tests executed.")
    except GroqHTTPError as e:
        print(f"\n❌ GroqHTTPError: HTTP {e.status}, code={e.code}, req={e.request_id}\nBody: {e.body}")
        sys.exit(2)
    except GroqSDKError as e:
        print(f"\n❌ GroqSDKError: {e}")
        sys.exit(3)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e.__class__.__name__}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
