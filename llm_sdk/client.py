# # llm_service/client.py
# from __future__ import annotations

# import json, os, time, uuid, math
# from dataclasses import dataclass, asdict
# from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, AsyncIterator

# import httpx

# __all__ = [
#     "GroqLLM",
#     "AsyncGroqLLM",
#     "LLMInput",
#     "LLMResult",
#     "Message",
#     "llm_input_for_chat",
#     "llm_input_for_prompt",
#     "GroqSDKError",
# ]
# __version__ = "0.2.0"

# # ======================
# # Errors / Basic Models
# # ======================

# class GroqSDKError(RuntimeError):
#     pass


# @dataclass
# class Message:
#     role: str
#     content: str
#     name: Optional[str] = None
#     tool_call_id: Optional[str] = None
#     # For assistant messages containing tool calls:
#     tool_calls: Optional[List[Dict[str, Any]]] = None


# @dataclass
# class LLMInput:
#     # Common
#     model: Optional[str] = None
#     idempotency_key: Optional[str] = None
#     # Chat
#     messages: Optional[List[Message]] = None
#     tools: Optional[List[Dict[str, Any]]] = None
#     tool_choice: Optional[Any] = None
#     response_format: Optional[Dict[str, Any]] = None
#     temperature: Optional[float] = None
#     top_p: Optional[float] = None
#     max_completion_tokens: Optional[int] = None
#     seed: Optional[int] = None
#     stop: Optional[Iterable[str]] = None
#     parallel_tool_calls: Optional[bool] = None
#     reasoning_effort: Optional[str] = None
#     reasoning_format: Optional[str] = None
#     include_reasoning: Optional[bool] = None
#     search_settings: Optional[Dict[str, Any]] = None
#     user: Optional[str] = None
#     stream: Optional[bool] = None
#     # Responses API
#     prompt: Optional[str] = None
#     instructions: Optional[str] = None
#     max_output_tokens: Optional[int] = None
#     service_tier: Optional[str] = None
#     text: Optional[Dict[str, Any]] = None  # not used by default (vendor-specific)
#     # Batches
#     batch: Optional[Dict[str, Any]] = None


# @dataclass
# class LLMResult:
#     kind: str  # "message" | "stream_chunk" | "batch" | "raw"
#     text: str = ""
#     delta: str = ""
#     tool_calls: Optional[List[Dict[str, Any]]] = None
#     finish_reason: Optional[str] = None
#     id: Optional[str] = None
#     raw: Optional[Dict[str, Any]] = None


# def llm_input_for_chat(
#     model: str,
#     messages: List[Message],
#     *,
#     temperature: Optional[float] = None,
#     top_p: Optional[float] = None,
#     max_completion_tokens: Optional[int] = None,
#     response_format: Optional[Dict[str, Any]] = None,
#     tools: Optional[List[Dict[str, Any]]] = None,
#     tool_choice: Optional[Any] = None,
#     stop: Optional[Iterable[str]] = None,
#     seed: Optional[int] = None,
#     user: Optional[str] = None,
#     idempotency_key: Optional[str] = None,
#     parallel_tool_calls: Optional[bool] = None,
#     reasoning_effort: Optional[str] = None,
#     reasoning_format: Optional[str] = None,
#     include_reasoning: Optional[bool] = None,
#     search_settings: Optional[Dict[str, Any]] = None,
# ) -> LLMInput:
#     return LLMInput(
#         model=model,
#         messages=messages,
#         temperature=temperature,
#         top_p=top_p,
#         max_completion_tokens=max_completion_tokens,
#         response_format=response_format,
#         tools=tools,
#         tool_choice=tool_choice,
#         stop=stop,
#         seed=seed,
#         user=user,
#         idempotency_key=idempotency_key,
#         parallel_tool_calls=parallel_tool_calls,
#         reasoning_effort=reasoning_effort,
#         reasoning_format=reasoning_format,
#         include_reasoning=include_reasoning,
#         search_settings=search_settings,
#     )


# def llm_input_for_prompt(
#     model: str,
#     prompt: str,
#     *,
#     temperature: Optional[float] = None,
#     top_p: Optional[float] = None,
#     max_output_tokens: Optional[int] = None,
#     instructions: Optional[str] = None,
#     user: Optional[str] = None,
#     idempotency_key: Optional[str] = None,
#     service_tier: Optional[str] = None,
#     tool_choice: Optional[Any] = None,
#     tools: Optional[List[Dict[str, Any]]] = None,
#     stream: Optional[bool] = None,
# ) -> LLMInput:
#     return LLMInput(
#         model=model,
#         prompt=prompt,
#         temperature=temperature,
#         top_p=top_p,
#         max_output_tokens=max_output_tokens,
#         instructions=instructions,
#         user=user,
#         idempotency_key=idempotency_key,
#         service_tier=service_tier,
#         tool_choice=tool_choice,
#         tools=tools,
#         stream=stream,
#     )

# # ======================
# # Config / HTTP helpers
# # ======================

# def _env(key: str, default: Optional[str] = None) -> Optional[str]:
#     v = os.getenv(key)
#     return v if (v is not None and v != "") else default

# def _api_base() -> str:
#     return _env("GROQ_API_BASE", "https://api.groq.com/openai")

# def _api_key() -> str:
#     key = _env("GROQ_API_KEY")
#     if not key:
#         raise GroqSDKError("GROQ_API_KEY is not set.")
#     return key

# def _headers(idempotency_key: Optional[str] = None) -> Dict[str, str]:
#     h = {
#         "Authorization": f"Bearer {_api_key()}",
#         "Content-Type": "application/json",
#     }
#     if idempotency_key:
#         h["Idempotency-Key"] = idempotency_key
#     return h

# def _drop_none(d: Dict[str, Any]) -> Dict[str, Any]:
#     return {k: v for k, v in d.items() if v is not None}

# def _msg_drop_none(m: Message) -> Dict[str, Any]:
#     d = asdict(m)
#     # Remove None and empty tool_calls if not present
#     d = {k: v for k, v in d.items() if v is not None}
#     if "tool_calls" in d and not d["tool_calls"]:
#         d.pop("tool_calls", None)
#     return d

# # Known per-model completion caps (from your matrix)
# _MODEL_COMPLETION_CAP: Dict[str, int] = {
#     "llama-3.1-8b-instant": 131072,
#     "llama-3.3-70b-versatile": 32768,
#     "meta-llama/llama-guard-4-12b": 1024,
#     "deepseek-r1-distill-llama-70b": 131072,
#     "meta-llama/llama-4-maverick-17b-128e-instruct": 8192,
#     "meta-llama/llama-4-scout-17b-16e-instruct": 8192,
#     "meta-llama/llama-prompt-guard-2-22m": 512,
#     "meta-llama/llama-prompt-guard-2-86m": 512,
#     "moonshotai/kimi-k2-instruct": 16384,
#     # whisper models: N/A
# }

# def _cap_tokens(model: Optional[str], key: str, val: Optional[int]) -> Optional[int]:
#     if val is None or not model:
#         return val
#     cap = _MODEL_COMPLETION_CAP.get(model)
#     if cap is None:
#         return val
#     return min(val, cap)

# # =========================
# # Payload build / extraction
# # =========================

# def _build_chat_payload(inp: LLMInput) -> Dict[str, Any]:
#     if not inp.model or not inp.messages:
#         raise GroqSDKError("Chat payload requires model and messages.")
#     payload: Dict[str, Any] = {
#         "model": inp.model,
#         "messages": [_msg_drop_none(m) for m in inp.messages],
#         "temperature": inp.temperature,
#         "top_p": inp.top_p,
#         "max_completion_tokens": _cap_tokens(inp.model, "max_completion_tokens", inp.max_completion_tokens),
#         "seed": inp.seed,
#         "stop": list(inp.stop) if inp.stop else None,
#         "response_format": inp.response_format,
#         "tools": inp.tools,
#         "tool_choice": inp.tool_choice,
#         "parallel_tool_calls": inp.parallel_tool_calls,
#         "reasoning_effort": inp.reasoning_effort,
#         "reasoning_format": inp.reasoning_format,
#         "include_reasoning": inp.include_reasoning,
#         "search_settings": inp.search_settings,
#         "user": inp.user,
#     }
#     return _drop_none(payload)

# def _build_response_payload(inp: LLMInput) -> Dict[str, Any]:
#     if not inp.model or inp.prompt is None:
#         raise GroqSDKError("Responses payload requires model and prompt.")
#     payload: Dict[str, Any] = {
#         "model": inp.model,
#         "input": inp.prompt,
#         "temperature": inp.temperature,
#         "top_p": inp.top_p,
#         "max_output_tokens": _cap_tokens(inp.model, "max_output_tokens", inp.max_output_tokens),
#         "instructions": inp.instructions,
#         "user": inp.user,
#         "tool_choice": inp.tool_choice,
#         "tools": inp.tools,
#         # Do NOT default service_tier=auto (orgs may not have access)
#         "service_tier": inp.service_tier,
#         # Avoid vendor-specific text formatting by default
#         # "text": inp.text,
#     }
#     return _drop_none(payload)

# def _extract_chat_text(res: Dict[str, Any]) -> Tuple[str, Optional[List[Dict[str, Any]]], Optional[str]]:
#     text = ""
#     tool_calls = None
#     finish_reason = None
#     choices = res.get("choices") or []
#     if choices:
#         c0 = choices[0] or {}
#         finish_reason = c0.get("finish_reason")
#         msg = c0.get("message") or {}
#         text = msg.get("content") or ""
#         tool_calls = msg.get("tool_calls")
#     return text, tool_calls, finish_reason

# def _extract_responses_text(res: Dict[str, Any]) -> Tuple[str, Optional[List[Dict[str, Any]]], Optional[str]]:
#     # 1) Try chat-like shape first
#     if "choices" in res:
#         return _extract_chat_text(res)

#     text = ""
#     tool_calls = None
#     finish_reason = res.get("finish_reason") or res.get("status")

#     # 2) Common Groq Responses shapes
#     # a) Flat string field
#     if isinstance(res.get("output_text"), str) and res["output_text"]:
#         return res["output_text"], tool_calls, finish_reason

#     # b) output list with text parts
#     out = res.get("output")
#     if isinstance(out, list) and out:
#         parts: List[str] = []

#         def pull_text(obj: Dict[str, Any]) -> None:
#             t = obj.get("type")
#             if t == "output_text" and isinstance(obj.get("text"), str):
#                 parts.append(obj["text"])
#             # nested content arrays
#             if "content" in obj and isinstance(obj["content"], list):
#                 for c in obj["content"]:
#                     if isinstance(c, dict) and c.get("type") in ("output_text", "output_text_delta"):
#                         if isinstance(c.get("text"), str):
#                             parts.append(c["text"])

#         for seg in out:
#             if isinstance(seg, dict):
#                 pull_text(seg)

#         if parts:
#             text = "".join(parts)

#     return text, tool_calls, finish_reason

# def _sse_iter_lines_from_chunks(chunks: Iterable[bytes]) -> Iterable[str]:
#     buf = b""
#     for chunk in chunks:
#         if not chunk:
#             continue
#         buf += chunk
#         while True:
#             i = buf.find(b"\n")
#             if i == -1:
#                 break
#             line, buf = buf[:i], buf[i+1:]
#             yield line.decode("utf-8", "ignore")

# def _sse_iter_lines(content: Iterable[bytes]) -> Iterable[str]:
#     buf = b""
#     for chunk in content:
#         if not chunk:
#             continue
#         buf += chunk
#         while b"\n" in buf:
#             line, buf = buf.split(b"\n", 1)
#             yield line.decode("utf-8", "ignore")

# def _parse_stream_event(data_json: Dict[str, Any]) -> Tuple[str, Optional[List[Dict[str, Any]]], Optional[str]]:
#     """
#     Extract streaming delta text and tool_calls from either Chat or Responses SSE shapes.
#     Returns (delta_text, tool_calls, finish_reason)
#     """

#     # 1) OpenAI/Chat-style SSE
#     if "choices" in data_json:
#         ch = data_json["choices"][0] if data_json.get("choices") else {}
#         delta = ch.get("delta") or {}
#         if isinstance(delta, dict):
#             return delta.get("content", "") or "", delta.get("tool_calls"), ch.get("finish_reason")
#         # fallback
#         return (ch.get("text") or ""), None, ch.get("finish_reason")

#     # 2) Groq Responses-style SSE
#     t = data_json.get("type")

#     # Incremental text deltas
#     if t == "response.output_text.delta":
#         d = data_json.get("delta")
#         if isinstance(d, dict):
#             return d.get("text", "") or "", None, None
#         if isinstance(d, str):
#             return d, None, None
#         return "", None, None

#     # Sometimes a full text lands as a response update
#     if t in {"response.output_text", "response.completed"}:
#         # Some shapes place text directly in 'response'
#         resp = data_json.get("response")
#         if isinstance(resp, dict) and isinstance(resp.get("output_text"), str):
#             return resp["output_text"], None, resp.get("finish_reason") or "stop"
#         # Or as a top-level field
#         if isinstance(data_json.get("output_text"), str):
#             return data_json["output_text"], None, "stop"
#         return "", None, "stop"

#     # Fallback: nested response container with text deltas
#     resp = data_json.get("response")
#     if isinstance(resp, dict):
#         if isinstance(resp.get("output_text_delta"), str):
#             return resp["output_text_delta"], None, None
#         if isinstance(resp.get("output_text"), str):
#             return resp["output_text"], None, resp.get("finish_reason")

#     # Unknown event; no text
#     return "", None, None
# # =================
# # Sync HTTP client
# # =================

# class GroqLLM:
#     def __init__(self, timeout: float = 60.0, max_retries: int = 3):
#         self._client = httpx.Client(timeout=timeout)
#         self._max_retries = max_retries

#     def __enter__(self) -> "GroqLLM":
#         return self

#     def __exit__(self, exc_type, exc, tb) -> None:
#         self.close()

#     def close(self) -> None:
#         self._client.close()

#     # ---- Chat ----
#     def chat(self, inp: LLMInput) -> LLMResult:
#         url = f"{_api_base()}/v1/chat/completions"
#         payload = _build_chat_payload(inp)
#         res = self._request("POST", url, payload, inp.idempotency_key)
#         text, tool_calls, finish = _extract_chat_text(res)
#         return LLMResult(kind="message", text=text or "", tool_calls=tool_calls, finish_reason=finish, id=res.get("id"), raw=res)

#     def chat_stream(self, inp: LLMInput) -> Iterator[LLMResult]:
#         url = f"{_api_base()}/v1/chat/completions"
#         payload = _build_chat_payload(inp)
#         payload["stream"] = True
#         headers = _headers(inp.idempotency_key)
#         with self._client.stream("POST", url, headers=headers, json=payload) as r:
#             if r.status_code >= 400:
#                 err = r.read().decode("utf-8", "ignore")
#                 raise GroqSDKError(f"HTTP {r.status_code}: {err}")
#             for line in _sse_iter_lines_from_chunks(r.iter_raw()):
#                 line = line.strip()
#                 if not line or not line.startswith("data:"):
#                     continue
#                 data = line[5:].strip()
#                 if data == "[DONE]":
#                     break
#                 try:
#                     obj = json.loads(data)
#                 except Exception:
#                     continue
#                 delta, tool_calls, finish = _parse_stream_event(obj)
#                 yield LLMResult(kind="stream_chunk", delta=delta or "", tool_calls=tool_calls, finish_reason=finish, raw=obj)

#     # ---- Responses ----
#     def response(self, inp: LLMInput) -> LLMResult:
#         url = f"{_api_base()}/v1/responses"
#         payload = _build_response_payload(inp)
#         res = self._request("POST", url, payload, inp.idempotency_key)
#         text, tool_calls, finish = _extract_responses_text(res)
#         return LLMResult(kind="message", text=text or "", tool_calls=tool_calls, finish_reason=finish, id=res.get("id"), raw=res)

#     def response_stream(self, inp: LLMInput) -> Iterator[LLMResult]:
#         url = f"{_api_base()}/v1/responses"
#         payload = _build_response_payload(inp)
#         payload["stream"] = True
#         headers = _headers(inp.idempotency_key)
#         with self._client.stream("POST", url, headers=headers, json=payload) as r:
#             if r.status_code >= 400:
#                 err = r.read().decode("utf-8", "ignore")
#                 raise GroqSDKError(f"HTTP {r.status_code}: {err}")
#             for line in _sse_iter_lines(r.iter_raw()):
#                 line = line.strip()
#                 if not line or not line.startswith("data:"):
#                     continue
#                 data = line[5:].strip()
#                 if data == "[DONE]":
#                     break
#                 try:
#                     obj = json.loads(data)
#                 except Exception:
#                     continue
#                 delta, tool_calls, finish = _parse_stream_event(obj)
#                 yield LLMResult(kind="stream_chunk", delta=delta or "", tool_calls=tool_calls, finish_reason=finish, raw=obj)

#     # ---- Batches ----
#     def batch_create(self, inp: LLMInput) -> LLMResult:
#         if not inp.batch:
#             raise GroqSDKError("batch_create requires LLMInput.batch")
#         url = f"{_api_base()}/v1/batches"
#         res = self._request("POST", url, inp.batch, inp.idempotency_key)
#         return LLMResult(kind="batch", id=res.get("id"), raw=res)

#     def batch_retrieve(self, batch_id: str) -> LLMResult:
#         url = f"{_api_base()}/v1/batches/{batch_id}"
#         res = self._request("GET", url, None, None)
#         return LLMResult(kind="batch", id=res.get("id"), raw=res)

#     def batch_list(self) -> LLMResult:
#         url = f"{_api_base()}/v1/batches"
#         res = self._request("GET", url, None, None)
#         return LLMResult(kind="batch", raw=res)

#     def batch_cancel(self, batch_id: str) -> LLMResult:
#         url = f"{_api_base()}/v1/batches/{batch_id}/cancel"
#         res = self._request("POST", url, {}, str(uuid.uuid4()))
#         return LLMResult(kind="batch", id=res.get("id"), raw=res)

#     # ---- Low-level request with retries ----
#     def _request(self, method: str, url: str, json_body: Optional[Dict[str, Any]], idem_key: Optional[str]) -> Dict[str, Any]:
#         backoff = 0.5
#         for attempt in range(self._max_retries):
#             try:
#                 r = self._client.request(method, url, headers=_headers(idem_key), json=json_body)
#                 if r.status_code >= 400:
#                     # 429/5xx => retry; others raise immediately
#                     if r.status_code in (429, 500, 502, 503, 504) and attempt < self._max_retries - 1:
#                         time.sleep(backoff)
#                         backoff = min(backoff * 2, 4.0)
#                         continue
#                     raise GroqSDKError(f"HTTP {r.status_code}: {r.text}")
#                 return r.json()
#             except httpx.TimeoutException as e:
#                 if attempt < self._max_retries - 1:
#                     time.sleep(backoff)
#                     backoff = min(backoff * 2, 4.0)
#                     continue
#                 raise GroqSDKError(f"Timeout: {e}") from e

# # ===================
# # Async HTTP client
# # ===================

# class AsyncGroqLLM:
#     def __init__(self, timeout: float = 60.0, max_retries: int = 3):
#         self._client = httpx.AsyncClient(timeout=timeout)
#         self._max_retries = max_retries

#     async def __aenter__(self) -> "AsyncGroqLLM":
#         return self

#     async def __aexit__(self, exc_type, exc, tb) -> None:
#         await self.close()

#     async def close(self) -> None:
#         await self._client.aclose()

#     # ---- Chat ----
#     async def chat(self, inp: LLMInput) -> LLMResult:
#         url = f"{_api_base()}/v1/chat/completions"
#         payload = _build_chat_payload(inp)
#         res = await self._request("POST", url, payload, inp.idempotency_key)
#         text, tool_calls, finish = _extract_chat_text(res)
#         return LLMResult(kind="message", text=text or "", tool_calls=tool_calls, finish_reason=finish, id=res.get("id"), raw=res)

#     async def chat_stream(self, inp: LLMInput) -> AsyncIterator[LLMResult]:
#         url = f"{_api_base()}/v1/chat/completions"
#         payload = _build_chat_payload(inp)
#         payload["stream"] = True
#         headers = _headers(inp.idempotency_key)
#         async with self._client.stream("POST", url, headers=headers, json=payload) as r:
#             if r.status_code >= 400:
#                 err = (await r.aread()).decode("utf-8", "ignore")
#                 raise GroqSDKError(f"HTTP {r.status_code}: {err}")
#             async for raw in r.aiter_raw():
#                 for line in _sse_iter_lines([raw]):
#                     line = line.strip()
#                     if not line or not line.startswith("data:"):
#                         continue
#                     data = line[5:].strip()
#                     if data == "[DONE]":
#                         return
#                     try:
#                         obj = json.loads(data)
#                     except Exception:
#                         continue
#                     delta, tool_calls, finish = _parse_stream_event(obj)
#                     yield LLMResult(kind="stream_chunk", delta=delta or "", tool_calls=tool_calls, finish_reason=finish, raw=obj)

#     # ---- Responses ----
#     async def response(self, inp: LLMInput) -> LLMResult:
#         url = f"{_api_base()}/v1/responses"
#         payload = _build_response_payload(inp)
#         res = await self._request("POST", url, payload, inp.idempotency_key)
#         text, tool_calls, finish = _extract_responses_text(res)
#         return LLMResult(kind="message", text=text or "", tool_calls=tool_calls, finish_reason=finish, id=res.get("id"), raw=res)

#     async def response_stream(self, inp: LLMInput) -> AsyncIterator[LLMResult]:
#         url = f"{_api_base()}/v1/responses"
#         payload = _build_response_payload(inp)
#         payload["stream"] = True
#         headers = _headers(inp.idempotency_key)
#         async with self._client.stream("POST", url, headers=headers, json=payload) as r:
#             if r.status_code >= 400:
#                 err = (await r.aread()).decode("utf-8", "ignore")
#                 raise GroqSDKError(f"HTTP {r.status_code}: {err}")
#             async for raw in r.aiter_raw():
#                 for line in _sse_iter_lines([raw]):
#                     line = line.strip()
#                     if not line or not line.startswith("data:"):
#                         continue
#                     data = line[5:].strip()
#                     if data == "[DONE]":
#                         return
#                     try:
#                         obj = json.loads(data)
#                     except Exception:
#                         continue
#                     delta, tool_calls, finish = _parse_stream_event(obj)
#                     yield LLMResult(kind="stream_chunk", delta=delta or "", tool_calls=tool_calls, finish_reason=finish, raw=obj)

#     # ---- Batches ----
#     async def batch_create(self, inp: LLMInput) -> LLMResult:
#         if not inp.batch:
#             raise GroqSDKError("batch_create requires LLMInput.batch")
#         url = f"{_api_base()}/v1/batches"
#         res = await self._request("POST", url, inp.batch, inp.idempotency_key)
#         return LLMResult(kind="batch", id=res.get("id"), raw=res)

#     async def batch_retrieve(self, batch_id: str) -> LLMResult:
#         url = f"{_api_base()}/v1/batches/{batch_id}"
#         res = await self._request("GET", url, None, None)
#         return LLMResult(kind="batch", id=res.get("id"), raw=res)

#     async def batch_list(self) -> LLMResult:
#         url = f"{_api_base()}/v1/batches"
#         res = await self._request("GET", url, None, None)
#         return LLMResult(kind="batch", raw=res)

#     async def batch_cancel(self, batch_id: str) -> LLMResult:
#         url = f"{_api_base()}/v1/batches/{batch_id}/cancel"
#         res = await self._request("POST", url, {}, str(uuid.uuid4()))
#         return LLMResult(kind="batch", id=res.get("id"), raw=res)

#     # ---- Low-level request with retries ----
#     async def _request(self, method: str, url: str, json_body: Optional[Dict[str, Any]], idem_key: Optional[str]) -> Dict[str, Any]:
#         backoff = 0.5
#         for attempt in range(self._max_retries):
#             try:
#                 r = await self._client.request(method, url, headers=_headers(idem_key), json=json_body)
#                 if r.status_code >= 400:
#                     if r.status_code in (429, 500, 502, 503, 504) and attempt < self._max_retries - 1:
#                         await _async_sleep(backoff)
#                         backoff = min(backoff * 2, 4.0)
#                         continue
#                     raise GroqSDKError(f"HTTP {r.status_code}: {await r.aread()}")
#                 return r.json()
#             except httpx.TimeoutException as e:
#                 if attempt < self._max_retries - 1:
#                     await _async_sleep(backoff)
#                     backoff = min(backoff * 2, 4.0)
#                     continue
#                 raise GroqSDKError(f"Timeout: {e}") from e

# async def _async_sleep(sec: float) -> None:
#     # tiny inline sleep to avoid importing asyncio at module import time
#     import asyncio as _a
#     await _a.sleep(sec)

# llm_service/client.py
from __future__ import annotations

import json, os, time, uuid, math, random
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, AsyncIterator, Callable, NamedTuple

import httpx
from email.utils import parsedate_to_datetime

__all__ = [
    "GroqLLM",
    "AsyncGroqLLM",
    "LLMInput",
    "LLMResult",
    "Message",
    "llm_input_for_chat",
    "llm_input_for_prompt",
    "GroqSDKError",
    "GroqHTTPError",
]
__version__ = "0.3.0"

# ======================
# Errors / Basic Models
# ======================

class GroqSDKError(RuntimeError):
    pass


class GroqHTTPError(GroqSDKError):
    def __init__(self, status: int, message: str, code: Optional[str], request_id: Optional[str], body: str):
        super().__init__(f"HTTP {status} [{code or 'unknown'}]: {message}")
        self.status = status
        self.code = code
        self.request_id = request_id
        self.body = body


@dataclass(frozen=True)
class Message:
    role: str
    content: str
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    # For assistant messages containing tool calls:
    tool_calls: Optional[List[Dict[str, Any]]] = None


@dataclass(frozen=True)
class LLMInput:
    # Common
    model: Optional[str] = None
    idempotency_key: Optional[str] = None
    # Chat
    messages: Optional[List[Message]] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Any] = None
    response_format: Optional[Dict[str, Any]] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_completion_tokens: Optional[int] = None
    seed: Optional[int] = None
    stop: Optional[Iterable[str]] = None
    parallel_tool_calls: Optional[bool] = None
    reasoning_effort: Optional[str] = None
    reasoning_format: Optional[str] = None
    include_reasoning: Optional[bool] = None
    search_settings: Optional[Dict[str, Any]] = None
    user: Optional[str] = None
    stream: Optional[bool] = None
    # Responses API
    prompt: Optional[str] = None
    instructions: Optional[str] = None
    max_output_tokens: Optional[int] = None
    service_tier: Optional[str] = None
    text: Optional[Dict[str, Any]] = None  # not used by default (vendor-specific)
    # Batches
    batch: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class LLMResult:
    kind: str  # "message" | "stream_chunk" | "batch" | "raw"
    text: str = ""
    delta: str = ""
    tool_calls: Optional[List[Dict[str, Any]]] = None
    finish_reason: Optional[str] = None
    id: Optional[str] = None
    raw: Optional[Dict[str, Any]] = None


def llm_input_for_chat(
    model: str,
    messages: List[Message],
    *,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    max_completion_tokens: Optional[int] = None,
    response_format: Optional[Dict[str, Any]] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[Any] = None,
    stop: Optional[Iterable[str]] = None,
    seed: Optional[int] = None,
    user: Optional[str] = None,
    idempotency_key: Optional[str] = None,
    parallel_tool_calls: Optional[bool] = None,
    reasoning_effort: Optional[str] = None,
    reasoning_format: Optional[str] = None,
    include_reasoning: Optional[bool] = None,
    search_settings: Optional[Dict[str, Any]] = None,
) -> LLMInput:
    return LLMInput(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_completion_tokens=max_completion_tokens,
        response_format=response_format,
        tools=tools,
        tool_choice=tool_choice,
        stop=stop,
        seed=seed,
        user=user,
        idempotency_key=idempotency_key,
        parallel_tool_calls=parallel_tool_calls,
        reasoning_effort=reasoning_effort,
        reasoning_format=reasoning_format,
        include_reasoning=include_reasoning,
        search_settings=search_settings,
    )


def llm_input_for_prompt(
    model: str,
    prompt: str,
    *,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    max_output_tokens: Optional[int] = None,
    instructions: Optional[str] = None,
    user: Optional[str] = None,
    idempotency_key: Optional[str] = None,
    service_tier: Optional[str] = None,
    tool_choice: Optional[Any] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    stream: Optional[bool] = None,
) -> LLMInput:
    return LLMInput(
        model=model,
        prompt=prompt,
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_output_tokens,
        instructions=instructions,
        user=user,
        idempotency_key=idempotency_key,
        service_tier=service_tier,
        tool_choice=tool_choice,
        tools=tools,
        stream=stream,
    )

# ======================
# Config / HTTP helpers
# ======================

def _env(key: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(key)
    return v if (v is not None and v != "") else default

def _api_base() -> str:
    return _env("GROQ_API_BASE", "https://api.groq.com/openai")

def _api_key() -> str:
    key = _env("GROQ_API_KEY")
    if not key:
        raise GroqSDKError("GROQ_API_KEY is not set.")
    return key

def _headers(idempotency_key: Optional[str] = None) -> Dict[str, str]:
    h = {
        "Authorization": f"Bearer {_api_key()}",
        "Content-Type": "application/json",
        "User-Agent": f"codeincubate-llm-sdk/{__version__} (python-httpx)",
    }
    if idempotency_key:
        h["Idempotency-Key"] = idempotency_key
    return h

def _drop_none(d: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in d.items() if v is not None}

def _msg_drop_none(m: Message) -> Dict[str, Any]:
    d = asdict(m)
    # Remove None and empty tool_calls if not present
    d = {k: v for k, v in d.items() if v is not None}
    if "tool_calls" in d and not d["tool_calls"]:
        d.pop("tool_calls", None)
    if "name" in d and not d["name"]:
        d.pop("name", None)
    return d

# Known per-model completion caps (from your matrix)
_MODEL_COMPLETION_CAP: Dict[str, int] = {
    "llama-3.1-8b-instant": 131072,
    "llama-3.3-70b-versatile": 32768,
    "meta-llama/llama-guard-4-12b": 1024,
    "deepseek-r1-distill-llama-70b": 131072,
    "meta-llama/llama-4-maverick-17b-128e-instruct": 8192,
    "meta-llama/llama-4-scout-17b-16e-instruct": 8192,
    "meta-llama/llama-prompt-guard-2-22m": 512,
    "meta-llama/llama-prompt-guard-2-86m": 512,
    "moonshotai/kimi-k2-instruct": 16384,
    # whisper models: N/A
}

def _cap_tokens(model: Optional[str], key: str, val: Optional[int]) -> Optional[int]:
    if val is None or not model:
        return val
    cap = _MODEL_COMPLETION_CAP.get(model)
    if cap is None:
        return val
    return min(val, cap)

# =========================
# Payload build / extraction
# =========================

def _build_chat_payload(inp: LLMInput) -> Dict[str, Any]:
    if not inp.model or not inp.messages:
        raise GroqSDKError("Chat payload requires model and messages.")
    payload: Dict[str, Any] = {
        "model": inp.model,
        "messages": [_msg_drop_none(m) for m in inp.messages],
        "temperature": inp.temperature,
        "top_p": inp.top_p,
        "max_completion_tokens": _cap_tokens(inp.model, "max_completion_tokens", inp.max_completion_tokens),
        "seed": inp.seed,
        "stop": list(inp.stop) if inp.stop else None,
        "response_format": inp.response_format,
        "tools": inp.tools,
        "tool_choice": inp.tool_choice,
        "parallel_tool_calls": inp.parallel_tool_calls,
        "reasoning_effort": inp.reasoning_effort,
        "reasoning_format": inp.reasoning_format,
        "include_reasoning": inp.include_reasoning,
        "search_settings": inp.search_settings,
        "user": inp.user,
    }
    return _drop_none(payload)

def _build_response_payload(inp: LLMInput) -> Dict[str, Any]:
    if not inp.model or inp.prompt is None:
        raise GroqSDKError("Responses payload requires model and prompt.")
    payload: Dict[str, Any] = {
        "model": inp.model,
        "input": inp.prompt,
        "temperature": inp.temperature,
        "top_p": inp.top_p,
        "max_output_tokens": _cap_tokens(inp.model, "max_output_tokens", inp.max_output_tokens),
        "instructions": inp.instructions,
        "user": inp.user,
        "tool_choice": inp.tool_choice,
        "tools": inp.tools,
        # Do NOT default service_tier=auto (orgs may not have access)
        "service_tier": inp.service_tier,
        # Avoid vendor-specific text formatting by default
        # "text": inp.text,
    }
    return _drop_none(payload)

def _extract_chat_text(res: Dict[str, Any]) -> Tuple[str, Optional[List[Dict[str, Any]]], Optional[str]]:
    text = ""
    tool_calls = None
    finish_reason = None
    choices = res.get("choices") or []
    if choices:
        c0 = choices[0] or {}
        finish_reason = c0.get("finish_reason")
        msg = c0.get("message") or {}
        text = msg.get("content") or ""
        tool_calls = msg.get("tool_calls")
    return text, tool_calls, finish_reason

def _extract_responses_text(res: Dict[str, Any]) -> Tuple[str, Optional[List[Dict[str, Any]]], Optional[str]]:
    # 1) Try chat-like shape first
    if "choices" in res:
        return _extract_chat_text(res)

    text = ""
    tool_calls = None
    finish_reason = res.get("finish_reason") or res.get("status")

    # 2) Common Groq Responses shapes
    # a) Flat string field
    if isinstance(res.get("output_text"), str) and res["output_text"]:
        return res["output_text"], tool_calls, finish_reason

    # b) output list with text parts
    out = res.get("output")
    if isinstance(out, list) and out:
        parts: List[str] = []

        def pull_text(obj: Dict[str, Any]) -> None:
            t = obj.get("type")
            if t == "output_text" and isinstance(obj.get("text"), str):
                parts.append(obj["text"])
            # nested content arrays
            if "content" in obj and isinstance(obj["content"], list):
                for c in obj["content"]:
                    if isinstance(c, dict) and c.get("type") in ("output_text", "output_text_delta"):
                        if isinstance(c.get("text"), str):
                            parts.append(c["text"])

        for seg in out:
            if isinstance(seg, dict):
                pull_text(seg)

        if parts:
            text = "".join(parts)

    return text, tool_calls, finish_reason

# ===============
# SSE line parsers
# ===============

def _iter_sse_lines_from_chunks(chunks: Iterable[bytes]) -> Iterable[str]:
    """
    Robustly yield UTF-8 lines from an iterable of byte chunks.
    Maintains a shared buffer across chunks (important for SSE over TCP frames).
    """
    buf = b""
    for chunk in chunks:
        if not chunk:
            continue
        buf += chunk
        while True:
            i = buf.find(b"\n")
            if i == -1:
                break
            line, buf = buf[:i], buf[i+1:]
            yield line.decode("utf-8", "ignore")
    # SSE lines are newline-terminated; trailing partial is kept (discarded).

def _parse_stream_event(data_json: Dict[str, Any]) -> Tuple[str, Optional[List[Dict[str, Any]]], Optional[str]]:
    """
    Extract streaming delta text and tool_calls from either Chat or Responses SSE shapes.
    Returns (delta_text, tool_calls, finish_reason)
    """
    # 1) OpenAI/Chat-style SSE
    if "choices" in data_json:
        ch = data_json["choices"][0] if data_json.get("choices") else {}
        delta = ch.get("delta") or {}
        if isinstance(delta, dict):
            return delta.get("content", "") or "", delta.get("tool_calls"), ch.get("finish_reason")
        # fallback
        return (ch.get("text") or ""), None, ch.get("finish_reason")

    # 2) Groq Responses-style SSE
    t = data_json.get("type")

    # Incremental text deltas
    if t == "response.output_text.delta":
        d = data_json.get("delta")
        if isinstance(d, dict):
            return d.get("text", "") or "", None, None
        if isinstance(d, str):
            return d, None, None
        return "", None, None

    # Sometimes a full text lands as a response update
    if t in {"response.output_text", "response.completed"}:
        # Some shapes place text directly in 'response'
        resp = data_json.get("response")
        if isinstance(resp, dict) and isinstance(resp.get("output_text"), str):
            return resp["output_text"], None, resp.get("finish_reason") or "stop"
        # Or as a top-level field
        if isinstance(data_json.get("output_text"), str):
            return data_json["output_text"], None, "stop"
        return "", None, "stop"

    # Fallback: nested response container with text deltas
    resp = data_json.get("response")
    if isinstance(resp, dict):
        if isinstance(resp.get("output_text_delta"), str):
            return resp["output_text_delta"], None, None
        if isinstance(resp.get("output_text"), str):
            return resp["output_text"], None, resp.get("finish_reason")

    # Unknown event; no text
    return "", None, None

# ===============
# Retry helpers
# ===============

def _retry_sleep_seconds(backoff: float, retry_after_header: Optional[str]) -> float:
    # If Retry-After present, prefer it
    if retry_after_header:
        try:
            # seconds form
            return max(0.0, float(retry_after_header))
        except ValueError:
            # HTTP-date form
            try:
                dt = parsedate_to_datetime(retry_after_header)
                delta = (dt - getattr(dt, "now")()).total_seconds()  # naive best-effort
                return max(0.0, delta)
            except Exception:
                pass
    # full-jitter on current backoff window
    return random.uniform(0.0, backoff)

def _mk_http_error(r: httpx.Response) -> GroqHTTPError:
    rid = r.headers.get("x-request-id") or r.headers.get("x-requestid")
    code = None
    message = r.text
    body = r.text
    try:
        j = r.json()
        if isinstance(j, dict):
            err = j.get("error")
            if isinstance(err, dict):
                message = err.get("message") or message
                code = err.get("type") or err.get("code")
    except Exception:
        pass
    return GroqHTTPError(r.status_code, message, code, rid, body)

# ==============
# Observability
# ==============

class HttpEvent(NamedTuple):
    method: str
    url: str
    status: int
    duration_s: float
    bytes: int
    attempt: int

# =================
# Sync HTTP client
# =================

class GroqLLM:
    def __init__(
        self,
        timeout: float | httpx.Timeout = httpx.Timeout(connect=5.0, read=60.0, write=30.0, pool=5.0),
        max_retries: int = 3,
        max_retry_elapsed: float = 30.0,
        on_event: Optional[Callable[[HttpEvent], None]] = None,
    ):
        if isinstance(timeout, (int, float)):
            timeout = httpx.Timeout(float(timeout))
        # Disable httpx internal retries; we own the policy
        self._client = httpx.Client(http2=True, timeout=timeout, transport=httpx.HTTPTransport(retries=0))
        self._max_retries = max_retries
        self._max_retry_elapsed = max_retry_elapsed
        self._on_event = on_event

    def __enter__(self) -> "GroqLLM":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    # ---- Chat ----
    def chat(self, inp: LLMInput) -> LLMResult:
        url = f"{_api_base()}/v1/chat/completions"
        payload = _build_chat_payload(inp)
        # default an idempotency key for non-stream API
        idem = inp.idempotency_key or str(uuid.uuid4())
        res = self._request("POST", url, payload, idem)
        text, tool_calls, finish = _extract_chat_text(res)
        return LLMResult(kind="message", text=text or "", tool_calls=tool_calls, finish_reason=finish, id=res.get("id"), raw=res)

    def chat_stream(self, inp: LLMInput) -> Iterator[LLMResult]:
        url = f"{_api_base()}/v1/chat/completions"
        payload = _build_chat_payload(inp)
        payload["stream"] = True
        headers = _headers(inp.idempotency_key)  # caller decides idempotency for streams
        t0 = time.monotonic()
        with self._client.stream("POST", url, headers=headers, json=payload) as r:
            if r.status_code >= 400:
                err = r.read().decode("utf-8", "ignore")
                raise _mk_http_error(r)
            attempt = 0
            try:
                for line in _iter_sse_lines_from_chunks(r.iter_raw()):
                    line = line.strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                    except Exception:
                        continue
                    delta, tool_calls, finish = _parse_stream_event(obj)
                    yield LLMResult(kind="stream_chunk", delta=delta or "", tool_calls=tool_calls, finish_reason=finish, raw=obj)
            finally:
                if self._on_event:
                    dur = time.monotonic() - t0
                    # Can't know bytes easily without wrapping; use 0 for streams unless needed
                    self._on_event(HttpEvent("POST", url, r.status_code, dur, 0, attempt))

    # ---- Responses ----
    def response(self, inp: LLMInput) -> LLMResult:
        url = f"{_api_base()}/v1/responses"
        payload = _build_response_payload(inp)
        idem = inp.idempotency_key or str(uuid.uuid4())
        res = self._request("POST", url, payload, idem)
        text, tool_calls, finish = _extract_responses_text(res)
        return LLMResult(kind="message", text=text or "", tool_calls=tool_calls, finish_reason=finish, id=res.get("id"), raw=res)

    def response_stream(self, inp: LLMInput) -> Iterator[LLMResult]:
        url = f"{_api_base()}/v1/responses"
        payload = _build_response_payload(inp)
        payload["stream"] = True
        headers = _headers(inp.idempotency_key)
        t0 = time.monotonic()
        with self._client.stream("POST", url, headers=headers, json=payload) as r:
            if r.status_code >= 400:
                err = r.read().decode("utf-8", "ignore")
                raise _mk_http_error(r)
            attempt = 0
            try:
                for line in _iter_sse_lines_from_chunks(r.iter_raw()):
                    line = line.strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                    except Exception:
                        continue
                    delta, tool_calls, finish = _parse_stream_event(obj)
                    yield LLMResult(kind="stream_chunk", delta=delta or "", tool_calls=tool_calls, finish_reason=finish, raw=obj)
            finally:
                if self._on_event:
                    dur = time.monotonic() - t0
                    self._on_event(HttpEvent("POST", url, r.status_code, dur, 0, attempt))

    # ---- Batches ----
    def batch_create(self, inp: LLMInput) -> LLMResult:
        if not inp.batch:
            raise GroqSDKError("batch_create requires LLMInput.batch")
        url = f"{_api_base()}/v1/batches"
        res = self._request("POST", url, inp.batch, inp.idempotency_key or str(uuid.uuid4()))
        return LLMResult(kind="batch", id=res.get("id"), raw=res)

    def batch_retrieve(self, batch_id: str) -> LLMResult:
        url = f"{_api_base()}/v1/batches/{batch_id}"
        res = self._request("GET", url, None, None)
        return LLMResult(kind="batch", id=res.get("id"), raw=res)

    def batch_list(self) -> LLMResult:
        url = f"{_api_base()}/v1/batches"
        res = self._request("GET", url, None, None)
        return LLMResult(kind="batch", raw=res)

    def batch_cancel(self, batch_id: str) -> LLMResult:
        url = f"{_api_base()}/v1/batches/{batch_id}/cancel"
        res = self._request("POST", url, {}, str(uuid.uuid4()))
        return LLMResult(kind="batch", id=res.get("id"), raw=res)

    # ---- Low-level request with retries ----
    def _request(self, method: str, url: str, json_body: Optional[Dict[str, Any]], idem_key: Optional[str]) -> Dict[str, Any]:
        backoff = 0.5
        started = time.monotonic()
        attempt = 0
        last_exc: Optional[Exception] = None
        while attempt < self._max_retries:
            attempt += 1
            t0 = time.monotonic()
            try:
                r = self._client.request(method, url, headers=_headers(idem_key), json=json_body)
                dur = time.monotonic() - t0
                if r.status_code >= 400:
                    if r.status_code in (429, 500, 502, 503, 504) and attempt < self._max_retries:
                        sleep_s = _retry_sleep_seconds(backoff, r.headers.get("Retry-After"))
                        if self._on_event:
                            self._on_event(HttpEvent(method, url, r.status_code, dur, len(r.content or b""), attempt))
                        if time.monotonic() - started + sleep_s > self._max_retry_elapsed:
                            raise _mk_http_error(r)
                        time.sleep(sleep_s)
                        backoff = min(backoff * 2, 8.0)
                        continue
                    raise _mk_http_error(r)
                # success
                if self._on_event:
                    self._on_event(HttpEvent(method, url, r.status_code, dur, len(r.content or b""), attempt))
                try:
                    return r.json()
                except Exception:
                    # Non-JSON success; return raw
                    return {"raw": r.text}
            except (httpx.TimeoutException, httpx.TransportError) as e:
                last_exc = e
                if attempt < self._max_retries:
                    sleep_s = _retry_sleep_seconds(backoff, None)
                    if time.monotonic() - started + sleep_s > self._max_retry_elapsed:
                        break
                    time.sleep(sleep_s)
                    backoff = min(backoff * 2, 8.0)
                    continue
                raise GroqSDKError(f"Network error: {e}") from e
        # Exhausted retry budget
        if isinstance(last_exc, Exception):
            raise GroqSDKError(f"Retry budget exceeded: {last_exc}") from last_exc
        raise GroqSDKError("Retry budget exceeded")

# ===================
# Async HTTP client
# ===================

async def _async_sleep(sec: float) -> None:
    # tiny inline sleep to avoid importing asyncio at module import time
    import asyncio as _a
    await _a.sleep(sec)

class AsyncGroqLLM:
    def __init__(
        self,
        timeout: float | httpx.Timeout = httpx.Timeout(connect=5.0, read=60.0, write=30.0, pool=5.0),
        max_retries: int = 3,
        max_retry_elapsed: float = 30.0,
        on_event: Optional[Callable[[HttpEvent], None]] = None,
    ):
        if isinstance(timeout, (int, float)):
            timeout = httpx.Timeout(float(timeout))
        self._client = httpx.AsyncClient(http2=True, timeout=timeout, transport=httpx.AsyncHTTPTransport(retries=0))
        self._max_retries = max_retries
        self._max_retry_elapsed = max_retry_elapsed
        self._on_event = on_event

    async def __aenter__(self) -> "AsyncGroqLLM":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def close(self) -> None:
        await self._client.aclose()

    # ---- Chat ----
    async def chat(self, inp: LLMInput) -> LLMResult:
        url = f"{_api_base()}/v1/chat/completions"
        payload = _build_chat_payload(inp)
        idem = inp.idempotency_key or str(uuid.uuid4())
        res = await self._request("POST", url, payload, idem)
        text, tool_calls, finish = _extract_chat_text(res)
        return LLMResult(kind="message", text=text or "", tool_calls=tool_calls, finish_reason=finish, id=res.get("id"), raw=res)

    async def chat_stream(self, inp: LLMInput) -> AsyncIterator[LLMResult]:
        url = f"{_api_base()}/v1/chat/completions"
        payload = _build_chat_payload(inp)
        payload["stream"] = True
        headers = _headers(inp.idempotency_key)
        t0 = time.monotonic()
        async with self._client.stream("POST", url, headers=headers, json=payload) as r:
            if r.status_code >= 400:
                err = (await r.aread()).decode("utf-8", "ignore")
                raise _mk_http_error(r)
            attempt = 0
            # shared buffer across chunks
            buf = b""
            async for raw in r.aiter_raw():
                if not raw:
                    continue
                buf += raw
                while True:
                    i = buf.find(b"\n")
                    if i == -1:
                        break
                    line, buf = buf[:i], buf[i+1:]
                    s = line.decode("utf-8", "ignore").strip()
                    if not s or not s.startswith("data:"):
                        continue
                    data = s[5:].strip()
                    if data == "[DONE]":
                        if self._on_event:
                            dur = time.monotonic() - t0
                            self._on_event(HttpEvent("POST", url, r.status_code, dur, 0, attempt))
                        return
                    try:
                        obj = json.loads(data)
                    except Exception:
                        continue
                    delta, tool_calls, finish = _parse_stream_event(obj)
                    yield LLMResult(kind="stream_chunk", delta=delta or "", tool_calls=tool_calls, finish_reason=finish, raw=obj)

    # ---- Responses ----
    async def response(self, inp: LLMInput) -> LLMResult:
        url = f"{_api_base()}/v1/responses"
        payload = _build_response_payload(inp)
        idem = inp.idempotency_key or str(uuid.uuid4())
        res = await self._request("POST", url, payload, idem)
        text, tool_calls, finish = _extract_responses_text(res)
        return LLMResult(kind="message", text=text or "", tool_calls=tool_calls, finish_reason=finish, id=res.get("id"), raw=res)

    async def response_stream(self, inp: LLMInput) -> AsyncIterator[LLMResult]:
        url = f"{_api_base()}/v1/responses"
        payload = _build_response_payload(inp)
        payload["stream"] = True
        headers = _headers(inp.idempotency_key)
        t0 = time.monotonic()
        async with self._client.stream("POST", url, headers=headers, json=payload) as r:
            if r.status_code >= 400:
                err = (await r.aread()).decode("utf-8", "ignore")
                raise _mk_http_error(r)
            attempt = 0
            buf = b""
            async for raw in r.aiter_raw():
                if not raw:
                    continue
                buf += raw
                while True:
                    i = buf.find(b"\n")
                    if i == -1:
                        break
                    line, buf = buf[:i], buf[i+1:]
                    s = line.decode("utf-8", "ignore").strip()
                    if not s or not s.startswith("data:"):
                        continue
                    data = s[5:].strip()
                    if data == "[DONE]":
                        if self._on_event:
                            dur = time.monotonic() - t0
                            self._on_event(HttpEvent("POST", url, r.status_code, dur, 0, attempt))
                        return
                    try:
                        obj = json.loads(data)
                    except Exception:
                        continue
                    delta, tool_calls, finish = _parse_stream_event(obj)
                    yield LLMResult(kind="stream_chunk", delta=delta or "", tool_calls=tool_calls, finish_reason=finish, raw=obj)

    # ---- Batches ----
    async def batch_create(self, inp: LLMInput) -> LLMResult:
        if not inp.batch:
            raise GroqSDKError("batch_create requires LLMInput.batch")
        url = f"{_api_base()}/v1/batches"
        res = await self._request("POST", url, inp.batch, inp.idempotency_key or str(uuid.uuid4()))
        return LLMResult(kind="batch", id=res.get("id"), raw=res)

    async def batch_retrieve(self, batch_id: str) -> LLMResult:
        url = f"{_api_base()}/v1/batches/{batch_id}"
        res = await self._request("GET", url, None, None)
        return LLMResult(kind="batch", id=res.get("id"), raw=res)

    async def batch_list(self) -> LLMResult:
        url = f"{_api_base()}/v1/batches"
        res = await self._request("GET", url, None, None)
        return LLMResult(kind="batch", raw=res)

    async def batch_cancel(self, batch_id: str) -> LLMResult:
        url = f"{_api_base()}/v1/batches/{batch_id}/cancel"
        res = await self._request("POST", url, {}, str(uuid.uuid4()))
        return LLMResult(kind="batch", id=res.get("id"), raw=res)

    # ---- Low-level request with retries ----
    async def _request(self, method: str, url: str, json_body: Optional[Dict[str, Any]], idem_key: Optional[str]) -> Dict[str, Any]:
        backoff = 0.5
        started = time.monotonic()
        attempt = 0
        last_exc: Optional[Exception] = None
        while attempt < self._max_retries:
            attempt += 1
            t0 = time.monotonic()
            try:
                r = await self._client.request(method, url, headers=_headers(idem_key), json=json_body)
                dur = time.monotonic() - t0
                if r.status_code >= 400:
                    if r.status_code in (429, 500, 502, 503, 504) and attempt < self._max_retries:
                        sleep_s = _retry_sleep_seconds(backoff, r.headers.get("Retry-After"))
                        if self._on_event:
                            content = await r.aread()
                            self._on_event(HttpEvent(method, url, r.status_code, dur, len(content or b""), attempt))
                        if time.monotonic() - started + sleep_s > self._max_retry_elapsed:
                            raise _mk_http_error(r)
                        await _async_sleep(sleep_s)
                        backoff = min(backoff * 2, 8.0)
                        continue
                    raise _mk_http_error(r)
                if self._on_event:
                    content = await r.aread()
                    self._on_event(HttpEvent(method, url, r.status_code, dur, len(content or b""), attempt))
                    # reset the stream so .json() can work afterwards
                    r = httpx.Response(r.status_code, headers=r.headers, request=r.request, content=content)
                try:
                    return r.json()
                except Exception:
                    return {"raw": (await r.aread()).decode("utf-8", "ignore")}
            except (httpx.TimeoutException, httpx.TransportError) as e:
                last_exc = e
                if attempt < self._max_retries:
                    sleep_s = _retry_sleep_seconds(backoff, None)
                    if time.monotonic() - started + sleep_s > self._max_retry_elapsed:
                        break
                    await _async_sleep(sleep_s)
                    backoff = min(backoff * 2, 8.0)
                    continue
                raise GroqSDKError(f"Network error: {e}") from e
        if isinstance(last_exc, Exception):
            raise GroqSDKError(f"Retry budget exceeded: {last_exc}") from last_exc
        raise GroqSDKError("Retry budget exceeded")

