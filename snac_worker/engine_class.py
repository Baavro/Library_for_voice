import asyncio
import torch
import os
import time
import uuid
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from transformers import AutoTokenizer
from .decoder import tokens_decoder # Use the async decoder
from .decoder import TTSDecoderProfiler

class OrpheusModel:
    def __init__(
        self,
        model_path: str = "/home/ubuntu/Orpheus/Orpheus/orpheus_unsloth/hf_cache",
        *,
        tokenizer_path: str | None = None,
        quantization: str | None = None,          # "awq" | "gptq" | "bnb" | None
        dtype=torch.float16,                      # activations dtype (fp16/bf16)
        max_model_len: int = 512,
        gpu_memory_utilization: float = 0.50,
        enforce_eager: bool = True,
        max_num_seqs: int = 64,
        tensor_parallel_size: int | None = None,
        kv_cache_dtype: str | None = None,        # e.g., "fp8" or "int8" (if supported by your vLLM)
        swap_space: int | None = None,            # bytes, e.g., 8<<30
        trust_remote_code: bool = True,
        **engine_kwargs,
    ):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path or model_path
        self.quantization = quantization
        self.dtype = dtype
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.enforce_eager = enforce_eager
        self.max_num_seqs = max_num_seqs
        self.tensor_parallel_size = tensor_parallel_size
        self.kv_cache_dtype = kv_cache_dtype
        self.swap_space = swap_space
        self.engine_kwargs = engine_kwargs

        self.available_voices = ["zoe", "zac", "jess", "leo", "mia", "julia", "leah"]

        self.tokenizer = self._load_tokenizer(self.tokenizer_path, trust_remote_code=trust_remote_code)
        self.engine = self._setup_engine(trust_remote_code=trust_remote_code)


    def _load_tokenizer(self, tokenizer_path, trust_remote_code=True):
        try:
            if os.path.isdir(tokenizer_path):
                return AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True, trust_remote_code=trust_remote_code)
            else:
                return AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=trust_remote_code)
        except Exception as e:
            print(f"Error loading tokenizer: {e}\nFalling back to default tokenizer")
            return AutoTokenizer.from_pretrained("gpt2")
        

    def _setup_engine(self, trust_remote_code=True):
        # Build args dict in a version-safe way (only pass keys that have values)
        args = {
            "model": self.model_path,
            "tokenizer": self.tokenizer_path,
            "dtype": self.dtype,  # activations dtype
            "max_model_len": self.max_model_len,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "enforce_eager": self.enforce_eager,
            "max_num_seqs": self.max_num_seqs,
            "trust_remote_code": trust_remote_code,
        }

        if self.quantization:            # e.g., "awq", "gptq", "bnb"
            args["quantization"] = self.quantization
        if self.tensor_parallel_size:
            args["tensor_parallel_size"] = self.tensor_parallel_size
        if self.swap_space is not None:
            args["swap_space"] = self.swap_space

        # kv_cache_dtype is only available on newer vLLM; pass it carefully
        if self.kv_cache_dtype:
            try:
                engine_args = AsyncEngineArgs(**args, kv_cache_dtype=self.kv_cache_dtype, **self.engine_kwargs)
            except TypeError:
                print("[OrpheusModel] kv_cache_dtype not supported by this vLLM version; ignoring.")
                engine_args = AsyncEngineArgs(**args, **self.engine_kwargs)
        else:
            engine_args = AsyncEngineArgs(**args, **self.engine_kwargs)

        return AsyncLLMEngine.from_engine_args(engine_args)

    
    def _format_prompt(self, prompt, voice="tara", model_type="larger"):
        # This function is unchanged
        adapted_prompt = f"{voice}: {prompt}" if voice else prompt
        prompt_tokens = self.tokenizer(adapted_prompt, return_tensors="pt")
        start_token = torch.tensor([[128259]], dtype=torch.int64)
        end_tokens = torch.tensor([[128009, 128260, 128261, 128257]], dtype=torch.int64)
        all_input_ids = torch.cat([start_token, prompt_tokens.input_ids, end_tokens], dim=1)
        return self.tokenizer.decode(all_input_ids[0])

    async def generate_tokens_async(self, prompt, voice=None, request_id="req-001", **kwargs):
        """Yields ONLY the newly generated text (delta) from vLLM."""
        prompt_string = self._format_prompt(prompt, voice)
        sampling_params = SamplingParams(
            temperature=kwargs.get("temperature", 0.4),
            top_p=kwargs.get("top_p", 0.9),
            max_tokens=kwargs.get("max_tokens", 2000),
            stop_token_ids=kwargs.get("stop_token_ids", [128258]),
            repetition_penalty=kwargs.get("repetition_penalty", 1.1),
        )
        print(f"DEBUG: [generate_tokens_async] Starting vLLM generation for request {request_id}")
        results_generator = self.engine.generate(prompt_string, sampling_params, request_id)

        previous_text = ""
        token_count = 0
        async for result in results_generator:
            current_text = result.outputs[0].text
            newly_generated_text = current_text[len(previous_text):]
            if newly_generated_text:
                yield newly_generated_text
                previous_text = current_text
                token_count += 1

        print(f"DEBUG: [generate_tokens_async] Finished vLLM generation for request {request_id}. Total steps: {token_count}")

    async def generate_speech_async(self, **kwargs):
        """Orchestrates async token generation and async audio decoding with deep profiling."""
        profiler = TTSDecoderProfiler()
        request_id = kwargs.get("request_id", f"tts-{uuid.uuid4()}")
        kwargs["request_id"] = request_id

        print(f"  [TTS Backend] Starting for request {request_id}")
        profiler.tick("init")

        token_generator = self.generate_tokens_async(**kwargs)
        profiler.tick("token_generator_created")

        audio_chunk_generator = tokens_decoder(token_generator, profiler)

        chunk_count = 0
        async for audio_chunk in audio_chunk_generator:
            yield audio_chunk
            chunk_count += 1

        profiler.tick("finished_yielding")
        print(f"--- 🔬 TTS DEEP DIVE for Request {request_id} ---")
        total_time = sum(profiler.timings.values())
        for name, timing in profiler.timings.items():
            percentage = (timing / total_time) * 100 if total_time > 0 else 0
            print(f"  - {name:<22}: {timing:.4f}s ({percentage:.1f}%)")
        print("-------------------------------------------------")
