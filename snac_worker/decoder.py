from snac import SNAC
import numpy as np
import torch
import asyncio
import os
import re
import time

class TTSDecoderProfiler:
    def __init__(self):
        self.timings = {}
        self.last_tick = time.perf_counter()

    def tick(self, name):
        now = time.perf_counter()
        self.timings[name] = self.timings.get(name, 0) + (now - self.last_tick)
        self.last_tick = now

# SNAC Model Loading (at module level for efficiency)
snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()
snac_device = os.environ.get("SNAC_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
if snac_device == "cuda":
    snac_model = snac_model.to(dtype=torch.float16, device=snac_device)
else:
    snac_model = snac_model.to(snac_device)



MF = 7
PRIMER_MF = int(os.environ.get("SNAC_PRIMER_MF", "2"))      # first chunk uses 2*7=14 tokens (set "1" for even faster)
STEADY_MF = int(os.environ.get("SNAC_STEADY_MF", "4"))      # later chunks use 4*7=28 (original behavior)
SLICE_START = int(os.environ.get("SNAC_SLICE_START", "2048"))
SLICE_LEN   = int(os.environ.get("SNAC_SLICE_LEN",   "2048"))  # keep 2048 (~85ms @ 24k) for stable overlap-add

torch.set_grad_enabled(False)
torch.backends.cudnn.benchmark = True

                
def convert_to_audio(multiframe, count):
    if len(multiframe) < MF: return
    codes_0 = torch.zeros(len(multiframe) // 7, device=snac_device, dtype=torch.int32)
    codes_1 = torch.zeros(2 * (len(multiframe) // 7), device=snac_device, dtype=torch.int32)
    codes_2 = torch.zeros(4 * (len(multiframe) // 7), device=snac_device, dtype=torch.int32)
    
    for j in range(len(multiframe) // 7):
        i = 7 * j
        codes_0[j] = multiframe[i]
        codes_1[2*j], codes_1[2*j+1] = multiframe[i+1], multiframe[i+4]
        codes_2[4*j:4*j+4] = torch.tensor(multiframe[i+2:i+4] + multiframe[i+5:i+7], device=snac_device)

    codes = [codes_0.unsqueeze(0), codes_1.unsqueeze(0), codes_2.unsqueeze(0)]
    if torch.any((codes[0] < 0) | (codes[0] > 4096)) or \
       torch.any((codes[1] < 0) | (codes[1] > 4096)) or \
       torch.any((codes[2] < 0) | (codes[2] > 4096)):
        print("WARNING: [convert_to_audio] Token ID out of valid range [0, 4096]. Skipping frame.")
        return None

    with torch.inference_mode(), torch.cuda.amp.autocast(enabled=(snac_device=='cuda'), dtype=torch.float16):
        audio_hat = snac_model.decode(codes)
    audio_slice = audio_hat[:, :, SLICE_START:SLICE_START+SLICE_LEN]
    return (audio_slice.detach().cpu().numpy() * 32767).astype(np.int16).tobytes()

def turn_token_into_ids(token_string, current_token_count):
    """Finds all custom tokens, converts them to IDs, and returns a list."""
    ids = []
    found_tokens = re.findall(r"<custom_token_(\d+)>", token_string)
    if not found_tokens:
        return []

    for i, number_str in enumerate(found_tokens):
        try:
            index = current_token_count + i
            raw_token_val = int(number_str)
            # This is the core mapping formula
            token_id = raw_token_val - 10 - ((index % 7) * 4096)
            
            # Enhanced Debugging Print
            # print(f"DEBUG: [ID Calc] Raw: {raw_token_val}, Index: {index}, Modulo: {index % 7}, Final ID: {token_id}")

            if 0 <= token_id <= 4096:
                ids.append(token_id)
            else:
                # This will tell you exactly which token failed and why
                print(f"WARNING: [ID Invalid] Raw: {raw_token_val}, Index: {index} -> Final ID: {token_id} is out of range.")

        except ValueError:
            continue
    return ids
  
async def tokens_decoder(token_gen, profiler: TTSDecoderProfiler):
    buffer, count = [], 0
    profiler.tick("start_decode_loop")

    async for token_sim in token_gen:
        profiler.tick("received_token_text")
        token_ids = turn_token_into_ids(token_sim, count)
        profiler.tick("parsed_token_ids")
        if not token_ids: 
            continue

        for token in token_ids:
            buffer.append(token)
            count += 1

            if count % MF != 0:
                continue

            min_needed = PRIMER_MF * MF
            if count < min_needed:
                continue

            # Early ramp: use smaller window for first chunk(s), then steady state.
            decode_mf = PRIMER_MF if count < STEADY_MF * MF else STEADY_MF
            buffer_to_proc = buffer[-decode_mf * MF:]

            profiler.tick("pre_convert_audio")
            audio_samples = convert_to_audio(buffer_to_proc, count)
            profiler.tick("post_convert_audio")

            if audio_samples:
                yield audio_samples
                profiler.tick("yielded_audio")
