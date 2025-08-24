# quantize_orpheus_awq.py
# Usage:
#   python quantize_orpheus_awq.py \
#     --base canopylabs/orpheus-3b-0.1-ft \
#     --calib calib_prompts.json \
#     --out ./orpheus-3b-0.1-ft-awq-w4g128-zp
#
# Produces a vLLM-friendly AWQ 4-bit checkpoint.

import argparse, json, random
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

def load_calib_texts(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Keep only non-empty strings
    texts = [s.strip() for s in data if isinstance(s, str) and s.strip()]
    # (Optional) de-dup and shuffle
    texts = list(dict.fromkeys(texts))
    random.shuffle(texts)
    return texts

def maybe_lengthen(texts, tokenizer, target_tokens=128):
    """
    AWQ works fine with short prompts, but calibration is a bit stabler if
    each sample has ~128-256 tokens. This lightly repeats content to reach the target.
    Disable if you prefer original length.
    """
    out = []
    for t in texts:
        ids = tokenizer(t, add_special_tokens=True)["input_ids"]
        if len(ids) >= target_tokens:
            out.append(t)
            continue
        # naive lengthening: repeat with a separator so we don't create a single long token
        reps = max(1, target_tokens // max(1, len(ids)))
        out.append((" " + t).join([""] * reps) or t)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", help="Base HF model (e.g., canopylabs/orpheus-3b-0.1-ft)", default="canopylabs/orpheus-3b-0.1-ft")
    ap.add_argument("--calib", help="Path to JSON array of strings",default="orpheus_calib_texts_prod.json")
    ap.add_argument("--out", help="Output dir for quantized model",default="./orpheus-3b-0.1-ft-awq-w4g128-zp")
    ap.add_argument("--gsize", type=int, default=128, help="AWQ quant group size (default: 128)")
    ap.add_argument("--wbit", type=int, default=4, help="AWQ weight bits (default: 4)")
    ap.add_argument("--no_zero_point", action="store_true", help="Disable zero-point (defaults to enabled)")
    ap.add_argument("--no_lengthen", action="store_true", help="Use texts as-is (no token lengthening)")
    args = ap.parse_args()

    print("→ Loading model/tokenizer...")
    model = AutoAWQForCausalLM.from_pretrained(args.base, trust_remote_code=True)
    tok   = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)

    print(f"→ Loading calibration texts from {args.calib} ...")
    calib_texts = load_calib_texts(args.calib)
    if not args.no_lengthen:
        calib_texts = maybe_lengthen(calib_texts, tok, target_tokens=128)

    quant_config = {
        "w_bit": args.wbit,
        "q_group_size": args.gsize,
        "zero_point": (not args.no_zero_point),
        "version": "GEMM",      # vLLM-friendly kernels
    }

    print(f"→ Quantizing with {quant_config} on {len(calib_texts)} samples ...")
    model.quantize(tok, quant_config=quant_config, calib_data=calib_texts)

    print(f"→ Saving to {args.out} ...")
    model.save_quantized(args.out)
    tok.save_pretrained(args.out)
    print("✔ Done.")

if __name__ == "__main__":
    main()
