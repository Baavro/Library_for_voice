from huggingface_hub import snapshot_download
import os

model_repo = "unsloth/orpheus-3b-0.1-ft-unsloth-bnb-4bit"
local_dir = "snac_worker/Orpheus/orpheus_unsloth/hf_cache"

os.makedirs(local_dir, exist_ok=True)

snapshot_download(
    repo_id=model_repo,
    local_dir=local_dir,
    local_dir_use_symlinks=False,
)
