"""Inspect the actual LoRA setup at training time:
- target_modules from the warm-start adapter
- saved alpha (deltacoder's training alpha)
- effective scaling = alpha / rank
- count of trainable parameters
- which modules have requires_grad=True
"""
import json, os
from pathlib import Path

# Try to find deltacoder's adapter_config.json from the HF cache
home = Path.home()
hf_cache = home / ".cache/huggingface/hub"
candidates = list(hf_cache.glob("**/adapter_config.json"))
print(f"Found {len(candidates)} adapter_config.json files in HF cache")
for p in candidates:
    name = str(p)
    if "DeltaCoder" in name or "deltacoder" in name.lower():
        print(f"\n=== {p} ===")
        cfg = json.loads(p.read_text())
        for k in ['r','lora_alpha','target_modules','lora_dropout','base_model_name_or_path','task_type','peft_type']:
            print(f"  {k}: {cfg.get(k)}")
        # Compute scaling
        r = cfg.get('r')
        a = cfg.get('lora_alpha')
        if r and a:
            print(f"  → scaling (alpha/r) = {a/r:.3f}")

# Also check huggingface API for the model
print("\n=== HF API: danielcherubini/Qwen3.5-DeltaCoder-9B ===")
try:
    from huggingface_hub import HfApi
    api = HfApi()
    info = api.model_info("danielcherubini/Qwen3.5-DeltaCoder-9B")
    print(f"  pipeline_tag: {info.pipeline_tag}")
    print(f"  tags: {info.tags[:10] if info.tags else '-'}")
    print(f"  card data keys: {list(info.cardData.keys()) if info.cardData else '-'}")
except Exception as e:
    print(f"  err: {e}")
