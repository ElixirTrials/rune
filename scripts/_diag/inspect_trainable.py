"""Verify trainable parameter count + which modules actually got LoRA layers."""
import sys

sys.path.insert(0, "libs/model-training/src")
sys.path.insert(0, "libs/shared/src")

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

MODEL_ID = "Qwen/Qwen3.5-9B"
ADAPTER = "danielcherubini/Qwen3.5-DeltaCoder-9B"

print(f"Loading base {MODEL_ID} ...")
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                          bnb_4bit_compute_dtype=torch.bfloat16,
                          bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb, torch_dtype=torch.bfloat16)
print(f"Loading adapter {ADAPTER} ...")
model = PeftModel.from_pretrained(model, ADAPTER)
# Enable training on adapter
for n, p in model.named_parameters():
    if "lora_" in n:
        p.requires_grad_(True)

# Count
n_total, n_train = 0, 0
lora_params = 0
modules_with_lora = set()
for n, p in model.named_parameters():
    n_total += p.numel()
    if p.requires_grad:
        n_train += p.numel()
    if "lora_" in n:
        lora_params += p.numel()
        # Extract module path
        parts = n.split(".lora_")[0]
        # Get the module type — last segment
        modules_with_lora.add(parts.split('.')[-1])

print("\n=== PARAM COUNTS ===")
print(f"Total params:     {n_total/1e9:.3f}B")
print(f"Trainable params: {n_train/1e6:.3f}M ({100*n_train/n_total:.4f}%)")
print(f"LoRA params:      {lora_params/1e6:.3f}M")

print("\n=== MODULES WITH LoRA ===")
for m in sorted(modules_with_lora):
    print(f"  {m}")

# Sanity: print one example LoRA layer's shape
print("\n=== SAMPLE LoRA LAYER SHAPES ===")
shown = 0
for n, p in model.named_parameters():
    if "lora_A" in n or "lora_B" in n:
        print(f"  {n[:80]}: shape={tuple(p.shape)}")
        shown += 1
        if shown >= 6:
            break

# Check the active adapter's effective alpha + scaling
if hasattr(model, 'peft_config'):
    for name, cfg in model.peft_config.items():
        a = cfg.lora_alpha
        r = cfg.r
        print(f"\n=== adapter '{name}' ===")
        print(f"  r={r}, alpha={a}, scaling={a/r:.3f}")
        print(f"  target_modules: {cfg.target_modules}")
