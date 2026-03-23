#!/usr/bin/env python3
"""End-to-end training smoke test for Rune.

Validates the full training pipeline with a tiny model (~135M params) on
whatever hardware is available (CUDA > MPS > CPU). Exercises:

  1. Activation extraction — hidden states from base model
  2. Hypernetwork forward — generate LoRA weights from activations
  3. Functional LoRA injection — patch model with generated weights
  4. Loss computation — shift-aware KL+CE over answer span
  5. Backward pass — gradients flow through LoRA to hypernetwork
  6. TIES/DARE merging — merge adapter state dicts with dtype preservation

Uses HuggingFaceTB/SmolLM2-135M (auto-downloaded, ~270MB) so this runs
in under 60 seconds on an M4 Mac and under 30 seconds on GPU.

Run: uv run python scripts/e2e_training_smoke.py
"""

# ruff: noqa: E402, D
# mypy: ignore-errors
from __future__ import annotations

import sys
import time
from pathlib import Path

# Add workspace src paths so libs are importable outside pytest
_root = Path(__file__).resolve().parent.parent
for _src in [
    "libs/shared/src",
    "libs/model-training/src",
    "libs/adapter-registry/src",
    "libs/inference/src",
    "libs/evaluation/src",
]:
    sys.path.insert(0, str(_root / _src))

passed = 0
failed = 0

MODEL_NAME = "HuggingFaceTB/SmolLM2-135M"


def check(name: str, condition: bool, detail: str = "") -> None:
    global passed, failed
    if condition:
        passed += 1
        print(f"  ✓ {name}")
    else:
        failed += 1
        print(f"  ✗ {name}{': ' + detail if detail else ''}")


def main() -> int:
    global passed, failed
    t0 = time.time()

    import torch
    from shared.hardware import get_best_device

    device_name = get_best_device()
    device = torch.device(device_name)
    print(f"\n{'=' * 60}")
    print("  Rune Training E2E Smoke Test")
    print(f"  Device: {device_name} | Model: {MODEL_NAME}")
    print(f"{'=' * 60}\n")

    # ── 1. Load tiny model and tokenizer ─────────────────────────
    print("[1/6] Loading model and tokenizer...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base_model = (
        AutoModelForCausalLM.from_pretrained(MODEL_NAME, output_hidden_states=True)
        .eval()
        .to(device)
    )

    num_layers = base_model.config.num_hidden_layers
    hidden_dim = base_model.config.hidden_size
    check("Model loaded", num_layers > 0, f"{num_layers} layers, hidden={hidden_dim}")

    # Discover attention projection modules and their dimensions
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    available_targets = []
    module_dims: dict[str, tuple[int, int]] = {}  # {name: (in_features, out_features)}
    for name, mod in base_model.named_modules():
        short = name.split(".")[-1]
        if short in target_modules and hasattr(mod, "weight"):
            if short not in available_targets:
                available_targets.append(short)
                module_dims[short] = (mod.weight.shape[1], mod.weight.shape[0])

    check(
        "Found attention projections",
        len(available_targets) >= 2,
        f"found: {available_targets} dims: {module_dims}",
    )

    # Pick a subset of layers to target (first and last)
    layer_indices = [0, num_layers - 1]

    # ── 2. Activation extraction ─────────────────────────────────
    print("\n[2/6] Extracting activations...")
    from model_training.d2l_probe import extract_activations_with_model

    activation_text = "def fibonacci(n):\n    if n <= 1:\n        return n"
    answer_text = "\n    return fibonacci(n-1) + fibonacci(n-2)"
    teacher_text = activation_text + answer_text

    features, attn_mask = extract_activations_with_model(
        text=activation_text,
        model=base_model,
        tokenizer=tokenizer,
        layer_indices=layer_indices,
        max_length=128,
    )

    check(
        "Features shape",
        features.shape[1] == len(layer_indices),
        f"shape={list(features.shape)}",
    )
    check(
        "Features on device",
        str(features.device).startswith(device_name) or device_name == "cpu",
        f"features on {features.device}, expected {device_name}",
    )

    seq_len = features.shape[2]
    check("Sequence length > 0", seq_len > 0, f"seq_len={seq_len}")

    # ── 3. Hypernetwork forward ──────────────────────────────────
    print("\n[3/6] Hypernetwork forward pass...")

    # Build a minimal hypernetwork that produces LoRA weights for our model
    # We use a simple learned projection rather than the full Sakana perceiver
    import torch.nn as nn

    lora_rank = 4

    class MiniHypernet(nn.Module):
        """Tiny hypernetwork: pool activations → project to LoRA A/B weights."""

        def __init__(self, hidden_dim, n_layers, targets, rank, dims):
            super().__init__()
            self.n_layers = n_layers
            self.targets = targets
            self.rank = rank
            self.dims = dims  # {target: (in_features, out_features)}
            self.projectors_a = nn.ModuleDict()
            self.projectors_b = nn.ModuleDict()
            for t in targets:
                in_f, out_f = dims[t]
                self.projectors_a[t] = nn.Linear(hidden_dim, n_layers * rank * in_f)
                self.projectors_b[t] = nn.Linear(hidden_dim, n_layers * rank * out_f)

        def forward(self, features, attn_mask):
            # features: (1, n_layers, seq_len, hidden_dim)
            pooled = features.mean(dim=(1, 2))  # (1, hidden_dim)

            lora_dict = {}
            for t in self.targets:
                in_f, out_f = self.dims[t]
                raw_a = self.projectors_a[t](pooled)
                raw_b = self.projectors_b[t](pooled)
                lora_dict[t] = {
                    "A": raw_a.view(1, self.n_layers, self.rank, in_f),
                    "B": raw_b.view(1, self.n_layers, self.rank, out_f),
                }
            return lora_dict

    hypernet = MiniHypernet(
        hidden_dim=hidden_dim,
        n_layers=len(layer_indices),
        targets=available_targets,
        rank=lora_rank,
        dims=module_dims,
    ).to(device)
    hypernet.train()

    lora_dict = hypernet(features, attn_mask)

    check("LoRA dict has all targets", set(lora_dict.keys()) == set(available_targets))
    first_target = available_targets[0]
    expected_in_f = module_dims[first_target][0]
    check(
        "LoRA A shape correct",
        lora_dict[first_target]["A"].shape
        == (1, len(layer_indices), lora_rank, expected_in_f),
        f"got {list(lora_dict[first_target]['A'].shape)}",
    )

    # ── 4. Functional LoRA injection + loss computation ──────────
    print("\n[4/6] Functional LoRA + loss computation...")
    from types import SimpleNamespace

    from model_training.d2l_lora import apply_functional_lora
    from model_training.d2l_train import D2LTrainConfig, _compute_kl_ce_loss

    # Build a fake HypernetConfig with the fields apply_functional_lora needs
    hc = SimpleNamespace(
        lora_config=SimpleNamespace(
            target_modules=available_targets,
            r=lora_rank,
            lora_alpha=lora_rank * 2,
        ),
        layer_indices=layer_indices,
    )

    # Tokenize teacher text
    teacher_inputs = tokenizer(
        teacher_text, return_tensors="pt", truncation=True, max_length=128
    )
    teacher_inputs = {k: v.to(device) for k, v in teacher_inputs.items()}

    # Teacher forward (frozen)
    with torch.no_grad():
        teacher_out = base_model(**teacher_inputs, output_hidden_states=False)
    teacher_logits = teacher_out.logits

    # Student forward (with LoRA patches)
    with apply_functional_lora(base_model, lora_dict, hc):
        student_out = base_model(**teacher_inputs, output_hidden_states=False)
    student_logits = student_out.logits

    check(
        "Student logits shape matches teacher",
        student_logits.shape == teacher_logits.shape,
        f"student={list(student_logits.shape)}, teacher={list(teacher_logits.shape)}",
    )

    # Compute answer_start
    answer_start = len(
        tokenizer(activation_text, truncation=True, max_length=128)["input_ids"]
    )

    config = D2LTrainConfig(
        sakana_checkpoint_path="dummy",
        alpha=0.5,
        temperature=2.0,
    )
    loss, metrics = _compute_kl_ce_loss(
        student_logits, teacher_logits, answer_start, config
    )

    check("Loss is finite", torch.isfinite(loss).item(), f"loss={loss.item():.6f}")
    check("Loss is positive", loss.item() > 0, f"loss={loss.item():.6f}")
    check("KL loss computed", metrics["kl_loss"] >= 0, f"kl={metrics['kl_loss']:.6f}")
    check("CE loss computed", metrics["ce_loss"] >= 0, f"ce={metrics['ce_loss']:.6f}")

    # ── 5. Backward pass — gradients flow to hypernetwork ────────
    print("\n[5/6] Backward pass + gradient check...")

    # Re-run with fresh graph (previous student_out graph may be stale)
    features2, attn_mask2 = extract_activations_with_model(
        text=activation_text,
        model=base_model,
        tokenizer=tokenizer,
        layer_indices=layer_indices,
        max_length=128,
    )
    lora_dict2 = hypernet(features2, attn_mask2)

    with apply_functional_lora(base_model, lora_dict2, hc):
        student_out2 = base_model(**teacher_inputs, output_hidden_states=False)

    loss2, _ = _compute_kl_ce_loss(
        student_out2.logits, teacher_logits, answer_start, config
    )
    loss2.backward()

    # Check gradients reached the hypernetwork
    has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in hypernet.parameters()
        if p.requires_grad
    )
    check("Gradients flow to hypernetwork", has_grad)

    # Verify hypernet grad magnitude is non-trivial
    hypernet_grad_norm = sum(
        p.grad.norm().item() for p in hypernet.parameters() if p.grad is not None
    )
    check(
        "Hypernet gradient norm > 0",
        hypernet_grad_norm > 0,
        f"grad_norm={hypernet_grad_norm:.6f}",
    )

    # ── 6. TIES/DARE merging ────────────────────────────────────
    print("\n[6/6] Adapter merging...")
    from model_training.merging import dare_merge, ties_merge

    # Create two fake adapter state dicts (simulating trained adapters)
    sd1 = {f"layer.{i}.weight": torch.randn(lora_rank, hidden_dim) for i in range(2)}
    sd2 = {f"layer.{i}.weight": torch.randn(lora_rank, hidden_dim) for i in range(2)}

    # Test with bfloat16 to verify dtype preservation
    sd1_bf16 = {k: v.to(torch.bfloat16) for k, v in sd1.items()}
    sd2_bf16 = {k: v.to(torch.bfloat16) for k, v in sd2.items()}

    ties_result = ties_merge([sd1_bf16, sd2_bf16], density=0.5)
    check(
        "TIES merge preserves bfloat16",
        ties_result["layer.0.weight"].dtype == torch.bfloat16,
    )
    check(
        "TIES merge preserves shape",
        ties_result["layer.0.weight"].shape == (lora_rank, hidden_dim),
    )

    dare_result = dare_merge([sd1_bf16, sd2_bf16], drop_rate=0.1, seed=42)
    check(
        "DARE merge preserves bfloat16",
        dare_result["layer.0.weight"].dtype == torch.bfloat16,
    )

    # Verify DARE seed reproducibility
    dare_result2 = dare_merge([sd1_bf16, sd2_bf16], drop_rate=0.1, seed=42)
    check(
        "DARE merge is reproducible with seed",
        torch.equal(dare_result["layer.0.weight"], dare_result2["layer.0.weight"]),
    )

    # ── Summary ──────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"  {passed} passed, {failed} failed  ({elapsed:.1f}s on {device_name})")
    print(f"{'=' * 60}\n")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
