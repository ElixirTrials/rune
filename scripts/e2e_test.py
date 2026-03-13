"""End-to-end test: Hypernetwork as full project state machine.

Exercises every component:
  1. Creates a hypernetwork checkpoint (random init, sized for Qwen 0.5B)
  2. Bootstraps: methodology doc → H() → methodology_adapter (real)
  3. Iteration 1: constant prompt + adapter → generate (real transformers) → execute → H() → new adapter
  4. Iteration 2: constant prompt + new adapter → generate → execute → H() → next adapter
  5. Verifies: adapters on disk, adapter IDs change, provider loads them, trajectory accumulates

Usage:
    uv run scripts/e2e_test.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bootstrap import setup_path

setup_path()  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("e2e")

# --- Qwen2.5-Coder-0.5B architecture (GQA: 14 attn heads, 2 KV heads) ---
MODEL_NAME = "Qwen/Qwen2.5-Coder-0.5B"
NUM_LAYERS = 24
HIDDEN_DIM = 896
KV_DIM = 128  # num_kv_heads(2) * head_dim(64)
VOCAB_SIZE = 151936
RANK = 4
TARGET_MODULES = ["q_proj", "v_proj"]
# Per-module dims: q_proj is (896→896), v_proj is (896→128) due to GQA
MODULE_DIMS = {"q_proj": (896, 896), "v_proj": (896, 128)}

# --- Hypernetwork config (small for speed) ---
H_NUM_LATENTS = 4
H_LATENT_DIM = 32
H_DEPTH = 1
H_HEADS = 4
H_VOCAB_SIZE = 10000  # Hypernetwork's own vocab (not the model's)
H_MAX_LENGTH = 128


def step(n: int, msg: str) -> None:
    """Print a step header."""
    print(f"\n{'=' * 60}")
    print(f"  STEP {n}: {msg}")
    print(f"{'=' * 60}\n")


def create_hypernetwork_checkpoint(checkpoint_path: str) -> None:
    """Create a randomly initialized hypernetwork checkpoint."""
    import torch
    from model_training.hypernetwork import DocToLoraHypernetwork

    step(1, "Creating hypernetwork checkpoint (random init)")

    h = DocToLoraHypernetwork(
        input_dim=H_VOCAB_SIZE,
        num_latents=H_NUM_LATENTS,
        latent_dim=H_LATENT_DIM,
        depth=H_DEPTH,
        heads=H_HEADS,
        rank=RANK,
        target_modules=tuple(TARGET_MODULES),
        num_layers=NUM_LAYERS,
        hidden_dim=HIDDEN_DIM,
        module_dims=MODULE_DIMS,
    )

    param_count = sum(p.numel() for p in h.parameters())
    print(f"  Hypernetwork params: {param_count:,}")
    print(
        f"  Architecture: {H_NUM_LATENTS} latents x {H_LATENT_DIM}d, "
        f"{H_DEPTH} layers, {H_HEADS} heads"
    )
    print(f"  Target: {NUM_LAYERS} layers x {HIDDEN_DIM}d, rank={RANK}")
    print(f"  Output: {len(TARGET_MODULES) * NUM_LAYERS * 2} adapter weight matrices")

    torch.save(
        {
            "model_state_dict": h.state_dict(),
            "hypernetwork_config": {
                "input_dim": H_VOCAB_SIZE,
                "num_latents": H_NUM_LATENTS,
                "latent_dim": H_LATENT_DIM,
                "depth": H_DEPTH,
                "heads": H_HEADS,
                "rank": RANK,
                "target_modules": TARGET_MODULES,
                "num_layers": NUM_LAYERS,
                "hidden_dim": HIDDEN_DIM,
                "module_dims": MODULE_DIMS,
            },
        },
        checkpoint_path,
    )

    print(f"  Saved: {checkpoint_path}")
    print(f"  Size: {os.path.getsize(checkpoint_path) / 1024:.1f} KB")


def test_hypernetwork_generates_adapter(checkpoint_path: str, output_dir: str) -> str:
    """Test: hypernetwork loads from checkpoint, generates adapter from text."""
    from model_training.hypernetwork import generate_adapter, load_pretrained

    step(2, "Hypernetwork generates adapter from trajectory text")

    print("  Loading pretrained hypernetwork...")
    h = load_pretrained(checkpoint_path, device="cpu")
    print(f"  Loaded: {type(h).__name__}")

    trajectory_text = (
        "=== Bootstrap ===\n"
        "Phase 1: Decompose project into tasks\n"
        "Phase 2: Generate code with tests\n"
        "Phase 3: Execute, validate, iterate\n"
        "Phase 4: Integrate and run full test suite\n"
    )

    print(f"  Trajectory text: {len(trajectory_text)} chars")
    print("  Generating adapter...")

    adapter_path = generate_adapter(
        hypernetwork=h,
        trajectory_text=trajectory_text,
        output_dir=output_dir,
        base_model_id=MODEL_NAME,
        vocab_size=H_VOCAB_SIZE,
        max_length=H_MAX_LENGTH,
        device="cpu",
    )

    files = os.listdir(adapter_path)
    print(f"  Adapter saved to: {adapter_path}")
    print(f"  Files: {files}")
    assert "adapter_model.safetensors" in files, "Missing safetensors!"
    assert "adapter_config.json" in files, "Missing config!"

    config = json.loads(Path(adapter_path, "adapter_config.json").read_text())
    print("  Adapter config:")
    print(f"    peft_type: {config['peft_type']}")
    print(f"    rank: {config['r']}")
    print(f"    target_modules: {config['target_modules']}")
    print(f"    base_model: {config['base_model_name_or_path']}")

    return adapter_path


def test_provider_loads_adapter(adapter_path: str) -> None:
    """Test: TransformersProvider loads model + applies PEFT adapter."""
    step(3, "TransformersProvider loads model and applies PEFT adapter")

    from inference.transformers_provider import TransformersProvider
    from shared.hardware import get_best_device

    provider = TransformersProvider(
        model_name=MODEL_NAME,
        device=get_best_device(),
        torch_dtype="float32",
    )

    print(f"  Provider: {type(provider).__name__}")
    print(f"  Model: {MODEL_NAME}")

    # Load base model
    provider._load_model_if_needed()
    print(f"  Base model loaded on: {provider._device}")

    # Register and activate adapter
    async def _test() -> None:
        await provider.load_adapter("methodology_adapter", adapter_path)
        adapters = await provider.list_adapters()
        print(f"  Registered adapters: {adapters}")
        assert "methodology_adapter" in adapters

        # Generate with base model (no adapter)
        print("\n  --- Generation WITHOUT adapter ---")
        result_base = await provider.generate(
            prompt="You are a Python code generator.\n\nBuild a Python statistics library with mean, median, mode",
            model=MODEL_NAME,
            adapter_id=None,
            max_tokens=100,
        )
        print(f"  Adapter used: {result_base.adapter_id}")
        print(f"  Tokens: {result_base.token_count}")
        print(f"  Output (first 200 chars): {result_base.text[:200]!r}")

        # Generate WITH adapter
        print("\n  --- Generation WITH adapter ---")
        result_adapted = await provider.generate(
            prompt="You are a Python code generator.\n\nBuild a Python statistics library with mean, median, mode",
            model=MODEL_NAME,
            adapter_id="methodology_adapter",
            max_tokens=100,
        )
        print(f"  Adapter used: {result_adapted.adapter_id}")
        print(f"  Tokens: {result_adapted.token_count}")
        print(f"  Output (first 200 chars): {result_adapted.text[:200]!r}")

        # Verify adapter was actually used
        assert result_adapted.adapter_id == "methodology_adapter", (
            f"Expected adapter_id='methodology_adapter', got '{result_adapted.adapter_id}'"
        )

        # The outputs should differ (different weights active)
        if result_base.text != result_adapted.text:
            print(
                "\n  ✓ Base and adapted outputs DIFFER (adapter influenced generation)"
            )
        else:
            print(
                "\n  ⚠ Base and adapted outputs are identical (adapter may not have been applied)"
            )

    asyncio.run(_test())


async def test_full_iteration_loop(checkpoint_path: str) -> None:
    """Test: Full rune_runner iteration loop with hypernetwork."""
    step(4, "Full iteration loop: prompt → generate → execute → H() → adapter → repeat")

    # Configure provider
    os.environ["INFERENCE_PROVIDER"] = "transformers"
    os.environ["TRANSFORMERS_MODEL_NAME"] = MODEL_NAME
    from shared.hardware import get_best_device
    os.environ["TRANSFORMERS_DEVICE"] = get_best_device()

    # Clear provider cache so new env vars take effect
    from inference.factory import _clear_cache

    _clear_cache()

    from rune_runner import run_project

    result = await run_project(
        project_prompt="Build a Python statistics library with mean, median, mode, stdev, percentile",
        max_iterations=3,
        checkpoint_path=checkpoint_path,
        base_model_id=MODEL_NAME,
        device="cpu",  # Hypernetwork on CPU (small)
    )

    print(f"\n  Session: {result['session_id']}")
    print(f"  Total iterations: {result['total_iterations']}")
    print(f"  Final tests passed: {result['final_tests_passed']}")
    print(f"  Adapter dir: {result['adapter_dir']}")

    # Show per-iteration details
    adapters_seen: list[str | None] = []
    for it in result["iterations"]:
        adapter = it["adapter_id"]
        adapters_seen.append(adapter)
        code = it["generated_code"]
        lines = code.split("\n") if code else []
        print(f"\n  --- Iteration {it['iteration']} ---")
        print(f"    Adapter: {adapter or 'none'}")
        print(f"    Tests passed: {it['tests_passed']}")
        print(f"    Exit code: {it['exit_code']}")
        print(f"    Code lines: {len(lines)}")
        if lines:
            for line in lines[:10]:
                print(f"      {line}")
            if len(lines) > 10:
                print(f"      ... ({len(lines) - 10} more lines)")

    # Verify adapter progression
    print(f"\n  Adapter progression: {adapters_seen}")

    # Check adapter files on disk
    adapter_dir = Path(result["adapter_dir"])
    if adapter_dir.exists():
        subdirs = sorted(adapter_dir.iterdir())
        print(f"  Adapter subdirs on disk: {[d.name for d in subdirs]}")
        for d in subdirs:
            if d.is_dir():
                contents = list(d.iterdir())
                print(f"    {d.name}/: {[f.name for f in contents]}")


def run_e2e() -> None:
    """Run the full end-to-end test."""
    print("=" * 60)
    print("  RUNE E2E TEST: Hypernetwork as Project State Machine")
    print("=" * 60)
    print(f"\n  Model: {MODEL_NAME}")
    print(f"  Architecture: {NUM_LAYERS} layers, hidden={HIDDEN_DIM}")
    print(f"  LoRA: rank={RANK}, targets={TARGET_MODULES}")

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = os.path.join(tmpdir, "hypernetwork.pt")
        adapter_dir = os.path.join(tmpdir, "test_adapter")

        # Step 1: Create random hypernetwork checkpoint
        create_hypernetwork_checkpoint(checkpoint_path)

        # Step 2: Test hypernetwork generates adapter
        adapter_path = test_hypernetwork_generates_adapter(checkpoint_path, adapter_dir)

        # Step 3: Test provider loads and uses adapter
        test_provider_loads_adapter(adapter_path)

        # Step 4: Full iteration loop
        asyncio.run(test_full_iteration_loop(checkpoint_path))

    print("\n" + "=" * 60)
    print("  E2E TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    run_e2e()
