"""Diagnostic: verify CE-only loss can overfit on a single record.

Test 1 (alpha=0.0): 20 steps, same record, loss = CE only.
Test 2 (alpha=1.0): 20 steps, same record, loss = KL only, log both KL and CE.

Reuses the same model/tokenizer/hypernet setup as train_hypernet_hpo.py.
"""

from __future__ import annotations

import gc
import logging
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
warnings.filterwarnings("ignore", message=".*guard_size_oblivious.*")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bootstrap import setup_path  # type: ignore[import-not-found]

setup_path()

# Reuse chunked loss from the training script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_hypernet_hpo import _chunked_kl_ce_loss  # type: ignore[import-not-found]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

NUM_STEPS = 20
BASE_MODEL = "Qwen/Qwen3.5-9B"
MODEL_CONFIG_NAME = "qwen3.5-9b"
LR = 3e-4
GRAD_CLIP = 1.0
MAX_LENGTH = 512
ACTIVATION_MAX_LENGTH = 512
TEMPERATURE = 2.0


def _build_env():
    """Load model, tokenizer, hypernet, and a single needle record."""
    import torch  # noqa: PLC0415
    from ctx_to_lora.modeling.hypernet import HyperLoRA  # noqa: PLC0415
    from model_training.d2l_config import (  # noqa: PLC0415
        build_from_scratch_hypernet_config,
        load_hypernet_defaults,
    )
    from model_training.d2l_data import generate_needle_dataset  # noqa: PLC0415
    from model_training.hypernetwork import _patch_flash_attention  # noqa: PLC0415
    from shared.hardware import get_best_device  # noqa: PLC0415
    from transformers import (  # noqa: PLC0415
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    _patch_flash_attention()

    dfl = load_hypernet_defaults()
    _l = dfl["lora"]

    # Build HyperLoRA
    logger.info("Building HyperLoRA for %s", MODEL_CONFIG_NAME)
    hc = build_from_scratch_hypernet_config(
        model_name=MODEL_CONFIG_NAME,
        lora_r=_l["r"],
        target_modules=_l["target_modules"],
    )
    hypernet = HyperLoRA(hc).to(torch.float32)
    hypernet.train()

    # Warm-start scaler_B
    with torch.no_grad():
        for name, param in hypernet.named_parameters():
            if "scaler_B" in name and param.abs().max() == 0:
                param.fill_(0.01)
                logger.info("Warm-started %s -> 0.01", name)

    n_params = sum(p.numel() for p in hypernet.parameters())
    logger.info("HyperLoRA params: %d", n_params)

    # Load NF4 base model
    logger.info("Loading base model: %s (NF4)", BASE_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        llm_int8_enable_fp32_cpu_offload=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        output_hidden_states=True,
        quantization_config=bnb_config,
        device_map="auto",
    ).eval()
    base_model.requires_grad_(False)

    # Gradient checkpointing (same config as training script)
    base_model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={
            "use_reentrant": False,
            "determinism_check": "none",
        }
    )
    base_model.train()

    # Device
    device = torch.device(get_best_device())
    hypernet = hypernet.to(device)

    # Data: single needle record (index 0)
    records = generate_needle_dataset(n=5)
    record = records[0]
    logger.info("Record 0 activation_text: %s", record["activation_text"][:120])
    logger.info("Record 0 teacher_text: %s", record["teacher_text"][:120])

    return base_model, tokenizer, hypernet, hc, record, device


def _run_overfit_test(
    base_model,
    tokenizer,
    hypernet,
    hc,
    record,
    device,
    alpha: float,
    label: str,
):
    """Run NUM_STEPS of overfitting on a single record and print loss per step."""
    import torch  # noqa: PLC0415
    from model_training.d2l_lora import apply_functional_lora  # noqa: PLC0415
    from model_training.d2l_probe import extract_activations_with_model  # noqa: PLC0415
    from torch.nn.utils import clip_grad_norm_  # noqa: PLC0415

    logger.info("=" * 70)
    logger.info("TEST: %s (alpha=%.1f) — %d steps on record 0", label, alpha, NUM_STEPS)
    logger.info("=" * 70)

    # Reset hypernet to fresh state for each test
    hypernet_state = {k: v.clone() for k, v in hypernet.state_dict().items()}

    # Warm-start scaler_B again after reset
    with torch.no_grad():
        for name, param in hypernet.named_parameters():
            if "scaler_B" in name and param.abs().max() == 0:
                param.fill_(0.01)

    trainable_params = list(hypernet.parameters())
    from bitsandbytes.optim import PagedAdamW8bit  # noqa: PLC0415

    optimizer = PagedAdamW8bit(trainable_params, lr=LR)

    layer_indices = list(hc.layer_indices)

    # Precompute teacher logits once (no teacher adapter — use base model itself)
    teacher_inputs = tokenizer(
        record["teacher_text"],
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
    )
    teacher_inputs = {k: v.to(device) for k, v in teacher_inputs.items()}

    answer_start = len(
        tokenizer(
            record["activation_text"],
            truncation=True,
            max_length=MAX_LENGTH,
        )["input_ids"]
    )
    seq_len = teacher_inputs["input_ids"].shape[1]
    logit_start = max(0, answer_start - 1)

    logger.info(
        "answer_start=%d, seq_len=%d, logit_start=%d",
        answer_start,
        seq_len,
        logit_start,
    )

    with torch.no_grad():
        teacher_logits_full = base_model(
            **teacher_inputs,
            output_hidden_states=False,
            use_cache=False,
        ).logits.detach()
        if answer_start < seq_len:
            teacher_logits_cached = teacher_logits_full[:, logit_start:, :].to("cpu")
        else:
            teacher_logits_cached = teacher_logits_full.to("cpu")
        del teacher_logits_full
    gc.collect()
    torch.cuda.empty_cache()

    results = []

    for step in range(1, NUM_STEPS + 1):
        optimizer.zero_grad(set_to_none=True)

        # Extract activations
        features, attn_mask = extract_activations_with_model(
            text=record["activation_text"],
            model=base_model,
            tokenizer=tokenizer,
            layer_indices=layer_indices,
            max_length=ACTIVATION_MAX_LENGTH,
        )

        # Hypernetwork forward
        lora_dict, _ = hypernet.generate_weights(features, attn_mask, None)

        # Student forward + backward inside functional LoRA context
        with apply_functional_lora(base_model, lora_dict, hc):
            student_logits = base_model(
                **teacher_inputs,
                output_hidden_states=False,
                use_cache=False,
            ).logits

            if answer_start < seq_len:
                student_logits = student_logits[:, logit_start:, :].contiguous()

            loss, metrics = _chunked_kl_ce_loss(
                student_logits,
                teacher_logits_cached,
                alpha=alpha,
                temperature=TEMPERATURE,
            )

            loss.backward()

        raw_grad_norm = clip_grad_norm_(trainable_params, float("inf"))
        clip_grad_norm_(trainable_params, GRAD_CLIP)
        optimizer.step()

        results.append(metrics)
        logger.info(
            "[%s] Step %02d/%d — total=%.6f  kl=%.6f  ce=%.6f  grad_norm=%.4e",
            label,
            step,
            NUM_STEPS,
            metrics["total_loss"],
            metrics["kl_loss"],
            metrics["ce_loss"],
            raw_grad_norm.item(),
        )

        del features, attn_mask, lora_dict, student_logits, loss, metrics
        gc.collect()
        torch.cuda.empty_cache()

    # Restore original hypernet state for next test
    hypernet.load_state_dict(hypernet_state)

    # Summary
    first_ce = results[0]["ce_loss"]
    last_ce = results[-1]["ce_loss"]
    first_kl = results[0]["kl_loss"]
    last_kl = results[-1]["kl_loss"]
    first_total = results[0]["total_loss"]
    last_total = results[-1]["total_loss"]

    logger.info("")
    logger.info("SUMMARY [%s] alpha=%.1f:", label, alpha)
    logger.info(
        "  CE:    %.6f -> %.6f  (delta=%.6f)", first_ce, last_ce, last_ce - first_ce
    )
    logger.info(
        "  KL:    %.6f -> %.6f  (delta=%.6f)", first_kl, last_kl, last_kl - first_kl
    )
    logger.info(
        "  Total: %.6f -> %.6f  (delta=%.6f)",
        first_total,
        last_total,
        last_total - first_total,
    )
    logger.info("")

    return results


def main():
    base_model, tokenizer, hypernet, hc, record, device = _build_env()

    # Test 1: CE-only (alpha=0.0)
    ce_results = _run_overfit_test(
        base_model,
        tokenizer,
        hypernet,
        hc,
        record,
        device,
        alpha=0.0,
        label="CE-ONLY",
    )

    # Test 2: KL-only (alpha=1.0)
    kl_results = _run_overfit_test(
        base_model,
        tokenizer,
        hypernet,
        hc,
        record,
        device,
        alpha=1.0,
        label="KL-ONLY",
    )

    # Final comparison
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(
        f"\n{'Step':>4}  {'CE-only total':>14}  {'CE-only CE':>12}  {'KL-only total':>14}  {'KL-only KL':>12}  {'KL-only CE':>12}"
    )
    print("-" * 80)
    for i in range(NUM_STEPS):
        print(
            f"{i + 1:4d}  {ce_results[i]['total_loss']:14.6f}  {ce_results[i]['ce_loss']:12.6f}"
            f"  {kl_results[i]['total_loss']:14.6f}  {kl_results[i]['kl_loss']:12.6f}"
            f"  {kl_results[i]['ce_loss']:12.6f}"
        )

    ce_first = ce_results[0]["ce_loss"]
    ce_last = ce_results[-1]["ce_loss"]
    print(
        f"\nCE-only path: CE went from {ce_first:.6f} to {ce_last:.6f} (delta={ce_last - ce_first:.6f})"
    )
    print(
        f"  -> CE {'DECREASED' if ce_last < ce_first else 'DID NOT DECREASE'} over {NUM_STEPS} steps"
    )

    kl_ce_first = kl_results[0]["ce_loss"]
    kl_ce_last = kl_results[-1]["ce_loss"]
    print(
        f"\nKL-only path: CE side-effect went from {kl_ce_first:.6f} to {kl_ce_last:.6f} (delta={kl_ce_last - kl_ce_first:.6f})"
    )
    print(
        f"  -> CE {'DECREASED' if kl_ce_last < kl_ce_first else 'DID NOT DECREASE'} as KL side-effect"
    )


if __name__ == "__main__":
    main()
