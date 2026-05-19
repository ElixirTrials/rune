"""Diagnostic: measure gradient norms from KL and CE separately on hypernetwork.

Sets up the same training infrastructure as train_hypernet_hpo.py (bootstrap,
HyperLoRA, NF4 base model, gradient checkpointing), runs separate forward
passes on needle record 0, and measures gradient norms through each loss
component independently.

This tells us whether CE gradients reach the hypernetwork at all and their
relative magnitude vs KL gradients.

Run: uv run python scripts/_diag/test_grad_norms.py
"""

from __future__ import annotations

import gc
import logging
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("RUNE_DISABLE_MLFLOW", "1")
warnings.filterwarnings("ignore", message=".*guard_size_oblivious.*")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bootstrap import setup_path  # type: ignore[import-not-found]

setup_path()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

CHUNK_SIZE = 48


def _chunked_kl_ce_separate(student_logits, teacher_logits, *, temperature):
    """Compute KL and CE as separate grad-bearing tensors.

    Returns (kl, ce) where both are scalar tensors with gradient.
    """
    import torch
    from torch.nn import functional as f
    from torch.utils.checkpoint import checkpoint

    n_tokens = student_logits.shape[1]
    vocab = student_logits.shape[-1]
    device = student_logits.device
    teacher_ref = teacher_logits

    def _chunk_fn(s_chunk, t_chunk):
        t_chunk = t_chunk.to(s_chunk.device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            t_soft = f.softmax(t_chunk / temperature, dim=-1)
            t_hard = t_chunk.argmax(-1).reshape(-1)
            del t_chunk
            s_log = f.log_softmax(s_chunk / temperature, dim=-1)
            kl = f.kl_div(s_log, t_soft, reduction="sum")
            del s_log, t_soft
            ce = f.cross_entropy(s_chunk.reshape(-1, vocab), t_hard, reduction="sum")
            del t_hard
        return kl, ce

    kl_sum = torch.zeros(1, device=device)
    ce_sum = torch.zeros(1, device=device)
    total_elements = 0

    for i in range(0, n_tokens, CHUNK_SIZE):
        s = student_logits[:, i : i + CHUNK_SIZE, :]
        t = teacher_ref[:, i : i + CHUNK_SIZE, :].contiguous()
        chunk_elems = s.shape[0] * s.shape[1]
        kl, ce = checkpoint(_chunk_fn, s, t, use_reentrant=False)
        kl_sum = kl_sum + kl
        ce_sum = ce_sum + ce
        total_elements += chunk_elems

    kl_final = kl_sum / total_elements * temperature**2
    ce_final = ce_sum / total_elements
    return kl_final, ce_final


def _grad_norm(hypernet):
    """Total L2 grad norm across all hypernetwork parameters."""

    total = 0.0
    for p in hypernet.parameters():
        if p.grad is not None:
            total += p.grad.detach().float().pow(2).sum().item()
    return total**0.5


def _count_nonzero_grads(hypernet):
    """Count params with nonzero grad and total grad-bearing params."""
    n_with_grad = 0
    n_nonzero = 0
    for p in hypernet.parameters():
        if p.grad is not None:
            n_with_grad += 1
            if p.grad.abs().max().item() > 0:
                n_nonzero += 1
    return n_nonzero, n_with_grad


def main():
    import torch
    from ctx_to_lora.modeling.hypernet import HyperLoRA
    from model_training.d2l_config import build_from_scratch_hypernet_config
    from model_training.d2l_data import generate_needle_dataset
    from model_training.d2l_lora import apply_functional_lora
    from model_training.d2l_probe import extract_activations_with_model
    from model_training.hypernetwork import _patch_flash_attention
    from shared.hardware import get_best_device
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    _patch_flash_attention()

    base_model_name = "Qwen/Qwen3.5-9B"
    model_config_name = "qwen3.5-9b"
    activation_max_length = 512
    max_length = 512
    temperature = 4.0

    # Build HyperLoRA
    logger.info("Building HyperLoRA from scratch")
    hc = build_from_scratch_hypernet_config(model_name=model_config_name)
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
    logger.info("Loading base model: %s (NF4)", base_model_name)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        llm_int8_enable_fp32_cpu_offload=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        output_hidden_states=True,
        quantization_config=bnb_config,
        device_map="auto",
    ).eval()
    base_model.requires_grad_(False)
    base_model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={
            "use_reentrant": False,
            "determinism_check": "none",
        }
    )
    base_model.train()

    device = torch.device(get_best_device())
    logger.info("Device: %s", device)
    hypernet = hypernet.to(device)

    # Data: needle record 0
    records = generate_needle_dataset(n=20)
    record = records[0]
    layer_indices = list(hc.layer_indices)

    # Tokenize
    teacher_inputs = tokenizer(
        record["teacher_text"],
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    answer_start = len(
        tokenizer(
            record["activation_text"],
            truncation=True,
            max_length=max_length,
        )["input_ids"]
    )
    seq_len = teacher_inputs["input_ids"].shape[1]
    logit_start = max(0, answer_start - 1)
    logger.info(
        "answer_start=%d, seq_len=%d, logit_start=%d, answer_tokens=%d",
        answer_start,
        seq_len,
        logit_start,
        seq_len - answer_start,
    )

    # Pre-compute teacher logits once (no grad, reusable)
    logger.info("Computing teacher logits (no grad, reusable)...")
    inp = {k: v.to(device) for k, v in teacher_inputs.items()}
    with torch.no_grad():
        t_logits = base_model(
            **inp, output_hidden_states=False, use_cache=False
        ).logits.detach()
    t_logits = t_logits[:, logit_start:, :].to("cpu")
    del inp
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("Teacher logits shape: %s (on CPU)", t_logits.shape)

    def do_forward_backward(loss_mode, label):
        """Full forward + loss + backward inside LoRA context.

        loss_mode: "kl_only", "ce_only", "combined_10", "combined_05"
        Returns (grad_norm, n_nonzero, n_with_grad, kl_val, ce_val).
        """
        hypernet.zero_grad(set_to_none=True)
        inp_d = {k: v.to(device) for k, v in teacher_inputs.items()}

        features, attn_mask = extract_activations_with_model(
            text=record["activation_text"],
            model=base_model,
            tokenizer=tokenizer,
            layer_indices=layer_indices,
            max_length=activation_max_length,
        )
        lora_dict, _ = hypernet.generate_weights(features, attn_mask, None)
        del features, attn_mask

        # Student forward + backward INSIDE the LoRA context (required for
        # gradient checkpointing recomputation to see the LoRA patches).
        with apply_functional_lora(base_model, lora_dict, hc):
            s_logits = base_model(
                **inp_d, output_hidden_states=False, use_cache=False
            ).logits
            del inp_d, lora_dict

            s_logits = s_logits[:, logit_start:, :].contiguous()
            kl, ce = _chunked_kl_ce_separate(
                s_logits, t_logits, temperature=temperature
            )
            del s_logits

            kl_val = kl.item()
            ce_val = ce.item()

            if loss_mode == "kl_only":
                loss = kl
            elif loss_mode == "ce_only":
                loss = ce
            elif loss_mode == "combined_10":
                loss = 1.0 * kl + 0.0 * ce
            elif loss_mode == "combined_05":
                loss = 0.5 * kl + 0.5 * ce
            else:
                raise ValueError(loss_mode)

            loss.backward()

        gn = _grad_norm(hypernet)
        nz, total = _count_nonzero_grads(hypernet)
        logger.info(
            "%s: grad_norm=%.4e, loss=%.6f (kl=%.6f, ce=%.6f), %d/%d nonzero",
            label,
            gn,
            loss.item(),
            kl_val,
            ce_val,
            nz,
            total,
        )

        del kl, ce, loss
        gc.collect()
        torch.cuda.empty_cache()
        return gn, nz, total, kl_val, ce_val

    # --- Run 4 measurements ---
    kl_gn, kl_nz, kl_total, kl_val, ce_val = do_forward_backward("kl_only", "KL-only")
    ce_gn, ce_nz, ce_total, _, _ = do_forward_backward("ce_only", "CE-only")
    c10_gn, _, _, _, _ = do_forward_backward("combined_10", "Combined(1.0)")
    c05_gn, _, _, _, _ = do_forward_backward("combined_05", "Combined(0.5)")

    # --- Results ---
    print("\n" + "=" * 60)
    print("GRADIENT NORM DIAGNOSTIC RESULTS")
    print("=" * 60)
    print(f"KL loss value:               {kl_val:.6f}")
    print(f"CE loss value:               {ce_val:.6f}")
    print()
    print(f"KL-only grad norm:           {kl_gn:.4e}")
    print(f"CE-only grad norm:           {ce_gn:.4e}")
    print(f"Combined (alpha=1.0) grad norm: {c10_gn:.4e}")
    print(f"Combined (alpha=0.5) grad norm: {c05_gn:.4e}")
    print()
    print(f"KL: {kl_nz}/{kl_total} params with nonzero grad")
    print(f"CE: {ce_nz}/{ce_total} params with nonzero grad")
    print()
    if ce_gn > 0:
        print(f"KL/CE grad norm ratio:       {kl_gn / ce_gn:.4f}")
    else:
        print("KL/CE grad norm ratio:       CE grad norm is ZERO")
    print("=" * 60)


if __name__ == "__main__":
    main()
