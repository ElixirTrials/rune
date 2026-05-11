"""Train HyperLoRA from scratch using HPO DeltaCoder adapter as teacher.

Builds a fresh perceiver-based HyperLoRA for Qwen 3.5 9B and trains it
via KL+CE distillation against a teacher model composed of the base model
plus the HPO-tuned DeltaCoder LoRA adapter.

Follows the Sakana training pattern:
  1. Extract per-layer activations from the base model
  2. Feed activations through the perceiver to generate LoRA weights
  3. Apply generated LoRA to student via functional injection
  4. Teacher forward pass (base + HPO adapter, no_grad)
  5. Student forward pass (base + generated LoRA)
  6. Loss = alpha * KL + (1-alpha) * CE

Usage:
    uv run python scripts/train_hypernet_hpo.py \
        --teacher-adapter hpo_artifacts/best_diffloss_v1 \
        --dataset data/mined/all_unrolled.jsonl \
        --num-steps 1000 \
        --checkpoint-dir checkpoints/hypernet_hpo
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
warnings.filterwarnings("ignore", message=".*guard_size_oblivious.*")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bootstrap import setup_path

setup_path()

logger = logging.getLogger(__name__)

_LOSS_CHUNK = 128


def _chunked_kl_ce_loss(
    student_logits: Any,
    teacher_logits: Any,
    *,
    alpha: float,
    temperature: float,
) -> tuple[Any, dict[str, float]]:
    """KL+CE loss chunked along the token dimension to cap VRAM.

    With vocab ~152K, a single log_softmax over the full answer span
    allocates answer_tokens × 152K × 2 bytes — easily 600+ MB.
    Chunking to _LOSS_CHUNK tokens keeps peak intermediates at ~37 MB.
    """
    import torch  # noqa: PLC0415
    import torch.nn.functional as F  # noqa: PLC0415, N812

    n_tokens = student_logits.shape[1]
    if n_tokens == 0:
        zero = torch.tensor(0.0, device=student_logits.device, requires_grad=True)
        return zero, {"kl_loss": 0.0, "ce_loss": 0.0, "total_loss": 0.0}

    vocab = student_logits.shape[-1]
    kl_sum = torch.tensor(0.0, device=student_logits.device)
    ce_sum = torch.tensor(0.0, device=student_logits.device)
    total_elements = 0

    for i in range(0, n_tokens, _LOSS_CHUNK):
        s = student_logits[:, i : i + _LOSS_CHUNK, :]
        t = teacher_logits[:, i : i + _LOSS_CHUNK, :]
        chunk_elems = s.shape[0] * s.shape[1]

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            kl_sum = kl_sum + F.kl_div(
                F.log_softmax(s / temperature, dim=-1),
                F.softmax(t / temperature, dim=-1),
                reduction="sum",
            )
            ce_sum = ce_sum + F.cross_entropy(
                s.reshape(-1, vocab),
                t.argmax(-1).reshape(-1),
                reduction="sum",
            )
        total_elements += chunk_elems

    kl = kl_sum / total_elements * temperature**2
    ce = ce_sum / total_elements
    total = alpha * kl + (1.0 - alpha) * ce

    return total, {
        "kl_loss": kl.item(),
        "ce_loss": ce.item(),
        "total_loss": total.item(),
    }


def main() -> None:  # noqa: C901
    from model_training.d2l_config import load_hypernet_defaults  # noqa: PLC0415

    dfl = load_hypernet_defaults()
    _t = dfl["training"]
    _l = dfl["lora"]

    parser = argparse.ArgumentParser(
        description="Train HyperLoRA from scratch with HPO DeltaCoder teacher"
    )
    parser.add_argument(
        "--teacher-adapter",
        type=str,
        required=True,
        help="Path to HPO-tuned LoRA adapter directory",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/mined/all_unrolled.jsonl",
        help="Path to training JSONL (activation_text + teacher_text fields)",
    )
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen3.5-9B")
    parser.add_argument("--model-config-name", type=str, default="qwen3.5-9b")
    parser.add_argument("--lora-r", type=int, default=_l["r"])
    parser.add_argument("--num-steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=_t["lr"])
    parser.add_argument("--alpha", type=float, default=_t["alpha"])
    parser.add_argument("--temperature", type=float, default=_t["temperature"])
    parser.add_argument("--grad-clip", type=float, default=_t["grad_clip"])
    parser.add_argument("--warmup-steps", type=int, default=_t["warmup_steps"])
    parser.add_argument("--max-length", type=int, default=_t["max_length"])
    parser.add_argument(
        "--activation-max-length",
        type=int,
        default=_t.get("activation_max_length", 512),
        help="Max tokens for perceiver activation extraction (shorter to save VRAM)",
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default="checkpoints/hypernet_hpo"
    )
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--experiment-name", type=str, default="hypernet-hpo")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--target-modules", nargs="+", default=_l["target_modules"])
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    # Skip if final checkpoint already exists
    final_ckpt = Path(args.checkpoint_dir) / "checkpoint.pt"
    if final_ckpt.exists() and not args.smoke_test:
        logger.info(
            "Final checkpoint already exists: %s — skipping training. "
            "Delete the file to re-train.",
            final_ckpt,
        )
        return

    # Auto-fetch teacher adapter from MLflow/S3 if missing
    hpo_run_id = "e9c9760f816f46948197519e1c905273"
    hpo_s3_prefix = (
        "s3://elixirtrials-949678234935-eu-west-2-artifacts"
        f"/mlflow/artifacts/3/{hpo_run_id}/artifacts"
    )
    teacher_path = Path(args.teacher_adapter)
    if not (teacher_path / "adapter_config.json").exists():
        import subprocess  # noqa: PLC0415

        teacher_path.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Teacher adapter not found at %s — fetching from S3...", teacher_path
        )
        s3_ok = (
            subprocess.run(  # noqa: S603, S607
                [
                    "aws",
                    "s3",
                    "cp",
                    f"{hpo_s3_prefix}/",
                    f"{teacher_path}/",
                    "--recursive",
                ],
                capture_output=True,
            ).returncode
            == 0
        )
        if not s3_ok:
            logger.info("S3 download failed — trying MLflow CLI...")
            subprocess.run(  # noqa: S603
                [
                    "uv",
                    "run",
                    "mlflow",
                    "artifacts",
                    "download",
                    "--run-id",
                    hpo_run_id,
                    "--dst-path",
                    str(teacher_path),
                ],
                check=True,
            )
        if not (teacher_path / "adapter_config.json").exists():
            raise FileNotFoundError(
                f"Could not fetch teacher adapter. Neither S3 nor MLflow download "
                f"produced adapter_config.json in {teacher_path}"
            )
        logger.info("Teacher adapter downloaded to %s", teacher_path)

    import json  # noqa: PLC0415
    import re  # noqa: PLC0415
    from collections import defaultdict  # noqa: PLC0415
    from types import SimpleNamespace  # noqa: PLC0415

    import torch  # noqa: PLC0415
    from bitsandbytes.optim import PagedAdamW8bit  # noqa: PLC0415
    from ctx_to_lora.modeling.hypernet import HyperLoRA  # noqa: PLC0415
    from model_training.d2l_config import (  # noqa: PLC0415
        build_from_scratch_hypernet_config,
    )
    from model_training.d2l_data import load_jsonl, split_by_task_id  # noqa: PLC0415
    from model_training.d2l_lora import apply_functional_lora  # noqa: PLC0415
    from model_training.d2l_probe import extract_activations_with_model  # noqa: PLC0415
    from model_training.sakana_d2l import _patch_flash_attention  # noqa: PLC0415
    from model_training.training_common import (  # noqa: PLC0415
        _log_failure,
        setup_mlflow,
    )
    from safetensors.torch import load_file  # noqa: PLC0415
    from shared.hardware import get_best_device  # noqa: PLC0415
    from torch.nn.utils import clip_grad_norm_  # noqa: PLC0415
    from torch.optim.lr_scheduler import (  # noqa: PLC0415
        CosineAnnealingLR,
        LinearLR,
        SequentialLR,
    )
    from transformers import (  # noqa: PLC0415
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    _patch_flash_attention()

    num_steps = min(args.num_steps, 5) if args.smoke_test else args.num_steps

    # --- Build HyperLoRA from scratch ---
    logger.info("Building HyperLoRA from scratch for %s", args.model_config_name)
    hc = build_from_scratch_hypernet_config(
        model_name=args.model_config_name,
        lora_r=args.lora_r,
        target_modules=args.target_modules,
    )
    hypernet = HyperLoRA(hc).to(torch.float32)
    hypernet.train()

    n_params = sum(p.numel() for p in hypernet.parameters())
    logger.info("HyperLoRA params: %d (all trainable)", n_params)

    # --- Load base model (single copy, NF4 quantized to fit in VRAM) ---
    logger.info("Loading base model: %s (NF4 quantized)", args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        llm_int8_enable_fp32_cpu_offload=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        output_hidden_states=True,
        quantization_config=bnb_config,
        device_map="auto",
    ).eval()
    base_model.requires_grad_(False)
    base_model.gradient_checkpointing_enable()

    # --- Load teacher adapter as functional lora_dict ---
    logger.info("Loading teacher adapter weights: %s", args.teacher_adapter)
    teacher_cfg_path = Path(args.teacher_adapter) / "adapter_config.json"
    with open(teacher_cfg_path) as f:
        teacher_adapter_cfg = json.load(f)

    teacher_weights = load_file(
        str(Path(args.teacher_adapter) / "adapter_model.safetensors"),
    )

    teacher_target_modules = teacher_adapter_cfg["target_modules"]
    teacher_r = teacher_adapter_cfg["r"]
    teacher_alpha = teacher_adapter_cfg["lora_alpha"]

    # Parse PEFT keys into lora_dict[short_name]["A"|"B"][layer_idx]
    _peft_key_re = re.compile(
        r"base_model\.model\.model\.layers\.(\d+)\..+\.(\w+)\.(lora_A|lora_B)\.weight"
    )
    _teacher_per_layer: dict[str, dict[str, dict[int, torch.Tensor]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for key, tensor in teacher_weights.items():
        m = _peft_key_re.match(key)
        if not m:
            continue
        layer_idx, short_name, ab = int(m.group(1)), m.group(2), m.group(3)
        ab_key = "A" if ab == "lora_A" else "B"
        _teacher_per_layer[short_name][ab_key][layer_idx] = tensor

    teacher_layer_indices = sorted(
        {idx for mod in _teacher_per_layer.values() for idx in mod["A"]}
    )

    # Stack into lora_dict format: [1, n_layers, r, dim]
    # Zero-pad layers where a module doesn't exist (hybrid attention arch).
    # PEFT stores lora_A as (r, d_in) and lora_B as (d_out, r).
    # apply_functional_lora expects both as (r, d), so B must be transposed.
    teacher_lora_dict: dict[str, dict[str, torch.Tensor]] = {}
    for mod_name in teacher_target_modules:
        if mod_name not in _teacher_per_layer:
            continue
        mod_a = _teacher_per_layer[mod_name]["A"]
        mod_b = _teacher_per_layer[mod_name]["B"]
        sample_a = next(iter(mod_a.values()))
        sample_b = next(iter(mod_b.values()))
        a_stack = torch.stack(
            [mod_a.get(i, torch.zeros_like(sample_a)) for i in teacher_layer_indices]
        ).unsqueeze(0)
        b_stack = torch.stack(
            [
                mod_b.get(i, torch.zeros_like(sample_b)).t()
                for i in teacher_layer_indices
            ]
        ).unsqueeze(0)
        teacher_lora_dict[mod_name] = {"A": a_stack, "B": b_stack}

    teacher_hc = SimpleNamespace(
        lora_config=SimpleNamespace(
            r=teacher_r,
            lora_alpha=teacher_alpha,
            target_modules=list(teacher_lora_dict.keys()),
        ),
        layer_indices=teacher_layer_indices,
    )
    logger.info(
        "Teacher adapter: r=%d, alpha=%d, %d modules, %d layers",
        teacher_r,
        teacher_alpha,
        len(teacher_lora_dict),
        len(teacher_layer_indices),
    )
    del teacher_weights

    # --- Device (base_model already on GPU via device_map="auto") ---
    # Teacher LoRA stays on CPU; transferred to GPU per-step (saves ~0.5 GB).
    device = torch.device(get_best_device())
    logger.info("Using device: %s", device)
    hypernet = hypernet.to(device)

    # --- Optimizer (8-bit paged Adam to fit in VRAM) ---
    trainable_params = list(hypernet.parameters())
    optimizer = PagedAdamW8bit(trainable_params, lr=args.lr)

    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=0.01, total_iters=args.warmup_steps),
            CosineAnnealingLR(
                optimizer,
                T_max=max(1, num_steps - args.warmup_steps),
                eta_min=1e-6,
            ),
        ],
        milestones=[args.warmup_steps],
    )

    # --- Data ---
    if args.smoke_test or not Path(args.dataset).exists():
        from model_training.d2l_data import generate_needle_dataset  # noqa: PLC0415

        records = generate_needle_dataset(n=20)
        logger.info("Using needle dataset (%d records)", len(records))
    else:
        all_records = load_jsonl(args.dataset)
        records, _ = split_by_task_id(all_records)
        logger.info("Loaded %d training records from %s", len(records), args.dataset)

    if not records:
        raise ValueError("No training records loaded.")

    # --- MLflow ---
    mlflow_ok = setup_mlflow(args.experiment_name, tracking_uri=None)
    if mlflow_ok:
        import mlflow  # noqa: PLC0415

        mlflow.start_run(run_name=f"{args.experiment_name}-{num_steps}steps")
        mlflow.log_params(
            {
                "base_model": args.base_model,
                "teacher_adapter": args.teacher_adapter,
                "lora_r": args.lora_r,
                "num_steps": num_steps,
                "lr": args.lr,
                "alpha": args.alpha,
                "temperature": args.temperature,
                "max_length": args.max_length,
                "activation_max_length": args.activation_max_length,
                "n_records": len(records),
                "n_params": n_params,
                "target_modules": ",".join(args.target_modules),
                "n_layers": len(hc.layer_indices),
                "base_model_quant": "nf4",
                "optimizer": "PagedAdamW8bit",
            }
        )

    layer_indices = list(hc.layer_indices)
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")
    final_loss = float("inf")
    step_losses: list[float] = []

    # --- Training loop ---
    logger.info("Starting training: %d steps", num_steps)
    skipped = 0
    step = 0
    try:
        for step in range(1, num_steps + 1):
            record = records[(step - 1) % len(records)]

            # Pre-check: skip records where context fills the entire
            # max_length, leaving no answer tokens for the loss.
            answer_start = len(
                tokenizer(
                    record["activation_text"],
                    truncation=True,
                    max_length=args.max_length,
                )["input_ids"]
            )
            teacher_tok_len = len(
                tokenizer(
                    record["teacher_text"],
                    truncation=True,
                    max_length=args.max_length,
                )["input_ids"]
            )
            if answer_start >= teacher_tok_len:
                skipped += 1
                if skipped <= 5 or skipped % 50 == 0:
                    logger.info(
                        "Step %d skipped (answer_start=%d >= seq_len=%d, total skipped=%d)",
                        step,
                        answer_start,
                        teacher_tok_len,
                        skipped,
                    )
                continue

            # Tokenize teacher_text (context + answer) at full max_length
            teacher_inputs = tokenizer(
                record["teacher_text"],
                return_tensors="pt",
                truncation=True,
                max_length=args.max_length,
            )
            teacher_inputs = {k: v.to(device) for k, v in teacher_inputs.items()}

            # Extract activations from base model (context only).
            # Uses shorter activation_max_length because output_hidden_states=True
            # stores all 32 layers of hidden states — O(n_layers * seq_len * hidden).
            features, attn_mask = extract_activations_with_model(
                text=record["activation_text"],
                model=base_model,
                tokenizer=tokenizer,
                layer_indices=layer_indices,
                max_length=args.activation_max_length,
            )

            # Hypernetwork forward (preserves autograd graph)
            lora_dict, _ = hypernet.generate_weights(features, attn_mask, None)

            # Teacher forward: move LoRA to GPU, run under no_grad, move back
            teacher_lora_gpu = {
                m: {k: v.to(device) for k, v in ab.items()}
                for m, ab in teacher_lora_dict.items()
            }
            with torch.no_grad():
                with apply_functional_lora(
                    base_model,
                    teacher_lora_gpu,
                    teacher_hc,
                ):
                    teacher_logits = base_model(
                        **teacher_inputs,
                        output_hidden_states=False,
                        use_cache=False,
                    ).logits.detach()
            del teacher_lora_gpu

            # Pre-slice teacher logits to answer span to free VRAM before
            # the memory-heavy student forward.  Full logits at 3072 tokens
            # with vocab ~152K = ~934 MB; the prefix before answer_start is
            # never used by the loss function.
            logit_start = max(0, answer_start - 1)
            seq_len = teacher_logits.shape[1]
            if answer_start < seq_len:
                teacher_logits = teacher_logits[:, logit_start:, :].contiguous()
            torch.cuda.empty_cache()

            # Student forward: base + hypernetwork-generated LoRA
            with apply_functional_lora(base_model, lora_dict, hc):
                student_logits = base_model(
                    **teacher_inputs,
                    output_hidden_states=False,
                    use_cache=False,
                ).logits
            del teacher_inputs

            # Slice student logits to match pre-sliced teacher span
            if answer_start < seq_len:
                student_logits = student_logits[:, logit_start:, :]

            # Chunked KL+CE loss: vocab=152K makes full-span softmax
            # intermediates too large (~600 MB per tensor).  Processing
            # 128 tokens at a time keeps peak at ~37 MB.
            loss, metrics = _chunked_kl_ce_loss(
                student_logits,
                teacher_logits,
                alpha=args.alpha,
                temperature=args.temperature,
            )

            loss.backward()
            clip_grad_norm_(trainable_params, args.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            step_loss = metrics["total_loss"]
            step_losses.append(step_loss)
            final_loss = step_loss
            if step_loss < best_loss:
                best_loss = step_loss

            if mlflow_ok:
                mlflow.log_metrics(metrics, step=step)

            logger.info(
                "Step %d/%d — loss=%.4f (kl=%.4f, ce=%.4f)",
                step,
                num_steps,
                metrics["total_loss"],
                metrics["kl_loss"],
                metrics["ce_loss"],
            )

            # Free graph-connected tensors to prevent cross-step accumulation
            del features, attn_mask, lora_dict, student_logits
            del teacher_logits, loss
            torch.cuda.empty_cache()

            # Checkpoint
            if step % args.checkpoint_every == 0 or step == num_steps:
                ckpt_path = ckpt_dir / f"ckpt-{step}.pt"
                torch.save(
                    {
                        "hypernet_state_dict": hypernet.state_dict(),
                        "hypernet_config": hc,
                        "base_model_name_or_path": args.base_model,
                        "teacher_adapter_path": args.teacher_adapter,
                        "step": step,
                        "attention_layer_indices": layer_indices,
                        "best_loss": best_loss,
                        "lora_r": args.lora_r,
                    },
                    ckpt_path,
                )
                logger.info("Checkpoint saved: %s", ckpt_path)
                if mlflow_ok:
                    mlflow.log_artifact(str(ckpt_path))

    except BaseException as exc:
        logger.error("Training failed at step %d: %s", step, exc)
        if mlflow_ok:
            mlflow.log_metrics({"failed_at_step": step}, step=step)
            _log_failure(exc)
        raise

    # Smoke test assertions
    if args.smoke_test:
        for i, sl in enumerate(step_losses):
            assert torch.isfinite(torch.tensor(sl)), (  # noqa: S101
                f"Smoke test: loss at step {i + 1} is not finite: {sl}"
            )

    # Save final checkpoint
    final_ckpt = ckpt_dir / "checkpoint.pt"
    torch.save(
        {
            "hypernet_state_dict": hypernet.state_dict(),
            "hypernet_config": hc,
            "base_model_name_or_path": args.base_model,
            "teacher_adapter_path": args.teacher_adapter,
            "step": num_steps,
            "attention_layer_indices": layer_indices,
            "best_loss": best_loss,
            "lora_r": args.lora_r,
        },
        final_ckpt,
    )
    logger.info("Final checkpoint: %s", final_ckpt)

    if mlflow_ok:
        mlflow.log_artifact(str(final_ckpt))
        mlflow.log_metrics(
            {
                "final_loss": final_loss,
                "best_loss": best_loss,
                "skipped_records": skipped,
            }
        )
        mlflow.end_run()

    logger.info(
        "Training complete: %d steps, best_loss=%.4f, final_loss=%.4f, skipped=%d",
        num_steps,
        best_loss,
        final_loss,
        skipped,
    )


if __name__ == "__main__":
    main()
