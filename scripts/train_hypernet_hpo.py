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
import gc
import io
import logging
import os
import signal
import sys
import warnings
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
warnings.filterwarnings("ignore", message=".*guard_size_oblivious.*")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bootstrap import setup_path  # type: ignore[import-not-found]

setup_path()

logger = logging.getLogger(__name__)

_LOSS_CHUNK = 48


def _chunked_kl_ce_loss(
    student_logits: Any,
    teacher_logits: Any,
    *,
    alpha: float,
    temperature: float,
) -> tuple[Any, dict[str, float]]:
    """KL+CE loss chunked along the token dimension to cap VRAM.

    Each chunk is gradient-checkpointed so autograd discards forward
    intermediates (log_softmax, softmax — ~14 MB each at vocab 152K)
    and recomputes them during backward.

    Teacher logits may arrive on CPU (offload path, default) or GPU
    (large-VRAM path). Chunks are moved to the student device via
    .to(device) which is a no-op when already co-located.
    """
    import torch  # noqa: PLC0415
    import torch.nn.functional as F  # noqa: PLC0415, N812
    from torch.utils.checkpoint import checkpoint  # noqa: PLC0415

    n_tokens = student_logits.shape[1]
    if n_tokens == 0:
        zero = torch.tensor(0.0, device=student_logits.device, requires_grad=True)
        return zero, {
            "kl_loss": 0.0,
            "ce_loss": 0.0,
            "total_loss": 0.0,
            "top1_agreement": 0.0,
        }

    vocab = student_logits.shape[-1]

    # Teacher logits may be on CPU (offload path) or GPU (large-VRAM path).
    # _chunk_fn handles both via .to(device) which is a no-op when already there.
    teacher_ref = teacher_logits
    del teacher_logits

    def _chunk_fn(
        s_chunk: torch.Tensor,
        t_chunk: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        t_chunk = t_chunk.to(s_chunk.device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            t_soft = F.softmax(t_chunk / temperature, dim=-1)
            t_hard = t_chunk.argmax(-1).reshape(-1)
            del t_chunk

            s_log = F.log_softmax(s_chunk / temperature, dim=-1)
            kl = F.kl_div(s_log, t_soft, reduction="sum")
            del s_log, t_soft

            ce = F.cross_entropy(
                s_chunk.reshape(-1, vocab),
                t_hard,
                reduction="sum",
            )
            s_hard = s_chunk.argmax(-1).reshape(-1)
            top1_matches = (s_hard == t_hard).sum()
            del s_hard, t_hard
        return kl, ce, top1_matches

    device = student_logits.device
    kl_sum = torch.zeros(1, device=device)
    ce_sum = torch.zeros(1, device=device)
    top1_match_sum = 0
    total_elements = 0

    for i in range(0, n_tokens, _LOSS_CHUNK):
        s = student_logits[:, i : i + _LOSS_CHUNK, :]
        t = teacher_ref[:, i : i + _LOSS_CHUNK, :].contiguous()
        chunk_elems = s.shape[0] * s.shape[1]

        kl, ce, matches = checkpoint(_chunk_fn, s, t, use_reentrant=False)
        kl_sum = kl_sum + kl
        ce_sum = ce_sum + ce
        top1_match_sum += matches.item()
        total_elements += chunk_elems

    del teacher_ref

    kl = kl_sum / total_elements * temperature**2
    ce = ce_sum / total_elements
    total = alpha * kl + (1.0 - alpha) * ce

    return total, {
        "kl_loss": kl.item(),
        "ce_loss": ce.item(),
        "total_loss": total.item(),
        "top1_agreement": top1_match_sum / total_elements
        if total_elements > 0
        else 0.0,
    }


def _full_kl_ce_loss(
    student_logits: Any,
    teacher_logits: Any,
    *,
    alpha: float,
    temperature: float,
) -> tuple[Any, dict[str, float]]:
    """KL+CE loss computed in one pass — no chunking, no per-chunk checkpointing.

    Requires ~2.5 GB extra VRAM (full softmax intermediates at 152K vocab)
    but avoids recomputation overhead. Use on >=40 GB GPUs.
    """
    import torch  # noqa: PLC0415
    import torch.nn.functional as F  # noqa: PLC0415, N812

    n_tokens = student_logits.shape[1]
    if n_tokens == 0:
        zero = torch.tensor(0.0, device=student_logits.device, requires_grad=True)
        return zero, {
            "kl_loss": 0.0,
            "ce_loss": 0.0,
            "total_loss": 0.0,
            "top1_agreement": 0.0,
        }

    vocab = student_logits.shape[-1]
    total_elements = student_logits.shape[0] * n_tokens

    teacher_logits = teacher_logits.to(student_logits.device)

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        t_soft = F.softmax(teacher_logits / temperature, dim=-1)
        t_hard = teacher_logits.argmax(-1).reshape(-1)
        del teacher_logits

        s_log = F.log_softmax(student_logits / temperature, dim=-1)
        kl = F.kl_div(s_log, t_soft, reduction="sum") / total_elements * temperature**2
        del s_log, t_soft

        ce = (
            F.cross_entropy(student_logits.reshape(-1, vocab), t_hard, reduction="sum")
            / total_elements
        )
        s_hard = student_logits.argmax(-1).reshape(-1)
        top1_agreement = (s_hard == t_hard).float().mean().item()
        del s_hard, t_hard

    total = alpha * kl + (1.0 - alpha) * ce
    return total, {
        "kl_loss": kl.item(),
        "ce_loss": ce.item(),
        "total_loss": total.item(),
        "top1_agreement": top1_agreement,
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
        required=False,
        default="hpo_artifacts/best_diffloss_v1",
        help="Path to HPO-tuned LoRA adapter directory (unused with --teacher-logits-dir)",
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
        "--checkpoint-dir",
        type=str,
        default=os.environ.get("SM_HP_CHECKPOINT_DIR", "checkpoints/hypernet_hpo"),
    )
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--experiment-name", type=str, default="hypernet-hpo")
    parser.add_argument(
        "--mlflow-tracking-uri",
        type=str,
        default=None,
        help="MLflow tracking server URI. Defaults to MLFLOW_TRACKING_URI env var.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=0,
        help="Early-stop after this many steps without loss improvement. 0 = disabled.",
    )
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--target-modules", nargs="+", default=_l["target_modules"])

    # --- Precomputed teacher logits ---
    parser.add_argument(
        "--teacher-logits-dir",
        type=str,
        default=None,
        help="Path or S3 URI (s3://bucket/prefix) to precomputed teacher logits. "
        "When set, skips teacher LoRA loading and teacher forward pass entirely.",
    )

    # --- VRAM configuration ---
    # Defaults are None so --high-vram can set them. Resolved after parse.
    parser.add_argument(
        "--high-vram",
        action="store_true",
        help=">=40 GB GPU: bf16 model, fp32 AdamW, no gradient checkpointing, "
        "full loss, everything on GPU. Individual flags override.",
    )
    parser.add_argument(
        "--base-model-precision",
        choices=["nf4", "bf16"],
        default=None,
        help="nf4 (~5 GB, default) or bf16 (~18 GB, better distillation quality).",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Gradient checkpointing on base model. Saves VRAM, costs ~30%% compute.",
    )
    parser.add_argument(
        "--optimizer-type",
        choices=["adamw-8bit", "adamw"],
        default=None,
        help="adamw-8bit (default) saves ~50%% optimizer VRAM; adamw uses fp32 states.",
    )
    parser.add_argument(
        "--offload-teacher-logits",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Move teacher logits to CPU before student forward. Saves ~590 MB VRAM.",
    )
    parser.add_argument(
        "--offload-teacher-lora",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Keep teacher LoRA on CPU, transfer per-step. Saves ~0.5 GB.",
    )
    parser.add_argument(
        "--chunk-loss",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Chunk loss computation (48 tokens). Saves ~2 GB; adds recompute overhead.",
    )
    args = parser.parse_args()

    # --- Resolve VRAM defaults: --high-vram sets aggressive defaults,
    # individual flags (when explicitly provided) override. ---
    high = args.high_vram
    vram_defaults = {
        "base_model_precision": ("bf16" if high else "nf4"),
        "gradient_checkpointing": (False if high else True),
        "optimizer_type": ("adamw" if high else "adamw-8bit"),
        "offload_teacher_logits": (False if high else True),
        "offload_teacher_lora": (False if high else True),
        "chunk_loss": (False if high else True),
    }
    for attr, default in vram_defaults.items():
        if getattr(args, attr) is None:
            setattr(args, attr, default)

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

    use_precomputed = args.teacher_logits_dir is not None
    logits_is_s3 = False
    logits_s3_bucket = ""
    logits_s3_prefix = ""
    logits_s3_client: Any = None
    logits_dir: Path | None = None

    # Auto-fetch teacher adapter from MLflow/S3 if missing (skip if precomputed)
    if not use_precomputed:
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

    import torch  # noqa: PLC0415
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
        mlflow_download_latest_checkpoint,
        mlflow_log_checkpoint,
        setup_mlflow,
    )
    from shared.hardware import get_best_device  # noqa: PLC0415
    from torch.nn.utils import clip_grad_norm_  # noqa: PLC0415
    from torch.optim.lr_scheduler import (  # noqa: PLC0415
        CosineAnnealingLR,
        LinearLR,
        SequentialLR,
    )
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

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

    # Warm-start scaler_B: zero init creates a dead gradient bottleneck where
    # all gradients through the LoRA B path (and thus the perceiver/head) are
    # zero until scaler_B grows from 0.  Initialising to a small positive value
    # unblocks gradient flow from step 1.
    with torch.no_grad():
        for name, param in hypernet.named_parameters():
            if "scaler_B" in name and param.abs().max() == 0:
                param.fill_(0.01)
                logger.info("Warm-started %s → 0.01 (was zeros)", name)

    n_params = sum(p.numel() for p in hypernet.parameters())
    logger.info("HyperLoRA params: %d (all trainable)", n_params)

    # --- Load base model ---
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if args.base_model_precision == "bf16":
        logger.info("Loading base model: %s (bf16)", args.base_model)
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            output_hidden_states=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        ).eval()
    else:
        from transformers import BitsAndBytesConfig  # noqa: PLC0415

        logger.info("Loading base model: %s (NF4 quantized)", args.base_model)
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
    if args.gradient_checkpointing:
        # use_reentrant=False: non-reentrant checkpointing is required for
        # functional LoRA because all LoRA tensors share an upstream graph
        # (the hypernetwork output).  Reentrant mode creates nested
        # torch.autograd.backward() calls per block; the first block's
        # backward frees the shared graph, causing "backward through the
        # graph a second time" on subsequent blocks.
        #
        # determinism_check="none": NF4 dequantization is not bitwise
        # deterministic across calls (cached vs recomputed), so the default
        # tensor-count check would fail on recomputation.
        base_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={
                "use_reentrant": False,
                "determinism_check": "none",
            }
        )
        # gradient_checkpointing activates only when self.training is True
        # (transformers checks `self.gradient_checkpointing and self.training`).
        # .eval() above set training=False. Safe to re-enable: all params have
        # requires_grad=False, Qwen3 uses 0 dropout, no batch norm.
        base_model.train()

    # --- Load teacher adapter (skip if using precomputed logits) ---
    teacher_lora_dict: dict[str, dict[str, torch.Tensor]] | None = None
    teacher_hc: Any = None
    if not use_precomputed:
        import re  # noqa: PLC0415
        from collections import defaultdict  # noqa: PLC0415
        from types import SimpleNamespace  # noqa: PLC0415

        from safetensors.torch import load_file  # noqa: PLC0415

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

        teacher_lora_dict = {}
        for mod_name in teacher_target_modules:
            if mod_name not in _teacher_per_layer:
                continue
            mod_a = _teacher_per_layer[mod_name]["A"]
            mod_b = _teacher_per_layer[mod_name]["B"]
            sample_a = next(iter(mod_a.values()))
            sample_b = next(iter(mod_b.values()))
            a_stack = torch.stack(
                [
                    mod_a.get(i, torch.zeros_like(sample_a))
                    for i in teacher_layer_indices
                ]
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

        # Optionally keep teacher LoRA on GPU permanently
        if not args.offload_teacher_lora:
            device_tmp = torch.device(get_best_device())
            teacher_lora_dict = {
                m: {k: v.to(device_tmp) for k, v in ab.items()}
                for m, ab in teacher_lora_dict.items()
            }
            logger.info("Teacher LoRA pinned to GPU")
    else:
        logits_src = args.teacher_logits_dir
        logits_is_s3 = logits_src.startswith("s3://")
        if logits_is_s3:
            import boto3  # noqa: PLC0415

            _s3_parts = logits_src[5:].split("/", 1)
            logits_s3_bucket = _s3_parts[0]
            logits_s3_prefix = _s3_parts[1].rstrip("/") if len(_s3_parts) > 1 else ""
            logits_s3_client = boto3.client("s3")
            try:
                resp = logits_s3_client.get_object(
                    Bucket=logits_s3_bucket,
                    Key=f"{logits_s3_prefix}/manifest.json",
                )
                _manifest = json.loads(resp["Body"].read())
                logger.info(
                    "Using precomputed teacher logits from %s (%d records)",
                    logits_src,
                    _manifest.get("n_valid", "?"),
                )
            except Exception:
                logger.info("Using precomputed teacher logits from %s", logits_src)
        else:
            logits_dir = Path(logits_src)
            manifest_path = logits_dir / "manifest.json"
            if manifest_path.exists():
                with open(manifest_path) as f:
                    _manifest = json.load(f)
                logger.info(
                    "Using precomputed teacher logits from %s (%d records)",
                    logits_dir,
                    _manifest.get("n_valid", "?"),
                )
            else:
                logger.info("Using precomputed teacher logits from %s", logits_dir)

    # --- Device ---
    device = torch.device(get_best_device())
    logger.info("Using device: %s", device)
    hypernet = hypernet.to(device)

    # --- Optimizer ---
    trainable_params = list(hypernet.parameters())
    optimizer: torch.optim.Optimizer
    if args.optimizer_type == "adamw":
        optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)
        logger.info("Optimizer: AdamW (fp32)")
    else:
        from bitsandbytes.optim import PagedAdamW8bit  # noqa: PLC0415

        optimizer = PagedAdamW8bit(trainable_params, lr=args.lr)
        logger.info("Optimizer: PagedAdamW8bit")

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
    mlflow_ok = setup_mlflow(
        args.experiment_name, tracking_uri=args.mlflow_tracking_uri
    )
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
                "base_model_precision": args.base_model_precision,
                "optimizer_type": args.optimizer_type,
                "gradient_checkpointing": args.gradient_checkpointing,
                "offload_teacher_logits": args.offload_teacher_logits,
                "offload_teacher_lora": args.offload_teacher_lora,
                "chunk_loss": args.chunk_loss,
                "use_precomputed_logits": use_precomputed,
                "high_vram": args.high_vram,
                "patience": args.patience,
            }
        )

    layer_indices = list(hc.layer_indices)
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")
    final_loss = float("inf")
    steps_without_improvement = 0
    step_losses: list[float] = []

    # --- Atomic checkpoint helpers ---
    def _build_ckpt_state(step_num: int) -> dict[str, Any]:
        return {
            "hypernet_state_dict": hypernet.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "hypernet_config": hc,
            "base_model_name_or_path": args.base_model,
            "teacher_adapter_path": args.teacher_adapter,
            "step": step_num,
            "attention_layer_indices": layer_indices,
            "best_loss": best_loss,
            "steps_without_improvement": steps_without_improvement,
            "lora_r": args.lora_r,
        }

    def _save_atomic(path: Path, state: dict[str, Any]) -> None:
        tmp = path.with_suffix(".pt.tmp")
        torch.save(state, tmp)
        os.replace(tmp, path)

    def _prune_checkpoints(keep: int = 3) -> None:
        ckpts = sorted(
            (p for p in ckpt_dir.glob("ckpt-[0-9]*.pt") if "-emergency" not in p.name),
            key=lambda p: int(p.stem.split("-")[1]),
        )
        for old in ckpts[:-keep]:
            old.unlink()
            logger.debug("Pruned checkpoint: %s", old)

    # --- SIGTERM handler (SageMaker spot reclamation) ---
    _shutdown = [False]

    def _handle_sigterm(signum: int, frame: Any) -> None:
        _shutdown[0] = True
        logger.warning("SIGTERM received — will checkpoint and exit after current step")

    signal.signal(signal.SIGTERM, _handle_sigterm)

    # Clean up leftover temp files from prior atomic writes
    for _tmp in ckpt_dir.glob("*.pt.tmp"):
        _tmp.unlink()

    # --- Resume from checkpoint (local first, then MLflow/S3) ---
    start_step = 0
    ckpt_files = sorted(
        (p for p in ckpt_dir.glob("ckpt-[0-9]*.pt") if "-emergency" not in p.name),
        key=lambda p: int(p.stem.split("-")[1]),
    )
    if not ckpt_files and not args.smoke_test and mlflow_ok:
        mlflow_ckpt = mlflow_download_latest_checkpoint(args.experiment_name, ckpt_dir)
        if mlflow_ckpt is not None:
            ckpt_files = [mlflow_ckpt]
            logger.info("Downloaded checkpoint from MLflow: %s", mlflow_ckpt)
    if ckpt_files and not args.smoke_test:
        latest = ckpt_files[-1]
        logger.info("Resuming from checkpoint: %s", latest)
        ckpt_data = torch.load(latest, map_location=device, weights_only=False)
        hypernet.load_state_dict(ckpt_data["hypernet_state_dict"])
        if "optimizer_state_dict" in ckpt_data:
            optimizer.load_state_dict(ckpt_data["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt_data:
            scheduler.load_state_dict(ckpt_data["scheduler_state_dict"])
        start_step = ckpt_data["step"]
        best_loss = ckpt_data.get("best_loss", float("inf"))
        steps_without_improvement = ckpt_data.get("steps_without_improvement", 0)
        logger.info("Resumed at step %d (best_loss=%.4f)", start_step, best_loss)
        del ckpt_data
        if mlflow_ok:
            mlflow.log_metrics({"resumed_from_step": start_step})

    # --- Training loop ---
    logger.info("Starting training: %d steps (from step %d)", num_steps, start_step)
    skipped = 0
    step = start_step
    try:
        for step in range(start_step + 1, num_steps + 1):
            record = records[(step - 1) % len(records)]

            record_idx = (step - 1) % len(records)

            # --- Load or compute teacher logits ---
            if use_precomputed:
                filename = f"{record_idx:06d}.pt"
                if logits_is_s3:
                    try:
                        obj = logits_s3_client.get_object(
                            Bucket=logits_s3_bucket,
                            Key=f"{logits_s3_prefix}/{filename}",
                        )
                        cached = torch.load(
                            io.BytesIO(obj["Body"].read()),
                            map_location="cpu",
                            weights_only=True,
                        )
                    except logits_s3_client.exceptions.NoSuchKey:
                        skipped += 1
                        if skipped <= 5 or skipped % 50 == 0:
                            logger.info(
                                "Step %d skipped (no precomputed logits at s3://.../%s, total skipped=%d)",
                                step,
                                filename,
                                skipped,
                            )
                        continue
                else:
                    assert logits_dir is not None  # noqa: S101
                    cache_path = logits_dir / filename
                    if not cache_path.exists():
                        skipped += 1
                        if skipped <= 5 or skipped % 50 == 0:
                            logger.info(
                                "Step %d skipped (no precomputed logits at %s, total skipped=%d)",
                                step,
                                cache_path,
                                skipped,
                            )
                        continue
                    cached = torch.load(
                        cache_path, map_location="cpu", weights_only=True
                    )
                answer_start: int = int(cached["answer_start"])
                seq_len: int = int(cached["seq_len"])
                logit_start = max(0, answer_start - 1)
                if "logits" in cached:
                    teacher_logits = cached["logits"]
                    del cached
                else:
                    tv = cached["top_values"]
                    ti = cached["top_indices"].long()
                    vs = int(cached["vocab_size"])
                    del cached
                    teacher_logits = torch.full(
                        (1, tv.shape[0], vs), -1e4, dtype=tv.dtype
                    )
                    teacher_logits[0].scatter_(1, ti, tv)
                    del tv, ti
            else:
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
                seq_len = teacher_tok_len

            # Tokenize teacher_text (context + answer) for the student forward
            teacher_inputs = tokenizer(
                record["teacher_text"],
                return_tensors="pt",
                truncation=True,
                max_length=args.max_length,
            )
            teacher_inputs = {k: v.to(device) for k, v in teacher_inputs.items()}

            # Extract activations from base model (context only)
            features, attn_mask = extract_activations_with_model(
                text=record["activation_text"],
                model=base_model,
                tokenizer=tokenizer,
                layer_indices=layer_indices,
                max_length=args.activation_max_length,
            )

            # Hypernetwork forward (preserves autograd graph)
            lora_dict, _ = hypernet.generate_weights(features, attn_mask, None)

            # --- Teacher forward (only when not using precomputed logits) ---
            if not use_precomputed:
                assert teacher_lora_dict is not None  # noqa: S101
                if args.offload_teacher_lora:
                    teacher_lora_gpu = {
                        m: {k: v.to(device) for k, v in ab.items()}
                        for m, ab in teacher_lora_dict.items()
                    }
                else:
                    teacher_lora_gpu = teacher_lora_dict
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
                if args.offload_teacher_lora:
                    del teacher_lora_gpu

                logit_start = max(0, answer_start - 1)
                if args.offload_teacher_logits:
                    if answer_start < seq_len:
                        teacher_logits = teacher_logits[:, logit_start:, :].to("cpu")
                    else:
                        teacher_logits = teacher_logits.to("cpu")
                    gc.collect()
                    torch.cuda.empty_cache()
                else:
                    if answer_start < seq_len:
                        teacher_logits = teacher_logits[:, logit_start:, :].clone()
                    gc.collect()
                    torch.cuda.empty_cache()

            # Student forward + backward must both live inside the functional
            # LoRA context: reentrant gradient checkpointing recomputes each
            # transformer block during backward, so the LoRA monkey-patches
            # must still be active or the recomputed graph omits the LoRA path
            # and the hypernetwork receives zero gradients.
            with apply_functional_lora(base_model, lora_dict, hc):
                student_logits = base_model(
                    **teacher_inputs,
                    output_hidden_states=False,
                    use_cache=False,
                ).logits
                del teacher_inputs
                gc.collect()
                torch.cuda.empty_cache()

                # Slice student logits to match the teacher answer span
                if answer_start < seq_len:
                    student_logits = student_logits[:, logit_start:, :].contiguous()

                # Loss
                _loss_fn = _chunked_kl_ce_loss if args.chunk_loss else _full_kl_ce_loss
                loss, metrics = _loss_fn(
                    student_logits,
                    teacher_logits,
                    alpha=args.alpha,
                    temperature=args.temperature,
                )

                loss.backward()

            # Gradient diagnostics: log raw norm before clipping so we can
            # verify gradient flow through the hypernetwork.
            raw_grad_norm = clip_grad_norm_(trainable_params, float("inf"))
            clip_grad_norm_(trainable_params, args.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            step_loss = metrics["total_loss"]
            step_losses.append(step_loss)
            final_loss = step_loss
            if step_loss < best_loss:
                best_loss = step_loss
                steps_without_improvement = 0
            else:
                steps_without_improvement += 1

            if mlflow_ok:
                metrics["grad_norm_raw"] = raw_grad_norm.item()
                mlflow.log_metrics(metrics, step=step)

            logger.info(
                "Step %d/%d — loss=%.4f (kl=%.4f, ce=%.4f, top1=%.3f) grad_norm=%.4e",
                step,
                num_steps,
                metrics["total_loss"],
                metrics["kl_loss"],
                metrics["ce_loss"],
                metrics["top1_agreement"],
                raw_grad_norm.item(),
            )

            # Free graph-connected tensors to prevent cross-step accumulation
            del features, attn_mask, lora_dict, student_logits
            del teacher_logits, loss, metrics
            gc.collect()
            torch.cuda.empty_cache()

            # Checkpoint (atomic write, every step during warmup)
            in_warmup = step <= args.warmup_steps
            should_ckpt = (
                step % args.checkpoint_every == 0 or step == num_steps or in_warmup
            )
            if should_ckpt:
                ckpt_path = ckpt_dir / f"ckpt-{step}.pt"
                _save_atomic(ckpt_path, _build_ckpt_state(step))
                logger.info("Checkpoint saved: %s", ckpt_path)
                gc.collect()
                torch.cuda.empty_cache()
                if not in_warmup:
                    _prune_checkpoints()
                should_upload = step % args.checkpoint_every == 0 or step == num_steps
                if mlflow_ok and should_upload:
                    mlflow_log_checkpoint(str(ckpt_path))

            if _shutdown[0]:
                logger.warning("Shutting down after step %d (SIGTERM)", step)
                break

            if (
                args.patience > 0
                and step > args.warmup_steps
                and steps_without_improvement >= args.patience
            ):
                logger.info(
                    "Early stopping at step %d: no improvement for %d steps "
                    "(best_loss=%.4f)",
                    step,
                    args.patience,
                    best_loss,
                )
                if mlflow_ok:
                    mlflow.log_metrics({"early_stopped_at_step": step}, step=step)
                break

    except KeyboardInterrupt:
        logger.warning(
            "KeyboardInterrupt at step %d — saving emergency checkpoint", step
        )
        if step > start_step:
            _save_atomic(
                ckpt_dir / f"ckpt-{step}-emergency.pt",
                _build_ckpt_state(step),
            )
        if mlflow_ok:
            mlflow.log_metrics({"interrupted_at_step": step}, step=step)
            mlflow.end_run(status="KILLED")
        raise
    except Exception as exc:
        logger.error("Training failed at step %d: %s", step, exc)
        is_oom = isinstance(exc, torch.cuda.OutOfMemoryError)
        if step > start_step and not is_oom:
            _save_atomic(
                ckpt_dir / f"ckpt-{step}-emergency.pt",
                _build_ckpt_state(step),
            )
        elif is_oom:
            logger.warning("Skipping emergency checkpoint (GPU OOM)")
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
    final_step = step if _shutdown[0] else num_steps
    final_ckpt = ckpt_dir / "checkpoint.pt"
    _save_atomic(final_ckpt, _build_ckpt_state(final_step))
    logger.info("Final checkpoint: %s", final_ckpt)

    if mlflow_ok:
        mlflow_log_checkpoint(str(final_ckpt))
        mlflow_log_checkpoint(str(final_ckpt), artifact_path="")
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
