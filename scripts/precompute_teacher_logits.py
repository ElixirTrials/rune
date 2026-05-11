"""Pre-compute teacher logits for hypernetwork training.

Runs the base model + teacher LoRA adapter over every training record and
saves answer-span logits.  The training script loads these via
``--teacher-logits-dir`` instead of running a teacher forward pass each step,
freeing ~2 GB VRAM and eliminating the per-step teacher computation.

Supports two output modes:

- **Local** (``--output-dir``): writes .pt files to a local directory.
- **S3-direct** (``--s3-uri``): streams each .pt to S3 via in-memory buffer,
  zero local disk usage. On startup, lists S3 prefix to find already-done
  indices and skips them (idempotent resume).

Output layout (local or S3)::

    <prefix>/
        manifest.json      # metadata (model, adapter, max_length, record count)
        000000.pt          # {logits: Tensor, answer_start: int, seq_len: int}
        000001.pt
        ...

Usage:
    # Local:
    uv run python scripts/precompute_teacher_logits.py \\
        --teacher-adapter hpo_artifacts/best_diffloss_v1 \\
        --dataset data/mined/all_unrolled.jsonl \\
        --output-dir data/teacher_logits

    # Direct to S3 (no local disk usage, idempotent resume):
    uv run python scripts/precompute_teacher_logits.py \\
        --teacher-adapter hpo_artifacts/best_diffloss_v1 \\
        --dataset data/mined/all_unrolled.jsonl \\
        --s3-uri s3://my-bucket/rune-logit-cache/
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import re
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
warnings.filterwarnings("ignore", message=".*guard_size_oblivious.*")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bootstrap import setup_path  # type: ignore[import-not-found]

setup_path()

logger = logging.getLogger(__name__)


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    """Parse ``s3://bucket/prefix/`` into (bucket, prefix)."""
    if not uri.startswith("s3://"):
        raise ValueError(f"Expected s3:// URI, got: {uri}")
    without_scheme = uri[5:]
    bucket, _, prefix = without_scheme.partition("/")
    return bucket, prefix.rstrip("/")


def _s3_list_existing(bucket: str, prefix: str) -> set[str]:
    """List .pt object keys under *prefix* and return the set of basenames."""
    import boto3  # noqa: PLC0415

    s3 = boto3.client("s3")
    existing: set[str] = set()
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix + "/"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            basename = key.rsplit("/", 1)[-1]
            if basename.endswith(".pt"):
                existing.add(basename)
    return existing


def _s3_upload_bytes(bucket: str, key: str, data: bytes) -> None:
    """Upload raw bytes to S3."""
    import boto3  # noqa: PLC0415

    s3 = boto3.client("s3")
    s3.put_object(Bucket=bucket, Key=key, Body=data)


def _s3_upload_tensor_dict(
    bucket: str, prefix: str, filename: str, tensor_dict: dict
) -> None:
    """Serialize a dict via torch.save into memory and upload to S3."""
    import torch  # noqa: PLC0415

    buf = io.BytesIO()
    torch.save(tensor_dict, buf)
    _s3_upload_bytes(bucket, f"{prefix}/{filename}", buf.getvalue())


def main() -> None:
    from model_training.d2l_data import load_jsonl, split_by_task_id  # noqa: PLC0415

    parser = argparse.ArgumentParser(
        description="Pre-compute teacher logits for hypernetwork training"
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
        help="Path to training JSONL",
    )
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen3.5-9B")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Local directory to write .pt files. Required unless --s3-uri is set.",
    )
    parser.add_argument(
        "--s3-uri",
        type=str,
        default=None,
        help="S3 URI (e.g. s3://bucket/prefix/) to stream .pt files directly. "
        "Zero local disk usage. Resumes by listing existing S3 objects.",
    )
    parser.add_argument(
        "--base-model-precision",
        choices=["nf4", "bf16"],
        default="nf4",
        help="nf4 (~5 GB) or bf16 (~18 GB, better quality teacher signal).",
    )
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    if not args.output_dir and not args.s3_uri:
        parser.error("Either --output-dir or --s3-uri is required.")

    use_s3 = args.s3_uri is not None
    s3_bucket = ""
    s3_prefix = ""
    if use_s3:
        s3_bucket, s3_prefix = _parse_s3_uri(args.s3_uri)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    import torch  # noqa: PLC0415
    from model_training.d2l_lora import apply_functional_lora  # noqa: PLC0415
    from model_training.sakana_d2l import _patch_flash_attention  # noqa: PLC0415
    from safetensors.torch import load_file  # noqa: PLC0415
    from shared.hardware import get_best_device  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    _patch_flash_attention()

    # --- Load model ---
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if args.base_model_precision == "bf16":
        logger.info("Loading base model: %s (bf16)", args.base_model)
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        ).eval()
    else:
        from transformers import BitsAndBytesConfig  # noqa: PLC0415

        logger.info("Loading base model: %s (NF4)", args.base_model)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            llm_int8_enable_fp32_cpu_offload=True,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            quantization_config=bnb_config,
            device_map="auto",
        ).eval()

    device = torch.device(get_best_device())

    # --- Load teacher adapter as functional lora_dict ---
    logger.info("Loading teacher adapter: %s", args.teacher_adapter)
    teacher_path = Path(args.teacher_adapter)
    with open(teacher_path / "adapter_config.json") as f:
        teacher_cfg = json.load(f)

    teacher_weights = load_file(str(teacher_path / "adapter_model.safetensors"))
    teacher_r = teacher_cfg["r"]
    teacher_alpha = teacher_cfg["lora_alpha"]
    teacher_target_modules = teacher_cfg["target_modules"]

    _peft_key_re = re.compile(
        r"base_model\.model\.model\.layers\.(\d+)\..+\.(\w+)\.(lora_A|lora_B)\.weight"
    )
    _per_layer: dict[str, dict[str, dict[int, torch.Tensor]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for key, tensor in teacher_weights.items():
        m = _peft_key_re.match(key)
        if not m:
            continue
        layer_idx, short_name, ab = int(m.group(1)), m.group(2), m.group(3)
        ab_key = "A" if ab == "lora_A" else "B"
        _per_layer[short_name][ab_key][layer_idx] = tensor

    layer_indices = sorted(
        {idx for mod in _per_layer.values() for idx in mod["A"]}
    )

    teacher_lora_dict: dict[str, dict[str, torch.Tensor]] = {}
    for mod_name in teacher_target_modules:
        if mod_name not in _per_layer:
            continue
        mod_a = _per_layer[mod_name]["A"]
        mod_b = _per_layer[mod_name]["B"]
        sample_a = next(iter(mod_a.values()))
        sample_b = next(iter(mod_b.values()))
        a_stack = torch.stack(
            [mod_a.get(i, torch.zeros_like(sample_a)) for i in layer_indices]
        ).unsqueeze(0)
        b_stack = torch.stack(
            [mod_b.get(i, torch.zeros_like(sample_b)).t() for i in layer_indices]
        ).unsqueeze(0)
        teacher_lora_dict[mod_name] = {"A": a_stack, "B": b_stack}

    teacher_hc = SimpleNamespace(
        lora_config=SimpleNamespace(
            r=teacher_r,
            lora_alpha=teacher_alpha,
            target_modules=list(teacher_lora_dict.keys()),
        ),
        layer_indices=layer_indices,
    )
    del teacher_weights

    # Move teacher LoRA to GPU
    teacher_lora_dict = {
        m: {k: v.to(device) for k, v in ab.items()}
        for m, ab in teacher_lora_dict.items()
    }
    logger.info(
        "Teacher adapter: r=%d, alpha=%d, %d modules, %d layers",
        teacher_r,
        teacher_alpha,
        len(teacher_lora_dict),
        len(layer_indices),
    )

    # --- Load data ---
    if args.smoke_test:
        from model_training.d2l_data import generate_needle_dataset  # noqa: PLC0415

        records = generate_needle_dataset(n=10)
    else:
        all_records = load_jsonl(args.dataset)
        records, _ = split_by_task_id(all_records)
    logger.info("Processing %d records", len(records))

    # --- Output setup ---
    out_dir: Path | None = None
    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    # Build skip-set: indices already completed (idempotent resume)
    already_done: set[str] = set()
    if use_s3:
        logger.info("Listing existing objects in s3://%s/%s/ ...", s3_bucket, s3_prefix)
        already_done = _s3_list_existing(s3_bucket, s3_prefix)
        logger.info("Found %d existing .pt files in S3 — will skip those", len(already_done))
    elif out_dir is not None:
        already_done = {f.name for f in out_dir.glob("*.pt")}
        if already_done:
            logger.info("Found %d existing .pt files locally — will skip those", len(already_done))

    # --- Process records ---
    n_valid = 0
    n_skipped = 0
    n_resumed = 0
    total_logit_tokens = 0

    for idx, record in enumerate(records):
        filename = f"{idx:06d}.pt"

        if filename in already_done:
            n_resumed += 1
            n_valid += 1
            continue

        answer_start = len(
            tokenizer(
                record["activation_text"],
                truncation=True,
                max_length=args.max_length,
            )["input_ids"]
        )
        inputs = tokenizer(
            record["teacher_text"],
            return_tensors="pt",
            truncation=True,
            max_length=args.max_length,
        )
        seq_len = inputs["input_ids"].shape[1]

        if answer_start >= seq_len:
            n_skipped += 1
            continue

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            with apply_functional_lora(base_model, teacher_lora_dict, teacher_hc):
                logits = base_model(
                    **inputs, output_hidden_states=False, use_cache=False
                ).logits

        logit_start = max(0, answer_start - 1)
        span_logits = logits[:, logit_start:, :].to(torch.bfloat16).cpu()
        del logits, inputs
        torch.cuda.empty_cache()

        payload = {
            "logits": span_logits,
            "answer_start": answer_start,
            "seq_len": seq_len,
        }

        if use_s3:
            _s3_upload_tensor_dict(s3_bucket, s3_prefix, filename, payload)
        else:
            assert out_dir is not None  # noqa: S101
            torch.save(payload, out_dir / filename)

        n_valid += 1
        total_logit_tokens += span_logits.shape[1]

        if (idx + 1) % 50 == 0 or idx == len(records) - 1:
            logger.info(
                "Progress: %d/%d (valid=%d, skipped=%d, resumed=%d)",
                idx + 1,
                len(records),
                n_valid,
                n_skipped,
                n_resumed,
            )

    # --- Write manifest ---
    manifest = {
        "base_model": args.base_model,
        "teacher_adapter": str(args.teacher_adapter),
        "max_length": args.max_length,
        "base_model_precision": args.base_model_precision,
        "n_records": len(records),
        "n_valid": n_valid,
        "n_skipped": n_skipped,
        "n_resumed": n_resumed,
        "total_logit_tokens": total_logit_tokens,
        "storage_dtype": "bfloat16",
    }

    if use_s3:
        manifest_bytes = json.dumps(manifest, indent=2).encode()
        _s3_upload_bytes(s3_bucket, f"{s3_prefix}/manifest.json", manifest_bytes)
        dest = f"s3://{s3_bucket}/{s3_prefix}/"
    else:
        assert out_dir is not None  # noqa: S101
        with open(out_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        dest = str(out_dir)

    logger.info(
        "Done: %d valid records saved to %s (skipped %d, resumed %d, ~%d total tokens)",
        n_valid,
        dest,
        n_skipped,
        n_resumed,
        total_logit_tokens,
    )


if __name__ == "__main__":
    main()
