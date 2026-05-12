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
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any

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


def _s3_upload_bytes(s3_client: Any, bucket: str, key: str, data: bytes) -> None:
    """Upload raw bytes to S3 using a shared client."""
    s3_client.put_object(Bucket=bucket, Key=key, Body=data)


def _estimate_batch_size(max_length: int) -> int:
    """Estimate inference batch size from free VRAM after model loading."""
    import torch  # noqa: PLC0415

    if not torch.cuda.is_available():
        return 1
    free_bytes, _ = torch.cuda.mem_get_info()
    usable = int(free_bytes * 0.65)
    per_seq = int(1.6e9 * (max_length / 2048))
    bs = max(1, min(usable // per_seq, 32))
    gpu_name = torch.cuda.get_device_name()
    logger.info(
        "Auto batch size: %d (%s, %.1f GB free, ~%.1f GB/seq @ %d tokens)",
        bs,
        gpu_name,
        free_bytes / 1e9,
        per_seq / 1e9,
        max_length,
    )
    return bs


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
    parser.add_argument(
        "--top-k",
        type=int,
        default=64,
        help="Save only top-k logits per token (0 = full vocab). "
        "k=64 reduces per-token storage ~3880x with negligible KL error.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Inference batch size (0 = auto-detect from free VRAM).",
    )
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
    nf4_cache = (
        Path.home()
        / ".cache"
        / "rune"
        / "quantized_models"
        / args.base_model.replace("/", "--")
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if args.base_model_precision == "bf16":
        logger.info("Loading base model: %s (bf16)", args.base_model)
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        ).eval()
    elif nf4_cache.exists():
        logger.info("Loading cached NF4 model from %s", nf4_cache)
        base_model = AutoModelForCausalLM.from_pretrained(
            str(nf4_cache),
            device_map="auto",
        ).eval()
    else:
        from transformers import BitsAndBytesConfig  # noqa: PLC0415

        logger.info("Loading base model: %s (NF4, will cache)", args.base_model)
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
        try:
            nf4_cache.mkdir(parents=True, exist_ok=True)
            base_model.save_pretrained(str(nf4_cache))
            tokenizer.save_pretrained(str(nf4_cache))
            logger.info("Cached NF4 model to %s", nf4_cache)
        except Exception:
            logger.warning(
                "Failed to cache NF4 model — will re-quantize next run", exc_info=True
            )

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

    layer_indices = sorted({idx for mod in _per_layer.values() for idx in mod["A"]})

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
        logger.info(
            "Found %d existing .pt files in S3 — will skip those", len(already_done)
        )
    elif out_dir is not None:
        already_done = {f.name for f in out_dir.glob("*.pt")}
        if already_done:
            logger.info(
                "Found %d existing .pt files locally — will skip those",
                len(already_done),
            )

    # --- Pre-filter and pre-tokenize (CPU) ---
    pending: list[tuple[int, int, int, list[int]]] = []
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
        teacher_ids: list[int] = tokenizer(
            record["teacher_text"],
            truncation=True,
            max_length=args.max_length,
        )["input_ids"]
        seq_len = len(teacher_ids)

        if answer_start >= seq_len:
            n_skipped += 1
            continue

        pending.append((idx, answer_start, seq_len, teacher_ids))

    pending.sort(key=lambda x: x[2])

    batch_size = (
        args.batch_size
        if args.batch_size > 0
        else _estimate_batch_size(args.max_length)
    )
    logger.info(
        "%d records to process (batch_size=%d, resumed=%d, skipped=%d)",
        len(pending),
        batch_size,
        n_resumed,
        n_skipped,
    )

    # --- Batched inference under a single LoRA context ---
    pad_id = tokenizer.pad_token_id or 0
    top_k = args.top_k
    io_workers = 16 if use_s3 else 2
    max_pending = io_workers * 3
    io_pool = ThreadPoolExecutor(max_workers=io_workers)
    pending_futures: list[Future[Any]] = []

    # Shared S3 client — thread-safe, avoids per-upload client creation
    s3_client: object | None = None
    if use_s3:
        import boto3  # noqa: PLC0415

        s3_client = boto3.client("s3")

    def _drain_futures(force: bool = False) -> None:
        """Check completed futures for exceptions; block if *force*."""
        nonlocal pending_futures
        still_pending: list[Future[None]] = []
        for fut in pending_futures:
            if force or fut.done():
                fut.result()
            else:
                still_pending.append(fut)
        pending_futures = still_pending
        if not force and len(pending_futures) >= max_pending:
            pending_futures[0].result()
            pending_futures = pending_futures[1:]

    vocab_size: int = 0

    with (
        torch.inference_mode(),
        apply_functional_lora(base_model, teacher_lora_dict, teacher_hc),
    ):
        for batch_start in range(0, len(pending), batch_size):
            batch = pending[batch_start : batch_start + batch_size]
            max_len = max(sl for _, _, sl, _ in batch)

            input_ids = torch.full(
                (len(batch), max_len), pad_id, dtype=torch.long, device=device
            )
            attention_mask = torch.zeros(
                len(batch), max_len, dtype=torch.long, device=device
            )
            for i, (_, _, _, ids) in enumerate(batch):
                input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
                attention_mask[i, : len(ids)] = 1

            logits = base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,
                use_cache=False,
            ).logits

            if not vocab_size:
                vocab_size = logits.shape[-1]

            for i, (orig_idx, answer_start, seq_len, _) in enumerate(batch):
                filename = f"{orig_idx:06d}.pt"
                logit_start = max(0, answer_start - 1)
                raw_span = logits[i, logit_start:seq_len, :]  # (span, vocab)

                if top_k > 0:
                    vals, idxs = raw_span.to(torch.bfloat16).topk(
                        min(top_k, raw_span.shape[-1]), dim=-1
                    )
                    payload: dict = {
                        "top_values": vals.cpu(),
                        "top_indices": idxs.cpu().to(torch.int32),
                        "answer_start": answer_start,
                        "seq_len": seq_len,
                        "vocab_size": vocab_size,
                        "top_k": top_k,
                    }
                    span_tokens = raw_span.shape[0]
                else:
                    span_logits = raw_span.to(torch.bfloat16).cpu()
                    payload = {
                        "logits": span_logits.unsqueeze(0),
                        "answer_start": answer_start,
                        "seq_len": seq_len,
                    }
                    span_tokens = span_logits.shape[0]

                # Serialize on main thread — avoids 3x copy amplification in workers
                buf = io.BytesIO()
                torch.save(payload, buf)
                payload_bytes = buf.getvalue()
                del payload, buf

                if use_s3:
                    assert s3_client is not None  # noqa: S101
                    pending_futures.append(
                        io_pool.submit(
                            _s3_upload_bytes,
                            s3_client,
                            s3_bucket,
                            f"{s3_prefix}/{filename}",
                            payload_bytes,
                        )
                    )
                else:
                    assert out_dir is not None  # noqa: S101
                    pending_futures.append(
                        io_pool.submit(
                            (out_dir / filename).write_bytes,
                            payload_bytes,
                        )
                    )

                n_valid += 1
                total_logit_tokens += span_tokens

            del logits, input_ids, attention_mask
            _drain_futures()

            done = batch_start + len(batch)
            if done % (batch_size * 10) < batch_size or done >= len(pending):
                logger.info(
                    "Progress: %d/%d (valid=%d, skipped=%d, resumed=%d, io_pending=%d)",
                    done,
                    len(pending),
                    n_valid,
                    n_skipped,
                    n_resumed,
                    len(pending_futures),
                )

    _drain_futures(force=True)
    io_pool.shutdown()

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
        "top_k": top_k,
        "vocab_size": vocab_size,
    }

    if use_s3:
        assert s3_client is not None  # noqa: S101
        manifest_bytes = json.dumps(manifest, indent=2).encode()
        _s3_upload_bytes(
            s3_client, s3_bucket, f"{s3_prefix}/manifest.json", manifest_bytes
        )
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
