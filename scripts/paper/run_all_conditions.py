"""Master runner for paper Table 2: all 5 conditions.

Conditions:
    (i)   Frozen base — Qwen 3.5 9B + DeltaCoder warm-start LoRA (substrate)
    (ii)  Trajectory-aware RAG — substrate + retrieved trajectory context
    (iii) Direct PEFT QLoRA — substrate + best HPO-tuned LoRA
    (iv)  TTT-E2E — substrate + test-time MLP fine-tuning
    (v)   Rune — iterative adapter generation with test-feedback retry loop

Usage:
    uv run python scripts/paper/run_all_conditions.py \
        --conditions i ii iii iv v \
        --benchmarks humaneval livecodebench \
        --hypernet-checkpoint path/to/checkpoint.bin \
        --output evaluation_results/table2.json
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bootstrap import setup_path  # type: ignore[import-not-found]

setup_path()

logger = logging.getLogger(__name__)

HPO_BEST_RUN_ID = "e9c9760f816f46948197519e1c905273"
HPO_S3_PREFIX = (
    "s3://elixirtrials-949678234935-eu-west-2-artifacts"
    f"/mlflow/artifacts/3/{HPO_BEST_RUN_ID}/artifacts"
)


def flush_partial_results(
    all_results: dict[str, dict[str, float | None]],
    output_path: Path,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Write current results to disk after each condition completes.

    Ensures partial results survive crashes. The file is overwritten
    on each call with the latest cumulative results.
    """
    table = assemble_table2(all_results)
    if metadata:
        table["metadata"] = metadata
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(table, indent=2))
    logger.info("Flushed %d condition(s) to %s", len(all_results), output_path)


def fetch_best_hpo_adapter(
    dest: Path,
    run_id: str = HPO_BEST_RUN_ID,
    s3_prefix: str = HPO_S3_PREFIX,
    mlflow_tracking_uri: str | None = None,
) -> Path:
    """Fetch the best HPO adapter from S3 or MLflow if not already on disk.

    Tries S3 first, falls back to MLflow CLI. Returns the path to the
    adapter directory containing adapter_config.json.

    Raises:
        FileNotFoundError: If neither S3 nor MLflow download succeeds.
    """
    if (dest / "adapter_config.json").exists():
        logger.info("HPO adapter already present at %s", dest)
        return dest

    dest.mkdir(parents=True, exist_ok=True)
    logger.info("HPO adapter not found at %s — fetching from S3...", dest)

    s3_ok = (
        subprocess.run(
            ["aws", "s3", "cp", f"{s3_prefix}/", f"{dest}/", "--recursive"],
            capture_output=True,
        ).returncode
        == 0
    )

    if not s3_ok:
        logger.info("S3 download failed — trying MLflow CLI...")
        env = None
        if mlflow_tracking_uri:
            import os

            env = {**os.environ, "MLFLOW_TRACKING_URI": mlflow_tracking_uri}
        subprocess.run(
            [
                "uv",
                "run",
                "mlflow",
                "artifacts",
                "download",
                "--run-id",
                run_id,
                "--dst-path",
                str(dest),
            ],
            capture_output=True,
            env=env,
        )

    if not (dest / "adapter_config.json").exists():
        raise FileNotFoundError(
            f"Could not fetch HPO adapter. Neither S3 nor MLflow download "
            f"produced adapter_config.json in {dest}"
        )
    logger.info("HPO adapter downloaded to %s", dest)
    return dest


CONDITION_LABELS = {
    "i": "Frozen base (substrate)",
    "ii": "Trajectory-aware RAG",
    "iii": "Direct PEFT QLoRA",
    "iv": "TTT-E2E",
    "v": "Rune iterative (ours)",
}

DEFAULT_BASE_MODEL = "Qwen/Qwen3.5-9B"
DEFAULT_WARM_START = "danielcherubini/Qwen3.5-DeltaCoder-9B"


def _run_benchmarks(
    stack: Any,
    benchmarks: list[str],
    max_samples: int | None = None,
    checkpoint_dir: Path | str | None = None,
    condition_label: str | None = None,
) -> dict[str, float | None]:
    """Run benchmarks against an AdapterStack, returning {benchmark: pass_at_1}."""
    from evaluation.benchmarks import run_benchmark

    try:
        import mlflow

        mlflow_active = mlflow.active_run() is not None
    except Exception:
        mlflow_active = False

    def _on_verdict(
        bench_id: str, verdict: Any, running_p1: float,
        n_done: int, n_total: int,
    ) -> None:
        prefix = f"{condition_label}_" if condition_label else ""
        if mlflow_active:
            mlflow.log_metric(
                f"{prefix}{bench_id}_running_pass_at_1",
                running_p1, step=n_done,
            )
        if n_done % 50 == 0 or n_done == n_total:
            print(
                f"  [{prefix}{bench_id}] {n_done}/{n_total} "
                f"running Pass@1={running_p1:.2%}"
            )

    results: dict[str, float | None] = {}
    for bench_id in benchmarks:
        try:
            result = run_benchmark(
                stack, bench_id, max_samples=max_samples,
                checkpoint_dir=checkpoint_dir,
                on_verdict=_on_verdict,
            )
            results[bench_id] = result.pass_at_1
            if mlflow_active and checkpoint_dir:
                ckpt_file = Path(checkpoint_dir) / f"{bench_id}.jsonl"
                if ckpt_file.exists():
                    mlflow.log_artifact(str(ckpt_file), "checkpoints")
        except Exception as exc:
            logger.error("Benchmark %s failed: %s", bench_id, exc, exc_info=True)
            results[bench_id] = None
    return results


def run_condition_static(
    benchmarks: list[str],
    model: str,
    adapter_ids: list[str],
    provider: Any,
    adapter_paths: dict[str, str] | None = None,
    max_samples: int | None = None,
    checkpoint_dir: Path | str | None = None,
) -> dict[str, float | None]:
    """Evaluate with a fixed adapter stack (conditions i, iii).

    Args:
        benchmarks: Benchmark IDs to evaluate.
        model: Base model ID.
        adapter_ids: Adapter IDs/paths to stack on top of base model.
            IDs without a corresponding entry in adapter_paths are treated
            as metadata only (not sent to the inference provider).
        provider: InferenceProvider instance.
        adapter_paths: Optional dict mapping adapter_id -> local filesystem
            path. Adapters with paths are loaded into the provider before
            evaluation and unloaded after.
        max_samples: Cap on problems per benchmark.
        checkpoint_dir: Directory for per-problem JSONL checkpoints.

    Returns:
        Dict of {benchmark: pass_at_1}.
    """
    import asyncio

    from evaluation.benchmarks.adapter_stack import AdapterStack

    paths = adapter_paths or {}

    loop = asyncio.new_event_loop()
    try:
        for aid, path in paths.items():
            loop.run_until_complete(provider.load_adapter(aid, path))

        stack = AdapterStack(
            base_model=model,
            adapter_ids=adapter_ids,
            adapter_paths=paths,
            provider=provider,
        )

        return _run_benchmarks(stack, benchmarks, max_samples, checkpoint_dir, condition_label="static")
    finally:
        for aid in paths:
            try:
                loop.run_until_complete(provider.unload_adapter(aid))
            except Exception:
                logger.warning("Failed to unload adapter %s", aid, exc_info=True)
        loop.close()


def run_condition_rag(
    benchmarks: list[str],
    model: str,
    warm_start_adapter: str,
    corpus_path: Path,
    top_k: int,
    provider: Any,
    max_samples: int | None = None,
    checkpoint_dir: Path | str | None = None,
) -> dict[str, float | None]:
    """Evaluate with trajectory-aware RAG prompt augmentation (condition ii).

    Builds a FAISS vector store from the trajectory corpus, then for each
    benchmark problem retrieves top-k trajectory chunks and prepends them
    to the prompt as context.

    Args:
        benchmarks: Benchmark IDs to evaluate.
        model: Base model ID.
        warm_start_adapter: Warm-start LoRA adapter ID.
        corpus_path: Path to JSONL trajectory corpus.
        top_k: Number of chunks to retrieve per query.
        provider: InferenceProvider instance.
        max_samples: Cap on problems per benchmark.
        checkpoint_dir: Directory for per-problem JSONL checkpoints.

    Returns:
        Dict of {benchmark: pass_at_1}.
    """
    from evaluation.benchmarks.adapter_stack import AdapterStack
    from model_training.rag_pipeline import (
        RAGConfig,
        _get_encoder,
        build_vector_store,
        query_trajectory_rag,
    )

    config = RAGConfig(top_k=top_k)
    print(f"  Building vector store from {corpus_path}...")
    store = build_vector_store(corpus_path, config)
    encoder = _get_encoder(config.embedding_model)
    print(f"  Built index with {store['n_chunks']} chunks")

    def prompt_augmenter(prompt: str) -> str:
        chunks = query_trajectory_rag(
            query=prompt,
            index=store["index"],
            chunks=store["chunks"],
            encoder=encoder,
            top_k=top_k,
        )
        if not chunks:
            return prompt
        context = "\n---\n".join(chunks)
        return f"# Retrieved trajectory context\n{context}\n\n# Task\n{prompt}"

    stack = AdapterStack(
        base_model=model,
        adapter_ids=[warm_start_adapter],
        adapter_paths={},
        provider=provider,
        prompt_augmenter=prompt_augmenter,
    )

    return _run_benchmarks(stack, benchmarks, max_samples, checkpoint_dir, condition_label="rag")


def run_condition_ttt(
    benchmarks: list[str],
    model: str,
    warm_start_adapter: str,
    ttt_lr: float,
    ttt_steps: int,
    ttt_mlp_fraction: float,
    provider: Any,
    max_samples: int | None = None,
    checkpoint_dir: Path | str | None = None,
) -> dict[str, float | None]:
    """Evaluate with test-time training on MLP layers (condition iv).

    For each benchmark problem, runs inner-loop gradient updates on a
    fraction of MLP layers using the prompt as self-supervised context,
    then generates the completion.

    Args:
        benchmarks: Benchmark IDs to evaluate.
        model: Base model ID.
        warm_start_adapter: Warm-start LoRA adapter ID.
        ttt_lr: Inner-loop learning rate.
        ttt_steps: Number of inner-loop gradient steps.
        ttt_mlp_fraction: Fraction of MLP layers to train.
        provider: InferenceProvider instance.
        max_samples: Cap on problems per benchmark.
        checkpoint_dir: Directory for per-problem JSONL checkpoints.

    Returns:
        Dict of {benchmark: pass_at_1}.
    """
    from evaluation.benchmarks.adapter_stack import AdapterStack
    from model_training.ttt_e2e import TTTConfig, ttt_forward_pass

    ttt_config = TTTConfig(
        mlp_fraction=ttt_mlp_fraction,
        inner_lr=ttt_lr,
        inner_steps=ttt_steps,
    )

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"  Loading model {model} for TTT inner-loop...")
    tokenizer = AutoTokenizer.from_pretrained(model)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        llm_int8_enable_fp32_cpu_offload=True,
    )
    ttt_model = AutoModelForCausalLM.from_pretrained(
        model,
        quantization_config=bnb_config,
        device_map="auto",
    )

    from model_training.ttt_e2e import select_mlp_layers

    all_mlp_names = [
        name
        for name, _ in ttt_model.named_parameters()
        if "mlp" in name and "weight" in name
    ]
    trainable_names = set(
        select_mlp_layers(all_mlp_names, ttt_config.mlp_fraction)
    )
    original_sd = {
        k: v.detach().cpu().clone()
        for k, v in ttt_model.state_dict().items()
        if k in trainable_names
    }

    def completion_override(prompt: str, max_tokens: int) -> str:
        for k, v in original_sd.items():
            param = ttt_model.get_parameter(k)
            param.data.copy_(v.to(param.device))
        result = ttt_forward_pass(
            model=ttt_model,
            tokenizer=tokenizer,
            context=prompt,
            query=prompt,
            config=ttt_config,
        )
        return result["generation"]

    stack = AdapterStack(
        base_model=model,
        adapter_ids=[warm_start_adapter],
        adapter_paths={},
        provider=provider,
        completion_override=completion_override,
    )

    return _run_benchmarks(stack, benchmarks, max_samples, checkpoint_dir, condition_label="ttt")


def run_condition_rune(
    benchmarks: list[str],
    model: str,
    warm_start_adapter: str,
    adapter_dir: str,
    provider: Any,
    max_samples: int | None = None,
    checkpoint_dir: Path | str | None = None,
) -> dict[str, float | None]:
    """Evaluate with pre-generated hypernetwork adapters (condition v, one-shot).

    Expects adapters were already generated by ``pregenerate_rune_adapters``
    and a manifest.json exists in *adapter_dir*.
    """
    return _rune_eval_with_pregenerated(
        benchmarks, model, warm_start_adapter, provider,
        adapter_dir, max_samples, checkpoint_dir,
    )


def run_condition_rune_iterative(
    benchmarks: list[str],
    model: str,
    warm_start_adapter: str,
    hypernet_checkpoint: str,
    device: str = "cuda",
    max_attempts: int = 3,
    max_samples: int | None = None,
    checkpoint_dir: Path | str | None = None,
    scaling_factor: float = 0.16,
) -> dict[str, float | None]:
    """Condition (v) — iterative Rune: adapter generation with test-feedback retry.

    Runs the core Rune algorithm per benchmark problem:
      1. Build trajectory from code.j2 template
      2. Generate adapter via hypernetwork
      3. Generate code with adapter-augmented model
      4. Run test assertions in sandbox
      5. On failure: build code_retry.j2 trajectory with error feedback → new adapter → retry

    Requires exclusive GPU (no vLLM). Loads base model (4-bit NF4) +
    hypernetwork directly, similar to TTT condition.
    """
    import gc
    import re
    import tempfile

    import ctx_to_lora.modeling.hypernet as _hypernet_mod
    import torch
    from ctx_to_lora.modeling.lora_merger import combine_lora as _combine_lora
    from evaluation.benchmarks.protocol import BenchmarkResult, PassVerdict
    from evaluation.benchmarks.runner import (
        _ADAPTER_REGISTRY,
        _STOP_BY_BENCHMARK,
        _import_adapter,
    )
    from model_training.d2l_probe import extract_activations_with_model
    from model_training.sakana_d2l import _save_sakana_adapter, load_sakana_checkpoint
    from peft import PeftModel
    from shared.sandbox import (
        SubprocessBackend,
        count_test_results,
        extract_failed_tests,
    )
    from shared.template_loader import render_trajectory
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    # Monkey-patch HyperLoRA.forward for device safety
    _orig_forward = _hypernet_mod.HyperLoRA.forward

    def _device_safe_forward(self, features, attn_mask=None, position_ids=None, n_ctx_chunks=None):
        dev = "cuda" if features.is_cuda else "cpu"
        with torch.autocast(device_type=dev, dtype=torch.bfloat16):
            if self.aggregator.layer_to_layer and self.iterative_mode:
                bs, n_layers = features.shape[0:2]
                lora_emb = torch.empty(
                    (bs, n_layers, self.num_modules, self.r, self.config.latent_size),
                    device=features.device,
                )
                for i in range(n_layers):
                    lora_emb[:, i], _ = self.aggregator(
                        features[:, i], attn_mask, position_ids
                    )
            else:
                lora_emb, _ = self.aggregator(features, attn_mask, position_ids)
        flat_loras = None
        if self.target_modules:
            lora_emb = self.layers(lora_emb)
            norm = torch.norm(lora_emb, dim=-1, keepdim=True)
            norm_lora_emb = lora_emb / norm
            flat_loras = self.head(norm_lora_emb)
        return flat_loras, None

    _hypernet_mod.HyperLoRA.forward = _device_safe_forward

    try:
        import mlflow
        mlflow_active = mlflow.active_run() is not None
    except Exception:
        mlflow_active = False

    logger.info("Loading hypernetwork from %s", hypernet_checkpoint)
    hypernet, hc = load_sakana_checkpoint(hypernet_checkpoint, device=device)
    layer_indices = list(hc.layer_indices)

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    logger.info("Loading base model %s (4-bit NF4) for iterative Rune", model)
    tokenizer = AutoTokenizer.from_pretrained(model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        model, quantization_config=bnb_cfg, device_map="auto",
    )
    base_model.eval()
    sandbox = SubprocessBackend()
    tmp_root = Path(tempfile.mkdtemp(prefix="rune_iter_"))
    default_stop = ["\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif"]

    def _generate_text(prompt: str, adapter_dir: str | None, max_tokens: int, stop: list[str]) -> str:
        """Generate text using base model + optional PEFT adapter."""
        if adapter_dir is not None:
            peft_model = PeftModel.from_pretrained(base_model, adapter_dir)  # noqa: F821
            peft_model.eval()
            gen_model = peft_model
        else:
            gen_model = base_model  # noqa: F821

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        model_device = next(gen_model.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = gen_model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)

        if adapter_dir is not None:
            del peft_model
            gc.collect()
            torch.cuda.empty_cache()

        # Truncate at first stop sequence
        min_idx = len(text)
        for tok in stop:
            idx = text.find(tok)
            if idx != -1 and idx < min_idx:
                min_idx = idx
        return text[:min_idx]

    def _generate_adapter(trajectory: str, out_dir: str) -> str:
        """Run hypernetwork to produce a PEFT adapter from trajectory text."""
        features, attn_mask = extract_activations_with_model(
            text=trajectory,
            model=base_model,  # noqa: F821
            tokenizer=tokenizer,
            layer_indices=layer_indices,
        )
        with torch.no_grad():
            lora_dict, _ = hypernet.generate_weights(features, attn_mask, None)  # noqa: F821
        n_chunks = torch.ones(1, dtype=torch.int32)
        lora_bias = hypernet.get_head_bias() if hypernet.config.use_bias else None  # noqa: F821
        lora_dict = _combine_lora(lora_dict, n_chunks, lora_bias=lora_bias)
        _save_sakana_adapter(
            lora_dict=lora_dict,
            output_dir=out_dir,
            base_model_name=model,
            hc=hc,
            scaling_factor=scaling_factor,
        )
        return out_dir

    def _extract_error_summary(stderr: str) -> str:
        if not stderr:
            return ""
        lines = stderr.strip().splitlines()
        error_lines = [
            ln for ln in lines
            if "Error" in ln or "Exception" in ln or "assert" in ln.lower()
        ]
        final_error = error_lines[-1].strip() if error_lines else lines[-1].strip()
        failed = [
            ln.strip() for ln in lines
            if ln.strip().startswith("FAIL:") or ln.strip().startswith("ERROR:")
        ]
        parts = []
        if failed:
            parts.append("Failed: " + "; ".join(failed[:5]))
        parts.append(final_error)
        return "\n".join(parts)

    def _safe_id(name: str) -> str:
        return re.sub(r"[^\w\-]", "_", name)

    results: dict[str, float | None] = {}
    ckpt_base = Path(checkpoint_dir) if checkpoint_dir else None
    if ckpt_base:
        ckpt_base.mkdir(parents=True, exist_ok=True)

    for bench_id in benchmarks:
        if bench_id not in _ADAPTER_REGISTRY:
            logger.error("Unknown benchmark %s", bench_id)
            results[bench_id] = None
            continue

        bench_stop = _STOP_BY_BENCHMARK.get(bench_id, default_stop)
        adapter = _import_adapter(_ADAPTER_REGISTRY[bench_id])
        problems = adapter.load_problems(max_samples=max_samples, seed=42)
        verdicts: list[PassVerdict] = []
        n_passed = 0

        logger.info("Rune iterative: %s — %d problems, up to %d attempts each",
                     bench_id, len(problems), max_attempts)

        for pi, problem in enumerate(problems):
            subtask = {
                "name": _safe_id(problem.problem_id),
                "description": problem.metadata.get("description", problem.prompt[:200]),
            }
            best_generation = ""
            passed = False
            test_passed = 0
            test_total = 0
            error_summary = ""
            failed_tests = ""
            fix_guidance = ""

            for attempt in range(max_attempts):
                adapter_out = str(tmp_root / bench_id / f"{subtask['name']}_v{attempt}")

                if attempt == 0:
                    trajectory = render_trajectory(
                        "code",
                        subtask=subtask,
                        subtask_index=1,
                        total_subtasks=1,
                        plan=problem.prompt,
                        existing_code="",
                        project=problem.prompt,
                        dependency_interfaces="",
                    )
                else:
                    trajectory = render_trajectory(
                        "code_retry",
                        subtask=subtask,
                        attempt=attempt + 1,
                        max_retries=max_attempts,
                        plan=problem.prompt,
                        existing_code=best_generation,
                        passed=test_passed,
                        total=test_total,
                        tests_passed=False,
                        error_summary=error_summary,
                        failed_tests=failed_tests,
                        fix_guidance=fix_guidance,
                        history=None,
                        project=problem.prompt,
                        dependency_interfaces="",
                    )

                _generate_adapter(trajectory, adapter_out)  # noqa: F821
                generation = _generate_text(  # noqa: F821
                    problem.prompt, adapter_out, max_tokens=512, stop=bench_stop,
                )
                best_generation = generation

                setup = problem.metadata.get("test_setup_code", "")
                setup_block = f"{setup}\n" if setup else ""
                test_code = f"{setup_block}{generation}\n\n{problem.test_code}\n"
                sandbox_result = sandbox.run(test_code, timeout=30)
                passed = sandbox_result.exit_code == 0 and not sandbox_result.is_timed_out

                if passed:
                    logger.debug("  %s PASSED on attempt %d", problem.problem_id, attempt + 1)
                    break

                test_passed, test_total = count_test_results(
                    sandbox_result.stdout, sandbox_result.stderr,
                )
                error_summary = _extract_error_summary(sandbox_result.stderr)  # noqa: F821
                failed_tests = extract_failed_tests(sandbox_result.stderr)
                if test_total == 0:
                    fix_guidance = "Code failed to execute. Check syntax, imports, indentation."
                elif test_passed == 0:
                    fix_guidance = "No tests pass. Fix basic structure."
                else:
                    fix_guidance = f"{test_total - test_passed} test(s) failing. Fix without breaking passing tests."

                logger.debug("  %s FAILED attempt %d/%d", problem.problem_id, attempt + 1, max_attempts)

            verdict = PassVerdict(
                problem_id=problem.problem_id,
                passed=passed,
                generation=best_generation,
                error=None if passed else error_summary or None,
                timed_out=False,
            )
            verdicts.append(verdict)
            if passed:
                n_passed += 1

            running_p1 = n_passed / (pi + 1)
            if mlflow_active:
                mlflow.log_metric(f"v_iter_{bench_id}_running_pass_at_1", running_p1, step=pi + 1)
            if (pi + 1) % 10 == 0 or (pi + 1) == len(problems):
                print(f"  [rune_iter/{bench_id}] {pi + 1}/{len(problems)} "
                      f"running Pass@1={running_p1:.2%}")

        bench_result = BenchmarkResult(benchmark_id=bench_id, verdicts=verdicts)
        results[bench_id] = bench_result.pass_at_1
        logger.info("Rune iterative %s: Pass@1=%.2f%%", bench_id, bench_result.pass_at_1 * 100)

        if mlflow_active:
            mlflow.log_metric(f"v_iter_{bench_id}_pass_at_1", bench_result.pass_at_1)

    # Cleanup
    del hypernet, base_model
    gc.collect()
    torch.cuda.empty_cache()
    _hypernet_mod.HyperLoRA.forward = _orig_forward

    return results


def pregenerate_rune_adapters(
    benchmarks: list[str],
    model: str,
    hypernet_checkpoint: str,
    device: str,
    output_dir: str,
    max_samples: int | None = None,
) -> None:
    """Phase 1: Pre-generate per-problem adapters on GPU (no vLLM needed).

    Saves adapters to output_dir/<bench_id>/<problem_id>/ and writes a
    manifest.json mapping problem_id → adapter_path.
    """
    import json as _json

    import ctx_to_lora.modeling.hypernet as _hypernet_mod
    import torch
    from ctx_to_lora.modeling.lora_merger import combine_lora as _combine_lora
    from evaluation.benchmarks.runner import _ADAPTER_REGISTRY, _import_adapter
    from model_training.d2l_probe import extract_activations_with_model
    from model_training.sakana_d2l import _save_sakana_adapter, load_sakana_checkpoint
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # HyperLoRA.forward hardcodes torch.autocast(device_type="cuda").
    # Patch it to derive device_type from the input tensor instead.

    def _device_safe_forward(self, features, attn_mask=None, position_ids=None, n_ctx_chunks=None):
        dev = "cuda" if features.is_cuda else "cpu"
        with torch.autocast(device_type=dev, dtype=torch.bfloat16):
            if self.aggregator.layer_to_layer and self.iterative_mode:
                bs, n_layers = features.shape[0:2]
                lora_emb = torch.empty(
                    (bs, n_layers, self.num_modules, self.r, self.config.latent_size),
                    device=features.device,
                )
                for i in range(n_layers):
                    lora_emb[:, i], _ = self.aggregator(
                        features[:, i], attn_mask, position_ids
                    )
            else:
                lora_emb, _ = self.aggregator(features, attn_mask, position_ids)

        flat_loras = None
        if self.target_modules:
            lora_emb = self.layers(lora_emb)
            norm = torch.norm(lora_emb, dim=-1, keepdim=True)
            norm_lora_emb = lora_emb / norm
            flat_loras = self.head(norm_lora_emb)

        return flat_loras, None

    _hypernet_mod.HyperLoRA.forward = _device_safe_forward

    logger.info("Pre-loading HyperLoRA perceiver from %s", hypernet_checkpoint)
    hypernet, hc = load_sakana_checkpoint(hypernet_checkpoint, device=device)
    layer_indices = list(hc.layer_indices)
    scaling_factor = 0.16

    # bf16 weights (~18 GB) + forward pass activations (~4 GB) exceed the
    # 22 GB L4.  Use 4-bit NF4 quantization (~4.5 GB) to leave headroom.
    from transformers import BitsAndBytesConfig  # noqa: PLC0415

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    logger.info("Loading base model %s (4-bit NF4) on %s for activation extraction", model, device)
    tokenizer = AutoTokenizer.from_pretrained(model)
    base_model = AutoModelForCausalLM.from_pretrained(
        model, quantization_config=bnb_cfg, device_map="auto",
    )
    base_model.eval()

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict[str, str]] = {}

    for bench_id in benchmarks:
        adapter = _import_adapter(_ADAPTER_REGISTRY[bench_id])
        problems = adapter.load_problems(max_samples=max_samples, seed=42)
        manifest[bench_id] = {}
        logger.info("Generating %d adapters for %s", len(problems), bench_id)

        for i, problem in enumerate(problems):
            problem_dir = str(Path(output_dir) / bench_id / problem.problem_id)
            Path(problem_dir).mkdir(parents=True, exist_ok=True)

            features, attn_mask = extract_activations_with_model(
                text=problem.prompt,
                model=base_model,
                tokenizer=tokenizer,
                layer_indices=layer_indices,
            )
            with torch.no_grad():
                lora_dict, _ = hypernet.generate_weights(features, attn_mask, None)

            n_chunks = torch.ones(1, dtype=torch.int32)
            lora_bias = hypernet.get_head_bias() if hypernet.config.use_bias else None
            lora_dict = _combine_lora(lora_dict, n_chunks, lora_bias=lora_bias)

            _save_sakana_adapter(
                lora_dict=lora_dict,
                output_dir=problem_dir,
                base_model_name=model,
                hc=hc,
                scaling_factor=scaling_factor,
            )
            manifest[bench_id][problem.problem_id] = problem_dir
            if (i + 1) % 50 == 0:
                logger.info("  %s: %d/%d adapters generated", bench_id, i + 1, len(problems))

    manifest_path = Path(output_dir) / "manifest.json"
    manifest_path.write_text(_json.dumps(manifest, indent=2))
    logger.info("Wrote adapter manifest to %s", manifest_path)

    del hypernet, base_model
    import gc  # noqa: PLC0415
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("GPU freed — %d adapters pre-generated", sum(len(v) for v in manifest.values()))


def _rune_eval_with_pregenerated(
    benchmarks: list[str],
    model: str,
    warm_start_adapter: str,
    provider: Any,
    adapter_dir: str,
    max_samples: int | None = None,
    checkpoint_dir: Path | str | None = None,
) -> dict[str, float | None]:
    """Phase 2: Run benchmarks using pre-generated adapters (vLLM required)."""
    import json as _json

    from evaluation.benchmarks.adapter_stack import AdapterStack
    from evaluation.benchmarks.runner import _ADAPTER_REGISTRY, _import_adapter

    manifest_path = Path(adapter_dir) / "manifest.json"
    manifest = _json.loads(manifest_path.read_text())

    # Build prompt → adapter_path lookup across all benchmarks.
    prompt_to_adapter: dict[str, str] = {}
    for bench_id in benchmarks:
        if bench_id not in manifest:
            logger.warning("No pre-generated adapters for %s", bench_id)
            continue
        pid_map = manifest[bench_id]
        adapter = _import_adapter(_ADAPTER_REGISTRY[bench_id])
        problems = adapter.load_problems(max_samples=max_samples, seed=42)
        for problem in problems:
            if problem.problem_id in pid_map:
                prompt_to_adapter[problem.prompt] = pid_map[problem.problem_id]
    logger.info("Loaded %d pre-generated adapter paths", len(prompt_to_adapter))

    def adapter_generator(prompt: str) -> str | None:
        return prompt_to_adapter.get(prompt)

    stack = AdapterStack(
        base_model=model,
        adapter_ids=[warm_start_adapter],
        adapter_paths={},
        provider=provider,
        adapter_generator=adapter_generator,
    )

    return _run_benchmarks(stack, benchmarks, max_samples, checkpoint_dir, condition_label="v")


def assemble_table2(
    all_results: dict[str, dict[str, float | None]],
) -> dict[str, Any]:
    """Assemble Table 2 data from per-condition results.

    Args:
        all_results: {condition: {benchmark: pass_at_1}}.

    Returns:
        Table 2 structured data with deltas vs condition (i) substrate baseline.
    """
    table: dict[str, Any] = {"conditions": {}}
    base_i = all_results.get("i", {})

    for cond, scores in all_results.items():
        deltas: dict[str, float | None] = {}
        for bench, score in scores.items():
            i_score = base_i.get(bench)
            if score is not None and i_score is not None:
                deltas[bench] = score - i_score
            else:
                deltas[bench] = None

        table["conditions"][cond] = {
            "label": CONDITION_LABELS.get(cond, cond),
            "scores": scores,
            "delta_vs_substrate": deltas,
        }

    return table


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all paper conditions (Table 2)")
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["i", "ii", "iii", "iv", "v"],
        choices=["i", "ii", "iii", "iv", "v"],
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["humaneval", "livecodebench"],
    )
    parser.add_argument("--model", default=DEFAULT_BASE_MODEL)
    parser.add_argument(
        "--warm-start-adapter",
        default=DEFAULT_WARM_START,
        help="Warm-start LoRA for substrate (DeltaCoder)",
    )
    parser.add_argument(
        "--adapter-iii",
        type=str,
        default=None,
        help="Path to HPO-tuned QLoRA adapter for Condition (iii)",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=None,
        help="Path to JSONL trajectory corpus for Condition (ii) RAG",
    )
    parser.add_argument("--rag-top-k", type=int, default=5)
    parser.add_argument(
        "--hypernet-checkpoint",
        type=str,
        default=None,
        help="Path to trained hypernetwork checkpoint for Condition (v) pregeneration",
    )
    parser.add_argument(
        "--rune-adapter-dir",
        type=str,
        default=None,
        help="Path to pre-generated adapter dir (with manifest.json) for Condition (v) eval",
    )
    parser.add_argument(
        "--pregenerate",
        action="store_true",
        help="Pre-generate Rune adapters (GPU phase) and exit without running benchmarks",
    )
    parser.add_argument(
        "--rune-max-attempts",
        type=int,
        default=3,
        help="Max retry attempts per problem for iterative Rune (condition v)",
    )
    parser.add_argument(
        "--rune-one-shot",
        action="store_true",
        help="Use one-shot adapter generation instead of iterative Rune for condition (v)",
    )
    parser.add_argument("--ttt-lr", type=float, default=1e-4)
    parser.add_argument("--ttt-steps", type=int, default=5)
    parser.add_argument("--ttt-mlp-fraction", type=float, default=0.25)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Cap problems per benchmark (useful for quick smoke runs)",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("evaluation_results/table2.json")
    )
    args = parser.parse_args()

    # --pregenerate: GPU-only phase — generate adapters and exit.
    if args.pregenerate:
        if not args.hypernet_checkpoint:
            parser.error("--pregenerate requires --hypernet-checkpoint")
        adapter_out = args.rune_adapter_dir or str(
            args.output.parent / "rune_adapters"
        )
        pregenerate_rune_adapters(
            benchmarks=args.benchmarks,
            model=args.model,
            hypernet_checkpoint=args.hypernet_checkpoint,
            device=args.device,
            output_dir=adapter_out,
            max_samples=args.max_samples,
        )
        print(f"Adapters written to {adapter_out}")
        return

    from inference.factory import get_provider
    from model_training.training_common import (
        mlflow_log_artifact,
        mlflow_log_params,
        setup_mlflow,
    )

    mlflow_ok = setup_mlflow("paper-table2", tracking_uri=None)
    if mlflow_ok:
        import mlflow

        mlflow.start_run(run_name="table2")
        mlflow_log_params(
            {
                "model": args.model,
                "warm_start_adapter": args.warm_start_adapter,
                "benchmarks": ",".join(args.benchmarks),
                "conditions": ",".join(args.conditions),
                "git_commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"],
                    text=True,
                ).strip(),
                "rag_top_k": args.rag_top_k,
                "ttt_lr": args.ttt_lr,
                "ttt_steps": args.ttt_steps,
                "ttt_mlp_fraction": args.ttt_mlp_fraction,
            }
        )

    # Auto-fetch HPO adapter for conditions that need it (iii, v)
    hpo_adapter_path: Path | None = None
    needs_hpo = {"iii"} & set(args.conditions)
    if needs_hpo:
        hpo_dest = (
            Path(args.adapter_iii)
            if args.adapter_iii
            else Path("hpo_artifacts/best_diffloss_v1")
        )
        try:
            hpo_adapter_path = fetch_best_hpo_adapter(hpo_dest)
            if not args.adapter_iii:
                args.adapter_iii = str(hpo_adapter_path)
        except FileNotFoundError as exc:
            print(f"  WARNING: Could not fetch HPO adapter: {exc}")
            if "iii" in needs_hpo:
                print("  Condition (iii) will be skipped.")

    provider = get_provider()

    metadata = {
        "model": args.model,
        "warm_start_adapter": args.warm_start_adapter,
        "benchmarks": args.benchmarks,
        "hypernet_checkpoint": args.hypernet_checkpoint,
    }
    all_results: dict[str, dict[str, float | None]] = {}
    ckpt_base = args.output.parent / "checkpoints"

    for cond in args.conditions:
        print(f"\n{'=' * 60}")
        print(f"Condition ({cond}): {CONDITION_LABELS[cond]}")
        print(f"{'=' * 60}")

        cond_ckpt = ckpt_base / cond
        start = time.time()

        if cond == "i":
            results = run_condition_static(
                args.benchmarks,
                args.model,
                adapter_ids=[args.warm_start_adapter],
                provider=provider,
                max_samples=args.max_samples,
                checkpoint_dir=cond_ckpt,
            )

        elif cond == "ii":
            if not args.corpus:
                print("  SKIPPED: --corpus not provided")
                continue
            try:
                results = run_condition_rag(
                    args.benchmarks,
                    args.model,
                    warm_start_adapter=args.warm_start_adapter,
                    corpus_path=args.corpus,
                    top_k=args.rag_top_k,
                    provider=provider,
                    max_samples=args.max_samples,
                    checkpoint_dir=cond_ckpt,
                )
            except ImportError as exc:
                print(f"  SKIPPED: missing dependency for RAG — {exc}")
                continue

        elif cond == "iii":
            if not args.adapter_iii:
                print("  SKIPPED: --adapter-iii not provided and auto-fetch failed")
                continue
            iii_id = "hpo_qlora"
            results = run_condition_static(
                args.benchmarks,
                args.model,
                adapter_ids=[args.warm_start_adapter, iii_id],
                adapter_paths={iii_id: str(args.adapter_iii)},
                provider=provider,
                max_samples=args.max_samples,
                checkpoint_dir=cond_ckpt,
            )

        elif cond == "iv":
            results = run_condition_ttt(
                args.benchmarks,
                args.model,
                warm_start_adapter=args.warm_start_adapter,
                ttt_lr=args.ttt_lr,
                ttt_steps=args.ttt_steps,
                ttt_mlp_fraction=args.ttt_mlp_fraction,
                provider=provider,
                max_samples=args.max_samples,
                checkpoint_dir=cond_ckpt,
            )

        elif cond == "v":
            if args.rune_one_shot:
                if not args.rune_adapter_dir:
                    print("  SKIPPED: --rune-adapter-dir not provided (one-shot mode)")
                    continue
                results = run_condition_rune(
                    args.benchmarks,
                    args.model,
                    warm_start_adapter=args.warm_start_adapter,
                    adapter_dir=args.rune_adapter_dir,
                    provider=provider,
                    max_samples=args.max_samples,
                    checkpoint_dir=cond_ckpt,
                )
            else:
                if not args.hypernet_checkpoint:
                    print("  SKIPPED: --hypernet-checkpoint required for iterative Rune")
                    continue
                results = run_condition_rune_iterative(
                    args.benchmarks,
                    args.model,
                    warm_start_adapter=args.warm_start_adapter,
                    hypernet_checkpoint=args.hypernet_checkpoint,
                    device=args.device,
                    max_attempts=args.rune_max_attempts,
                    max_samples=args.max_samples,
                    checkpoint_dir=cond_ckpt,
                )

        else:
            continue

        elapsed = time.time() - start
        for bench_id, score in results.items():
            label = f"{score:.2%}" if score is not None else "FAILED"
            print(f"  {bench_id}: {label}")
        print(f"  Elapsed: {elapsed:.1f}s")

        all_results[cond] = results
        flush_partial_results(all_results, args.output, metadata)

    print(f"\nTable 2 written to {args.output}")

    if mlflow_ok:
        for cond, scores in all_results.items():
            for bench, score in scores.items():
                if score is not None:
                    mlflow.log_metric(f"{cond}_{bench}_pass_at_1", score)
        mlflow_log_artifact(str(args.output))
        mlflow.end_run()


if __name__ == "__main__":
    main()
