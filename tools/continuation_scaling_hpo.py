# tools/continuation_scaling_hpo.py
"""Optuna HPO for adapter-encoded continuation feasibility and tuning.

Validates whether the hypernetwork can encode partial output well enough
for the model to continue generation from a short prompt + grammar state.
Sweeps scaling multiplier, prompt shape (first M / last N lines), prompt
strategy, and trajectory flavor.

Run:
  uv run python tools/continuation_scaling_hpo.py \
      [--config benchmarks/bench.yaml] [--n-trials 20]

Results logged to MLflow experiment 'continuation-scaling-hpo'.
"""

from __future__ import annotations

import argparse
import asyncio
import faulthandler
import json
import os
import signal
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import logging
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

_handler = logging.StreamHandler(sys.stderr)
_fmt = "%(asctime)s %(name)s %(levelname)s %(message)s"
_handler.setFormatter(logging.Formatter(_fmt))
_handler.setLevel(logging.DEBUG)
logging.root.addHandler(_handler)
logging.root.setLevel(logging.INFO)

logger = logging.getLogger(__name__)

_STATUS_FILE = Path("/tmp/continuation_hpo_status.txt")


def _status(msg: str) -> None:
    """Write a checkpoint status to a file (survives SIGKILL) and log it."""
    logger.info("STATUS: %s", msg)
    try:
        with open(_STATUS_FILE, "w") as f:
            f.write(msg + "\n")
            f.flush()
            os.fsync(f.fileno())
    except OSError:
        pass


def _log_memory(label: str) -> None:
    """Log GPU and system memory usage."""
    try:
        import torch  # noqa: PLC0415
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            max_alloc = torch.cuda.max_memory_allocated() / 1e9
            logger.info(
                "MEM [%s] GPU alloc=%.2fGB reserved=%.2fGB peak=%.2fGB",
                label, alloc, reserved, max_alloc,
            )
    except Exception:
        pass
    try:
        import psutil  # noqa: PLC0415
        proc = psutil.Process()
        rss = proc.memory_info().rss / 1e9
        logger.info("MEM [%s] RSS=%.2fGB", label, rss)
    except Exception:
        pass

_DEFAULT_CONFIG = Path("benchmarks/bench.yaml")
_CONT_MAX_TOKENS = 256
_RSS_LIMIT_FRACTION = 0.80


def _rss_gb() -> float:
    try:
        import psutil  # noqa: PLC0415
        return psutil.Process().memory_info().rss / 1e9
    except Exception:
        return 0.0


def _check_rss_limit() -> None:
    """Raise if RSS exceeds _RSS_LIMIT_FRACTION of total system memory."""
    try:
        import psutil  # noqa: PLC0415
        mem = psutil.virtual_memory()
        fraction = psutil.Process().memory_info().rss / mem.total
        if fraction > _RSS_LIMIT_FRACTION:
            raise MemoryError(
                f"RSS at {fraction:.0%} of system memory "
                f"({_rss_gb():.1f}GB / {mem.total / 1e9:.1f}GB) — "
                f"aborting trial to prevent OOM"
            )
    except MemoryError:
        raise
    except Exception:
        pass


def _force_gc() -> None:
    """Force garbage collection and release PyTorch caches."""
    import gc  # noqa: PLC0415
    gc.collect()
    try:
        import torch  # noqa: PLC0415
        torch.cuda.empty_cache()
    except Exception:
        pass


@dataclass(frozen=True)
class ContinuationTask:
    name: str
    task: str
    system_prompt: str
    expected_min_chars: int


TASKS = [
    ContinuationTask(
        name="calculator_divide",
        task=(
            "Write a Python Calculator class with add, subtract, multiply, "
            "divide (with ZeroDivisionError), and a history method."
        ),
        system_prompt="You are a code generator.",
        expected_min_chars=400,
    ),
    ContinuationTask(
        name="stack_class",
        task=(
            "Write a Python Stack class with push, pop, peek, is_empty, "
            "and size methods using a list internally."
        ),
        system_prompt="You are a code generator.",
        expected_min_chars=300,
    ),
    ContinuationTask(
        name="validators",
        task=(
            "Write Python functions: validate_email, validate_phone, "
            "validate_url. Each returns True/False. Use regex."
        ),
        system_prompt="You are a code generator.",
        expected_min_chars=300,
    ),
]


def _first_n_lines(text: str, n: int) -> str:
    lines = text.splitlines()
    return "\n".join(lines[:n])


def _last_n_lines(text: str, n: int) -> str:
    lines = text.splitlines()
    return "\n".join(lines[-n:]) if n > 0 else ""


def _scale_b_only_inplace(sd: dict[str, Any], factor: float) -> dict[str, Any]:
    for k, v in sd.items():
        if "lora_B" in k:
            sd[k] = v * factor
    return sd


_TRAJ_CODE_CAP = 4000


def _cap_code(accumulated: str) -> str:
    if len(accumulated) <= _TRAJ_CODE_CAP:
        return accumulated
    return "...\n" + accumulated[-_TRAJ_CODE_CAP:]


# --- Trajectory flavors: what the hypernetwork sees ---

def _traj_minimal(task: str, accumulated: str, attempt: int, max_cont: int) -> str:
    return f"GOAL: {task[:200]}\nCODE SO FAR:\n{_cap_code(accumulated)}"


def _traj_with_counter(task: str, accumulated: str, attempt: int, max_cont: int) -> str:
    return (
        f"CONTINUATION {attempt + 1}/{max_cont}\n"
        f"GOAL: {task[:200]}\n"
        f"CODE SO FAR:\n{_cap_code(accumulated)}"
    )


def _traj_with_structure(
    task: str, accumulated: str, attempt: int, max_cont: int,
) -> str:
    lines = accumulated.splitlines()
    n_lines = len(lines)
    n_defs = sum(1 for ln in lines if ln.strip().startswith("def "))
    n_classes = sum(1 for ln in lines if ln.strip().startswith("class "))
    return (
        f"CONTINUATION {attempt + 1}/{max_cont}\n"
        f"GOAL: {task[:200]}\n"
        f"STRUCTURE: {n_lines} lines, {n_classes} classes, {n_defs} functions\n"
        f"CODE SO FAR:\n{_cap_code(accumulated)}"
    )


def _traj_code_template(
    task: str, accumulated: str, attempt: int, max_cont: int,
) -> str:
    """Matches the code.j2 template format the hypernetwork was trained on."""
    return (
        f"ROLE: coder\n"
        f"PROJECT: {task[:300]}\n"
        f"SUBTASK: continuation ({attempt + 1}/{max_cont})\n"
        f"DESCRIPTION: Continue generating code from where it left off.\n"
        f"\n"
        f"PLAN:\n"
        f"Continue the implementation. The code below is partially complete.\n"
        f"\n"
        f"EXISTING CODE:\n"
        f"{_cap_code(accumulated)}\n"
        f"PRACTICES: Clean layered architecture, no stubs or placeholders, no dead\n"
        f"code, specific exceptions with context."
    )


TRAJECTORY_FLAVORS: dict[str, Any] = {
    "minimal_goal_code": _traj_minimal,
    "with_attempt_counter": _traj_with_counter,
    "with_structural_summary": _traj_with_structure,
    "code_template": _traj_code_template,
}


# --- Prompt strategies: what the model sees in the user turn ---

def _prompt_head_tail(
    accumulated: str, first_lines: int, last_lines: int, task: str,
) -> str:
    head = _first_n_lines(accumulated, first_lines) if first_lines > 0 else ""
    tail = _last_n_lines(accumulated, last_lines)
    parts = [p for p in [head, "...", tail] if p]
    return "Continue the code:\n" + "\n".join(parts)


def _prompt_tail_only(
    accumulated: str, first_lines: int, last_lines: int, task: str,
) -> str:
    tail = _last_n_lines(accumulated, last_lines)
    return "Continue the code:\n" + tail


def _prompt_instruction_wrapped(
    accumulated: str, first_lines: int, last_lines: int, task: str,
) -> str:
    tail = _last_n_lines(accumulated, last_lines)
    return (
        f"Original task: {task[:150]}\n"
        f"Continue from where the code left off:\n{tail}"
    )


PROMPT_STRATEGIES: dict[str, Any] = {
    "head_tail": _prompt_head_tail,
    "tail_only": _prompt_tail_only,
    "instruction_wrapped": _prompt_instruction_wrapped,
}


def _check_completion(
    accumulated: str,
    matcher: Any,
    new_token_count: int,
    max_tokens: int,
) -> tuple[bool, dict[str, bool]]:
    grammar_done = matcher.is_completed()

    schema_valid = False
    try:
        from rune.engine.parse import CodeResult  # noqa: PLC0415

        CodeResult.model_validate_json(accumulated)
        schema_valid = True
    except Exception:
        pass

    gen_stopped_early = new_token_count < max_tokens

    signals = {
        "grammar_completed": grammar_done,
        "schema_valid": schema_valid,
        "gen_stopped_early": gen_stopped_early,
    }
    all_agree = grammar_done and schema_valid and gen_stopped_early
    return all_agree, signals


def _edit_distance(a: str, b: str) -> float:
    if not a and not b:
        return 0.0
    return 1.0 - SequenceMatcher(None, a, b).ratio()


def _coherence_at_boundary(before: str, after: str) -> float:
    if not before or not after:
        return 0.0
    last_line = before.rstrip().splitlines()[-1] if before.strip() else ""
    first_line = after.lstrip().splitlines()[0] if after.strip() else ""
    if not last_line or not first_line:
        return 0.0
    indent_before = len(last_line) - len(last_line.lstrip())
    indent_after = len(first_line) - len(first_line.lstrip())
    indent_ok = 1.0 if abs(indent_before - indent_after) <= 8 else 0.0
    printable = sum(1 for c in after if c.isprintable() or c in "\n\t")
    ascii_ratio = printable / max(len(after), 1)
    return indent_ok * 0.5 + ascii_ratio * 0.5


def _run_continuation_trial(
    model: Any,
    task: ContinuationTask,
    scaling: float,
    first_lines: int,
    last_lines: int,
    prompt_strategy: str,
    trajectory_flavor: str,
    gen_kwargs: dict[str, Any],
    initial_gen_kwargs: dict[str, Any],
    compiled_grammar: Any,
    max_continuations: int = 2,
) -> dict[str, Any]:
    import torch  # noqa: PLC0415
    import xgrammar as xgr  # noqa: PLC0415

    _check_rss_limit()

    _status(f"task {task.name}: generating initial adapter")
    _log_memory(f"task {task.name} pre-adapter")
    initial_adapter = model.generate_adapter(
        f"ROLE: coder\nTASK: {task.task}\nPLAN: Implement with tests.",
        offload_base=True,
    )
    model.hotswap_adapter(
        _scale_b_only_inplace(initial_adapter.state_dict, scaling),
    )
    del initial_adapter
    _force_gc()

    _status(f"task {task.name}: initial generation")
    _log_memory(f"task {task.name} pre-generate")
    initial_result = asyncio.run(
        model.generate(
            task.task,
            system_prompt=task.system_prompt,
            **initial_gen_kwargs,
        )
    )
    logger.info(
        "task %s: initial gen done, %d tokens, truncated=%s, %d chars",
        task.name, initial_result.tokens_used, initial_result.truncated,
        len(initial_result.text),
    )

    if not initial_result.truncated:
        return {
            "completed": True,
            "continuations_used": 0,
            "completion_signals": {
                "grammar_completed": True,
                "schema_valid": True,
                "gen_stopped_early": True,
            },
            "total_tokens": initial_result.tokens_used,
            "accumulated_chars": len(initial_result.text),
            "coherence": 1.0,
            "note": "completed_without_continuation",
        }

    _force_gc()
    base_model_obj = model._base_model
    tokenizer = model._tokenizer

    accumulated = initial_result.text
    total_tokens = initial_result.tokens_used
    initial_text = initial_result.text
    del initial_result
    coherence_scores: list[float] = []
    traj_fn = TRAJECTORY_FLAVORS[trajectory_flavor]
    prompt_fn = PROMPT_STRATEGIES[prompt_strategy]
    completed = False
    completion_signals: dict[str, bool] = {}

    sampling: dict[str, Any] = {}
    temp = gen_kwargs.get("temperature", 0.7)
    if temp > 0:
        sampling = {
            "do_sample": True,
            "temperature": temp,
            "top_p": gen_kwargs.get("top_p", 0.8),
            "top_k": gen_kwargs.get("top_k", 20),
        }
    else:
        sampling = {"do_sample": False}

    for attempt in range(max_continuations):
        _check_rss_limit()
        _status(f"task {task.name}: continuation {attempt}/{max_continuations}")
        _log_memory(f"task {task.name} cont {attempt}")
        _force_gc()
        trajectory_text = traj_fn(task.task, accumulated, attempt, max_continuations)
        cont_adapter = model.generate_adapter(trajectory_text, offload_base=True)
        model.hotswap_adapter(
            _scale_b_only_inplace(cont_adapter.state_dict, scaling),
        )
        del cont_adapter
        _force_gc()

        matcher = xgr.GrammarMatcher(compiled_grammar)
        if not matcher.accept_string(accumulated):
            logger.error(
                "Grammar reject at continuation %d for %s",
                attempt, task.name,
            )
            del matcher
            break

        adv_processor = xgr.contrib.hf.LogitsProcessor(compiled_grammar)
        adv_processor.matchers = [matcher]
        adv_processor.token_bitmask = xgr.allocate_token_bitmask(  # type: ignore[assignment]
            1, adv_processor.full_vocab_size,
        )
        adv_processor.prefilled = False
        adv_processor.batch_size = 1

        short_prompt = prompt_fn(accumulated, first_lines, last_lines, task.task)
        messages = [
            {"role": "system", "content": task.system_prompt},
            {"role": "user", "content": short_prompt},
        ]
        # TODO: enable_thinking=True could let the adapter continue
        # thinking past the context window — test once code continuation
        # is proven.
        encoded = tokenizer.apply_chat_template(
            messages, return_tensors="pt", enable_thinking=False,
        )
        if hasattr(encoded, "input_ids"):
            input_ids = encoded["input_ids"].to(base_model_obj.device)
        else:
            input_ids = encoded.to(base_model_obj.device)

        cont_max = gen_kwargs.get("max_tokens", _CONT_MAX_TOKENS)
        attention_mask = torch.ones_like(input_ids)
        _status(f"task {task.name}: cont {attempt} generating ({cont_max} max tokens)")
        _log_memory(f"task {task.name} cont {attempt} pre-generate")
        with torch.no_grad():
            output = base_model_obj.generate(
                input_ids,
                attention_mask=attention_mask,
                pad_token_id=tokenizer.eos_token_id,
                max_new_tokens=cont_max,
                repetition_penalty=gen_kwargs.get("repetition_penalty", 1.0),
                no_repeat_ngram_size=gen_kwargs.get("no_repeat_ngram_size", 12),
                logits_processor=[adv_processor],
                **sampling,
            )

        new_tokens = output[0][input_ids.shape[1]:]
        continuation = tokenizer.decode(new_tokens, skip_special_tokens=True)
        n_new = len(new_tokens)
        total_tokens += n_new
        del output, input_ids, attention_mask, new_tokens, adv_processor

        coh = _coherence_at_boundary(accumulated, continuation)
        coherence_scores.append(coh)

        accumulated += continuation

        completed, completion_signals = _check_completion(
            accumulated, matcher, n_new, cont_max,
        )
        del matcher

        logger.info(
            "Task %s cont %d: +%d tokens, coh=%.2f, signals=%s",
            task.name, attempt, n_new, coh, completion_signals,
        )

        if completed:
            break

    _force_gc()
    avg_coherence = (
        sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0.0
    )

    return {
        "completed": completed,
        "continuations_used": (
            min(attempt + 1, max_continuations)
            if accumulated != initial_text
            else 0
        ),
        "completion_signals": completion_signals,
        "total_tokens": total_tokens,
        "accumulated_chars": len(accumulated),
        "coherence": round(avg_coherence, 4),
    }


def _run_trial(
    model: Any,
    scaling: float,
    first_lines: int,
    last_lines: int,
    prompt_strategy: str,
    trajectory_flavor: str,
    gen_kwargs: dict[str, Any],
    initial_gen_kwargs: dict[str, Any],
    compiled_grammar: Any,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    completion_count = 0
    coherence_scores: list[float] = []
    token_counts: list[int] = []

    for task in TASKS:
        result = _run_continuation_trial(
            model=model,
            task=task,
            scaling=scaling,
            first_lines=first_lines,
            last_lines=last_lines,
            prompt_strategy=prompt_strategy,
            trajectory_flavor=trajectory_flavor,
            gen_kwargs=gen_kwargs,
            initial_gen_kwargs=initial_gen_kwargs,
            compiled_grammar=compiled_grammar,
        )

        prefix = f"task/{task.name}"
        metrics[f"{prefix}/completed"] = float(result["completed"])
        metrics[f"{prefix}/continuations"] = float(result["continuations_used"])
        metrics[f"{prefix}/coherence"] = result["coherence"]
        metrics[f"{prefix}/total_tokens"] = float(result["total_tokens"])
        metrics[f"{prefix}/accumulated_chars"] = float(result["accumulated_chars"])

        for sig_name, sig_val in result["completion_signals"].items():
            metrics[f"{prefix}/signal_{sig_name}"] = float(sig_val)

        if result["completed"]:
            completion_count += 1
        coherence_scores.append(result["coherence"])
        token_counts.append(result["total_tokens"])
        _force_gc()

    n = len(TASKS)
    completion_rate = completion_count / n
    avg_coherence = sum(coherence_scores) / n
    avg_tokens = sum(token_counts) / n

    metrics["avg/completion_rate"] = round(completion_rate, 4)
    metrics["avg/coherence"] = round(avg_coherence, 4)
    metrics["avg/total_tokens"] = round(avg_tokens, 1)
    metrics["objective"] = round(completion_rate * 0.7 + avg_coherence * 0.3, 4)

    return metrics


def main() -> None:
    faulthandler.enable()

    def _sigterm_handler(signum: int, frame: Any) -> None:
        _status(f"received signal {signum} — exiting")
        logger.critical("Received signal %d, terminating", signum)
        sys.exit(128 + signum)

    signal.signal(signal.SIGTERM, _sigterm_handler)

    from rune.config import load_config  # noqa: PLC0415

    parser = argparse.ArgumentParser(description="Continuation scaling HPO")
    parser.add_argument(
        "--config",
        type=Path,
        default=_DEFAULT_CONFIG,
        help=f"Config YAML path (default: {_DEFAULT_CONFIG})",
    )
    parser.add_argument("--n-trials", type=int, default=None)
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Delete the Optuna DB and start the study from scratch",
    )
    args = parser.parse_args()

    _status("loading config")
    cfg = load_config(args.config)
    if not cfg.checkpoint_path:
        parser.error("Config must set checkpoint_path")

    hpo = cfg.hpo
    n_trials = args.n_trials or hpo.get("n_trials", 20)
    logger.info("Config loaded: model=%s, n_trials=%d", cfg.model_id, n_trials)

    from rune.engine.parse import CodeResult  # noqa: PLC0415

    cont_max = cfg.hpo.get("cont_max_tokens", _CONT_MAX_TOKENS)
    no_repeat_ngram = cfg.hpo.get("no_repeat_ngram_size", 12)
    gen_kwargs: dict[str, Any] = {
        "output_schema": CodeResult,
        "max_tokens": cont_max,
        "temperature": cfg.temperature,
        "repetition_penalty": cfg.repetition_penalty,
        "top_p": cfg.top_p,
        "top_k": cfg.top_k,
        "no_repeat_ngram_size": no_repeat_ngram,
    }
    initial_gen_kwargs: dict[str, Any] = {
        "output_schema": CodeResult,
        "max_tokens": cont_max,
        "temperature": cfg.temperature,
        "repetition_penalty": cfg.repetition_penalty,
    }

    import mlflow  # noqa: PLC0415
    import optuna  # noqa: PLC0415
    import xgrammar as xgr  # noqa: PLC0415

    from rune.model.wrapper import ModelWrapper  # noqa: PLC0415
    from rune.tracking import configure_mlflow  # noqa: PLC0415

    _status("configuring mlflow")
    configure_mlflow("continuation-scaling-hpo")

    _status("loading model")
    _log_memory("pre-model-load")
    model = ModelWrapper.from_config(cfg)
    _status("model loaded")
    _log_memory("post-model-load")

    _status("compiling grammar (reused across all trials)")
    base_model_obj = model._base_model
    tokenizer = model._tokenizer
    base_model_inner = getattr(base_model_obj, "base_model", base_model_obj)
    model_config = getattr(base_model_inner, "config", None)
    text_cfg = getattr(model_config, "text_config", model_config)
    vocab_size = getattr(text_cfg, "vocab_size", None) or tokenizer.vocab_size
    tokenizer_info = xgr.TokenizerInfo.from_huggingface(
        tokenizer, vocab_size=vocab_size,
    )
    grammar_compiler = xgr.GrammarCompiler(tokenizer_info)
    schema_json = json.dumps(CodeResult.model_json_schema())
    compiled_grammar = grammar_compiler.compile_json_schema(
        schema_json, max_whitespace_cnt=16,
    )
    del tokenizer_info, grammar_compiler

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    db_path = Path("optuna_continuation_scaling.db")
    if args.fresh and db_path.exists():
        db_path.unlink()
        logger.info("Deleted existing Optuna DB: %s", db_path)
    storage = f"sqlite:///{db_path}"

    scaling_low = cfg.hpo.get("continuation_scaling", {}).get("low", 0.8)
    scaling_high = cfg.hpo.get("continuation_scaling", {}).get("high", 2.0)

    print("\n=== CONTINUATION HPO CONFIG ===", flush=True)
    print(f"  model: {cfg.model_id}", flush=True)
    print(f"  n_trials: {n_trials}", flush=True)
    task_names = ", ".join(t.name for t in TASKS)
    print(f"  n_tasks: {len(TASKS)} ({task_names})", flush=True)
    print("  max_continuations: 2", flush=True)
    print("  --- fixed generation params ---", flush=True)
    print(f"  temperature: {cfg.temperature}", flush=True)
    print(f"  repetition_penalty: {cfg.repetition_penalty}", flush=True)
    print(f"  top_p: {cfg.top_p}", flush=True)
    print(f"  top_k: {cfg.top_k}", flush=True)
    print(f"  cont_max_tokens: {cont_max}", flush=True)
    print(f"  no_repeat_ngram_size: {no_repeat_ngram}", flush=True)
    print("  enable_thinking: False", flush=True)
    print("  --- search ranges ---", flush=True)
    print(
        f"  scaling: [{scaling_low}, {scaling_high}] log=True",
        flush=True,
    )
    print("  first_lines: [0, 5]", flush=True)
    print("  last_lines: [1, 10]", flush=True)
    ps = list(PROMPT_STRATEGIES.keys())
    tf = list(TRAJECTORY_FLAVORS.keys())
    print(f"  prompt_strategies: {ps}", flush=True)
    print(f"  trajectory_flavors: {tf}", flush=True)
    print("=" * 35, flush=True)
    print(flush=True)

    def objective(trial: optuna.Trial) -> float:
        _check_rss_limit()
        scaling = trial.suggest_float(
            "continuation_scaling", scaling_low, scaling_high, log=True,
        )
        first_lines = trial.suggest_int("first_lines", 0, 5)
        last_lines = trial.suggest_int("last_lines", 1, 10)
        prompt_strategy = trial.suggest_categorical(
            "prompt_strategy", list(PROMPT_STRATEGIES.keys()),
        )
        trajectory_flavor = trial.suggest_categorical(
            "trajectory_flavor", list(TRAJECTORY_FLAVORS.keys()),
        )

        _status(
            f"trial {trial.number}: scaling={scaling:.2f} "
            f"first={first_lines} last={last_lines} "
            f"prompt={prompt_strategy} traj={trajectory_flavor} "
            f"RSS={_rss_gb():.1f}GB"
        )

        try:
            metrics = _run_trial(
                model=model,
                scaling=scaling,
                first_lines=first_lines,
                last_lines=last_lines,
                prompt_strategy=prompt_strategy,
                trajectory_flavor=trajectory_flavor,
                gen_kwargs=gen_kwargs,
                initial_gen_kwargs=initial_gen_kwargs,
                compiled_grammar=compiled_grammar,
            )
        except MemoryError:
            logger.error("Trial %d aborted: RSS limit exceeded", trial.number)
            _status(f"trial {trial.number} aborted — RSS limit")
            raise optuna.TrialPruned() from None
        except BaseException:
            logger.exception("Trial %d FAILED with exception", trial.number)
            _status(f"trial {trial.number} FAILED — see traceback above")
            raise
        finally:
            _force_gc()

        with mlflow.start_run(
            run_name=f"trial-{trial.number}",
            nested=True,
        ):
            mlflow.log_params({
                "continuation_scaling": scaling,
                "first_lines": first_lines,
                "last_lines": last_lines,
                "prompt_strategy": prompt_strategy,
                "trajectory_flavor": trajectory_flavor,
                "trial": trial.number,
            })
            mlflow.log_metrics(metrics)

        logger.info(
            "Trial %d: obj=%.4f (completion=%.0f%% coherence=%.2f)",
            trial.number,
            metrics["objective"],
            metrics["avg/completion_rate"] * 100,
            metrics["avg/coherence"],
        )
        _status(f"trial {trial.number} done: obj={metrics['objective']:.4f}")
        return metrics["objective"]

    _status("creating optuna study")
    study = optuna.create_study(
        direction="maximize",
        study_name="continuation-scaling-hpo",
        storage=storage,
        load_if_exists=not args.fresh,
    )

    _status(f"starting study.optimize with {n_trials} trials")
    with mlflow.start_run(run_name="continuation-scaling-hpo"):
        mlflow.log_params({
            "n_trials": n_trials,
            "n_tasks": len(TASKS),
            "cont_max_tokens": _CONT_MAX_TOKENS,
            "prompt_strategies": ",".join(PROMPT_STRATEGIES.keys()),
            "trajectory_flavors": ",".join(TRAJECTORY_FLAVORS.keys()),
        })
        study.optimize(objective, n_trials=n_trials)

        n_complete = len([t for t in study.trials if t.state.name == "COMPLETE"])
        n_fail = len([t for t in study.trials if t.state.name == "FAIL"])
        logger.info(
            "Study finished: %d complete, %d failed out of %d trials",
            n_complete, n_fail, len(study.trials),
        )

        if n_complete == 0:
            _status("ALL TRIALS FAILED — no successful results")
            logger.error(
                "All %d trials failed. Check tracebacks above for root cause.",
                n_fail,
            )
            for t in study.trials:
                if t.state.name == "FAIL":
                    reason = t.system_attrs.get(
                        "fail_reason", "unknown",
                    )
                    logger.error(
                        "  Trial %d: %s", t.number, reason,
                    )
            return

        mlflow.log_params({f"best/{k}": v for k, v in study.best_params.items()})
        mlflow.log_metric("best/objective", study.best_value)

    best = study.best_params
    best_obj = (
        study.best_trial.values[0] if study.best_trial.values else 0
    )

    _status("done")
    print("\n=== FEASIBILITY RESULT ===", flush=True)
    if best_obj >= 0.5:
        print(f"  FEASIBLE: best objective={best_obj:.4f}", flush=True)
    else:
        print(f"  NOT FEASIBLE: best objective={best_obj:.4f}", flush=True)
        print("  Adapter cannot recover partial output reliably.", flush=True)
    print("\n=== BEST CONFIG ===", flush=True)
    print(f"  continuation_scaling: {best['continuation_scaling']:.4f}", flush=True)
    print(f"  first_lines: {best['first_lines']}", flush=True)
    print(f"  last_lines: {best['last_lines']}", flush=True)
    print(f"  prompt_strategy: {best['prompt_strategy']}", flush=True)
    print(f"  trajectory_flavor: {best['trajectory_flavor']}", flush=True)
    print("\n=== ALL TRIALS ===", flush=True)
    for t in sorted(
        study.trials,
        key=lambda t: t.value or 0,
        reverse=True,
    ):
        print(
            f"  #{t.number}:"
            f" scaling={t.params['continuation_scaling']:.2f}"
            f" first={t.params['first_lines']}"
            f" last={t.params['last_lines']}"
            f" prompt={t.params['prompt_strategy']}"
            f" traj={t.params['trajectory_flavor']}"
            f" obj={t.value:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except BaseException:
        logger.exception("FATAL: unhandled exception in main()")
        _status("FATAL: unhandled exception — see traceback above")
        sys.exit(1)
