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


_SYSTEM_PROMPT = (
    "Output only Python code. No commentary, no explanations, "
    "no markdown fences. Continue exactly from where the code left off."
)

TASKS = [
    ContinuationTask(
        name="calculator_divide",
        task=(
            "Write a Python Calculator class with add, subtract, multiply, "
            "divide (with ZeroDivisionError), power, and a history method "
            "that returns the last 10 operations."
        ),
        system_prompt=_SYSTEM_PROMPT,
        expected_min_chars=400,
    ),
    ContinuationTask(
        name="stack_class",
        task=(
            "Write a Python Stack class with push, pop, peek, is_empty, "
            "and size methods using a list internally."
        ),
        system_prompt=_SYSTEM_PROMPT,
        expected_min_chars=300,
    ),
    ContinuationTask(
        name="validators",
        task=(
            "Write Python functions: validate_email, validate_phone, "
            "validate_url. Each returns True/False. Use regex."
        ),
        system_prompt=_SYSTEM_PROMPT,
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


def _prompt_bare(
    accumulated: str, first_lines: int, last_lines: int, task: str,
) -> str:
    tail = _last_n_lines(accumulated, last_lines) if last_lines > 0 else ""
    if tail:
        return f"Continue the code:\n{tail}"
    return "Continue writing code."


def _prompt_task_only(
    accumulated: str, first_lines: int, last_lines: int, task: str,
) -> str:
    """Task spec + tail. Adapter carries what's done."""
    tail = _last_n_lines(accumulated, last_lines) if last_lines > 0 else ""
    parts = [f"Task: {task[:200]}"]
    parts.append("Write ONLY the next unimplemented method. No redefinitions.")
    if tail:
        parts.append(f"Resume:\n{tail}")
    return "\n".join(parts)


PROMPT_STRATEGIES: dict[str, Any] = {
    "head_tail": _prompt_head_tail,
    "tail_only": _prompt_tail_only,
    "instruction_wrapped": _prompt_instruction_wrapped,
    "bare": _prompt_bare,
    "task_only": _prompt_task_only,
}


def _edit_distance(a: str, b: str) -> float:
    if not a and not b:
        return 0.0
    return 1.0 - SequenceMatcher(None, a, b).ratio()


def _degeneration_score(text: str, n: int = 4) -> float:
    """Fraction of repeated n-grams. 0 = no repetition, 1 = all repeated."""
    words = text.split()
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    if not ngrams:
        return 0.0
    return 1.0 - len(set(ngrams)) / len(ngrams)


def _extract_code(raw: str) -> str:
    """Strip <think> blocks, markdown fences, and assistant prefix."""
    import re as _re  # noqa: PLC0415
    text = _re.sub(r"^assistant\s*", "", raw.strip())
    text = _re.sub(r"<think>.*?</think>", "", text, flags=_re.DOTALL)
    if "<think>" in text:
        text = text[:text.index("<think>")]
    text = text.strip()
    text = _re.sub(r"^Here(?:'s| is)[^\n]*\n", "", text).strip()
    blocks = _re.findall(r"```(?:python)?\n(.*?)```", text, _re.DOTALL)
    if not blocks:
        m = _re.search(r"```(?:python)?\n(.*)", text, _re.DOTALL)
        if m:
            blocks = [m.group(1)]
    if blocks:
        return "\n".join(b.rstrip() for b in blocks)
    if text:
        lines = text.splitlines()
        return "\n".join(l for l in lines if not l.startswith("```")).rstrip()
    return ""


def _dedup_code(new_code: str, accumulated: str) -> str:
    """Remove class/function re-definitions and __main__ blocks."""
    existing_defs: set[str] = set()
    for line in accumulated.splitlines():
        stripped = line.strip()
        if stripped.startswith("class ") or stripped.startswith("def "):
            name = stripped.split("(")[0].split(":")[0]
            name = name.replace("class ", "").replace("def ", "").strip()
            existing_defs.add(name)
    lines = new_code.splitlines(keepends=True)
    result: list[str] = []
    skip_until_dedent = False
    skip_indent: int | None = None
    for line in lines:
        stripped = line.strip()
        if skip_until_dedent:
            indent = len(line) - len(line.lstrip()) if line.strip() else 999
            if indent <= skip_indent and stripped:  # type: ignore[operator]
                skip_until_dedent = False
                skip_indent = None
            else:
                continue
        if stripped.startswith('if __name__'):
            skip_until_dedent = True
            skip_indent = len(line) - len(line.lstrip())
            continue
        if stripped.startswith("class ") or stripped.startswith("def "):
            name = stripped.split("(")[0].split(":")[0]
            name = name.replace("class ", "").replace("def ", "").strip()
            if name in existing_defs:
                skip_until_dedent = True
                skip_indent = len(line) - len(line.lstrip())
                continue
        result.append(line)
    return "".join(result)


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
    max_continuations: int = 5,
) -> dict[str, Any]:
    import torch  # noqa: PLC0415

    _check_rss_limit()

    base_model_obj = model._base_model
    tokenizer = model._tokenizer

    _status(f"task {task.name}: generating initial adapter")
    _log_memory(f"task {task.name} pre-adapter")
    traj_fn = TRAJECTORY_FLAVORS[trajectory_flavor]
    prompt_fn = PROMPT_STRATEGIES[prompt_strategy]

    initial_traj = traj_fn(task.task, "", 0, max_continuations)
    initial_adapter = model.generate_adapter(initial_traj, offload_base=False)
    adapter_template = {
        k: torch.zeros_like(v) for k, v in initial_adapter.state_dict.items()
    }
    model.hotswap_adapter(
        _scale_b_only_inplace(initial_adapter.state_dict, scaling),
    )
    del initial_adapter
    _force_gc()

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

    cont_max = gen_kwargs.get("max_tokens", _CONT_MAX_TOKENS)
    accumulated = ""
    total_tokens = 0
    degen_scores: list[float] = []
    empty_rounds = 0

    for attempt in range(max_continuations):
        _check_rss_limit()
        _status(f"task {task.name}: round {attempt}/{max_continuations}")
        _log_memory(f"task {task.name} round {attempt}")
        _force_gc()

        trajectory_text = traj_fn(task.task, accumulated, attempt, max_continuations)
        cont_adapter = model.generate_adapter(trajectory_text, offload_base=False)
        model.hotswap_adapter(
            _scale_b_only_inplace(cont_adapter.state_dict, scaling),
        )
        del cont_adapter
        _force_gc()

        short_prompt = prompt_fn(accumulated, first_lines, last_lines, task.task)
        messages = [
            {"role": "system", "content": task.system_prompt},
            {"role": "user", "content": short_prompt},
        ]
        encoded = tokenizer.apply_chat_template(
            messages, return_tensors="pt", enable_thinking=False,
            add_generation_prompt=True,
        )
        if hasattr(encoded, "input_ids"):
            input_ids = encoded["input_ids"].to(base_model_obj.device)
        else:
            input_ids = encoded.to(base_model_obj.device)

        attention_mask = torch.ones_like(input_ids)
        _status(f"task {task.name}: round {attempt} generating ({cont_max} tokens)")
        with torch.no_grad():
            output = base_model_obj.generate(
                input_ids,
                attention_mask=attention_mask,
                pad_token_id=tokenizer.eos_token_id,
                max_new_tokens=cont_max,
                repetition_penalty=gen_kwargs.get("repetition_penalty", 1.0),
                no_repeat_ngram_size=gen_kwargs.get("no_repeat_ngram_size", 12),
                **sampling,
            )

        new_tokens = output[0][input_ids.shape[1]:]
        raw_continuation = tokenizer.decode(new_tokens, skip_special_tokens=True)
        n_new = len(new_tokens)
        total_tokens += n_new
        stopped_early = n_new < cont_max
        del output, input_ids, attention_mask, new_tokens

        code = _extract_code(raw_continuation)
        code = _dedup_code(code, accumulated)

        degen = _degeneration_score(raw_continuation)
        degen_scores.append(degen)

        if code.strip():
            accumulated = accumulated.rstrip() + "\n" + code.strip() + "\n" if accumulated else code.strip() + "\n"
            empty_rounds = 0
        else:
            empty_rounds += 1

        logger.info(
            "Task %s round %d: +%d tokens, %d code chars, degen=%.2f, eos=%s",
            task.name, attempt, n_new, len(code), degen, stopped_early,
        )

        if stopped_early:
            break
        if empty_rounds >= 2:
            break

    # Zero-adapter baseline: same prompt from start, no adapter.
    _force_gc()
    model.hotswap_adapter(adapter_template)
    baseline_prompt = prompt_fn("", first_lines, last_lines, task.task)
    messages = [
        {"role": "system", "content": task.system_prompt},
        {"role": "user", "content": baseline_prompt},
    ]
    encoded = tokenizer.apply_chat_template(
        messages, return_tensors="pt", enable_thinking=False,
        add_generation_prompt=True,
    )
    if hasattr(encoded, "input_ids"):
        input_ids = encoded["input_ids"].to(base_model_obj.device)
    else:
        input_ids = encoded.to(base_model_obj.device)
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        output = base_model_obj.generate(
            input_ids,
            attention_mask=attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=cont_max,
            repetition_penalty=gen_kwargs.get("repetition_penalty", 1.0),
            **sampling,
        )
    baseline_tokens = output[0][input_ids.shape[1]:]
    baseline_text = _extract_code(
        tokenizer.decode(baseline_tokens, skip_special_tokens=True),
    )
    del output, input_ids, attention_mask, baseline_tokens
    adapter_diff = _edit_distance(baseline_text, accumulated)
    del baseline_text, adapter_template
    _force_gc()

    avg_degen = (
        sum(degen_scores) / len(degen_scores) if degen_scores else 0.0
    )

    completed = len(accumulated) >= task.expected_min_chars

    return {
        "completed": completed,
        "continuations_used": attempt + 1,
        "completion_signals": {
            "gen_stopped_early": stopped_early,
            "enough_chars": completed,
            "empty_rounds": empty_rounds,
        },
        "total_tokens": total_tokens,
        "accumulated_chars": len(accumulated),
        "degeneration": round(avg_degen, 4),
        "adapter_diff": round(adapter_diff, 4),
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
    compiled_grammar: Any = None,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    completion_count = 0
    degen_scores: list[float] = []
    diff_scores: list[float] = []
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
        metrics[f"{prefix}/degeneration"] = result["degeneration"]
        metrics[f"{prefix}/adapter_diff"] = result["adapter_diff"]
        metrics[f"{prefix}/total_tokens"] = float(result["total_tokens"])
        metrics[f"{prefix}/accumulated_chars"] = float(result["accumulated_chars"])

        for sig_name, sig_val in result["completion_signals"].items():
            metrics[f"{prefix}/signal_{sig_name}"] = float(sig_val)

        if result["completed"]:
            completion_count += 1
        degen_scores.append(result["degeneration"])
        diff_scores.append(result["adapter_diff"])
        token_counts.append(result["total_tokens"])
        _force_gc()

    n = len(TASKS)
    completion_rate = completion_count / n
    avg_degen = sum(degen_scores) / n
    avg_diff = sum(diff_scores) / n
    avg_tokens = sum(token_counts) / n

    metrics["avg/completion_rate"] = round(completion_rate, 4)
    metrics["avg/degeneration"] = round(avg_degen, 4)
    metrics["avg/adapter_diff"] = round(avg_diff, 4)
    metrics["avg/total_tokens"] = round(avg_tokens, 1)
    metrics["objective"] = round(
        completion_rate * 0.4
        + (1.0 - avg_degen) * 0.3
        + avg_diff * 0.3,
        4,
    )

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

    cont_max = cfg.hpo.get("cont_max_tokens", _CONT_MAX_TOKENS)
    no_repeat_ngram = cfg.hpo.get("no_repeat_ngram_size", 12)
    gen_kwargs: dict[str, Any] = {
        "max_tokens": cont_max,
        "temperature": cfg.temperature,
        "repetition_penalty": cfg.repetition_penalty,
        "top_p": cfg.top_p,
        "top_k": cfg.top_k,
        "no_repeat_ngram_size": no_repeat_ngram,
    }
    initial_gen_kwargs: dict[str, Any] = {
        "max_tokens": cont_max,
        "temperature": cfg.temperature,
        "repetition_penalty": cfg.repetition_penalty,
    }

    import mlflow  # noqa: PLC0415
    import optuna  # noqa: PLC0415

    from rune.model.wrapper import ModelWrapper  # noqa: PLC0415
    from rune.tracking import configure_mlflow  # noqa: PLC0415

    _status("configuring mlflow")
    configure_mlflow("continuation-scaling-hpo")

    _status("loading model")
    _log_memory("pre-model-load")
    model = ModelWrapper.from_config(cfg)
    _status("model loaded")
    _log_memory("post-model-load")

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
    print("  max_continuations: 5", flush=True)
    print("  mode: plaintext (no xgrammar)", flush=True)
    print("  --- fixed generation params ---", flush=True)
    print(f"  temperature: {cfg.temperature}", flush=True)
    print(f"  repetition_penalty: {cfg.repetition_penalty}", flush=True)
    print(f"  top_p: {cfg.top_p}", flush=True)
    print(f"  top_k: {cfg.top_k}", flush=True)
    print(f"  cont_max_tokens: {cont_max}", flush=True)
    print(f"  no_repeat_ngram_size: {no_repeat_ngram}", flush=True)
    print("  enable_thinking: False (add_generation_prompt=True)", flush=True)
    print("  stop: EOS or 2 consecutive empty rounds", flush=True)
    print("  --- search ranges ---", flush=True)
    print(
        f"  scaling: [{scaling_low}, {scaling_high}] log=True",
        flush=True,
    )
    print("  last_lines: [2, 8]", flush=True)
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
        last_lines = trial.suggest_int("last_lines", 2, 8)
        prompt_strategy = trial.suggest_categorical(
            "prompt_strategy", list(PROMPT_STRATEGIES.keys()),
        )
        trajectory_flavor = trial.suggest_categorical(
            "trajectory_flavor", list(TRAJECTORY_FLAVORS.keys()),
        )

        _status(
            f"trial {trial.number}: scaling={scaling:.2f} "
            f"last={last_lines} "
            f"prompt={prompt_strategy} traj={trajectory_flavor} "
            f"RSS={_rss_gb():.1f}GB"
        )

        try:
            metrics = _run_trial(
                model=model,
                scaling=scaling,
                first_lines=0,
                last_lines=last_lines,
                prompt_strategy=prompt_strategy,
                trajectory_flavor=trajectory_flavor,
                gen_kwargs=gen_kwargs,
                initial_gen_kwargs=initial_gen_kwargs,
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
                "last_lines": last_lines,
                "prompt_strategy": prompt_strategy,
                "trajectory_flavor": trajectory_flavor,
                "trial": trial.number,
            })
            mlflow.log_metrics(metrics)

        logger.info(
            "Trial %d: obj=%.4f (completion=%.0f%% degen=%.2f diff=%.2f)",
            trial.number,
            metrics["objective"],
            metrics["avg/completion_rate"] * 100,
            metrics["avg/degeneration"],
            metrics["avg/adapter_diff"],
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
