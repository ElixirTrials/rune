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
import json
import logging
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

logging.basicConfig(
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

_DEFAULT_CONFIG = Path("benchmarks/bench.yaml")
_CONT_MAX_TOKENS = 256


@dataclass(frozen=True)
class ContinuationTask:
    name: str
    task: str
    system_prompt: str
    expected_min_chars: int


TASKS = [
    ContinuationTask(
        name="calculator_class",
        task=(
            "Write a Python module with a Calculator class that supports "
            "add, subtract, multiply, divide (with ZeroDivisionError handling), "
            "power, and a history method that returns the last 10 operations. "
            "Include type hints and a test class with at least 8 test methods."
        ),
        system_prompt="You are a code generator.",
        expected_min_chars=1500,
    ),
    ContinuationTask(
        name="linked_list",
        task=(
            "Write a Python module with a generic doubly-linked list supporting "
            "append, prepend, insert_at, delete, find, reverse, and __iter__. "
            "Include a comprehensive test suite with edge cases for empty list, "
            "single element, and boundary conditions."
        ),
        system_prompt="You are a code generator.",
        expected_min_chars=2500,
    ),
    ContinuationTask(
        name="rest_api_models",
        task=(
            "Write a Python module with Pydantic models for a REST API: User, "
            "UserCreate, UserUpdate, PaginatedResponse[T], ErrorResponse. "
            "Include validators, a UserRepository class with in-memory CRUD "
            "operations, and test coverage for all validation rules and CRUD paths."
        ),
        system_prompt="You are a code generator.",
        expected_min_chars=3500,
    ),
]


def _first_n_lines(text: str, n: int) -> str:
    lines = text.splitlines()
    return "\n".join(lines[:n])


def _last_n_lines(text: str, n: int) -> str:
    lines = text.splitlines()
    return "\n".join(lines[-n:]) if n > 0 else ""


def _scale_b_only(sd: dict[str, Any], factor: float) -> dict[str, Any]:
    return {k: v * factor if "lora_B" in k else v for k, v in sd.items()}


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


TRAJECTORY_FLAVORS: dict[str, Any] = {
    "minimal_goal_code": _traj_minimal,
    "with_attempt_counter": _traj_with_counter,
    "with_structural_summary": _traj_with_structure,
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
    max_continuations: int = 5,
) -> dict[str, Any]:
    import torch  # noqa: PLC0415
    import xgrammar as xgr  # noqa: PLC0415

    from rune.engine.parse import CodeResult  # noqa: PLC0415

    initial_adapter = model.generate_adapter(
        f"ROLE: coder\nTASK: {task.task}\nPLAN: Implement with tests."
    )
    model.hotswap_adapter(
        _scale_b_only(initial_adapter.state_dict, scaling),
    )

    initial_result = asyncio.run(
        model.generate(
            task.task,
            system_prompt=task.system_prompt,
            **gen_kwargs,
        )
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

    torch.cuda.empty_cache()
    base_model_obj = model._base_model
    tokenizer = model._tokenizer

    base_model_inner = getattr(base_model_obj, "base_model", base_model_obj)
    model_config = getattr(base_model_inner, "config", None)
    text_cfg = getattr(model_config, "text_config", model_config)
    vocab_size = getattr(text_cfg, "vocab_size", None) or tokenizer.vocab_size

    tokenizer_info = xgr.TokenizerInfo.from_huggingface(
        tokenizer, vocab_size=vocab_size,
    )
    compiler = xgr.GrammarCompiler(tokenizer_info)
    schema_json = json.dumps(CodeResult.model_json_schema())
    compiled = compiler.compile_json_schema(schema_json, max_whitespace_cnt=16)

    accumulated = initial_result.text
    total_tokens = initial_result.tokens_used
    coherence_scores: list[float] = []
    traj_fn = TRAJECTORY_FLAVORS[trajectory_flavor]
    prompt_fn = PROMPT_STRATEGIES[prompt_strategy]
    completed = False
    completion_signals: dict[str, bool] = {}

    sampling: dict[str, Any] = {}
    temp = gen_kwargs.get("temperature", 0.3)
    if temp > 0:
        sampling = {"do_sample": True, "temperature": temp, "top_p": 0.9}
    else:
        sampling = {"do_sample": False}

    for attempt in range(max_continuations):
        torch.cuda.empty_cache()
        trajectory_text = traj_fn(task.task, accumulated, attempt, max_continuations)
        cont_adapter = model.generate_adapter(trajectory_text)
        model.hotswap_adapter(
            _scale_b_only(cont_adapter.state_dict, scaling),
        )

        matcher = xgr.GrammarMatcher(compiled)
        if not matcher.accept_string(accumulated):
            logger.error(
                "Grammar reject at continuation %d for %s",
                attempt, task.name,
            )
            break

        adv_processor = xgr.contrib.hf.LogitsProcessor(compiled)
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
        encoded = tokenizer.apply_chat_template(messages, return_tensors="pt")
        if hasattr(encoded, "input_ids"):
            input_ids = encoded["input_ids"].to(base_model_obj.device)
        else:
            input_ids = encoded.to(base_model_obj.device)

        cont_max = gen_kwargs.get("max_tokens", _CONT_MAX_TOKENS)
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            output = base_model_obj.generate(
                input_ids,
                attention_mask=attention_mask,
                pad_token_id=tokenizer.eos_token_id,
                max_new_tokens=cont_max,
                repetition_penalty=gen_kwargs.get("repetition_penalty", 1.1),
                logits_processor=[adv_processor],
                **sampling,
            )

        new_tokens = output[0][input_ids.shape[1]:]
        continuation = tokenizer.decode(new_tokens, skip_special_tokens=True)
        n_new = len(new_tokens)
        total_tokens += n_new
        del output, input_ids, attention_mask, new_tokens

        coh = _coherence_at_boundary(accumulated, continuation)
        coherence_scores.append(coh)

        accumulated += continuation

        completed, completion_signals = _check_completion(
            accumulated, matcher, n_new, cont_max,
        )

        logger.info(
            "Task %s cont %d: +%d tokens, coh=%.2f, signals=%s",
            task.name, attempt, n_new, coh, completion_signals,
        )

        if completed:
            break

    avg_coherence = (
        sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0.0
    )

    return {
        "completed": completed,
        "continuations_used": (
            min(attempt + 1, max_continuations)
            if accumulated != initial_result.text
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
    from rune.config import load_config  # noqa: PLC0415

    parser = argparse.ArgumentParser(description="Continuation scaling HPO")
    parser.add_argument(
        "--config",
        type=Path,
        default=_DEFAULT_CONFIG,
        help=f"Config YAML path (default: {_DEFAULT_CONFIG})",
    )
    parser.add_argument("--n-trials", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if not cfg.checkpoint_path:
        parser.error("Config must set checkpoint_path")

    hpo = cfg.hpo
    n_trials = args.n_trials or hpo.get("n_trials", 20)

    from rune.engine.parse import CodeResult  # noqa: PLC0415

    gen_kwargs: dict[str, Any] = {
        "output_schema": CodeResult,
        "max_tokens": _CONT_MAX_TOKENS,
        "temperature": cfg.bench.get("gen_temperature", 0.01),
        "repetition_penalty": cfg.repetition_penalty,
    }

    import mlflow  # noqa: PLC0415
    import optuna  # noqa: PLC0415

    from rune.model.wrapper import ModelWrapper  # noqa: PLC0415
    from rune.tracking import configure_mlflow  # noqa: PLC0415

    configure_mlflow("continuation-scaling-hpo")
    model = ModelWrapper.from_config(cfg)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        scaling = trial.suggest_float(
            "continuation_scaling", 0.8, 10.0, log=True,
        )
        first_lines = trial.suggest_int("first_lines", 0, 5)
        last_lines = trial.suggest_int("last_lines", 1, 10)
        prompt_strategy = trial.suggest_categorical(
            "prompt_strategy", list(PROMPT_STRATEGIES.keys()),
        )
        trajectory_flavor = trial.suggest_categorical(
            "trajectory_flavor", list(TRAJECTORY_FLAVORS.keys()),
        )

        logger.info(
            "Trial %d: scaling=%.2f first=%d last=%d prompt=%s traj=%s",
            trial.number, scaling, first_lines, last_lines,
            prompt_strategy, trajectory_flavor,
        )

        metrics = _run_trial(
            model=model,
            scaling=scaling,
            first_lines=first_lines,
            last_lines=last_lines,
            prompt_strategy=prompt_strategy,
            trajectory_flavor=trajectory_flavor,
            gen_kwargs=gen_kwargs,
        )

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
        return metrics["objective"]

    study = optuna.create_study(
        direction="maximize",
        study_name="continuation-scaling-hpo",
    )

    with mlflow.start_run(run_name="continuation-scaling-hpo"):
        mlflow.log_params({
            "n_trials": n_trials,
            "n_tasks": len(TASKS),
            "cont_max_tokens": _CONT_MAX_TOKENS,
            "prompt_strategies": ",".join(PROMPT_STRATEGIES.keys()),
            "trajectory_flavors": ",".join(TRAJECTORY_FLAVORS.keys()),
        })
        study.optimize(objective, n_trials=n_trials)

        mlflow.log_params({f"best/{k}": v for k, v in study.best_params.items()})
        mlflow.log_metric("best/objective", study.best_value)

    best = study.best_params
    best_obj = (
        study.best_trial.values[0] if study.best_trial.values else 0
    )

    print("\n=== FEASIBILITY RESULT ===")
    if best_obj >= 0.5:
        print(f"  FEASIBLE: best objective={best_obj:.4f}")
    else:
        print(f"  NOT FEASIBLE: best objective={best_obj:.4f}")
        print("  Adapter cannot recover partial output reliably.")
    print("\n=== BEST CONFIG ===")
    print(f"  continuation_scaling: {best['continuation_scaling']:.4f}")
    print(f"  first_lines: {best['first_lines']}")
    print(f"  last_lines: {best['last_lines']}")
    print(f"  prompt_strategy: {best['prompt_strategy']}")
    print(f"  trajectory_flavor: {best['trajectory_flavor']}")
    print("\n=== ALL TRIALS ===")
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
            f" obj={t.value:.4f}"
        )


if __name__ == "__main__":
    main()
