"""Capacity sweep: test adapter-as-memory at increasing trajectory sizes.

Loads the model once, runs each scenario at scaling=0.75 (adapter active)
and scaling=0.01 (baseline, ~no adapter), saves results for comparison.

Run:
  uv run python tools/capacity_sweep.py
"""

from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from tools.cont_probe import (
    PROMPT_TEMPLATES,
    SCENARIOS,
    TRAJECTORY_FLAVORS,
    _extract_code,
    _extract_think,
    _generate_plaintext,
    _scale_b_only_inplace,
    _summarize_think,
)


CONFIGS = [
    {"scenario": "mid_fn", "label": "tiny (~170 tok)"},
    {"scenario": "large_small", "label": "small (~1000 tok)"},
    {"scenario": "large_medium", "label": "medium (~1165 tok)"},
]

SCALING_LEVELS = [
    (0.25, "adapter_025"),
    (0.49, "adapter_049"),
    (0.01, "baseline"),
]
MAX_TOKENS = 512
TRAJECTORY = "code_template"
PROMPT = "structural"
LAST_LINES = 4


SUCCESS_CRITERIA = {
    "mid_fn": {
        "expected_funcs": ["power", "history"],
        "must_reference": ["self", "_history"],
        "must_not_redefine": ["add", "subtract", "multiply"],
    },
    "large_small": {
        "expected_funcs": ["filter_by_tags", "deduplicate", "sort_by_timestamp"],
        "must_reference": ["Record", "list"],
        "must_not_redefine": ["Pipeline", "Record", "validate", "load_json"],
    },
    "large_medium": {
        "expected_funcs": ["export_json", "export_csv"],
        "must_reference": ["Record", "json", "csv"],
        "must_not_redefine": ["Pipeline", "Record", "validate", "filter_by_tags"],
    },
}


def evaluate(code: str, scenario: str) -> dict[str, Any]:
    """Check extracted code against success criteria."""
    criteria = SUCCESS_CRITERIA[scenario]
    found_funcs = []
    redefined = []
    for func in criteria["expected_funcs"]:
        if f"def {func}" in code:
            found_funcs.append(func)
    for name in criteria["must_not_redefine"]:
        if f"class {name}" in code or f"def {name}" in code:
            redefined.append(name)
    refs_found = [ref for ref in criteria["must_reference"] if ref in code]
    return {
        "expected_funcs": criteria["expected_funcs"],
        "found_funcs": found_funcs,
        "func_coverage": len(found_funcs) / max(len(criteria["expected_funcs"]), 1),
        "redefined_existing": redefined,
        "references_found": refs_found,
        "ref_coverage": len(refs_found) / max(len(criteria["must_reference"]), 1),
        "has_syntax_error": _check_syntax(code),
        "code_chars": len(code),
    }


def _check_syntax(code: str) -> bool:
    try:
        compile(code, "<string>", "exec")
        return False
    except SyntaxError:
        return True


def main() -> None:
    from rune.config import load_config
    from rune.model.wrapper import ModelWrapper

    cfg = load_config(Path("benchmarks/bench.yaml"))
    print("Loading model...", file=sys.stderr, flush=True)
    model = ModelWrapper.from_config(cfg)
    base_model = model._base_model
    tokenizer = model._tokenizer

    traj_fn = TRAJECTORY_FLAVORS[TRAJECTORY]
    prompt_fn = PROMPT_TEMPLATES[PROMPT]
    system_prompt = (
        "Output only Python code. No commentary, no explanations, "
        "no markdown fences. Continue exactly from where the code left off."
    )
    no_repeat_ngram = cfg.hpo.get("no_repeat_ngram_size", 12)

    ts = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    run_dir = Path("runs") / "capacity_sweep" / ts
    run_dir.mkdir(parents=True, exist_ok=True)

    results = []

    import torch

    for conf in CONFIGS:
        scenario = conf["scenario"]
        task_text, truncate_fn = SCENARIOS[scenario]
        _full, accumulated = truncate_fn()

        for scaling, tag in SCALING_LEVELS:
            torch.cuda.empty_cache()
            name = f"{scenario}_{tag}"
            print(f"\n=== {name} (scaling={scaling}) ===", file=sys.stderr, flush=True)
            print(f"Accumulated: {len(accumulated)} chars", file=sys.stderr, flush=True)

            trajectory = traj_fn(task_text, accumulated, 10)
            print(f"Trajectory: {len(trajectory)} chars", file=sys.stderr, flush=True)

            traj_tokens = tokenizer(
                trajectory, truncation=True, max_length=2048, return_tensors="pt",
            )
            actual_traj_tokens = traj_tokens["input_ids"].shape[1]
            print(f"Trajectory tokens: {actual_traj_tokens}/2048", file=sys.stderr, flush=True)

            adapter = model.generate_adapter(trajectory, offload_base=False)
            model.hotswap_adapter(
                _scale_b_only_inplace(adapter.state_dict, scaling),
            )

            prompt = prompt_fn(accumulated, 0, LAST_LINES, task_text)

            continuation, n_tokens = _generate_plaintext(
                base_model, tokenizer, prompt, system_prompt,
                MAX_TOKENS, cfg.temperature, cfg.repetition_penalty,
                cfg.top_p, cfg.top_k, no_repeat_ngram,
            )

            code = _extract_code(continuation)
            think = _extract_think(continuation)
            think_summary = _summarize_think(think)
            eval_result = evaluate(code, scenario)

            out_dir = run_dir / name
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "accumulated.txt").write_text(accumulated)
            (out_dir / "trajectory.txt").write_text(trajectory)
            (out_dir / "prompt.txt").write_text(prompt)
            (out_dir / "raw_continuation.txt").write_text(continuation)
            (out_dir / "code.txt").write_text(code)
            (out_dir / "eval.json").write_text(json.dumps(eval_result, indent=2) + "\n")

            row = {
                "name": name,
                "scenario": scenario,
                "scaling": scaling,
                "tag": tag,
                "accumulated_chars": len(accumulated),
                "traj_tokens": actual_traj_tokens,
                "gen_tokens": n_tokens,
                "stopped_early": n_tokens < MAX_TOKENS,
                "think_summary": think_summary[:100],
                **eval_result,
            }
            results.append(row)

            print(f"  Tokens: {n_tokens}, stopped_early={n_tokens < MAX_TOKENS}", flush=True)
            if think_summary:
                print(f"  Think: {think_summary[:80]}", flush=True)
            print(f"  Code: {len(code)} chars", flush=True)
            print(f"  Funcs found: {eval_result['found_funcs']}", flush=True)
            print(f"  Coverage: {eval_result['func_coverage']:.0%}", flush=True)
            print(f"  Redefined: {eval_result['redefined_existing']}", flush=True)

    (run_dir / "summary.json").write_text(json.dumps(results, indent=2) + "\n")

    print(f"\n{'='*60}", flush=True)
    print(f"CAPACITY SWEEP RESULTS", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"{'Name':<30} {'Tok':>5} {'TrTok':>5} {'Funcs':>10} {'Cov':>5} {'Redef':>6} {'Syn':>4}", flush=True)
    print("-" * 70, flush=True)
    for r in results:
        funcs = f"{len(r['found_funcs'])}/{len(r['expected_funcs'])}"
        print(
            f"{r['name']:<30} {r['gen_tokens']:>5} {r['traj_tokens']:>5} "
            f"{funcs:>10} {r['func_coverage']:>4.0%} {len(r['redefined_existing']):>6} "
            f"{'ERR' if r['has_syntax_error'] else 'ok':>4}",
            flush=True,
        )
    print(f"\nResults saved to: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
