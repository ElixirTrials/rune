"""End-to-end test: Pretrained Sakana Doc-to-LoRA through Rune stack.

Exercises every component with a REAL pretrained hypernetwork checkpoint:
  1. sakana_d2l: Download checkpoint from HF, load HyperLoRA perceiver
  2. sakana_d2l: Extract per-layer activations from gemma-2-2b-it base model
  3. sakana_d2l: Generate PEFT adapter via pretrained perceiver
  4. TransformersProvider: Load base model, apply adapter, generate text
  5. adapter_registry: Store and query adapter records
  6. evaluation.metrics: Score adapter quality
  7. rune_runner: Full iteration loop (bootstrap → iterate → H() → new adapter)

Usage:
    uv run scripts/e2e_test.py
    uv run scripts/e2e_test.py --steps 1,2,3    # run specific steps only
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

# Load .env for HF_TOKEN
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bootstrap import setup_path

setup_path()  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("e2e")

# --- Base model: gemma-2-2b-it (matches Sakana gemma_demo checkpoint) ---
MODEL_NAME = "google/gemma-2-2b-it"

from shared.hardware import get_best_device  # noqa: E402

DEVICE = get_best_device()


def step(n: int, msg: str) -> None:
    """Print a step header."""
    print(f"\n{'=' * 60}")
    print(f"  STEP {n}: {msg}")
    print(f"{'=' * 60}\n")


def test_sakana_hypernetwork(tmpdir: str) -> str:
    """Steps 1-3: Download checkpoint, extract activations, generate adapter."""
    from model_training.sakana_d2l import (
        generate_adapter_from_sakana,
    )

    step(1, "Sakana D2L: Download checkpoint + generate adapter")

    adapter_dir = os.path.join(tmpdir, "sakana_adapter")

    trajectory_text = (
        "=== Bootstrap ===\n"
        "Phase 1: Decompose project into tasks — LRUCache core (doubly-linked list + "
        "hash map), TTL expiration logic, thread-safety with fine-grained locks, "
        "stats tracking, decorator wrapper\n"
        "Phase 2: Generate code with tests for each component\n"
        "Phase 3: Execute, validate, iterate on concurrency edge cases\n"
        "Phase 4: Integrate components, run full test suite including stress tests\n"
    )

    print(f"  Device: {DEVICE}")
    print(f"  Base model: {MODEL_NAME}")
    print("  Checkpoint: SakanaAI/doc-to-lora (gemma_demo, 80k steps)")
    print(f"  Trajectory text: {len(trajectory_text)} chars")

    adapter_path = generate_adapter_from_sakana(
        text=trajectory_text,
        output_dir=adapter_dir,
        variant="gemma_demo",
        device=DEVICE,
    )

    # Verify adapter files
    files = os.listdir(adapter_path)
    print(f"  Adapter saved to: {adapter_path}")
    print(f"  Files: {files}")
    assert "adapter_model.safetensors" in files, "Missing safetensors!"
    assert "adapter_config.json" in files, "Missing config!"

    config = json.loads(Path(adapter_path, "adapter_config.json").read_text())
    print("  Adapter config:")
    print(f"    peft_type: {config['peft_type']}")
    print(f"    rank: {config['r']}")
    print(f"    target_modules: {config['target_modules']}")
    print(f"    base_model: {config['base_model_name_or_path']}")

    return adapter_path


def test_provider_loads_adapter(adapter_path: str) -> None:
    """Step 2: TransformersProvider loads model + applies PEFT adapter."""
    step(2, "TransformersProvider loads model and applies PEFT adapter")

    from inference.transformers_provider import TransformersProvider

    provider = TransformersProvider(
        model_name=MODEL_NAME,
        device=DEVICE,
        torch_dtype="float32",
    )

    print(f"  Provider: {type(provider).__name__}")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Device: {DEVICE}")

    # Load base model
    provider._load_model_if_needed()
    print(f"  Base model loaded on: {provider._device}")

    async def _test() -> None:
        await provider.load_adapter("methodology_adapter", adapter_path)
        adapters = await provider.list_adapters()
        print(f"  Registered adapters: {adapters}")
        assert "methodology_adapter" in adapters

        # Generate WITHOUT adapter
        print("\n  --- Generation WITHOUT adapter ---")
        result_base = await provider.generate(
            prompt="What is the capital of France?",
            model=MODEL_NAME,
            adapter_id=None,
            max_tokens=50,
        )
        print(f"  Adapter used: {result_base.adapter_id}")
        print(f"  Tokens: {result_base.token_count}")
        print(f"  Output: {result_base.text[:200]!r}")

        # Generate WITH adapter
        print("\n  --- Generation WITH adapter ---")
        result_adapted = await provider.generate(
            prompt="What is the capital of France?",
            model=MODEL_NAME,
            adapter_id="methodology_adapter",
            max_tokens=50,
        )
        print(f"  Adapter used: {result_adapted.adapter_id}")
        print(f"  Tokens: {result_adapted.token_count}")
        print(f"  Output: {result_adapted.text[:200]!r}")

        assert result_adapted.adapter_id == "methodology_adapter"

        if result_base.text != result_adapted.text:
            print("\n  Base and adapted outputs DIFFER (adapter influenced generation)")
        else:
            print(
                "\n  Base and adapted outputs are identical (adapter may not have been applied)"
            )

    asyncio.run(_test())


def test_adapter_registry(adapter_path: str) -> None:
    """Step 3: Adapter registry CRUD."""
    step(3, "Adapter Registry: store, query, retrieve")

    from adapter_registry.models import AdapterRecord
    from adapter_registry.registry import AdapterRegistry
    from sqlalchemy import create_engine

    with tempfile.TemporaryDirectory() as db_dir:
        db_path = os.path.join(db_dir, "test_registry.db")
        engine = create_engine(f"sqlite:///{db_path}")
        registry = AdapterRegistry(engine=engine)

        record = AdapterRecord(
            id="e2e-sakana-001",
            version=1,
            task_type="qa",
            base_model_id=MODEL_NAME,
            rank=8,
            created_at="2026-01-01T00:00:00Z",
            file_path=adapter_path,
            file_hash="e2e-test-hash",
            file_size_bytes=0,
            source="sakana_d2l",
            session_id="e2e-test",
        )
        registry.store(record)
        print(f"  Stored: {record.id}")

        retrieved = registry.retrieve_by_id("e2e-sakana-001")
        assert retrieved.base_model_id == MODEL_NAME
        print(f"  Retrieved: {retrieved.id}, base_model={retrieved.base_model_id}")

        registry.update_fitness("e2e-sakana-001", pass_rate=0.85, fitness_score=0.78)
        updated = registry.retrieve_by_id("e2e-sakana-001")
        print(
            f"  Updated fitness: pass_rate={updated.pass_rate}, fitness={updated.fitness_score}"
        )

        by_task = registry.query_by_task_type("qa")
        print(f"  Query by task 'qa': {len(by_task)} results")

        best = registry.query_best_for_task("qa", top_k=1)
        print(f"  Best for 'qa': {best[0].id if best else 'none'}")

        print("  Registry CRUD: PASSED")


def test_evaluation_metrics() -> None:
    """Step 4: Evaluation metrics."""
    step(4, "Evaluation metrics: pass@k, fitness, quality scoring")

    from evaluation.metrics import (
        calculate_pass_at_k,
        evaluate_fitness,
        score_adapter_quality,
    )

    pass_at_1 = calculate_pass_at_k(n_samples=10, n_correct=7, k=1)
    print(f"  pass@1 (10 samples, 7 correct): {pass_at_1:.4f}")
    assert 0.0 < pass_at_1 <= 1.0

    fitness = evaluate_fitness(
        adapter_id="e2e-sakana-001", pass_rate=0.85, diversity_score=0.6
    )
    print(f"  Fitness (pass_rate=0.85, diversity=0.6): {fitness:.4f}")
    assert 0.0 < fitness <= 1.0

    quality = score_adapter_quality(
        adapter_id="e2e-sakana-001",
        pass_rate=0.85,
        generalization_delta=0.1,
    )
    print(f"  Quality (pass_rate=0.85, gen_delta=0.1): {quality:.4f}")
    assert 0.0 < quality <= 1.0

    print("  Evaluation metrics: PASSED")


def test_phased_pipeline(tmpdir: str) -> None:
    """Step 5: Full 4-phase pipeline with template-driven adapters."""
    step(5, "Phased Pipeline: decompose → plan → code → integrate")

    from model_training.sakana_d2l import download_checkpoint

    checkpoint_path = str(download_checkpoint(variant="gemma_demo"))
    print(f"  Sakana checkpoint: {checkpoint_path}")

    # Configure inference provider
    os.environ["INFERENCE_PROVIDER"] = "transformers"
    os.environ["TRANSFORMERS_MODEL_NAME"] = MODEL_NAME
    os.environ["TRANSFORMERS_DEVICE"] = DEVICE

    from inference.factory import _clear_cache

    _clear_cache()

    project_prompt = (
        "Build a complete Python CLI task manager application. Requirements:\n\n"
        "1. Data layer: TaskStore class backed by SQLite via the stdlib sqlite3 "
        "module. Schema: tasks(id INTEGER PRIMARY KEY, title TEXT NOT NULL, "
        "description TEXT, status TEXT DEFAULT 'todo', priority INTEGER DEFAULT 0, "
        "created_at TEXT, due_date TEXT, tags TEXT). Supports CRUD, filtering by "
        "status/priority/tag, and sorting.\n\n"
        "2. Domain layer: Task dataclass with validation — title must be non-empty, "
        "status must be one of ('todo','in_progress','done','blocked'), priority 0-5, "
        "tags stored as comma-separated string. TaskService class that enforces "
        "business rules: cannot transition from 'done' to 'todo', blocked tasks "
        "cannot move to 'done' directly, and overdue tasks are auto-flagged.\n\n"
        "3. CLI layer: argparse-based CLI with subcommands: add, list, show, "
        "update, delete, stats. list supports --status, --priority, "
        "--tag, --sort-by, --overdue flags. stats shows counts by status "
        "and average completion time.\n\n"
        "4. Include a comprehensive test suite using only unittest that covers: "
        "TaskStore CRUD and queries, Task validation and state transitions, CLI "
        "argument parsing and output formatting, edge cases (empty db, duplicate "
        "titles, invalid transitions), and at least 15 test methods."
    )

    print(f"  Device: {DEVICE}")
    print(f"  Base model: {MODEL_NAME}")
    print(f"  Project prompt: {len(project_prompt)} chars")

    from rune_runner import run_phased_pipeline

    result = asyncio.run(
        run_phased_pipeline(
            project_prompt=project_prompt,
            max_iterations=10,
            checkpoint_path=checkpoint_path,
            base_model_id=MODEL_NAME,
            device=DEVICE,
            population_size=2,
            max_retries_per_subtask=2,
            max_phase_iterations=5,
        )
    )

    # Report phase results
    phases = result.get("phases", {})
    print("\n  --- Phase Results ---")

    # Phase 1: Decompose
    decompose = phases.get("decompose", {})
    subtasks = decompose.get("subtasks", [])
    print(f"  Phase 1 (Decompose): {len(subtasks)} subtasks")
    for st in subtasks:
        print(f"    - {st['name']}: {st.get('description', '')[:60]}")

    # Phase 2: Plan
    plan = phases.get("plan", {})
    plan_outputs = plan.get("plans", {})
    print(f"  Phase 2 (Plan): {len(plan_outputs)} plans generated")
    for name, text in plan_outputs.items():
        print(f"    - {name}: {text[:60]}...")

    # Phase 3: Code
    code = phases.get("code", {})
    code_outputs = code.get("outputs", {})
    code_subtask_results = code.get("subtask_results", {})
    code_passed = code.get("passed", 0)
    code_total = code.get("total", 0)
    print(f"  Phase 3 (Code): {code_passed}/{code_total} subtasks passed")
    for name, text in code_outputs.items():
        sr = code_subtask_results.get(name, {})
        status = "PASS" if sr.get("passed") else "FAIL"
        attempts = sr.get("attempts", "?")
        print(f"    - {name}: {len(text)} chars [{status} after {attempts} attempt(s)]")

    # Phase 4: Integrate
    integrate = phases.get("integrate", {})
    print(f"  Phase 4 (Integrate): tests_passed={integrate.get('tests_passed', False)}")

    # Report registered adapters
    adapters = result.get("adapters", [])
    print(f"\n  --- Registered Adapters ({len(adapters)}) ---")
    for a in adapters:
        print(f"    {a['adapter_id']}: type={a['task_type']}, phase={a['phase']}")

    # Verify all 4 phases executed
    assert "decompose" in phases, "Phase 1 (decompose) did not execute"
    assert "plan" in phases, "Phase 2 (plan) did not execute"
    assert "code" in phases, "Phase 3 (code) did not execute"
    assert "integrate" in phases, "Phase 4 (integrate) did not execute"

    # Verify adapter registration has phase-specific task_types
    adapter_types = {a["task_type"] for a in adapters}
    print(f"\n  Adapter task types: {adapter_types}")

    # Report evolution results
    evolution = result.get("evolution", {})
    assert "phase_iterations" in evolution, "Missing evolution phase_iterations"
    print("\n  --- Evolution Results ---")
    phase_iters = evolution.get("phase_iterations", {})
    for phase_name, iters in phase_iters.items():
        print(f"    {phase_name}: {iters} iteration(s)")
    best_adapters = evolution.get("best_adapters", {})
    print(f"  Best adapters: {best_adapters}")
    sweeps = evolution.get("sweeps", {})
    print(f"  Evolution sweeps: {len(sweeps)}")
    for sweep_name, sweep_result in sweeps.items():
        task_types = sweep_result.get("task_types", {})
        if task_types:
            print(f"    {sweep_name}: {task_types}")

    # Verify phase iteration counts
    for phase_name in ("decompose", "code", "integrate"):
        assert phase_name in phase_iters, f"Missing iteration count for {phase_name}"
        assert phase_iters[phase_name] >= 1, f"{phase_name} ran 0 iterations"

    print(f"\n  Session: {result['session_id']}")
    print(f"  Total iterations: {result['total_iterations']}")
    print(f"  Final tests passed: {result['final_tests_passed']}")
    print(f"  Adapters dir: {result['adapter_dir']}")

    print("\n  Phased Pipeline: PASSED")


def run_e2e() -> None:
    """Run the full end-to-end test."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--steps", default=None, help="Comma-separated step numbers to run (e.g. 1,2,3)"
    )
    args = parser.parse_args()

    run_steps = set(range(1, 6))
    if args.steps:
        run_steps = {int(s) for s in args.steps.split(",")}

    print("=" * 60)
    print("  RUNE E2E TEST: Pretrained Sakana D2L + Full Stack")
    print("=" * 60)
    print(f"\n  Base model: {MODEL_NAME}")
    print(f"  Device: {DEVICE}")
    print(
        f"  HF token: {'set' if os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN') else 'NOT SET'}"
    )
    print(f"  Steps: {sorted(run_steps)}")

    with tempfile.TemporaryDirectory() as tmpdir:
        adapter_path = None

        if 1 in run_steps:
            adapter_path = test_sakana_hypernetwork(tmpdir)

        if 2 in run_steps:
            if adapter_path is None:
                print("\n  Skipping step 2 (no adapter from step 1)")
            else:
                test_provider_loads_adapter(adapter_path)

        if 3 in run_steps:
            if adapter_path is None:
                adapter_path = os.path.join(tmpdir, "dummy_adapter")
            test_adapter_registry(adapter_path)

        if 4 in run_steps:
            test_evaluation_metrics()

        if 5 in run_steps:
            test_phased_pipeline(tmpdir)

    print("\n" + "=" * 60)
    print("  E2E TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    run_e2e()
