#!/usr/bin/env python3
"""End-to-end smoke test for Rune v5.0 milestone.

Exercises the full pipeline WITHOUT GPU or network:
  1. Adapter Registry — CRUD lifecycle
  2. Inference — provider factory, ABC contracts
  3. Agent Loop — trajectory persistence, node functions
  4. Model Training — QLoRA config, hypernetwork forward pass
  5. Evaluation — pass@k, kill-switch gate, HumanEval subset
  6. Training Service — HTTP endpoints via TestClient

Run: uv run python scripts/e2e_smoke.py
"""

# ruff: noqa: E402
# mypy: ignore-errors
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

passed = 0
failed = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global passed, failed
    if condition:
        passed += 1
        print(f"  [PASS] {name}")
    else:
        failed += 1
        print(f"  [FAIL] {name} — {detail}")


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


# =========================================================================
#  1. ADAPTER REGISTRY
# =========================================================================
section("1. Adapter Registry — CRUD lifecycle")

from adapter_registry import AdapterRecord, AdapterRegistry
from adapter_registry.exceptions import (
    AdapterAlreadyExistsError,
    AdapterNotFoundError,
)
from sqlalchemy.engine import Engine
from sqlmodel import create_engine

engine: Engine = create_engine("sqlite:///:memory:")
registry = AdapterRegistry(engine=engine)

# Store
record = AdapterRecord(
    id="adapter-001",
    version=1,
    task_type="code-gen",
    base_model_id="Qwen/Qwen2.5-Coder-7B-Instruct",
    rank=8,
    created_at="2026-03-06T00:00:00Z",
    file_path="/tmp/adapters/adapter-001/adapter_model.safetensors",
    file_hash="abc123",
    file_size_bytes=1024,
    source="qlora",
    session_id="session-001",
    is_archived=False,
)
registry.store(record)
check("store adapter", True)

# Retrieve
retrieved = registry.retrieve_by_id("adapter-001")
check("retrieve by id", retrieved.id == "adapter-001")
check("retrieve fields intact", retrieved.task_type == "code-gen")

# Duplicate raises
try:
    registry.store(record)
    check("duplicate raises", False, "should have raised")
except AdapterAlreadyExistsError:
    check("duplicate raises AdapterAlreadyExistsError", True)

# Query by task type
results = registry.query_by_task_type("code-gen")
check("query by task_type", len(results) == 1)
check("query returns correct record", results[0].id == "adapter-001")

# List all (non-archived)
all_records = registry.list_all()
check("list_all returns non-archived", len(all_records) == 1)

# Not found raises
try:
    registry.retrieve_by_id("nonexistent")
    check("not found raises", False, "should have raised")
except AdapterNotFoundError:
    check("not found raises AdapterNotFoundError", True)


# =========================================================================
#  2. INFERENCE PROVIDER ABSTRACTION
# =========================================================================
section("2. Inference — provider factory, ABC contracts")

from inference import (
    GenerationResult,
    InferenceProvider,
    OllamaProvider,
    VLLMProvider,
    get_provider,
)

# Factory creates correct types
vllm = get_provider("vllm", base_url="http://localhost:8100/v1")
check("get_provider('vllm') returns VLLMProvider", isinstance(vllm, VLLMProvider))

ollama = get_provider("ollama", base_url="http://localhost:11434/v1")
check(
    "get_provider('ollama') returns OllamaProvider",
    isinstance(ollama, OllamaProvider),
)

# Caching works
vllm2 = get_provider("vllm", base_url="http://localhost:8100/v1")
check("provider caching (same instance)", vllm is vllm2)

vllm3 = get_provider("vllm", base_url="http://other:8100/v1")
check("different URL gives different instance", vllm is not vllm3)

# ABC enforcement
check("VLLMProvider is InferenceProvider", isinstance(vllm, InferenceProvider))
check("OllamaProvider is InferenceProvider", isinstance(ollama, InferenceProvider))

# GenerationResult construction
result = GenerationResult(
    text="def foo(): pass",
    model="qwen-7b",
    adapter_id="adapter-001",
    token_count=10,
    finish_reason="stop",
)
check("GenerationResult fields", result.text == "def foo(): pass")
check("GenerationResult adapter_id", result.adapter_id == "adapter-001")

# Invalid provider
try:
    get_provider("invalid_provider")
    check("invalid provider raises", False)
except ValueError:
    check("invalid provider raises ValueError", True)


# =========================================================================
#  3. AGENT LOOP — trajectory + nodes
# =========================================================================
section("3. Agent Loop — trajectory persistence, node functions")

with tempfile.TemporaryDirectory() as tmpdir:
    os.environ["RUNE_TRAJECTORY_DIR"] = tmpdir

    from model_training.trajectory import (
        format_for_sft,
        load_trajectory,
        record_trajectory,
    )

    # Record a trajectory
    steps = [
        {
            "code": "def add(a, b): return a + b",
            "stdout": "",
            "stderr": "",
            "exit_code": 0,
            "tests_passed": True,
        }
    ]
    session_id = "e2e-test-session"
    meta = record_trajectory(
        session_id=session_id,
        steps=steps,
        outcome="success",
        task_description="Add two numbers",
        task_type="code-gen",
        adapter_ids=["adapter-001"],
    )
    check("record_trajectory returns metadata", "file_path" in meta)
    check(
        "trajectory file exists",
        Path(meta["file_path"]).exists(),
    )

    # Load it back
    loaded = load_trajectory(session_id)
    check("load_trajectory round-trip", loaded["outcome"] == "success")
    check("trajectory has steps", len(loaded["steps"]) == 1)

    # Format for SFT
    sft = format_for_sft(loaded)
    check("format_for_sft returns messages", len(sft) == 3)
    check("SFT has system role", sft[0]["role"] == "system")
    check("SFT has user role", sft[1]["role"] == "user")
    check("SFT has assistant role", sft[2]["role"] == "assistant")

    del os.environ["RUNE_TRAJECTORY_DIR"]

# Node functions (mock inference provider)
import asyncio

from rune_agent.nodes import (
    execute_node,
    generate_node,
    reflect_node,
)

mock_provider = AsyncMock()
mock_provider.generate.return_value = GenerationResult(
    text='```python\nprint("hello")\n```',
    model="test",
    adapter_id=None,
    token_count=5,
    finish_reason="stop",
)

with patch("rune_agent.nodes.get_provider", return_value=mock_provider):
    state = {
        "task_description": "Print hello",
        "test_suite": "",
        "generated_code": "",
        "stdout": "",
        "stderr": "",
        "exit_code": -1,
        "tests_passed": False,
        "attempt_count": 0,
        "max_attempts": 3,
        "trajectory": [],
        "session_id": "e2e-node-test",
        "adapter_ids": [],
    }
    gen_result = asyncio.run(generate_node(state))
    check("generate_node returns generated_code", "generated_code" in gen_result)
    check(
        "generate_node extracts code",
        'print("hello")' in gen_result["generated_code"],
    )

# Execute node — actually runs subprocess
state2 = {
    **state,
    "generated_code": 'print("hello world")',
    "test_suite": "",
}
exec_result = asyncio.run(execute_node(state2))
check("execute_node returns stdout", "hello world" in exec_result["stdout"])
check("execute_node exit_code 0", exec_result["exit_code"] == 0)
check("execute_node tests_passed", exec_result["tests_passed"] is True)

# Reflect node
state3 = {**state2, **exec_result, "attempt_count": 0, "trajectory": []}
reflect_result = asyncio.run(reflect_node(state3))
check("reflect_node increments attempt", reflect_result["attempt_count"] == 1)
check("reflect_node appends to trajectory", len(reflect_result["trajectory"]) == 1)


# =========================================================================
#  4. MODEL TRAINING — config, hypernetwork
# =========================================================================
section("4. Model Training — QLoRA config, hypernetwork forward pass")

from model_training.config import get_training_config, validate_config

config = get_training_config(task_type="code-gen")
check("get_training_config returns dict", isinstance(config, dict))
check("config has rank", "rank" in config)
check("config alpha is 2x rank", config["alpha"] == config["rank"] * 2)

is_valid = validate_config(config)
check("validate_config on defaults passes", is_valid is True)

bad_config = {**config, "rank": 0}
try:
    validate_config(bad_config)
    check("validate_config catches bad rank", False, "should have raised")
except ValueError:
    check("validate_config catches bad rank", True)

# Hypernetwork (requires torch)
try:
    import torch
    from model_training.hypernetwork import (
        DocToLoraHypernetwork,
        save_hypernetwork_adapter,
    )

    # Small model for CPU test
    hyper = DocToLoraHypernetwork(input_dim=1000, hidden_dim=32, num_layers=1, rank=4)
    check("DocToLoraHypernetwork instantiates", True)

    token_ids = torch.randint(0, 1000, (1, 16))
    weights = hyper(token_ids)
    check("hypernetwork forward returns dict", isinstance(weights, dict))

    # Check PEFT key format
    expected_key = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
    check("output has PEFT-compatible keys", expected_key in weights)

    lora_a = weights[expected_key]
    check("lora_A shape is (rank, hidden_dim)", lora_a.shape == (4, 32))

    # Save adapter
    with tempfile.TemporaryDirectory() as tmpdir:
        save_hypernetwork_adapter(
            weights=weights,
            output_dir=tmpdir,
            base_model_id="test-model",
            rank=4,
        )
        check(
            "adapter_model.safetensors written",
            (Path(tmpdir) / "adapter_model.safetensors").exists(),
        )
        check(
            "adapter_config.json written",
            (Path(tmpdir) / "adapter_config.json").exists(),
        )

        cfg = json.loads((Path(tmpdir) / "adapter_config.json").read_text())
        check("adapter config peft_type is LORA", cfg["peft_type"] == "LORA")
        check("adapter config modules_to_save is None", cfg["modules_to_save"] is None)

except ImportError:
    print("  [SKIP] torch not installed — skipping hypernetwork tests")


# =========================================================================
#  5. EVALUATION — pass@k, kill-switch, HumanEval
# =========================================================================
section("5. Evaluation — pass@k, kill-switch gate, HumanEval subset")

from evaluation import calculate_pass_at_k, run_kill_switch_gate

# pass@k
score = calculate_pass_at_k(n_samples=10, n_correct=10, k=1)
check("pass@k perfect score = 1.0", score == 1.0)

score = calculate_pass_at_k(n_samples=10, n_correct=0, k=1)
check("pass@k zero score = 0.0", score == 0.0)

score = calculate_pass_at_k(n_samples=10, n_correct=5, k=1)
check("pass@k partial score in (0, 1)", 0.0 < score < 1.0)

# Kill-switch gate
result = run_kill_switch_gate(adapter_pass1=0.60, baseline_pass1=0.50, threshold=0.05)
check("kill-switch PASS when adapter > baseline*1.05", result["verdict"] == "PASS")

result = run_kill_switch_gate(adapter_pass1=0.50, baseline_pass1=0.50, threshold=0.05)
check("kill-switch FAIL when adapter == baseline", result["verdict"] == "FAIL")

# HumanEval subset — run with canonical solutions
from evaluation.metrics import run_humaneval_subset

data_path = (
    Path(__file__).resolve().parent.parent
    / "libs"
    / "evaluation"
    / "src"
    / "evaluation"
    / "data"
    / "humaneval_subset.json"
)
with data_path.open() as f:
    tasks = json.load(f)

check("humaneval_subset.json has 20 tasks", len(tasks) == 20)

# Run with canonical solutions — should get high pass rate
completions = {t["task_id"]: t["canonical_solution"] for t in tasks[:5]}
result = run_humaneval_subset(
    adapter_id="test-adapter", completions=completions, subset_size=20
)
check("run_humaneval_subset returns pass_rate", "pass_rate" in result)
check(
    "canonical solutions pass rate > 0.5",
    result["pass_rate"] > 0.5,
)


# =========================================================================
#  6. TRAINING SERVICE — HTTP endpoints
# =========================================================================
section("6. Training Service — HTTP endpoints via TestClient")

from unittest.mock import patch as mock_patch

from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel
from training_svc.jobs import JOB_STORE

svc_engine = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
SQLModel.metadata.create_all(bind=svc_engine)

with mock_patch("training_svc.storage.engine", svc_engine):
    from training_svc.main import app

    with TestClient(app) as client:
        # POST /train/lora
        JOB_STORE.clear()
        with mock_patch("training_svc.routers.training._run_training_job"):
            resp = client.post(
                "/train/lora",
                json={
                    "session_id": "e2e-session",
                    "task_type": "code-gen",
                    "rank": 8,
                    "epochs": 3,
                },
            )
        check("POST /train/lora returns 200", resp.status_code == 200)
        data = resp.json()
        check("response has job_id", "job_id" in data)
        check("response status is queued", data["status"] == "queued")

        # GET /jobs/{job_id}
        job_id = data["job_id"]
        resp2 = client.get(f"/jobs/{job_id}")
        check("GET /jobs/{job_id} returns 200", resp2.status_code == 200)
        check("job status is queued", resp2.json()["status"] == "queued")

        # GET /jobs/nonexistent
        resp3 = client.get("/jobs/nonexistent")
        check("GET /jobs/nonexistent returns 404", resp3.status_code == 404)

        # POST /train/hypernetwork
        JOB_STORE.clear()
        with mock_patch("training_svc.routers.training._run_hypernetwork_job"):
            resp4 = client.post(
                "/train/hypernetwork",
                json={"task_type": "gen", "trajectory_ids": ["t-1"]},
            )
        check("POST /train/hypernetwork returns 200", resp4.status_code == 200)
        check(
            "hypernetwork job status queued",
            resp4.json()["status"] == "queued",
        )

        # Poll hypernetwork job
        hn_job_id = resp4.json()["job_id"]
        resp5 = client.get(f"/jobs/{hn_job_id}")
        check("hypernetwork job pollable", resp5.status_code == 200)

        # Validation errors
        resp6 = client.post("/train/lora", json={"task_type": "code-gen"})
        check(
            "POST /train/lora without session_id returns 422",
            resp6.status_code == 422,
        )

        resp7 = client.post("/train/hypernetwork", json={"task_type": "gen"})
        check(
            "POST /train/hypernetwork without trajectory_ids returns 422",
            resp7.status_code == 422,
        )

        # Health check
        resp8 = client.get("/health")
        check("GET /health returns 200", resp8.status_code == 200)
        check(
            "health shows training-svc",
            resp8.json()["service"] == "training-svc",
        )


# =========================================================================
#  SUMMARY
# =========================================================================
print(f"\n{'=' * 60}")
print(f"  RESULTS: {passed} passed, {failed} failed")
print(f"{'=' * 60}")

if failed > 0:
    print("\nSome checks failed!")
    sys.exit(1)
else:
    print("\nAll checks passed. v5.0 milestone is solid.")
    sys.exit(0)
