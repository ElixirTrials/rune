"""Adapter-compressed reasoning loop LangGraph subgraph.

When accumulated context exceeds the model's context window, this graph
splits long reasoning into many short turns. After each turn, the current
code state is compressed into LoRA adapter weights via the hypernetwork,
the adapted model is reloaded, and generation continues from a fresh
short prompt with a sliding window of recent output.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Literal, cast

from langgraph.graph import START, StateGraph
from typing_extensions import TypedDict

logger = logging.getLogger(__name__)

_EXECUTING_PHASES = frozenset({"code", "code_repair", "integrate"})


def _extract_code_skeleton(code: str) -> str:
    """Compress code into a structural skeleton (signatures, imports, decorators)."""
    if not code:
        return ""
    skeleton_lines: list[str] = []
    for line in code.splitlines():
        stripped = line.strip()
        if (
            stripped.startswith(("import ", "from "))
            or stripped.startswith(("class ", "def "))
            or stripped.startswith('"""')
            or stripped.startswith("@")
            or stripped.startswith(("raise ", "return "))
            or (
                "self." in stripped and "=" in stripped and stripped.startswith("self.")
            )
        ):
            skeleton_lines.append(line)
    return "\n".join(skeleton_lines[:80])


class ReasoningLoopState(TypedDict):
    """State for the adapter-compressed reasoning loop."""

    task_description: str
    phase: str
    adapter_ids: list[str]
    session_id: str
    generated_code: str
    stdout: str
    stderr: str
    exit_code: int
    tests_passed: bool
    test_count: int
    tests_ran: bool
    finish_reason: str | None
    outcome: str | None
    prompt_context: dict[str, Any] | None

    artifact: dict[str, Any] | None
    trajectory_state: dict[str, Any] | None

    turn_count: int
    max_turns: int
    sliding_window: str
    sliding_window_size: int
    base_sliding_window_size: int
    current_adapter_path: str | None
    current_adapter_weights: dict[str, list[float]] | None
    prior_adapter_weights: dict[str, list[float]] | None
    first_adapter_norm: float | None
    scaling_factor: float
    enable_chunk_composition: bool
    chunk_threshold: int
    phase_executes: bool
    code_scaling_boost: float
    default_merge_method: str
    turn_history: list[dict[str, Any]]

    adapter_cosine_sim: float
    adapter_norm_ratio: float
    output_repetition: float
    consecutive_high_similarity: int
    recovery_attempted: bool


async def reason_node(state: ReasoningLoopState) -> dict[str, Any]:
    """Generate with sliding window prompt + current adapter."""
    from shared.template_loader import render_prompt

    phase = state["phase"]
    turn = state["turn_count"]

    user_prompt = render_prompt(
        "reasoning_continue",
        task_description=state["task_description"],
        current_phase=phase,
        turn=turn,
    )

    from .nodes import generate_node

    gen_state: dict[str, Any] = {
        "task_description": user_prompt,
        "task_type": "project",
        "test_suite": "",
        "adapter_ids": state["adapter_ids"],
        "session_id": state["session_id"],
        "attempt_count": 0,
        "max_attempts": 1,
        "generated_code": "",
        "stdout": "",
        "stderr": "",
        "exit_code": 0,
        "tests_passed": False,
        "test_count": 0,
        "tests_ran": False,
        "trajectory": [],
        "phase": phase,
        "prompt_context": state.get("prompt_context"),
        "finish_reason": None,
        "outcome": None,
    }

    from .state import RuneState

    result = await generate_node(cast(RuneState, gen_state))

    new_code = result.get("generated_code", "")
    if phase in _EXECUTING_PHASES and state["generated_code"]:
        accumulated = state["generated_code"] + "\n" + new_code
    else:
        accumulated = new_code

    return {
        "generated_code": accumulated,
        "finish_reason": result.get("finish_reason"),
    }


async def execute_reason_node(state: ReasoningLoopState) -> dict[str, Any]:
    """Execute generated code in sandbox (code phases only)."""
    from .nodes import execute_node

    exec_state: dict[str, Any] = {
        "generated_code": state["generated_code"],
        "test_suite": "",
        "task_description": state["task_description"],
        "task_type": "project",
        "adapter_ids": state["adapter_ids"],
        "session_id": state["session_id"],
        "attempt_count": 0,
        "max_attempts": 1,
        "stdout": "",
        "stderr": "",
        "exit_code": 0,
        "tests_passed": False,
        "test_count": 0,
        "tests_ran": False,
        "trajectory": [],
        "phase": state["phase"],
        "prompt_context": None,
        "finish_reason": None,
        "outcome": None,
    }

    from .state import RuneState

    return await execute_node(cast(RuneState, exec_state))


async def reflect_reason_node(state: ReasoningLoopState) -> dict[str, Any]:
    """Lightweight reflection: record turn results."""
    return {
        "turn_history": state["turn_history"]
        + [
            {
                "turn": state["turn_count"],
                "tests_passed": state["tests_passed"],
                "exit_code": state["exit_code"],
                "finish_reason": state["finish_reason"],
                "generated_code": state["generated_code"],
                "generated_code_len": len(state["generated_code"]),
            }
        ],
    }


async def build_artifact_node(state: ReasoningLoopState) -> dict[str, Any]:
    """Construct ArtifactState or TrajectoryState from current turn."""
    phase = state["phase"]

    if phase in _EXECUTING_PHASES:
        from .artifact_state import ArtifactState, build_artifact_state

        prev_art = None
        art_dict = state.get("artifact")
        if art_dict is not None:
            prev_art = ArtifactState.from_dict(art_dict)

        artifact = build_artifact_state(
            generated_code=state["generated_code"],
            stdout=state["stdout"],
            stderr=state["stderr"],
            tests_passed=state["tests_passed"],
            turn=state["turn_count"],
            previous_artifact=prev_art,
        )
        return {"artifact": artifact.to_dict()}
    else:
        from .artifact_state import TrajectoryState

        ts = TrajectoryState(
            turn=state["turn_count"],
            output=state["generated_code"],
            feedback=state.get("stdout", ""),
            diagnosis="",
        )
        return {"trajectory_state": ts.to_dict()}


async def compress_to_adapter_node(state: ReasoningLoopState) -> dict[str, Any]:
    """Render artifact via template, encode via H(), load new adapter."""
    from shared.template_loader import render_trajectory

    phase = state["phase"]
    start = time.monotonic()

    art_dict = state.get("artifact")
    if phase in _EXECUTING_PHASES and art_dict is not None:
        from .artifact_state import ArtifactState

        art = ArtifactState.from_dict(art_dict)
        patches_text = "\n".join(
            f"Turn {p.turn}: {p.description} ({p.diff_summary})" for p in art.patches
        )

        code_skeleton = _extract_code_skeleton(art.file_contents)

        text = render_trajectory(
            "artifact_compress",
            import_block=art.import_block,
            interface_summary=art.interface_summary,
            patches=patches_text,
            test_results=art.test_results,
            stderr_summary=art.stderr_summary,
            code_skeleton=code_skeleton,
        )
    elif (ts := state.get("trajectory_state")) is not None:
        text = render_trajectory(
            "trajectory_compress",
            turn=ts.get("turn", 0),
            output=ts.get("output", ""),
            feedback=ts.get("feedback", ""),
            diagnosis=ts.get("diagnosis", ""),
        )
    else:
        return {}

    from .adapter_strategy import ChunkComposition, resolve_adapter_strategy

    word_count = len(text.split())
    strategy = resolve_adapter_strategy(
        phase=phase,
        artifact_tokens=word_count * 4,
        chunk_threshold=state["chunk_threshold"],
        base_scaling=state["scaling_factor"],
        enable_chunk_composition=state["enable_chunk_composition"],
        code_scaling_boost=state["code_scaling_boost"],
        default_merge_method=state["default_merge_method"],
    )

    import os  # noqa: PLC0415
    import tempfile  # noqa: PLC0415

    adapter_id = f"reasoning-loop-turn-{state['turn_count']}"
    output_dir = os.path.join(tempfile.gettempdir(), "rune_rl_adapters", adapter_id)

    from model_training.adapter_generator import generate_adapter  # noqa: PLC0415

    if isinstance(strategy, ChunkComposition):
        from model_training.merging import (  # noqa: PLC0415
            dare_merge,
            load_adapter_state_dict,
            ties_merge,
        )
        from safetensors.torch import save_file as _save_chunk  # noqa: PLC0415

        from .artifact_state import ArtifactState, chunk_code_state  # noqa: PLC0415

        assert state["artifact"] is not None
        art = ArtifactState.from_dict(state["artifact"])
        chunks = chunk_code_state(art, max_chunk_tokens=state["chunk_threshold"])
        if not chunks:
            return {}
        chunk_sds: list[dict[str, Any]] = []
        for i, chunk in enumerate(chunks):
            chunk_dir = os.path.join(output_dir, f"chunk_{i}")
            generate_adapter(
                text=chunk.content,
                output_dir=chunk_dir,
                scaling_factor=strategy.scaling,
            )
            chunk_sds.append(load_adapter_state_dict(chunk_dir))
        if chunk_sds:
            merge_fn = ties_merge if strategy.merge_method == "ties" else dare_merge
            merged_sd = merge_fn(chunk_sds)
            os.makedirs(output_dir, exist_ok=True)
            sd_out = os.path.join(output_dir, "adapter_model.safetensors")
            _save_chunk(merged_sd, sd_out)
    else:
        generate_adapter(
            text=text,
            output_dir=output_dir,
            scaling_factor=strategy.scaling,
        )

    from safetensors.torch import load_file  # noqa: PLC0415

    sd_path = os.path.join(output_dir, "adapter_model.safetensors")
    raw = load_file(sd_path)
    current_weights = {k: v.flatten().tolist() for k, v in raw.items()}

    first_norm = state.get("first_adapter_norm")
    if first_norm is None:
        from .adapter_health import compute_adapter_norm

        first_norm = compute_adapter_norm(current_weights)

    elapsed_ms = (time.monotonic() - start) * 1000

    _log_mlflow_metrics(state, elapsed_ms)

    return {
        "current_adapter_path": output_dir,
        "adapter_ids": [adapter_id],
        "prior_adapter_weights": state.get("current_adapter_weights"),
        "current_adapter_weights": current_weights,
        "first_adapter_norm": first_norm,
    }


def _log_mlflow_metrics(state: ReasoningLoopState, compress_latency_ms: float) -> None:
    """Log per-turn metrics to MLflow."""
    try:
        import mlflow
    except ImportError:
        return
    if mlflow.active_run() is None:
        return

    turn = state["turn_count"]
    mlflow.log_metrics(
        {
            "reasoning_loop/turn": float(turn),
            "reasoning_loop/adapter_cosine_sim": state.get("adapter_cosine_sim", 0.0),
            "reasoning_loop/adapter_norm_ratio": state.get("adapter_norm_ratio", 1.0),
            "reasoning_loop/output_repetition": state.get("output_repetition", 0.0),
            "reasoning_loop/hypernetwork_latency_ms": compress_latency_ms,
            "reasoning_loop/sliding_window_tokens": float(state["sliding_window_size"]),
            "reasoning_loop/scaling_factor": state["scaling_factor"],
        },
        step=turn,
    )


async def check_health_node(state: ReasoningLoopState) -> dict[str, Any]:
    """Compute adapter health signals."""
    from .adapter_health import (
        compute_cosine_similarity,
        compute_norm_ratio,
        compute_output_repetition,
    )

    current_weights = state.get("current_adapter_weights")
    if current_weights is None:
        return {
            "adapter_cosine_sim": 0.0,
            "adapter_norm_ratio": 1.0,
            "output_repetition": 0.0,
            "consecutive_high_similarity": 0,
            "turn_count": state["turn_count"] + 1,
        }

    prior_weights = state.get("prior_adapter_weights")
    cosine_sim = compute_cosine_similarity(current_weights, prior_weights)
    norm_ratio = compute_norm_ratio(current_weights, state.get("first_adapter_norm"))

    prev_output = state.get("sliding_window", "")
    output_rep = compute_output_repetition(state["generated_code"], prev_output)

    words = state["generated_code"].split()
    window_words = state["sliding_window_size"] // 4
    new_window = " ".join(words[-window_words:]) if words else ""

    consec = state["consecutive_high_similarity"]
    if cosine_sim > 0.95:
        consec += 1
    else:
        consec = 0

    return {
        "adapter_cosine_sim": cosine_sim,
        "adapter_norm_ratio": norm_ratio,
        "output_repetition": output_rep,
        "consecutive_high_similarity": consec,
        "turn_count": state["turn_count"] + 1,
        "sliding_window": new_window,
    }


def should_continue(
    state: ReasoningLoopState,
) -> Literal["reason", "recover", "__end__"]:
    """Check termination conditions."""
    if state.get("finish_reason") == "stop":
        return "__end__"

    if state["turn_count"] >= state["max_turns"]:
        return "__end__"

    from .adapter_health import check_health

    health = check_health(
        cosine_sim=state.get("adapter_cosine_sim", 0.0),
        norm_ratio=state.get("adapter_norm_ratio", 1.0),
        output_repetition=state.get("output_repetition", 0.0),
        consecutive_high_similarity=state.get("consecutive_high_similarity", 0),
    )

    if health.is_collapsed:
        if health.collapse_reason and (
            "norm_collapse" in health.collapse_reason
            or "norm_explosion" in health.collapse_reason
        ):
            return "__end__"
        if not state.get("recovery_attempted", False):
            return "recover"
        return "__end__"

    return "reason"


async def recover_node(state: ReasoningLoopState) -> dict[str, Any]:
    """Expand sliding window and reset scaling for recovery."""
    max_window = state.get("base_sliding_window_size", 1024) * 4
    new_window_size = min(state["sliding_window_size"] * 2, max_window)
    return {
        "sliding_window_size": new_window_size,
        "scaling_factor": state["scaling_factor"],
        "recovery_attempted": True,
    }


def route_after_reason(
    state: ReasoningLoopState,
) -> Literal["execute_reason", "reflect"]:
    """Route based on whether this phase executes code."""
    if state["phase_executes"]:
        return "execute_reason"
    return "reflect"


def select_best_output(turn_history: list[dict[str, Any]]) -> int | None:
    """Select the best turn index based on test results."""
    best: int | None = None
    for entry in turn_history:
        if entry.get("tests_passed", False):
            best = entry["turn"]
    return best


async def finalize_node(state: ReasoningLoopState) -> dict[str, Any]:
    """Restore best-passing-turn output if last turn did not pass."""
    if state.get("tests_passed"):
        return {}
    best_turn = select_best_output(state["turn_history"])
    if best_turn is None:
        return {}
    for entry in state["turn_history"]:
        if entry["turn"] == best_turn and entry.get("generated_code"):
            return {"generated_code": entry["generated_code"]}
    return {}


def create_reasoning_loop_graph() -> Any:
    """Create and compile the reasoning loop subgraph."""
    workflow = StateGraph(ReasoningLoopState)

    workflow.add_node("reason", reason_node)
    workflow.add_node("execute_reason", execute_reason_node)
    workflow.add_node("reflect", reflect_reason_node)
    workflow.add_node("build_artifact", build_artifact_node)
    workflow.add_node("compress_to_adapter", compress_to_adapter_node)
    workflow.add_node("check_health", check_health_node)
    workflow.add_node("recover", recover_node)
    workflow.add_node("finalize", finalize_node)

    workflow.add_edge(START, "reason")
    workflow.add_conditional_edges("reason", route_after_reason)
    workflow.add_edge("execute_reason", "reflect")
    workflow.add_edge("reflect", "build_artifact")
    workflow.add_edge("build_artifact", "compress_to_adapter")
    workflow.add_edge("compress_to_adapter", "check_health")
    workflow.add_conditional_edges(
        "check_health",
        should_continue,
        {"reason": "reason", "recover": "recover", "__end__": "finalize"},
    )
    workflow.add_edge("recover", "reason")

    return workflow.compile()
