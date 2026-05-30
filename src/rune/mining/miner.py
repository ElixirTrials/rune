"""Mining pipeline: session scanning, trajectory extraction, JSONL shard writing."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

from rune.mining.session_log import SESSION_SCHEMA_VERSION  # single source of truth

logger = logging.getLogger(__name__)


def scan_sessions(sessions_dir: Path) -> list[Path]:
    """Return subdirectories of sessions_dir that contain session.jsonl."""
    return sorted(
        p.parent
        for p in sessions_dir.rglob("session.jsonl")
        if p.parent != sessions_dir
    )


def load_session(session_dir: Path) -> tuple[list[dict], dict]:  # type: ignore[type-arg]
    """Read session.jsonl and metadata.json from session_dir."""
    steps_path = session_dir / "session.jsonl"
    meta_path = session_dir / "metadata.json"

    steps: list[dict] = []  # type: ignore[type-arg]
    for raw_line in steps_path.read_text().splitlines():
        stripped = raw_line.strip()
        if stripped:
            steps.append(json.loads(stripped))

    metadata: dict = json.loads(meta_path.read_text()) if meta_path.exists() else {}  # type: ignore[type-arg]
    return steps, metadata


def _select_training_steps(
    steps: list[dict],  # type: ignore[type-arg]
    pass_at_1: bool | None,
) -> list[dict]:  # type: ignore[type-arg]
    """STaR self-distillation selection (ports v1-final success_filter):
    pass (or unknown verdict, e.g. smoke) -> all steps; fail -> only diagnose
    steps for subtasks that recovered (have a passing feedback somewhere)."""
    if pass_at_1 is None or pass_at_1:
        return steps
    recovered = {
        s.get("target")
        for s in steps
        if s.get("target") and (s.get("feedback") or {}).get("exit_code") == 0
    }
    return [
        s
        for s in steps
        if s.get("action") == "diagnose" and s.get("target") in recovered
    ]


def extract_trajectories(
    steps: list[dict],  # type: ignore[type-arg]
    metadata: dict,  # type: ignore[type-arg]
) -> list[dict]:  # type: ignore[type-arg]
    """One SFT record per kept step (no joining); STaR-filtered by run pass@1."""
    version = metadata.get("schema_version")
    if version != SESSION_SCHEMA_VERSION:
        raise ValueError(
            f"session schema_version {version!r} != expected "
            f"{SESSION_SCHEMA_VERSION}; re-mine from current sessions "
            "(old corpora are intentionally not bridged)."
        )
    benchmark = metadata.get("benchmark", "unknown")
    problem_id = metadata.get("problem_id", "unknown")
    pass_at_1 = metadata.get("pass_at_1")

    records: list[dict] = []  # type: ignore[type-arg]
    dropped = 0
    for step in _select_training_steps(steps, pass_at_1):
        completion = step.get("output", "")
        if not completion:
            dropped += 1
            continue
        records.append(
            {
                "task_id": f"{benchmark}/{problem_id}/{step.get('step')}",
                "trajectory": step.get("trajectory", ""),
                "prompt": step.get("prompt", ""),
                "completion": completion,
                "metadata": {
                    "phase": step.get("action", "unknown"),
                    "target": step.get("target"),
                    "step": step.get("step"),
                    "benchmark": benchmark,
                    "problem_id": problem_id,
                    "pass_at_1": pass_at_1,
                    "schema_version": SESSION_SCHEMA_VERSION,
                },
            }
        )
    if dropped:
        logger.debug(
            "extract_trajectories: dropped %d step(s) with empty completion (%s/%s)",
            dropped,
            benchmark,
            problem_id,
        )
    return records


def mine_corpus(sessions_dir: Path, output_dir: Path) -> dict[str, int]:
    """Mine all sessions into JSONL shards. Return {bin_key: record_count}."""
    output_dir.mkdir(parents=True, exist_ok=True)

    bins: dict[str, list[dict]] = defaultdict(list)  # type: ignore[type-arg]

    for session_dir in scan_sessions(sessions_dir):
        steps, metadata = load_session(session_dir)
        benchmark = metadata.get("benchmark", "unknown")
        records = extract_trajectories(steps, metadata)
        for record in records:
            action = record["metadata"]["phase"]
            bin_key = f"{action}_{benchmark}"
            bins[bin_key].append(record)

    counts: dict[str, int] = {}
    for bin_key, records in bins.items():
        shard_path = output_dir / f"{bin_key}.jsonl"
        with shard_path.open("w") as fh:
            for record in records:
                fh.write(json.dumps(record) + "\n")
        counts[bin_key] = len(records)

    return counts
