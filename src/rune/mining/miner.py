"""Mining pipeline: session scanning, trajectory extraction, JSONL shard writing."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


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


def _render_trajectory(steps: list[dict], action: str) -> str:  # type: ignore[type-arg]
    """Render trajectory text for steps matching action."""
    parts: list[str] = []
    for step in steps:
        if step.get("action") != action:
            continue
        inp = step.get("input", "")
        out = step.get("output", "")
        parts.append(f"Input: {inp}\nOutput: {out}")
        fb = step.get("feedback")
        if fb:
            parts.append(f"Feedback: {json.dumps(fb)}")
    return "\n---\n".join(parts)


def extract_trajectories(
    steps: list[dict],  # type: ignore[type-arg]
    metadata: dict,  # type: ignore[type-arg]
) -> list[dict]:  # type: ignore[type-arg]
    """Produce one trajectory record per unique action in steps."""
    benchmark = metadata.get("benchmark", "unknown")
    problem_id = metadata.get("problem_id", "unknown")
    task_id = f"{benchmark}/{problem_id}"

    seen_actions: set[str] = set()
    records: list[dict] = []  # type: ignore[type-arg]

    for step in steps:
        action = step.get("action", "unknown")
        if action in seen_actions:
            continue
        seen_actions.add(action)

        trajectory_text = _render_trajectory(steps, action)
        records.append(
            {
                "task_id": task_id,
                "trajectory": trajectory_text,
                "metadata": {
                    "phase": action,
                    "benchmark": benchmark,
                    "problem_id": problem_id,
                },
            }
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
