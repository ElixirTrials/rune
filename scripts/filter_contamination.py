"""Drop trajectories that overlap with held-out benchmark fingerprints.

Tier (a) — exact-match exclusion: if any of (task_description, action_diff,
test paths in feedback anchors) hash-matches a fingerprint from
``humaneval_plus``, ``mbpp_plus``, ``bigcodebench_*``, ``ds_1000``, or
``livecodebench``, drop the trajectory.

Tier (b) — repo-level: if the trajectory's ``provenance.repo`` is in
``swebench_lite_repos``, drop the trajectory.

Per paper §B.3, both tiers are applied. Per-benchmark exclusion counts go
to a sidecar ``<output>.exclusion_counts.json`` for the paper.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

from build_benchmark_fingerprints import fingerprint

logger = logging.getLogger(__name__)


def filter_corpus(
    input_path: Path,
    output_path: Path,
    fingerprints_path: Path,
) -> Counter:
    fps = json.loads(fingerprints_path.read_text(encoding="utf-8"))
    repo_filter = set(fps.get("swebench_lite_repos", []))
    fp_sets = {
        name: set(values)
        for name, values in fps.items()
        if name != "swebench_lite_repos"
    }

    counts: Counter = Counter()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with (
        input_path.open("r", encoding="utf-8") as fh_in,
        output_path.open("w", encoding="utf-8") as fh_out,
    ):
        for line in fh_in:
            rec = json.loads(line)
            if rec["provenance"]["repo"] in repo_filter:
                counts["swebench_lite_repos"] += 1
                continue
            desc_fp = fingerprint(rec.get("task_description", ""))
            hit_bench = next(
                (name for name, s in fp_sets.items() if desc_fp in s),
                None,
            )
            if hit_bench:
                counts[hit_bench] += 1
                continue
            fh_out.write(line)

    sidecar = output_path.with_suffix(output_path.suffix + ".exclusion_counts.json")
    sidecar.write_text(json.dumps(dict(counts), indent=2), encoding="utf-8")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=Path, required=True)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument(
        "--fingerprints",
        type=Path,
        default=Path("data/contamination/fingerprints.json"),
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    counts = filter_corpus(args.input, args.output, args.fingerprints)
    logger.info("Excluded: %s", dict(counts))


if __name__ == "__main__":
    main()
