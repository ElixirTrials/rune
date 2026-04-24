"""Bin PhaseArtifacts into per-(phase, benchmark) oracle bins.

Bins are keyed:
  "<phase>_<benchmark>"  for decompose / plan / code / integrate
  "diagnose_pooled"      for diagnose (all benchmarks pooled per spec q4 decision)
"""

from __future__ import annotations

from collections import defaultdict

from corpus_producer.models import PhaseArtifact

DIAGNOSE_BIN_KEY = "diagnose_pooled"


def bin_artifacts(
    artifacts: list[PhaseArtifact],
) -> dict[str, list[PhaseArtifact]]:
    """Group artifacts into oracle training bins.

    Args:
        artifacts: Any mix of PhaseArtifacts (may span multiple runs / benchmarks).

    Returns:
        Dict mapping bin key -> list of artifacts in that bin. Diagnose
        artifacts from all benchmarks share the key ``"diagnose_pooled"``.
        Empty bins are not included.
    """
    bins: dict[str, list[PhaseArtifact]] = defaultdict(list)
    for art in artifacts:
        bins[art.bin_key()].append(art)
    return dict(bins)


def expected_bin_keys(
    benchmarks: list[str] | None = None,
) -> list[str]:
    """Return the complete list of expected bin keys for the 25-oracle target.

    Args:
        benchmarks: Override list of benchmark ids. Defaults to the 6 spec benchmarks.

    Returns:
        List of 25 bin keys: 4 phases × len(benchmarks) + 1 diagnose_pooled.
    """
    if benchmarks is None:
        benchmarks = [
            "humaneval",
            "mbpp",
            "apps",
            "bigcodebench",
            "ds_1000",
            "livecodebench",
        ]
    keys = [
        f"{phase}_{bm}"
        for phase in ("decompose", "plan", "code", "integrate")
        for bm in benchmarks
    ]
    keys.append(DIAGNOSE_BIN_KEY)
    return keys
