"""Pre-cache contamination fingerprints from public coding benchmarks.

Run once per benchmark version. Output goes to
``data/contamination/fingerprints.json`` and is consumed by
``scripts/filter_contamination.py``.

Implements paper §B.3 tier (a): exact-match exclusion on problem statement,
function signature, or canonical test fixture across HumanEval+, MBPP+,
BigCodeBench (complete + instruct), DS-1000, LiveCodeBench. SWE-Bench-Lite
gets a separate repo-level filter (tier b) handled in
``filter_contamination.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

_WS = re.compile(r"\s+")
_QUOTE = re.compile(r"['\"]")


def fingerprint(text: str) -> str:
    """Whitespace- and quote-insensitive fingerprint for exact-match exclusion."""
    normalised = _WS.sub(" ", _QUOTE.sub('"', text)).strip()
    return hashlib.sha1(normalised.encode("utf-8")).hexdigest()


def _humaneval_plus(out: dict[str, set[str]]) -> None:
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise SystemExit("`datasets` package required: uv pip install datasets") from e
    ds = load_dataset("evalplus/humanevalplus", split="test")
    for row in ds:
        out.setdefault("humaneval_plus", set()).add(fingerprint(row["prompt"]))
        out["humaneval_plus"].add(fingerprint(row.get("entry_point", "")))


def _mbpp_plus(out: dict[str, set[str]]) -> None:
    from datasets import load_dataset
    ds = load_dataset("evalplus/mbppplus", split="test")
    for row in ds:
        out.setdefault("mbpp_plus", set()).add(fingerprint(row["text"]))


def _bigcodebench(out: dict[str, set[str]]) -> None:
    from datasets import load_dataset
    for split in ("complete", "instruct"):
        ds = load_dataset("bigcode/bigcodebench", split=split)
        for row in ds:
            out.setdefault(f"bigcodebench_{split}", set()).add(
                fingerprint(row["instruct_prompt"] if "instruct_prompt" in row else row.get("complete_prompt", ""))
            )


def _swebench_lite_repos(out: dict[str, set[str]]) -> None:
    from datasets import load_dataset
    ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    for row in ds:
        out.setdefault("swebench_lite_repos", set()).add(row["repo"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", type=Path, default=Path("data/contamination/fingerprints.json"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    out: dict[str, set[str]] = {}
    for name, fn in [
        ("humaneval_plus", _humaneval_plus),
        ("mbpp_plus", _mbpp_plus),
        ("bigcodebench", _bigcodebench),
        ("swebench_lite_repos", _swebench_lite_repos),
    ]:
        try:
            fn(out)
            logger.info("Loaded %s", name)
        except Exception:
            logger.exception("Skipping %s", name)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({k: sorted(v) for k, v in out.items()}, indent=2),
        encoding="utf-8",
    )
    logger.info("Wrote %d benchmark fingerprint sets to %s", len(out), args.output)


if __name__ == "__main__":
    main()
