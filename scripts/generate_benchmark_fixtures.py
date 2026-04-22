"""Generate mini parquet fixtures for benchmark adapter tests.

Downloads 5 rows from each benchmark dataset and saves them as parquet
files in libs/evaluation/tests/fixtures/. These fixtures are checked
into the repo so adapter tests run offline (HF_DATASETS_OFFLINE=1).

Run once (with network access):
    uv run python scripts/generate_benchmark_fixtures.py

Prerequisites:
    uv sync --all-extras  (installs datasets, pandas)

Generates:
    libs/evaluation/tests/fixtures/humaneval_mini.parquet
    libs/evaluation/tests/fixtures/mbpp_mini.parquet
    libs/evaluation/tests/fixtures/apps_mini.parquet
    libs/evaluation/tests/fixtures/bigcodebench_mini.parquet
    libs/evaluation/tests/fixtures/ds1000_mini.parquet
    libs/evaluation/tests/fixtures/livecodebench_mini.parquet
    libs/evaluation/tests/fixtures/swe_bench_lite_mini.parquet
    libs/evaluation/tests/fixtures/codecontests_mini.parquet
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent / "libs" / "evaluation" / "src"))

FIXTURE_DIR = Path(__file__).parent.parent / "libs" / "evaluation" / "tests" / "fixtures"
FIXTURE_DIR.mkdir(parents=True, exist_ok=True)

# Allow HF downloads during fixture generation
os.environ["HF_DATASETS_OFFLINE"] = "0"

import datasets  # noqa: E402
import pandas as pd  # noqa: E402


def save_mini(
    dataset_name: str,
    config: str | None,
    split: str,
    n: int,
    out_name: str,
) -> None:
    """Load a dataset split, take first n rows, save as parquet fixture.

    Args:
        dataset_name: HuggingFace dataset identifier.
        config: Dataset config/subset name, or None.
        split: Dataset split to load (e.g. "test", "train").
        n: Number of rows to keep.
        out_name: Output parquet filename in FIXTURE_DIR.
    """
    out_path = FIXTURE_DIR / out_name
    if out_path.exists():
        print(f"  [skip] {out_path} already exists")
        return
    print(f"  Loading {dataset_name} ({config or 'default'}, {split}) ...")
    kwargs: dict[str, object] = {"split": split, "trust_remote_code": True}
    if config:
        ds = datasets.load_dataset(dataset_name, config, **kwargs)
    else:
        ds = datasets.load_dataset(dataset_name, **kwargs)
    df = ds.to_pandas().head(n)  # type: ignore[union-attr]
    df.to_parquet(out_path, index=False)
    print(f"  Wrote {out_path} ({len(df)} rows, {out_path.stat().st_size} bytes)")


def main() -> None:
    """Generate all benchmark fixtures."""
    print("Generating benchmark fixtures...")

    # Training-oracle benchmarks
    save_mini("openai/openai_humaneval", None, "test", 5, "humaneval_mini.parquet")
    save_mini("google-research-datasets/mbpp", "full", "train", 5, "mbpp_mini.parquet")
    save_mini("codeparrot/apps", "all", "train", 5, "apps_mini.parquet")

    try:
        save_mini("bigcode/bigcodebench", None, "v0.1.2", 5, "bigcodebench_mini.parquet")
    except Exception:
        try:
            save_mini("bigcode/bigcodebench", None, "train", 5, "bigcodebench_mini.parquet")
        except Exception as exc:
            print(f"  [warn] bigcodebench fixture failed: {exc}")

    try:
        save_mini("xlangai/DS-1000", None, "test", 5, "ds1000_mini.parquet")
    except Exception as exc:
        print(f"  [warn] ds1000 fixture failed: {exc}")

    try:
        save_mini(
            "livecodebench/code_generation_lite",
            "release_v4",
            "test",
            5,
            "livecodebench_mini.parquet",
        )
    except Exception as exc:
        print(f"  [warn] livecodebench fixture failed: {exc}")

    # Held-out benchmarks
    try:
        save_mini(
            "princeton-nlp/SWE-bench_Lite", None, "test", 5, "swe_bench_lite_mini.parquet"
        )
    except Exception as exc:
        print(f"  [warn] swe_bench_lite fixture failed: {exc}")

    try:
        save_mini("deepmind/code_contests", None, "test", 5, "codecontests_mini.parquet")
    except Exception as exc:
        print(f"  [warn] codecontests fixture failed: {exc}")

    print("\nAll fixtures generated. Check them into git:")
    print("  git add libs/evaluation/tests/fixtures/*.parquet")


if __name__ == "__main__":
    main()
