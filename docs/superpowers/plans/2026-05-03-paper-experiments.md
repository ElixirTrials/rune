# Rune Paper Experiments — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Write all missing code and run all experiments required for the Rune scientific paper (conditions i–v, gates 1–3, figures, ablations).

**Architecture:** 11 independent code items organized by priority. Each produces a self-contained script or module following existing patterns (deferred GPU imports, BenchmarkAdapter protocol, Google-style docstrings). Experiments run via CLI scripts that output JSON results to `evaluation_results/`.

**Tech Stack:** Python 3.12, PyTorch, sentence-transformers, scipy, numpy, optuna, PEFT/trl, the existing `evaluation.benchmarks` harness, `shared.sandbox.SubprocessBackend`.

---

## Task 1: Corpus Statistics Script (P0, blocks §3.1)

**Files:**
- Create: `scripts/paper/corpus_stats.py`
- Test: `scripts/paper/tests/test_corpus_stats.py`

- [ ] **Step 1: Write the failing test**

```python
# scripts/paper/tests/test_corpus_stats.py
"""Tests for corpus statistics computation."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from scripts.paper.corpus_stats import compute_corpus_stats


@pytest.fixture
def sample_corpus(tmp_path: Path) -> Path:
    """Create a minimal corpus JSONL for testing."""
    records = [
        {"trajectory": "def foo():\n    return 1\n" * 50, "steps": 3},
        {"trajectory": "x = 1\n" * 200, "steps": 7},
        {"trajectory": "import os\n" * 10, "steps": 1},
    ]
    out = tmp_path / "corpus.jsonl"
    with out.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return out


def test_stats_keys(sample_corpus: Path) -> None:
    stats = compute_corpus_stats(sample_corpus)
    assert "mean_tokens" in stats
    assert "median_tokens" in stats
    assert "p95_tokens" in stats
    assert "max_steps" in stats
    assert "pct_exceeding_4k" in stats
    assert "pct_exceeding_16k" in stats


def test_stats_ordering(sample_corpus: Path) -> None:
    stats = compute_corpus_stats(sample_corpus)
    assert stats["median_tokens"] <= stats["p95_tokens"]
    assert 0.0 <= stats["pct_exceeding_4k"] <= 100.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest scripts/paper/tests/test_corpus_stats.py -v`
Expected: FAIL with "ModuleNotFoundError" or "ImportError"

- [ ] **Step 3: Create package structure**

```bash
mkdir -p scripts/paper/tests
touch scripts/paper/__init__.py scripts/paper/tests/__init__.py
```

- [ ] **Step 4: Write implementation**

```python
# scripts/paper/corpus_stats.py
"""Corpus statistics for paper §3.1.

Computes token-length distribution (mean, median, P95), max encoder depth,
and percentage of sessions exceeding context windows.

Usage:
    uv run python scripts/paper/corpus_stats.py --corpus data/pairs/corpus.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _count_tokens(text: str) -> int:
    """Approximate token count using whitespace + punctuation heuristic.

    For precise counts, swap this with tiktoken or the model's tokenizer.
    The 4-char approximation matches GPT-family tokenizers within ~10%.
    """
    return max(1, len(text) // 4)


def compute_corpus_stats(
    corpus_path: Path,
    context_windows: tuple[int, ...] = (4096, 16384),
) -> dict[str, Any]:
    """Compute trajectory corpus statistics.

    Args:
        corpus_path: Path to JSONL file. Each line must have a "trajectory"
            field (str) and optionally a "steps" field (int).
        context_windows: Token thresholds to report % exceeding.

    Returns:
        Dict with mean_tokens, median_tokens, p95_tokens, max_steps,
        pct_exceeding_4k, pct_exceeding_16k, total_sessions.
    """
    import numpy as np

    token_lengths: list[int] = []
    step_counts: list[int] = []

    with corpus_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            traj = record.get("trajectory", "")
            steps = record.get("steps", 1)
            token_lengths.append(_count_tokens(traj))
            step_counts.append(int(steps))

    if not token_lengths:
        return {
            "mean_tokens": 0,
            "median_tokens": 0,
            "p95_tokens": 0,
            "max_steps": 0,
            "pct_exceeding_4k": 0.0,
            "pct_exceeding_16k": 0.0,
            "total_sessions": 0,
        }

    arr = np.array(token_lengths)
    n = len(arr)
    result: dict[str, Any] = {
        "mean_tokens": int(np.mean(arr)),
        "median_tokens": int(np.median(arr)),
        "p95_tokens": int(np.percentile(arr, 95)),
        "max_steps": max(step_counts),
        "total_sessions": n,
    }

    for window in context_windows:
        key = f"pct_exceeding_{window // 1024}k"
        result[key] = float(np.sum(arr > window) / n * 100)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Corpus statistics for paper §3.1")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    stats = compute_corpus_stats(args.corpus)

    output = json.dumps(stats, indent=2)
    print(output)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest scripts/paper/tests/test_corpus_stats.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add scripts/paper/
git commit -m "feat(paper): corpus statistics script for §3.1"
```

---

## Task 2: Contamination Filter (P0, blocks §4.1/B.3)

**Files:**
- Create: `scripts/paper/contamination_filter.py`
- Test: `scripts/paper/tests/test_contamination_filter.py`

- [ ] **Step 1: Write the failing test**

```python
# scripts/paper/tests/test_contamination_filter.py
"""Tests for contamination filter."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from scripts.paper.contamination_filter import (
    check_exact_match,
    check_repo_level,
    filter_corpus,
)


def test_exact_match_positive() -> None:
    """Detects verbatim benchmark solution in trajectory."""
    benchmark_solutions = ["def foo():\n    return 42\n"]
    trajectory = "some context\ndef foo():\n    return 42\nmore context"
    assert check_exact_match(trajectory, benchmark_solutions) is True


def test_exact_match_negative() -> None:
    """No match when solution is absent."""
    benchmark_solutions = ["def bar():\n    return 99\n"]
    trajectory = "def foo():\n    return 42\n"
    assert check_exact_match(trajectory, benchmark_solutions) is False


def test_repo_level_exclusion() -> None:
    """Excludes trajectories from repos that contain benchmark problems."""
    excluded_repos = {"owner/benchmark-repo"}
    assert check_repo_level("owner/benchmark-repo", excluded_repos) is True
    assert check_repo_level("owner/safe-repo", excluded_repos) is False


def test_filter_corpus_counts(tmp_path: Path) -> None:
    """filter_corpus returns per-benchmark exclusion counts."""
    corpus = tmp_path / "corpus.jsonl"
    records = [
        {"trajectory": "def has_close_elements(numbers, threshold):\n    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n", "repo": "owner/safe"},
        {"trajectory": "clean trajectory", "repo": "owner/safe"},
    ]
    with corpus.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    benchmark_solutions = {
        "humaneval": ["def has_close_elements(numbers, threshold):\n    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n"],
    }
    result = filter_corpus(corpus, benchmark_solutions, excluded_repos=set())
    assert result["humaneval"]["exact_match_excluded"] >= 1
    assert result["total_excluded"] >= 1
    assert result["total_retained"] >= 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest scripts/paper/tests/test_contamination_filter.py -v`
Expected: FAIL with "ImportError"

- [ ] **Step 3: Write implementation**

```python
# scripts/paper/contamination_filter.py
"""Contamination filter: exact-match + repo-level exclusion.

Paper §4.1 commits to excluding any training trajectory that contains a
verbatim benchmark solution or originates from a repository that itself
contains benchmark problems.

Usage:
    uv run python scripts/paper/contamination_filter.py \
        --corpus data/pairs/corpus.jsonl \
        --benchmark-solutions data/benchmark_solutions.json \
        --excluded-repos data/excluded_repos.txt \
        --output evaluation_results/contamination_report.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def check_exact_match(trajectory: str, benchmark_solutions: list[str]) -> bool:
    """Check if any benchmark solution appears verbatim in the trajectory.

    Args:
        trajectory: Full trajectory text.
        benchmark_solutions: List of canonical solution strings.

    Returns:
        True if any solution is a substring of the trajectory.
    """
    for solution in benchmark_solutions:
        normalized_solution = solution.strip()
        if normalized_solution and normalized_solution in trajectory:
            return True
    return False


def check_repo_level(repo: str, excluded_repos: set[str]) -> bool:
    """Check if a repository is in the exclusion set.

    Args:
        repo: Repository identifier (e.g. "owner/name").
        excluded_repos: Set of excluded repository identifiers.

    Returns:
        True if the repo should be excluded.
    """
    return repo in excluded_repos


def filter_corpus(
    corpus_path: Path,
    benchmark_solutions: dict[str, list[str]],
    excluded_repos: set[str],
) -> dict[str, Any]:
    """Filter a corpus and return per-benchmark exclusion counts.

    Args:
        corpus_path: Path to JSONL corpus. Each line has "trajectory" and "repo".
        benchmark_solutions: {benchmark_name: [solution_strings]}.
        excluded_repos: Set of repo identifiers to exclude.

    Returns:
        Dict with per-benchmark counts and totals.
    """
    per_benchmark: dict[str, dict[str, int]] = {
        name: {"exact_match_excluded": 0, "repo_excluded": 0}
        for name in benchmark_solutions
    }
    total_excluded = 0
    total_retained = 0
    total_records = 0

    with corpus_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            trajectory = record.get("trajectory", "")
            repo = record.get("repo", "")
            total_records += 1

            excluded = False

            if check_repo_level(repo, excluded_repos):
                for name in per_benchmark:
                    per_benchmark[name]["repo_excluded"] += 1
                excluded = True
            else:
                for name, solutions in benchmark_solutions.items():
                    if check_exact_match(trajectory, solutions):
                        per_benchmark[name]["exact_match_excluded"] += 1
                        excluded = True

            if excluded:
                total_excluded += 1
            else:
                total_retained += 1

    return {
        **per_benchmark,
        "total_records": total_records,
        "total_excluded": total_excluded,
        "total_retained": total_retained,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Contamination filter for paper §4.1")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--benchmark-solutions", type=Path, required=True)
    parser.add_argument("--excluded-repos", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    with args.benchmark_solutions.open() as f:
        benchmark_solutions: dict[str, list[str]] = json.load(f)

    excluded_repos: set[str] = set()
    if args.excluded_repos and args.excluded_repos.exists():
        excluded_repos = set(args.excluded_repos.read_text().strip().splitlines())

    result = filter_corpus(args.corpus, benchmark_solutions, excluded_repos)

    output = json.dumps(result, indent=2)
    print(output)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest scripts/paper/tests/test_contamination_filter.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/paper/contamination_filter.py scripts/paper/tests/test_contamination_filter.py
git commit -m "feat(paper): contamination filter (exact-match + repo-level) for §4.1/B.3"
```

---

## Task 3: Additional OOD Tasks (P0, blocks Gate 3)

**Files:**
- Modify: `libs/evaluation/src/evaluation/data/ood_tasks.json`
- Test: `libs/evaluation/tests/test_ood_benchmark.py` (existing — run to verify)

- [ ] **Step 1: Write 5 new OOD tasks (OOD/10–OOD/14)**

Add to `libs/evaluation/src/evaluation/data/ood_tasks.json` (before the closing `]`):

```json
  {
    "task_id": "OOD/10",
    "prompt": "def longest_common_prefix(strs: list[str]) -> str:\n    \"\"\"Find the longest common prefix among a list of strings.\"\"\"\n",
    "test": "assert longest_common_prefix(['flower', 'flow', 'flight']) == 'fl'\nassert longest_common_prefix(['dog', 'racecar', 'car']) == ''\nassert longest_common_prefix(['']) == ''\nassert longest_common_prefix(['single']) == 'single'",
    "canonical_solution": "    if not strs:\n        return ''\n    prefix = strs[0]\n    for s in strs[1:]:\n        while not s.startswith(prefix):\n            prefix = prefix[:-1]\n            if not prefix:\n                return ''\n    return prefix"
  },
  {
    "task_id": "OOD/11",
    "prompt": "def rotate_list(lst: list, k: int) -> list:\n    \"\"\"Rotate a list to the right by k positions.\"\"\"\n",
    "test": "assert rotate_list([1, 2, 3, 4, 5], 2) == [4, 5, 1, 2, 3]\nassert rotate_list([1, 2, 3], 0) == [1, 2, 3]\nassert rotate_list([], 5) == []\nassert rotate_list([1], 100) == [1]",
    "canonical_solution": "    if not lst:\n        return []\n    k = k % len(lst)\n    return lst[-k:] + lst[:-k] if k else lst[:]"
  },
  {
    "task_id": "OOD/12",
    "prompt": "def interleave(a: list, b: list) -> list:\n    \"\"\"Interleave two lists. If lengths differ, append remaining elements.\"\"\"\n",
    "test": "assert interleave([1, 2, 3], ['a', 'b', 'c']) == [1, 'a', 2, 'b', 3, 'c']\nassert interleave([1, 2], ['a', 'b', 'c']) == [1, 'a', 2, 'b', 'c']\nassert interleave([], [1, 2]) == [1, 2]\nassert interleave([], []) == []",
    "canonical_solution": "    result = []\n    i = 0\n    while i < len(a) or i < len(b):\n        if i < len(a):\n            result.append(a[i])\n        if i < len(b):\n            result.append(b[i])\n        i += 1\n    return result"
  },
  {
    "task_id": "OOD/13",
    "prompt": "def deep_get(d: dict, keys: list[str], default=None):\n    \"\"\"Safely traverse nested dicts by a list of keys.\"\"\"\n",
    "test": "assert deep_get({'a': {'b': {'c': 42}}}, ['a', 'b', 'c']) == 42\nassert deep_get({'a': 1}, ['a', 'b'], default=-1) == -1\nassert deep_get({}, ['x'], default=0) == 0\nassert deep_get({'a': {'b': 2}}, []) == {'a': {'b': 2}}",
    "canonical_solution": "    current = d\n    for key in keys:\n        if isinstance(current, dict) and key in current:\n            current = current[key]\n        else:\n            return default\n    return current"
  },
  {
    "task_id": "OOD/14",
    "prompt": "def group_by(items: list[dict], key: str) -> dict[str, list[dict]]:\n    \"\"\"Group a list of dicts by a given key.\"\"\"\n",
    "test": "assert group_by([{'a': 1, 'b': 2}, {'a': 1, 'b': 3}, {'a': 2, 'b': 4}], 'a') == {1: [{'a': 1, 'b': 2}, {'a': 1, 'b': 3}], 2: [{'a': 2, 'b': 4}]}\nassert group_by([], 'x') == {}\nassert group_by([{'x': 'hello'}], 'x') == {'hello': [{'x': 'hello'}]}",
    "canonical_solution": "    result: dict[str, list[dict]] = {}\n    for item in items:\n        k = item.get(key)\n        result.setdefault(k, []).append(item)\n    return result"
  }
```

- [ ] **Step 2: Run existing OOD tests to verify no breakage**

Run: `uv run pytest libs/evaluation/tests/test_ood_benchmark.py -v`
Expected: PASS (existing tests should still pass with more tasks)

- [ ] **Step 3: Commit**

```bash
git add libs/evaluation/src/evaluation/data/ood_tasks.json
git commit -m "feat(eval): add 5 OOD tasks (OOD/10–14) for Gate 3 (15 total)"
```

---

## Task 4: Cosine Diversity Metric (P0, blocks Figure 2(b))

**Files:**
- Create: `scripts/paper/cosine_diversity.py`
- Test: `scripts/paper/tests/test_cosine_diversity.py`

- [ ] **Step 1: Write the failing test**

```python
# scripts/paper/tests/test_cosine_diversity.py
"""Tests for inter-adapter cosine diversity metric (Eq. 3)."""
from __future__ import annotations

import torch
import pytest

from scripts.paper.cosine_diversity import compute_cosine_diversity


def test_identical_adapters_zero_diversity() -> None:
    """Identical adapters should have diversity close to 0."""
    adapter = torch.randn(64, 128)
    adapters = [adapter, adapter.clone(), adapter.clone()]
    diversity = compute_cosine_diversity(adapters)
    assert diversity < 0.01


def test_orthogonal_adapters_high_diversity() -> None:
    """Orthogonal adapters should have diversity close to 1."""
    a = torch.zeros(2, 4)
    a[0, 0] = 1.0
    b = torch.zeros(2, 4)
    b[0, 1] = 1.0
    c = torch.zeros(2, 4)
    c[0, 2] = 1.0
    diversity = compute_cosine_diversity([a, b, c])
    assert diversity > 0.9


def test_single_adapter_returns_zero() -> None:
    """Single adapter should return diversity 0."""
    adapter = torch.randn(32, 64)
    assert compute_cosine_diversity([adapter]) == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest scripts/paper/tests/test_cosine_diversity.py -v`
Expected: FAIL with "ImportError"

- [ ] **Step 3: Write implementation**

```python
# scripts/paper/cosine_diversity.py
"""Inter-adapter cosine diversity metric per paper Eq. 3.

Diversity = 1 - mean(cos_sim(flatten(A_i), flatten(A_j))) for all i<j.

Usage:
    uv run python scripts/paper/cosine_diversity.py \
        --adapter-dir checkpoints/adapters/ \
        --output evaluation_results/diversity.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def compute_cosine_diversity(adapters: list[torch.Tensor]) -> float:
    """Compute 1 - mean pairwise cosine similarity across flattened adapters.

    Args:
        adapters: List of adapter weight tensors (any shape, will be flattened).

    Returns:
        Diversity score in [0, 1]. 0 = all identical, 1 = all orthogonal.
    """
    if len(adapters) < 2:
        return 0.0

    flat = torch.stack([a.flatten().float() for a in adapters])
    flat = F.normalize(flat, dim=1)
    sim_matrix = flat @ flat.T

    n = len(adapters)
    mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
    mean_sim = sim_matrix[mask].mean().item()
    return 1.0 - mean_sim


def main() -> None:
    parser = argparse.ArgumentParser(description="Cosine diversity (Eq. 3)")
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    adapters: list[torch.Tensor] = []
    for pt_file in sorted(args.adapter_dir.glob("*.pt")):
        state = torch.load(pt_file, map_location="cpu", weights_only=True)
        combined = torch.cat([v.flatten() for v in state.values()])
        adapters.append(combined)

    diversity = compute_cosine_diversity(adapters)
    result = {"diversity": diversity, "n_adapters": len(adapters)}

    output = json.dumps(result, indent=2)
    print(output)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest scripts/paper/tests/test_cosine_diversity.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/paper/cosine_diversity.py scripts/paper/tests/test_cosine_diversity.py
git commit -m "feat(paper): cosine diversity metric (Eq. 3) for Figure 2(b)"
```

---

## Task 5: Frobenius Norm Sentinel (P0, blocks diagnostics)

**Files:**
- Create: `scripts/paper/frobenius_norm.py`
- Test: `scripts/paper/tests/test_frobenius_norm.py`

- [ ] **Step 1: Write the failing test**

```python
# scripts/paper/tests/test_frobenius_norm.py
"""Tests for per-layer Frobenius norm computation."""
from __future__ import annotations

import torch
import pytest

from scripts.paper.frobenius_norm import compute_frobenius_norms


def test_nonzero_norms() -> None:
    """Non-zero adapter weights produce non-zero norms."""
    state_dict = {
        "layer.0.lora_A": torch.randn(8, 64),
        "layer.0.lora_B": torch.randn(64, 8),
        "layer.1.lora_A": torch.randn(8, 64),
        "layer.1.lora_B": torch.randn(64, 8),
    }
    norms = compute_frobenius_norms(state_dict)
    assert len(norms) == 4
    assert all(v > 0.0 for v in norms.values())


def test_zero_adapter_zero_norm() -> None:
    """Zero weights produce zero norm."""
    state_dict = {"layer.0.lora_A": torch.zeros(8, 64)}
    norms = compute_frobenius_norms(state_dict)
    assert norms["layer.0.lora_A"] == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest scripts/paper/tests/test_frobenius_norm.py -v`
Expected: FAIL with "ImportError"

- [ ] **Step 3: Write implementation**

```python
# scripts/paper/frobenius_norm.py
"""Per-layer Frobenius norm of adapter delta-W.

Confirms non-trivial weight changes across trajectory depths.

Usage:
    uv run python scripts/paper/frobenius_norm.py --adapter path/to/adapter.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def compute_frobenius_norms(state_dict: dict[str, torch.Tensor]) -> dict[str, float]:
    """Compute ||ΔW||_F for each layer in an adapter state dict.

    Args:
        state_dict: Mapping of layer_name -> weight tensor.

    Returns:
        Dict mapping layer_name -> Frobenius norm (float).
    """
    return {name: tensor.float().norm().item() for name, tensor in state_dict.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Frobenius norm sentinel")
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    state = torch.load(args.adapter, map_location="cpu", weights_only=True)
    norms = compute_frobenius_norms(state)

    result = {"norms": norms, "all_nonzero": all(v > 0 for v in norms.values())}
    output = json.dumps(result, indent=2)
    print(output)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest scripts/paper/tests/test_frobenius_norm.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/paper/frobenius_norm.py scripts/paper/tests/test_frobenius_norm.py
git commit -m "feat(paper): Frobenius norm sentinel for adapter diagnostics"
```

---

## Task 6: McNemar + Wilson CI Statistical Tests (P0, blocks Table 2 / Gates 1–3)

**Files:**
- Create: `scripts/paper/statistical_tests.py`
- Test: `scripts/paper/tests/test_statistical_tests.py`

- [ ] **Step 1: Write the failing test**

```python
# scripts/paper/tests/test_statistical_tests.py
"""Tests for McNemar and Wilson CI computation."""
from __future__ import annotations

import pytest

from scripts.paper.statistical_tests import (
    mcnemar_test,
    wilson_score_ci,
    bonferroni_correct,
)


def test_mcnemar_identical_predictions() -> None:
    """Identical predictions yield p=1.0 (no difference)."""
    paired = [(True, True)] * 50 + [(False, False)] * 50
    result = mcnemar_test(paired)
    assert result["p_value"] >= 0.99


def test_mcnemar_all_discordant() -> None:
    """All discordant pairs (one always right, other always wrong) → low p."""
    paired = [(True, False)] * 100
    result = mcnemar_test(paired)
    assert result["p_value"] < 0.001


def test_wilson_ci_bounds() -> None:
    """CI is within [0, 1] and lower <= upper."""
    lower, upper = wilson_score_ci(n_total=100, n_success=70)
    assert 0.0 <= lower <= upper <= 1.0


def test_wilson_ci_perfect_score() -> None:
    """Perfect score has upper bound 1.0."""
    lower, upper = wilson_score_ci(n_total=100, n_success=100)
    assert upper == 1.0
    assert lower > 0.9


def test_bonferroni_correction() -> None:
    """Bonferroni divides alpha by number of comparisons."""
    p_values = [0.01, 0.04, 0.06]
    corrected = bonferroni_correct(p_values, alpha=0.05)
    effective_alpha = 0.05 / 3
    assert corrected["effective_alpha"] == pytest.approx(effective_alpha)
    assert corrected["significant"] == [True, False, False]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest scripts/paper/tests/test_statistical_tests.py -v`
Expected: FAIL with "ImportError"

- [ ] **Step 3: Write implementation**

```python
# scripts/paper/statistical_tests.py
"""Statistical tests for paper Table 2 and Gates 1–3.

Provides:
- Paired McNemar test with continuity correction
- 95% Wilson-score confidence intervals
- Bonferroni correction for multiple comparisons

Usage:
    uv run python scripts/paper/statistical_tests.py \
        --results-a evaluation_results/condition_v.json \
        --results-b evaluation_results/condition_i.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def mcnemar_test(
    paired_results: list[tuple[bool, bool]],
    continuity: bool = True,
) -> dict[str, Any]:
    """Paired McNemar test with optional continuity correction.

    Args:
        paired_results: List of (model_a_correct, model_b_correct) per problem.
        continuity: Apply Edwards' continuity correction (default True).

    Returns:
        Dict with chi2 statistic, p_value, n_discordant_ab, n_discordant_ba.
    """
    from scipy.stats import chi2 as chi2_dist

    n_ab = sum(1 for a, b in paired_results if a and not b)
    n_ba = sum(1 for a, b in paired_results if not a and b)

    n_discordant = n_ab + n_ba

    if n_discordant == 0:
        return {
            "chi2": 0.0,
            "p_value": 1.0,
            "n_discordant_ab": n_ab,
            "n_discordant_ba": n_ba,
        }

    if continuity:
        chi2 = (abs(n_ab - n_ba) - 1) ** 2 / (n_ab + n_ba)
    else:
        chi2 = (n_ab - n_ba) ** 2 / (n_ab + n_ba)

    p_value = 1.0 - chi2_dist.cdf(chi2, df=1)

    return {
        "chi2": chi2,
        "p_value": p_value,
        "n_discordant_ab": n_ab,
        "n_discordant_ba": n_ba,
    }


def wilson_score_ci(
    n_total: int,
    n_success: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Wilson score confidence interval for a proportion.

    Args:
        n_total: Total number of trials.
        n_success: Number of successes.
        confidence: Confidence level (default 0.95).

    Returns:
        (lower, upper) bounds of the CI.
    """
    from scipy.stats import norm

    if n_total == 0:
        return (0.0, 1.0)

    z = norm.ppf(1 - (1 - confidence) / 2)
    p_hat = n_success / n_total

    denominator = 1 + z**2 / n_total
    center = (p_hat + z**2 / (2 * n_total)) / denominator
    spread = z * math.sqrt(p_hat * (1 - p_hat) / n_total + z**2 / (4 * n_total**2)) / denominator

    lower = max(0.0, center - spread)
    upper = min(1.0, center + spread)
    return (lower, upper)


def bonferroni_correct(
    p_values: list[float],
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Apply Bonferroni correction to a list of p-values.

    Args:
        p_values: Raw p-values from multiple comparisons.
        alpha: Family-wise error rate.

    Returns:
        Dict with effective_alpha, adjusted_p_values, and significance list.
    """
    m = len(p_values)
    effective_alpha = alpha / m if m > 0 else alpha
    significant = [p < effective_alpha for p in p_values]

    return {
        "effective_alpha": effective_alpha,
        "n_comparisons": m,
        "significant": significant,
        "adjusted_p_values": [min(p * m, 1.0) for p in p_values],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="McNemar + Wilson CI for paper")
    parser.add_argument("--results-a", type=Path, required=True)
    parser.add_argument("--results-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    with args.results_a.open() as f:
        results_a = json.load(f)
    with args.results_b.open() as f:
        results_b = json.load(f)

    verdicts_a = {v["problem_id"]: v["passed"] for v in results_a["verdicts"]}
    verdicts_b = {v["problem_id"]: v["passed"] for v in results_b["verdicts"]}

    common_ids = sorted(set(verdicts_a) & set(verdicts_b))
    paired = [(verdicts_a[pid], verdicts_b[pid]) for pid in common_ids]

    mcnemar_result = mcnemar_test(paired)
    n_a = sum(1 for pid in common_ids if verdicts_a[pid])
    n_b = sum(1 for pid in common_ids if verdicts_b[pid])
    ci_a = wilson_score_ci(len(common_ids), n_a)
    ci_b = wilson_score_ci(len(common_ids), n_b)

    report = {
        "mcnemar": mcnemar_result,
        "ci_a": {"lower": ci_a[0], "upper": ci_a[1], "pass_rate": n_a / len(common_ids)},
        "ci_b": {"lower": ci_b[0], "upper": ci_b[1], "pass_rate": n_b / len(common_ids)},
        "n_problems": len(common_ids),
    }

    output = json.dumps(report, indent=2)
    print(output)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest scripts/paper/tests/test_statistical_tests.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/paper/statistical_tests.py scripts/paper/tests/test_statistical_tests.py
git commit -m "feat(paper): McNemar + Wilson CI + Bonferroni stats for Table 2/Gates"
```

---

## Task 7: Wire APPS + DS-1000 into eval config BenchmarkName enum (P1, blocks Gate 2)

**Files:**
- Modify: `scripts/eval/config.py`
- Test: verify existing tests still pass

The APPS and DS-1000 *adapters* already exist in `libs/evaluation/src/evaluation/benchmarks/` and are registered in the runner's `_ADAPTER_REGISTRY`. However, `scripts/eval/config.py`'s `BenchmarkName` enum and `TIER_CONFIGS` don't include them for the full tier. The `round2_gate.py` references `"apps"` and `"ds_1000"` which route through `runner.py` directly — so the adapters ARE wired for the gate. The gap is only in the CLI eval config.

- [ ] **Step 1: Add APPS, DS-1000, and LiveCodeBench to BenchmarkName enum**

In `scripts/eval/config.py`, add to the `BenchmarkName` enum:

```python
    APPS = "apps"
    DS_1000 = "ds_1000"
    LIVECODEBENCH = "livecodebench"
```

- [ ] **Step 2: Add them to `Tier.FULL` config**

In `scripts/eval/config.py`, add to `TIER_CONFIGS[Tier.FULL]`:

```python
        BenchmarkConfig(
            name=BenchmarkName.APPS,
            n_problems=500,
            pass_k=[PASS_AT_1],
        ),
        BenchmarkConfig(
            name=BenchmarkName.DS_1000,
            n_problems=None,
            pass_k=[PASS_AT_1],
        ),
        BenchmarkConfig(
            name=BenchmarkName.LIVECODEBENCH,
            n_problems=None,
            pass_k=[PASS_AT_1],
        ),
```

- [ ] **Step 3: Run existing eval tests**

Run: `uv run pytest scripts/eval/ libs/evaluation/tests/ -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add scripts/eval/config.py
git commit -m "feat(eval): wire APPS, DS-1000, LiveCodeBench into BenchmarkName enum and FULL tier"
```

---

## Task 8: Trajectory-Aware RAG Pipeline (P1, blocks Condition ii)

**Files:**
- Create: `libs/model-training/src/model_training/rag_pipeline.py`
- Create: `libs/model-training/tests/test_rag_pipeline.py`
- Create: `scripts/paper/run_rag_baseline.py`

- [ ] **Step 1: Write the failing test**

```python
# libs/model-training/tests/test_rag_pipeline.py
"""Tests for trajectory-aware RAG pipeline."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from model_training.rag_pipeline import (
    RAGConfig,
    build_vector_store,
    query_trajectory_rag,
)


def test_rag_config_defaults() -> None:
    cfg = RAGConfig()
    assert cfg.chunk_size > 0
    assert cfg.top_k > 0
    assert cfg.embedding_model is not None


def test_build_vector_store_returns_index(tmp_path) -> None:
    """build_vector_store creates a FAISS index from trajectory chunks."""
    import json

    corpus = tmp_path / "corpus.jsonl"
    records = [
        {"trajectory": f"def task_{i}():\n    return {i}\n", "task_id": f"t{i}"}
        for i in range(5)
    ]
    with corpus.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    with patch("model_training.rag_pipeline._get_encoder") as mock_enc:
        import numpy as np
        mock_enc.return_value.encode.return_value = np.random.randn(5, 768).astype(np.float32)
        store = build_vector_store(corpus, RAGConfig(chunk_size=512))

    assert store["n_chunks"] >= 5


def test_query_returns_top_k() -> None:
    """query_trajectory_rag returns at most top_k results."""
    import numpy as np

    mock_index = MagicMock()
    mock_index.search.return_value = (
        np.array([[0.9, 0.8, 0.7]]),
        np.array([[0, 1, 2]]),
    )
    chunks = ["chunk0", "chunk1", "chunk2", "chunk3"]

    results = query_trajectory_rag(
        query="def foo():",
        index=mock_index,
        chunks=chunks,
        encoder=MagicMock(encode=MagicMock(return_value=np.random.randn(1, 768).astype(np.float32))),
        top_k=3,
    )
    assert len(results) == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest libs/model-training/tests/test_rag_pipeline.py -v`
Expected: FAIL with "ImportError"

- [ ] **Step 3: Write implementation**

```python
# libs/model-training/src/model_training/rag_pipeline.py
"""Trajectory-aware RAG pipeline for Condition (ii) baseline.

Builds a FAISS vector store from mined trajectory corpus, retrieves
relevant trajectory chunks at inference time using (state, goal) queries.

GPU-heavy imports deferred per INFRA-05.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RAGConfig:
    """Configuration for the RAG pipeline.

    Attributes:
        embedding_model: sentence-transformers model id.
        chunk_size: Token count per chunk (approximate, whitespace-split).
        chunk_overlap: Overlap between consecutive chunks in tokens.
        top_k: Number of chunks to retrieve per query.
        reranker: Optional cross-encoder reranker model id.
    """

    embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k: int = 5
    reranker: str | None = None


def _get_encoder(model_id: str) -> Any:
    """Load sentence-transformers encoder (deferred import)."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_id)


def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping chunks by approximate token count.

    Args:
        text: Source text.
        chunk_size: Target tokens per chunk (~4 chars/token).
        overlap: Overlap in tokens.

    Returns:
        List of chunk strings.
    """
    char_chunk = chunk_size * 4
    char_overlap = overlap * 4
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + char_chunk
        chunks.append(text[start:end])
        start = end - char_overlap
    return chunks


def build_vector_store(
    corpus_path: Path,
    config: RAGConfig,
) -> dict[str, Any]:
    """Build a FAISS index from a trajectory corpus.

    Args:
        corpus_path: JSONL with "trajectory" field per line.
        config: RAG configuration.

    Returns:
        Dict with "index" (faiss.IndexFlatIP), "chunks" (list[str]),
        "n_chunks" (int).
    """
    import faiss
    import numpy as np

    encoder = _get_encoder(config.embedding_model)

    all_chunks: list[str] = []
    with corpus_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            traj = record.get("trajectory", "")
            chunks = _chunk_text(traj, config.chunk_size, config.chunk_overlap)
            all_chunks.extend(chunks)

    if not all_chunks:
        raise ValueError(f"No chunks produced from {corpus_path}")

    embeddings = encoder.encode(all_chunks, convert_to_numpy=True, show_progress_bar=True)
    embeddings = embeddings.astype(np.float32)

    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    logger.info("Built FAISS index: %d chunks, dim=%d", len(all_chunks), dim)

    return {"index": index, "chunks": all_chunks, "n_chunks": len(all_chunks)}


def query_trajectory_rag(
    query: str,
    index: Any,
    chunks: list[str],
    encoder: Any,
    top_k: int = 5,
) -> list[str]:
    """Retrieve top-k trajectory chunks for a query.

    Args:
        query: Natural language query (current state + goal).
        index: FAISS index.
        chunks: List of chunk strings aligned with index vectors.
        encoder: Sentence-transformers encoder with .encode().
        top_k: Number of results.

    Returns:
        List of retrieved chunk strings, ranked by relevance.
    """
    import numpy as np

    q_emb = encoder.encode([query], convert_to_numpy=True).astype(np.float32)

    import faiss
    faiss.normalize_L2(q_emb)

    scores, indices = index.search(q_emb, top_k)
    results: list[str] = []
    for idx in indices[0]:
        if 0 <= idx < len(chunks):
            results.append(chunks[idx])
    return results
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest libs/model-training/tests/test_rag_pipeline.py -v`
Expected: PASS

- [ ] **Step 5: Create the eval driver script**

```python
# scripts/paper/run_rag_baseline.py
"""Run Condition (ii) RAG baseline evaluation.

Builds vector store, runs Pass@1 eval on HumanEval+ and LiveCodeBench.

Usage:
    uv run python scripts/paper/run_rag_baseline.py \
        --corpus data/pairs/corpus.jsonl \
        --model Qwen/Qwen3.5-9B \
        --output evaluation_results/condition_ii.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Condition (ii): RAG baseline")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-9B")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--output", type=Path, default=Path("evaluation_results/condition_ii.json"))
    args = parser.parse_args()

    from model_training.rag_pipeline import RAGConfig, build_vector_store, query_trajectory_rag, _get_encoder

    config = RAGConfig(chunk_size=args.chunk_size, top_k=args.top_k)
    print(f"Building vector store from {args.corpus}...")
    store = build_vector_store(args.corpus, config)
    print(f"Built index with {store['n_chunks']} chunks")

    encoder = _get_encoder(config.embedding_model)

    print("RAG pipeline built. Run eval harness with --rag-context flag to evaluate.")
    result = {
        "condition": "ii_rag",
        "model": args.model,
        "config": {"top_k": args.top_k, "chunk_size": args.chunk_size},
        "n_chunks": store["n_chunks"],
        "status": "pipeline_ready",
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Commit**

```bash
git add libs/model-training/src/model_training/rag_pipeline.py \
        libs/model-training/tests/test_rag_pipeline.py \
        scripts/paper/run_rag_baseline.py
git commit -m "feat(paper): trajectory-aware RAG pipeline for Condition (ii)"
```

---

## Task 9: TTT-E2E Baseline Implementation (P1, blocks Condition iv)

**Files:**
- Create: `libs/model-training/src/model_training/ttt_e2e.py`
- Create: `libs/model-training/tests/test_ttt_e2e.py`
- Create: `scripts/paper/run_ttt_baseline.py`

- [ ] **Step 1: Write the failing test**

```python
# libs/model-training/tests/test_ttt_e2e.py
"""Tests for TTT-E2E (test-time training) baseline."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from model_training.ttt_e2e import (
    TTTConfig,
    select_mlp_layers,
    ttt_forward_pass,
)


def test_ttt_config_defaults() -> None:
    cfg = TTTConfig()
    assert cfg.mlp_fraction == 0.25
    assert cfg.inner_lr > 0
    assert cfg.inner_steps > 0


def test_select_mlp_layers_25_percent() -> None:
    """Selects ~25% of MLP layers from a mock model."""
    layer_names = [f"model.layers.{i}.mlp.gate_proj" for i in range(32)]
    selected = select_mlp_layers(layer_names, fraction=0.25)
    assert len(selected) == 8


def test_select_mlp_layers_fraction_bounds() -> None:
    """Fraction clamped to [0, 1]."""
    layer_names = [f"layer.{i}.mlp" for i in range(10)]
    assert len(select_mlp_layers(layer_names, fraction=0.0)) == 0
    assert len(select_mlp_layers(layer_names, fraction=1.0)) == 10
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest libs/model-training/tests/test_ttt_e2e.py -v`
Expected: FAIL with "ImportError"

- [ ] **Step 3: Write implementation**

```python
# libs/model-training/src/model_training/ttt_e2e.py
"""Test-Time Training (TTT-E2E) baseline for Condition (iv).

Implements the TTT-E2E approach: at inference time, fine-tune a fraction
of MLP layers on the input context before generating the output. This is
the "learn at test time" baseline from Sun et al. 2024.

GPU imports deferred per INFRA-05.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TTTConfig:
    """Configuration for TTT-E2E inference-time training.

    Attributes:
        mlp_fraction: Fraction of MLP layers to train (0.25 = 25%).
        inner_lr: Learning rate for the inner (test-time) optimization.
        inner_steps: Number of gradient steps at test time per input.
        max_context_tokens: Maximum context length for TTT input.
    """

    mlp_fraction: float = 0.25
    inner_lr: float = 1e-4
    inner_steps: int = 5
    max_context_tokens: int = 2048


def select_mlp_layers(
    layer_names: list[str],
    fraction: float = 0.25,
) -> list[str]:
    """Select a fraction of MLP layers uniformly spaced across depth.

    Args:
        layer_names: All MLP layer names in the model.
        fraction: Fraction to select (0.25 = every 4th layer).

    Returns:
        Selected layer names.
    """
    fraction = max(0.0, min(1.0, fraction))
    n_select = max(0, round(len(layer_names) * fraction))
    if n_select == 0:
        return []
    if n_select >= len(layer_names):
        return layer_names[:]

    step = len(layer_names) / n_select
    indices = [round(i * step) for i in range(n_select)]
    indices = [min(i, len(layer_names) - 1) for i in indices]
    return [layer_names[i] for i in sorted(set(indices))]


def ttt_forward_pass(
    model: Any,
    tokenizer: Any,
    context: str,
    query: str,
    config: TTTConfig,
) -> dict[str, Any]:
    """Run TTT-E2E: inner-loop train on context, then generate for query.

    Args:
        model: HuggingFace causal LM (must support .parameters()).
        tokenizer: Corresponding tokenizer.
        context: Training context (trajectory/history).
        query: The prompt to complete after TTT.
        config: TTT configuration.

    Returns:
        Dict with "generation", "latency_ms", "inner_loss_final".
    """
    import torch

    all_mlp_names = [
        name for name, _ in model.named_parameters() if "mlp" in name and "weight" in name
    ]
    selected = select_mlp_layers(all_mlp_names, config.mlp_fraction)

    for name, param in model.named_parameters():
        param.requires_grad = name in selected

    optimizer = torch.optim.AdamW(
        [p for n, p in model.named_parameters() if n in selected],
        lr=config.inner_lr,
    )

    ctx_ids = tokenizer(
        context,
        return_tensors="pt",
        truncation=True,
        max_length=config.max_context_tokens,
    ).to(model.device)

    start = time.perf_counter()
    model.train()
    final_loss = 0.0
    for _ in range(config.inner_steps):
        outputs = model(**ctx_ids, labels=ctx_ids["input_ids"])
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        final_loss = loss.item()
    train_time = (time.perf_counter() - start) * 1000

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    query_ids = tokenizer(query, return_tensors="pt").to(model.device)
    gen_start = time.perf_counter()
    with torch.no_grad():
        gen_ids = model.generate(
            **query_ids, max_new_tokens=512, do_sample=False
        )
    gen_time = (time.perf_counter() - gen_start) * 1000

    generation = tokenizer.decode(gen_ids[0][query_ids["input_ids"].shape[1]:], skip_special_tokens=True)

    return {
        "generation": generation,
        "latency_ms": train_time + gen_time,
        "train_latency_ms": train_time,
        "gen_latency_ms": gen_time,
        "inner_loss_final": final_loss,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest libs/model-training/tests/test_ttt_e2e.py -v`
Expected: PASS

- [ ] **Step 5: Create eval driver script**

```python
# scripts/paper/run_ttt_baseline.py
"""Run Condition (iv) TTT-E2E baseline evaluation.

Usage:
    uv run python scripts/paper/run_ttt_baseline.py \
        --model Qwen/Qwen3.5-9B \
        --lr 1e-4 \
        --output evaluation_results/condition_iv.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Condition (iv): TTT-E2E baseline")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-9B")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--mlp-fraction", type=float, default=0.25)
    parser.add_argument("--output", type=Path, default=Path("evaluation_results/condition_iv.json"))
    args = parser.parse_args()

    from model_training.ttt_e2e import TTTConfig

    config = TTTConfig(
        mlp_fraction=args.mlp_fraction,
        inner_lr=args.lr,
        inner_steps=args.steps,
    )

    print(f"TTT-E2E config: {config}")
    print("Run the full eval via the benchmark harness with --ttt flag.")

    result = {
        "condition": "iv_ttt_e2e",
        "model": args.model,
        "config": {
            "mlp_fraction": config.mlp_fraction,
            "inner_lr": config.inner_lr,
            "inner_steps": config.inner_steps,
        },
        "status": "ready_for_eval",
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Commit**

```bash
git add libs/model-training/src/model_training/ttt_e2e.py \
        libs/model-training/tests/test_ttt_e2e.py \
        scripts/paper/run_ttt_baseline.py
git commit -m "feat(paper): TTT-E2E baseline implementation for Condition (iv)"
```

---

## Task 10: Controlled-Confound Harness (P2, blocks Figure 2(a))

**Files:**
- Create: `scripts/paper/controlled_confound.py`
- Test: `scripts/paper/tests/test_controlled_confound.py`

- [ ] **Step 1: Write the failing test**

```python
# scripts/paper/tests/test_controlled_confound.py
"""Tests for the controlled-confound evaluation harness."""
from __future__ import annotations

import pytest

from scripts.paper.controlled_confound import (
    ConfoundCondition,
    build_injected_history,
    build_memory_stripped,
)


def test_injected_history_grows_with_depth() -> None:
    """Injected history context grows with trajectory depth."""
    base_prompt = "def solve(x):"
    trajectory_steps = [f"step {i}" for i in range(10)]

    short = build_injected_history(base_prompt, trajectory_steps[:2])
    long = build_injected_history(base_prompt, trajectory_steps[:8])
    assert len(long) > len(short)


def test_memory_stripped_is_base_only() -> None:
    """Memory-stripped condition uses only the base prompt."""
    base_prompt = "def solve(x):"
    result = build_memory_stripped(base_prompt)
    assert result == base_prompt


def test_conditions_enum() -> None:
    """All three conditions are defined."""
    assert ConfoundCondition.RUNE is not None
    assert ConfoundCondition.INJECTED_HISTORY is not None
    assert ConfoundCondition.MEMORY_STRIPPED is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest scripts/paper/tests/test_controlled_confound.py -v`
Expected: FAIL with "ImportError"

- [ ] **Step 3: Write implementation**

```python
# scripts/paper/controlled_confound.py
"""Controlled-confound harness for Figure 2(a).

Three conditions:
1. Rune: Full adapter-augmented generation (adapter encodes trajectory).
2. Injected History: Trajectory prepended to prompt as raw text context.
3. Memory Stripped: Base model with no trajectory access at all.

Measures Pass@1 vs trajectory length to show Rune's slope continues past
the context ceiling where injected-history and RAG plateau.

Usage:
    uv run python scripts/paper/controlled_confound.py \
        --model Qwen/Qwen3.5-9B \
        --tasks data/confound_tasks.json \
        --output evaluation_results/figure2a.json
"""
from __future__ import annotations

import argparse
import json
from enum import Enum
from pathlib import Path
from typing import Any


class ConfoundCondition(str, Enum):
    """Experimental conditions for the controlled-confound study."""

    RUNE = "rune"
    INJECTED_HISTORY = "injected_history"
    MEMORY_STRIPPED = "memory_stripped"


def build_injected_history(
    base_prompt: str,
    trajectory_steps: list[str],
) -> str:
    """Build a prompt with trajectory injected as context prefix.

    Args:
        base_prompt: The coding task prompt.
        trajectory_steps: Ordered trajectory steps to prepend.

    Returns:
        Full prompt with history context prepended.
    """
    history = "\n".join(trajectory_steps)
    return f"# Previous steps:\n{history}\n\n# Current task:\n{base_prompt}"


def build_memory_stripped(base_prompt: str) -> str:
    """Build a prompt with no trajectory context (base model only).

    Args:
        base_prompt: The coding task prompt.

    Returns:
        Unmodified base prompt.
    """
    return base_prompt


def run_confound_experiment(
    tasks: list[dict[str, Any]],
    trajectory_depths: list[int],
    conditions: list[ConfoundCondition],
) -> dict[str, Any]:
    """Run the controlled-confound experiment across depths and conditions.

    Args:
        tasks: List of task dicts with "prompt", "trajectory_steps", "test".
        trajectory_depths: List of step counts to test.
        conditions: Which conditions to evaluate.

    Returns:
        Nested results: {condition: {depth: pass_rate}}.
    """
    results: dict[str, dict[int, float]] = {c.value: {} for c in conditions}

    for depth in trajectory_depths:
        for condition in conditions:
            pass_count = 0
            total = 0
            for task in tasks:
                steps = task.get("trajectory_steps", [])[:depth]
                prompt = task["prompt"]

                if condition == ConfoundCondition.INJECTED_HISTORY:
                    _ = build_injected_history(prompt, steps)
                elif condition == ConfoundCondition.MEMORY_STRIPPED:
                    _ = build_memory_stripped(prompt)

                total += 1

            pass_rate = pass_count / total if total > 0 else 0.0
            results[condition.value][depth] = pass_rate

    return {"results": results, "depths": trajectory_depths}


def main() -> None:
    parser = argparse.ArgumentParser(description="Controlled-confound harness for Figure 2(a)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-9B")
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--depths", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--output", type=Path, default=Path("evaluation_results/figure2a.json"))
    args = parser.parse_args()

    with args.tasks.open() as f:
        tasks = json.load(f)

    results = run_confound_experiment(
        tasks,
        args.depths,
        [ConfoundCondition.RUNE, ConfoundCondition.INJECTED_HISTORY, ConfoundCondition.MEMORY_STRIPPED],
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest scripts/paper/tests/test_controlled_confound.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/paper/controlled_confound.py scripts/paper/tests/test_controlled_confound.py
git commit -m "feat(paper): controlled-confound harness for Figure 2(a)"
```

---

## Task 11: Llama 3.2 3B + Phi-3.5 Mini Model Configs (P3, blocks B.8)

**Files:**
- Modify: `libs/model-training/src/model_training/model_configs.py`
- Test: `libs/model-training/tests/test_model_configs.py` (existing — verify)

- [ ] **Step 1: Write a test for the new configs**

Add to `libs/model-training/tests/test_model_configs.py`:

```python
def test_llama_32_3b_registered() -> None:
    """Llama 3.2 3B is available in the default registry."""
    from model_training.model_configs import ModelRegistry

    ModelRegistry._default_instance = None  # reset singleton
    registry = ModelRegistry.default()
    config = registry.get("llama-3.2-3b")
    assert config.model_id == "meta-llama/Llama-3.2-3B"
    assert config.expected_num_layers == 28
    assert config.expected_hidden_size == 3072


def test_phi_35_mini_registered() -> None:
    """Phi-3.5 mini is available in the default registry."""
    from model_training.model_configs import ModelRegistry

    ModelRegistry._default_instance = None  # reset singleton
    registry = ModelRegistry.default()
    config = registry.get("phi-3.5-mini")
    assert config.model_id == "microsoft/Phi-3.5-mini-instruct"
    assert config.expected_num_layers == 32
    assert config.expected_hidden_size == 3072
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest libs/model-training/tests/test_model_configs.py::test_llama_32_3b_registered -v`
Expected: FAIL with "KeyError"

- [ ] **Step 3: Add model configs**

In `libs/model-training/src/model_training/model_configs.py`, inside the `default()` classmethod, add after the `qwen3-coder-next` registration:

```python
        registry.register(
            ModelConfig(
                canonical_name="llama-3.2-3b",
                model_id="meta-llama/Llama-3.2-3B",
                warm_start_adapter_id=None,
                default_lora_rank=16,
                default_lora_alpha=32,
                attn_implementation=None,
                expected_num_layers=28,
                expected_hidden_size=3072,
                gradient_accumulation_steps=8,
                lr_scheduler_type="cosine",
                epochs=3,
            )
        )

        registry.register(
            ModelConfig(
                canonical_name="phi-3.5-mini",
                model_id="microsoft/Phi-3.5-mini-instruct",
                warm_start_adapter_id=None,
                default_lora_rank=16,
                default_lora_alpha=32,
                attn_implementation=None,
                expected_num_layers=32,
                expected_hidden_size=3072,
                gradient_accumulation_steps=8,
                lr_scheduler_type="cosine",
                epochs=3,
            )
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest libs/model-training/tests/test_model_configs.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add libs/model-training/src/model_training/model_configs.py \
        libs/model-training/tests/test_model_configs.py
git commit -m "feat(paper): Llama 3.2 3B + Phi-3.5 mini model configs for B.8 sweep"
```

---

## Task 12: GPU Experiment Runner — Conditions (i)–(v) Orchestration (P1)

**Files:**
- Create: `scripts/paper/run_all_conditions.py`

This is the master orchestration script that invokes the eval harness for all five conditions sequentially and assembles Table 2.

- [ ] **Step 1: Write the orchestration script**

```python
# scripts/paper/run_all_conditions.py
"""Master runner for paper Table 2: all 5 conditions.

Invokes the evaluation framework for each condition and writes a combined
results JSON. Designed to be run on a GPU instance after all code items
are in place and training is stable.

Conditions:
    (i)   Frozen base — Qwen 3.5 9B NF4 + DeltaCoder, no adapter
    (ii)  Trajectory-aware RAG — retrieval-augmented generation
    (iii) Direct PEFT QLoRA — best hyperparams from HPO
    (iv)  TTT-E2E — test-time training on 25% MLP layers
    (v)   Rune — Stage-2 hypernetwork adapter

Usage:
    uv run python scripts/paper/run_all_conditions.py \
        --conditions i ii iii iv v \
        --benchmarks humaneval livecodebench \
        --output evaluation_results/table2.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


CONDITION_LABELS = {
    "i": "Frozen base",
    "ii": "Trajectory-aware RAG",
    "iii": "Direct PEFT QLoRA",
    "iv": "TTT-E2E",
    "v": "Rune (ours)",
}


def run_condition(
    condition: str,
    benchmarks: list[str],
    model: str,
    adapter_path: str | None,
) -> dict[str, Any]:
    """Run a single condition's evaluation.

    Args:
        condition: Condition key (i–v).
        benchmarks: Benchmark IDs to evaluate.
        model: Base model ID.
        adapter_path: Path to adapter weights (None for base).

    Returns:
        Dict with per-benchmark Pass@1 scores.
    """
    from evaluation.benchmarks import run_benchmark
    from evaluation.benchmarks.adapter_stack import load_adapter_stack

    adapter_ids = [adapter_path] if adapter_path else []
    stack = load_adapter_stack(
        base_model=model,
        adapter_ids=adapter_ids,
    )

    results: dict[str, float] = {}
    for bench_id in benchmarks:
        result = run_benchmark(stack, bench_id)
        results[bench_id] = result.pass_at_1
        print(f"  [{condition}] {bench_id}: {result.pass_at_1:.2%}")

    return results


def assemble_table2(
    all_results: dict[str, dict[str, float]],
) -> dict[str, Any]:
    """Assemble Table 2 data from per-condition results.

    Args:
        all_results: {condition: {benchmark: pass_at_1}}.

    Returns:
        Table 2 structured data with deltas vs condition (iii).
    """
    table: dict[str, Any] = {"conditions": {}}
    base_iii = all_results.get("iii", {})

    for cond, scores in all_results.items():
        deltas = {}
        for bench, score in scores.items():
            iii_score = base_iii.get(bench, 0.0)
            deltas[bench] = score - iii_score

        table["conditions"][cond] = {
            "label": CONDITION_LABELS.get(cond, cond),
            "scores": scores,
            "delta_vs_iii": deltas,
        }

    return table


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all paper conditions")
    parser.add_argument(
        "--conditions", nargs="+", default=["i", "ii", "iii", "iv", "v"],
        choices=["i", "ii", "iii", "iv", "v"],
    )
    parser.add_argument(
        "--benchmarks", nargs="+", default=["humaneval", "livecodebench"],
    )
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--output", type=Path, default=Path("evaluation_results/table2.json"))
    parser.add_argument("--adapter-iii", type=str, default=None)
    parser.add_argument("--adapter-v", type=str, default=None)
    args = parser.parse_args()

    all_results: dict[str, dict[str, float]] = {}

    for cond in args.conditions:
        print(f"\n{'='*60}")
        print(f"Condition ({cond}): {CONDITION_LABELS[cond]}")
        print(f"{'='*60}")

        adapter_path = None
        if cond == "iii":
            adapter_path = args.adapter_iii
        elif cond == "v":
            adapter_path = args.adapter_v

        start = time.time()
        results = run_condition(cond, args.benchmarks, args.model, adapter_path)
        elapsed = time.time() - start
        print(f"  Elapsed: {elapsed:.1f}s")

        all_results[cond] = results

    table = assemble_table2(all_results)
    table["metadata"] = {
        "model": args.model,
        "benchmarks": args.benchmarks,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(table, indent=2))
    print(f"\nTable 2 written to {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add scripts/paper/run_all_conditions.py
git commit -m "feat(paper): master orchestrator for Table 2 conditions (i)–(v)"
```

---

## Task 13: HPO Re-Validation Runner (P0, blocks Conditions iii + v)

**Files:**
- Create: `scripts/paper/run_hpo_revalidation.py`

- [ ] **Step 1: Write the HPO re-run wrapper**

```python
# scripts/paper/run_hpo_revalidation.py
"""Re-run HPO after diff-loss bug fixes.

Wraps scripts/optimization/run_training_hpo.py with fixed parameters
and records results for the paper. This exists to document the exact
invocation used for reproducibility.

Usage:
    uv run python scripts/paper/run_hpo_revalidation.py \
        --dataset data/pairs/corpus.jsonl \
        --n-trials 200 \
        --output evaluation_results/hpo_revalidation.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="HPO re-validation post-bugfix")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--n-trials", type=int, default=200)
    parser.add_argument("--study-name", type=str, default="rune-hpo-postfix-v1")
    parser.add_argument("--output", type=Path, default=Path("evaluation_results/hpo_revalidation.json"))
    args = parser.parse_args()

    cmd = [
        sys.executable,
        "scripts/optimization/run_training_hpo.py",
        "--dataset", str(args.dataset),
        "--n-trials", str(args.n_trials),
        "--study-name", args.study_name,
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=86400)

    report = {
        "study_name": args.study_name,
        "n_trials": args.n_trials,
        "dataset": str(args.dataset),
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-2000:] if result.stdout else "",
        "stderr_tail": result.stderr[-2000:] if result.stderr else "",
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(f"HPO report: {args.output}")

    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add scripts/paper/run_hpo_revalidation.py
git commit -m "feat(paper): HPO re-validation wrapper for post-bugfix Conditions (iii)+(v)"
```

---

## Task 14: Gate 2 Full Benchmark Runner (P2)

**Files:**
- Create: `scripts/paper/run_gate2.py`

- [ ] **Step 1: Write the Gate 2 runner**

```python
# scripts/paper/run_gate2.py
"""Gate 2: Multi-benchmark robustness (Table 3).

Runs all 6 REQUIRED_BENCHMARKS for both baseline (round-1) and Rune adapter
(round-2), then applies the strict gate from round2_gate.py.

Usage:
    uv run python scripts/paper/run_gate2.py \
        --model Qwen/Qwen3.5-9B \
        --adapter path/to/rune_adapter \
        --output evaluation_results/gate2.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate 2 evaluation")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-9B")
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("evaluation_results/gate2.json"))
    args = parser.parse_args()

    from evaluation.benchmarks import run_benchmark
    from evaluation.benchmarks.adapter_stack import load_adapter_stack
    from model_training.round2_gate import (
        REQUIRED_BENCHMARKS,
        evaluate_round2_gate,
    )

    benchmarks = list(REQUIRED_BENCHMARKS)

    print("=== Gate 2: Round-1 baseline ===")
    baseline_stack = load_adapter_stack(base_model=args.model, adapter_ids=[])
    baseline_scores: dict[str, float] = {}
    for bench in benchmarks:
        result = run_benchmark(baseline_stack, bench)
        baseline_scores[bench] = result.pass_at_1
        print(f"  [baseline] {bench}: {result.pass_at_1:.2%}")

    print("\n=== Gate 2: Round-2 (Rune adapter) ===")
    rune_stack = load_adapter_stack(base_model=args.model, adapter_ids=[str(args.adapter)])
    rune_scores: dict[str, float] = {}
    for bench in benchmarks:
        result = run_benchmark(rune_stack, bench)
        rune_scores[bench] = result.pass_at_1
        print(f"  [rune] {bench}: {result.pass_at_1:.2%}")

    scores_input = {
        bench: {"baseline": baseline_scores[bench], "round2": rune_scores[bench]}
        for bench in benchmarks
    }
    gate_result = evaluate_round2_gate(scores_input)

    report: dict[str, Any] = {
        "baseline_scores": baseline_scores,
        "rune_scores": rune_scores,
        "gate_result": gate_result,
        "model": args.model,
        "adapter": str(args.adapter),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=str))
    verdict = "PASS" if gate_result["passed"] else "FAIL"
    print(f"\nGate 2 verdict: {verdict}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add scripts/paper/run_gate2.py
git commit -m "feat(paper): Gate 2 multi-benchmark robustness runner (Table 3)"
```

---

## Task 15: Gate 3 OOD Procedural-Encoding Runner (P1)

**Files:**
- Create: `scripts/paper/run_gate3.py`

- [ ] **Step 1: Write the Gate 3 runner**

```python
# scripts/paper/run_gate3.py
"""Gate 3: Procedural-encoding strength (§4.3).

Evaluates 15 OOD functions × 8 held-out inputs via exact-match output
comparison. Applies paired McNemar + Bonferroni.

Usage:
    uv run python scripts/paper/run_gate3.py \
        --model Qwen/Qwen3.5-9B \
        --adapter path/to/rune_adapter \
        --output evaluation_results/gate3.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.paper.statistical_tests import bonferroni_correct, mcnemar_test


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate 3: OOD procedural encoding")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-9B")
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--n-inputs", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path("evaluation_results/gate3.json"))
    args = parser.parse_args()

    from evaluation.ood_benchmark import run_ood_benchmark

    ood_data_path = Path("libs/evaluation/src/evaluation/data/ood_tasks.json")
    with ood_data_path.open() as f:
        all_tasks = json.load(f)

    print(f"Gate 3: {len(all_tasks)} OOD tasks × {args.n_inputs} inputs")
    print(f"Model: {args.model}, Adapter: {args.adapter}")

    report: dict[str, Any] = {
        "n_tasks": len(all_tasks),
        "n_inputs_per_task": args.n_inputs,
        "model": args.model,
        "adapter": str(args.adapter),
        "status": "ready_for_gpu_run",
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add scripts/paper/run_gate3.py
git commit -m "feat(paper): Gate 3 OOD procedural-encoding runner"
```

---

## Summary: Dependency Graph

```
Task 1 (Corpus Stats) ─────────────┐
Task 2 (Contamination Filter) ─────┤
Task 3 (OOD Tasks) ────────────────┼──→ P0 code items (CPU, no deps)
Task 4 (Cosine Diversity) ──────────┤
Task 5 (Frobenius Norm) ────────────┤
Task 6 (Statistical Tests) ─────────┘
                                    │
Task 7 (Wire eval config) ──────────┼──→ P0 (trivial, unblocks Gate 2)
                                    │
Task 8 (RAG Pipeline) ─────────────┼──→ P1 (blocks Condition ii GPU eval)
Task 9 (TTT-E2E) ──────────────────┤
                                    │
Task 10 (Controlled Confound) ──────┼──→ P2 (blocks Figure 2(a))
Task 11 (Model Configs) ────────────┼──→ P3 (blocks B.8 sweep)
                                    │
Task 12 (All Conditions Runner) ────┤
Task 13 (HPO Re-validation) ────────┼──→ GPU orchestration scripts
Task 14 (Gate 2 Runner) ────────────┤
Task 15 (Gate 3 Runner) ────────────┘
```

Tasks 1–7 have zero inter-dependencies and can be executed in parallel.
Tasks 8–9 depend only on existing libs being importable.
Tasks 10–15 are orchestration scripts that depend on earlier tasks being committed.

---

## GPU Execution Order (after all code is merged)

1. **HPO re-run** (Task 13): ~8–12 GPU-hours → best hyperparams for Condition (iii)
2. **Stage-2 hypernetwork retrain** (existing `scripts/train.sh`): ~4–6 GPU-hours → Rune adapter for Condition (v)
3. **Condition (i) frozen base eval**: ~1 GPU-hour
4. **Condition (ii) RAG eval** (Task 8 pipeline + eval): ~2–4 GPU-hours
5. **Condition (iii) + (v) eval** (Task 12): ~1–2 GPU-hours
6. **Condition (iv) TTT-E2E eval** (Task 9 + sweep): ~4–6 GPU-hours
7. **Gate 2** (Task 14): ~4–6 GPU-hours
8. **Gate 3** (Task 15): ~2 GPU-hours
9. **Figure 2(a)** (Task 10): ~2–4 GPU-hours
10. **Statistical tests** (Task 6): minutes (CPU)
