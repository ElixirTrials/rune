"""SWE-Bench-Lite benchmark adapter.

load_problems() loads the 300-problem Lite split from
princeton-nlp/SWE-bench_Lite and surfaces repo/test metadata needed for
scoring (FAIL_TO_PASS, PASS_TO_PASS, test_patch, environment_setup_commit).

score() implements the repo-checkout + git-apply + pytest preflight:
    1. Shallow-clone ``metadata['repo']`` into a temp dir at base_commit.
    2. Apply ``test_patch`` so the oracle tests are present.
    3. Apply the model's ``generation`` (unified diff) with ``git apply``.
    4. Install the checkout in editable mode (best-effort — failure is treated
       as an environment error, not a patch failure).
    5. Run pytest on ``FAIL_TO_PASS`` and ``PASS_TO_PASS`` node ids.
    6. PASS iff every FAIL_TO_PASS and PASS_TO_PASS test passes.

The score pipeline is gated behind the ``RUNE_SWE_BENCH_SCORE`` env var. When
unset, score() raises NotImplementedError to preserve Plan A's CI contract
(no network / Docker in CI). Set ``RUNE_SWE_BENCH_SCORE=1`` to enable.
"""

from __future__ import annotations

import json
import logging
import os
import random
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evaluation.benchmarks.protocol import PassVerdict, Problem

logger = logging.getLogger(__name__)

_DEFAULT_FIXTURE = (
    Path(__file__).parent.parent.parent.parent.parent.parent
    / "tests"
    / "fixtures"
    / "swe_bench_lite_mini.parquet"
)

_SCORE_ENV_FLAG = "RUNE_SWE_BENCH_SCORE"


@dataclass(frozen=True)
class _CmdResult:
    """Lightweight wrapper over subprocess.run for readability."""

    returncode: int
    stdout: str
    stderr: str
    timed_out: bool


class SWEBenchLiteAdapter:
    """Benchmark adapter for SWE-Bench-Lite (held-out generalization set).

    Attributes:
        benchmark_id: "swe_bench_lite".

    Note:
        score() performs a heavyweight preflight (git clone + pytest) when
        the ``RUNE_SWE_BENCH_SCORE`` env var is truthy. By default it still
        raises NotImplementedError so CI remains hermetic.
    """

    benchmark_id: str = "swe_bench_lite"
    _fixture_path: Path = _DEFAULT_FIXTURE

    def load_problems(
        self,
        max_samples: int | None = None,
        seed: int = 42,
    ) -> list[Problem]:
        """Load SWE-Bench-Lite problem metadata.

        Does NOT clone repositories or set up execution environments. The
        metadata field carries repo, base_commit, test_patch, FAIL_TO_PASS,
        PASS_TO_PASS, and environment_setup_commit so that score() can do
        its work later if enabled.

        Args:
            max_samples: Cap on returned problems.
            seed: Subsampling seed.

        Returns:
            List of Problem instances.
        """
        rows = self._load_rows()
        if max_samples is not None and max_samples < len(rows):
            rng = random.Random(seed)
            rows = rng.sample(rows, max_samples)
        return [self._row_to_problem(r) for r in rows]

    def score(
        self,
        problem: Problem,
        generation: str,
        timeout_s: int = 30,
    ) -> PassVerdict:
        """Score a generated patch by applying it and running the oracle tests.

        Gated by ``RUNE_SWE_BENCH_SCORE``: when unset, raises
        NotImplementedError (Plan A CI contract). When set, performs the
        full clone/apply/pytest pipeline.

        Args:
            problem: Problem instance with repo/base_commit/test_patch/
                FAIL_TO_PASS/PASS_TO_PASS metadata.
            generation: Proposed patch (unified diff format).
            timeout_s: Overall wall clock budget in seconds.

        Returns:
            PassVerdict. ``passed`` is True iff every FAIL_TO_PASS and
            PASS_TO_PASS test passes after applying the generation.

        Raises:
            NotImplementedError: When the ``RUNE_SWE_BENCH_SCORE`` env var
                is not truthy, preserving the hermetic-CI default.
        """
        if not _score_enabled():
            raise NotImplementedError(
                "preflight clone/apply not yet implemented — see follow-on plan. "
                "Set RUNE_SWE_BENCH_SCORE=1 to run clone + git-apply + pytest."
            )

        repo = str(problem.metadata.get("repo", ""))
        base_commit = str(problem.metadata.get("base_commit", ""))
        test_patch = str(problem.metadata.get("test_patch", ""))
        fail_to_pass = _decode_node_list(problem.metadata.get("FAIL_TO_PASS"))
        pass_to_pass = _decode_node_list(problem.metadata.get("PASS_TO_PASS"))

        if not repo or not base_commit:
            return PassVerdict(
                problem_id=problem.problem_id,
                passed=False,
                generation=generation,
                error="missing repo/base_commit metadata",
                timed_out=False,
            )
        if not fail_to_pass and not pass_to_pass:
            return PassVerdict(
                problem_id=problem.problem_id,
                passed=False,
                generation=generation,
                error="no FAIL_TO_PASS or PASS_TO_PASS tests declared",
                timed_out=False,
            )

        with tempfile.TemporaryDirectory(prefix="swe_bench_") as workdir:
            work_path = Path(workdir)
            clone = _run_git_clone(repo, base_commit, work_path, timeout_s)
            if clone.returncode != 0:
                return PassVerdict(
                    problem_id=problem.problem_id,
                    passed=False,
                    generation=generation,
                    error=f"clone failed: {clone.stderr.strip()[:400]}",
                    timed_out=clone.timed_out,
                )

            repo_dir = work_path / "repo"

            if test_patch:
                apply_tests = _apply_patch(repo_dir, test_patch, timeout_s)
                if apply_tests.returncode != 0:
                    return PassVerdict(
                        problem_id=problem.problem_id,
                        passed=False,
                        generation=generation,
                        error=f"test_patch did not apply: "
                        f"{apply_tests.stderr.strip()[:400]}",
                        timed_out=apply_tests.timed_out,
                    )

            apply_gen = _apply_patch(repo_dir, generation, timeout_s)
            if apply_gen.returncode != 0:
                return PassVerdict(
                    problem_id=problem.problem_id,
                    passed=False,
                    generation=generation,
                    error=f"generation did not apply: "
                    f"{apply_gen.stderr.strip()[:400]}",
                    timed_out=apply_gen.timed_out,
                )

            # Best-effort editable install; tolerate failure so scoring still
            # runs for repos that are already importable.
            _run_pip_install_editable(repo_dir, timeout_s)

            all_tests = fail_to_pass + pass_to_pass
            pytest_res = _run_pytest(repo_dir, all_tests, timeout_s)
            if pytest_res.timed_out:
                return PassVerdict(
                    problem_id=problem.problem_id,
                    passed=False,
                    generation=generation,
                    error="pytest timed out",
                    timed_out=True,
                )

            return PassVerdict(
                problem_id=problem.problem_id,
                passed=pytest_res.returncode == 0,
                generation=generation,
                error=None
                if pytest_res.returncode == 0
                else f"pytest returncode={pytest_res.returncode}: "
                f"{pytest_res.stdout.strip()[-400:]}",
                timed_out=False,
            )

    def _load_rows(self) -> list[dict[str, Any]]:
        """Load rows from HF or local fixture."""
        offline = os.environ.get("HF_DATASETS_OFFLINE", "0") == "1"
        if offline:
            return self._load_from_fixture()
        try:
            return self._load_from_hf()
        except Exception:
            return self._load_from_fixture()

    def _load_from_hf(self) -> list[dict[str, Any]]:
        """Load from HuggingFace datasets."""
        import datasets as hf_datasets  # deferred

        ds = hf_datasets.load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
        return list(ds)

    def _load_from_fixture(self) -> list[dict[str, Any]]:
        """Load from local parquet fixture."""
        import pandas as pd

        records: list[dict[Any, Any]] = pd.read_parquet(self._fixture_path).to_dict(
            orient="records"
        )
        return records

    def _row_to_problem(self, row: dict[str, Any]) -> Problem:
        """Convert a raw SWE-Bench-Lite row to a Problem instance.

        Carries the full scoring-relevant metadata (test_patch,
        FAIL_TO_PASS, PASS_TO_PASS, environment_setup_commit) alongside
        the basics (repo, base_commit, issue_url).
        """
        return Problem(
            problem_id=f"swe_bench_lite/{row.get('instance_id', '')}",
            prompt=str(row.get("problem_statement", "")),
            test_code="",  # scoring uses metadata, not test_code
            metadata={
                "repo": str(row.get("repo", "")),
                "base_commit": str(row.get("base_commit", "")),
                "issue_url": str(row.get("issue_url", row.get("instance_id", ""))),
                "test_patch": str(row.get("test_patch", "")),
                "FAIL_TO_PASS": row.get("FAIL_TO_PASS", "[]"),
                "PASS_TO_PASS": row.get("PASS_TO_PASS", "[]"),
                "environment_setup_commit": str(
                    row.get("environment_setup_commit", "")
                ),
            },
        )


# ---------------------------------------------------------------------------
# Module-level helpers (overridable in tests via monkeypatch).


def _score_enabled() -> bool:
    """Whether score() should run the full pipeline (env-gated)."""
    return os.environ.get(_SCORE_ENV_FLAG, "").lower() in {"1", "true", "yes"}


def _decode_node_list(raw: Any) -> list[str]:
    """Decode FAIL_TO_PASS / PASS_TO_PASS (JSON-encoded list or list)."""
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(x) for x in raw]
    try:
        decoded = json.loads(str(raw))
    except json.JSONDecodeError:
        return []
    if isinstance(decoded, list):
        return [str(x) for x in decoded]
    return []


def _run(
    cmd: list[str],
    cwd: Path | None,
    timeout_s: int,
) -> _CmdResult:
    """Run a command with a timeout, capturing stdout/stderr."""
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=max(timeout_s, 1),
            check=False,
        )
        return _CmdResult(
            returncode=proc.returncode,
            stdout=proc.stdout or "",
            stderr=proc.stderr or "",
            timed_out=False,
        )
    except subprocess.TimeoutExpired as exc:
        return _CmdResult(
            returncode=-1,
            stdout=(exc.stdout or b"").decode("utf-8", errors="replace")
            if isinstance(exc.stdout, bytes)
            else (exc.stdout or ""),
            stderr=(exc.stderr or b"").decode("utf-8", errors="replace")
            if isinstance(exc.stderr, bytes)
            else (exc.stderr or ""),
            timed_out=True,
        )
    except FileNotFoundError as exc:
        return _CmdResult(
            returncode=-1,
            stdout="",
            stderr=str(exc),
            timed_out=False,
        )


def _run_git_clone(
    repo: str,
    base_commit: str,
    workdir: Path,
    timeout_s: int,
) -> _CmdResult:
    """Clone the repo and check out the base commit. Uses partial clone."""
    repo_dir = workdir / "repo"
    clone_url = f"https://github.com/{repo}.git"
    clone = _run(
        [
            "git",
            "clone",
            "--filter=blob:none",
            "--no-checkout",
            clone_url,
            str(repo_dir),
        ],
        cwd=None,
        timeout_s=timeout_s,
    )
    if clone.returncode != 0:
        return clone
    # Need to fetch the specific base_commit — partial clones don't fetch all.
    _run(
        ["git", "fetch", "--depth=1", "origin", base_commit],
        cwd=repo_dir,
        timeout_s=timeout_s,
    )
    return _run(
        ["git", "checkout", base_commit],
        cwd=repo_dir,
        timeout_s=timeout_s,
    )


def _apply_patch(repo_dir: Path, patch_text: str, timeout_s: int) -> _CmdResult:
    """Apply a unified-diff patch via ``git apply``."""
    if not patch_text.strip():
        return _CmdResult(
            returncode=0, stdout="", stderr="empty patch", timed_out=False
        )
    patch_file = repo_dir / "_rune.patch"
    patch_file.write_text(patch_text)
    try:
        return _run(
            ["git", "apply", "--whitespace=nowarn", str(patch_file)],
            cwd=repo_dir,
            timeout_s=timeout_s,
        )
    finally:
        try:
            patch_file.unlink(missing_ok=True)
        except OSError:
            pass


def _run_pip_install_editable(repo_dir: Path, timeout_s: int) -> _CmdResult:
    """Best-effort editable install. Failures are logged but not fatal."""
    result = _run(
        [sys.executable, "-m", "pip", "install", "-e", ".", "--quiet"],
        cwd=repo_dir,
        timeout_s=timeout_s,
    )
    if result.returncode != 0:
        logger.debug(
            "pip install -e . failed (continuing): %s",
            result.stderr.strip()[:200],
        )
    return result


def _run_pytest(
    repo_dir: Path,
    node_ids: list[str],
    timeout_s: int,
) -> _CmdResult:
    """Run pytest against the given node ids, quiet mode."""
    if not node_ids:
        return _CmdResult(returncode=0, stdout="", stderr="", timed_out=False)
    return _run(
        [sys.executable, "-m", "pytest", "-q", "--no-header", "-x", *node_ids],
        cwd=repo_dir,
        timeout_s=timeout_s,
    )


# Expose the shutil import so it is not flagged as unused by ruff when the
# scoring path is exercised entirely via tests that monkeypatch helpers.
_ = shutil
