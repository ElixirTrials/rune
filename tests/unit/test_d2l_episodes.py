"""Schema/shape + fact-presence + extraction tests for the d2l control episodes.

Spec §8: TDD the deterministic pieces — episode dataset builder (schema/shape),
extraction (file path, diff hunk). The load-bearing invariant: every queried
answer text actually occurs in its source (no fact that isn't present), checked
from the episode object alone so the test survives CI without the corpus.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

# Load the module by path (tools/ is not a package). Register it in sys.modules
# before exec so dataclass field-type resolution can find the module.
_MOD_PATH = Path(__file__).resolve().parents[2] / "tools" / "d2l_control" / "episodes.py"
_spec = importlib.util.spec_from_file_location("d2l_episodes", _MOD_PATH)
assert _spec is not None and _spec.loader is not None
episodes = importlib.util.module_from_spec(_spec)
sys.modules["d2l_episodes"] = episodes
_spec.loader.exec_module(episodes)


# --------------------------------------------------------------------------
# Doc-fact episodes (probe validation set).
# --------------------------------------------------------------------------

def test_doc_fact_count_in_range() -> None:
    n = len(episodes.DOC_FACT_EPISODES)
    assert 8 <= n <= 12, n


def test_doc_fact_schema_and_fact_present() -> None:
    for ep in episodes.DOC_FACT_EPISODES:
        assert ep.doc and ep.query and ep.answer_fact
        # The needle must actually be in the haystack.
        assert ep.answer_fact in ep.doc
        # The answer must not be trivially restated in the query (needle is
        # unguessable / not handed to the model in the question).
        assert ep.answer_fact not in ep.query


# --------------------------------------------------------------------------
# Pure extraction (hand-built pairs).
# --------------------------------------------------------------------------

def test_extract_file_path() -> None:
    at = (
        "## Task\nReview and revise code from acme/widgets "
        "(PR #7, file: src/foo/bar.py)\n\n## Current Code\nx = 1\n"
    )
    assert episodes.extract_file_path(at) == "src/foo/bar.py"


def test_extract_file_path_absent() -> None:
    assert episodes.extract_file_path("## Task\nno file here\n") == ""


def test_extract_review_feedback() -> None:
    at = "## Task\nt\n\n## Current Code\nx=1\n\n## Review Feedback\nFix the bug."
    assert episodes.extract_review_feedback(at) == "Fix the bug."


def test_extract_diff_hunk_replace() -> None:
    pre = "a = 1\nb = 2\nc = 3\n"
    post = "a = 1\nb = 22\nc = 3\n"
    assert episodes.extract_diff_hunk(pre, post) == "b = 22"


def test_extract_diff_hunk_insert() -> None:
    pre = "a = 1\nc = 3\n"
    post = "a = 1\nb = 2\nc = 3\n"
    assert episodes.extract_diff_hunk(pre, post) == "b = 2"


def test_extract_diff_hunk_multiline() -> None:
    pre = "def f():\n    return 1\n"
    post = "def f():\n    if True:\n        return 2\n    return 1\n"
    hunk = episodes.extract_diff_hunk(pre, post)
    # Post-side of the first changed region; each line is present in post.
    assert hunk
    for line in hunk.split("\n"):
        assert line in post


def test_extract_diff_hunk_identical_is_empty() -> None:
    assert episodes.extract_diff_hunk("a = 1\n", "a = 1\n") == ""


def test_extract_diff_hunk_pure_delete_is_empty() -> None:
    # Only deletions, no post-side lines introduced.
    assert episodes.extract_diff_hunk("a\nb\nc\n", "a\nc\n") == ""


# --------------------------------------------------------------------------
# Rune episodes: schema/shape + every answer present in its source.
# --------------------------------------------------------------------------

_RUNE_PARAMS = [
    ("synthetic (fallback)", episodes.build_rune_episodes("/nonexistent/path.jsonl", 12)),
    ("corpus-or-fallback", episodes.build_rune_episodes(episodes.DEFAULT_CORPUS, 12)),
]


def test_fallback_used_when_corpus_absent() -> None:
    eps = episodes.build_rune_episodes("/nonexistent/path.jsonl", 12)
    assert len(eps) >= 3  # the committed synthetic set


def test_rune_episode_schema_and_facts_present() -> None:
    for label, eps in _RUNE_PARAMS:
        assert eps, f"{label}: no episodes built"
        for ep in eps:
            # Schema/shape.
            assert ep.doc, label
            assert set(ep.queries) == {"goal", "file", "diff"}, label
            assert ep.source, label
            for name, q in ep.queries.items():
                assert q["query"] and q["answer"], f"{label}:{name}"
                ans = q["answer"]
                # Every queried answer text actually occurs in the source.
                if name == "diff":
                    assert episodes._hunk_in_source(ans, ep.source), f"{label}:{name}"
                else:
                    assert ans in ep.source, f"{label}:{name}"
            # goal + file ARE carried by the compact doc (queries answerable
            # from the doc alone, per the patch reformulation).
            assert ep.queries["goal"]["answer"] in ep.doc, label
            assert ep.queries["file"]["answer"] in ep.doc, label
            # The diff is the recovery TARGET (it lives in post_code). The
            # compact doc does NOT append post_code, so the diff is not handed
            # to the model via the appended revision — though a row whose review
            # feedback quotes a code-suggestion block may legitimately echo it.
            assert episodes._hunk_in_source(
                ep.queries["diff"]["answer"], ep.source
            ), label


def test_build_respects_n() -> None:
    eps = episodes.build_rune_episodes(episodes.DEFAULT_CORPUS, 5)
    assert len(eps) <= 5


def test_synthetic_diff_is_held_out_of_doc() -> None:
    # On the hand-built fallback set (full control), the diff recovery target is
    # genuinely absent from the doc — the doc is pre_code + feedback only.
    eps = episodes.build_rune_episodes("/nonexistent/path.jsonl", 12)
    for ep in eps:
        assert ep.queries["diff"]["answer"] not in ep.doc
