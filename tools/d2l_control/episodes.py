"""Episode datasets for the Doc2LoRA positive control (spec §5 parts 2-3, §9).

Two datasets, both CPU-only / import-safe (no GPU, no model load):

  DOC_FACT_EPISODES  — small synthetic needle-in-a-haystack set for PROBE
    VALIDATION. Each episode is {doc, query, answer_fact}: an unguessable
    fact buried in a doc, and a query whose gold answer IS that fact. Used
    to confirm the scorecard detects recall when it genuinely exists.

  build_rune_episodes(corpus_path, n) — reformulate a handful of
    external_codereview rows into tiny patch+QA episodes (spec §9 default:
    reformulate real rows, doubling as an early read on the patch
    reformulation the data track needs). Each episode is
    {doc, queries: {goal, file, diff}, source} where every query carries an
    answer span that is an EXACT slice of `source` (provenance), so the
    "no fact that isn't present" invariant holds by construction and is
    checkable from the episode object alone (corpus may be absent in CI).

The Rune-episode `doc` is the COMPACT patch reformulation: the Task header
(carrying the file path) + the pre-edit code + the review feedback — i.e.
roughly the activation_text. It deliberately does NOT include post_code /
the full revised file; the diff is the recovery TARGET, not given in the doc.
"""

from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_CORPUS = "/tmp/rune-corpus/external_codereview.val.clean.jsonl"


# --------------------------------------------------------------------------
# Doc-fact episodes (probe validation): synthetic NIAH/QA literals.
# answer_fact is an exact substring of doc, so the probe has a present needle.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class DocFactEpisode:
    doc: str
    query: str
    answer_fact: str


def _haystack(needle_sentence: str) -> str:
    """Wrap a needle sentence in unrelated filler so the fact is unguessable."""
    pre = (
        "The committee met on a Tuesday to review the quarterly logistics report. "
        "Several unrelated matters concerning catering and parking were discussed. "
    )
    post = (
        " Afterwards the minutes were filed and the meeting was adjourned without "
        "objection. The next session was tentatively scheduled for the spring."
    )
    return pre + needle_sentence + post


# Each fact is unguessable (random codes / names) and appears verbatim as a
# substring of its doc; the query's gold answer is exactly that span.
DOC_FACT_EPISODES: tuple[DocFactEpisode, ...] = (
    DocFactEpisode(
        doc=_haystack("The vault access code for building 7 is QX-4417-ZD."),
        query="What is the vault access code for building 7?",
        answer_fact="QX-4417-ZD",
    ),
    DocFactEpisode(
        doc=_haystack("Dr. Marisol Venn was appointed lead auditor for the project."),
        query="Who was appointed lead auditor for the project?",
        answer_fact="Dr. Marisol Venn",
    ),
    DocFactEpisode(
        doc=_haystack("The shipment was routed through the port of Valparaiso."),
        query="Through which port was the shipment routed?",
        answer_fact="Valparaiso",
    ),
    DocFactEpisode(
        doc=_haystack("Server node KR-09 holds the primary replica of the ledger."),
        query="Which server node holds the primary replica of the ledger?",
        answer_fact="KR-09",
    ),
    DocFactEpisode(
        doc=_haystack("The encryption rotation interval was set to 37 days."),
        query="What was the encryption rotation interval set to?",
        answer_fact="37 days",
    ),
    DocFactEpisode(
        doc=_haystack("The reagent batch was labelled with lot number ZTH-88201."),
        query="What lot number was the reagent batch labelled with?",
        answer_fact="ZTH-88201",
    ),
    DocFactEpisode(
        doc=_haystack("The fallback DNS resolver was configured at 198.51.100.42."),
        query="What address was the fallback DNS resolver configured at?",
        answer_fact="198.51.100.42",
    ),
    DocFactEpisode(
        doc=_haystack("Operative codename Halcyon was assigned to the Lisbon cell."),
        query="What codename was assigned to the Lisbon cell?",
        answer_fact="Halcyon",
    ),
    DocFactEpisode(
        doc=_haystack("The warranty on unit 14 expires on 2031-08-19."),
        query="When does the warranty on unit 14 expire?",
        answer_fact="2031-08-19",
    ),
    DocFactEpisode(
        doc=_haystack("The signing key fingerprint ends in F2:9A:CC:01."),
        query="What does the signing key fingerprint end in?",
        answer_fact="F2:9A:CC:01",
    ),
)


# --------------------------------------------------------------------------
# Pure extraction helpers (TDD'd: spec §8 — schema/shape + extraction).
# --------------------------------------------------------------------------

_FILE_RE = re.compile(r"file:\s*([^)\n]+)\)")


def extract_file_path(activation_text: str) -> str:
    """Pull the file path from the `## Task` header line.

    The header reads `... (PR #N, file: PATH)`; the path lives ONLY there
    (not in pre_code or feedback). Returns "" if no match.
    """
    m = _FILE_RE.search(activation_text)
    return m.group(1).strip() if m else ""


def extract_review_feedback(activation_text: str) -> str:
    """Return the text under the `## Review Feedback` heading, verbatim.

    Exact slice (no normalization) so it stays a substring of the source.
    """
    marker = "## Review Feedback"
    if marker not in activation_text:
        return ""
    return activation_text.split(marker, 1)[1].lstrip("\n").strip()


def extract_diff_hunk(pre_code: str, post_code: str) -> str:
    """The post-side text of the first changed (replace/insert) line region.

    Line-level (not char-level) so a "hunk" is a clean block of lines. We take
    the FIRST opcode that introduces post-side lines — a `replace` or `insert`
    whose post range is non-empty — and return those post lines joined by "\\n".
    The result is by construction a slice of post_code's lines, so it is a
    substring of post_code (modulo the join — see _hunk_in_source). Returns ""
    if pre and post are identical or only delete lines.
    """
    pre_lines = pre_code.splitlines()
    post_lines = post_code.splitlines()
    sm = difflib.SequenceMatcher(None, pre_lines, post_lines)
    for tag, _i1, _i2, j1, j2 in sm.get_opcodes():
        if tag in ("replace", "insert") and j2 > j1:
            return "\n".join(post_lines[j1:j2])
    return ""


# --------------------------------------------------------------------------
# Rune episodes.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class RuneEpisode:
    doc: str
    # queries: name -> {"answer": span, ...}. answer is an exact slice of source.
    queries: dict[str, dict[str, str]]
    source: str  # provenance: every answer span is a substring of this
    task_id: str = ""
    meta: dict[str, object] = field(default_factory=dict)


def _build_episode_from_row(raw: dict[str, object]) -> RuneEpisode | None:
    """Reformulate one corpus row into a tiny patch+QA episode, or None.

    None when the row does not carry all three queried facts (file, feedback,
    diff hunk) — skipping those avoids the data-artifact failure §9 warns of.
    """
    activation_text = str(raw.get("activation_text", ""))
    pre_code = str(raw.get("pre_code", ""))
    post_code = str(raw.get("post_code", ""))

    file_path = extract_file_path(activation_text)
    feedback = extract_review_feedback(activation_text)
    diff_hunk = extract_diff_hunk(pre_code, post_code)
    if not (file_path and feedback and diff_hunk):
        return None

    # Compact doc = the patch reformulation: Task header (carries file path) +
    # pre-edit code + feedback. NOT post_code / the full revised file.
    doc = activation_text

    # Provenance: every answer span must be an exact substring of `source`.
    # file + feedback live in activation_text; the diff hunk lives in post_code.
    source = activation_text + "\n" + post_code

    queries = {
        "goal": {"query": "What change does the reviewer request?", "answer": feedback},
        "file": {"query": "Which file is being edited?", "answer": file_path},
        "diff": {"query": "What is the edited code?", "answer": diff_hunk},
    }
    return RuneEpisode(
        doc=doc,
        queries=queries,
        source=source,
        task_id=str(raw.get("task_id", "")),
        meta={"quality_score": raw.get("quality_score")},
    )


# Committed fallback: tiny synthetic Rune-style episodes, used when the corpus
# is absent (CPU CI). Hand-built so file/feedback live in the doc and the diff
# hunk is the post-side of a real line change. Same schema + provenance as
# corpus-derived episodes.
_SYNTHETIC_ROWS: tuple[dict[str, str], ...] = (
    {
        "task_id": "synthetic_0",
        "activation_text": (
            "## Task\n"
            "Review and revise code from acme/widgets "
            "(PR #1, file: src/widgets/parser.py)\n\n"
            "## Current Code\n"
            "def parse(line):\n"
            "    parts = line.split(',')\n"
            "    return parts[0]\n\n"
            "## Review Feedback\n"
            "Strip whitespace from the field before returning it."
        ),
        "pre_code": (
            "def parse(line):\n    parts = line.split(',')\n    return parts[0]\n"
        ),
        "post_code": (
            "def parse(line):\n    parts = line.split(',')\n"
            "    return parts[0].strip()\n"
        ),
    },
    {
        "task_id": "synthetic_1",
        "activation_text": (
            "## Task\n"
            "Review and revise code from acme/server "
            "(PR #2, file: server/handlers.py)\n\n"
            "## Current Code\n"
            "def handle(req):\n"
            "    data = req.json\n"
            "    return data\n\n"
            "## Review Feedback\n"
            "Add a None check before accessing req.json."
        ),
        "pre_code": "def handle(req):\n    data = req.json\n    return data\n",
        "post_code": (
            "def handle(req):\n"
            "    if req is None:\n"
            "        return None\n"
            "    data = req.json\n"
            "    return data\n"
        ),
    },
    {
        "task_id": "synthetic_2",
        "activation_text": (
            "## Task\n"
            "Review and revise code from acme/utils "
            "(PR #3, file: utils/math_helpers.py)\n\n"
            "## Current Code\n"
            "def average(xs):\n"
            "    return sum(xs) / len(xs)\n\n"
            "## Review Feedback\n"
            "Guard against an empty list to avoid a division by zero."
        ),
        "pre_code": "def average(xs):\n    return sum(xs) / len(xs)\n",
        "post_code": (
            "def average(xs):\n"
            "    if not xs:\n"
            "        return 0.0\n"
            "    return sum(xs) / len(xs)\n"
        ),
    },
)


def _synthetic_rune_episodes() -> list[RuneEpisode]:
    eps: list[RuneEpisode] = []
    for row in _SYNTHETIC_ROWS:
        ep = _build_episode_from_row(dict(row))
        if ep is not None:
            eps.append(ep)
    return eps


def build_rune_episodes(
    corpus_path: str | Path = DEFAULT_CORPUS, n: int = 12
) -> list[RuneEpisode]:
    """Build up to `n` Rune patch+QA episodes from the corpus.

    Falls back to the committed synthetic set when `corpus_path` is absent, so
    this module is import-safe and the loader works in CPU CI. Returns the
    first `n` VALID rows (those carrying file + feedback + diff hunk); the
    actual count may be < n if rows get filtered.
    """
    path = Path(corpus_path)
    if not path.exists():
        return _synthetic_rune_episodes()[:n]

    episodes: list[RuneEpisode] = []
    with path.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            ep = _build_episode_from_row(json.loads(line))
            if ep is not None:
                episodes.append(ep)
            if len(episodes) >= n:
                break
    return episodes


def _hunk_in_source(answer: str, source: str) -> bool:
    """Diff-hunk containment: the join newline may not match source line seps.

    extract_diff_hunk joins post lines with "\\n"; source may use "\\r\\n". A
    multi-line hunk is present iff each of its lines is present in source.
    """
    return all(part in source for part in answer.split("\n"))
