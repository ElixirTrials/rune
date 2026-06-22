"""RepoBench v1.1 (Python) loader + adapter-conditioning templates (issue #52).

Cross-file next-line completion: the long-context probe for the
adapter-as-unbounded-context thesis (PRODUCT.md current bet / JTBD #3). The
hypernetwork compresses the cross-file ``context`` into a constant-length LoRA
adapter (arm A3); the arms differ only in how that context is delivered. The
heavy ``datasets`` dependency imports inside the loader so the module stays
CPU-importable (PRODUCT.md invariant 2).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rune.bench.identifier_match import extract_identifiers

_SIG_RE = re.compile(r"^\s*(class|def|async def)\s")

_HF_DATASET = "tianyang/repobench_python_v1.1"
_CROSS_FILE_FIRST = "cross_file_first"
_ANCHOR_CHARS = 600  # in-file tail kept as the `## Current Code` anchor (training mode)


@dataclass(frozen=True)
class RepoBenchRow:
    """One cross-file completion example (the RepoBench fields this probe uses)."""

    task_id: str
    cropped_code: str  # in-file prefix up to the cursor
    import_statement: str  # the file's import block (bridges to cross-file names)
    context: tuple[dict[str, str], ...]  # snippets: {identifier, path, snippet}
    gold_snippet_index: int  # index into context of the needed definition
    next_line: str  # ground-truth completion
    level: str  # RepoBench length bucket (e.g. "2k", "8k")
    token_num: int  # cross-file context token count
    repo_name: str
    file_path: str

    @property
    def gold_identifier(self) -> str:
        """Identifier of the gold cross-file snippet (the API the next line needs)."""
        if 0 <= self.gold_snippet_index < len(self.context):
            return self.context[self.gold_snippet_index].get("identifier", "")
        return ""

    @property
    def gold_in_import(self) -> bool:
        """Whether the gold identifier already appears in the import block.

        A confound for bare-identifier recovery: when the cross-file name is in
        the file's own imports, even the no-context arm can echo it, so EM /
        edit-similarity are the robust discriminators on those rows.
        """
        return self.gold_identifier in set(extract_identifiers(self.import_statement))


def load_repobench_rows(
    *, limit: int = 0, level: str | None = None, split: str = _CROSS_FILE_FIRST
) -> list[RepoBenchRow]:
    """Load RepoBench v1.1 Python rows; ``limit`` / ``level`` filter the split."""
    import datasets as hf  # noqa: PLC0415

    ds = hf.load_dataset(_HF_DATASET, split=split)
    rows: list[RepoBenchRow] = []
    for i, r in enumerate(ds):
        if level is not None and r["level"] != level:
            continue
        ctx = tuple(
            {"identifier": c["identifier"], "path": c["path"], "snippet": c["snippet"]}
            for c in r["context"]
        )
        rows.append(
            RepoBenchRow(
                task_id=f"{split}/{i}",
                cropped_code=r["cropped_code"],
                import_statement=r["import_statement"],
                context=ctx,
                gold_snippet_index=int(r["gold_snippet_index"]),
                next_line=r["next_line"],
                level=r["level"],
                token_num=int(r["token_num"]),
                repo_name=r["repo_name"],
                file_path=r["file_path"],
            )
        )
        if limit and len(rows) >= limit:
            break
    return rows


def order_context(
    row: RepoBenchRow, *, gold_first: bool = False
) -> tuple[dict[str, str], ...]:
    """Context snippets, optionally with the gold snippet moved to the front.

    The hypernet truncates conditioning at 2048 tokens (front-kept), so at long
    levels a late gold snippet is evicted from the adapter's view. ``gold_first``
    guarantees the needed definition survives within the budget.
    """
    ctx = row.context
    gi = row.gold_snippet_index
    if gold_first and 0 <= gi < len(ctx):
        return (ctx[gi], *ctx[:gi], *ctx[gi + 1 :])
    return ctx


def render_xfile_adapter(
    row: RepoBenchRow, mode: str = "structured", *, gold_first: bool = False
) -> str:
    """Render cross-file ``context`` into hypernetwork-conditioning text.

    The template-tuning surface (no weight training): ``mode`` controls only the
    formatting of the same content; ``gold_first`` controls snippet ordering.

    - ``raw``: snippets concatenated.
    - ``structured``: per-snippet ``## File: <path>`` headers (mirrors the
      reference-adapter section format in ``engine/graph.py``).
    - ``training``: the hypernet's distillation surface
      (``## Task / ## Current Code / ## Review Feedback``, see
      ``engine/graph.render_training_format_trajectory``) with the cross-file
      definitions as the Task and a short in-file anchor as Current Code, so the
      conditioning is in-distribution for the frozen c3 hypernet.
    """
    ctx = order_context(row, gold_first=gold_first)
    if mode == "raw":
        return "\n\n".join(c["snippet"] for c in ctx)
    structured = "\n\n".join(f"## File: {c['path']}\n{c['snippet']}" for c in ctx)
    if mode == "structured":
        return structured
    if mode == "training":
        task = (
            "Complete the next line of the current file. Relevant cross-file "
            f"definitions:\n\n{structured}"
        )
        anchor = row.cropped_code[-_ANCHOR_CHARS:]
        return f"## Task\n{task}\n\n## Current Code\n{anchor}\n\n## Review Feedback\n"
    raise ValueError(f"unknown render mode {mode!r}")


def render_context_prompt(row: RepoBenchRow) -> str:
    """Cross-file context as a prompt prefix (arm A2: context-in-prompt)."""
    return render_xfile_adapter(row, mode="structured")


def _gold_signature(snippet: str) -> str:
    """Callable signature lines (class/def headers) of a snippet — the API surface."""
    sigs = [ln for ln in snippet.splitlines() if _SIG_RE.match(ln)]
    if sigs:
        return "\n".join(sigs)
    first = next((ln for ln in snippet.splitlines() if ln.strip()), "")
    return first


def render_episodic_adapter(
    row: RepoBenchRow, *, signature_only: bool = False, anchor_chars: int = 400
) -> str:
    """Episodic, per-task adapter conditioning (the corrected template).

    The adapter is episodic and per-task: it needs to know *what this task must
    call*, not the whole repo. This names the single gold cross-file API and its
    definition, in the hypernet's distillation surface
    (``## Task / ## Current Code / ## Review Feedback``) — far smaller than the
    multi-file ``render_xfile_adapter`` dump (which the 2048-token cap shredded).

    ``signature_only`` keeps just the class/def headers (the call surface), not the
    full body.
    """
    gi = row.gold_snippet_index
    if not (0 <= gi < len(row.context)):
        return ""
    g = row.context[gi]
    body = _gold_signature(g["snippet"]) if signature_only else g["snippet"]
    task = (
        "Complete the next line of the current file. It must call "
        f"`{g['identifier']}` defined in {g['path']}:\n\n{body}"
    )
    anchor = row.cropped_code[-anchor_chars:]
    return f"## Task\n{task}\n\n## Current Code\n{anchor}\n\n## Review Feedback\n"
