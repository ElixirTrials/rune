"""Issue #52 — E2 directionality dataset: builder / loader + construction docs.

Loads tools/_e2_counterfactuals.json and emits per-episode records shaped to
drop into the E2 harness (third_party/doc-to-lora/rune_episode_recall.py). NO
model is loaded; this is pure data. CPU-only, import-safe (stdlib only).

DESIGN (FROZEN spec docs/issue52-predeclared-spec-T0-E1-E2-2026-06-02.md, §E2)
--------------------------------------------------------------------------
Per episode the harness conditions on a ctx (a doc) and scores ONE fixed answer
span — the NEXT-STEP action/code the *matched* direction implies — under four
arms:
  matched        = own doc            -> ctxs[i]
  counterfactual = direction flipped  -> REPLACES the structured negative
                                          ctxs[(i+1)%len] at rune_episode_recall.py:94
  control        = same bag of events, neutral reorder, NO flip -> sibling ctx
  zero           = base, no adapter
  ceiling        = doc-in-prompt, no adapter (already in the harness)

Validity invariant (the load-bearing property; verified structurally by
``check_episode`` below): the scored answer is the next edit under the MATCHED
direction; under the COUNTERFACTUAL direction a DIFFERENT edit is correct
(documented per episode as ``counterfactual_next_action``), so the scored span
is WRONG under the flip. Episodes that fail this are not scoreable as
directionality and are excluded by construction.

Discriminator (spec §E2): ``control_lp - counterfactual_lp > 0``. The control
absorbs lexical/positional overlap with matched, so the residual gap to the
counterfactual is the causal flip rather than lexical drift. ``matched_lp -
counterfactual_lp`` is the weaker, lexically-confounded contrast and is reported
only as a secondary number. The action-binding score is read as a fraction of
(ceiling_lp - zero_lp) per the spec ceiling arm.

FORBIDDEN (spec §E2): bare time-reversal; were<->heading text swaps. Every flip
here lives in STATE/ROLE semantics (what currently exists vs is missing; which
file changed first vs is stale; which side leads) distributed across the doc so
no single sentinel token reveals the direction. counterfactual_doc and
control_doc are authored to be ~equidistant lexically from matched_doc.

Provenance: all episodes are SYNTHETIC content — the provable
direction-flips-action property requires authored content. Their doc STRUCTURE
(## Task / ## Current Code / ## Review Feedback) is adapted from real
external_codereview rows (/tmp/rune-corpus/external_codereview.val.clean.jsonl)
and tools/d2l_control/episodes.py::_SYNTHETIC_ROWS, so the doc shape matches the
episodes the trained adapter saw. ``synthetic`` is True for every row.

Run (CPU; prints a construction report, loads no model):
  uv run python tools/_e2_build.py
  uv run python tools/_e2_build.py --json    # machine-readable summary
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

DATA_PATH = Path(__file__).with_name("_e2_counterfactuals.json")

# Cap from the spec (§E2): MAX_ANS_TOK=48 on the scored span. We have no
# tokenizer on the CPU box, so flag items whose answer is long enough to risk
# truncation by a conservative whitespace-token proxy (real check is on GPU).
MAX_ANS_TOK = 48
_TRUNC_WARN_WORDS = 40  # ~ proxy threshold; whitespace tokens << BPE tokens


@dataclass(frozen=True)
class E2Episode:
    """One E2 directionality episode, harness-ready.

    ``doc`` mirrors RuneEpisode.doc (the matched conditioning doc). ``queries``
    mirrors the RuneEpisode shape: a single 'next_action' target whose ``answer``
    is the scored span. ``counterfactual_doc`` / ``control_doc`` are the two
    extra ctxs the E2 arms condition on.
    """

    id: str
    synthetic: bool
    is_positive_control: bool
    adapted_from_real: bool
    doc: str  # matched conditioning doc (== RuneEpisode.doc)
    counterfactual_doc: str
    control_doc: str
    queries: dict[str, dict[str, str]]  # {"next_action": {"query","answer"}}
    counterfactual_next_action: str
    scored_span: str
    notes: str
    meta: dict[str, object] = field(default_factory=dict)

    @property
    def answer(self) -> str:
        return self.queries["next_action"]["answer"]


def _word_count(s: str) -> int:
    return len(s.split())


# Stopwords / scaffold tokens that are not "distinguishing" — their presence in
# the counterfactual doc is harmless. Everything else in the scored answer that
# is also in the counterfactual doc is a priming leak (see check_episode #5).
_NONDISTINGUISHING = frozenset(
    {
        "the",
        "a",
        "an",
        "to",
        "in",
        "of",
        "and",
        "or",
        "is",
        "be",
        "add",
        "edit",
        "fill",
        "write",
        "return",
        "def",
        "if",
        "not",
        "with",
        "for",
        "self",
        "true",
        "false",
        "none",
    }
)


def _tokens(s: str) -> set[str]:
    """Lowercased alnum/underscore/dot tokens (identifiers, paths, literals)."""
    import re

    return {t for t in re.split(r"[^\w.]+", s.lower()) if t}


def check_episode(ep: dict) -> list[str]:
    """Structural validity checks (no model). Returns a list of problems; empty
    list == the episode satisfies the construction contract.
    """
    problems: list[str] = []
    req = (
        "id",
        "query",
        "matched_doc",
        "counterfactual_doc",
        "control_doc",
        "answer_next_action",
        "counterfactual_next_action",
        "scored_span",
    )
    for k in req:
        if not str(ep.get(k, "")).strip():
            problems.append(f"missing/empty field: {k}")
    if problems:
        return problems

    # 1) The flip must change the correct next action: the matched answer and the
    #    counterfactual's correct action must DIFFER (else it tests recall, not
    #    direction). Structural proxy for the GPU-checked invariant.
    if ep["answer_next_action"].strip() == ep["counterfactual_next_action"].strip():
        problems.append(
            "answer_next_action == counterfactual_next_action: flip does not "
            "change the correct next action (recall, not direction)"
        )

    # 2) The three docs must be distinct (a counterfactual/control identical to
    #    matched cannot separate direction from lexical overlap).
    docs = {
        "matched": ep["matched_doc"],
        "counterfactual": ep["counterfactual_doc"],
        "control": ep["control_doc"],
    }
    if len({v.strip() for v in docs.values()}) < 3:
        problems.append("matched/counterfactual/control docs are not all distinct")

    # 3) Lexical-equidistance sanity: control and counterfactual should be
    #    roughly equidistant (in token-set Jaccard) from matched, so the only
    #    thing separating control from counterfactual is the causal flip, not
    #    raw lexical overlap. We only WARN on a large imbalance.
    def jac(a: str, b: str) -> float:
        sa, sb = set(a.split()), set(b.split())
        return len(sa & sb) / max(1, len(sa | sb))

    j_ctrl = jac(ep["matched_doc"], ep["control_doc"])
    j_cf = jac(ep["matched_doc"], ep["counterfactual_doc"])
    if abs(j_ctrl - j_cf) > 0.30:
        problems.append(
            f"WARN lexical imbalance: Jaccard(matched,control)={j_ctrl:.2f} vs "
            f"Jaccard(matched,counterfactual)={j_cf:.2f} (|diff|>0.30); the "
            "control may not absorb the lexical component evenly"
        )

    # 5) PRIMING LEAK (the load-bearing structural proxy for the logprob
    #    asymmetry): the adapter ENCODES facts stated in its conditioning doc and
    #    raises their logprob (the project's own +2.235 goal recall). So if the
    #    counterfactual doc states the scored answer's DISTINGUISHING tokens, the
    #    counterfactual adapter would LIFT the scored span — the wrong direction,
    #    inverting control_lp - counterfactual_lp. Assert the answer's
    #    distinguishing tokens do NOT appear in counterfactual_doc, AND are cued
    #    by matched_doc (matched must support the action more than counterfactual).
    #    Shared SCAFFOLD (a token present in matched_doc too — e.g. the target
    #    file path, which both docs reference by design) is not a leak: matched
    #    supports it at least as much. The leak that inverts the discriminator is
    #    an answer token the COUNTERFACTUAL supplies but the MATCHED doc does NOT
    #    (the counterfactual uniquely primes the scored span). Assert there are
    #    none, and warn if matched cues none of the body tokens at all.
    ans_tokens = _tokens(ep["answer_next_action"]) - _NONDISTINGUISHING
    cf_tokens = _tokens(ep["counterfactual_doc"])
    matched_tokens = _tokens(ep["matched_doc"])
    leaked = sorted(ans_tokens & cf_tokens - matched_tokens)
    if leaked:
        problems.append(
            "priming leak: counterfactual_doc supplies scored-answer tokens "
            f"{leaked} that matched_doc does NOT -> the flipped adapter would "
            "LIFT the scored span (inverts control-counterfactual). De-quote them."
        )
    # Body tokens = answer tokens that are NOT shared scaffold (not in both docs).
    # Cueing may be on the dotted ROOT (matched says 'cache'; answer says
    # 'cache.set') or in prose, so split dotted identifiers to their roots when
    # testing whether matched cues the action.
    def _roots(toks: set[str]) -> set[str]:
        out: set[str] = set()
        for t in toks:
            out.update(t.split("."))
        return out

    body_tokens = ans_tokens - (matched_tokens & cf_tokens)
    if body_tokens and not (_roots(body_tokens) & _roots(matched_tokens)):
        problems.append(
            "WARN matched_doc cues none of the answer's body tokens "
            f"{sorted(body_tokens)}; verify matched supports the action more than "
            "counterfactual (the asymmetry the discriminator relies on)"
        )

    # 4) Truncation proxy on the scored span (real MAX_ANS_TOK=48 check is on GPU).
    wc = _word_count(ep["answer_next_action"])
    if wc > _TRUNC_WARN_WORDS:
        problems.append(
            f"WARN answer_next_action ~{wc} whitespace-words; verify <= "
            f"{MAX_ANS_TOK} BPE tokens on GPU (set truncation_flag)"
        )

    return problems


def load_e2_episodes(path: str | Path = DATA_PATH) -> list[E2Episode]:
    """Load + validate the E2 episodes. Raises on a hard (non-WARN) problem so a
    malformed dataset can never reach the GPU arm silently.
    """
    raw = json.loads(Path(path).read_text())
    out: list[E2Episode] = []
    for ep in raw["episodes"]:
        problems = check_episode(ep)
        hard = [p for p in problems if not p.startswith("WARN")]
        if hard:
            raise ValueError(f"episode {ep.get('id')!r} invalid: {hard}")
        out.append(
            E2Episode(
                id=ep["id"],
                synthetic=bool(ep.get("synthetic", True)),
                is_positive_control=bool(ep.get("is_positive_control", False)),
                adapted_from_real=bool(ep.get("adapted_from_real", False)),
                doc=ep["matched_doc"],
                counterfactual_doc=ep["counterfactual_doc"],
                control_doc=ep["control_doc"],
                queries={
                    "next_action": {
                        "query": ep["query"],
                        "answer": ep["answer_next_action"],
                    }
                },
                counterfactual_next_action=ep["counterfactual_next_action"],
                scored_span=ep["scored_span"],
                notes=ep.get("notes", ""),
                meta={"warnings": [p for p in problems if p.startswith("WARN")]},
            )
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DATA_PATH)
    ap.add_argument("--json", action="store_true", help="machine-readable summary")
    args = ap.parse_args()

    eps = load_e2_episodes(args.data)
    n_pc = sum(1 for e in eps if e.is_positive_control)
    n_syn = sum(1 for e in eps if e.synthetic)

    if args.json:
        print(
            json.dumps(
                {
                    "n_episodes": len(eps),
                    "n_positive_control": n_pc,
                    "n_synthetic": n_syn,
                    "n_adapted_from_real": sum(1 for e in eps if e.adapted_from_real),
                    "episodes": [
                        {
                            "id": e.id,
                            "is_positive_control": e.is_positive_control,
                            "synthetic": e.synthetic,
                            "n_words_answer": _word_count(e.answer),
                            "warnings": e.meta.get("warnings", []),
                        }
                        for e in eps
                    ],
                },
                indent=2,
            )
        )
        return 0

    print(f"E2 directionality dataset: {len(eps)} episodes", flush=True)
    print(
        f"  positive-control={n_pc}  synthetic={n_syn}  "
        f"adapted_from_real={sum(1 for e in eps if e.adapted_from_real)}",
        flush=True,
    )
    print(
        "  scored span = next-step action/code (matched direction); "
        "discriminator = control_lp - counterfactual_lp > 0",
        flush=True,
    )
    for e in eps:
        tag = "  [POSITIVE CONTROL]" if e.is_positive_control else ""
        print(f"\n- {e.id}{tag}", flush=True)
        print(f"    answer ({_word_count(e.answer)} ws-words): {e.answer!r}", flush=True)
        print(f"    counterfactual next action: {e.counterfactual_next_action!r}", flush=True)
        warns = e.meta.get("warnings", [])
        if warns:
            for w in warns:
                print(f"    {w}", flush=True)
    print(
        "\nLoader emits E2Episode objects: .doc -> matched ctxs[i]; "
        ".counterfactual_doc -> replace ctxs[(i+1)%len] (rune_episode_recall.py:94); "
        ".control_doc -> sibling ctx for the matched-vs-control contrast.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
