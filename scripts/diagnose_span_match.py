"""Span-match failure diagnostic tool.

Classifies every ``_find_post_in_span`` failure into one of seven
mutually-exclusive buckets (TRUNCATION_FRONT, TRUNCATION_TAIL,
BPE_DRIFT_START, BPE_DRIFT_END, BPE_DRIFT_BOTH, WRONG_TURN_LOOKUP,
CONTENT_MISMATCH) and writes per-failure records to a JSON file.

Usage::

    uv run python scripts/diagnose_span_match.py \\
        --dataset data/github-pairs/_merged/pairs_all.jsonl \\
        --max-length 3072

Writes ``artifacts/span_match_diagnosis.json`` and prints the aggregate
histogram to stdout.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Path bootstrap — libs are not installed as packages in this script context.
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "libs" / "model-training" / "src"))
sys.path.insert(0, str(_ROOT / "libs" / "shared" / "src"))

from model_training.d2l_data import pairs_to_chat_messages  # noqa: E402
from model_training.diff_loss import (  # noqa: E402
    IGNORE_INDEX,
    _find_post_in_span,
    _iter_assistant_spans,
)
from model_training.model_configs import ModelRegistry  # noqa: E402
from model_training.span_match_classifier import (  # noqa: E402
    FailureBucket,
    classify_failure,
)
from model_training.trajectory import compute_assistant_masks  # noqa: E402

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Diagnose _find_post_in_span failures across a dataset."
    )
    p.add_argument(
        "--dataset",
        default="data/github-pairs/_merged/pairs_all.jsonl",
        help="Path to the JSONL pairs dataset.",
    )
    p.add_argument(
        "--max-length",
        type=int,
        default=3072,
        dest="max_length",
        help="Sequence length cap passed to compute_assistant_masks.",
    )
    p.add_argument(
        "--truncation-mode",
        choices=["keep_start", "keep_end"],
        default="keep_end",
        dest="truncation_mode",
        help="Truncation mode (default: keep_end, matching the diff-aware trainer).",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of conversations to process (for fast smoke).",
    )
    p.add_argument(
        "--output",
        default="artifacts/span_match_diagnosis.json",
        help="Path for the per-failure JSON output.",
    )
    p.add_argument(
        "--model",
        default="qwen3.5-9b",
        help="Model key in ModelRegistry (default: qwen3.5-9b).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:  # noqa: C901 (complexity acceptable for a diagnostic script)
    """Run the span-match failure diagnosis and write the output JSON."""
    args = _parse_args()

    # Resolve tokeniser from model registry.
    registry = ModelRegistry.default()
    model_cfg = registry.get(args.model)
    from transformers import AutoTokenizer  # noqa: PLC0415 — GPU-free, deferred for CI

    print(f"Loading tokenizer: {model_cfg.model_id}", flush=True)
    tok = AutoTokenizer.from_pretrained(model_cfg.model_id)
    print(f"Loaded tokenizer: {model_cfg.model_id}", flush=True)

    # Load dataset.
    dataset_path = Path(args.dataset)
    records = [
        json.loads(line)
        for line in dataset_path.read_text().splitlines()
        if line.strip()
    ]
    print(f"Loaded {len(records)} records from {dataset_path}", flush=True)

    # Build chat conversations with per-turn pre/post codes.
    convs, pre_post = pairs_to_chat_messages(records, mode="multi_turn")
    n_convs = len(convs)
    print(f"multi_turn: {n_convs} conversations", flush=True)

    if args.limit is not None:
        convs = convs[: args.limit]
        pre_post = pre_post[: args.limit]
        print(f"  (limited to {args.limit} conversations)", flush=True)

    # Process conversations.
    total_spans = 0
    failures: list[dict] = []
    bucket_counts: dict[str, int] = {b.value: 0 for b in FailureBucket}

    for conv_idx, (messages, pp) in enumerate(zip(convs, pre_post)):
        pre_codes: list[str] = pp.get("pre_codes", [])
        post_codes: list[str] = pp.get("post_codes", [])

        result = compute_assistant_masks(
            tok,
            messages,
            max_length=args.max_length,
            truncation_mode=args.truncation_mode,
        )
        input_ids: list[int] = result["input_ids"]
        assistant_masks: list[int] = result["assistant_masks"]
        labels = [
            tid if m else IGNORE_INDEX for tid, m in zip(input_ids, assistant_masks)
        ]
        spans = list(_iter_assistant_spans(labels))

        # Mirror _weights_via_hunk_path offset semantics.
        n_turns = max(len(pre_codes), len(post_codes))
        if args.truncation_mode == "keep_end" and n_turns >= len(spans):
            offset = n_turns - len(spans)
        else:
            offset = 0

        # Pre-tokenise all post_codes for WRONG_TURN_LOOKUP.
        all_post_ids: list[list[int]] = [
            list(tok(pc, add_special_tokens=False)["input_ids"]) if pc else []
            for pc in post_codes
        ]

        for span_idx, (span_start, span_end) in enumerate(spans):
            turn_idx = offset + span_idx
            if turn_idx >= len(post_codes):
                continue
            post = post_codes[turn_idx]
            if not post:
                continue

            total_spans += 1
            post_ids: list[int] = all_post_ids[turn_idx]
            match_pos = _find_post_in_span(input_ids, span_start, span_end, post_ids)

            if match_pos >= 0:
                # Successful match — not a failure.
                continue

            span_ids = input_ids[span_start:span_end]
            cls_result = classify_failure(
                span_ids=span_ids,
                post_ids=post_ids,
                span_start=span_start,
                span_end=span_end,
                input_ids=input_ids,
                all_post_ids_lists=all_post_ids,
                turn_idx=turn_idx,
                conv_idx=conv_idx,
                span_idx=span_idx,
            )
            bucket_counts[cls_result.bucket.value] += 1

            # Build decoded diagnostic text.
            head4_post = (
                tok.decode(cls_result.post_ids_head) if cls_result.post_ids_head else ""
            )
            head4_span = (
                tok.decode(cls_result.span_ids_head) if cls_result.span_ids_head else ""
            )
            tail4_post = (
                tok.decode(cls_result.post_ids_tail) if cls_result.post_ids_tail else ""
            )
            tail4_span = (
                tok.decode(cls_result.span_ids_tail) if cls_result.span_ids_tail else ""
            )

            rec = asdict(cls_result)
            rec["post_ids_head_text"] = head4_post
            rec["span_ids_head_text"] = head4_span
            rec["post_ids_tail_text"] = tail4_post
            rec["span_ids_tail_text"] = tail4_span
            # Convert enum to string for JSON serialisation.
            rec["bucket"] = cls_result.bucket.value
            failures.append(rec)

    # Write output.
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(failures, indent=2))
    print(f"\nWrote {len(failures)} failure records to {out_path}", flush=True)

    # Print aggregate report.
    n_fail = len(failures)
    fail_pct = 100.0 * n_fail / total_spans if total_spans > 0 else 0.0
    print(
        f"\n=== SPAN-MATCH FAILURE BREAKDOWN ===\n"
        f"Total spans:          {total_spans}\n"
        f"Total failures:       {n_fail}  ({n_fail}/{total_spans} = {fail_pct:.1f}%)"
    )
    for bucket in FailureBucket:
        cnt = bucket_counts[bucket.value]
        pct = 100.0 * cnt / n_fail if n_fail > 0 else 0.0
        print(f"  {bucket.value:<22} {cnt}  ({pct:.1f}%)")


if __name__ == "__main__":
    main()
