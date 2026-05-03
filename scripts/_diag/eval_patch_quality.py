"""Patch-applicability + structural-quality evaluator for diff-target SFT.

Addresses advisor critique: token-level cross-entropy on unified-diff strings
does not measure whether the generated patch is a syntactically valid /
applicable diff. Two records with similar CE loss can produce one applicable
patch and one garbage patch.

Three tiers of check (in increasing strictness):

  Tier 1 — syntactic validity (no repo state needed):
    parses_ok      = at least one `--- file ---` and one `@@ ... @@` line,
                      every hunk header's ±count matches actual +/- line counts
    file_count     = number of file headers
    hunk_count     = number of `@@` hunks

  Tier 2 — content overlap with ground truth (no repo state needed):
    hunk_iou        = Jaccard over the SET of (file, +line, -line) triples
                       between generated and ground truth
    char_similarity = Levenshtein ratio on the diff text
    exact_match     = strings equal byte-for-byte

  Tier 3 — `git apply --check` against the source file (deferred; needs the
            parent-commit file state which our corpus does not include).

Usage:
  uv run python scripts/_diag/eval_patch_quality.py \\
      --heldout data/_ab/pairs_heldout_100.jsonl \\
      --adapter ./hpo_artifacts/<run>/adapter \\
      --n-rows 25 --max-new 1024
"""
from __future__ import annotations

import argparse
import json
import re
from difflib import SequenceMatcher
from pathlib import Path

# Unified diff structure regexes.
_FILE_HEADER = re.compile(r"^---\s+(.+?)\s*$", re.MULTILINE)
_FILE_HEADER_PLUS = re.compile(r"^\+\+\+\s+(.+?)\s*$", re.MULTILINE)
_HUNK_HEADER = re.compile(
    r"^@@\s+-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@",
    re.MULTILINE,
)


def parse_diff_structure(text: str) -> dict:
    """Tier 1 + Tier 2 helpers — extract structure from a diff string."""
    file_headers = _FILE_HEADER.findall(text)
    list(_HUNK_HEADER.finditer(text))

    # Walk line-by-line to count + and - per hunk and verify ±count matches.
    lines = text.splitlines()
    hunks: list[dict] = []
    cur: dict | None = None
    for ln in lines:
        if ln.startswith("@@"):
            if cur is not None:
                hunks.append(cur)
            m = _HUNK_HEADER.match(ln)
            if m:
                old_count = int(m.group(2) or "1")
                new_count = int(m.group(4) or "1")
                cur = {
                    "old_count_declared": old_count,
                    "new_count_declared": new_count,
                    "added": [],
                    "removed": [],
                    "context": [],
                }
        elif cur is not None:
            if ln.startswith("+++") or ln.startswith("---"):
                # File header line, ends the current hunk
                hunks.append(cur)
                cur = None
            elif ln.startswith("+"):
                cur["added"].append(ln[1:])
            elif ln.startswith("-"):
                cur["removed"].append(ln[1:])
            elif ln.startswith(" ") or ln == "":
                cur["context"].append(ln[1:] if ln else "")
    if cur is not None:
        hunks.append(cur)

    # Validate ±counts per hunk.
    counts_match = True
    for h in hunks:
        actual_old = len(h["removed"]) + len(h["context"])
        actual_new = len(h["added"]) + len(h["context"])
        if actual_old != h["old_count_declared"] or actual_new != h["new_count_declared"]:
            counts_match = False
            h["counts_ok"] = False
        else:
            h["counts_ok"] = True

    return {
        "n_files": len(file_headers),
        "n_hunks": len(hunks),
        "hunk_counts_match": counts_match,
        "hunks": hunks,
        "files": file_headers,
    }


def syntactic_validity(text: str) -> tuple[bool, dict]:
    """Tier 1: bool + breakdown of which sub-checks failed."""
    s = parse_diff_structure(text)
    has_file = s["n_files"] >= 1
    has_hunk = s["n_hunks"] >= 1
    counts_ok = s["hunk_counts_match"]
    valid = bool(has_file and has_hunk and counts_ok)
    return valid, {
        "has_file_header": has_file,
        "has_hunk": has_hunk,
        "hunk_counts_match": counts_ok,
        "n_files": s["n_files"],
        "n_hunks": s["n_hunks"],
    }


def hunk_iou(gen_text: str, gt_text: str) -> float:
    """Tier 2: Jaccard over (file, +/- line) triples."""
    def signatures(t: str) -> set:
        parse_diff_structure(t)
        sig = set()
        # Pair each hunk with the most recent file header.
        cur_file = ""
        for ln in t.splitlines():
            if ln.startswith("--- "):
                cur_file = ln[4:].strip()
            elif ln.startswith("+") and not ln.startswith("+++"):
                sig.add(("+", cur_file, ln[1:].rstrip()))
            elif ln.startswith("-") and not ln.startswith("---"):
                sig.add(("-", cur_file, ln[1:].rstrip()))
        return sig

    a = signatures(gen_text)
    b = signatures(gt_text)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def char_similarity(gen_text: str, gt_text: str) -> float:
    """Tier 2: SequenceMatcher ratio over chars (cheap; not edit distance proper)."""
    return SequenceMatcher(a=gen_text, b=gt_text, autojunk=False).ratio()


def evaluate_record(generated: str, ground_truth: str) -> dict:
    valid, sub = syntactic_validity(generated)
    iou = hunk_iou(generated, ground_truth)
    char_sim = char_similarity(generated, ground_truth)
    exact = generated.strip() == ground_truth.strip()
    return {
        "syntactic_valid": valid,
        "exact_match": exact,
        "hunk_iou": iou,
        "char_similarity": char_sim,
        **sub,
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--heldout", default="data/_ab/pairs_heldout_100.jsonl")
    p.add_argument("--n-rows", type=int, default=25)
    p.add_argument("--max-new", type=int, default=1024)
    p.add_argument("--adapter", default=None,
                   help="PEFT adapter dir; defaults to deltacoder if omitted.")
    p.add_argument("--deltacoder-id", default="danielcherubini/Qwen3.5-DeltaCoder-9B")
    p.add_argument("--model", default="Qwen/Qwen3.5-9B")
    p.add_argument("--max-prompt-tokens", type=int, default=2000,
                   help="Truncate prompts longer than this (tokens, not chars).")
    p.add_argument("--print-failures", type=int, default=2,
                   help="Print N worst-IOU examples for inspection.")
    p.add_argument(
        "--mlflow-experiment",
        default="rune-eval-patch",
        help="MLflow experiment to log eval metrics under. Set to '' to disable.",
    )
    p.add_argument("--mlflow-uri", default=None)
    return p.parse_args()


def extract_response(teacher: str, prompt: str) -> str:
    """Lift the response (assistant content) from teacher_text minus prompt."""
    if teacher.startswith(prompt):
        return teacher[len(prompt):].lstrip("\n")
    if prompt in teacher:
        return teacher.split(prompt, 1)[1].lstrip("\n")
    return teacher


def main():
    args = parse_args()
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print("Loading model+adapter ...")
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=bnb, dtype=torch.bfloat16,
    )
    adapter_path = args.adapter or args.deltacoder_id
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    rows = [json.loads(line) for line in Path(args.heldout).read_text().splitlines() if line.strip()]
    rows = rows[: args.n_rows]
    print(f"Evaluating {len(rows)} held-out records (adapter={adapter_path[:60]})")

    results = []
    failures: list[dict] = []
    for i, r in enumerate(rows):
        prompt = r["activation_text"]
        gt = extract_response(r["teacher_text"], prompt)
        if not gt.strip():
            continue

        # Build prompt + generation prompt
        msgs = [{"role": "user", "content": prompt}]
        prompt_text = tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
        )
        ids = tok(prompt_text, add_special_tokens=False, return_tensors="pt")
        # Truncate if prompt too long
        if ids["input_ids"].shape[1] > args.max_prompt_tokens:
            ids = {k: v[:, -args.max_prompt_tokens:] for k, v in ids.items()}
        ids = {k: v.to(model.device) for k, v in ids.items()}

        with torch.no_grad():
            out = model.generate(
                **ids,
                max_new_tokens=args.max_new,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tok.pad_token_id,
            )
        gen_full = tok.decode(out[0], skip_special_tokens=True)
        # Trim the prompt prefix from the generated text.
        prompt_decoded = tok.decode(ids["input_ids"][0], skip_special_tokens=True)
        if gen_full.startswith(prompt_decoded):
            gen_resp = gen_full[len(prompt_decoded):].lstrip("\n")
        else:
            # Fall back: strip everything before the first '## ' or '---'
            cut = max(gen_full.find("\n## "), gen_full.find("\n---"))
            gen_resp = gen_full[cut:].lstrip("\n") if cut > 0 else gen_full

        scores = evaluate_record(gen_resp, gt)
        scores["task_id"] = r.get("task_id", "?")
        scores["gen_len_chars"] = len(gen_resp)
        scores["gt_len_chars"] = len(gt)
        results.append(scores)

        if scores["hunk_iou"] < 0.10 and len(failures) < args.print_failures:
            failures.append({
                "task_id": scores["task_id"],
                "gen_preview": gen_resp[:600],
                "gt_preview": gt[:600],
                "scores": scores,
            })

        if (i + 1) % 5 == 0:
            print(
                f"  {i+1}/{len(rows)}: "
                f"valid={scores['syntactic_valid']} "
                f"iou={scores['hunk_iou']:.3f} "
                f"sim={scores['char_similarity']:.3f}"
            )

    n = len(results)
    print(f"\n{'='*60}\nPATCH-QUALITY SUMMARY (n={n})\n{'='*60}")
    if not n:
        print("No results.")
        return

    valid_rate = sum(1 for x in results if x["syntactic_valid"]) / n
    exact_rate = sum(1 for x in results if x["exact_match"]) / n
    has_file_rate = sum(1 for x in results if x["has_file_header"]) / n
    has_hunk_rate = sum(1 for x in results if x["has_hunk"]) / n
    counts_match_rate = sum(1 for x in results if x["hunk_counts_match"]) / n
    mean_iou = sum(x["hunk_iou"] for x in results) / n
    mean_char_sim = sum(x["char_similarity"] for x in results) / n
    median_iou = sorted(x["hunk_iou"] for x in results)[n // 2]

    print(f"  syntactically_valid:  {valid_rate:.1%}")
    print(f"    has_file_header:    {has_file_rate:.1%}")
    print(f"    has_hunk:           {has_hunk_rate:.1%}")
    print(f"    hunk_counts_match:  {counts_match_rate:.1%}")
    print(f"  exact_match:          {exact_rate:.1%}")
    print(f"  hunk_iou       mean={mean_iou:.3f}  median={median_iou:.3f}")
    print(f"  char_similarity mean={mean_char_sim:.3f}")

    for f in failures:
        print(f"\n--- FAILURE EXAMPLE: {f['task_id']} (iou={f['scores']['hunk_iou']:.3f}) ---")
        print("--- generated (first 600 chars) ---")
        print(f["gen_preview"])
        print("--- ground truth (first 600 chars) ---")
        print(f["gt_preview"])

    # MLflow logging — eval metrics live in their own experiment so dashboards
    # can compare adapters without polluting training runs. Skipped on
    # --mlflow-experiment='' or when mlflow is unavailable.
    if args.mlflow_experiment:
        try:
            import mlflow  # noqa: PLC0415

            if args.mlflow_uri:
                mlflow.set_tracking_uri(args.mlflow_uri)
            mlflow.set_experiment(args.mlflow_experiment)
            with mlflow.start_run(run_name=f"patch-{Path(args.adapter or 'deltacoder').name}"):
                mlflow.log_params({
                    "adapter_path": str(args.adapter or args.deltacoder_id),
                    "heldout_path": args.heldout,
                    "n_rows": args.n_rows,
                    "max_new": args.max_new,
                    "deltacoder_id": args.deltacoder_id,
                    "model": args.model,
                })
                mlflow.log_metrics({
                    "eval_patch/syntactic_valid_rate": valid_rate,
                    "eval_patch/has_file_header_rate": has_file_rate,
                    "eval_patch/has_hunk_rate": has_hunk_rate,
                    "eval_patch/hunk_counts_match_rate": counts_match_rate,
                    "eval_patch/exact_match_rate": exact_rate,
                    "eval_patch/hunk_iou_mean": mean_iou,
                    "eval_patch/hunk_iou_median": median_iou,
                    "eval_patch/char_similarity_mean": mean_char_sim,
                    "eval_patch/n_results": n,
                })
            print(f"\nLogged to MLflow experiment '{args.mlflow_experiment}'")
        except Exception as e:  # noqa: BLE001
            print(f"\n[warn] MLflow logging skipped: {e}")


if __name__ == "__main__":
    main()
