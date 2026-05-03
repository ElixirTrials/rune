"""Per-token loss + token_accuracy on a held-out split, for any LoRA adapter.

Addresses advisor critique: previous overfit probes measured *memorisation*
(2 records sampled from the 5 training rows). This evaluates *generalisation*
on a disjoint 100-row split sampled from records NOT in the training set.

Compares three states:
  1. Base Qwen3.5-9B alone (no adapter)
  2. + deltacoder warm-start (no fine-tune)
  3. + a fine-tuned adapter (path supplied via --adapter)

If state 3 beats state 2 on held-out tok_acc/loss, fine-tuning generalised.
If state 3 == state 2, fine-tuning only memorised training data.

Run: uv run python scripts/_diag/eval_heldout.py \\
    --heldout data/_ab/pairs_heldout_100.jsonl \\
    --adapter ./hpo_artifacts/<run>/adapter
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

IGNORE_INDEX = -100


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--heldout", default="data/_ab/pairs_heldout_100.jsonl")
    p.add_argument("--n-rows", type=int, default=50, help="Rows of heldout to evaluate.")
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument(
        "--adapter",
        default=None,
        help="Path to a fine-tuned PEFT adapter dir. If omitted, only the "
        "base+deltacoder eval runs (state 1 + 2).",
    )
    p.add_argument(
        "--skip-base",
        action="store_true",
        help="Skip the base-model-alone state (state 1) — saves ~1 min of forward.",
    )
    p.add_argument(
        "--deltacoder-id",
        default="danielcherubini/Qwen3.5-DeltaCoder-9B",
    )
    p.add_argument("--model", default="Qwen/Qwen3.5-9B")
    return p.parse_args()


def build_examples(records, tokenizer, max_length):
    """Same prompt+response masking as mimic_minimal_train.py — eval-time only."""
    examples = []
    for r in records:
        prompt = r["activation_text"]
        teacher = r["teacher_text"]
        if teacher.startswith(prompt):
            response = teacher[len(prompt):].lstrip("\n")
        elif prompt in teacher:
            response = teacher.split(prompt, 1)[1].lstrip("\n")
        else:
            response = teacher
        if not response:
            continue

        prompt_msg = [{"role": "user", "content": prompt}]
        response_msg = prompt_msg + [{"role": "assistant", "content": response}]
        prompt_text = tokenizer.apply_chat_template(
            prompt_msg, tokenize=False, add_generation_prompt=True,
        )
        full_text = tokenizer.apply_chat_template(
            response_msg, tokenize=False, add_generation_prompt=False,
        )
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]

        if len(full_ids) > max_length:
            full_ids = full_ids[-max_length:]
            prompt_len = min(len(prompt_ids), max_length // 2)
        else:
            prompt_len = len(prompt_ids)
            if prompt_len >= len(full_ids):
                continue

        labels = list(full_ids)
        for i in range(min(prompt_len, len(labels))):
            labels[i] = IGNORE_INDEX
        if all(line == IGNORE_INDEX for line in labels):
            continue

        examples.append({"input_ids": full_ids, "labels": labels})
    return examples


def evaluate(model, tokenizer, examples, *, label: str) -> dict:
    """Return loss + token_accuracy (totals across all examples) for a model."""
    import torch
    import torch.nn.functional as F  # noqa: N812

    model.eval()
    total_correct = 0
    total_labeled = 0
    total_ce = 0.0
    n_skipped = 0

    with torch.no_grad():
        for ex in examples:
            ids = torch.tensor([ex["input_ids"]], device=model.device, dtype=torch.long)
            labs = torch.tensor([ex["labels"]], device=model.device, dtype=torch.long)
            # Forward
            try:
                out = model(input_ids=ids)
            except Exception as e:  # noqa: BLE001
                n_skipped += 1
                if n_skipped < 3:
                    print(f"  [{label}] skipped 1 example: {type(e).__name__}: {e}")
                continue
            logits = out.logits[:, :-1, :]
            shift_labels = labs[:, 1:]
            mask = shift_labels != IGNORE_INDEX

            # CE
            ce = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                shift_labels.reshape(-1),
                ignore_index=IGNORE_INDEX,
                reduction="none",
            ).reshape(shift_labels.shape)
            n = mask.sum().item()
            total_ce += (ce * mask.float()).sum().item()
            total_labeled += n

            # Acc
            pred = logits.argmax(dim=-1)
            total_correct += ((pred == shift_labels) & mask).sum().item()

            # Free
            del out, logits, ce, pred
            torch.cuda.empty_cache()

    mean_loss = total_ce / max(1, total_labeled)
    acc = total_correct / max(1, total_labeled)
    return {
        "label": label,
        "n_examples": len(examples) - n_skipped,
        "n_labeled_tokens": total_labeled,
        "mean_loss": mean_loss,
        "token_accuracy": acc,
    }


def main():
    args = parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"Loading tokenizer + base {args.model}...")
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=bnb, dtype=torch.bfloat16,
    )

    print(f"Loading {args.heldout} ...")
    rows = [json.loads(line) for line in Path(args.heldout).read_text().splitlines() if line.strip()]
    rows = rows[: args.n_rows]
    examples = build_examples(rows, tok, max_length=args.max_length)
    print(f"Built {len(examples)} eval examples (from {len(rows)} rows)")
    if not examples:
        print("ERROR: no examples built")
        return

    results = []

    # State 1: base model alone
    if not args.skip_base:
        results.append(evaluate(model, tok, examples, label="base"))
        print(f"  [base] loss={results[-1]['mean_loss']:.4f} "
              f"tok_acc={results[-1]['token_accuracy']:.4f} "
              f"n_tokens={results[-1]['n_labeled_tokens']}")

    # State 2: + deltacoder warm-start
    print(f"Loading deltacoder warm-start: {args.deltacoder_id}")
    model = PeftModel.from_pretrained(model, args.deltacoder_id)
    results.append(evaluate(model, tok, examples, label="deltacoder"))
    print(f"  [deltacoder] loss={results[-1]['mean_loss']:.4f} "
          f"tok_acc={results[-1]['token_accuracy']:.4f}")

    # State 3: + fine-tuned adapter (replace deltacoder)
    if args.adapter:
        # Unload current adapter, load the fine-tune
        print(f"Unloading deltacoder, loading fine-tune adapter from {args.adapter}")
        # Get the inner base back
        base = model.unload()
        model = PeftModel.from_pretrained(base, args.adapter)
        results.append(
            evaluate(model, tok, examples, label="fine-tuned")
        )
        print(f"  [fine-tuned] loss={results[-1]['mean_loss']:.4f} "
              f"tok_acc={results[-1]['token_accuracy']:.4f}")

    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    print(f"{'state':<14} {'loss':>8} {'tok_acc':>9}  {'n_tokens':>10}")
    print(f"{'-'*14} {'-'*8} {'-'*9}  {'-'*10}")
    for r in results:
        print(
            f"{r['label']:<14} {r['mean_loss']:>8.4f} {r['token_accuracy']:>9.4f}  "
            f"{r['n_labeled_tokens']:>10}"
        )

    if len(results) >= 3:
        delta = results[2]["token_accuracy"] - results[1]["token_accuracy"]
        delta_loss = results[1]["mean_loss"] - results[2]["mean_loss"]
        print(
            f"\nFine-tuned vs deltacoder: tok_acc Δ={delta:+.4f}, "
            f"loss Δ={delta_loss:+.4f} (loss reduction)"
        )
        print(
            "Generalisation confirmed."
            if delta > 0.005 or delta_loss > 0.01
            else "Within noise — fine-tune did not generalise meaningfully."
        )


if __name__ == "__main__":
    main()
