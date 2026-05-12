"""Minimal end-to-end training script — vanilla HF/PEFT, no custom code path.

Goal: prove (or disprove) that the model + warm-start + corpus combination
can be trained at all. If this learns, our custom code has a bug. If this
also fails, the issue is upstream (data, model, warm-start saturation).

Test design — *overfitting probe*:
- 10 records from pairs_500_random.jsonl
- Plain warm-start, NO override-alpha (uses saved scaling=1.0)
- 30 epochs
- Standard HF Trainer with manual prompt+response tokenisation + label masking
- No diff-aware, no NEFTune, no chunked entropy, no _attach_assistant_masks,
  no DiffWeightedDataCollator, no DiffAwareSFTTrainer

If our optimizer + LoRA + corpus combo can fit any signal, we should see
loss drop monotonically over epochs on a 10-row dataset. If not, optimization
is broken at a level deeper than any of our custom code.

Run: uv run python scripts/_diag/mimic_minimal_train.py 2>&1 | tee /tmp/mimic.log
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

# Ensure our chat template helper is importable so we can build messages list,
# but do NOT use any of our trainer/collator/metric code.
sys.path.insert(0, "libs/model-training/src")

import torch
from peft import PeftModel
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

MODEL = "Qwen/Qwen3.5-9B"
ADAPTER = "danielcherubini/Qwen3.5-DeltaCoder-9B"

IGNORE_INDEX = -100


class TinyOverfitDataset(Dataset):
    """Hand-rolled dataset: prompt+response, label only the response tokens."""

    def __init__(self, records, tokenizer, max_length: int = 2048):
        self.examples = []
        n_skipped = 0
        for r in records:
            prompt = r["activation_text"]
            # Compute response = teacher_text minus the activation prefix
            teacher = r["teacher_text"]
            if not teacher.startswith(prompt):
                # fallback: assume teacher is "{prompt}\n\n{response}"
                # Find the last "\n## " before some heuristic separator
                # Just take everything after the activation if it's a prefix substring.
                if prompt in teacher:
                    response = teacher.split(prompt, 1)[1].lstrip("\n")
                else:
                    response = teacher  # whole teacher is response
            else:
                response = teacher[len(prompt) :].lstrip("\n")

            # Manual prompt+response with response-only labels
            prompt_msg = [{"role": "user", "content": prompt}]
            response_msg = prompt_msg + [{"role": "assistant", "content": response}]

            # Render via apply_chat_template(tokenize=False) to text, then
            # tokenize the text. This sidesteps the inconsistent return shape
            # of tokenize=True (sometimes a list, sometimes a BatchEncoding —
            # the latter has hasattr(.,'keys')=True but isinstance(.,dict)=False,
            # which silently breaks naive dict-checks).
            try:
                prompt_text = tokenizer.apply_chat_template(
                    prompt_msg,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                full_text = tokenizer.apply_chat_template(
                    response_msg,
                    tokenize=False,
                    add_generation_prompt=False,
                )
                prompt_ids = tokenizer(prompt_text, add_special_tokens=False)[
                    "input_ids"
                ]
                full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
            except Exception as e:
                if n_skipped == 0:
                    print(f"  first skip cause: {type(e).__name__}: {e}")
                n_skipped += 1
                continue

            # Truncate from the END (keep the response, drop prompt prefix if too long).
            # When truncated, clamp prompt_len so at least half the kept tensor is
            # response tokens — guarantees `labels` has non-IGNORE positions.
            if len(full_ids) > max_length:
                full_ids = full_ids[-max_length:]
                prompt_len = min(len(prompt_ids), max_length // 2)
            else:
                prompt_len = len(prompt_ids)
                if prompt_len >= len(full_ids):
                    n_skipped += 1
                    continue

            labels = list(full_ids)
            for i in range(min(prompt_len, len(labels))):
                labels[i] = IGNORE_INDEX
            # Skip if no labeled positions
            if all(x == IGNORE_INDEX for x in labels):
                n_skipped += 1
                continue

            self.examples.append(
                {
                    "input_ids": full_ids,
                    "labels": labels,
                    "attention_mask": [1] * len(full_ids),
                }
            )

        print(f"Built {len(self.examples)} examples; skipped {n_skipped}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class PadCollator:
    """Pad to longest in batch; pad labels with IGNORE_INDEX."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        max_len = max(len(b["input_ids"]) for b in batch)
        out_ids, out_labels, out_mask = [], [], []
        for b in batch:
            pad_n = max_len - len(b["input_ids"])
            out_ids.append(list(b["input_ids"]) + [self.pad_token_id] * pad_n)
            out_labels.append(list(b["labels"]) + [IGNORE_INDEX] * pad_n)
            out_mask.append([1] * len(b["input_ids"]) + [0] * pad_n)
        return {
            "input_ids": torch.tensor(out_ids, dtype=torch.long),
            "labels": torch.tensor(out_labels, dtype=torch.long),
            "attention_mask": torch.tensor(out_mask, dtype=torch.long),
        }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/_ab/pairs_500_random.jsonl")
    p.add_argument("--n-rows", type=int, default=10)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--output", default="/tmp/mimic-train-out")
    p.add_argument("--no-warm-start", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Loading tokenizer + base model {MODEL}...")
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        quantization_config=bnb,
        dtype=torch.bfloat16,
    )

    if args.no_warm_start:
        from peft import LoraConfig, get_peft_model

        cfg = LoraConfig(
            r=32,
            lora_alpha=32,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "v_proj"],
        )
        model = get_peft_model(model, cfg)
    else:
        print(f"Loading adapter {ADAPTER}...")
        model = PeftModel.from_pretrained(model, ADAPTER, is_trainable=True)
        for n, p in model.named_parameters():
            if "lora_" in n:
                p.requires_grad_(True)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {n_trainable / 1e6:.2f}M")

    # Sample N rows
    random.seed(42)
    rows = [
        json.loads(line)
        for line in Path(args.data).read_text().splitlines()
        if line.strip()
    ]
    sample = random.sample(rows, args.n_rows)
    print(f"Loaded {len(rows)} rows, sampled {len(sample)}")

    ds = TinyOverfitDataset(sample, tok, max_length=args.max_length)
    if len(ds) == 0:
        print("ERROR: no examples built")
        return

    collator = PadCollator(pad_token_id=tok.pad_token_id)

    # Diagnostic: run forward on one example BEFORE training to record initial loss/accuracy
    model.eval()
    with torch.no_grad():
        first_batch = collator([ds[i] for i in range(min(2, len(ds)))])
        first_batch = {k: v.cuda() for k, v in first_batch.items()}
        out = model(**first_batch)
        labels = first_batch["labels"]
        logits = out.logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        mask = shift_labels != IGNORE_INDEX
        per_token_loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            shift_labels.reshape(-1),
            ignore_index=IGNORE_INDEX,
            reduction="none",
        ).reshape(shift_labels.shape)
        per_token_loss = per_token_loss * mask.float()
        n = mask.sum().item()
        mean_loss = per_token_loss.sum().item() / max(1, n)
        pred = logits.argmax(dim=-1)
        correct = ((pred == shift_labels) & mask).sum().item()
        acc = correct / max(1, n)
        print("\n=== INITIAL (warm-start, no training) ===")
        print(f"  loss={mean_loss:.4f}  token_accuracy={acc:.4f}  n_labeled={n}")

    print(f"\n=== TRAINING ({args.epochs} epochs, {len(ds)} rows, lr={args.lr}) ===")
    targs = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,  # tiny so we see learning per epoch
        learning_rate=args.lr,
        logging_steps=1,
        save_strategy="no",
        bf16=True,
        optim="paged_adamw_32bit",  # 32-bit Adam to rule out 8-bit quantization issues
        lr_scheduler_type="constant",
        warmup_ratio=0.0,
        report_to="none",
        # Gradient checkpointing IS required on L4 22GB — without it, 9B + LoRA
        # OOMs at max_length>=2048. Re-enabling matches our production trainer.
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=collator,
    )
    model.train()
    trainer.train()

    # Final diagnostic on the same first batch
    model.eval()
    with torch.no_grad():
        out = model(**first_batch)
        logits = out.logits[:, :-1, :]
        shift_labels = first_batch["labels"][:, 1:]
        mask = shift_labels != IGNORE_INDEX
        per_token_loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            shift_labels.reshape(-1),
            ignore_index=IGNORE_INDEX,
            reduction="none",
        ).reshape(shift_labels.shape)
        per_token_loss = per_token_loss * mask.float()
        n = mask.sum().item()
        mean_loss = per_token_loss.sum().item() / max(1, n)
        pred = logits.argmax(dim=-1)
        correct = ((pred == shift_labels) & mask).sum().item()
        acc = correct / max(1, n)
        print(f"\n=== FINAL (after {args.epochs} epochs of overfitting) ===")
        print(f"  loss={mean_loss:.4f}  token_accuracy={acc:.4f}  n_labeled={n}")
        print("\nIf loss did NOT drop to near 0 and accuracy did NOT reach near 1.0,")
        print("the optimizer is not effective at fitting even a tiny dataset.")
        print("This indicates a fundamental training-pipeline problem.")


if __name__ == "__main__":
    main()
