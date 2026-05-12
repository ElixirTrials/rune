# A Short Report to My Advisor on the DeltaCoder Fine-Tune

**Date:** 2026-05-03
**Author's note:** Branch `fix/diff-loss-per-turn-alignment` (PR #35). Prior context lives in
- [`2026-05-03-diff-aware-span-match-rca.md`](2026-05-03-diff-aware-span-match-rca.md)
- [`2026-05-03-diff-aware-span-match-results.md`](2026-05-03-diff-aware-span-match-results.md)
- [`2026-05-03-training-deep-dive.md`](2026-05-03-training-deep-dive.md)

Dear advisor,

What follows is a tight write-up of an investigation I had set up as a "why isn't it learning?" debugging exercise. The headline turned out to be that the optimisation stack is fine and the supposed "flat training" we kept observing was an artefact of running every experiment for a single epoch over a small corpus — well below the visibility threshold for code-edit fine-tunes. Along the way I picked up two real bugs and one important corpus-shape finding that I think are worth your time. I would value your read on (1) whether the per-step "flat" measurement we kept making was a methodological mistake or a real ceiling, and (2) whether the corpus shape itself is worth re-mining. The 3-epoch validation run is still in flight as I write this — at step 78 of 147, ~50 % through, mid-epoch-2.

---

## 1. The presenting symptom

A QLoRA fine-tune of Qwen3.5-9B (a hybrid Mamba-Transformer) using the public `danielcherubini/Qwen3.5-DeltaCoder-9B` LoRA as a warm start. Trained on a corpus of 2,743 GitHub PRs mined into `(activation_text, teacher_text)` pairs. Goal: better PR-revision generation. Symptom: across half a dozen production runs the per-step `mean_token_accuracy` started at ~0.79 and stayed there. Loss values were all over the map (sometimes 30+, sometimes 0.8) which masked the underlying behaviour.

I had already worked through a span-match correctness bug in the diff-aware loss path (commits `b80c5f62` and earlier) and the model still wouldn't visibly learn. You had asked me to prove that the data wasn't bad, the trainer wasn't bugged, and that the fancy diff-aware loss wasn't masking a real problem.

---

## 2. Hypotheses, in priority order

| # | Hypothesis | Rationale | Status going in |
|---|---|---|---|
| H1 | LoRA scaling override `α=16` halves the canonical scaling vs deltacoder's saved `α=32 / r=32 → 1.0` | Examined adapter_config.json | Suspicious — hadn't been tested |
| H2 | LR `4.3e-5` too low; literature uses 1e-4 to 5e-4 for QLoRA SFT on code | Magicoder, QLoRA paper, community recipes | Pending |
| H3 | `paged_adamw_8bit` quantises optimizer state; tiny updates round to zero | Standard LoRA folklore | Cheap to test |
| H4 | NEFTune `α=5` adds noise that drowns small updates | Marginal gain in SFT literature; bigger risk on small fine-tunes | Pending |
| H5 | `gradient_checkpointing` interaction with PEFT, breaking grad flow | Has known PyTorch interaction issues | Cheap to test |
| H6 | Wrong `target_modules` — adapter not attached to all needed projections | Inspect trainable params | Refuted (86.5 M params on 12 modules including the Mamba-specific `in_proj_*`) |
| H7 | Diff-aware loss is silently dropping training signal | We were running with `--diff-aware-loss` enabled but corpus has `changed_token_frac ≈ 0.98` | Tested in earlier work; the path is a near-no-op on this corpus |
| H8 | Custom data path (`_attach_assistant_masks`, `DiffWeightedDataCollator`) silently destroys labels | The code is novel and untested against vanilla TRL | The most-likely "real bug" candidate |

I ruled H1 out empirically (α=32 had effectively no effect vs α=16 on a 9-step run). The deeper question — whether *anything* in our pipeline was wrong — needed a control with all our custom code stripped out.

---

## 3. Method: the minimal-mimic overfit probe

I wrote a vanilla-HuggingFace training script (`scripts/_diag/mimic_minimal_train.py`, ~280 lines) that uses the **standard** `transformers.Trainer` — not our `DiffAwareSFTTrainer`, not TRL's `SFTTrainer`, no `_attach_assistant_masks`, no `DiffWeightedDataCollator`, no NEFTune. Just: load the model, attach the adapter, hand-roll prompt+response tokenisation with response-only labels, and run `Trainer.train()`.

The key snippet is the dataset constructor (response-only label masking):

```python
class TinyOverfitDataset(Dataset):
    """Hand-rolled dataset: prompt+response, label only the response tokens."""

    def __init__(self, records, tokenizer, max_length: int = 2048):
        self.examples = []
        for r in records:
            prompt = r["activation_text"]
            teacher = r["teacher_text"]
            response = teacher[len(prompt):].lstrip("\n") if teacher.startswith(prompt) \
                       else teacher.split(prompt, 1)[1].lstrip("\n") if prompt in teacher \
                       else teacher

            prompt_msg = [{"role": "user", "content": prompt}]
            response_msg = prompt_msg + [{"role": "assistant", "content": response}]

            # Render to text first, THEN tokenize. apply_chat_template(tokenize=True)
            # returns a BatchEncoding (NOT a dict, NOT a list) — len() returns the
            # dict-key count of 2, which silently breaks naive dict-checks.
            prompt_text = tokenizer.apply_chat_template(prompt_msg, tokenize=False,
                                                       add_generation_prompt=True)
            full_text = tokenizer.apply_chat_template(response_msg, tokenize=False,
                                                     add_generation_prompt=False)
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
            self.examples.append({"input_ids": full_ids, "labels": labels,
                                  "attention_mask": [1] * len(full_ids)})
```

The training loop is the textbook configuration:

```python
targs = TrainingArguments(
    output_dir=args.output,
    num_train_epochs=args.epochs,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    learning_rate=args.lr,
    bf16=True,
    optim="paged_adamw_32bit",      # 32-bit Adam: rules out 8-bit quantisation issue
    lr_scheduler_type="constant",
    warmup_ratio=0.0,
    gradient_checkpointing=True,    # required on L4 22GB
    gradient_checkpointing_kwargs={"use_reentrant": False},
)
if hasattr(model, "enable_input_require_grads"):
    model.enable_input_require_grads()  # PEFT + checkpointing compat
trainer = Trainer(model=model, args=targs, train_dataset=ds, data_collator=collator)
trainer.train()
```

The probe configuration is **5 records × 20 epochs** with `gradient_accumulation_steps=1`. That gives 100 optimisation steps at 20 passes per row — far more than enough to memorise. I measure `loss` and `token_accuracy` on a fixed 2-record held-out batch *before training starts* and *after the run finishes*. If 20 passes per row can't move loss toward zero, optimisation is broken at a layer deeper than any of our custom code.

---

## 4. Headline results (already in)

### 4.1 Mimic warm-start

| | Initial | Final (after 20 epochs) |
|---|---|---|
| `loss` | 1.069 | **0.002** (535× reduction) |
| `mean_token_accuracy` | 0.7625 | **1.0000** |

### 4.2 Mimic cold-start (`--no-warm-start`, fresh `LoraConfig(r=32, α=32, q_proj+v_proj)`)

| | Initial | Final |
|---|---|---|
| `loss` | 1.071 | **0.016** |
| `mean_token_accuracy` | 0.7593 | **0.9946** |

Both reach near-perfect memorisation. **The optimisation stack — quantisation, LoRA, gradient flow, paged_adamw_32bit, checkpointing — is unambiguously healthy.**

### 4.3 Direct A/B against our trainer (the critical one)

I then ran our pipeline (`scripts/train.sh`, the production code path that uses `_attach_assistant_masks`, `DiffWeightedDataCollator`, `DiffAwareSFTTrainer` falling back to plain SFT when `--diff-aware-loss` is off) on **the exact same 5 records × 20 epochs** with deltacoder warm-start, lr=2e-4, grad_accum=1.

| Step | Epoch | Loss | tok_acc |
|---|---|---|---|
| 1 | 0.4 | 1.50 | 0.684 |
| 9 | 1.8 | 0.39 | 0.902 |
| 17 | 3.4 | 0.10 | (rising) |
| 50 | 10.0 | **0.009** | **1.0000** |
| 100 | 20.0 | **0.0002** | 1.0000 |

`train_loss` (epoch sum reported by HF Trainer) ended at 0.1516, `train_runtime=336s`. **The same data path that "couldn't learn" on 500 rows × 1 epoch reaches the same near-zero-loss / 100 %-accuracy floor as the vanilla mimic.** Our pipeline is not bugged.

### 4.4 What the "flat training" actually was

Looking at all our prior production runs through the lens of *passes per row*:

| Setup | Passes per row | Observed tok_acc trajectory |
|---|---|---|
| Mimic, 5 × 20 ep, ga=1 | 20 | 0.76 → 1.00 |
| **Our trainer, 5 × 20 ep, ga=1** | **20** | **0.68 → 1.00** |
| Our trainer, 500 × 1 ep, ga=8 | 1 | 0.79 → 0.84 (modest, real) |
| Our trainer, 500 × 5 ep, ga=8 | <1 (only 49 of 245 steps when killed) | 0.81 → 0.80 (looked flat — but didn't even complete one epoch) |
| The 28-step run we killed earlier | <0.1 (8 % of one epoch) | 7.4 → 7.5 (was always going to look flat) |

Every prior "flat training" run was at <1 pass per row. The model simply had not seen each example even once on average. The Magicoder / QLoRA-paper / community-consensus recipe is **2-3 epochs minimum**. We were running 1 epoch and then claiming the model wasn't learning. It was learning — barely, because each row had been seen exactly once. With more passes, it would lift.

---

## 5. Two real bugs found along the way

### 5.1 The `apply_chat_template` BatchEncoding trap (in my own mimic script)

`tokenizer.apply_chat_template(messages, tokenize=True, return_tensors=None)` returns a `BatchEncoding` object, **not a dict and not a list.** `isinstance(enc, dict) → False`, but `enc["input_ids"]` works. `len(enc) → 2` (the dict-key count, not the token count). The first version of my mimic script used `len(prompt_ids)` and then a guard `if prompt_len >= len(full_ids)`, which made every record look like an empty response and silently skipped all of them. Concretely:

```python
# WRONG — len() returns 2 for both, every record skipped:
prompt_ids = tokenizer.apply_chat_template(msgs, tokenize=True, return_tensors=None)
if len(prompt_ids) >= len(full_ids): continue   # always true, dict-key counts
```

The robust workaround is to render the chat-formatted text first and tokenize the text separately:

```python
# RIGHT — guaranteed list[int]:
prompt_text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
```

I checked our production code path: `compute_assistant_masks` in `libs/model-training/src/model_training/trajectory.py` uses `tokenize=True, return_dict=False` and explicitly `list(...)`-wraps the result, so it's safe. The bug was confined to my new diagnostic script.

### 5.2 HPO was hardcoded to `epochs=1` per trial

`scripts/optimization/run_training_hpo.py` had

```python
"epochs": 1,  # proxy mode; operators choose final epochs on the winner
```

The "proxy mode" assumption was that 1 epoch on 500 rows is a noisy-but-useful predictor of multi-epoch performance. The deep-dive above falsifies that assumption: at 1 epoch on 500 rows the per-step signal is below the visibility threshold — the optimiser is exploring search space against near-zero gradient information. **Every "winner" the HPO has ever produced was tuned under this regime.** The two best diff-aware HPO winners (t010, t016) happened to land on conservative HP that compensate for the gradient noise — exactly the symptom you'd see if the loss was flat.

I added a `--proxy-epochs` flag (default 3) and wired it through `HPORunArgs`, `_build_trial_kwargs`, and the existing CLI pass-through in `run_hpo.sh`. The full diff is in commit `139b610f`. After merging this PR I'd want to rerun the HPO from scratch on corrected code.

---

## 6. The data-shape question (separate but important)

The corpus stores `pre_code` and `post_code` as **unified diff strings** (with `--- file ---`, `@@ ... @@`, `+`/`-` lines), not raw file bodies. The diff-aware loss path expects raw file bodies; on diff-of-diffs `_compute_hunk_ranges` median is 0.977 — almost every line of post is "different from" pre because pre and post are separate diffs of the same file, often touching different regions. That makes diff-aware loss mathematically equivalent to plain SFT on this corpus (`changed_token_frac ≈ 0.98`).

Independent of the deep-dive, plain-SFT on this corpus shape is the right path until the corpus is re-mined to raw bodies (and we'd need to also redesign the loss to weight correctly on a unified-diff target — see [`2026-05-03-training-deep-dive.md`](2026-05-03-training-deep-dive.md) for the longer treatment of this).

---

## 7. The validation training currently running

I've launched a single 3-epoch validation training to confirm that with 3 passes per row we see actual generalisation (not just memorisation). Configuration:

```bash
bash scripts/train.sh \
  --dataset data/_ab/pairs_500_random.jsonl \
  --warm-start deltacoder \
  --epochs 3 \
  --lr 2e-4 \
  --grad-accum 8 \
  --lr-scheduler cosine \
  --warmup-ratio 0.05 \
  --max-seq-length 2048 \
  --encoding-mode multi_turn
```

State at write time: step 78 / 147, ~50 % through (mid-epoch-2). LR ramped 0 → 2e-4 over the first 7 steps then started cosine decay. tok_acc bouncing 0.78–0.84 with mean ~0.80 — same regime as Plain SFT control through epoch 1. The deep-dive's prediction is that epochs 2 and 3 lift the metric meaningfully; we'll know in ~45 minutes.

If the run ends with `tok_acc > 0.85`, I'll proceed to the HPO sweep with `proxy_epochs=3` to find a tuned recipe. If it ends at the warm-start prior of ~0.79, the prediction was wrong and the right next step is the corpus question rather than HPO.

---

## 8. Where I'd value your read

1. **Was the methodological mistake (1-epoch-per-trial HPO + 1-epoch validation) inevitable?** This was the original config; nobody flagged it, and the literature recipes all use 2-3 epochs. I'm wondering if there's a more general lesson about how to design HPO probes in the small-corpus regime where 1 epoch is below the signal threshold.

2. **The corpus-shape decision.** Plain SFT works with the current diff-format corpus, but the loss surface is asymmetric (every line of a unified diff matters equally for the patch to apply, so the "weight context less" assumption of body-shape diff-aware loss is wrong here). I'm leaning toward (a) ship plain SFT on current corpus, (b) corpus expansion (10K records via more PRs or synthetic OSS-Instruct), (c) defer body-shape re-mining indefinitely.

3. **Whether the LoRA scaling-override default (`α=16` on top of saved `α=32`) is worth a follow-up experiment.** I tested α=32 (canonical scaling=1.0) and it didn't move the needle in 9 steps — but 9 steps is back below the visibility threshold. A proper test would need 3 epochs at each setting, which is HPO territory.

I'll send the validation result and HPO launch as a follow-up.

— Your student

---

## Addendum: Implementing the advisor's recommendations (later that day)

After your reply (`instructions/review_of_arguments.md`), addressed each item in the order of cost.

### 1. `grad_norm` spike at epoch boundary — verified, not a problem

Pulled the `grad_norm` history from MLflow for steps 70–95 (the boundary you flagged). Plateau is 0.20–0.30; the largest excursions are isolated single-step spikes to 0.40–0.50 with immediate return. No sustained excursion to ~1.0. The `max_grad_norm=1.0` default in HF `TrainingArguments` is in effect (we never override it). Healthy.

### 2. End-of-data shuffle — verified, not a problem

`SFTConfig.shuffle_dataset=False` is a one-time pre-shuffle, not per-epoch. HF Trainer's default `_get_train_sampler` returns a `RandomSampler` (since `group_by_length=False`), which DOES re-shuffle each epoch with a fresh seed. So the loss drop at the right edge is not a same-batch-as-end-of-epoch-1 artefact.

### 3. Memorisation vs. generalisation — addressed via two evaluators

The blocking item from your review. Built a disjoint 100-row held-out split (`data/_ab/pairs_heldout_100.jsonl`, seed=99, sampled from the 2,243 records NOT in `pairs_500_random.jsonl`) and two evaluators that run against it after training finishes:

- **`scripts/_diag/eval_heldout.py`** — per-token CE + token-accuracy on the held-out split, comparing base / +deltacoder / +fine-tuned. Decision rule: tok_acc Δ > 0.005 OR loss Δ > 0.01 ⇒ generalisation confirmed.
- **`scripts/_diag/eval_patch_quality.py`** — generates completions and scores against ground truth on three tiers. Tier 1 is syntactic validity (file headers, hunk headers, ±count consistency — checks for the "off-by-one hunk header / wrong context" pathology you flagged). Tier 2 is hunk-IoU (Jaccard over `(file, ±, line)` triples) + char-similarity + exact-match. Tier 3 (`git apply --check` against parent-commit file state) is deferred — our corpus doesn't carry the parent-commit file content.
- **`scripts/_diag/eval_full.sh`** — runs both back-to-back on a given adapter dir.

Both committed at `<HEAD>`. Will run them on the v3 checkpoint as soon as training completes (~10 min).

### 4. Patch applicability as the *real* eval metric

You're right that token CE is loss-aligned but not metric-aligned. The Tier-1 syntactic-validity check in `eval_patch_quality.py` is a strong proxy without needing parent-commit file states: a diff with mismatched ±counts will fail `git apply` 100 % of the time, so the rate is a useful upper bound on patch applicability. Will report this number alongside the Tier-2 IoU after the run.

### 5. t010/t016 HP retest at 3 epochs — queued

Agreed that the 9-step α A/B was below the visibility threshold. Plan: after v3 finishes and the held-out eval lands, run t016's exact HP (lr=4.3e-5, α=16, dropout=0.1, ga=32, constant LR, NEFTune=5) for 3 epochs on `pairs_500_random.jsonl` and compare held-out tok_acc against v3's. If t016's HP at 3 epochs hits or beats v3, the original HPO winners deserve a fairer treatment than my "tuned under noise" framing implied.

### 6. Per-epoch val_loss curve — deferred to follow-up

Acknowledging this is the proper way to detect overfit/divergence during training. The plumbing requires changes in `_build_sft_config` (eval_strategy=epoch), `train_qlora` (build eval dataset), `_construct_sft_trainer` (pass through), and `build_diff_aware_sft_trainer` (accept eval_dataset kwarg). About 30–60 minutes of careful work; queued as a separate commit. For now, `eval_heldout.py` after training gives end-of-run held-out metrics.

### 7. The methodology lesson on HPO probes

Your point — *"HPO probes must operate in the same regime where the metric is informative"* — is one I want to bake into our docs as a precondition check. Concretely: before any HPO sweep, run a single 2-epoch sanity training and verify `val_loss` moves at least 0.05 between epochs. If it doesn't, HPO is searching against zero-information signal; lengthen proxy_epochs first. Will add this to the HPO docs once the rest settles.

### Updated open questions

The original three open questions remain, but I'd refocus them given the additional work:

1. **(Was) Was the 1-epoch-per-trial methodology a foreseeable mistake?** I now think yes — the cheapest pre-HPO check (a sanity training that measures val_loss across epochs) would have caught it. Adding to docs.

2. **(Was) Corpus shape decision.** Once `eval_patch_quality.py` runs on v3, we'll have a concrete patch-applicability rate. If it's > 70 % on plain SFT with the current diff-format corpus, defer re-mining. If < 30 %, re-mining is justified. The number will tell us, not principle.

3. **(New) Does t010/t016's "conservative" HP at 3 epochs match or beat the literature-canonical recipe?** Direct test queued. The result decides whether the historical HPO findings are salvageable or need full re-run.

— Your student (still)
