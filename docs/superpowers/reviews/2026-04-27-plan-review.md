# Skeptical Review of Training-Issues Fix Plan

Reviewer: adversarial pass against `docs/superpowers/plans/2026-04-27-training-issues-fix.md`

---

## Critical Issues (BLOCK merge)

### 1. [Task 1] — Test does NOT test the fix; will pass before AND after the change

The failing-test step imports `DataCollatorForLanguageModeling` from
`trl.trainer.sft_trainer` directly, constructs it with `completion_only_loss=False`
(the post-fix value), and asserts labels are not all -100. The test is written
around the *fixed* collator, not the *broken* one. Step 2 says "run test to verify
it fails" — but this test will PASS on the current codebase too, because the test
constructs the collator directly with the already-correct kwarg, bypassing
`build_diff_aware_sft_trainer` entirely.

The test exercises the TRL collator's own behavior, not whether
`build_diff_aware_sft_trainer` at `diff_loss.py:636` passes `completion_only_loss=False`.
The correct failing test would call `build_diff_aware_sft_trainer(...)` and inspect the
`inner_collator` attribute on the returned trainer's `data_collator`, or invoke
`build_diff_aware_sft_trainer` end-to-end with a minimal dataset and assert loss > 0.
As written, the test has zero regression value — the bug could be re-introduced at
`diff_loss.py:638` and the test would still pass.

Additionally: the test imports `build_diff_aware_sft_trainer` but then doesn't call
it. That import is dead code in the test.

### 2. [Task 1] — `completion_only_loss=False` means ALL tokens are learning targets (prompt + system turn)

The fix rationale says "DiffWeightedDataCollator owns masking via the `labels ==
IGNORE_INDEX` check." But the identity fallback path in `DiffWeightedDataCollator.__call__`
(`diff_loss.py:422-426`) computes weights as `1.0 if lab != -100 else 0.0`. If the
inner collator with `completion_only_loss=False` sets ALL labels to their token IDs
(no masking), then every prompt token — including system turn, user messages,
`<|im_start|>` markers — gets weight `1.0` and gradient signal. The diff path should
only train on the assistant's revised code, not on the entire conversation.

The plan states this is fine because "SFTTrainer chat-template preprocessing already
produces correct labels for prompt+completion." This is not verified. `SFTConfig` has
`assistant_only_loss=False` (`trainer.py:601`) AND the diff path skips
`_attach_assistant_masks` (`trainer.py:877-878`). With both disabled and the inner
collator set to `completion_only_loss=False`, there is **no mechanism** that produces
`-100` labels for prompt tokens. The fix trades zero-loss-every-step for
training-on-everything. Whether that is better or worse for adapter quality is
unverified; it is definitely not the intended behavior described by the RCA.

The RCA's own alternative fix direction — "On the diff path, still run
`_attach_assistant_masks` and include `assistant_masks` in the dataset before passing
to `DiffWeightedDataCollator`" — is the correct fix. The plan chose the other
alternative but did not verify the masking invariant holds.

### 3. [Task 8] — `_build_sft_config` tuple-return change breaks the only caller

Task 8 Step 5 says: "change `_build_sft_config` to return a tuple
`(sft_config, total_steps_or_none)`, and update its single caller in `train_qlora`."
At `trainer.py:899-912`, `_build_sft_config` is called and its return value is
assigned to `training_args` directly. There is exactly one call site.

The plan assigns wave B (Tasks 2, 6, 7) to one agent for `trainer.py`, and wave D
(Task 8) to a separate agent after B completes. Task 8 modifies `_build_sft_config`
**and** its call site at `trainer.py:899`. Wave B's agent also commits to `trainer.py`
(Tasks 2, 6, 7 all commit `trainer.py`). When wave D runs and changes the return type
of `_build_sft_config`, the wave B commits on `trainer.py` are already in; the wave D
agent needs to find the exact same lines that wave B may have shifted/modified. This is
a merge conflict hazard, but even without conflict: the plan does not add a test that
the call site correctly unpacks the new tuple before the commit. If the implementer
forgets the call-site update, `training_args` becomes a `tuple` and the next line
`trainer = _construct_sft_trainer(..., args=training_args)` passes a tuple as SFTConfig,
which will either silently corrupt training args or raise at runtime — after all tests
pass on CPU (since `_build_sft_config` is not exercised in CPU unit tests).

The existing test in Task 8 only calls `_build_run_params` (which is unchanged by this
tuple refactor). There is no test that calls `train_qlora` end-to-end to catch the
unpacking failure. The plan also does not update the existing `test_trainer_mlflow.py`
tests that call `_build_run_params` with a fixed signature — those tests will need the
new `total_steps` parameter or they break (Task 8 adds `total_steps` to
`_build_run_params` signature as a required arg; the two existing calls at
`test_trainer_mlflow.py:121` and `test_trainer_mlflow.py:161` do not include it and
will fail with `TypeError`).

---

## High Priority (must address before execution)

### 4. [Task 5] — `byte_cap` silent-skip bug on last pad/special token

Task 3, Step 5 inserts:
```python
last_offset = enc["offset_mapping"][0][-1].tolist()
byte_cap = int(last_offset[1]) if last_offset[1] > 0 else len(teach)
```

In practice, Qwen tokenizers append an EOS token with offset `(0, 0)` as the final
element of `offset_mapping`. When truncation fires, the EOS is appended after the last
real token, meaning `enc["offset_mapping"][0][-1]` is `(0, 0)`, `last_offset[1]` is
`0`, and the fallback fires: `byte_cap = len(teach)`. That is fine.

But if the sequence is short enough NOT to truncate, and the tokenizer does NOT append
a trailing `(0, 0)` token, `last_offset[1]` is the byte end of the last real token —
which is correct. However, for Qwen3.5's sentencepiece tokenizer, the last real token
in the offset_mapping may not reach `len(teach)` exactly (whitespace/newline bytes after
the final token are often unaccounted for). If `byte_cap` < the start of a hunk that
begins near the end of the text, that hunk gets incorrectly dropped even though it was
not truncated. This is not catastrophic — eval skips that pair — but it silently
degrades eval coverage on non-truncated pairs, hiding the effect.

The correct guard is `enc["offset_mapping"][0][-2].tolist()` (second-to-last, skipping
the padding `(0,0)` terminal entry), or iterate backwards to find the last non-zero
offset. The current code is fragile for the non-truncation case.

### 5. [Task 9 (b)] — `llm_int8_enable_fp32_cpu_offload=True` will silently CPU-offload on low-VRAM trials

The plan dismisses this: "This does NOT mask Issue #2 — the CUDA OOM there is real
and Tasks 3/5/6 fix it." That is partially true, but: with `device_map="auto"` and
`llm_int8_enable_fp32_cpu_offload=True`, if any HPO trial starts while VRAM is more
constrained than usual (e.g. Trial 2 after a partially-freed Trial 1), `accelerate`
will silently split layers across GPU+CPU rather than raising. The trial will "succeed"
but run at 1/100th throughput with CPU-resident layers doing compute in float32 — not
bfloat16. The per-step training loss will be correct but the trial will consume 30-60x
more wall time, exhausting the HPO budget before enough trials complete to give Optuna
signal.

The plan explicitly says "a slow trial is recoverable; a load-time crash is not."
But in a time-budgeted HPO study, a trial that takes 10 hours instead of 10 minutes is
worse than a fast crash. The RCA-4 (b) fix direction also suggested "gate `device_map`
with a VRAM check" as an alternative. The plan should at least log a WARNING when CPU
offload actually triggers (detectable by checking `model.hf_device_map` after load).

### 6. [Task 9 (a)] — Double-save writes adapter weights twice, is not idempotent

The fix for Issue #1 calls `trainer.save_model(output_dir)` first (the TRL path,
which calls `PeftModel.save_pretrained(output_dir)` internally), then immediately calls
`inner.save_pretrained(output_dir, save_embedding_layers=False)` again. The second call
overwrites the files written by the first call — with the same weights, so the result
is correct, but this doubles the disk write and can corrupt the adapter directory if
interrupted between the two writes (e.g. the `adapter_config.json` from the first
write is deleted and the process dies before the second write completes).

The correct fix is to override `save_model` on `DiffAwareSFTTrainer` (or pass
`save_embedding_layers=False` to the TRL path directly). TRL's `SFTTrainer.save_model`
calls `model.save_pretrained(output_dir, **kwargs)` — the `**kwargs` path is available.
The current double-save approach should be flagged as unclean and replaced with a
single-call solution.

### 7. [Task 8] — `caplog.at_level("WARNING", logger="model_training.diff_loss")` may not capture

`diff_loss.py:23` declares `logger = logging.getLogger(__name__)`. When the module is
imported as `model_training.diff_loss`, `__name__` resolves to `"model_training.diff_loss"`.
The `caplog.at_level("WARNING", logger="model_training.diff_loss")` call should match.

However, the test imports `_compute_weighted_loss` directly, not through the full
package path. In CPU-only test environments where the package is installed as an
editable wheel under `model_training.*`, `__name__` will be `model_training.diff_loss`.
This is fine. But the test also has `logger="model_training.diff_loss"` as a string
literal — if the module structure ever changes (e.g. namespace package), this silently
stops capturing. This is low-severity but the test should be written as
`logger=logging.getLogger("model_training.diff_loss")` or use `caplog.records` with
a broader capture. Not a block, but worth noting.

---

## Medium Priority (should address, can ship without)

### 8. [Task 2] — Test's `_release_trial_state` signature mismatch

The test at Task 2 Step 1 calls:
```python
_release_trial_state(trainer, wrapper, dataset=None, persist_base=True)
```

The actual signature at `trainer.py:68` is:
```python
def _release_trial_state(trainer, model, dataset, *, persist_base) -> None
```

The `dataset` parameter is positional, not keyword-only. The test passes it as a
keyword argument (`dataset=None`), which works in Python (keyword arguments can match
positional params). This is not broken, but it establishes a false signature expectation
for future tests. If someone later re-orders the parameters (making `model`
keyword-only), the test would silently test a different argument. Minor issue.

### 9. [Task 4] — `${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}` does not OVERRIDE, just defaults

If a user has already exported `PYTORCH_CUDA_ALLOC_CONF` with a different or
incompatible value (e.g. `max_split_size_mb:256`), the `:-` default syntax leaves it
unchanged. The RCA-2 Cause 4 concern is specifically that the flag needs to be in
place before any `torch` import. If the user's pre-existing value doesn't include
`expandable_segments:True`, the fragmentation fix won't apply. The plan should use
`export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` unconditionally (or warn if
the existing value doesn't contain it) rather than silently defer to whatever the user
set.

### 10. [Task 6] — Test requires `hpo._run_single_trial_smoke()` which doesn't exist and is not required by the fix

The test calls `hpo._run_single_trial_smoke()` — a function the implementer is asked
to add as a "thin smoke wrapper." This couples the test to a new production function
that has no functional purpose beyond being a test hook. The fallback the plan offers
— "replace this assertion with one that checks `_flush_gpu_between_phases` is
referenced in `_run_single_trial`'s body via `inspect.getsource`" — is asserting
source code text, not behavior. Neither option is a proper behavior test. A better test
would monkeypatch `_flush_gpu_between_phases` into the actual `_run_single_trial`
fixture and assert it gets called, exercising the real production path.

### 11. [Task 3] — `_tokenize_for_eval` test uses a dict-returning fake tokenizer but TRL expects tensors

The test's `_FakeTok.__call__` returns `{"input_ids": [[1]], ...}` as plain Python
lists, not tensors. The production code at `run_training_hpo.py:541-543` does:
```python
input_ids = enc["input_ids"].to(model.device)
attention_mask = enc["attention_mask"].to(model.device)
offsets = enc["offset_mapping"][0].tolist()
```
These call `.to()` and `.tolist()` on the return values. The test only calls
`_tokenize_for_eval` and asserts `captured` kwargs — it never processes the return
value, so the type mismatch doesn't fail the test. But the test does not verify that
the truncation-clipped hunks path (`byte_cap` logic) works correctly, only that
`max_length` is forwarded. The test title says "passes max_length_and_truncation" but
doesn't verify truncation behavior at all. Low severity but misleading.

---

## Low / Nits

### 12. [Task 5] — Test raises `RuntimeError` but expects `pytest.raises(RuntimeError)` — fine, but `_FakeAdapterModel.__call__` signature mismatch

`_FakeAdapterModel.__call__` signature is `def __call__(self, *a, **k)` returning
`None`. The actual call in `_forward_hunk_metrics` at `run_training_hpo.py:545` does
`model(input_ids=..., attention_mask=...).logits[0]` — accessing `.logits` on `None`
would raise `AttributeError`, not the desired `RuntimeError`. The test expects
`RuntimeError("simulated OOM")` but the fake raises before `.logits` is accessed
(from `raise RuntimeError` in `__call__`), so it works by coincidence. If the
implementation ever checks `isinstance(error, torch.cuda.OutOfMemoryError)`, the fake
raises a plain `RuntimeError` and the test still passes but the behavior contract is
wrong.

### 13. [Wave assignments] — Task 6 modifies `run_training_hpo.py` (wave C) and also belongs to wave C with Task 3/5; no conflict

Confirmed: Task 6 modifies `run_training_hpo.py`, same as Tasks 3 and 5. But the
plan correctly assigns all three to wave C with "single agent serialises edits." No
actual conflict — the file ownership is correct.

### 14. [Task 9] — `_build_bnb_config` does not exist yet; test at Step 2 expects `ImportError` not `AssertionError`

The plan says "Expected: FAIL — `_build_bnb_config` does not exist." An `ImportError`
is not a test failure — `pytest` will mark it as `ERROR` not `FAIL`. The TDD discipline
described (write failing test, verify it fails, fix, verify it passes) only works if
the test fails with `AssertionError`. `pytest.raises(ImportError)` should wrap the
import, or the test should guard the import with `pytest.importorskip`. This is a minor
process issue but means Step 2 will produce an ERROR, not a FAIL, and the implementer
may not recognize it as the expected state.

---

## Plan-vs-RCA Coverage Gaps

- **RCA-4 (a): MLflow callback stale run_id after Trial 1 OOM.** The plan
  acknowledges this: "MLflow callback re-init is not explicitly addressed." The
  `MLflowCallback` is instantiated inside `SFTTrainer.__init__`, which runs inside
  `train_and_register` which is inside the HPO `try:` block at `run_training_hpo.py:751`.
  When Trial 1 OOMs in `_evaluate_adapter_on_heldout` (after `train_and_register`
  returns), `mlflow.end_run(status="FAILED")` IS called correctly at line 769 — so
  the run is terminated. Trial 2 opens a fresh `mlflow.start_run()` at line 731. The
  `MLflowCallback` is freshly constructed for Trial 2's `SFTTrainer`. The plan's
  dismissal ("user logs only ever show the truncated-message artifact, not measurable
  lost metrics") appears defensible: the `MLflowCallback` attaches at `SFTTrainer`
  construction time to whatever run is active, and Trial 2's trainer is constructed
  with Trial 2's run active. This is not a gap that needs addressing.

- **RCA-2 Fix Direction 1: `torch.inference_mode()` for eval.** Not addressed. The
  plan uses `torch.no_grad()` (already present at `run_training_hpo.py:524`). Using
  `inference_mode()` instead would reduce VRAM slightly (no version tracking). Not a
  correctness gap, and the OOM is addressed by Tasks 3/5/6, so this is genuinely
  optional.

- **RCA-3 Fix Direction 3: `delete_adapter` instead of `unload`.** The RCA notes
  that `BaseTuner.delete_adapter` removes the adapter from `peft_config` AND the LoRA
  layers, which is cleaner than `unload()` + `delattr`. The plan uses the `unload()` +
  `delattr` approach. This is defensible as it doesn't change the `peft_config` dict
  in ways that might affect the registry, but `delete_adapter` is the documented
  idiomatic approach. Not a correctness gap, but the chosen approach is more fragile
  (relies on PEFT's internal attribute name `peft_config` remaining stable).

- **RCA-5 H3: Too-few training steps.** The plan addresses this only with
  observability (log `total_steps`). No fix for the underlying cause (multi-turn
  clustering collapses 500 pairs to ~50 conversations → 2 gradient steps). The plan
  explicitly accepts this. Defensible for an HPO setup where the user controls
  subsample size, but the observability (Task 8) is blocked by the Task 8 tuple-return
  issue noted in Critical Issue #3.

---

## Verdict

**NEEDS REVISION**

Block on Critical Issues 1, 2, and 3 before execution:

1. Rewrite the Task 1 test to call `build_diff_aware_sft_trainer` and assert the
   returned trainer's `data_collator.inner_collator.completion_only_loss` is `False`,
   or run an end-to-end batch through the collator with no `assistant_masks` column
   and assert `(labels != -100).any()`. The current test bypasses the fix site entirely.

2. Decide: does the diff path want to train on ALL tokens (prompt + completion) with
   `completion_only_loss=False`, or only on assistant turns? If completion-only is
   desired, restore `_attach_assistant_masks` on the diff path (RCA's first alternative
   fix direction) and keep `completion_only_loss=True`. The current fix unblocks the
   zero-loss bug but may introduce a different regression (training on system/user
   prompts).

3. Update the two existing `_build_run_params` call sites in `test_trainer_mlflow.py`
   (lines 121 and 161) to include `total_steps=None` before running Task 8, and ensure
   the `_build_sft_config` tuple-return change has a test covering the `train_qlora`
   call site unpacking — not just a unit test on `_build_run_params` in isolation.
