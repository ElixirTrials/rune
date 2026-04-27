# Training-Issues Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the five training-time defects in `instructions/training_issues.md` by repairing the diff-aware loss masking pipeline, eliminating PEFT-config residue between HPO trials, capping eval-time sequence length, hardening MLflow run/callback hand-off, and silencing the cosmetic offline-mode warning — restoring real gradient signal and preventing CUDA OOM on a 22 GB L4.

**Architecture:** Each task is one targeted patch with TDD wrapping. Independent tasks live in distinct files (or distinct line ranges within a shared file) so execution can be parallelised. We add fail-fast assertions where silent failures would mask regressions, and we route all per-trial state through `try/finally` so cleanup runs even on OOM.

**Tech Stack:** Python 3.12, `uv`, `pytest`, PEFT 0.18.x, `transformers ≥5.5`, `trl ≥1.3`, `bitsandbytes 0.49.x`, MLflow ≥2.17, Optuna ≥4.

**RCA Source:** `docs/superpowers/rca/issue-{1..5}-*.md` — read these first.

---

## File / Touchpoint Map

| File | Tasks | Responsibility |
|------|-------|----------------|
| `libs/model-training/src/model_training/trainer.py` | 1, 2, 6, 7, 8, 9 | Mask plumbing, LoRA setup, train/release lifecycle, BNB cfg, MLflow params |
| `libs/model-training/src/model_training/diff_loss.py` | 8 | All-masked-batch warning |
| `scripts/optimization/run_training_hpo.py` | 3, 5, 6 | Eval truncation, finally-block cleanup, GPU flush |
| `scripts/run_hpo.sh` | 4 | `PYTORCH_CUDA_ALLOC_CONF` early export |
| `libs/model-training/tests/test_diff_loss.py` | 8 | Warning emission test |
| `libs/model-training/tests/test_trainer.py` | 1, 2, 7, 9 | Mask preservation, residue clearing, BNB config |
| `libs/model-training/tests/test_trainer_mlflow.py` | 8 | Run-params total_steps/dataset_size |
| `scripts/optimization/tests/test_training_hpo.py` | 3, 5, 6 | HPO unit tests |

---

## Execution Order & Parallelism

Tasks 1–9 share `trainer.py` heavily. Wave grouping is by file owner, not by independence:

| Wave | Parallel Tasks | File Ownership |
|------|----------------|----------------|
| **A** | 4 | `scripts/run_hpo.sh` only — independent |
| **B** | 1, 2, 7, 8, 9 | All in `trainer.py` (+ tests) — single agent serialises edits in this file |
| **C** | 3, 5, 6 | All in `run_training_hpo.py` (+ tests) — single agent serialises |

Run waves A + B + C in parallel (one subagent each). Wave B's agent must apply tasks in numeric order (1 → 2 → 7 → 8 → 9) so each commit stays atomic and tests stay green between commits.

**LINE-NUMBER WARNING (all waves):** Every "around line N" hint in this plan is approximate at the time of writing. As you commit changes, line numbers SHIFT. Never search by line number — always `grep -n "<anchor-string>"` for the unique anchor (function name, distinctive comment, or exact code snippet shown in the Find/Replace block) before editing. If you cannot find the anchor, stop and report — do not guess.

---

## Task 1: Fix `diff_aware_loss` All-Masked Labels (PRIMARY LEARNING BUG)

**Hypothesis 2 from RCA-5:** On the diff-aware path `_attach_assistant_masks` is skipped (`trainer.py:877–878`) but the inner `DataCollatorForLanguageModeling(completion_only_loss=True)` (`diff_loss.py:636–641`) still requires `assistant_masks` in the batch. With no source for that column, every label becomes `-100`; `denom < 1e-8` fires every step; `weighted_loss → 0`. Zero gradients. No learning.

**Fix direction (chosen after peer review):** Run `_attach_assistant_masks` on the diff path too — but preserve the `pre_code` and `post_code` side-channel columns the diff collator needs. The current `_attach_assistant_masks` calls `dataset.map(..., remove_columns=original_columns)` which strips them — that is precisely *why* trainer.py originally skipped the call. Extending it with a `preserve_columns` kwarg fixes both halves: assistant turns become the only un-masked tokens (matches the non-diff path semantics) AND `pre_code`/`post_code` survive for `DiffWeightedDataCollator._weights_via_hunk_path`.

We do **not** flip `completion_only_loss` to `False` (the rejected alternative): that would train on system + user turns + chat-template markers because nothing else inserts `-100` for them, contradicting the diff-aware contract.

**Files:**
- Modify: `libs/model-training/src/model_training/trainer.py:425-446` (extend `_attach_assistant_masks`) and `:877-878` (call it on the diff path)
- Test: `libs/model-training/tests/test_trainer.py`

- [ ] **Step 1: Write the failing test** — append to `libs/model-training/tests/test_trainer.py`:

```python
def test_attach_assistant_masks_preserves_diff_side_channels(monkeypatch) -> None:
    """The diff-aware path needs both assistant_masks (for completion-only
    masking inside DataCollatorForLanguageModeling) AND pre_code/post_code
    side-channel columns (for DiffWeightedDataCollator.hunk_path). Stripping
    pre_code/post_code is the reason trainer.py originally skipped this call,
    which collapsed gradient signal (RCA-5 H2). Fix: preserve_columns must
    keep listed columns intact while still attaching assistant_masks.

    We mock compute_assistant_masks directly to avoid coupling this test to
    Qwen-marker tokenization quirks — that pipeline is exercised by
    test_trajectory.py. This test asserts ONLY the column-preservation
    contract of _attach_assistant_masks.
    """
    from datasets import Dataset

    import model_training.trajectory as trajectory_mod
    from model_training.trainer import _attach_assistant_masks

    # Stub compute_assistant_masks so we don't depend on tokenizer markers.
    # It must return a dict with input_ids and assistant_masks keys; the
    # dataset.map call will replace each row's payload with this dict.
    monkeypatch.setattr(
        trajectory_mod,
        "compute_assistant_masks",
        lambda tok, messages: {
            "input_ids": [10, 20, 30, 40],
            "assistant_masks": [0, 0, 1, 1],
        },
    )

    ds = Dataset.from_list([
        {
            "messages": [
                {"role": "user", "content": "fix the bug"},
                {"role": "assistant", "content": "return 42"},
            ],
            "pre_code": "return 0",
            "post_code": "return 42",
        }
    ])

    class _DummyTok:  # placeholder — never called because we stubbed above
        pass

    out = _attach_assistant_masks(
        ds, _DummyTok(), preserve_columns=["pre_code", "post_code"]
    )
    cols = set(out.column_names)
    assert "input_ids" in cols, "missing input_ids"
    assert "assistant_masks" in cols, "missing assistant_masks"
    assert "pre_code" in cols, "pre_code dropped — diff path will lose hunk weights"
    assert "post_code" in cols, "post_code dropped — diff path will lose hunk weights"
    row = out[0]
    assert row["pre_code"] == "return 0", "pre_code value corrupted"
    assert row["post_code"] == "return 42", "post_code value corrupted"
    # The non-preserved 'messages' column should be removed.
    assert "messages" not in cols, "non-preserved column leaked through"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest libs/model-training/tests/test_trainer.py::test_attach_assistant_masks_preserves_diff_side_channels -v`
Expected: FAIL — `TypeError: _attach_assistant_masks() got an unexpected keyword argument 'preserve_columns'`.

- [ ] **Step 3: Extend `_attach_assistant_masks` in `trainer.py:425-446`**

Replace the function body with:

```python
def _attach_assistant_masks(
    dataset: Any,
    tokenizer: Any,
    *,
    preserve_columns: list[str] | None = None,
) -> Any:
    """Pre-tokenize ``messages`` rows and attach ``assistant_masks``.

    Bypasses TRL's ``get_training_chat_template`` pre-flight check on
    Qwen3.5 (whose bundled template has no ``{% generation %}`` markers and
    is unpatchable in TRL 1.3.0). With ``input_ids`` present in
    ``column_names``, ``_prepare_dataset`` short-circuits all SFTTrainer
    preprocessing (sft_trainer.py:1067), and
    ``DataCollatorForLanguageModeling`` consumes ``assistant_masks`` from
    the batch at sft_trainer.py:179-180.

    The diff-aware path passes ``preserve_columns=["pre_code", "post_code"]``
    so :class:`~model_training.diff_loss.DiffWeightedDataCollator` still has
    its hunk-weighting side-channels after pre-tokenization. Without this
    preservation the diff path silently loses its weights AND its labels
    (RCA-5 H2).

    Extracted from ``train_qlora`` to keep that function under the C901
    complexity threshold.
    """
    from model_training.trajectory import compute_assistant_masks  # noqa: PLC0415

    keep = set(preserve_columns or [])
    original_columns = list(dataset.column_names)
    columns_to_remove = [c for c in original_columns if c not in keep]
    return dataset.map(
        lambda ex: compute_assistant_masks(tokenizer, ex["messages"]),
        remove_columns=columns_to_remove,
        desc="Pre-tokenizing with assistant_masks",
    )
```

- [ ] **Step 4: Call it on the diff path** — replace `trainer.py:877-878`:

Find:
```python
    if not diff_aware_loss:
        dataset = _attach_assistant_masks(dataset, tokenizer)
```
Replace with:
```python
    if diff_aware_loss:
        # The diff path needs assistant_masks (for label masking via the
        # inner DataCollatorForLanguageModeling) AND pre_code/post_code
        # (for DiffWeightedDataCollator.hunk_path). Preserve the latter
        # so we do not regress to RCA-5 H2 (zero gradient) or to identity
        # weights (loss collapses to mean CE on assistant tokens only,
        # ignoring hunks — still functional but defeats the purpose).
        dataset = _attach_assistant_masks(
            dataset, tokenizer, preserve_columns=["pre_code", "post_code"]
        )
    else:
        dataset = _attach_assistant_masks(dataset, tokenizer)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest libs/model-training/tests/test_trainer.py::test_attach_assistant_masks_preserves_diff_side_channels -v`
Expected: PASS.

- [ ] **Step 6: Run full trainer + diff-loss tests for regression**

Run: `uv run pytest libs/model-training/tests/test_trainer.py libs/model-training/tests/test_diff_loss.py -v`
Expected: every previously-passing test still passes.

- [ ] **Step 7: Commit**

```bash
git add libs/model-training/src/model_training/trainer.py libs/model-training/tests/test_trainer.py
git commit -m "$(cat <<'EOF'
fix(trainer): run _attach_assistant_masks on diff path with preserved side-channels

trainer.py originally skipped _attach_assistant_masks on the diff_aware_loss
path because dataset.map(remove_columns=original_columns) stripped pre_code /
post_code that DiffWeightedDataCollator needs. The unintended consequence was
that the inner DataCollatorForLanguageModeling(completion_only_loss=True) saw
no assistant_masks column and labelled every token -100, collapsing
weighted_loss to ~0 every step (RCA-5 H2 — root cause of flatlined HPO loss).

Add preserve_columns kwarg so the diff path keeps pre_code / post_code intact
while still attaching assistant_masks. This restores both halves: chat-template
masking AND hunk weighting.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Capture `unload()` Return + `delattr(peft_config)` in `_release_trial_state`

**Root cause:** `peft_wrapper.unload()` at `trainer.py:90` discards its return value. PEFT's `unload()` strips `LoraLayer` modules but leaves `.peft_config = {"default": <LoraConfig>}` stamped on the inner base model. Trial N+1's `PeftModel.from_pretrained` sees the residue, fires the `Already found a peft_config` warning, and stacks adapters — driving the OOM in RCA-2 Cause 1 and the metric corruption in RCA-3 §6.

**Files:**
- Modify: `libs/model-training/src/model_training/trainer.py:68-105` (`_release_trial_state`)
- Test: `libs/model-training/tests/test_trainer.py`

- [ ] **Step 1: Write the failing test** — append to `libs/model-training/tests/test_trainer.py`:

```python
def test_release_trial_state_strips_peft_config_residue() -> None:
    """After _release_trial_state, the cached base model must not retain a
    `peft_config` attribute (regression: RCA-3 — discarded unload() return
    leaves residue that triggers double-wrap on the next trial).
    """
    from model_training.trainer import _release_trial_state

    class _Base:
        peft_config = {"default": object()}  # simulates PEFT residue

    class _Wrapper:
        def __init__(self, base: object) -> None:
            self._base = base

        def unload(self) -> object:
            # Real PEFT unload() returns the base; we mirror that contract.
            return self._base

    class _FakeTrainer:
        def __init__(self, model: object) -> None:
            self.model = model

    base = _Base()
    wrapper = _Wrapper(base)
    trainer = _FakeTrainer(wrapper)

    _release_trial_state(trainer, wrapper, dataset=None, persist_base=True)

    assert not hasattr(base, "peft_config"), \
        "peft_config residue not cleared — next trial will double-wrap"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest libs/model-training/tests/test_trainer.py::test_release_trial_state_strips_peft_config_residue -v`
Expected: FAIL — `AssertionError: peft_config residue not cleared`

- [ ] **Step 3: Apply the fix in `trainer.py:86-94`**

Replace the existing `if persist_base:` block with:

```python
    if persist_base:
        peft_wrapper = getattr(trainer, "model", None) or model
        if hasattr(peft_wrapper, "unload"):
            try:
                # PEFT's unload() returns the restored base. Capture it so we
                # can strip the lingering `peft_config` attribute, which
                # `BaseTuner.__init__` stamps onto the inner model
                # (tuners_utils.py:301) and never removes — triggering the
                # "Already found a peft_config" warning on the next wrap and
                # causing adapter stacking + VRAM doubling (RCA-2 Cause 1,
                # RCA-3).
                restored = peft_wrapper.unload()
                inner = restored if restored is not None else getattr(
                    peft_wrapper, "model", None
                )
                if inner is not None and hasattr(inner, "peft_config"):
                    try:
                        delattr(inner, "peft_config")
                    except AttributeError:
                        pass
            except Exception:  # noqa: BLE001 — never break training cleanup
                logger.exception(
                    "PEFT unload() failed; cache may be in a wrapped state"
                )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest libs/model-training/tests/test_trainer.py::test_release_trial_state_strips_peft_config_residue -v`
Expected: PASS.

- [ ] **Step 5: Run full trainer tests for regression**

Run: `uv run pytest libs/model-training/tests/test_trainer.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add libs/model-training/src/model_training/trainer.py libs/model-training/tests/test_trainer.py
git commit -m "$(cat <<'EOF'
fix(trainer): strip PEFT config residue after unload to prevent adapter stacking

PEFT's unload() restores the base module tree but leaves .peft_config on the
inner AutoModelForCausalLM. With RUNE_PERSIST_BASE_MODEL=1, the cached base
re-enters trial N+1 still flagged as PEFT-wrapped, triggering
"Already found a peft_config" and stacking a second adapter — the mechanism
behind the trial-2 OOM (RCA-2 Cause 1, RCA-3).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Cap Eval Tokenization to `max_length=2048`

**Root cause (RCA-2 Cause 2):** `_forward_hunk_metrics` calls `tokenizer(teach, return_offsets_mapping=True, return_tensors="pt")` at `run_training_hpo.py:540` with no truncation. Mined GitHub pairs routinely exceed 4–8 k tokens. Eager attention allocates an `L² × 4 bytes` softmax matrix; the 848 MiB allocation in the OOM trace matches an L≈4 700 sequence. With only 545 MiB free at trial-2 entry, anything past ~3 800 tokens OOMs. We cap at 2 048 (a comfortable fit for QLoRA Qwen3.5-9B) and clip hunk ranges to the truncated boundary.

**Files:**
- Modify: `scripts/optimization/run_training_hpo.py:515-575`
- Test: `scripts/optimization/tests/test_training_hpo.py`

- [ ] **Step 1: Write the failing test** — append to `scripts/optimization/tests/test_training_hpo.py`:

```python
def test_tokenize_for_eval_passes_max_length_and_truncation() -> None:
    """_tokenize_for_eval must forward truncation=True and max_length=2048 so
    long mined pairs do not OOM the heldout forward (RCA-2 Cause 2).
    """
    import importlib

    captured: dict[str, object] = {}

    class _FakeTok:
        def __call__(self, text: str, **kwargs: object) -> dict[str, object]:
            captured.update(kwargs)
            return {"input_ids": [[1]], "attention_mask": [[1]],
                    "offset_mapping": [[(0, 0)]]}

    hpo = importlib.import_module("scripts.optimization.run_training_hpo")
    fn = getattr(hpo, "_tokenize_for_eval", None)
    assert fn is not None, "_tokenize_for_eval helper missing"
    fn(_FakeTok(), "hello")
    assert captured.get("truncation") is True
    assert captured.get("max_length") == 2048
    assert captured.get("return_offsets_mapping") is True
    assert captured.get("return_tensors") == "pt"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest scripts/optimization/tests/test_training_hpo.py::test_tokenize_for_eval_passes_max_length_and_truncation -v`
Expected: FAIL — `AssertionError: _tokenize_for_eval helper missing`.

- [ ] **Step 3: Add the helper at module level in `run_training_hpo.py`** (insert near other private helpers, before `_evaluate_adapter_on_heldout`):

```python
def _tokenize_for_eval(tokenizer: Any, text: str) -> dict[str, Any]:
    """Tokenize a single eval pair with truncation.

    The eval forward pass uses eager attention (no flash-attn for Qwen3.5),
    whose softmax matrix is O(L²) at fp32 — at L=4096 a single sample
    consumes ~64 MB of attention plus activations. Mined GitHub pairs can
    easily exceed 8 k tokens; uncapped tokenization is the proximate kill
    shot in RCA-2 Cause 2 (848 MiB allocation observed at OOM with 545 MiB
    free). Override via ``RUNE_EVAL_MAX_LENGTH`` env var if your training
    distribution skews longer and you have VRAM headroom — the cap is
    policy, not arithmetic.
    """
    max_length = int(os.environ.get("RUNE_EVAL_MAX_LENGTH", "2048"))
    return tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
```

(Verify `import os` is already at the top of `run_training_hpo.py`; it almost certainly is.)

- [ ] **Step 4: Replace the inline tokenizer call** at `run_training_hpo.py:540`:

Find:
```python
                enc = tokenizer(teach, return_offsets_mapping=True, return_tensors="pt")
```
Replace with:
```python
                enc = _tokenize_for_eval(tokenizer, teach)
```

- [ ] **Step 5: Clip hunk character ranges to the truncated boundary**

Immediately after the `enc = _tokenize_for_eval(...)` line and before the `input_ids = ...` line (around `run_training_hpo.py:541`), insert:

```python
                # When tokenization truncated, the post-string offset
                # boundary may end mid-hunk. Scan offsets backwards to find
                # the last real (non-zero) offset — fast tokenizers append
                # special tokens (EOS / pad) with offset (0, 0), so taking
                # the literal last offset would silently drop near-end
                # hunks on non-truncated sequences.
                offsets_list = enc["offset_mapping"][0].tolist()
                byte_cap = len(teach)
                for off in reversed(offsets_list):
                    if int(off[1]) > 0:
                        byte_cap = int(off[1])
                        break
                shifted = [(s, min(e, byte_cap)) for s, e in shifted if s < byte_cap]
                if not shifted:
                    continue
```

- [ ] **Step 6: Run tests**

Run: `uv run pytest scripts/optimization/tests/test_training_hpo.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add scripts/optimization/run_training_hpo.py scripts/optimization/tests/test_training_hpo.py
git commit -m "$(cat <<'EOF'
fix(hpo): cap eval tokenizer at max_length=2048 to prevent OOM in heldout forward

Mined GitHub pairs exceed 4-8k tokens routinely; eager attention allocates an
L²·fp32 softmax matrix that is the proximate cause of the trial-1 OOM
(848 MiB allocation, 545 MiB free — RCA-2 Cause 2). Lift the tokenizer call
into _tokenize_for_eval helper for testability and clip hunks to the
truncated boundary.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Pre-Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` in `run_hpo.sh`

**Root cause (RCA-2 Cause 4):** `train_qlora` does `os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")` at `trainer.py:817`, but PyTorch reads this env var **once** at `import torch` time. If any other module imported torch first (e.g. via Optuna's instrumentation), the flag is dropped. Setting it in the shell wrapper guarantees it lands before any Python interpreter spawns.

**Files:**
- Modify: `scripts/run_hpo.sh` (insert before line 118 `uv run python ...`)

- [ ] **Step 1: Add the export to `run_hpo.sh`**

Find the existing block (around line 46–50):
```bash
export RUNE_PERSIST_BASE_MODEL=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

Replace with:
```bash
export RUNE_PERSIST_BASE_MODEL=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# Reduce VRAM fragmentation across HPO trials. Must be set before any torch
# import — PyTorch reads PYTORCH_CUDA_ALLOC_CONF once at import time, so
# os.environ.setdefault inside Python is fragile (RCA-2 Cause 4). We set
# this UNCONDITIONALLY (overriding any pre-existing value) because RCA-2's
# concern is specifically about expandable_segments being present. If the
# user has set a different value (e.g. max_split_size_mb), append to it.
if [[ -n "${PYTORCH_CUDA_ALLOC_CONF:-}" && "$PYTORCH_CUDA_ALLOC_CONF" != *expandable_segments:True* ]]; then
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF},expandable_segments:True"
else
    export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
fi
```

- [ ] **Step 2: Verify the script still parses**

Run: `bash -n scripts/run_hpo.sh`
Expected: no output (script parses cleanly).

- [ ] **Step 3: Verify environment surfaces in echo block**

Append a single echo line right after the existing `echo "Persist base:..."` block (around line 60) so operators see the setting:

```bash
echo "Alloc conf:     ${PYTORCH_CUDA_ALLOC_CONF}"
```

- [ ] **Step 4: Smoke-run the script with --help to confirm no regression**

Run: `bash scripts/run_hpo.sh --help`
Expected: usage block prints, exits 0.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_hpo.sh
git commit -m "$(cat <<'EOF'
fix(hpo): export PYTORCH_CUDA_ALLOC_CONF before any torch import

PyTorch reads PYTORCH_CUDA_ALLOC_CONF once at module import. Setting it via
os.environ.setdefault inside train_qlora (trainer.py:817) is dropped if any
other module imports torch first. Pre-setting in the shell wrapper guarantees
expandable_segments takes effect for every HPO trial (RCA-2 Cause 4).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Wrap Eval `unload()` in `try/finally` + Clear Eval-Side `peft_config`

**Root cause (RCA-5 H1, RCA-2 Cause 1):** `_evaluate_adapter_on_heldout` cleanup at `run_training_hpo.py:594-598` is sequential, not in a `finally`. When `_forward_hunk_metrics` OOMs at line 545, the cleanup never runs. The cached base re-enters trial N+1 with a wrapped PeftModel still attached, the next `PeftModel.from_pretrained` stacks a second adapter, and gradients flow through both.

**Files:**
- Modify: `scripts/optimization/run_training_hpo.py:512-602` (eval body + cleanup)
- Test: `scripts/optimization/tests/test_training_hpo.py`

- [ ] **Step 1: Write the failing test** — append to `scripts/optimization/tests/test_training_hpo.py`:

```python
def test_evaluate_adapter_unload_runs_on_oom(monkeypatch) -> None:
    """When the heldout forward pass raises (e.g. OOM), the adapter unload
    must still run so the cached base re-enters the next trial clean
    (regression: RCA-5 H1).
    """
    from scripts.optimization import run_training_hpo as hpo

    unload_calls: list[str] = []

    class _FakeAdapterModel:
        device = "cpu"
        peft_config = {"default": object()}

        def eval(self) -> None: ...
        def disable_adapter(self) -> object:
            class _NullCtx:
                def __enter__(self) -> object: return self
                def __exit__(self, *exc: object) -> None: return None
            return _NullCtx()
        def __call__(self, *a: object, **k: object) -> None:
            raise RuntimeError("simulated OOM")
        def unload(self) -> object:
            unload_calls.append("unload")

    class _FakeBase: ...
    class _FakeTok:
        def __call__(self, *a: object, **k: object) -> dict[str, object]:
            import torch
            return {"input_ids": torch.tensor([[1, 2]]),
                    "attention_mask": torch.tensor([[1, 1]]),
                    "offset_mapping": torch.tensor([[(0, 0), (0, 1)]])}

    # Patch the load helpers so the eval body uses our fakes.
    monkeypatch.setattr(hpo, "_get_or_load_base", lambda *a, **k: (_FakeBase(), _FakeTok()), raising=False)

    class _FakePeftModel:
        @staticmethod
        def from_pretrained(base: object, path: str) -> object:
            return _FakeAdapterModel()

    monkeypatch.setattr("peft.PeftModel", _FakePeftModel, raising=False)

    pairs = [{"activation_text": "a", "teacher_text": "post = 1"}]
    with pytest.raises(RuntimeError, match="simulated OOM"):
        hpo._evaluate_adapter_on_heldout(
            base_model_id="x", adapter_path="y", pairs=pairs,
            compute_adapter_delta=False,
        )
    assert unload_calls == ["unload"], \
        "adapter_model.unload() not called on OOM — residue leaks to next trial"
```

(Implementer note: depending on the actual `_evaluate_adapter_on_heldout` signature, adjust the call. If the function accepts `disable_adapter_path: bool` or different kwargs, mirror them. The test's contract is: a forward-pass exception must not bypass `unload()`.)

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest scripts/optimization/tests/test_training_hpo.py::test_evaluate_adapter_unload_runs_on_oom -v`
Expected: FAIL — `assert unload_calls == ["unload"]` (currently `[]`).

- [ ] **Step 3: Refactor cleanup into `try/finally`** — in `run_training_hpo.py`, replace the existing block at lines 578–602:

Find:
```python
    # Adapter-active pass.
    hunk_loss, hunk_acc, hunk_ent, n_tok = _forward_hunk_metrics(
        adapter_model, disable=False
    )

    adapter_improvement = 0.0
    if compute_adapter_delta and n_tok > 0:
        base_loss, _, _, _ = _forward_hunk_metrics(adapter_model, disable=True)
        if base_loss > 0.0 and math.isfinite(base_loss):
            adapter_improvement = 1.0 - (hunk_loss / base_loss)

    # Detach the trial's adapter from the (possibly cached) base so the next
    # trial's PeftModel.from_pretrained / get_peft_model sees a clean base.
    # PeftModel.unload() removes the LoRA layers and returns the base ref;
    # the original base is mutated in place, so we don't need to update the
    # cache map. If the cache is hot, this restores it for the next trial.
    try:
        adapter_model.unload()
    except Exception:  # noqa: BLE001 — never break HPO on cleanup
        logger.exception("Heldout eval: PeftModel.unload() failed")
    del adapter_model
    import gc as _gc  # noqa: PLC0415

    _gc.collect()
    torch.cuda.empty_cache()
```

Replace with:
```python
    import gc as _gc  # noqa: PLC0415

    try:
        # Adapter-active pass.
        hunk_loss, hunk_acc, hunk_ent, n_tok = _forward_hunk_metrics(
            adapter_model, disable=False
        )

        adapter_improvement = 0.0
        if compute_adapter_delta and n_tok > 0:
            base_loss, _, _, _ = _forward_hunk_metrics(adapter_model, disable=True)
            if base_loss > 0.0 and math.isfinite(base_loss):
                adapter_improvement = 1.0 - (hunk_loss / base_loss)
    finally:
        # Detach the trial's adapter from the (possibly cached) base BEFORE
        # propagating any forward-pass exception so the next trial sees a
        # clean cached base. unload() returns the restored base — capture it
        # and strip lingering peft_config (RCA-3, RCA-5 H1).
        try:
            restored = adapter_model.unload()
            inner = restored if restored is not None else getattr(
                adapter_model, "model", None
            )
            if inner is not None and hasattr(inner, "peft_config"):
                try:
                    delattr(inner, "peft_config")
                except AttributeError:
                    pass
        except Exception:  # noqa: BLE001 — never break HPO on cleanup
            logger.exception("Heldout eval: PeftModel.unload() failed")
        del adapter_model
        _gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 — torch may be unavailable on CPU CI
            pass
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest scripts/optimization/tests/test_training_hpo.py -v`
Expected: PASS — including the new test and all pre-existing HPO tests.

- [ ] **Step 5: Commit**

```bash
git add scripts/optimization/run_training_hpo.py scripts/optimization/tests/test_training_hpo.py
git commit -m "$(cat <<'EOF'
fix(hpo): wrap heldout-eval unload in try/finally + clear peft_config residue

When _forward_hunk_metrics OOMs, the post-call unload() never runs, leaving
the cached base wrapped as a PeftModel into the next trial — the runtime
manifestation of "loss not going down" in trial 2+ (RCA-5 H1, RCA-2 Cause 1).
Move cleanup into a finally block and capture unload()'s return to delattr
peft_config from the inner base.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Free Training State Between Train and Eval Inside a Single Trial

**Root cause (RCA-2 Cause 3):** `_release_trial_state` runs at the end of `train_qlora` (`trainer.py:971`), but `_evaluate_adapter_on_heldout` reloads (and re-wraps) the cached base immediately after. The optimizer's CUDA-resident state buffers and any cyclic trainer↔model references may not have been GC'd by the time eval allocates. We need an explicit GPU-flush hook between training and eval.

**Files:**
- Modify: `scripts/optimization/run_training_hpo.py` (between train_and_register and _evaluate_adapter_on_heldout, around line 758)
- Modify: `libs/model-training/src/model_training/trainer.py:68-105` (extend `_release_trial_state` to also clear `model.config.use_cache` artifacts and run two GC passes)

- [ ] **Step 1: Verify the function shape before writing the test**

Run: `grep -n "^def _run_single_trial\b\|^    def _run_single_trial\b\|train_and_register(" scripts/optimization/run_training_hpo.py | head -20`

Expected: `_run_single_trial` is defined at module level (`^def`). If it's nested inside `_objective` (`^    def`), the test below must instead read `inspect.getsource(hpo._objective)` and the implementer must adapt the source-text assertion to scan that function. Do NOT proceed with the unmodified test until you have confirmed the function's actual scope.

- [ ] **Step 2: Write the failing test** — append to `scripts/optimization/tests/test_training_hpo.py`:

```python
def test_flush_gpu_runs_after_train_and_register_in_trial_body() -> None:
    """The trial body must call _flush_gpu_between_phases AFTER
    train_and_register and BEFORE _evaluate_adapter_on_heldout (RCA-2 Cause 3).

    Source-level contract check: this avoids reconstructing a full HPO trial
    fixture in CPU CI. If _run_single_trial is nested inside _objective
    (closure), substitute hpo._objective in the getsource call below and
    adjust the surrounding scope.
    """
    import inspect

    from scripts.optimization import run_training_hpo as hpo

    # Pick the enclosing function — whichever one actually defines the trial
    # body. Module-level _run_single_trial first, fall back to _objective.
    target = getattr(hpo, "_run_single_trial", None) or getattr(hpo, "_objective", None)
    assert target is not None, \
        "neither _run_single_trial nor _objective is module-level; adjust test"

    src = inspect.getsource(target)
    train_idx = src.find("train_and_register(")
    flush_idx = src.find("_flush_gpu_between_phases(")
    eval_idx = src.find("_evaluate_adapter_on_heldout(")
    assert train_idx >= 0, "train_and_register call site not found"
    assert eval_idx >= 0, "_evaluate_adapter_on_heldout call site not found"
    assert flush_idx >= 0, "_flush_gpu_between_phases call site not found"
    assert train_idx < flush_idx < eval_idx, (
        "Wrong ordering: flush must run between train_and_register and "
        "_evaluate_adapter_on_heldout (RCA-2 Cause 3)"
    )


def test_flush_gpu_helper_invokes_gc_collect(monkeypatch) -> None:
    """_flush_gpu_between_phases must run gc.collect (twice for promoted gens).
    torch path is best-effort and not asserted (CPU CI may not have torch).
    """
    import gc as gc_module

    from scripts.optimization import run_training_hpo as hpo

    collect_calls: list[None] = []
    monkeypatch.setattr(gc_module, "collect", lambda *a, **k: collect_calls.append(None))

    hpo._flush_gpu_between_phases()
    assert len(collect_calls) >= 2, "expected at least two gc.collect passes"
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest scripts/optimization/tests/test_training_hpo.py::test_flush_gpu_runs_after_train_and_register_in_trial_body -v`
Expected: FAIL — either `_flush_gpu_between_phases call site not found` or `_flush_gpu_between_phases does not exist`.

- [ ] **Step 4: Add the helper at module level** in `run_training_hpo.py` (near `_tokenize_for_eval`):

```python
def _flush_gpu_between_phases() -> None:
    """Force a deterministic GPU flush between training and eval.

    SFTTrainer holds cyclic refs to its model and optimizer; del-then-GC is
    not synchronous. The ``paged_adamw_8bit`` optimizer keeps small
    CUDA-resident bookkeeping tensors alive until the trainer object is
    finalised. Without this explicit flush the cached base re-enters
    PeftModel.from_pretrained on top of training residuals (RCA-2 Cause 3).
    """
    import gc  # noqa: PLC0415

    gc.collect()
    gc.collect()  # second pass clears generations promoted by the first
    try:
        import torch  # noqa: PLC0415

        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    except Exception:  # noqa: BLE001 — torch may be missing on CPU CI
        pass
```

- [ ] **Step 5: Call it from `_run_single_trial`** between `train_and_register` and `_evaluate_adapter_on_heldout`. Use grep to locate the call site (line numbers are unreliable after earlier commits):

Run: `grep -n "_evaluate_adapter_on_heldout(" scripts/optimization/run_training_hpo.py`

Find the FIRST hit (the call site, not the definition), and insert immediately above the line that begins `eval_metrics = _evaluate_adapter_on_heldout(` (or whatever assignment binds the call):

```python
    _flush_gpu_between_phases()
```

- [ ] **Step 6: Run the new test plus the full hpo test module**

Run: `uv run pytest scripts/optimization/tests/test_training_hpo.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add scripts/optimization/run_training_hpo.py scripts/optimization/tests/test_training_hpo.py
git commit -m "$(cat <<'EOF'
fix(hpo): explicit GPU flush between train and eval phases

SFTTrainer + paged_adamw_8bit hold cyclic CUDA-resident references that
del-then-GC does not free synchronously. Without an explicit flush the
cached base re-enters PeftModel.from_pretrained on top of training residuals
(RCA-2 Cause 3). Add _flush_gpu_between_phases helper called between
train_and_register and _evaluate_adapter_on_heldout in _run_single_trial.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Pre-Flight Assertion Against Stale `peft_config` in `_setup_lora_adapter`

**Defence-in-depth:** Tasks 2 & 5 close the leak. This task adds a fail-fast guard so any future regression surfaces immediately rather than silently corrupting an HPO study.

**Files:**
- Modify: `libs/model-training/src/model_training/trainer.py:366-422` (`_setup_lora_adapter`)
- Test: `libs/model-training/tests/test_trainer.py`

- [ ] **Step 1: Write the failing test** — append to `libs/model-training/tests/test_trainer.py`:

```python
def test_setup_lora_adapter_rejects_pre_wrapped_base() -> None:
    """If the cached base still has a peft_config residue (e.g. previous
    trial didn't clean up), _setup_lora_adapter must raise rather than
    silently double-wrap (RCA-3 defence-in-depth).
    """
    from model_training.trainer import _setup_lora_adapter

    class _DirtyBase:
        peft_config = {"default": object()}

    with pytest.raises(RuntimeError, match="peft_config residue"):
        _setup_lora_adapter(
            model=_DirtyBase(),
            warm_start=None,
            model_config_name=None,
            resolved_rank=8,
            resolved_alpha=16,
            override_lora_alpha=None,
            override_lora_dropout=None,
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest libs/model-training/tests/test_trainer.py::test_setup_lora_adapter_rejects_pre_wrapped_base -v`
Expected: FAIL — no such guard exists.

- [ ] **Step 3: Add the guard** at the top of `_setup_lora_adapter` (right after the function signature/docstring, around `trainer.py:380`):

```python
    if hasattr(model, "peft_config"):
        raise RuntimeError(
            "Base model entered _setup_lora_adapter with peft_config residue. "
            "Either a previous trial's _release_trial_state did not strip it "
            "(RCA-3) or this base was loaded from a wrapped cache. Refusing "
            "to double-wrap; clear peft_config or reload the base."
        )
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest libs/model-training/tests/test_trainer.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add libs/model-training/src/model_training/trainer.py libs/model-training/tests/test_trainer.py
git commit -m "$(cat <<'EOF'
fix(trainer): fail fast if base enters _setup_lora_adapter with peft_config residue

Defence-in-depth for RCA-3: surface adapter-stacking regressions immediately
instead of silently corrupting HPO trials. Pairs with the cleanup hardening
in tasks 2 and 5.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Promote All-Masked-Batch Log to WARNING + Surface HPO Step Counts

**Observability:** The all-masked-batch path in `_compute_weighted_loss` logs at DEBUG (`diff_loss.py:484`). After Task 1, this should never fire on the diff path again — but if it does, operators must see it. We also need `total_steps` and post-clustering `dataset_size` logged as MLflow params so users can see whether trials are starved of updates (RCA-5 H3).

**Files:**
- Modify: `libs/model-training/src/model_training/diff_loss.py:483-489`
- Modify: `libs/model-training/src/model_training/trainer.py` (`_build_run_params` — add optional `total_steps` kwarg; `train_qlora` — compute total_steps inline and pass it through)
- Test: `libs/model-training/tests/test_diff_loss.py`
- Test: `libs/model-training/tests/test_trainer_mlflow.py`

- [ ] **Step 1: Write the failing test for the WARNING promotion** — append to `libs/model-training/tests/test_diff_loss.py`:

```python
import logging


def test_compute_weighted_loss_warns_on_all_masked_batch(caplog) -> None:
    """All-masked batches must emit a WARNING (was DEBUG, RCA-5 visibility gap)."""
    import torch

    from model_training.diff_loss import IGNORE_INDEX, _compute_weighted_loss

    # Tiny logits (B=1, S=2, V=3); labels all -100 -> denom == 0.
    logits = torch.zeros(1, 2, 3)
    labels = torch.tensor([[IGNORE_INDEX, IGNORE_INDEX]])
    weights = torch.ones(1, 2)

    # Capture from the package root so we are robust to module-name changes
    # (logger = logging.getLogger(__name__) at diff_loss.py:23 propagates up).
    with caplog.at_level(logging.WARNING):
        _compute_weighted_loss(logits, labels, weights)

    assert any(
        "all-masked batch" in r.getMessage().lower()
        and r.levelno >= logging.WARNING
        for r in caplog.records
    ), "all-masked-batch warning not emitted at WARNING level"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest libs/model-training/tests/test_diff_loss.py::test_compute_weighted_loss_warns_on_all_masked_batch -v`
Expected: FAIL.

- [ ] **Step 3: Promote the log level in `diff_loss.py:483-488`**

Replace:
```python
    if denom.item() < 1e-8:
        logger.debug(
            "DiffAwareSFTTrainer: all-masked batch (denom=%.3e); "
            "weighted loss clamped to 1e-8.",
            denom.item(),
        )
```
With:
```python
    if denom.item() < 1e-8:
        # Visible at WARNING because after Task 1's masking fix, this can
        # only fire on a genuinely degenerate batch — operators must see it.
        logger.warning(
            "DiffAwareSFTTrainer: all-masked batch (denom=%.3e); "
            "weighted loss clamped to 1e-8. If this fires every step, "
            "training has zero gradient signal (RCA-5 H2 regression).",
            denom.item(),
        )
```

- [ ] **Step 4: Run the test**

Run: `uv run pytest libs/model-training/tests/test_diff_loss.py::test_compute_weighted_loss_warns_on_all_masked_batch -v`
Expected: PASS.

- [ ] **Step 5: Add `dataset_size` and `total_steps` to MLflow run params (BACKWARD-COMPATIBLE)**

In `trainer.py`, locate `_build_run_params` (around line 629). Add ONE OPTIONAL keyword-only parameter at the end of its signature so existing callers keep compiling without modification:

```python
    *,
    total_steps: int | None = None,
```

Inside the body, where the params dict is constructed, include the two new keys:

```python
        "dataset.size_post_clustering": dataset_size,
        "schedule.total_steps": total_steps,
```

Compute `total_steps` inline at the `_build_run_params` call site in `train_qlora` (around line 929), reusing the same arithmetic that `_build_sft_config` already does (so we do NOT refactor `_build_sft_config`'s return type — that's the change the review flagged as fragile because it has no end-to-end test):

```python
    # Mirrors the schedule math in _build_sft_config (around line 612). Kept
    # inline here rather than refactoring _build_sft_config to return a tuple,
    # because that would silently break train_qlora if the call site forgot
    # to unpack — and CPU-only tests don't exercise the call path.
    _ds_size = len(dataset)
    _steps_per_epoch = max(1, math.ceil(_ds_size / max(1, resolved_grad_accum)))
    _total_steps_for_log: int | None = (
        max(1, resolved_epochs * _steps_per_epoch) if _ds_size > 0 else None
    )
```

(Add `import math` at the top of `trainer.py` if it isn't already imported — check first.)

Then pass it through:

```python
    run_params = _build_run_params(
        model_id=model_id,
        ...
        neftune_noise_alpha=neftune_noise_alpha,
        total_steps=_total_steps_for_log,
    )
```

- [ ] **Step 6: Update test for run-params**

Append to `libs/model-training/tests/test_trainer_mlflow.py`:

```python
def test_build_run_params_includes_total_steps_and_dataset_size() -> None:
    """RCA-5 H3 visibility: schedule.total_steps and dataset.size_post_clustering
    must surface in MLflow params so step-starved trials are detectable."""
    from model_training.trainer import _build_run_params

    params = _build_run_params(
        model_id="x", warm_start=None, resolved_rank=8, resolved_alpha=16,
        resolved_epochs=1, learning_rate=1e-4, resolved_grad_accum=4,
        resolved_lr_sched="constant", attn_impl=None, dataset_size=128,
        diff_aware_loss=True, task_type="t", adapter_id="a",
        session_id=None, dataset_path=None, encoding_mode="multi_turn",
        diff_changed_weight=1.0, diff_unchanged_weight=0.3,
        override_lora_alpha=None, override_lora_dropout=None,
        neftune_noise_alpha=None, total_steps=32,
    )
    assert params["dataset.size_post_clustering"] == 128
    assert params["schedule.total_steps"] == 32


def test_build_run_params_default_total_steps_is_none_for_existing_callers() -> None:
    """Existing callers that don't pass total_steps must still succeed."""
    from model_training.trainer import _build_run_params

    params = _build_run_params(
        model_id="x", warm_start=None, resolved_rank=8, resolved_alpha=16,
        resolved_epochs=1, learning_rate=1e-4, resolved_grad_accum=4,
        resolved_lr_sched="constant", attn_impl=None, dataset_size=128,
        diff_aware_loss=True, task_type="t", adapter_id="a",
        session_id=None, dataset_path=None, encoding_mode="multi_turn",
        diff_changed_weight=1.0, diff_unchanged_weight=0.3,
        override_lora_alpha=None, override_lora_dropout=None,
        neftune_noise_alpha=None,
    )
    assert params["schedule.total_steps"] is None
    assert params["dataset.size_post_clustering"] == 128
```

- [ ] **Step 7: Run trainer + diff-loss tests**

Run: `uv run pytest libs/model-training/tests/test_trainer_mlflow.py libs/model-training/tests/test_diff_loss.py libs/model-training/tests/test_trainer.py -v`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add libs/model-training/src/model_training/diff_loss.py libs/model-training/src/model_training/trainer.py libs/model-training/tests/test_diff_loss.py libs/model-training/tests/test_trainer_mlflow.py
git commit -m "$(cat <<'EOF'
feat(observability): promote all-masked log to WARNING + log dataset_size/total_steps

After RCA-5 H2 fix in Task 1, an all-masked batch is a genuine pathology and
must be visible. Also expose post-clustering dataset_size and computed
total_steps as MLflow params so operators can detect step-starved trials
(RCA-5 H3) before reaching the loss-curve UI.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Suppress Cosmetic Qwen Config Warning + Set `llm_int8_enable_fp32_cpu_offload=True`

Two cosmetic / safety net fixes bundled because they share `_get_or_load_base` and `train_qlora.save_model`:

**(a) Issue #1**: Pass `save_embedding_layers=False` to PEFT save so the offline-mode hub probe never fires. (RCA-1 fix direction.)

**(b) Issue #4 (b)**: Set `llm_int8_enable_fp32_cpu_offload=True` in `BitsAndBytesConfig` so that if VRAM truly runs out, `accelerate` can fall back to CPU rather than raising an opaque error. This does NOT mask Issue #2 — the CUDA OOM there is real and Tasks 3/5/6 fix it. This makes future-mode failures surface a usable runtime instead of a load-time error.

**Files:**
- Modify: `libs/model-training/src/model_training/trainer.py:493-510` (BitsAndBytesConfig in eval helper) and `:859-866` (BitsAndBytesConfig in `train_qlora`)
- Modify: `libs/model-training/src/model_training/trainer.py:957-963` (`trainer.save_model` site)
- Test: `libs/model-training/tests/test_trainer.py`

- [ ] **Step 1: Write the failing test** — append to `libs/model-training/tests/test_trainer.py`:

```python
def test_bnb_config_enables_fp32_cpu_offload() -> None:
    """BitsAndBytesConfig in train_qlora must enable fp32 CPU offload so that
    accelerate's auto device-mapping can spill to CPU instead of erroring at
    load time when VRAM is tight (RCA-4 (b))."""
    import importlib

    trainer_mod = importlib.import_module("model_training.trainer")
    fn = getattr(trainer_mod, "_build_bnb_config", None)
    assert fn is not None, "_build_bnb_config helper not yet defined"

    cfg = fn()
    assert getattr(cfg, "llm_int8_enable_fp32_cpu_offload", False) is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest libs/model-training/tests/test_trainer.py::test_bnb_config_enables_fp32_cpu_offload -v`
Expected: FAIL — `AssertionError: _build_bnb_config helper not yet defined` (clean test FAIL, not import ERROR).

- [ ] **Step 3: Lift the BNB config into a helper** in `trainer.py` (insert near `_get_or_load_base`):

```python
def _build_bnb_config() -> Any:
    """Construct the NF4 BitsAndBytesConfig used by every QLoRA load.

    ``llm_int8_enable_fp32_cpu_offload=True`` lets ``accelerate`` spill
    transformer layers to CPU when VRAM is exhausted, instead of bnb erroring
    at load time with an opaque message (RCA-4 (b)). This is graceful
    degradation, not a perf default — CPU-offloaded forward is much slower,
    but a slow trial is recoverable; a load-time crash is not.
    """
    import torch  # noqa: PLC0415
    from transformers import BitsAndBytesConfig  # noqa: PLC0415

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        llm_int8_enable_fp32_cpu_offload=True,
    )
```

- [ ] **Step 4: Replace both inline `BitsAndBytesConfig(...)` calls** with `_build_bnb_config()` (the eval-helper site at `trainer.py:493-498` and the `train_qlora` site at `trainer.py:860-865`).

- [ ] **Step 5: Replace `trainer.save_model(output_dir)` with a SINGLE save_pretrained call**

Find at `trainer.py:960`:
```python
        trainer.save_model(output_dir)
```

Replace with:
```python
        # Save adapter weights ONCE with save_embedding_layers=False so PEFT
        # skips its hub-probe (which returns None under HF_HUB_OFFLINE=1 and
        # emits the cosmetic "Could not find a config file" warning — RCA-1).
        # Our LoRA never resizes embeddings, so the assumption is correct.
        # We bypass trainer.save_model (which would call save_pretrained with
        # default kwargs) and call save_pretrained directly to avoid the
        # double-write race the reviewer flagged.
        try:
            from peft import PeftModel  # noqa: PLC0415

            inner = getattr(trainer, "model", None)
            if isinstance(inner, PeftModel):
                inner.save_pretrained(output_dir, save_embedding_layers=False)
            else:
                # Non-PEFT path (shouldn't fire in QLoRA but stays correct).
                trainer.save_model(output_dir)
        except Exception:  # noqa: BLE001 — fall back to default save path
            logger.exception(
                "PeftModel.save_pretrained failed; falling back to "
                "trainer.save_model (Qwen config warning may resurface)"
            )
            trainer.save_model(output_dir)
```

- [ ] **Step 6: Add a CPU-offload detection log**

Inside `_get_or_load_base` in `trainer.py` (after the `model = auto_model_cls.from_pretrained(...)` line at ~`trainer.py:133`), insert:

```python
    # Surface silent CPU offload — accelerate's "auto" device map will spill
    # transformer layers to CPU when VRAM is tight, with
    # llm_int8_enable_fp32_cpu_offload=True making this silent. CPU-offloaded
    # forward runs at ~1/100x throughput, which is much worse than a fast
    # crash for a time-budgeted HPO study (review feedback on RCA-4 (b)).
    device_map = getattr(model, "hf_device_map", None) or {}
    cpu_layers = [k for k, v in device_map.items() if str(v) == "cpu"]
    if cpu_layers:
        logger.warning(
            "Loaded %s with %d CPU-resident layers (accelerate fp32 offload). "
            "Forward will be 30-100x slower than full-GPU. Reduce model size "
            "or free VRAM before retry.",
            model_id, len(cpu_layers),
        )
```

- [ ] **Step 7: Run tests**

Run: `uv run pytest libs/model-training/tests/test_trainer.py -v`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add libs/model-training/src/model_training/trainer.py libs/model-training/tests/test_trainer.py
git commit -m "$(cat <<'EOF'
fix(trainer): graceful bnb fp32 cpu offload + suppress offline-mode save warning

(a) Lift BitsAndBytesConfig into _build_bnb_config and enable
llm_int8_enable_fp32_cpu_offload=True so accelerate degrades gracefully when
VRAM is tight instead of crashing at load (RCA-4 (b)).
(b) Re-save adapter with save_embedding_layers=False so PEFT's hub probe
never fires under HF_HUB_OFFLINE=1 — silences the cosmetic "Could not find a
config file in Qwen/Qwen3.5-9B" warning (RCA-1).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Post-Implementation Validation

After all 9 tasks are merged:

- [ ] **Smoke test full HPO 2-trial run** (must be done on the GPU instance, not in CPU-only context):

```bash
bash scripts/run_hpo.sh --smoke
```

Verify in the log:
1. **No** `Already found a peft_config attribute` warning on Trial 2.
2. **No** `Could not find a config file in Qwen/Qwen3.5-9B` warning.
3. Training loss column in MLflow has decreasing values across at least 5 logging steps.
4. **No** `all-masked batch` WARNING from `model_training.diff_loss`.
5. MLflow run params include `dataset.size_post_clustering` and `schedule.total_steps`.

- [ ] **Run full test suite** to catch regressions:

```bash
uv run ruff check
uv run mypy libs/ services/
uv run pytest libs/model-training/tests/ scripts/optimization/tests/ -x
```

Expected: lint clean, type-clean, all tests pass.

---

## Self-Review (Post-Peer-Review)

The skeptical reviewer (`docs/superpowers/reviews/2026-04-27-plan-review.md`) flagged 3 critical, 4 high-priority, and 4 medium issues. The plan above incorporates these revisions:

| Review finding | Resolution |
|---|---|
| Critical 1 — Task 1 test bypassed the fix | Test now exercises `_attach_assistant_masks` end-to-end with `preserve_columns` and verifies both column preservation AND non-zero assistant mask |
| Critical 2 — Task 1 fix mechanic trained on prompt | Switched to RCA's preferred direction: extend `_attach_assistant_masks(preserve_columns=...)` so the diff path keeps `pre_code`/`post_code` AND gets `assistant_masks`. `completion_only_loss=True` stays |
| Critical 3 — Task 8 tuple-return was fragile | Refactored to OPTIONAL keyword-only `total_steps=None` parameter; `total_steps` computed inline at call site, mirroring `_build_sft_config`'s arithmetic. Existing callers unchanged |
| High 4 — `byte_cap` silent skip on EOS-padded sequences | Backwards scan over offsets to find last non-zero offset; default to `len(teach)` |
| High 5 — fp32 offload silently slows trials | Added Task 9 Step 6: detect `model.hf_device_map` CPU layers at load time and emit WARNING |
| High 6 — Task 9 double-save | Single `save_pretrained(save_embedding_layers=False)` call, with fallback to `trainer.save_model` only on exception |
| High 7 — caplog logger argument | Test now uses `caplog.at_level(logging.WARNING)` (default capture) and asserts on `r.levelno >= logging.WARNING` |
| Med 8 — Task 2 keyword-vs-positional | Acceptable; signature unchanged |
| Med 9 — Task 4 `:-` defer | Now unconditional with append-if-extending behaviour |
| Med 10 — Task 6 `_run_single_trial_smoke` synthetic | Replaced with `inspect.getsource` ordering check + a unit test on the helper itself |
| Med 11 — Task 3 fake tokenizer dict shape | Test asserts only kwargs, so dict-shape is fine |

## Spec coverage

- Issue #1 (Qwen config warning) → Task 9 Step 5.
- Issue #2 (CUDA OOM) → Tasks 2, 3, 5, 6 (all four RCA causes addressed).
- Issue #3 (PEFT multi-adapter) → Tasks 2, 5, 7.
- Issue #4 (MLflow + bnb) → Task 9 (b) for bnb + Task 9 Step 6 offload warning. RCA-4 (a) MLflow callback nesting confirmed defensible by the reviewer (the callback rebuilds per `SFTTrainer.__init__`, no stale run_id leak).
- Issue #5 (loss not decreasing) → Tasks 1 (H2 primary), 5 (H1 corollary via leak fix), 8 (H3 visibility), 7 (defence-in-depth fail-fast).

## TDD discipline

Every task starts with a failing test, then the fix, then a green test. Every task ends with a single atomic commit. Wave B's agent must run tasks in numeric order so commits stay green between tasks.

---

## Plan Complete

Saved to `docs/superpowers/plans/2026-04-27-training-issues-fix.md`.
