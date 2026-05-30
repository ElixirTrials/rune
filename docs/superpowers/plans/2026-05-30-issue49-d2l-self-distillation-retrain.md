# Issue #49 — D2L Self-Distillation Retrain Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recover a non-collapsed HyperLoRA training path via privileged-context self-distillation (Sakana D2L) and produce a checkpoint proven to retrieve trajectory content by retrieval/contrast gates.

**Architecture:** Teacher = frozen base model with the trajectory in-context (`disable_adapter()`); student = base + hypernet-generated adapter with the trajectory removed from the prompt; top-K KL over the answer span, masked to diff tokens. Three anti-degeneracy fixes: init `scaler_B` non-zero, drop the L1 sink, diff-token masking. Discriminator-first: prove the loop on a synthetic needle corpus before touching the real S3 corpus.

**Tech Stack:** `uv`, Python 3.12, pytest, ruff, mypy (strict), PEFT, transformers, `ctx_to_lora` (Sakana doc-to-lora), existing `tools/diag_*.py` probes.

**Spec:** `docs/superpowers/specs/2026-05-30-issue49-d2l-self-distillation-retrain-design.md`

---

## Global execution rules

- [ ] Before any model load: `free -g`. Keep `offload_base=False`. This VM has ~15GB CPU RAM — moving the 18GB base model to CPU OOM-kills it.
- [ ] Long GPU jobs run under `tools/run_guarded.sh` (Task 1), never bare Python.
- [ ] Use `uv run` for every Python command. Install GPU extra with `uv sync --extra gpu`.
- [ ] After each slice: focused `uv run pytest …`, then `uv run ruff check .` and `uv run mypy src/`.
- [ ] Do NOT run Stage 1 (real corpus) until the Stage 0 synthetic gate passes (`real > zero` and `real > contradictory`).
- [ ] Magnitude signals (`scaler_B absmax`, ΔW norm, "output changed") are tripwires/diagnostics only — never promotion criteria.

## File structure

| File | Responsibility |
|---|---|
| `tools/run_guarded.sh` | RAM watchdog wrapper for GPU jobs (committed). |
| `src/rune/training/collapse_metrics.py` | Pure, GPU-import-free metric helpers (optimizer membership, grad-norm summary, diff_agreement, ΔW norm). |
| `src/rune/training/hypernet_distill.py` | D2L context-distillation entrypoint: teacher/student forward, top-K KL, diff-mask, negative contexts. |
| `src/rune/model/hypernetwork.py` | (modify) `scaler_B` init/reparam fix; strict-load key audit; `combine_lora`+`get_head_bias`. |
| `src/rune/model/wrapper.py` | (modify) `disable_adapter()` during activation extraction. |
| `tools/diag_synthetic_overfit.py` | Stage 0 synthetic NIAH overfit gate (GPU). |
| `src/rune/training/gate.py` | (modify) `evaluate_retrieval_gate()` content gates. |
| `src/rune/training/orchestrator.py` | (modify) Stage-2 → `hypernet_distill`; remove oracle stage + empty gate. |
| `src/rune/engine/graph.py` + templates | (modify) render inference trajectory in training format (§C). |
| **Removed** | `src/rune/training/oracle_cache.py`, `Round2TrainConfig`, `_run_oracle_training`, plain-SFT `run_distillation`/`to_sft_columns`. |

---

## Task 1: Commit the RAM watchdog

**Files:**
- Create: `tools/run_guarded.sh`
- Test: manual (shell)

- [ ] **Step 1: Write the watchdog script**

```bash
#!/usr/bin/env bash
# RAM watchdog: runs a python job under uv, kills it before the ~15GB VM OOMs.
# Usage: tools/run_guarded.sh <logfile> <python-script> [args...]
set -uo pipefail
LOG="${1:?logfile required}"; shift
SCRIPT="${1:?python script required}"; shift
THRESHOLD_KB="${RUNE_RAM_KILL_KB:-13500000}"   # ~13.5 GB RSS+cache ceiling

uv run python "$SCRIPT" "$@" >"$LOG" 2>&1 &
PID=$!
echo "guarded pid=$PID script=$SCRIPT log=$LOG threshold_kb=$THRESHOLD_KB"
while kill -0 "$PID" 2>/dev/null; do
  AVAIL_KB=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
  if [ "$AVAIL_KB" -lt $((16000000 - THRESHOLD_KB)) ]; then
    echo "WATCHDOG: MemAvailable ${AVAIL_KB}kB too low — killing $PID" | tee -a "$LOG"
    kill -9 "$PID" 2>/dev/null
    wait "$PID" 2>/dev/null
    exit 137
  fi
  sleep 2
done
wait "$PID"; exit $?
```

- [ ] **Step 2: Make executable and smoke it**

Run:
```bash
chmod +x tools/run_guarded.sh
tools/run_guarded.sh /tmp/guard_smoke.log -c "print('ok')" 2>&1 || true
echo "--- (the -c form is illustrative; real use passes a .py path) ---"
tools/run_guarded.sh /tmp/guard_smoke.log <(echo "print('ok')")
cat /tmp/guard_smoke.log
```
Expected: prints `ok`; watchdog loop exits 0.

- [ ] **Step 3: Commit**

```bash
git add tools/run_guarded.sh
git commit -m "feat(tools): commit RAM watchdog for guarded GPU jobs"
```

---

## Task 2: Pure collapse-metric helpers (CPU, no GPU imports)

**Files:**
- Create: `src/rune/training/collapse_metrics.py`
- Test: `tests/unit/test_collapse_metrics.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_collapse_metrics.py
import torch
from rune.training.collapse_metrics import (
    assert_optimizer_covers,
    diff_agreement,
    summarize_named_tensors,
)


def test_assert_optimizer_covers_flags_missing_scaler_b() -> None:
    p_in = torch.nn.Parameter(torch.zeros(2))
    p_missing = torch.nn.Parameter(torch.zeros(2))
    named = {"scaler_B": p_missing, "head": p_in}
    opt = torch.optim.SGD([p_in], lr=0.1)
    try:
        assert_optimizer_covers(named, opt)
        raise AssertionError("expected RuntimeError")
    except RuntimeError as exc:
        assert "scaler_B" in str(exc)


def test_assert_optimizer_covers_passes_when_all_present() -> None:
    p1 = torch.nn.Parameter(torch.zeros(2))
    p2 = torch.nn.Parameter(torch.zeros(2))
    opt = torch.optim.SGD([p1, p2], lr=0.1)
    assert_optimizer_covers({"a": p1, "b": p2}, opt)  # no raise


def test_diff_agreement_zero_when_student_equals_base() -> None:
    base = torch.tensor([1, 1, 1, 1])
    teacher = torch.tensor([1, 2, 3, 1])  # differs at positions 1,2
    student = base.clone()  # student == base everywhere
    # top1_agreement(student, teacher) is high (2/4), but diff_agreement must be 0.
    assert diff_agreement(student, teacher, base) == 0.0


def test_diff_agreement_one_when_student_matches_teacher_on_diffs() -> None:
    base = torch.tensor([1, 1, 1, 1])
    teacher = torch.tensor([1, 2, 3, 1])
    student = teacher.clone()
    assert diff_agreement(student, teacher, base) == 1.0


def test_summarize_named_tensors_reports_absmax() -> None:
    stats = summarize_named_tensors({"scaler_B": torch.tensor([0.0, -0.013, 0.005])})
    assert stats["scaler_B/absmax"] == 0.013
    assert "scaler_B/mean" in stats
    assert "scaler_B/l2" in stats
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/unit/test_collapse_metrics.py -q`
Expected: FAIL — `ModuleNotFoundError: rune.training.collapse_metrics`.

- [ ] **Step 3: Implement the module**

```python
# src/rune/training/collapse_metrics.py
"""Pure metric helpers for detecting hypernetwork adapter collapse.

No GPU imports at module load (CPU-importable invariant). torch is imported
lazily inside function bodies.
"""
from __future__ import annotations

from typing import Any


def assert_optimizer_covers(named_params: dict[str, Any], optimizer: Any) -> None:
    """Raise RuntimeError listing trainable params absent from the optimizer.

    Args:
        named_params: mapping of name -> nn.Parameter that must be optimized.
        optimizer: a torch optimizer whose param_groups are checked.
    """
    covered = {id(p) for group in optimizer.param_groups for p in group["params"]}
    missing = [name for name, p in named_params.items() if id(p) not in covered]
    if missing:
        raise RuntimeError(f"optimizer does not cover trainable params: {sorted(missing)}")


def summarize_named_tensors(named_tensors: dict[str, Any]) -> dict[str, float]:
    """Per-name mean/absmax/l2 stats for watched tensor groups."""
    out: dict[str, float] = {}
    for name, t in named_tensors.items():
        tf = t.detach().float()
        out[f"{name}/mean"] = float(tf.mean())
        out[f"{name}/absmax"] = float(tf.abs().max())
        out[f"{name}/l2"] = float(tf.norm())
    return out


def diff_agreement(student_top1: Any, teacher_top1: Any, base_top1: Any) -> float:
    """Fraction of diff positions where student matches teacher.

    Diff positions = where base_top1 != teacher_top1 (the tokens the trajectory
    is responsible for). Returns 0.0 when there are no diff positions.
    """
    mask = base_top1 != teacher_top1
    denom = int(mask.sum())
    if denom == 0:
        return 0.0
    agree = int(((student_top1 == teacher_top1) & mask).sum())
    return agree / denom
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/unit/test_collapse_metrics.py -q && uv run mypy src/rune/training/collapse_metrics.py`
Expected: PASS; mypy clean.

- [ ] **Step 5: Commit**

```bash
git add src/rune/training/collapse_metrics.py tests/unit/test_collapse_metrics.py
git commit -m "feat(training): pure collapse-metric helpers (diff_agreement, optimizer coverage)"
```

---

## Task 3: Static contract — scaler_B is trainable and survives load

**Files:**
- Modify: `src/rune/model/hypernetwork.py`
- Test: `tests/unit/test_hypernetwork_scaler_contract.py`

- [ ] **Step 1: Write failing test for the strict-load key audit**

```python
# tests/unit/test_hypernetwork_scaler_contract.py
import pytest
from rune.model.hypernetwork import audit_checkpoint_keys


def test_audit_checkpoint_keys_flags_dropped_scaler_b() -> None:
    model_keys = {"scaler_A.q_proj", "scaler_B.q_proj", "bias_A.q_proj", "head.weight"}
    ckpt_keys = {"scaler_A.q_proj", "head.weight"}  # scaler_B + bias_A missing
    missing = audit_checkpoint_keys(model_keys, ckpt_keys)
    assert "scaler_B.q_proj" in missing
    assert "bias_A.q_proj" in missing


def test_audit_checkpoint_keys_empty_when_all_present() -> None:
    keys = {"scaler_A.q_proj", "scaler_B.q_proj"}
    assert audit_checkpoint_keys(keys, keys) == set()
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/unit/test_hypernetwork_scaler_contract.py -q`
Expected: FAIL — `ImportError: cannot import name 'audit_checkpoint_keys'`.

- [ ] **Step 3: Add the audit helper and wire it into load**

Add to `src/rune/model/hypernetwork.py` (module level):

```python
def audit_checkpoint_keys(model_keys: set[str], ckpt_keys: set[str]) -> set[str]:
    """Return model params absent from the checkpoint among collapse-critical groups.

    Only watches scaler_*/bias_*/head keys — the ones a silent strict=False load
    would otherwise drop (issue #49 §D).
    """
    watched = tuple(("scaler_A", "scaler_B", "bias_A", "bias_B", "head"))
    relevant = {k for k in model_keys if any(w in k for w in watched)}
    return relevant - ckpt_keys
```

Then at the existing `hypernet.load_state_dict(weights, strict=False)` site (~`hypernetwork.py:267`), capture and warn on dropped keys:

```python
    missing = audit_checkpoint_keys(set(hypernet.state_dict().keys()), set(weights.keys()))
    hypernet.load_state_dict(weights, strict=False)
    if missing:
        logger.warning("checkpoint is missing collapse-critical keys: %s", sorted(missing))
```

- [ ] **Step 4: Run tests + lint**

Run: `uv run pytest tests/unit/test_hypernetwork_scaler_contract.py -q && uv run mypy src/rune/model/hypernetwork.py`
Expected: PASS; mypy clean.

- [ ] **Step 5: Commit**

```bash
git add src/rune/model/hypernetwork.py tests/unit/test_hypernetwork_scaler_contract.py
git commit -m "feat(model): audit strict-load for dropped scaler/bias/head keys (#49 §D)"
```

---

## Task 4: Fix the scaler_B gate (non-zero init / reparam)

**Files:**
- Modify: `src/rune/model/hypernetwork.py`
- Test: `tests/unit/test_scaler_b_init.py`

- [ ] **Step 1: Write failing test asserting non-inert init**

The hypernet ships `scaler_B` zero-init inside `ctx_to_lora`. Rather than patch the vendored package, we re-initialize after construction in our loader. Test the helper that does it:

```python
# tests/unit/test_scaler_b_init.py
import torch
from rune.model.hypernetwork import reinit_scaler_b_nonzero


class _FakeHypernet:
    def __init__(self) -> None:
        self.scaler_B = torch.nn.ParameterDict(
            {"q_proj": torch.nn.Parameter(torch.zeros((1, 2, 4, 1)))}
        )


def test_reinit_scaler_b_sets_ones() -> None:
    h = _FakeHypernet()
    reinit_scaler_b_nonzero(h, value=1.0)
    assert float(h.scaler_B["q_proj"].abs().min()) == 1.0
    assert h.scaler_B["q_proj"].requires_grad
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/unit/test_scaler_b_init.py -q`
Expected: FAIL — `cannot import name 'reinit_scaler_b_nonzero'`.

- [ ] **Step 3: Implement the re-init helper and call it post-load (train path only)**

Add to `src/rune/model/hypernetwork.py`:

```python
def reinit_scaler_b_nonzero(hypernet: Any, value: float = 1.0) -> None:
    """Re-initialize scaler_B away from the zero collapse basin (issue #49 §A).

    ctx_to_lora zero-inits scaler_B, so B = B_raw * scaler_B = 0 and B_raw gets
    no gradient. Setting scaler_B to a non-zero constant (mirroring scaler_A's
    ones-init) makes the adapter identity-active at init with gradient flowing to
    both B_raw and the gate. Call ONLY when (re)training, never when loading a
    trained checkpoint (its learned scaler_B must be preserved).
    """
    import torch  # noqa: PLC0415

    if not hasattr(hypernet, "scaler_B"):
        return
    with torch.no_grad():
        for name in list(hypernet.scaler_B.keys()):
            hypernet.scaler_B[name].fill_(value)
```

This helper is invoked by the training entrypoint (Task 6), not by inference load.

- [ ] **Step 4: Run tests + lint**

Run: `uv run pytest tests/unit/test_scaler_b_init.py -q && uv run mypy src/rune/model/hypernetwork.py`
Expected: PASS; mypy clean.

- [ ] **Step 5: Commit**

```bash
git add src/rune/model/hypernetwork.py tests/unit/test_scaler_b_init.py
git commit -m "feat(model): reinit scaler_B non-zero to escape collapse basin (#49 §A)"
```

---

## Task 5: Diff-mask + top-K KL loss primitives (CPU-testable)

**Files:**
- Create: `src/rune/training/hypernet_distill.py`
- Test: `tests/unit/test_hypernet_distill.py`

- [ ] **Step 1: Write failing tests for pure loss primitives**

```python
# tests/unit/test_hypernet_distill.py
import torch
from rune.training.hypernet_distill import (
    compute_diff_positions,
    topk_kl_loss,
)


def test_compute_diff_positions_masks_to_labeled_disagreements() -> None:
    base_top1 = torch.tensor([5, 5, 5, 5])
    teacher_top1 = torch.tensor([5, 9, 7, 5])  # differ at 1,2
    labels = torch.tensor([1, 1, -100, 1])     # pos 2 unsupervised
    mask = compute_diff_positions(base_top1, teacher_top1, labels)
    assert mask.tolist() == [False, True, False, False]


def test_topk_kl_loss_zero_when_student_equals_teacher() -> None:
    teacher_logits = torch.randn(3, 10)
    # student identical -> KL ~ 0
    loss = topk_kl_loss(teacher_logits.clone(), teacher_logits, k=5)
    assert float(loss) < 1e-5


def test_topk_kl_loss_positive_when_distributions_differ() -> None:
    teacher_logits = torch.zeros(3, 10)
    teacher_logits[:, 0] = 10.0  # teacher confident on token 0
    student_logits = torch.zeros(3, 10)
    student_logits[:, 1] = 10.0  # student confident on token 1
    loss = topk_kl_loss(student_logits, teacher_logits, k=5)
    assert float(loss) > 0.5
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/unit/test_hypernet_distill.py -q`
Expected: FAIL — module/functions missing.

- [ ] **Step 3: Implement loss primitives**

```python
# src/rune/training/hypernet_distill.py
"""D2L privileged-context self-distillation for the HyperLoRA hypernetwork.

Teacher = frozen base model with the trajectory in-context (adapters disabled);
student = base + generated adapter with the trajectory removed from the prompt.
Loss = top-K KL over the answer span, masked to diff tokens (where teacher != base).

GPU imports are deferred; only pure tensor helpers are import-safe.
"""
from __future__ import annotations

from typing import Any

IGNORE_INDEX = -100


def compute_diff_positions(base_top1: Any, teacher_top1: Any, labels: Any) -> Any:
    """Boolean mask: supervised positions where base and teacher top-1 disagree."""
    return (labels != IGNORE_INDEX) & (base_top1 != teacher_top1)


def topk_kl_loss(student_logits: Any, teacher_logits: Any, k: int = 50) -> Any:
    """KL(teacher || student) over the teacher's top-K tokens, mean over rows.

    Args:
        student_logits: [N, V] student logits at supervised positions.
        teacher_logits: [N, V] teacher logits at the same positions.
        k: number of top teacher tokens to match.
    """
    import torch  # noqa: PLC0415

    k = min(k, teacher_logits.shape[-1])
    topk_vals, topk_idx = teacher_logits.topk(k, dim=-1)
    t_denom = torch.logsumexp(teacher_logits.float(), dim=-1, keepdim=True)
    teacher_p = (topk_vals.float() - t_denom).exp()  # [N, K]
    s_denom = torch.logsumexp(student_logits.float(), dim=-1, keepdim=True)
    student_logq = student_logits.float().gather(-1, topk_idx) - s_denom  # [N, K]
    return -(teacher_p * student_logq).sum(dim=-1).mean()
```

- [ ] **Step 4: Run tests + lint**

Run: `uv run pytest tests/unit/test_hypernet_distill.py -q && uv run mypy src/rune/training/hypernet_distill.py`
Expected: PASS; mypy clean.

- [ ] **Step 5: Commit**

```bash
git add src/rune/training/hypernet_distill.py tests/unit/test_hypernet_distill.py
git commit -m "feat(training): top-K KL + diff-mask loss primitives for D2L distillation"
```

---

## Task 6: One-step synthetic gradient test (the GPU-edit gate)

**Files:**
- Modify: `src/rune/training/hypernet_distill.py`
- Test: `tests/unit/test_hypernet_distill_step.py`

- [ ] **Step 1: Write failing test using tiny fake modules**

```python
# tests/unit/test_hypernet_distill_step.py
import torch
from rune.training.hypernet_distill import distill_step_loss


def test_distill_step_loss_nonzero_and_backprops_to_scaler() -> None:
    torch.manual_seed(0)
    n, v = 4, 16
    teacher_logits = torch.zeros(n, v)
    teacher_logits[:, 3] = 8.0                 # teacher wants token 3
    base_top1 = torch.zeros(n, dtype=torch.long)  # base wants token 0 -> all diff
    teacher_top1 = torch.full((n,), 3, dtype=torch.long)
    labels = torch.ones(n, dtype=torch.long)

    scaler_b = torch.nn.Parameter(torch.ones(1))
    student_logits = torch.zeros(n, v) + scaler_b * 0.0  # depends on scaler_b
    student_logits = student_logits.clone()
    student_logits[:, 3] = scaler_b * 1.0       # student logit on token 3 scales with gate

    loss = distill_step_loss(student_logits, teacher_logits, base_top1, teacher_top1, labels)
    assert float(loss) > 0.0
    loss.backward()
    assert scaler_b.grad is not None and float(scaler_b.grad.abs().sum()) > 0.0
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/unit/test_hypernet_distill_step.py -q`
Expected: FAIL — `distill_step_loss` missing.

- [ ] **Step 3: Implement the masked step loss**

Append to `src/rune/training/hypernet_distill.py`:

```python
def distill_step_loss(
    student_logits: Any,
    teacher_logits: Any,
    base_top1: Any,
    teacher_top1: Any,
    labels: Any,
    k: int = 50,
) -> Any:
    """Top-K KL restricted to diff positions (base != teacher on supervised tokens).

    Returns a scalar loss. If there are no diff positions, returns 0 (no signal).
    """
    import torch  # noqa: PLC0415

    mask = compute_diff_positions(base_top1, teacher_top1, labels)
    if int(mask.sum()) == 0:
        return student_logits.sum() * 0.0
    return topk_kl_loss(student_logits[mask], teacher_logits[mask], k=k)
```

- [ ] **Step 4: Run tests + lint**

Run: `uv run pytest tests/unit/test_hypernet_distill_step.py -q && uv run mypy src/rune/training/hypernet_distill.py`
Expected: PASS; mypy clean.

- [ ] **Step 5: Commit**

```bash
git add src/rune/training/hypernet_distill.py tests/unit/test_hypernet_distill_step.py
git commit -m "feat(training): diff-masked distill step loss + gradient unit test"
```

---

## Task 7: Disable-adapter during activation extraction (§D contamination)

**Files:**
- Modify: `src/rune/model/hypernetwork.py` (`extract_activations_with_model`)
- Test: `tests/unit/test_activation_extraction_disable_adapter.py`

- [ ] **Step 1: Write failing test with a fake PEFT model**

```python
# tests/unit/test_activation_extraction_disable_adapter.py
from contextlib import contextmanager
from unittest.mock import MagicMock
import torch
from rune.model.hypernetwork import extract_activations_with_model


def _fake_model(with_disable: bool):
    m = MagicMock()
    m.parameters.return_value = iter([torch.zeros(1)])
    out = MagicMock()
    out.hidden_states = [torch.zeros(1, 3, 8) for _ in range(4)]
    m.return_value = out
    m.__call__ = lambda **kw: out
    if with_disable:
        called = {"n": 0}
        @contextmanager
        def _dis():
            called["n"] += 1
            yield
        m.disable_adapter = _dis
        m._disable_calls = called
    else:
        del m.disable_adapter
    return m


def test_disable_adapter_used_when_available() -> None:
    tok = MagicMock()
    tok.return_value = {"input_ids": torch.zeros(1, 3, dtype=torch.long),
                        "attention_mask": torch.ones(1, 3, dtype=torch.long)}
    model = _fake_model(with_disable=True)
    extract_activations_with_model("ctx", model, tok, layer_indices=[0, 1])
    assert model._disable_calls["n"] == 1


def test_non_peft_model_still_extracts() -> None:
    tok = MagicMock()
    tok.return_value = {"input_ids": torch.zeros(1, 3, dtype=torch.long),
                        "attention_mask": torch.ones(1, 3, dtype=torch.long)}
    model = _fake_model(with_disable=False)
    feats, mask = extract_activations_with_model("ctx", model, tok, layer_indices=[0, 1])
    assert feats is not None and mask is not None
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/unit/test_activation_extraction_disable_adapter.py -q`
Expected: FAIL — `disable_adapter` not called (no contextmanager wrap yet).

- [ ] **Step 3: Wrap extraction in disable_adapter when present**

In `extract_activations_with_model` (`hypernetwork.py` ~L271-307), wrap the forward:

```python
    import contextlib  # noqa: PLC0415

    ctx = model.disable_adapter() if hasattr(model, "disable_adapter") else contextlib.nullcontext()
    with torch.no_grad(), ctx:
        outputs = model(**inputs, output_hidden_states=True, use_cache=False)
```

- [ ] **Step 4: Run tests + lint**

Run: `uv run pytest tests/unit/test_activation_extraction_disable_adapter.py -q && uv run mypy src/rune/model/hypernetwork.py`
Expected: PASS; mypy clean.

- [ ] **Step 5: Commit**

```bash
git add src/rune/model/hypernetwork.py tests/unit/test_activation_extraction_disable_adapter.py
git commit -m "fix(model): extract activations under disable_adapter to avoid contamination (#49 §D)"
```

---

## Task 8: combine_lora + head bias in PEFT export (§D)

**Files:**
- Modify: `src/rune/model/hypernetwork.py` (`generate_adapter_weights`, `_to_peft_state_dict`)
- Test: `tests/unit/test_hypernetwork_peft_mapping.py`

- [ ] **Step 1: Write failing PEFT shape/key test**

```python
# tests/unit/test_hypernetwork_peft_mapping.py
import torch
from rune.model.hypernetwork import _to_peft_state_dict


def test_peft_keys_match_expected_pattern_and_no_truncation() -> None:
    r, d = 4, 8
    lora_dict = {"q_proj": {"A": torch.randn(1, 2, r, d), "B": torch.randn(1, 2, d, r)}}
    sd = _to_peft_state_dict(lora_dict, layer_indices=[0, 1], target_modules=["q_proj"])
    a_keys = [k for k in sd if k.endswith("lora_A.weight")]
    b_keys = [k for k in sd if k.endswith("lora_B.weight")]
    assert len(a_keys) == 2 and len(b_keys) == 2
    # B must be transposed to [out, r]; A stays [r, in]
    assert sd[a_keys[0]].shape == (r, d)
    assert sd[b_keys[0]].shape == (d, r)
```

- [ ] **Step 2: Run to verify failure or current behavior**

Run: `uv run pytest tests/unit/test_hypernetwork_peft_mapping.py -q`
Expected: PASS if existing mapping already matches; FAIL otherwise. If PASS, this test locks current behavior before adding bias. Proceed to add the bias-rank contract test below.

- [ ] **Step 3: Add bias-merge contract (rank expansion must be explicit)**

Add to the same test file:

```python
import pytest
from rune.model.hypernetwork import merge_head_bias_rank


def test_merge_head_bias_rank_raises_on_rank_mismatch() -> None:
    # Combining a rank-b bias into a rank-r adapter changes effective rank;
    # the PEFT config rank must match or we must raise (no silent misapply).
    with pytest.raises(ValueError, match="rank"):
        merge_head_bias_rank(adapter_rank=4, bias_rank=2, peft_config_rank=4)


def test_merge_head_bias_rank_ok_when_config_matches_combined() -> None:
    assert merge_head_bias_rank(adapter_rank=4, bias_rank=2, peft_config_rank=6) == 6
```

Implement in `hypernetwork.py`:

```python
def merge_head_bias_rank(adapter_rank: int, bias_rank: int, peft_config_rank: int) -> int:
    """Validate that the PEFT adapter rank accommodates bias concatenation.

    combine_lora concatenates the head bias as extra rank slices, so the effective
    rank becomes adapter_rank + bias_rank. The hot-swapped PEFT adapter must be
    created with that rank, or weights misapply silently. Returns the required rank.
    """
    required = adapter_rank + bias_rank
    if peft_config_rank != required:
        raise ValueError(
            f"PEFT rank {peft_config_rank} != adapter+bias rank {required}; "
            "recreate the PEFT adapter at the combined rank before hotswap"
        )
    return required
```

Wire `combine_lora` + `get_head_bias()` into `generate_adapter_weights` only when `hypernet.config.use_bias` is set, gating on `merge_head_bias_rank` so a mismatch raises before hotswap.

- [ ] **Step 4: Run tests + lint**

Run: `uv run pytest tests/unit/test_hypernetwork_peft_mapping.py -q && uv run mypy src/rune/model/hypernetwork.py`
Expected: PASS; mypy clean.

- [ ] **Step 5: Commit**

```bash
git add src/rune/model/hypernetwork.py tests/unit/test_hypernetwork_peft_mapping.py
git commit -m "feat(model): combine_lora + head-bias merge with explicit rank contract (#49 §D)"
```

---

## Task 9: Retrieval/contrast gate evaluator (content, not magnitude)

**Files:**
- Modify: `src/rune/training/gate.py`
- Test: `tests/unit/test_gate.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_gate.py (add to existing)
from rune.training.gate import evaluate_retrieval_gate


def _probe(real, zero, shuffled, contra, cosine, diff_ag):
    return {
        "real_hit_rate": real, "zero_hit_rate": zero,
        "shuffled_hit_rate": shuffled, "contradictory_hit_rate": contra,
        "adapter_cosine": cosine, "diff_agreement": diff_ag, "scaler_b_absmax": 0.2,
    }


def test_gate_passes_when_real_beats_all_controls() -> None:
    res = evaluate_retrieval_gate(_probe(0.8, 0.1, 0.1, 0.05, 0.4, 0.6))
    assert res.passed


def test_gate_fails_on_generic_perturbation() -> None:
    # real == zero == contra but output changed (cosine moved) -> must FAIL
    res = evaluate_retrieval_gate(_probe(0.1, 0.1, 0.1, 0.1, 0.4, 0.0))
    assert not res.passed


def test_gate_fails_when_adapters_near_identical() -> None:
    res = evaluate_retrieval_gate(_probe(0.8, 0.1, 0.1, 0.05, 0.999, 0.6))
    assert not res.passed
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/unit/test_gate.py -q`
Expected: FAIL — `evaluate_retrieval_gate` missing.

- [ ] **Step 3: Implement the gate**

```python
# add to src/rune/training/gate.py
from dataclasses import dataclass

COSINE_MAX = 0.95          # distinct trajectories must diverge below this
DIFF_AGREEMENT_MIN = 0.5


@dataclass(frozen=True)
class RetrievalGateResult:
    passed: bool
    reasons: tuple[str, ...]


def evaluate_retrieval_gate(probe: dict[str, float]) -> RetrievalGateResult:
    """Content-based promotion gate. Magnitude (scaler_b_absmax) is ignored here."""
    reasons: list[str] = []
    if not probe["real_hit_rate"] > probe["zero_hit_rate"]:
        reasons.append("real_hit_rate <= zero_hit_rate")
    if not probe["real_hit_rate"] > probe["shuffled_hit_rate"]:
        reasons.append("real_hit_rate <= shuffled_hit_rate")
    if not probe["real_hit_rate"] > probe["contradictory_hit_rate"]:
        reasons.append("real_hit_rate <= contradictory_hit_rate")
    if not probe["adapter_cosine"] < COSINE_MAX:
        reasons.append("adapters near-identical (cosine too high)")
    if not probe["diff_agreement"] >= DIFF_AGREEMENT_MIN:
        reasons.append("diff_agreement below threshold")
    return RetrievalGateResult(passed=len(reasons) == 0, reasons=tuple(reasons))
```

- [ ] **Step 4: Run tests + lint**

Run: `uv run pytest tests/unit/test_gate.py -q && uv run mypy src/rune/training/gate.py`
Expected: PASS; mypy clean.

- [ ] **Step 5: Commit**

```bash
git add src/rune/training/gate.py tests/unit/test_gate.py
git commit -m "feat(training): content-based retrieval/contrast promotion gate (#49 §B)"
```

---

## Task 10: Remove the dead oracle + plain-SFT paths (lean/DRY)

**Files:**
- Delete: `src/rune/training/oracle_cache.py`
- Modify: `src/rune/training/config.py` (drop `Round2TrainConfig`)
- Modify: `src/rune/training/orchestrator.py` (drop `_run_oracle_training`, empty-gate placeholder)
- Modify: `src/rune/training/d2l_train.py` (drop `to_sft_columns`, repurpose/remove `run_distillation`)
- Test: `tests/unit/test_orchestrator.py`

- [ ] **Step 1: Write failing test that Stage-2 dispatches to hypernet_distill**

```python
# tests/unit/test_orchestrator.py (replace oracle-stage assertions)
from unittest.mock import patch
from rune.training.orchestrator import _run_hypernetwork_distillation


def test_stage2_dispatches_to_hypernet_distill() -> None:
    with patch("rune.training.hypernet_distill.run_hypernet_distillation") as m:
        _run_hypernetwork_distillation(config=object())
        m.assert_called_once()
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/unit/test_orchestrator.py::test_stage2_dispatches_to_hypernet_distill -q`
Expected: FAIL — still dispatches to `run_distillation`.

- [ ] **Step 3: Delete dead code and rewire**

```bash
git rm src/rune/training/oracle_cache.py
```
- Remove `Round2TrainConfig` from `config.py` and any import of it.
- In `orchestrator.py`: delete `_run_oracle_training`; change `_run_hypernetwork_distillation` to `from rune.training.hypernet_distill import run_hypernet_distillation`; remove the empty-`{}`-score success-gate placeholder (gate is invoked from the bench path in Task 13).
- In `d2l_train.py`: remove `to_sft_columns` and the SFT `run_distillation` (or reduce the file to nothing and `git rm` it if fully replaced).
- Fix all references that mypy/pytest surface.

Add a `run_hypernet_distillation(config)` stub in `hypernet_distill.py` (full body in Task 11):

```python
def run_hypernet_distillation(config: Any) -> None:
    """Stage-2 entrypoint (D2L context distillation). Implemented in Task 11."""
    raise NotImplementedError("implemented in Task 11")
```

- [ ] **Step 4: Run unit suite + lint + types**

Run: `uv run pytest tests/unit/ -q && uv run ruff check . && uv run mypy src/`
Expected: PASS (update/delete any tests referencing removed symbols — e.g. `test_oracle_cache.py`, oracle assertions in `test_config.py`).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor(training): remove dead oracle + plain-SFT paths; Stage-2 -> hypernet_distill"
```

---

## Task 11: D2L training loop (GPU) wired into hypernet_distill

**Files:**
- Modify: `src/rune/training/hypernet_distill.py` (`run_hypernet_distillation`)
- Test: covered by Task 6 unit gate + Task 12 GPU smoke (no new CPU unit beyond a config-parse test)

- [ ] **Step 1: Add a CPU config-parse test**

```python
# tests/unit/test_hypernet_distill_config.py
from rune.training.hypernet_distill import DistillConfig


def test_distill_config_defaults() -> None:
    cfg = DistillConfig(corpus_path="/tmp/x.jsonl", checkpoint_dir="/tmp/ck")
    assert cfg.l1_reg_coef == 0.0          # L1 sink disabled (#49 §A)
    assert cfg.scaler_b_init == 1.0
    assert cfg.topk == 50
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/unit/test_hypernet_distill_config.py -q`
Expected: FAIL — `DistillConfig` missing.

- [ ] **Step 3: Implement config + loop**

Add `DistillConfig` (pydantic/dataclass mirroring fields) and the loop body. The loop, per slice (deferred GPU imports):
1. Load base model (`AutoModelForCausalLM`, bf16, flash-attn), tokenizer; `free -g` logged.
2. Load hypernet via `load_hypernetwork`; call `reinit_scaler_b_nonzero(hypernet, cfg.scaler_b_init)`.
3. Build optimizer over `[p for p in hypernet.parameters() if p.requires_grad]`; call `assert_optimizer_covers({"scaler_B": <one scaler param>, ...}, opt)`.
4. For each record: build teacher inputs (context + answer), student inputs (answer only); teacher forward under `disable_adapter()` + `no_grad`, take top-K; generate adapter from context activations, hot-swap, student forward; compute `distill_step_loss` over diff positions; `l1 = cfg.l1_reg_coef * ‖gen‖₁` (0 by default).
5. Backward, clip, step. Every N steps log `summarize_named_tensors({"scaler_B": ...})`, per-component grad norms, `diff_agreement`, ΔW norm, skipped-record count → JSONL.
6. Periodically save checkpoint (`hypernet_state_dict`, config, step).

Reuse `ctx_to_lora` primitives where possible (DRY): `combine_lora`, `generate_weights`, the `disable_adapter` pattern from `CtxDistillModel`.

- [ ] **Step 4: Run CPU config test + lint + types**

Run: `uv run pytest tests/unit/test_hypernet_distill_config.py -q && uv run mypy src/rune/training/hypernet_distill.py && uv run ruff check src/rune/training/hypernet_distill.py`
Expected: PASS; clean.

- [ ] **Step 5: Commit**

```bash
git add src/rune/training/hypernet_distill.py tests/unit/test_hypernet_distill_config.py
git commit -m "feat(training): D2L context-distillation loop (teacher in-context, diff-masked KL)"
```

---

## Task 12: Stage 0 — synthetic NIAH overfit gate (GPU, decisive)

**Files:**
- Create: `tools/diag_synthetic_overfit.py`
- Run: under `tools/run_guarded.sh`

- [ ] **Step 1: Write the synthetic-overfit harness**

Build a 3–5 record corpus with an unguessable needle present only in the context (e.g. `MAGIC_OFFSET = 73921`, `frobnicate_payload`), held-out recall prompts that omit the fact, and a contradictory variant (different needle). Train via `run_hypernet_distillation` for `--max-steps` (default 20). After training, generate adapters for real / zero / contradictory trajectories and measure needle recall. Emit `--json-out`:

```python
# tools/diag_synthetic_overfit.py  (skeleton — fill loop bodies)
"""Stage 0 discriminator: can the D2L loop make a non-inert, content-retrieving
adapter on an oracle-free synthetic needle corpus? Run under tools/run_guarded.sh."""
import argparse, json
# ... build corpus, train (max_steps), probe real/zero/contra recall ...
# Emit: real_hit_rate, zero_hit_rate, contradictory_hit_rate, scaler_b stats,
#       per-component grad norms, delta_w_norm, diff_agreement, skipped_records.
```

- [ ] **Step 2: Check RAM, then run guarded**

Run:
```bash
free -g
tools/run_guarded.sh /tmp/synth_overfit.log tools/diag_synthetic_overfit.py --max-steps 20 --json-out /tmp/rune-issue49-synth.json
cat /tmp/rune-issue49-synth.json
```
Expected gate: `real_hit_rate > zero_hit_rate` AND `real_hit_rate > contradictory_hit_rate`.

- [ ] **Step 3: Inspect collapse diagnostics**

Confirm the JSON includes non-flat `scaler_B` grad norms and a non-trivial ΔW norm. If gradients are flat or records are skipped → mechanical bug; fix before Stage 1. **Do not proceed past this gate on failure.**

- [ ] **Step 4: Commit harness + artifact**

```bash
git add tools/diag_synthetic_overfit.py
cp /tmp/rune-issue49-synth.json docs/superpowers/artifacts/2026-05-30-synth-overfit.json 2>/dev/null || mkdir -p docs/superpowers/artifacts && cp /tmp/rune-issue49-synth.json docs/superpowers/artifacts/
git add docs/superpowers/artifacts/2026-05-30-synth-overfit.json
git commit -m "feat(tools): Stage-0 synthetic NIAH overfit gate + passing artifact"
```

---

## Task 13: §C inference-format alignment + probe JSON output

**Files:**
- Modify: `src/rune/engine/graph.py` and the `code`/`code_continue` Jinja2 templates
- Modify: `tools/diag_retrieval_probe.py`, `diag_recall_probe.py`, `diag_continuation_probe.py` (JSON output)
- Test: `tests/unit/test_trajectory_format.py`

- [ ] **Step 1: Write failing test that inference trajectory uses training headers**

```python
# tests/unit/test_trajectory_format.py
from rune.engine.graph import render_training_format_trajectory


def test_inference_trajectory_uses_training_headers() -> None:
    txt = render_training_format_trajectory(
        task="implement find_tuples", current_code="def f(): pass",
        feedback="use all() not any()",
    )
    assert "## Task" in txt and "## Current Code" in txt and "## Review Feedback" in txt
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/unit/test_trajectory_format.py -q`
Expected: FAIL — function missing.

- [ ] **Step 3: Implement the training-format renderer and use it at adapter-generation time**

Add `render_training_format_trajectory(...)` producing the `## Task / ## Current Code / ## Review Feedback / ## Revision`-prefix format, and call it where `graph.py` builds the trajectory text fed to `model.generate_adapter(...)` (keep the human-facing prompt template separate). Add a `--json-out` flag to the three probes emitting the gate schema (`real_hit_rate`, `zero_hit_rate`, `shuffled_hit_rate`, `contradictory_hit_rate`, `adapter_cosine`, `diff_agreement`, `scaler_b_absmax`).

- [ ] **Step 4: Run tests + lint**

Run: `uv run pytest tests/unit/test_trajectory_format.py -q && uv run mypy src/rune/engine/graph.py`
Expected: PASS; mypy clean.

- [ ] **Step 5: Commit**

```bash
git add src/rune/engine/graph.py src/rune/engine/templates tools/diag_*_probe.py tests/unit/test_trajectory_format.py
git commit -m "feat(engine): align inference trajectory to training format; probe JSON output (#49 §C)"
```

---

## Task 14: Stage 1 — real-corpus D2L train + Stage 3/4 gate run (GPU)

**Files:**
- Run only (uses Tasks 11–13). Artifacts under `docs/superpowers/artifacts/`.

- [ ] **Step 1: Pull corpus from S3**

Run:
```bash
mkdir -p /tmp/rune-corpus
aws s3 cp s3://elixirtrials-949678234935-eu-west-2-artifacts/training-data/github-pairs/external_codereview.unrolled.jsonl /tmp/rune-corpus/  # 100% diff coverage
wc -l /tmp/rune-corpus/external_codereview.unrolled.jsonl
```
Expected: ~7,670 lines.

- [ ] **Step 2: Train guarded**

Run:
```bash
free -g
tools/run_guarded.sh /tmp/d2l_train.log -m rune.cli train --config <(printf 'corpus_path: /tmp/rune-corpus/external_codereview.unrolled.jsonl\ncheckpoint_dir: /tmp/rune-ck-issue49\n')
tail -40 /tmp/d2l_train.log
```
Expected: loss decreases; `scaler_B absmax` rises off zero; `diff_agreement` trends up; no skipped-record spike.

- [ ] **Step 3: §D scaling re-measure + gate run**

Run the retrieval/recall/continuation probes against the new checkpoint with `--json-out`, sweeping scaling fresh (do NOT reuse base≈7.84):
```bash
for p in retrieval recall continuation; do
  tools/run_guarded.sh /tmp/probe_$p.log tools/diag_${p}_probe.py --checkpoint /tmp/rune-ck-issue49/checkpoint.pt --json-out /tmp/probe_$p.json
done
uv run python -c "import json; from rune.training.gate import evaluate_retrieval_gate; \
  p=json.load(open('/tmp/probe_retrieval.json')); r=evaluate_retrieval_gate(p); print(r)"
```
Expected: `RetrievalGateResult(passed=True, ...)`.

- [ ] **Step 4: Tiny benchmark with controls**

Run a 2–3 task tiny benchmark across {base, zero, real, shuffled, contradictory}. Require no regression vs base/zero and directional lift for real.

- [ ] **Step 5: Save artifacts + commit**

```bash
mkdir -p docs/superpowers/artifacts
cp /tmp/probe_*.json docs/superpowers/artifacts/
git add docs/superpowers/artifacts/
git commit -m "test(issue49): gate-validated checkpoint artifacts (retrieval/recall/continuation + tiny bench)"
```

---

## Final verification before declaring Issue #49 milestone done

- [ ] `uv run ruff check .`
- [ ] `uv run mypy src/`
- [ ] `uv run pytest tests/unit/ -q`
- [ ] Stage 0 synthetic gate JSON shows `real > zero` and `real > contradictory`, saved with the run.
- [ ] Retrieval gate `passed=True` on the real-corpus checkpoint; magnitude used only as a tripwire.
- [ ] Tiny benchmark: no regression vs base/zero; directional lift for real over shuffled/contradictory.
- [ ] Final report states which hypotheses were confirmed (gate fix, L1 sink, diff-mask) and which remain ablation candidates.

## Self-review notes (author)

- Spec coverage: Stage 0 (Tasks 3,4,5,6,12), Stage 1 (Tasks 11,14), §C (Task 13), §D (Tasks 7,8,14-step3), gates (Tasks 9,12,14), lean-up (Task 10), watchdog (Task 1). All spec sections mapped.
- Types: `run_hypernet_distillation`/`DistillConfig`/`evaluate_retrieval_gate`/`distill_step_loss`/`compute_diff_positions`/`reinit_scaler_b_nonzero`/`audit_checkpoint_keys`/`merge_head_bias_rank` are consistent across tasks.
- GPU steps (11,12,14) are run-and-observe; their correctness gates are the committed JSON artifacts, not assertions in CI.
