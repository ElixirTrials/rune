# C4 Stage 1 Implementation Plan — capacity go/no-go for the episodic-continuation thesis

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run the C4 Stage-1 experiments demanded by the PR #60 specialist
reviews — the I0 symbol-reuse audit and the I5 capacity curve — producing a numeric
go/no-go for the multi-round continuation benchmark (Stage 2).

**Architecture:** Two new `tools/` harnesses reusing the frozen C1 keystone instrument.
The I0 audit is a CPU AST tool over engine session traces (plus a small GPU run to
regenerate real multi-round sessions — the committed fixtures are step-0-only). The I5
capacity runner extends `tools/_repobench_clamp_run.py` to bundles of K rows: K facts
compiled into ONE adapter (two build modes) vs K pointers in the prompt tail, scored
with the identical `_score`/`_gen_line` path, on the identical 60 keystone rows.
**No `src/` changes**: mode-(b) composition uses the probe's low-level assembly path
(`extract_activations_with_model` → `hyp.generate_weights` → `combine_lora` →
`_to_peft_state_dict`), which bypasses the `merge_head_bias_rank` guard, plus a PEFT
model built at an enlarged rank so rank-stacked adapters hot-swap cleanly.

**Tech Stack:** uv / Python 3.12, torch + PEFT + transformers (GPU imports deferred),
MLflow at `http://localhost:5000`, pytest for CPU-testable parts, existing stats
helpers (`_wilson_ci`, `_two_sided_binom_p`, `_paired_discordants` from the clamp
harness).

## Global Constraints

- Always `uv run` (with `--no-sync` on the GPU box — plain `uv sync` prunes the gpu extra).
- GPU imports deferred inside function bodies; `uv run pytest tests/unit/ -q` must stay CPU-green.
- CPU RAM ~15GB: base loads with `device_map={"": 0}` + `low_cpu_mem_usage=True`; hypernet ckpt loads via mmap; never `offload_base=True`.
- Base model id only via `rune.config.load_rune_config()`; never hardcoded.
- Frozen checkpoint: c3 = `/tmp/phase1/ckpt/c3_t07_lp2_lg1.pt`, sha256 `53e24af243a38dfbfad82f7293635bfc592922dd2058fefbbfa10714b5457a3f` — verified **before any forward pass**. No training, no new trajectory-corpus generation.
- `tools/_repobench_clamp_run.py` and `tools/_specificity_probe.py` stay **byte-untouched** (prior-run comparability). All new code in new files.
- C1-parity params pinned: window 768, levels `8k,32k`, per-level 30, offset 100, seed 0, `variant="use"`, `anchor=0`, `scaling=0.91`, `max_new=48` (MLflow exp 79, run `f37374906c5f…`).
- Every published number is recomputed independently from the raw per-task trace before it appears in a findings doc or PR comment.
- MLflow param/metric rows are not durable (DB snapshot loss precedent) — everything load-bearing also lands as an **artifact** and its sha256 goes to `docs/publication/hashes.txt`.
- ruff-clean for all new tools; `tools/` is mypy-excluded; one-line docstrings per house style.

---

## Context — what the spec is and what exploration established

**Spec** = PR #60 comment `4913682133` ("C4 — what's left to build and run", 2026-07-08),
shaped by three specialist reviews. Near-term commitment is **Stage 1 only**:

- **I0 fixture audit** — do engine trajectories reuse, in round t≥2, symbols introduced in round t−1? Threshold 60%; below it a synthetic accumulate-K task is mandatory for Stage 2. Also: state c3's training distribution and whether the audited sessions are OOD.
- **I5 capacity curve** — K∈{1,2,4,8} facts in one adapter vs K×~124 prompt tokens, in **(a) regenerate-from-scratch** and **(b) incremental-delta** build modes. K=1 reproduces the C1 loss (episodic 0.517 vs a2_tail 0.833).
- **Go/no-go** — if K=2 fails to beat floor by a pre-set margin (co-author to set), **stop**: redirect to the systems eval; the honest a2_tail reframe stands.

**Load-bearing facts established by code exploration (verified 2026-07-08):**

1. `tests/fixtures/lcb_engine_fixes/sessions/*/session.jsonl` are **single-record** files — step-0 `decompose` only (keys: `step, action, target, trajectory, prompt, output, feedback`). The I0 audit as worded cannot run on them; real multi-round sessions must be regenerated (Task 2). The six LCB rows needed to do so are committed in `tests/fixtures/lcb_engine_fixes/rows.jsonl`, and `tools/_lcb_run.py --qids … --sessions …` already writes full per-task `session.jsonl` dirs via `run_benchmark(..., sessions_dir=…)`.
2. The engine's continuation sub-loop (`src/rune/engine/graph.py:1002-1070`) feeds `accumulated_code` through **two channels**: adapter conditioning (`graph.py:1020-1026`, capped by `_ACCUMULATED_CODE_CAP = 3500` at `graph.py:156`) and `assistant_prefix=accumulated_code` (`graph.py:1031`, uncapped). The user-prompt template `prompt_code_continue.j2` is already clean (task description only). Isolating the adapter channel in Stage 2 is the one-line change `assistant_prefix=""` at `graph.py:1031`.
3. **No KV-cache persistence exists** anywhere in `src/rune/` (`use_cache=False` throughout; continuation re-primes purely textually). Stage 2's `kv_reinject` arm is new machinery, not a config flag.
4. PEFT state-dict layout (`hypernetwork._to_peft_state_dict`): `…lora_A.weight` is `(r_peft, in)`, `…lora_B.weight` is `(out, r_peft)`; with `use_bias`, ranks `0..r-1` are context and `r..2r-1` are the **conditioning-independent** head bias (`wrapper.py:40-50`). Rank-stacking is exact: `[B1 B2] @ [A1; A2] = B1@A1 + B2@A2`; zero-padding extra ranks is numerically inert.
5. `merge_head_bias_rank` (`hypernetwork.py:394`) hard-pins PEFT rank == `2r`, so composed (>2r) adapters cannot pass through `model.generate_adapter`. The probe's assembly path (`tools/_specificity_probe.py:277-283`) bypasses the guard — house precedent for tools-level low-level assembly.
6. C1 scoring surface to reuse byte-identically: `clamp._prefix`, `clamp._gen_line` (temperature 0.0, `torch.manual_seed(seed)` before every generation, `model.reset_adapter()` before non-adapter arms), `clamp._score` (identifier recovery via `gold_id_recovery`), `clamp._assemble_tail_prompt`, `clamp._load_stratified`, `clamp._wilson_ci`, `clamp._two_sided_binom_p`, `clamp._paired_discordants`. The module is import-safe (`if __name__ == "__main__"` guard; unit test `tests/unit/test_repobench_clamp_arms.py:16-21` already exec-loads it on CPU).

---

## Pre-registration (locked before any GPU run; copied verbatim into the findings doc)

**Rows:** the C1 keystone rows — `_load_stratified(["8k","32k"], 30, offset=100)`, N=60, frozen order. Bundles at each K are **consecutive disjoint index groups** in that frozen order; the `60 mod K` remainder rows are dropped from that K's denominators (K=8 → 56 rows; drop logged, never silent).

**Arms** (per row; labels frozen):
- `floor` — clamped current-file prompt only (K-independent; run once).
- `tail_k{K}` — the bundle's K episodic pointers (`render_episodic(row, "use", anchor_chars=0)` each, joined by `"\n\n"`) placed at the prompt tail via the C1 `_assemble_tail_prompt`. If the joined conditioning alone exceeds W: the arm is **infeasible and scored as not-recovered** (`recovered=False`, `infeasible=True`). This deviates deliberately from C1's exclusion guard: the linear prompt cost hitting the wall *is the measured quantity*. Sensitivity view excluding infeasible rows is also reported.
- `adapter_a_k{K}` — build mode (a), regenerate-from-scratch: ONE hypernet forward on the K pointers concatenated (`"\n\n".join`), `max_length=2048` unchanged (parity); `cond_tokens` and a `cond_truncated` flag logged (K=8 may clip).
- `adapter_b_k{K}` — build mode (b), incremental-delta: K per-row hypernet forwards; the K native state dicts composed by **rank-stacking** context slices with a single bias slice, zero-padded to the campaign rank.
- At K=1 the modes coincide; one arm `adapter_k1` is run and serves as the C1 anchor.

**Adapter application:** `hotswap_adapter(scale_lora_b(padded_sd, 0.91))` — same scaling as C1's `episodic_use` for all adapter arms and both modes.

**Anchor gates (run-validity, before the science gate is read):**
- **S1-ANCHOR-1:** sanity leg (native rank, engine `generate_adapter` path): `floor` and `adapter_k1` predictions match the C1 run `f37374906c5f` trace **token-for-token, 60/60**. Failure → stop, debug environment; no capacity numbers are valid.
- **S1-ANCHOR-2:** capacity leg (enlarged rank): `adapter_k1` prediction agreement with the sanity leg reported (expected ≈60/60; GEMM-shape drift is the only permitted source of difference). Agreement < 55/60 → investigate before reading gates.

**Science gate (the go/no-go):**
- **S1-GO:** at K=2, for **at least one** build mode: paired McNemar (`adapter_*_k2` vs `floor`, two-sided exact) p < 0.05 **AND** recovery-rate delta ≥ **M**.
- **M is the co-author-set margin.** Proposed default **M = +0.15** (≈40% of the K=1 episodic−floor lift of +0.367); pending sign-off. Changing M after seeing results is prohibited.
- **Sign-off gate:** M and this pre-registration section must be confirmed by the co-author **before Task 5 Step 4 (the full capacity leg) launches**. Tasks 1–4 and the sanity/smoke steps may proceed in parallel with that review; the gating run may not.
- **NO-GO** → Stage 2 is not built; findings doc records the capacity ceiling; the systems-eval redirect and the honest a2_tail reframe stand.

**Curve reporting (not gating):** per-K per-arm recovery + Wilson 95% CI; paired McNemars adapter-vs-floor and adapter-vs-tail at each K; crossover **K\*** = smallest K where the better adapter mode ≥ `tail_k{K}` on paired rows (with infeasibility counted as tail failure). Sensitivity: bundle-level exact sign test at K=2 (n=30 bundles, adapter bundle-mean vs floor bundle-mean) — reported because rows within a bundle share an adapter and are not independent; the row-level McNemar stays primary for C1 comparability.

**I0 threshold:** aggregate reuse fraction ≥ 0.60 over eligible rounds of the six regenerated LCB sessions → natural trajectories suffice for Stage 2's consistency metric; < 0.60 → synthetic accumulate-K task mandatory in Stage 2. Either way I5 proceeds (the capacity question is independent).

**Deviations from C1, pre-declared:** (1) tail infeasibility scored-as-failure instead of excluded (reason above); (2) adapter arms at K>1 share one adapter across bundle rows (the point of the experiment); (3) capacity leg runs at enlarged PEFT rank (anchored by S1-ANCHOR-2); (4) mode-(b) composition keeps ONE bias slice (bias is conditioning-independent; asserted in Task 5 by comparing bias slices across two per-fact adapters).

---

## File structure

| Path | Role |
|---|---|
| `tools/_c4_fixture_audit.py` | **Create** — I0 symbol-reuse audit (CPU, AST) |
| `tools/_c4_capacity_lib.py` | **Create** — pure composition/bundling primitives (CPU-testable) |
| `tools/_c4_capacity_run.py` | **Create** — I5 runner: sanity + capacity legs, stats, gate, MLflow |
| `tests/unit/test_c4_fixture_audit.py` | **Create** — audit TDD |
| `tests/unit/test_c4_capacity_lib.py` | **Create** — composition math TDD |
| `tests/unit/test_c4_capacity_stats.py` | **Create** — stats/gate on synthetic traces |
| `docs/publication/c4_stage1_findings.md` | **Create** (Task 6) — realized-gate findings |
| `docs/publication/hashes.txt` | **Modify** — append trace/audit shas |
| `mkdocs.yml` | **Modify** — nav entry for the findings doc |

Everything else — engine, clamp harness, probe — untouched.

---

### Task 1: I0 audit tool (`tools/_c4_fixture_audit.py`)

**Files:**
- Create: `tools/_c4_fixture_audit.py`
- Test: `tests/unit/test_c4_fixture_audit.py`

**Interfaces:**
- Consumes: `rune.engine.continuation.extract_partial_code(text: str) -> str` (the engine's own code extractor); session records with keys `step, action, output`.
- Produces: `introduced_symbols(code: str) -> set[str]`, `used_symbols(code: str) -> set[str]`, `code_rounds(records: list[dict]) -> list[str]`, `reuse_counts(rounds: list[str]) -> tuple[int, int]`, `audit_session(path: Path) -> dict`, CLI `--sessions-dir --out`. Task 2 runs the CLI; Task 6 quotes its JSON.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_c4_fixture_audit.py
"""I0 audit: symbol introduction/reuse over engine session traces."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_TOOL = Path(__file__).resolve().parents[2] / "tools" / "_c4_fixture_audit.py"
_spec = importlib.util.spec_from_file_location("_c4_fixture_audit", _TOOL)
assert _spec is not None and _spec.loader is not None
audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(audit)

_FIXTURES = (
    Path(__file__).resolve().parents[2]
    / "tests" / "fixtures" / "lcb_engine_fixes" / "sessions"
)


def _rec(step: int, action: str, output: str) -> dict:
    return {"step": step, "action": action, "target": "",
            "trajectory": "", "prompt": "", "output": output, "feedback": None}


def test_introduced_symbols_defs_classes_assignments() -> None:
    code = "def f(x):\n    y = 1\n    return y\n\nclass C:\n    pass\n\nz = f(2)\n"
    assert audit.introduced_symbols(code) == {"f", "y", "C", "z"}


def test_introduced_symbols_tolerates_syntax_error() -> None:
    assert audit.introduced_symbols("def broken(:") == set()


def test_reuse_detected_when_round2_calls_round1_symbol() -> None:
    r1 = "def helper(a):\n    return a + 1\n"
    r2 = "def solve(xs):\n    return [helper(x) for x in xs]\n"
    reused, eligible = audit.reuse_counts([r1, r2])
    assert (reused, eligible) == (1, 1)


def test_no_reuse_counts_zero() -> None:
    reused, eligible = audit.reuse_counts(
        ["def a():\n    return 1\n", "def b():\n    return 2\n"]
    )
    assert (reused, eligible) == (0, 1)


def test_decompose_only_session_has_no_eligible_rounds(tmp_path: Path) -> None:
    p = tmp_path / "s1" / "session.jsonl"
    p.parent.mkdir()
    p.write_text(json.dumps(_rec(0, "decompose", '{"subtasks": []}')) + "\n")
    rep = audit.audit_session(p)
    assert rep["eligible_rounds"] == 0
    assert rep["n_code_rounds"] == 0


def test_multi_round_session_end_to_end(tmp_path: Path) -> None:
    p = tmp_path / "s2" / "session.jsonl"
    p.parent.mkdir()
    lines = [
        _rec(0, "decompose", '{"subtasks": []}'),
        _rec(1, "code", "```python\ndef area(r):\n    return 3.14 * r * r\n```"),
        _rec(2, "repair", "```python\ndef main():\n    print(area(2))\n```"),
    ]
    p.write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    rep = audit.audit_session(p)
    assert rep["n_code_rounds"] == 2
    assert (rep["reused_rounds"], rep["eligible_rounds"]) == (1, 1)


def test_committed_fixtures_are_step0_only() -> None:
    """Regression-documents the I0 discovery: fixtures have zero code rounds."""
    reports = [audit.audit_session(p) for p in sorted(_FIXTURES.rglob("session.jsonl"))]
    assert len(reports) == 6
    assert all(r["eligible_rounds"] == 0 for r in reports)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_c4_fixture_audit.py -q`
Expected: FAIL — `FileNotFoundError` / module load error (tool does not exist yet).

- [ ] **Step 3: Implement the tool**

```python
# tools/_c4_fixture_audit.py
"""C4 I0 — symbol-reuse audit over engine session traces.

For each session.jsonl under --sessions-dir, extract the code payload of every
code-bearing round (actions: code, repair, integrate) and measure the fraction
of rounds t>=2 that reuse at least one symbol introduced in the previous
code-bearing round. The committed lcb_engine_fixes fixtures are step-0
decompose-only, so they report zero eligible rounds; the audit is meaningful
over full regenerated sessions (c4_implementation_plan.md Task 2).
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

_CODE_ACTIONS = frozenset({"code", "repair", "integrate"})


def introduced_symbols(code: str) -> set[str]:
    """Names bound in *code*: def/class names and assignment targets."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(node.name)
        elif isinstance(node, ast.Assign):
            out.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            if isinstance(node.target, ast.Name):
                out.add(node.target.id)
    return out


def used_symbols(code: str) -> set[str]:
    """All Name ids and attribute names referenced in *code*."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            out.add(node.id)
        elif isinstance(node, ast.Attribute):
            out.add(node.attr)
    return out


def code_rounds(records: list[dict[str, Any]]) -> list[str]:
    """Code payload per code-bearing record, in step order."""
    from rune.engine.continuation import extract_partial_code  # noqa: PLC0415

    recs = sorted(
        (r for r in records if r.get("action") in _CODE_ACTIONS),
        key=lambda r: r.get("step", 0),
    )
    out: list[str] = []
    for r in recs:
        code = extract_partial_code(r.get("output") or "")
        if code.strip():
            out.append(code)
    return out


def reuse_counts(rounds: list[str]) -> tuple[int, int]:
    """(rounds reusing a prev-round-introduced symbol, eligible rounds t>=2)."""
    reused = eligible = 0
    for prev, curr in zip(rounds, rounds[1:]):
        eligible += 1
        if introduced_symbols(prev) & used_symbols(curr):
            reused += 1
    return reused, eligible


def audit_session(path: Path) -> dict[str, Any]:
    """Per-session reuse report for one session.jsonl."""
    records = [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
    rounds = code_rounds(records)
    reused, eligible = reuse_counts(rounds)
    return {
        "session": path.parent.name,
        "n_records": len(records),
        "n_code_rounds": len(rounds),
        "eligible_rounds": eligible,
        "reused_rounds": reused,
        "introduced_per_round": [sorted(introduced_symbols(c)) for c in rounds],
    }


def main() -> None:
    """Scan --sessions-dir, print the per-session table, write --out JSON."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sessions-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    reports = [audit_session(p) for p in sorted(a.sessions_dir.rglob("session.jsonl"))]
    reused = sum(r["reused_rounds"] for r in reports)
    eligible = sum(r["eligible_rounds"] for r in reports)
    frac = (reused / eligible) if eligible else None
    for r in reports:
        print(
            f"{r['session']}: reuse {r['reused_rounds']}/{r['eligible_rounds']}"
            f" (code rounds: {r['n_code_rounds']})"
        )
    tail = f" = {frac:.3f}" if frac is not None else "  [no eligible rounds]"
    print(f"TOTAL: {reused}/{eligible}{tail}")
    if a.out:
        a.out.write_text(json.dumps(
            {"sessions": reports, "total_reused_rounds": reused,
             "total_eligible_rounds": eligible, "reuse_fraction": frac},
            indent=1,
        ))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_c4_fixture_audit.py -q`
Expected: `7 passed`. Then `uv run ruff check tools/_c4_fixture_audit.py tests/unit/test_c4_fixture_audit.py` → clean, and `uv run pytest tests/unit/ -q` → no regressions.

- [ ] **Step 5: Commit**

```bash
git add tools/_c4_fixture_audit.py tests/unit/test_c4_fixture_audit.py
git commit -m "feat(#52): C4 I0 symbol-reuse audit tool"
```

---

### Task 2: Regenerate real multi-round sessions and run the I0 audit (GPU, ~20–30 min)

**Files:**
- Create (runtime, not committed): `/tmp/c4/i0_sessions/`, `/tmp/c4/i0_audit.json`
- No repo files change in this task; results are folded into Task 6's findings doc.

**Interfaces:**
- Consumes: `tools/_lcb_run.py` CLI (`--arm c3 --qids … --sessions … --no-grade`), Task 1's audit CLI.
- Produces: `/tmp/c4/i0_audit.json` with `reuse_fraction` — the I0 number Task 6 reports against the 0.60 threshold.

- [ ] **Step 1: Pre-flight**

```bash
free -g && nvidia-smi
ls -la /tmp/phase1/ckpt/c3_t07_lp2_lg1.pt \
  && sha256sum /tmp/phase1/ckpt/c3_t07_lp2_lg1.pt
```
Expected: sha `53e24af2…`. If the checkpoint is missing (VM restart), re-restore per `docs/publication/c21_prep.md` §4 (`aws s3 cp s3://elixirtrials-949678234935-us-east-1-artifacts/mlflow/artifacts/45/fe72f9ddd69c4f7b8bd86b6b12372d47/artifacts/checkpoints/checkpoint_step48.pt /tmp/phase1/ckpt/c3_t07_lp2_lg1.pt`).

- [ ] **Step 2: Regenerate sessions for the six fixture qids**

```bash
mkdir -p /tmp/c4
nohup env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  uv run --no-sync python tools/_lcb_run.py \
    --arm c3 --qids 3748,3753,3754,3777,3799,3801 \
    --out /tmp/c4/i0_gens.jsonl --sessions /tmp/c4/i0_sessions \
    --no-grade --experiment issue52-c4 \
  > /tmp/c4/i0_run.log 2>&1 &
```
Monitor `/tmp/c4/i0_run.log`; expected wall ≈ 20–30 min (6 tasks, max-iters 24).
Expected on completion: `/tmp/c4/i0_sessions/<qid>/session.jsonl` exists for all six qids with **multiple records** (steps 0..n; actions decompose/plan/code/…).

- [ ] **Step 3: Run the audit**

```bash
uv run --no-sync python tools/_c4_fixture_audit.py \
  --sessions-dir /tmp/c4/i0_sessions --out /tmp/c4/i0_audit.json
cat /tmp/c4/i0_audit.json
```
Expected: per-session reuse lines and a `TOTAL: r/e = f` with `e > 0`. Record the fraction; the 0.60 threshold is read in Task 6, not here.

- [ ] **Step 4: Record the distribution statement (for the findings doc)**

Verbatim facts to carry: c3 was trained on `benchmarks/mbpp_recall_train.jsonl`
(40 MBPP tasks; MLflow exp 45 run `fe72f9ddd69c…`), conditioning surface
`render_training_format_trajectory` (`## Task / ## Current Code / ## Review Feedback`).
The audited LCB sessions are **OOD in task domain** (LCB-v6 vs MBPP) but
**in-distribution in conditioning surface** (same renderer). Per the review:
an OOD null ≠ mechanism null — which is why the I5 gate runs on the C1 keystone
instrument where c3's episodic capability is already demonstrated (0.517).

---

### Task 3: Composition primitives (`tools/_c4_capacity_lib.py`)

**Files:**
- Create: `tools/_c4_capacity_lib.py`
- Test: `tests/unit/test_c4_capacity_lib.py`

**Interfaces:**
- Consumes: PEFT flat state dicts as emitted by `rune.model.hypernetwork._to_peft_state_dict` — keys `…lora_A.weight` `(r, in)` / `…lora_B.weight` `(out, r)`.
- Produces (Task 4 imports all of these):
  - `make_bundles(n_rows: int, k: int) -> list[list[int]]`
  - `compose_rank_stacked(state_dicts: list[dict[str, Any]], ctx_rank: int) -> dict[str, Any]`
  - `pad_adapter_rank(state_dict: dict[str, Any], target_rank: int) -> dict[str, Any]`
  - `multi_cond_text(conds: list[str]) -> str`
  - `campaign_rank(ctx_rank: int, bias_rank: int, k_max: int) -> int`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_c4_capacity_lib.py
"""Rank-stacking composition math for the C4 capacity curve."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

_TOOL = Path(__file__).resolve().parents[2] / "tools" / "_c4_capacity_lib.py"
_spec = importlib.util.spec_from_file_location("_c4_capacity_lib", _TOOL)
assert _spec is not None and _spec.loader is not None
lib = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lib)


def _sd(a: torch.Tensor, b: torch.Tensor) -> dict:
    return {"m.lora_A.weight": a, "m.lora_B.weight": b}


def _delta(sd: dict) -> torch.Tensor:
    return sd["m.lora_B.weight"] @ sd["m.lora_A.weight"]


def test_rank_stacked_equals_sum_of_context_products_plus_one_bias() -> None:
    torch.manual_seed(0)
    ctx, bias, din, dout = 2, 1, 6, 5
    a_bias, b_bias = torch.randn(bias, din), torch.randn(dout, bias)
    a1, b1 = torch.randn(ctx, din), torch.randn(dout, ctx)
    a2, b2 = torch.randn(ctx, din), torch.randn(dout, ctx)
    sd1 = _sd(torch.cat([a1, a_bias]), torch.cat([b1, b_bias], dim=1))
    sd2 = _sd(torch.cat([a2, a_bias]), torch.cat([b2, b_bias], dim=1))
    comp = lib.compose_rank_stacked([sd1, sd2], ctx_rank=ctx)
    want = b1 @ a1 + b2 @ a2 + b_bias @ a_bias
    assert comp["m.lora_A.weight"].shape == (2 * ctx + bias, din)
    assert torch.allclose(_delta(comp), want, atol=1e-6)


def test_compose_single_adapter_is_identity() -> None:
    torch.manual_seed(1)
    sd = _sd(torch.randn(4, 6), torch.randn(5, 4))
    comp = lib.compose_rank_stacked([sd], ctx_rank=2)
    assert torch.equal(_delta(comp), _delta(sd))


def test_pad_adapter_rank_is_numerically_inert() -> None:
    torch.manual_seed(2)
    sd = _sd(torch.randn(3, 6), torch.randn(5, 3))
    padded = lib.pad_adapter_rank(sd, target_rank=10)
    assert padded["m.lora_A.weight"].shape == (10, 6)
    assert padded["m.lora_B.weight"].shape == (5, 10)
    assert torch.equal(_delta(padded), _delta(sd))


def test_pad_rejects_shrinking() -> None:
    sd = _sd(torch.zeros(4, 6), torch.zeros(5, 4))
    with pytest.raises(ValueError):
        lib.pad_adapter_rank(sd, target_rank=3)


def test_make_bundles_consecutive_disjoint_drops_remainder() -> None:
    assert lib.make_bundles(60, 1) == [[i] for i in range(60)]
    b8 = lib.make_bundles(60, 8)
    assert len(b8) == 7 and b8[0] == list(range(8)) and b8[-1] == list(range(48, 56))
    flat = [i for b in b8 for i in b]
    assert len(set(flat)) == len(flat) == 56


def test_campaign_rank() -> None:
    assert lib.campaign_rank(8, 8, 8) == 72
    assert lib.campaign_rank(8, 0, 8) == 64


def test_multi_cond_text_joins_blocks() -> None:
    assert lib.multi_cond_text(["a", "b"]) == "a\n\nb"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_c4_capacity_lib.py -q`
Expected: FAIL — module load error (file missing).

- [ ] **Step 3: Implement the library**

```python
# tools/_c4_capacity_lib.py
"""C4 I5 — composition and bundling primitives for the capacity curve.

Rank-stacking is exact: [B1 B2] @ [A1; A2] = B1@A1 + B2@A2, so composing K
per-fact adapters as concatenated rank slices reproduces the sum of their LoRA
deltas; zero-padding extra rank slices contributes exactly zero. PEFT layout
(hypernetwork._to_peft_state_dict): lora_A.weight is (r, in), lora_B.weight is
(out, r); with use_bias, ranks 0..ctx-1 are context and ctx.. are the
conditioning-independent head bias (wrapper.py:40-50) — so composition keeps
ONE bias slice (from the first adapter) or the bias would be applied K times.
"""

from __future__ import annotations

from typing import Any


def make_bundles(n_rows: int, k: int) -> list[list[int]]:
    """Disjoint consecutive index bundles of size k; remainder rows dropped."""
    if k < 1:
        raise ValueError("k must be >= 1")
    return [list(range(i * k, (i + 1) * k)) for i in range(n_rows // k)]


def compose_rank_stacked(
    state_dicts: list[dict[str, Any]], ctx_rank: int
) -> dict[str, Any]:
    """Compose K adapters: concat context rank slices; keep the first bias."""
    import torch  # noqa: PLC0415

    if not state_dicts:
        raise ValueError("no adapters to compose")
    keys = state_dicts[0].keys()
    if any(sd.keys() != keys for sd in state_dicts[1:]):
        raise ValueError("adapter key sets differ")
    out: dict[str, Any] = {}
    for key in keys:
        if "lora_A" in key:
            parts = [sd[key][:ctx_rank] for sd in state_dicts]
            parts.append(state_dicts[0][key][ctx_rank:])
            out[key] = torch.cat(parts, dim=0)
        elif "lora_B" in key:
            parts = [sd[key][:, :ctx_rank] for sd in state_dicts]
            parts.append(state_dicts[0][key][:, ctx_rank:])
            out[key] = torch.cat(parts, dim=1)
        else:
            out[key] = state_dicts[0][key]
    return out


def pad_adapter_rank(state_dict: dict[str, Any], target_rank: int) -> dict[str, Any]:
    """Zero-pad every lora_A/lora_B pair to target_rank (numerically inert)."""
    import torch  # noqa: PLC0415

    out: dict[str, Any] = {}
    for key, w in state_dict.items():
        if "lora_A" in key:
            r = w.shape[0]
            if r > target_rank:
                raise ValueError(f"{key}: rank {r} > target {target_rank}")
            pad = torch.zeros(
                target_rank - r, w.shape[1], dtype=w.dtype, device=w.device
            )
            out[key] = torch.cat([w, pad], dim=0)
        elif "lora_B" in key:
            r = w.shape[1]
            if r > target_rank:
                raise ValueError(f"{key}: rank {r} > target {target_rank}")
            pad = torch.zeros(
                w.shape[0], target_rank - r, dtype=w.dtype, device=w.device
            )
            out[key] = torch.cat([w, pad], dim=1)
        else:
            out[key] = w
    return out


def multi_cond_text(conds: list[str]) -> str:
    """Mode-(a) conditioning: the K per-row episodic blocks, joined."""
    return "\n\n".join(conds)


def campaign_rank(ctx_rank: int, bias_rank: int, k_max: int) -> int:
    """PEFT rank sized for K_max context slices plus one bias slice."""
    return k_max * ctx_rank + bias_rank
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_c4_capacity_lib.py -q`
Expected: `7 passed`. Then `uv run ruff check tools/_c4_capacity_lib.py tests/unit/test_c4_capacity_lib.py` → clean.

- [ ] **Step 5: Commit**

```bash
git add tools/_c4_capacity_lib.py tests/unit/test_c4_capacity_lib.py
git commit -m "feat(#52): C4 I5 rank-stacked adapter composition primitives"
```

---

### Task 4: Capacity runner (`tools/_c4_capacity_run.py`)

**Files:**
- Create: `tools/_c4_capacity_run.py`
- Test: `tests/unit/test_c4_capacity_stats.py`

**Interfaces:**
- Consumes: `clamp` = `tools/_repobench_clamp_run.py` (import as sibling module: `_prefix`, `_gen_line`, `_score`, `_assemble_tail_prompt`, `_load_stratified`, `_wilson_ci`, `_two_sided_binom_p`, `_paired_discordants`, `C3_CKPT`, `_TAIL_HEADER`, `_CURSOR_MARKER`, `_COND_CHAR_CAP`); `lib` = Task 3; `rune.model.hypernetwork.extract_activations_with_model`, `_to_peft_state_dict`; `ctx_to_lora.modeling.lora_merger.combine_lora`; `rune.model.adapter.scale_lora_b`; `rune.bench.repobench.render_episodic`.
- Produces: CLI with `--leg {sanity,capacity}`, `--ks`, `--stats-only`, `--smoke`, `--c1-traces`; traces JSON at `--out` (clamp-shaped: `{"task_id", …, "arms": {label: {"pred", "recovered", …}}}`); `stage1_gate(traces, margin) -> dict` (Task 6 reads its `go` field); MLflow run in experiment `issue52-c4`.

- [ ] **Step 1: Write the failing stats/gate tests**

```python
# tests/unit/test_c4_capacity_stats.py
"""C4 capacity runner: stats, gate, and label plumbing on synthetic traces."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_TOOL = Path(__file__).resolve().parents[2] / "tools" / "_c4_capacity_run.py"
_spec = importlib.util.spec_from_file_location("_c4_capacity_run", _TOOL)
assert _spec is not None and _spec.loader is not None
cap = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cap)


def _trace(i: int, arms: dict) -> dict:
    return {"task_id": f"t/{i}", "arms": {
        label: {"pred": "", "recovered": rec} for label, rec in arms.items()
    }}


def test_gate_go_when_one_mode_clears_margin_and_p() -> None:
    # 60 rows: floor recovers 9; adapter_b_k2 recovers 40 (superset). p ~ 2^-31.
    traces = [
        _trace(i, {
            "floor": i < 9,
            "adapter_a_k2": i < 12,
            "adapter_b_k2": i < 40,
        })
        for i in range(60)
    ]
    g = cap.stage1_gate(traces, margin=0.15)
    assert g["go"] is True
    assert g["adapter_b_k2"]["delta"] > 0.15
    assert g["adapter_b_k2"]["p"] < 0.05
    assert g["adapter_a_k2"]["passes"] is False  # delta 0.05 < margin


def test_gate_no_go_when_neither_mode_clears() -> None:
    traces = [
        _trace(i, {"floor": i < 9, "adapter_a_k2": i < 11, "adapter_b_k2": i < 10})
        for i in range(60)
    ]
    g = cap.stage1_gate(traces, margin=0.15)
    assert g["go"] is False


def test_metrics_report_per_arm_wilson_and_infeasible() -> None:
    traces = [
        _trace(0, {"floor": False, "tail_k8": False}),
        _trace(1, {"floor": True, "tail_k8": True}),
    ]
    traces[0]["arms"]["tail_k8"]["infeasible"] = True
    m = cap.capacity_metrics(traces)
    assert m["recovery_tail_k8"] == 0.5      # infeasible row scored as failure
    assert m["infeasible_tail_k8"] == 1
    assert 0.0 <= m["recovery_tail_k8_wilson_lo"] < 0.5


def test_bundle_sign_test_counts_bundle_means() -> None:
    # 4 bundles of 2: adapter better in 3 bundles, tied in 1 -> n_eff 3, pos 3.
    traces = []
    for b in range(4):
        for j in range(2):
            i = b * 2 + j
            adapter = b < 3  # bundles 0-2 recover both rows; bundle 3 none
            traces.append(_trace(i, {"floor": False, "adapter_b_k2": adapter}))
    pos, neg, n_eff = cap.bundle_sign_counts(traces, "adapter_b_k2", "floor", k=2)
    assert (pos, neg, n_eff) == (3, 0, 3)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_c4_capacity_stats.py -q`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement the runner**

```python
# tools/_c4_capacity_run.py
"""C4 I5 — capacity curve: K facts in one adapter vs K pointers in the tail.

Extends the frozen C1 keystone instrument (tools/_repobench_clamp_run.py,
byte-untouched, imported as a sibling module) to bundles of K rows on the same
60 keystone rows (levels 8k,32k x 30, offset 100, W=768, seed 0). Arms per K:

  floor          clamped current-file prompt (K-independent, run once)
  tail_k{K}      the bundle's K episodic pointers in the tail; if the joined
                 conditioning alone exceeds W the arm is infeasible and scored
                 as NOT recovered (pre-registered deviation from C1's guard)
  adapter_a_k{K} mode (a): ONE hypernet forward on the K pointers concatenated
  adapter_b_k{K} mode (b): K per-row forwards, rank-stacked composition
  adapter_k1     K=1 (modes coincide) - the C1 anchor arm

Legs: --leg sanity runs floor + adapter_k1 at NATIVE PEFT rank through the
engine path (model.generate_adapter) and must reproduce the C1 run
token-for-token (compare via --c1-traces). --leg capacity loads the PEFT
adapter at campaign_rank(r, bias_rank, K_max) and runs all Ks; per-fact and
mode-(a) adapters are assembled through the probe's low-level path (which the
merge_head_bias_rank guard does not constrain) and zero-padded before hotswap.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _c4_capacity_lib as lib  # noqa: E402
import _repobench_clamp_run as clamp  # noqa: E402

C3_SHA256 = "53e24af243a38dfbfad82f7293635bfc592922dd2058fefbbfa10714b5457a3f"
_KS_DEFAULT = "1,2,4,8"
_MARGIN_DEFAULT = 0.15  # proposed; co-author sign-off pre-registered in the plan


# ---------------------------------------------------------------- stats/gate

def _rate(traces: list[dict[str, Any]], label: str) -> tuple[int, int]:
    k = n = 0
    for rec in traces:
        arm = rec["arms"].get(label)
        if arm is None or arm.get("recovered") is None:
            continue
        n += 1
        k += int(bool(arm["recovered"]))
    return k, n


def capacity_metrics(traces: list[dict[str, Any]]) -> dict[str, float]:
    """Per-arm recovery + Wilson CI + infeasible counts + paired McNemars."""
    labels = sorted({lab for rec in traces for lab in rec["arms"]})
    m: dict[str, float] = {}
    for lab in labels:
        k, n = _rate(traces, lab)
        if not n:
            continue
        lo, hi = clamp._wilson_ci(k, n)
        m[f"recovery_{lab}"] = k / n
        m[f"recovery_{lab}_n"] = float(n)
        m[f"recovery_{lab}_wilson_lo"] = lo
        m[f"recovery_{lab}_wilson_hi"] = hi
        m[f"infeasible_{lab}"] = float(sum(
            1 for rec in traces if rec["arms"].get(lab, {}).get("infeasible")
        ))
    for lab in labels:
        if lab == "floor" or lab.startswith("tail_"):
            continue
        for other in ("floor", f"tail_{lab.rsplit('_', 1)[-1]}"):
            if other not in labels:
                continue
            a_only, b_only, n = clamp._paired_discordants(traces, lab, other)
            m[f"mcnemar_{lab}_vs_{other}_first_only"] = float(a_only)
            m[f"mcnemar_{lab}_vs_{other}_second_only"] = float(b_only)
            m[f"mcnemar_{lab}_vs_{other}_n"] = float(n)
            m[f"mcnemar_{lab}_vs_{other}_p"] = clamp._two_sided_binom_p(a_only, b_only)
    return m


def bundle_sign_counts(
    traces: list[dict[str, Any]], arm: str, other: str, k: int
) -> tuple[int, int, int]:
    """Bundle-level sign counts (sensitivity): mean recovery per k-bundle."""
    pos = neg = 0
    for bundle in lib.make_bundles(len(traces), k):
        d = 0.0
        ok = True
        for i in bundle:
            a = traces[i]["arms"].get(arm, {}).get("recovered")
            b = traces[i]["arms"].get(other, {}).get("recovered")
            if a is None or b is None:
                ok = False
                break
            d += float(bool(a)) - float(bool(b))
        if not ok or d == 0:
            continue
        pos += int(d > 0)
        neg += int(d < 0)
    return pos, neg, pos + neg


def stage1_gate(traces: list[dict[str, Any]], margin: float) -> dict[str, Any]:
    """Pre-registered S1-GO: at K=2 one build mode beats floor by margin, p<.05."""
    out: dict[str, Any] = {"margin": margin}
    go = False
    kf, nf = _rate(traces, "floor")
    for mode in ("adapter_a_k2", "adapter_b_k2"):
        km, nm = _rate(traces, mode)
        if not nm or not nf:
            out[mode] = {"passes": False, "reason": "arm missing"}
            continue
        a_only, b_only, _ = clamp._paired_discordants(traces, mode, "floor")
        p = clamp._two_sided_binom_p(a_only, b_only)
        delta = km / nm - kf / nf
        passes = p < 0.05 and delta >= margin
        out[mode] = {"rate": km / nm, "delta": delta, "p": p, "passes": passes}
        go = go or passes
    out["go"] = go
    return out


# ------------------------------------------------------------- model loading

def load_capacity_handles(cfg: Any, k_max: int) -> dict[str, Any]:
    """ModelWrapper + raw handles with the PEFT adapter at the campaign rank.

    Replicates ModelWrapper.from_config (wrapper.py) except LoraConfig.r —
    lora_alpha keeps the Sakana contract (alpha_peft = checkpoint_alpha *
    r_peft so PEFT's alpha/r quotient equals checkpoint_alpha at any rank).
    """
    import torch  # noqa: PLC0415
    from peft import LoraConfig, get_peft_model  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    from rune.model.hypernetwork import (  # noqa: PLC0415
        HypernetworkConfig,
        load_hypernetwork,
    )
    from rune.model.wrapper import ModelWrapper  # noqa: PLC0415

    device = "cuda" if torch.cuda.is_available() else "cpu"
    hyp = load_hypernetwork(
        HypernetworkConfig(
            checkpoint_path=cfg.checkpoint_path, model_config_name=cfg.model_id
        ),
        device=device,
    )
    hc = hyp.config
    rank = int(hc.lora_config.r)
    use_bias = bool(getattr(hc, "use_bias", False))
    bias_rank = rank if use_bias else 0
    alpha = float(getattr(hc.lora_config, "lora_alpha", rank * 2))
    r_camp = lib.campaign_rank(rank, bias_rank, k_max)
    raw = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        dtype=getattr(torch, cfg.dtype),
        attn_implementation=cfg.attn_implementation,
        low_cpu_mem_usage=True,
        device_map={"": 0},
    )
    peft_model = get_peft_model(raw, LoraConfig(
        r=r_camp, lora_alpha=alpha * r_camp,
        target_modules=list(hc.lora_config.target_modules),
        lora_dropout=0.0, use_rslora=False,
    ))
    tok = AutoTokenizer.from_pretrained(cfg.model_id)
    return {
        "model": ModelWrapper(peft_model, tok, hyp, config=cfg),
        "hyp": hyp, "base": peft_model, "tok": tok,
        "li": [int(x) for x in hc.layer_indices],
        "target_modules": list(hc.lora_config.target_modules),
        "head_bias": hyp.get_head_bias() if use_bias else None,
        "ctx_rank": rank, "bias_rank": bias_rank, "r_camp": r_camp,
    }


def assemble_native_sd(h: dict[str, Any], text: str) -> dict[str, Any]:
    """Per-conditioning PEFT state dict at NATIVE (ctx+bias) rank.

    The probe's assembly path (_specificity_probe.py:277-283): activation
    extraction -> hypernet forward -> combine_lora -> _to_peft_state_dict.
    Caller must reset_adapter() first so activations come from the base model.
    """
    import torch  # noqa: PLC0415
    from ctx_to_lora.modeling.lora_merger import combine_lora  # noqa: PLC0415

    from rune.model.hypernetwork import (  # noqa: PLC0415
        _to_peft_state_dict,
        extract_activations_with_model,
    )

    feats, am = extract_activations_with_model(
        text=text, model=h["base"], tokenizer=h["tok"],
        layer_indices=h["li"], max_length=2048,
    )
    dev = next(h["hyp"].parameters()).device
    dt = next(h["hyp"].parameters()).dtype
    with torch.no_grad():
        ld, _ = h["hyp"].generate_weights(feats.to(device=dev, dtype=dt), am.to(dev), None)
    merged = combine_lora(ld, torch.tensor([1]), lora_bias=h["head_bias"])
    return _to_peft_state_dict(merged, h["li"], h["target_modules"])


# ------------------------------------------------------------------ the legs

async def run_capacity(h: dict[str, Any], rows: list[Any], args: Any) -> list[dict]:
    import torch  # noqa: PLC0415

    from rune.bench.repobench import render_episodic  # noqa: PLC0415
    from rune.model.adapter import scale_lora_b  # noqa: PLC0415

    model = h["model"]
    w = args.window
    ks = [int(x) for x in args.ks.split(",")]
    conds = [
        render_episodic(r, args.variant, anchor_chars=args.anchor)[:clamp._COND_CHAR_CAP]
        for r in rows
    ]
    traces: list[dict[str, Any]] = [
        {"task_id": r.task_id, "level": r.level, "gold_identifier": r.gold_identifier,
         "next_line": r.next_line, "arms": {}} for r in rows
    ]
    per_fact: dict[int, dict[str, Any]] = {}

    def fact_sd(i: int) -> dict[str, Any]:
        if i not in per_fact:
            model.reset_adapter()
            per_fact[i] = assemble_native_sd(h, conds[i])
        return per_fact[i]

    async def gen_scored(i: int, prompt: str) -> dict[str, Any]:
        torch.manual_seed(args.seed)
        return clamp._score(await clamp._gen_line(model, prompt, args.max_new), rows[i])

    floor_prompts = [
        model.clamp_to_window(
            f"# Current file:\n{clamp._prefix(r)}\n# Next line:", w
        ) for r in rows
    ]
    for i in range(len(rows)):
        model.reset_adapter()
        traces[i]["arms"]["floor"] = await gen_scored(i, floor_prompts[i])

    for k in ks:
        for bundle in lib.make_bundles(len(rows), k):
            joined = lib.multi_cond_text([conds[i] for i in bundle])
            overhead = model.count_tokens(
                f"{clamp._TAIL_HEADER}\n{joined}{clamp._CURSOR_MARKER}"
            )
            arm_t = f"tail_k{k}"
            for i in bundle:
                if overhead > w:
                    traces[i]["arms"][arm_t] = {
                        "pred": "", "recovered": False, "infeasible": True,
                        "cond_tokens": overhead,
                    }
                    continue
                model.reset_adapter()
                prompt, _ = clamp._assemble_tail_prompt(
                    model, clamp._prefix(rows[i]), joined, w
                )
                s = await gen_scored(i, prompt)
                s["cond_tokens"] = overhead
                traces[i]["arms"][arm_t] = s

            modes: list[tuple[str, dict[str, Any]]] = []
            if k == 1:
                modes.append(("adapter_k1", fact_sd(bundle[0])))
            else:
                if args.mode in ("both", "a"):
                    model.reset_adapter()
                    sd_a = assemble_native_sd(h, joined)
                    modes.append((f"adapter_a_k{k}", sd_a))
                if args.mode in ("both", "b"):
                    comp = lib.compose_rank_stacked(
                        [fact_sd(i) for i in bundle], ctx_rank=h["ctx_rank"]
                    )
                    modes.append((f"adapter_b_k{k}", comp))
            for label, sd in modes:
                model.reset_adapter()
                model.hotswap_adapter(
                    scale_lora_b(lib.pad_adapter_rank(sd, h["r_camp"]), args.scaling)
                )
                for i in bundle:
                    s = await gen_scored(i, floor_prompts[i])
                    s["k"] = k
                    traces[i]["arms"][label] = s
    return traces


async def run_sanity(rows: list[Any], args: Any, cfg: Any) -> list[dict]:
    """Native-rank leg through the ENGINE path; must reproduce C1 bit-exactly."""
    import torch  # noqa: PLC0415

    from rune.bench.repobench import render_episodic  # noqa: PLC0415
    from rune.model.adapter import scale_lora_b  # noqa: PLC0415
    from rune.model.wrapper import ModelWrapper  # noqa: PLC0415

    model = ModelWrapper.from_config(cfg)
    traces: list[dict[str, Any]] = []
    for r in rows:
        floor_p = model.clamp_to_window(
            f"# Current file:\n{clamp._prefix(r)}\n# Next line:", args.window
        )
        rec: dict[str, Any] = {"task_id": r.task_id, "arms": {}}
        torch.manual_seed(args.seed)
        model.reset_adapter()
        rec["arms"]["floor"] = clamp._score(
            await clamp._gen_line(model, floor_p, args.max_new), r
        )
        cond = render_episodic(r, args.variant, anchor_chars=args.anchor)
        cond = cond[:clamp._COND_CHAR_CAP]
        model.reset_adapter()
        ar = model.generate_adapter(cond)
        torch.manual_seed(args.seed)
        model.hotswap_adapter(scale_lora_b(ar.state_dict, args.scaling))
        rec["arms"]["adapter_k1"] = clamp._score(
            await clamp._gen_line(model, floor_p, args.max_new), r
        )
        traces.append(rec)
    return traces


def compare_to_c1(traces: list[dict], c1_path: Path) -> dict[str, Any]:
    """Token-for-token prediction agreement vs the C1 trace artifact."""
    c1 = {t["task_id"]: t for t in json.loads(c1_path.read_text())}
    pairs = (("floor", "floor"), ("adapter_k1", "episodic_use"))
    out: dict[str, Any] = {}
    for ours, theirs in pairs:
        same = tot = 0
        for rec in traces:
            ref = c1.get(rec["task_id"], {}).get("arms", {}).get(theirs)
            arm = rec["arms"].get(ours)
            if not ref or not arm:
                continue
            tot += 1
            same += int(arm["pred"] == ref["pred"])
        out[f"match_{ours}"] = f"{same}/{tot}"
        out[f"match_{ours}_exact"] = same == tot
    return out


# ----------------------------------------------------------------------- cli

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--leg", choices=("sanity", "capacity"), default="capacity")
    ap.add_argument("--levels", default="8k,32k")
    ap.add_argument("--per-level", type=int, default=30)
    ap.add_argument("--offset", type=int, default=100)
    ap.add_argument("--window", type=int, default=768)
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variant", default="use")
    ap.add_argument("--anchor", type=int, default=0)
    ap.add_argument("--scaling", type=float, default=0.91)
    ap.add_argument("--ks", default=_KS_DEFAULT)
    ap.add_argument("--mode", choices=("both", "a", "b"), default="both")
    ap.add_argument("--margin", type=float, default=_MARGIN_DEFAULT)
    ap.add_argument("--experiment", default="issue52-c4")
    ap.add_argument("--out", default="/tmp/c4/capacity_traces.json")
    ap.add_argument("--c1-traces", default=None,
                    help="C1 run trace JSON for the sanity comparison")
    ap.add_argument("--stats-only", action="store_true",
                    help="recompute metrics + gate from an existing --out")
    ap.add_argument("--smoke", action="store_true",
                    help="first 8 rows, ks=1,2 (GPU plumbing check)")
    args = ap.parse_args()

    if args.stats_only:
        traces = json.loads(Path(args.out).read_text())
        m = capacity_metrics(traces)
        for key in sorted(m):
            print(f"{key} = {m[key]:.4f}")
        print(json.dumps(stage1_gate(traces, args.margin), indent=1))
        return

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import asyncio  # noqa: PLC0415

    import mlflow  # noqa: PLC0415

    from rune.config import load_rune_config  # noqa: PLC0415
    from rune.tracking import configure_mlflow, tracked_run  # noqa: PLC0415

    got = hashlib.sha256(Path(clamp.C3_CKPT).read_bytes()).hexdigest()
    if got != C3_SHA256:
        raise SystemExit(f"c3 ckpt sha {got} != pinned {C3_SHA256}")

    if args.smoke:
        args.ks = "1,2"
    levels = [x.strip() for x in args.levels.split(",") if x.strip()]
    rows = clamp._load_stratified(levels, args.per_level, args.offset)
    if args.smoke:
        rows = rows[:8]
    cfg = load_rune_config(None).override(
        checkpoint_path=clamp.C3_CKPT, thinking_budget=0, seed=args.seed,
        max_tokens=args.max_new, temperature=0.0,
    )
    engine_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False,
        cwd=str(Path(__file__).resolve().parent.parent),
    ).stdout.strip()

    if args.leg == "sanity":
        traces = asyncio.run(run_sanity(rows, args, cfg))
    else:
        k_max = max(int(x) for x in args.ks.split(","))
        h = load_capacity_handles(cfg, k_max)
        traces = asyncio.run(run_capacity(h, rows, args))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(traces, indent=1))
    m = capacity_metrics(traces)
    gate = stage1_gate(traces, args.margin) if args.leg == "capacity" else {}
    anchor = (
        compare_to_c1(traces, Path(args.c1_traces))
        if args.leg == "sanity" and args.c1_traces else {}
    )

    configure_mlflow(args.experiment)
    run_name = f"c4-{args.leg}-W{args.window}-K{args.ks}-off{args.offset}-seed{args.seed}"
    params = {
        "task": "C4-stage1-I5", "leg": args.leg, "window": args.window,
        "ks": args.ks, "mode": args.mode, "margin": args.margin,
        "levels": args.levels, "per_level": args.per_level, "offset": args.offset,
        "seed": args.seed, "episodic_variant": args.variant,
        "episodic_anchor": args.anchor, "episodic_scaling": args.scaling,
        "max_new": args.max_new, "n_rows": len(rows),
        "checkpoint_sha256": got, "engine_commit": engine_commit,
        "c1_anchor_run": "f37374906c5f",
    }
    with tracked_run(run_name, params=params):
        mlflow.log_metrics({k.replace("@", "_at_"): v for k, v in m.items()})
        mlflow.log_artifact(args.out)
        if gate:
            mlflow.log_dict(gate, "stage1_gate.json")
        if anchor:
            mlflow.log_dict(anchor, "c1_anchor.json")
    for key in sorted(m):
        print(f"{key} = {m[key]:.4f}")
    if anchor:
        print(json.dumps(anchor, indent=1))
    if gate:
        print(json.dumps(gate, indent=1))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the unit tests**

Run: `uv run pytest tests/unit/test_c4_capacity_stats.py tests/unit/test_c4_capacity_lib.py tests/unit/test_c4_fixture_audit.py -q`
Expected: all pass (stats/gate paths are pure; the module import must stay CPU-safe — heavy imports are all deferred).

- [ ] **Step 5: CPU dry-runs (house discipline, per c21_prep.md §3)**

```bash
uv run python tools/_c4_capacity_run.py --help
uv run ruff check tools/_c4_capacity_run.py tests/unit/test_c4_capacity_stats.py
uv run pytest tests/unit/ -q
```
Then exercise `--stats-only` end-to-end on a synthetic trace file:

```bash
uv run python - <<'EOF'
import json, pathlib
traces = [
    {"task_id": f"t/{i}", "arms": {
        "floor": {"pred": "x", "recovered": i < 9},
        "adapter_a_k2": {"pred": "x", "recovered": i < 12},
        "adapter_b_k2": {"pred": "x", "recovered": i < 40},
        "tail_k2": {"pred": "x", "recovered": i < 50},
    }} for i in range(60)
]
pathlib.Path("/tmp/c4").mkdir(exist_ok=True)
pathlib.Path("/tmp/c4/synth_traces.json").write_text(json.dumps(traces))
EOF
uv run python tools/_c4_capacity_run.py --stats-only --out /tmp/c4/synth_traces.json
```
Expected: metric lines for all four arms plus a gate JSON with `"go": true`.

- [ ] **Step 6: Commit**

```bash
git add tools/_c4_capacity_run.py tests/unit/test_c4_capacity_stats.py
git commit -m "feat(#52): C4 I5 capacity-curve runner (sanity + capacity legs)"
```

---

### Task 5: GPU campaign — sanity anchor, smoke, full capacity leg, independent verification (~1 GPU-hr)

**Files:**
- Create (runtime): `/tmp/c4/{c1_traces.json,sanity_traces.json,capacity_traces.json,*.log}`
- Modify: `docs/publication/hashes.txt` (append trace shas)

**Interfaces:**
- Consumes: Task 4's CLI; the C1 trace artifact from MLflow exp 79 run `f37374906c5f…`.
- Produces: verified trace files + MLflow runs in `issue52-c4`; the gate JSON Task 6 reads.

- [ ] **Step 1: Fetch the C1 trace artifact**

```bash
uv run --no-sync python - <<'EOF'
import mlflow
mlflow.set_tracking_uri("http://localhost:5000")
runs = mlflow.search_runs(
    experiment_names=["issue52-repobench-clamp"],
    filter_string="", output_format="list",
)
run = next(r for r in runs if r.info.run_id.startswith("f37374906c5f"))
p = mlflow.artifacts.download_artifacts(run_id=run.info.run_id, dst_path="/tmp/c4/c1_artifacts")
print(p)
EOF
ls /tmp/c4/c1_artifacts
```
Expected: the C1 traces JSON (the clamp run's `--out` artifact). Copy/rename it to `/tmp/c4/c1_traces.json`. If the tracking DB lacks the run (DB-loss precedent), fetch the artifact directly from the S3 store under `mlflow/artifacts/79/f37374906c5f*/artifacts/`.

- [ ] **Step 2: Sanity leg (native rank, engine path) — S1-ANCHOR-1**

```bash
free -g && nvidia-smi
nohup env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  uv run --no-sync python tools/_c4_capacity_run.py \
    --leg sanity --out /tmp/c4/sanity_traces.json \
    --c1-traces /tmp/c4/c1_traces.json \
  > /tmp/c4/sanity.log 2>&1 &
```
Expected (~10–15 min): `match_floor = 60/60`, `match_adapter_k1 = 60/60`, both `_exact: true`. **Any mismatch → stop; debug environment (ckpt, engine commit, flash-attn) before proceeding.**

- [ ] **Step 3: Capacity smoke (plumbing check)**

```bash
uv run --no-sync python tools/_c4_capacity_run.py \
  --leg capacity --smoke --out /tmp/c4/smoke_traces.json \
  > /tmp/c4/smoke.log 2>&1
```
Expected (~5 min): completes without exception; traces contain `floor`, `tail_k1/2`, `adapter_k1`, `adapter_a_k2`, `adapter_b_k2` entries with non-null `recovered`; no shape errors on hotswap (this is the first live test of pad-to-campaign-rank).

- [ ] **Step 4: Full capacity leg**

```bash
nohup env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  uv run --no-sync python tools/_c4_capacity_run.py \
    --leg capacity --out /tmp/c4/capacity_traces.json \
  > /tmp/c4/capacity.log 2>&1 &
```
Expected (~40–60 min): per-arm metrics printed; `stage1_gate` JSON printed; MLflow run `c4-capacity-W768-K1,2,4,8-off100-seed0` in `issue52-c4` with the traces artifact.

- [ ] **Step 5: S1-ANCHOR-2 + bias-invariance assertion**

Compare `adapter_k1` predictions between `/tmp/c4/capacity_traces.json` (enlarged rank) and `/tmp/c4/sanity_traces.json` (native rank):

```bash
uv run python - <<'EOF'
import json
cap = {t["task_id"]: t for t in json.load(open("/tmp/c4/capacity_traces.json"))}
san = json.load(open("/tmp/c4/sanity_traces.json"))
same = sum(
    cap[t["task_id"]]["arms"]["adapter_k1"]["pred"]
    == t["arms"]["adapter_k1"]["pred"]
    for t in san if t["task_id"] in cap
)
print(f"S1-ANCHOR-2: adapter_k1 agreement {same}/{len(san)}")
EOF
```
Expected: ≈60/60; <55 → investigate before reading gates. Also assert the mode-(b)
premise on live tensors — during the smoke run (or a 2-row probe), dump two per-fact
state dicts and check `torch.equal(sd1[key][ctx_rank:], sd2[key][ctx_rank:])` for one
`lora_A` key: the bias rank-slices of adapters built from *different* conditionings
must be identical (bias is conditioning-independent). Expected `True`; record both
results in the findings doc's Anchors section.

- [ ] **Step 6: Independent verification (house discipline)**

A stdlib-only script (no harness imports) recomputes from `/tmp/c4/capacity_traces.json`: every per-arm recovery count, every Wilson CI, every McNemar p, the bundle-level sign counts at K=2, and the gate booleans. Every number must match the runner's printout exactly; any discrepancy → stop and reconcile before Task 6. (At execution time this step should be run by a separate verifier agent, matching the C1/C3.2 verification pattern.)

- [ ] **Step 7: Append hashes + commit**

```bash
sha256sum /tmp/c4/sanity_traces.json /tmp/c4/capacity_traces.json /tmp/c4/i0_audit.json \
  >> docs/publication/hashes.txt
git add docs/publication/hashes.txt
git commit -m "docs(#52): C4 stage-1 trace + audit hashes"
```

---

### Task 6: Gate evaluation, findings doc, PR comment

**Files:**
- Create: `docs/publication/c4_stage1_findings.md`
- Modify: `mkdocs.yml` (nav entry under the publication docs section)

**Interfaces:**
- Consumes: `/tmp/c4/i0_audit.json`, `/tmp/c4/capacity_traces.json`, the gate JSON, verification report.
- Produces: the go/no-go record and the article-side handoff line; the PR #60 comment.

- [ ] **Step 1: Write the findings doc**

`docs/publication/c4_stage1_findings.md`, house style (mirror `c32_wsweep_findings.md`):
sections **Pre-registration** (copied verbatim from this plan), **I0 result**
(fixture discovery + regenerated-session reuse fraction ⟨r/e = f⟩ vs the 0.60
threshold + the OOD statement from Task 2 Step 4), **I5 result** (per-K per-arm
table with Wilson CIs; the two build modes side by side; infeasibility counts;
K\* crossover statement; bundle-level sensitivity), **Realized gate** (S1-GO or
NO-GO with the branch consequence exactly as pre-registered), **Anchors**
(S1-ANCHOR-1 60/60 statement, S1-ANCHOR-2 agreement, bias-invariance check),
**Verification** (independent recompute: 0 discrepancies), **Provenance**
(MLflow exp/run ids, trace sha256s, engine commit). All ⟨…⟩ slots are filled
from the verified numbers only.

- [ ] **Step 2: Add the doc to mkdocs nav and build**

Add `c4_stage1_findings.md` next to the other `docs/publication/` nav entries in `mkdocs.yml`.
Run: `uv run mkdocs build 2>&1 | tail -3` → expected: clean build, no warnings about the new page.

- [ ] **Step 3: Full quality gate**

```bash
uv run ruff check . && uv run mypy src/ && uv run pytest tests/unit/ -q
```
Expected: all clean/green.

- [ ] **Step 4: Commit and post the PR comment**

```bash
git add docs/publication/c4_stage1_findings.md mkdocs.yml
git commit -m "docs(#52): C4 stage-1 findings — I0 audit + I5 capacity go/no-go"
git push
```
Post to PR #60 (via `gh pr comment 60 --body-file …`): the realized-gate summary
in the same format as the C2.1/C3.2 comments — headline numbers, gate branch
taken, anchors, provenance — ending with the explicit Stage-2 decision request:
(1) confirm/adjust the margin M (was proposed +0.15, pre-registered before the
run), (2) adapter build mode (a) vs (b) for Stage 2 informed by the curve,
(3) whether the Doc-to-LoRA/SHINE niche suffices or SHINE gets benchmarked
directly. **Stage 2 is not started in this plan regardless of the gate outcome.**

---

## Stage 2 — deferred (separate plan, written only after S1-GO + co-author decisions)

Not part of this plan's tasks; recorded here so the interface anchors don't rot:

- **I3 clean adapter arm:** one-line change `assistant_prefix=""` at `src/rune/engine/graph.py:1031` (the parameter is required, keyword-only — pass empty, don't remove); `prompt_code_continue.j2` is already clean; assert no KV persistence per round (none exists today).
- **I1 scaffold:** raise continuation `max_new` 48 → ~384; contention threshold vs `W`; head-preserving truncation; same overflow policy on the full-reprompt arm; `_ACCUMULATED_CODE_CAP` (3500, `graph.py:156`) applies to the adapter channel only — revisit under the pre-registered build mode.
- **I2 prompt-side baseline family:** extractive-oracle, frozen-base self-summary (primary), structured-GT, summary+retrieval at budgets s₁ (systems parity) and s₂ = ⌊W/2⌋.
- **I4 systems + null arms:** full-reprompt, floor, `kv_reinject` (new machinery — nothing persists KV today); measure wall-clock, FLOPs, peak-KV-bytes per round.
- **I6 provenance:** new MLflow experiment; assert c3 state-dict sha == C1's; per-round JSONL trace.
- **Gates:** G1 adapter ≥ strongest prompt baseline (max of self-summary, summary+retrieval) at s₁, paired McNemar p<0.05 & Δ≥+0.05; G2 capacity crossover K\*≤8 (I5 delivers the K\* estimate). Consistency metric: per-identifier AST-level reuse, stratified boundary/surprise/control; N ≥ 80 tasks; Holm across the confirmatory family; tie ≡ null.

## Effort & GPU budget

| Task | Eng effort | GPU |
|---|---|---|
| 1 (audit tool) | ~2 h | — |
| 2 (session regen + audit) | ~1 h | ~0.5 h |
| 3 (composition lib) | ~2 h | — |
| 4 (runner) | ~4 h | — |
| 5 (campaign + verification) | ~2 h | ~1 h |
| 6 (findings + comment) | ~2 h | — |
| **Total** | **~1.5–2 eng-days** | **~1.5 GPU-hr** |

Slightly above the PR comment's ~1 day + 0.5 GPU-hr because I0 requires session
regeneration (the committed fixtures turned out to be step-0-only) and the run
adds the pre-registered sanity/anchor leg — both discovered necessities, both
cheap relative to the risk they retire.
