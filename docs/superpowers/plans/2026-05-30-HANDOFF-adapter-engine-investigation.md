# HANDOFF — finish non-training fixes + merge (`fix/prompt-layer-remediation`)

**For the next (context-cleared) session. Scope = the NON-training fixes on this branch,
then the merge. The hypernetwork/training work is OUT OF SCOPE here — it lives in
[issue #49](https://github.com/ElixirTrials/rune/issues/49) and a separate branch.**

## 0. Where we are
- **pass@1 0 → 1.0 is fixed and committed** (your commit `2faf73f8`): `wrapper.py`
  `lora_alpha=alpha` (revert `bce5f2fe`'s 8× over-scaling) + `hypernetwork.py`
  `torch.load(..., mmap=True)` (reliable load on the ~15GB-RAM box). ruff/mypy/216 unit
  tests were green at commit time.
- **The hypernetwork adapter is inert** (trained `scaler_B` collapsed to ~0 across *all*
  checkpoints). That is a **training** problem → **issue #49**, NOT this branch. Do not try
  to fix the adapter here. pass@1 currently works because the base model solves from the
  prompt; the adapter is a no-op.
- This branch still owes the original remediation **merge** (task #12) + a few small
  engine/quality fixes uncovered this session.

## 1. Outstanding fixes to finish on THIS branch (in order)

### (a) flash_attention_2 on the base-model load — low-risk perf
`src/rune/model/wrapper.py:88` loads `AutoModelForCausalLM.from_pretrained(model_id,
dtype=torch.bfloat16)` with **no `attn_implementation`** → defaults to `sdpa`. flash-attn
**2.8.3 is installed**. Add `attn_implementation="flash_attention_2"`. Verify it loads and a
smoke generation works (guarded GPU run). (This is the only base-model FA2 site; the
`hypernetwork.py` flash code is the perceiver and is intentionally `eager` — leave it.)

### (b) Self-test oracle / retry-exhaustion — DECIDE approach first, then fix
The in-loop success oracle runs the **model's own self-authored tests** (`prompt_code.j2:4`
"write tests FIRST"; pass = `parse.py:78` exit 0), while pass@1 uses **held-out** tests
(`runner.py:142`). On `mbpp/279` the repair loop ran to exhaustion (~8 min wasted) yet the
code passed at bench time. **The naive fix is a NO-GO** (validated by workflow
`wf_2cb5affa-71d`): appending held-out tests still fails if the model's wrong self-test
remains, unittest classes without `main()` pass *vacuously*, and injecting held-out tests
**leaks them into the repair prompt** via `graph.py:102 error_summary → code_repair.j2`.
Options (pick one, keep it leak-free):
  1. In-loop, exec **only the extracted implementation** (strip the model's tests) so a
     correct impl exits 0 → no spurious retries. Weaker signal ("runs" not "correct") but
     safe and simple.
  2. Thread `task.test_code` through `runner.py make_initial_state` into a new `RunState`
     field and run it in-loop, **but sanitize `error_summary`** so oracle asserts never reach
     `code_repair.j2`. More faithful, more work.
  Recommend (1) for this branch (speed fix, no eval leak); leave (2) as a follow-up.
  *Lower priority than the merge — if time-boxed, defer behind a tracking note.*

### (c) Diagnostic-tool cleanup + lint green
Many `tools/diag_*.py` probes were created this session. Keep the ones issue #49 references
(`diag_retrieval_probe.py`, `diag_continuation_probe.py`, `diag_recall_probe.py`,
`diag_scaling_mode_probe.py`, `diag_format_probe.py`) and make them **ruff-clean**; remove the
pure scratch (`diag_checkpointer_*.py`, and `/tmp/*_smoke.py`). `ruff check .` currently has
~12 errors, all in these diag tools — fix or delete the offenders so the tree is green.
Then **commit** the kept tools + the two new docs (this handoff + `docs/superpowers/issues/
2026-05-30-hypernetwork-collapse-two-stage-retraining.md`) so the working tree is clean before
the merge checkout.

### (d) Re-confirm CPU gates
`uv run ruff check .` && `uv run mypy src/` (was clean, 34 files) && `uv run pytest tests/unit/ -q`
(was 216 passed). All green before the merge.

## 2. The merge (task #12 / original plan) — do AFTER (a)-(d) and user confirm
```
git -C /workspaces/rune-gpu checkout fix/pr45-review-correctness
git -C /workspaces/rune-gpu merge --no-ff fix/prompt-layer-remediation
# resolve conflicts, then re-confirm gates:
uv run ruff check . && uv run mypy src/ && uv run pytest tests/unit/ -q
git -C /workspaces/rune-gpu push origin fix/pr45-review-correctness
```
End commit messages with the Co-Authored-By line per CLAUDE.md.

## 3. Explicitly OUT OF SCOPE here → issue #49 + new branch
- Hypernetwork `scaler_B` collapse / retraining / two-stage mining.
- **Adapter-application correctness** (latent until a good checkpoint, so can't verify here):
  apply `combine_lora` + `get_head_bias()`/`bias_A` (`hypernetwork.py:391`); un-contaminate
  activation extraction via `disable_adapter()` (`wrapper.py:104`); re-tune adapter scaling
  (base lever; continuation = 1.5× base). All tracked in issue #49 §D.
- Do NOT relaunch HPO — its objective was gamed by the collapse.

## 4. Environment / discipline (learned this session)
- **CPU RAM ~15GB.** `offload_base=True` OOM-kills the VM; `mmap=True` checkpoint load +
  `offload_base=False` is what makes base+hypernet load. Launch every GPU job under
  `/tmp/run_guarded.sh` (RAM watchdog, now kills the whole process group — no orphans).
- GPU single-tenant (23GB). `nvidia-smi ~0` before launch; one job at a time; wait with the
  Monitor tool, never sleep-chains. A live **MLflow** server is at `http://localhost:5000`.
- Results stream to `/tmp/*_results.jsonl` (survive idle-shutdown/reboot; `/tmp` persisted).

## 5. Task-list mapping
- #12 merge — PENDING (§2), after §1 fixes + confirm.
- #3 flash_attention_2 — §1(a).
- pass@1 > 0 — DONE (committed; caveat: prompt-driven, adapter inert).
- Hypernetwork/training — moved to issue #49 + a new branch (NOT here).
