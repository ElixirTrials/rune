# HANDOFF — pass@1 diagnosis + faster generation (2026-05-30)

**For the next session. Read this top to bottom before running anything.**

## 0. Current goal (set by user)
`/goal success = pass@1 probe + faster token generation`

Concretely: figure out **why the engine has never scored pass@1 > 0** and why a simple MBPP
task takes ~24 min, fix the root cause, demonstrate pass@1 > 0 on a tiny probe, then finish
the original remediation plan and do the local merge + push.

## 1. How this connects to the original plan
- Original plan: `docs/superpowers/plans/2026-05-29-prompt-layer-remediation.md` (Phases 0–3 +
  integration). All code tasks (#1–#11, #14, #15 in the task list) are **committed** on branch
  `fix/prompt-layer-remediation` (20 commits). CPU gates are green: `ruff` clean, `mypy` clean
  (34 files), **216 unit tests pass**.
- The plan's merge blocker is the **goal gate (task #13): "HPO runs without OOM + real pass@1."**
  Investigation this session showed that gate **cannot be met as written** until a deeper,
  pre-existing bug is fixed: the engine has essentially **never produced a passing benchmark
  result** (see §2). So the goal was refocused (by the user) onto: diagnose pass@1=0 + speed,
  fix, then merge.
- **Final integration still owed (original plan, task #12):** after the fix + a green pass@1
  probe, merge `fix/prompt-layer-remediation` → `fix/pr45-review-correctness` locally, then push.
  See §6 for exact steps.

## 2. What is VERIFIED (trust these — they are measured/read, not guessed)
1. **EOS is correct — NOT the bug.** `tokenizer.eos_token = <|im_end|>` (id 248046);
   `tokenizer.pad_token = <|endoftext|>` (248044); `</think>` is a single token (248069).
   `config.json` has `eos_token_id: null` but the tokenizer carries the right EOS, and
   `inference.generate()` passes `eos_token_id=tokenizer.eos_token_id`. The `generation_config.json
   → 404` in logs is benign. (Verified by reading the HF cache snapshot + loading the tokenizer.)
2. **The engine has ~never scored.** Across the ENTIRE MLflow history (`mlflow.db`, all
   experiments), exactly **one** run ever recorded a `pass_at_1` metric, and it was **0.0**
   (run `306f3399`, an ancient config). Every `bench-hpo` run FAILED or was orphaned with zero
   metrics. So pass@1=0 is **long-standing, not a regression from the remediation.**
3. **Optuna DB (`optuna_bench_hpo.db`): 1 trial, state=FAIL, 0 complete, no best_trial.** That
   trial used the OLD search space (`max_tokens`, no `presence_penalty`) — i.e. predates the
   Task-15 HPO fix. No HPO has run on current HEAD.
4. **MLflow exp 38 = 9 top-level runs, ZERO nested trial runs.** The last run `aa3e10b9` (the
   calibration bench, started 21:38, AFTER the OOM-fix commit 21:00 and HPO-fix 21:05) is the
   only run that exercised current HEAD. Its log shows **no OOM and no crash** — but the
   instance died ~22:09 mid-task-1-of-16, so **no pass@1 was emitted.**
5. **Calibration log (`/tmp/bench_calib.log`) facts:** one simple MBPP task ran 21:38→22:02
   (~24 min) before the first "JSON output truncated (11165 chars)" warning; the structured
   output hit the full `max_tokens` (3072) cap and never closed the JSON object. Continuation
   then added 3072 more tokens. So a trivial task produced a maxed-out, unclosed `{"code":...}`.
6. **Code facts (read, not run):**
   - `inference.py`: phase-2 `generate()` and `generate_continuation()` stop on
     `tokenizer.eos_token_id`. When `output_schema` is set, an `xgrammar` LogitsProcessor
     constrains output to the JSON schema.
   - `PresencePenaltyLogitsProcessor` (inference.py:24-31) subtracts the penalty from **every
     token seen so far, unconditionally** — including `"`, `}`, `\n`.
   - `bench/runner.py`: pass = sandbox exit 0 on `integrated_code` (or joined `code_results`)
     + `test_code`. If structured output never parses → no code → guaranteed fail.

## 3. PRIME HYPOTHESIS (strong, but NOT yet empirically confirmed)
**xgrammar JSON-schema decoding never terminates**, because:
- A JSON string value (`{"code": "<unbounded string>"}`) lets the grammar permit "more string"
  at every step, AND the grammar **masks the EOS token until the JSON object is closed**. The
  model would naturally emit `<|im_end|>` (it does in the raw, no-grammar case per the model's
  training) but the constraint suppresses it; nothing else forces the closing `"}`.
- Result → every structured generation runs to `max_tokens` → truncated → unparseable →
  0 code extracted → **pass@1=0**, AND every call burns the full token budget → **~24 min/task**.
- This single root cause would explain BOTH symptoms (the score and the speed).

**Sub-hypotheses already considered:** presence_penalty and thinking-mode are likely NOT the
root cause (raw decoding should stop fine without them); they may worsen the tail. Speed
(~tok/s) on this A10G is partly hardware (small card), so the real speed lever is **generating
fewer tokens** (stop-on-complete), not faster decode.

## 4. ⚠️ HONEST STATUS OF THE PROBE — it never produced data
I wrote `tools/diag_inference_probe.py` to test §3 (it calls the REAL `inference.generate()` on
base Qwen3.5-9B, no adapter, one task, one knob at a time: raw vs grammar vs +presence vs
+thinking, logging wall time / tok/s / hit-cap / parseable to `/tmp/probe_results.jsonl`).

**It NEVER successfully ran.** Every attempt was killed or OOM'd because I launched multiple
copies concurrently (two 9B loads = OOM on the 23GB card). **`/tmp/probe_results.jsonl` is
empty (0 bytes).** Any results table in `/tmp/diagnosis_notes.md` is **speculative, NOT
measured — do not trust those numbers.** The probe script itself is fine and is the right tool;
it just needs to be run **exactly once** (see §5).

## 5. NEXT STEPS (in order)
1. **Run the probe ONCE, cleanly** to confirm/refute §3:
   `nohup uv run python tools/diag_inference_probe.py > /tmp/probe_run.log 2>&1 &`
   Then wait with the **Monitor tool** (not sleep-chains). Expected ~5–6 min (11s load + 4
   configs at the card's real tok/s). Read `/tmp/probe_results.jsonl`.
   - KEY CONTRAST: does `raw_nothink_nopen_nogrammar` STOP on its own (truncated=false) while
     `grammar_nothink_nopen` runs to the cap (truncated=true, unparseable)? If yes → §3 confirmed.
2. **If confirmed, fix root cause** (highest value — fixes score AND speed):
   Make structured generation **stop as soon as the grammar accepts a complete object.**
   - Investigate the installed `xgrammar` API for matcher completion state (e.g.
     `matcher.is_terminated()` / accepting state) and add a HF `StoppingCriteria` that halts
     when the JSON object is complete. Confirm exact API in the installed version first.
   - Complement (defense in depth, already partly present): `extract_partial_code` /
     `json_repair` so a truncated-but-present code value can still be scored.
   - Secondary speed: disable thinking for code/structured actions; lower `max_tokens` for
     simple tasks.
   Use the **systematic-debugging skill**: confirm root cause from probe data BEFORE editing.
3. **Verify the fix** with a tiny pass@1 probe (3–4 MBPP tasks, e.g. a small slice of
   `/tmp/mbpp_subset16.json` if it still exists, else build from the cached MBPP dataset at
   `~/.cache/huggingface/hub/datasets--google-research-datasets--mbpp`). Demonstrate
   **pass@1 > 0** and reasonable per-task time. This is the goal gate.
4. **Fix Optuna→MLflow logging (user item 2):** nested run per trial + logged params +
   intermediate objective metric, visible mid-run. Today there are 0 nested runs because
   `MLflowCallback` only fires on trial **completion** and none completed. Make trials log
   their config and an objective even on failure / as they go, so we can check traces
   occasionally instead of waiting for the whole run.
5. **Launch the next HPO** (background, shutdown-resilient: persistent Optuna DB so trials
   resume; small n_trials first to validate the loop end-to-end).
6. **Then do the merge + push (§6).**

## 6. FINAL MERGE + PUSH (original plan task #12) — exact steps
Only after pass@1 > 0 is demonstrated and the user confirms.
```
# from /workspaces/rune-gpu, branch fix/prompt-layer-remediation
git -C /workspaces/rune-gpu checkout fix/pr45-review-correctness
git -C /workspaces/rune-gpu merge --no-ff fix/prompt-layer-remediation
# resolve any conflicts, then:
uv run ruff check . && uv run mypy src/ && uv run pytest tests/unit/ -q   # re-confirm gates
git -C /workspaces/rune-gpu push origin fix/pr45-review-correctness
```
(End commit messages with the Co-Authored-By line per CLAUDE.md. Decide whether
`tools/diag_inference_probe.py` should be committed or removed before merge.)

## 7. ⚠️ BASH DISCIPLINE — how I made a mess, and the rules to follow
What went wrong: I chained `sleep N; cmd` (the harness BLOCKS sleep-chaining), I sent many
**parallel** Bash calls where the first exited non-zero (e.g. `pkill` returns 1 when nothing
matches) which **cancelled all the others**, and I launched the GPU probe multiple times
concurrently → **OOM, killing my own runs and producing zero data.**

RULES for next session:
- **One Bash call per message** when commands depend on each other or on GPU state. Do NOT
  batch multiple GPU/process commands in parallel.
- **Never sleep-chain.** To wait for a background job, use `run_in_background: true` and then
  the **Monitor tool** with an until-condition (e.g. grep the results file for `"event":
  "done"`). Do not poll with `sleep`.
- **GPU is single-tenant (23GB, one 9B model fits once).** Before launching anything on GPU:
  `nvidia-smi --query-gpu=memory.used --format=csv,noheader` and confirm it's ~0. **Launch at
  most ONE GPU job at a time.** Never start a second while one is alive.
- `pkill`/`grep` return non-zero when there's no match — append `|| true`, or run them alone,
  so they don't cancel a batch.
- To kill a GPU job cleanly: `pkill -9 -f <script_name> || true`, then `sleep 4` in its OWN
  call, then check `nvidia-smi`.

## 8. Environment notes
- GPU machine, A10G 23GB. Run GPU jobs directly (CLAUDE.md allows it). Capture output to files.
- MLflow: the docker stack isn't runnable (no `docker` here). I started a local server with
  `uv run mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db`
  to read history; **kill any stray `mlflow server` you find** if not needed
  (`pkill -f "mlflow server" || true`). The real backend store is `./mlflow.db`; experiment 38
  is `rune-bench` / `bench-hpo`.
- `uv sync` prunes the gpu extra — use `uv sync --extra gpu` if deps look missing (memory
  `project_uv_sync_gpu_extra.md`).
- Working tree: only untracked file is `tools/diag_inference_probe.py`. Branch
  `fix/prompt-layer-remediation`, 20 commits, clean otherwise.

## 9. Task list mapping
- #12 Final integration (merge+push) — PENDING, blocked on goal gate (§6).
- #13 Goal gate — REDEFINED as: diagnose+fix pass@1=0 & speed, then demonstrate pass@1>0 (§5).
- #15 HPO trace findings — code committed but UNVALIDATED (no HPO has completed on HEAD).
- New work: probe-confirm root cause → fix grammar non-termination → pass@1 probe →
  Optuna/MLflow logging → HPO → merge.
