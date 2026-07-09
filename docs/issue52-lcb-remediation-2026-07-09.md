<!-- Companion to issue52-lcb-failure-modes-2026-07-08.md: the remediation campaign.
     Fable-orchestrated multi-agent workflows (Opus 4.8 workers) + interactive trace
     review; all changes flag-gated, adversarially verified, no task-specific logic. -->

# LCB failure-mode remediation — 2026-07-09 (`issue52-c4`)

**Input:** the six-task autopsy of 2026-07-08 (`issue52-lcb-failure-modes-2026-07-08.md`).
**Goal:** convert what is convertible without cheating; make every remaining failure honestly
measured and cheaply bounded. **Commits:** `a5ed9d5` (ship gate), `17310c5` (budget guards +
resume stamp), `d2bf020` (repair context), `8ed6cd9` (conditioning blackout + concise-code +
judge CLI), `b55a80b` (stricter instruction). **Runs (MLflow exp `issue52-c4`, c3, seed 0,
full mode):** i1 `f5b4a4a8`, i2 `2d5ca6c8`, i3 `a405f40e`; baseline i0 `8bfc80c4` (07-08).
Runs `9448b8ff`/`92fcb946` are orphans of a VM shutdown and an OOM kill (mark KILLED).

## 1. What changed, with verification

| wave | change (flag, default OFF) | verified by |
|---|---|---|
| 1a | Ship gate grades/ships the normalized entry form (A1) + grader-mirror imports (A2) | 5 TDD regression tests reproduce the doc's exact NameErrors pre-fix; shipped form byte-identical to `normalize_lcb_submission`; mirror preamble proven a strict subset of the official harness star-imports |
| 1b | Official grading env restored (A4) + documented re-provision steps | official grades reproduce the doc header table exactly (1/6, 3799 PASS 35/35); ~20 s to grade the slice |
| 2 | `repair_dedup_after` (floored ≥2), `complexity_repair_cap`, `continuation_structural_guard`; `grading_gate_version` resume stamp (not flagged) | session-replay against recorded i0 traces: cap fires at attempt 2 (3777) / 4 (3801), 3799's genuine 3-distinct-failure repair chain provably untouched; stale metadata re-serve blocked |
| 3 | `repair_context_fix`: thin repair prompt renders repair brief + last-failure line; history errors tail-cut (payload survives); spec label cut at line boundary + `[spec truncated]` marker | pre-fix prompts byte-identical empty-Diagnosis reproduction; i2 traces show brief+got/want in every repair prompt |
| 4a | `adapter_cond_budget_fix`: conditioning packed to ~6800 chars (hypernet encoder right-truncates at 2048 tokens) priority Task > Feedback > Code > Attempts | measured blackout in i1 (3754 s6–s10: hypernet saw Task+Code ONLY); i3: 0/55 attempts over budget, feedback section present in 55/55 |
| 4b | `concise_code_instruction` (zeroshot untouched, test-guarded) | i3 3753: comment ratio 0% (was 26–80%); 3754 stress case still 80% → wording tightened (`b55a80b`), pending next run |
| 4c | `--model-judge` CLI passthrough (judge itself pre-existed) | precedence audited: judge runs only on oracle-passing units, flips pass→fail only with a grounded failing input, fails open, quality-protected |

Cheating audit (every wave): no qid/entry-point literals in src/, thresholds config-only,
`test_code`/private suites never reach prompts, oracle feedback, briefs, or conditioning;
the in-loop signal remains public-checks-only. Independent adversarial verifiers confirmed
each wave; forensic re-probe of all 55 i1+i2 attempts found **zero functional false
negatives** (every recorded verdict reproduces deterministically).

## 2. Run-by-run (same seed 0; per-task outcomes are trajectory-stochastic)

| run | stack | wall | internal | official | notes |
|---|---|---|---|---|---|
| i0 (07-08) | none, `--no-grade` | ~95 min | 1/6 (wrong task: 3777) | 1/6 (3799) — graded post-hoc | 3754 alone ~62 min; 3 labels corrupted by A1/A2 |
| i1 | waves 1–2 | 74 min | 4/6 (honest publics-only) | **1/6 (3799, 35/35)** | first officially-recorded 3799 solve; guards fire exactly as replay predicted |
| i2 | + wave 3 | 83 min | 4/6 | 0/6 | 3799 shipped a publics-passing near-miss (B3) — the judge's exact target, judge off |
| i3 | + wave 4, judge on | **52 min** | 2/6 | 0/6 | judge never exercised (nothing passed publics); 3799 scattered (9 distinct failures); conditioning fix verified 55/55 |

Internal pass@1 is a publics-only advisory metric by construction (`test_code ==
public_checks`); official grading is the ground truth and now runs in-loop on every run.

## 3. Root causes established from traces (with smoke-test proof)

1. **Ship-form + import defects (A1/A2)** inverted 2 of 6 verdicts and hid the only real
   solve. Fixed + regression-tested.
2. **Full-mode repair prompts were signal-starved**: `repair_brief` suppresses the diagnose
   step (policy) but only episodic templates rendered the brief → `Diagnosis:` empty by
   construction; history truncation head-cut stderr so got/want appeared in **0/N** attempt
   blocks. Reproduced byte-identically, then fixed (wave 3).
3. **Adapter conditioning blackout**: encoder truncates at 2048 tokens; oversized
   `## Current Code` (the degenerate blob itself) evicted `## Review Feedback` entirely —
   i1 3754 repaired blind from s6 on. Self-amplifying. Fixed (wave 4a), verified 55/55.
4. **Thinking leaks into completions** (`thinking_budget=0` has no thinking phase at all):
   3754 spent 60–80% of generated lines on comment/prose reasoning → token-budget
   truncation → salvage → headless `-> None`. Instruction helps where the model knows what
   to write (3753: 0%); under uncertainty it is ignored (3754: 80%). A thinking-budget arm
   is the untried lever.
5. **Publics-passing near-misses are a coin flip** (B3): 3799 has both a correct basin
   (i1, 35/35) and near-miss basins (i2) reachable from identical config; 3753's abs-bug
   family likewise. Only the model judge (or stronger publics) can defend in-loop.
6. **Complexity wall (B2), dissected**: the *measurement* is correct — static floor
   analysis + (unused here) empirical big_o probe; both vetoes officially validated TLE;
   zero false vetoes. The *communication* is delivered — verdict + required class reach
   the model in stderr, repair brief, prompt, and (post-4a) provably in conditioning. The
   failure is **actionability**: "need O(log n)" names the what, not the how; a 4B without
   digit-DP in its repertoire re-submits brute force (i1 3777 ×9) or "fixes" the wrong
   thing (3801's digit-0 mutations breaking publics). Observed failure modes: (i)
   identical re-submission; (ii) misdirected semantic edits alternating with complexity
   rejections; (iii) guidance ceiling — the only legal enrichment left is a generic,
   task-agnostic class→technique-family hint table (log n → binary search / digit-DP /
   closed form; n³ → interval DP …) in the complexity brief; June's perfect-oracle result
   (0/11 conversions) caps expectations. Minor bug found: brief lags one attempt behind
   `Last failure` when failure types alternate (i2 3801 s4).

## 4. What converted, what did not

- **Converted to recorded fact:** 3799's solve (i1, official 35/35) — previously invisible.
  Measurement integrity across the board; label fidelity now version-stamped.
- **Converted to recovered budget:** wall time 95 → 52 min (−45%) with no real solve cut.
- **Not converted:** 3748/3754 (algorithm derivation), 3777/3801 (complexity wall) — the 4B
  capability ceiling, exactly as the source doc predicted; and per-task official outcomes
  on this adversarial slice fluctuate 0–1/6 across trajectories (June arms: same range).
  No legitimate engine change on the table converts these; anything that would is the
  retracted-answer-injection anti-pattern.

## 5a. Amendment — run i4 (`--no-complexity-repair-preserve-logic`, judge on)

User trace-review question ("identical resubmission under a complexity verdict seems
fishy") exposed that `complexity_repair_preserve_logic=True` (default since `db48504`,
2026-06-10) makes the complexity brief say **"Do NOT replace the algorithm with a
structurally different approach"** and hardcodes `replan_recommended=False` — so
`replan_on_complexity` was dead code, and identical brute-force resubmission was
*instructed compliance with an unsatisfiable directive*, not (only) capability. The
directive provably reached i2 3801's prompts 3/3. June's perfect-oracle 0/11 was
plausibly measured under the same directive, so the historical "complexity = pure
capability wall" attribution is contaminated. CLI passthrough added (`91c25a6`); run i4
(114 min, official 0/6, MLflow `issue52-c4` latest) exercised the never-tested
replace-and-replan branch:

- **3801**: mechanism works end-to-end — complexity reject → replan fired (plans at
  s3/s5) → REPLACE brief rendered → the model produced its first-ever digit-family
  attempt (still O(n), correctly rejected). `complexity_repair_cap=2` then halted
  exploration after two shots — the cap was tuned for the preserve-directive era and
  should be relaxed (or made replan-aware) under the replace treatment.
- **3777**: revealed a **judge defect** — running before the complexity oracle, the
  judge hallucinated counterexamples on correct-but-slow brute forces ("wrong on input
  [1,1,1,1,1,1]… uses itertools"), its synthetic assertion failures suppressed the
  deterministic complexity verdict, and the brief classified as "assertion" so the
  REPLACE directive never rendered there. Fixed (`0452266`): judge now runs LAST, only
  on units passing every deterministic oracle, and `prompt_judge` is scoped to
  functional correctness only (slow-but-correct = correct). Default path bit-identical
  (judge is opt-in).
- Conversions: still 0/6 official — with the directive finally delivered (3801) the 4B
  reached the right algorithm *family* but not the algorithm. Capability remains the
  wall for 3777/3801, but now *honestly tested* for 3801 and untested-cleanly for 3777
  (its i4 signal was judge-corrupted; re-test post-`0452266`).

## 5. Recommended next arms

1. Re-run 3777 (and the slice) post-`0452266` with `--no-complexity-repair-preserve-logic`
   and a relaxed/replan-aware `complexity_repair_cap` — the only arm where the complexity
   treatment is now clean end-to-end.
2. Pre-registered **escalate** arm on this slice with the full fix stack (only historical
   3753 solve came from escalate; rich episodic prompts already render brief/history).
3. **Multi-seed** (≥3) before attributing any per-task delta to a treatment.
4. **Thinking-budget arm** (`thinking_budget>0`) targeting failure mode 4.
5. Keep `--model-judge` on post-fix (near-miss defense — would have caught i2's 3799;
   its i4 hallucination mode is closed by precedence + scope).
6. Generic complexity→technique hint table in the repair brief (task-agnostic, reviewable).

## Appendix: environment re-provision (VM restarts wipe /tmp)

- Checkpoint: `aws s3 cp s3://elixirtrials-949678234935-us-east-1-artifacts/mlflow/artifacts/45/fe72f9ddd69c4f7b8bd86b6b12372d47/artifacts/checkpoints/checkpoint_step48.pt /tmp/phase1/ckpt/c3_t07_lp2_lg1.pt` (sha256 `53e24af2…`, per `docs/publication/c21_prep.md` §4).
- Dataset: `curl -L https://huggingface.co/datasets/livecodebench/code_generation_lite/resolve/main/test6.jsonl -o /tmp/lcb/test6.jsonl` (134,303,240 bytes).
- Grader: `git clone --depth 1 https://github.com/LiveCodeBench/LiveCodeBench.git /tmp/LiveCodeBench && uv venv /tmp/lcbenv --python 3.12 && VIRTUAL_ENV=/tmp/lcbenv uv pip install numpy tqdm`; invoke with `PYTHONPATH=/workspaces/rune-gpu/src:/tmp/LiveCodeBench /tmp/lcbenv/bin/python tools/_lcb_grade.py --gens <file> --timeout 6`.
- Sessions/gens: `/tmp/c4/{i1,i2,i3}_{sessions,gens.jsonl,run.log}`; forensic drivers under `/tmp/c4/scratch/`.
