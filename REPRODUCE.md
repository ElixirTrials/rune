# Issue-52 experiment reproduction code

Scratch harness + analysis scripts for the issue-52 LCB/MBPP benchmarking. Kept OFF
the main/PR branch (REMOVE-BEFORE-MERGE) but preserved here so any experiment can be
reproduced. This is an orphan branch (no shared history with main).

## Engine code version
These tools were run against rune engine commit:

    293980782949d745fcf39a164e0ed45a772420bb
    "Engine fixes: spec-cap, cont-scaling, judge-quality snapshot, budget,
     skip-diagnose, salvage ship"  (branch issue52-bf16-body-contrastive)

Check out that commit of the main repo, then run these tools against it.

## Layout
- `tools/`    — in-repo scratch harness (bridges LCB v6 -> rune engine -> official grade)
  - `_lcb_run.py`    : generate LCB solutions with the rune runner -> {question_id, code_list}
  - `_lcb_grade.py`  : official LCB grader (run in the isolated lcbenv)
  - `_subset_probe.py`: A/B subset grader vs a baseline session set
  - `_oracle_*.py`, `_real_repair_*.py`, `_repair_trace.py`, `_perfect_oracle_probe.py`,
    `_verify_*.py`, `_why_prize_small.py` : oracle/repair-signal probes
- `analysis/` — /tmp/goal3 + /tmp/goal3/overnight analysis & driver scripts
  (`_analyze.py`, `_lcb_grade_perqid.py`, `_falsepass.py`, `_overlap.py`,
   `run_postfix_bench.sh`, etc.)

## Canonical full-49 run (current code, judge-OFF, escalate, budget 24)

    UV_NO_SYNC=1 MLFLOW_ALLOW_FILE_STORE=true uv run --extra gpu python tools/_lcb_run.py \
      --arm c3 --prompt-mode escalate --functional-only --seed 0 --max-iters 24 --no-grade \
      --out OUT.json --sessions OUT_sessions --experiment <name>

Then grade with the official harness in lcbenv:

    PYTHONPATH=src:/tmp/LiveCodeBench /tmp/lcbenv/bin/python tools/_lcb_grade.py \
      --gens OUT.json --timeout 6

## Data locations (not archived here — large result JSONs)
- LCB v6 problems: /tmp/lcb/test6.jsonl
- c3 checkpoint:   /tmp/phase1/ckpt/c3_t07_lp2_lg1.pt
- official grader env: /tmp/lcbenv ; LCB harness: /tmp/LiveCodeBench
- prior result dumps: /tmp/goal3/overnight/*.json
