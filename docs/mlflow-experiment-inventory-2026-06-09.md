# MLflow Experiment Inventory — system of record for the paper

**Generated:** 2026-06-09 · **Tracking server:** `http://localhost:5000` (S3-backed store; the local `mlflow.db` holds only `Default` — always query the server, never the file).
**Why this doc exists:** the committed `docs/issue52-*.md` set captures the **June (issue #52 / PR #55)** track but is blind to the **May "paper" track** (`paper-table2`, `paper-gate2`, the baseline/HPO experiments the paper §4 is built around). Everything was logged to MLflow; this is the index. Use it together with [`paper-evidence-map-2026-06-09.md`](paper-evidence-map-2026-06-09.md).

> **Status (2026-07-08):** the campaign of record is now [`publication/handoff_realized_gates.md`](publication/handoff_realized_gates.md). Two supersessions since the 06-22 note below: (1) the tracking DB was restored from a snapshot predating experiments 78–86, so the param/metric rows for `issue52-repobench-clamp`, `issue52-repobench-template-hpo`, and `issue52-humanevalplus`-era runs in that range are lost — S3 artifacts survive and are hashed in [`publication/hashes.txt`](publication/hashes.txt); `issue52-repobench-clamp` is re-logged first-class as experiment id 79 (handoff §A-REPRO). (2) The §4 row "Phase-1 held-out recall +0.105" is superseded by the C2.1 fresh-pool re-estimate: mean Δlp_matched **+0.147**, n=120, sign test p=5.5e-14, CI [+0.109, +0.191] — exp 45 run `1769a1f8dedd43a789041536294c9825` ([`publication/c21_prep.md`](publication/c21_prep.md), handoff §A-OBJ). The registry below remains a correct snapshot of 2026-06-09.

> **Superseded in part (2026-06-22).** This 06-09 inventory predates the durable long-context / powered-eval work. Three MLflow experiments are now authoritative and are **not** listed in §1 below: `issue52-repobench-clamp`, `issue52-repobench-template-hpo`, `issue52-humanevalplus`. Also, the §4 LCB-49 "rune ties base 9/49" line is the old framing — the final clean run is **rune 12/49 > base 9/49** (strict superset; pooled N=63: c3 16/63 > base 12/63) — an uplift, not a significant win: McNemar two-sided p=0.125, underpowered, and the pre-registered significance endpoint was never reached, so the paper treats functional-49 as a tie/underpowered (see [`issue52-lcb-durable-findings-2026-06-19.md`](issue52-lcb-durable-findings-2026-06-19.md) §5). See [`issue52-results-longcontext-2026-06-22.md`](issue52-results-longcontext-2026-06-22.md), [`issue52-lcb-durable-findings-2026-06-19.md`](issue52-lcb-durable-findings-2026-06-19.md), and the 06-22 RCA docs for the full arc.

> **Reconciliation note.** Two distinct experimental tracks exist and **must not be merged**:
> - **May track** — `Qwen/Qwen3.5-9B` + `danielcherubini/Qwen3.5-DeltaCoder-9B` warm-start. This is the paper's *production* model line and the structure behind Table 2 (conditions i–v) and Gate 2.
> - **June track** — `Qwen/Qwen3-4B-Instruct-2507` + `body_recall_guarded` checkpoint (c3). This is the issue #52 recall-objective + LCB-49 work documented in `issue52-experimentation-log.md`.
> Different base models, different objectives, different benchmarks. The paper currently blends terminology from both plus a *third*, **Gemma 2 2B**, dev model that has **no experiment in MLflow at all** (see §3).

---

## 1. Full experiment registry (74 experiments)

Run counts as of 2026-06-09. "Track" = M(ay-paper) / J(une-issue52) / I(nfra-early).

| exp | name | runs | track | content / status |
|----:|------|-----:|:-----:|------------------|
| 0 | Default | 0 | — | empty |
| 3 | rune-qlora-hpo | 44 | I | early QLoRA HPO; no `pass@1` metric logged |
| 4 | rune-qlora-hpo-studies | 4 | I | study-summary runs |
| 5 | rune-qlora | 2 | I | QLoRA smoke |
| 6–17 | rune-difflearn-* / plainsft / h1 / ours-* / validate / eval-* | 1 ea | I | diff-aware loss dev + ablation singles (cold, overfit5, alpha32, heldout, patch) |
| 18 | codereview-single-turn-v1 | 18 | M | single-turn code-review eval |
| **19** | **paper-table2** | **31** | **M** | **Table 2 (conditions i–v). Mostly empty/failed; only (v) MBPP populated — see §2.** |
| 20 | hypernet-hpo | 29 | M | hypernet distillation HPO (Qwen3.5-9B); KL/CE loss only, no pass@1 |
| 21 | rune-encoder-pretrain-test | 38 | M | perceiver encoder pretrain smoke |
| 22 | rune-encoder-pretrain | 31 | M | perceiver encoder pretrain |
| 23–26 | kd-a09-t10 / kd-a10-t10 / kd-a099-t20 / hypernet-full-t10 | 1–5 | M | knowledge-distillation alpha/temp sweeps |
| **27** | **paper-gate2** | **7** | **M** | **Gate 2 (6-benchmark robustness). No metrics logged on any run — empty.** |
| 28 | benchmark-hpo-mbpp | 108 | M | engine pipeline HPO on MBPP; best `n_passed`=5 (small smoke pools); 945 metric rows |
| 29–37 | test-trace* / test-optuna* / single-mbpp / test-* / verify-fix / test-connectivity | 1–15 | I | plumbing/trace tests |
| 38 | rune-bench | 7 | M | engine bench smoke |
| 39 | adapter-probe | 2 | M | adapter logit probe |
| **40** | **adapter-scaling-hpo** | **31** | **M** | scaling sweep, **range 1.5×–9.9×**, best `objective`=0.387 (trial-20 @2.85×). *Not* the 0.16× source. |
| **41** | **rune-bench-hpo** | **64** | **M/J** | engine HPO; best `tuning_pass_at_1`=**0.588** @ `adapter_scaling`=**0.627**. Authoritative scaling optimum. |
| 42 | continuation-scaling-hpo | 27 | M | `cont_multiplier` sweep (→1.53 default); no pass@1 |
| 43 | issue52-body-crossover | 3 | J | Pilot 1 `body_derangement` |
| 44 | issue52-body-recall | 2 | J | Pilot 2 `body_recall_guarded` (train-on-test) |
| **45** | **issue52-phase1** | **4** | **J** | **c1–c4 grid; c3 best. Phase-1 held-out recall — the validated result.** |
| 46 | corpus-registry | 1 | J | `external_codereview` dataset lineage (`ea4f3c43`, 4 inputs) |
| 47 | issue52-goal2-scaling | 2 | J | corpus 40→80→160 scaling (n80, n160) |
| 48–62 | issue52-goal3-* | 1–6 ea | J | runner substrate, spec-in-adapter, HPO, oracle/judge, truefloor, hard-memory (see log §3.5) |
| 63–70 | issue52-goal3-episodic-* | 1 ea | J | episodic-prompt hard-task probes (smoke, 1task, hard..hard6) |
| 71 | issue52-goal3-multiturn | 19 | J | multiturn episodic driver |
| 72 | issue52-goal3-lcb | 15 | J | LCB pipeline bring-up |
| 73 | issue52-lcb-fix-rerun | 1 | J | LCB **6-task** smoke, c3@0.627, pass@1 1/6; oracle-fired diagnostics |
| 74 | issue52-lcb-fix-rerun-v2 | 1 | J | LCB **6-task** smoke rerun, identical 1/6 |

---

## 2. What `paper-table2` (exp 19) actually contains

31 runs, all `Qwen/Qwen3.5-9B` + DeltaCoder warm-start, 2026-05-06 → 05-13. The **only** non-trivial `pass@1` ever recorded:

| condition | benchmark | best `pass_at_1` logged | reading |
|-----------|-----------|------------------------:|--------|
| (i) base | mbpp | 0.008 | near-zero — harness/parse failure, not a real base score |
| (iii) PEFT QLoRA | mbpp | 0.008 | same near-zero failure |
| (iv) TTT-E2E | mbpp | 0.000 | did not produce passing output |
| **(v) Rune** | **mbpp** | **0.5136** (`v_iter_mbpp_pass_at_1`, 05-13 21:15) | the one substantive cell; iterative running pass@1 |

HumanEval / APPS / BigCodeBench / DS-1000 / LiveCodeBench: configured in `benchmarks` param but **no metric rows** — those arms never completed. **Conclusion: Table 2 of the paper is not populated by valid measured data.** Condition (v)'s 0.514 on MBPP is the only defensible number, and it has no comparable (i)/(ii)/(iii)/(iv) row to sit beside (the baselines logged 0.000–0.008, i.e. broken runs). The paper's Table 2 placeholders are therefore *honest* — there is no hidden filled version.

`paper-gate2` (exp 27): 7 runs, **zero metrics** on any. Gate 2 has not been measured.

---

## 3. The `0.16×` provenance problem (highest-priority author flag)

The paper's most-cited quantitative claim:

> "A 200-trial Optuna TPE search on Gemma 2 2B places optimal adapter scaling at 0.16×, roughly 280× below the 45.25× default for document-conditioned Doc-to-LoRA."

**Where I searched (this environment only):** all 74 MLflow experiments' params/metrics *and* the artifacts of the HPO/scaling/control runs (exps 3, 4, 20, 28, 40, 41); the local `optuna_bench_hpo.db`; every PR comment (open + closed); and a text/file search across both working trees (`/workspaces/content` and `/workspaces/rune-gpu` are the **same** tree). **Findings — scoped to what is reachable here:**

| paper element | status *in reachable artifacts* |
|---------------|--------------------|
| Gemma 2 2B experiment | **not present here** — no MLflow run, study, or config carries `gemma`; this is the **Qwen GPU fork**, and MLflow was only wired in at PR #28 (2026-04-22), so a Gemma *dev* sweep predating that would not be in MLflow at all |
| 200-trial Optuna search | **not present here** — largest scaling study reachable is `adapter-scaling-hpo` (31 trials) and `rune-bench-hpo` (64 runs across sessions; 16 in the local sqlite) |
| optimal scaling 0.16× | **not reproduced here** — `rune-bench-hpo` best = **0.627**; `adapter-scaling-hpo` best ≈ 2.85× (different objective, range 1.5–9.9×). The local 16-trial study's *minimum sampled* scaling was 0.146, but that trial scored **0.294** (near-worst), not the optimum. No "released HPO log" artifact found on the HPO runs. |
| 45.25× Doc-to-LoRA default | **correct** — `lora_alpha=45.2548` is the native scale of the Sakana `gemma_demo` / Qwen warm-start checkpoint (`test_adapter_contract.py`). This is the probe/contract scale, *not* a generation-scaling optimum. |

**RESOLVED (2026-06-09): author chose to re-anchor `paper_v9.tex` to the in-hand Qwen evidence.** I could not locate the Gemma 2 2B 200-trial study in any artifact reachable here (absence of evidence, not proof it never existed — it may live in an external/recycled store or predate the PR-#28 MLflow wiring). Rather than cite an unconfirmed store, the paper now reports the Qwen line: *generation-time* adapter scaling optimizes to **0.627×** on Qwen3-4B-Instruct easy-MBPP (`rune-bench-hpo`, 16 trials).

> **Unit semantics — do not repeat the original error.** `adapter_scaling` is a **multiplier on the checkpoint's native scale**: `scale_lora_b(state_dict, scaling)` is applied on top of PEFT's un-divided `lora_alpha=45.25` (`adapter.py:66`, `wrapper.py`; total = `lora_alpha × adapter_scaling`, so `1.0×` realises 45.25). Therefore **`0.627×` means 0.627× the document-conditioned default** (a mild ~0.6× attenuation), **not** "72× below 45.25." The paper's original "`0.16×` → 280× below 45.25×" was itself unit-confused (it divided 45.25 by the multiplier 0.16). The re-anchored paper drops the "Nx below" framing and reports `0.627×` as a sub-unity multiplier. This also means the structural conjecture is **weaker** than the paper implied — a 0.6× attenuation, not orders of magnitude, and outside the paper's old 0.1–0.3× prediction band (B.8 softened to "sub-unity" accordingly).

> **Compounding caveat (PR #45).** Early bench-HPO runs tuned `adapter_scaling` but the value **never reached the model** (`step_node` ignored it) until fixed mid-PR-#45. Any scaling-optimum run logged *before* that fix is invalid. Confirm the `0.627` study post-dates the fix (it does — `rune-bench-hpo` exp 41 / June) before citing it.

---

## 4. Authoritative numbers by claim (cite these, with their home)

| quantity | value | home of record | track |
|----------|-------|----------------|-------|
| Phase-1 held-out recall (Δlp_matched) | **+0.105**, CI [+0.033, +0.182] | exp 45 `issue52-phase1` run `fe72f9ddd69c` (c3); `_specificity_probe` post-hoc | J |
| Phase-1 absent pass@1 | 8/24 (c3) vs 0/24 (scale0) vs 3/24 (warm) | `issue52-experimentation-log.md` §3.2 (E-phase1) + scratchpad `21:37` | J |
| Spec-in-adapter scaling optimum | flat plateau `adapter_scaling`≈**0.6–0.9×** (0.627/0.673/0.685/0.815/0.921 all → tuning 0.588); 0.627 = lower edge; val 0.571 | exp 41 `rune-bench-hpo` | J |
| spec-absent floor vs adapter | 0.333 → 0.583 (+0.25) | exp 61 `truefloor`; `goal3-conclusions` | J |
| **LCB functional-49, base single-shot** | **9/49 (18.4%)** | official LCB harness via `tools/_lcb_grade.py` (PR #55 comment 2026-06-09 14:33Z) | J |
| **LCB functional-49, rune (de-overfit)** | **9/49 (18.4%)** — ties base | same; **not** in MLflow (exps 73/74 are 6-task smokes only) | J |
| Table 2 condition (v) MBPP | 0.5136 | exp 19 `paper-table2` 05-13 21:15 | M |
| Corpus scaling m-zero | 0.635→0.649→0.671 (N=40/80/160) | exp 47 `issue52-goal2-scaling` | J |

> **LCB-49 durability gap.** The headline benchmark number (base 9/49, rune 9/49) lives in a **PR comment + the official-harness JSON**, *not* an MLflow run — the MLflow LCB experiments (72/73/74) are pipeline bring-up and 6-task smokes. **Action:** log the full 49-task official-harness grade (both arms) as an MLflow run with `tools/_lcb_grade.py` output as an artifact, so the paper's eventual functional-subset number has a durable, versioned home.

---

## 5. How to query (reproducibility)

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
uv run python -c "
from mlflow.tracking import MlflowClient
c = MlflowClient()
for e in sorted(c.search_experiments(), key=lambda x:int(x.experiment_id)):
    print(e.experiment_id, e.name, len(c.search_runs([e.experiment_id], max_results=2000)))
"
```

Per-run metrics/params: `c.search_runs([exp_id])` → `run.data.metrics`, `run.data.params`, `run.data.tags['mlflow.runName']`.

---

*Maintainer note: regenerate run counts before citing externally; the server is mutable. Pair every paper number with its (exp, run, track) triple from §4.*
