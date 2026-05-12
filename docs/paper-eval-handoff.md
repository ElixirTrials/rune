# Paper Evaluation Handoff

**Date:** 2026-05-07  
**Branch:** `fix/diff-loss-per-turn-alignment`  
**Goal:** Run Table 2 (conditions i-v), Gate 2, Gate 3 evaluations and write results into `instructions/paper.tex`

---

## Problems Encountered and Resolutions

### 1. 0.00% Pass@1 on HumanEval and MBPP (conditions i, iii)

**Root cause:** `VLLMProvider.generate()` uses the OpenAI **chat completions API** (`/v1/chat/completions`), which wraps prompts in a chat template. `Qwen/Qwen3.5-9B` is a **base pretrained model** (not instruction-tuned), so it does not understand chat formatting. The model produced garbage or conversational responses instead of code continuations.

**Resolution:** Added `complete_text()` method to both `InferenceProvider` (ABC with fallback) and `VLLMProvider` (uses `/v1/completions` — raw text continuation). Updated `runner.py:_generate_completion()` to call `complete_text()` instead of `generate()`.

**Files changed:**
- `libs/inference/src/inference/provider.py` — added `complete_text()` default method
- `libs/inference/src/inference/vllm_provider.py` — added `complete_text()` override using `self._client.completions.create()`
- `libs/evaluation/src/evaluation/benchmarks/runner.py` — switched from `generate()` to `complete_text()`

**Verification:** Last run (20260507-114214) showed 201 successful `/v1/completions` 200 OK responses in vLLM log, zero `/v1/chat` calls. The fix is working but the run was killed before results were written.

### 2. Condition (iii) 404 on `/v1/load_lora_adapter`

**Root cause:** vLLM 0.20.1 gates the LoRA hot-loading endpoints behind `VLLM_ALLOW_RUNTIME_LORA_UPDATING` env var. Without it, the `/v1/load_lora_adapter` and `/v1/unload_lora_adapter` routes are never registered.

**Evidence:** `vllm/entrypoints/serve/lora/api_router.py:27` — `if not envs.VLLM_ALLOW_RUNTIME_LORA_UPDATING: return`

**Resolution:** Added `VLLM_ALLOW_RUNTIME_LORA_UPDATING=1` prefix to both vLLM launch commands in `collect_paper_data.sh` (initial launch and phase 2 restart).

**Files changed:**
- `scripts/paper/collect_paper_data.sh` — two vLLM launch blocks

**Verification:** Last run confirmed: `"LoRA dynamic loading & unloading is enabled in the API server"` appeared in vLLM log.

### 3. Condition (iv) TTT OOM on state_dict clone

**Root cause:** `run_condition_ttt()` clones the full model state_dict to restore weights between problems. Qwen 3.5 9B fp16 uses ~21.7 GB on a 22 GB GPU, leaving no room for a second copy.

**Resolution:** Changed `v.clone()` to `v.cpu().clone()` — state_dict backup lives in system RAM, not GPU memory.

**Files changed:**
- `scripts/paper/run_all_conditions.py:227` — `original_sd = {k: v.cpu().clone() for ...}`

**Status:** Not yet verified in a successful run.

### 4. Condition (ii) RAG — missing `sentence_transformers`

**Root cause:** RAG condition requires `sentence_transformers` for embedding queries, which is not installed.

**Resolution:** Wrapped `run_condition_rag()` call in `try/except ImportError` so it gracefully skips.

**Files changed:**
- `scripts/paper/run_all_conditions.py:433` — ImportError handler

**Status:** Working (logs show "SKIPPED: missing dependency for RAG"). To actually run condition (ii), install `sentence-transformers` package.

### 5. LiveCodeBench loading failure

**Root cause:** Neither the fixture (`tests/fixtures/livecodebench_mini.parquet`) nor the HF dataset loads correctly (legacy loading script).

**Resolution:** Changed default benchmark list in `collect_paper_data.sh` from 6 benchmarks to `humaneval mbpp`. Override with `BENCHMARKS="humaneval mbpp livecodebench" bash scripts/paper/collect_paper_data.sh` if fixture is added later.

**Files changed:**
- `scripts/paper/collect_paper_data.sh:16`

### 6. `set -euo pipefail` killing bash script on Python errors

**Root cause:** Any non-zero exit from `uv run python ...` piped to `tee` killed the entire script.

**Resolution:** Wrapped evaluation commands in `set +e` / `set -e` blocks.

**Files changed:**
- `scripts/paper/collect_paper_data.sh` — around conditions i-iii and condition iv blocks

### 7. Devpod instability killing long-running evaluations

**Root cause:** The devpod (remote development environment) goes down intermittently during multi-hour eval runs, killing all processes.

**Resolution:** Made the pipeline idempotent — each output file is checked before running. Intermediate results (`table2_vllm.json`, `table2_ttt.json`) are written before merging. If the run is interrupted, re-running skips completed steps.

**Status:** Partially effective. The current issue is that `run_all_conditions.py` writes its output only after ALL conditions complete, so if the process dies mid-condition, nothing is written. See outstanding issues below.

---

## Smoke Test Added

Added an inline smoke test to `collect_paper_data.sh` that runs immediately after vLLM starts:

```bash
curl -sf http://localhost:${VLLM_PORT}/v1/completions \
    -d '{"model": "...", "prompt": "def add(a, b):\n    return", "max_tokens": 16}'
```

Checks that: (a) the completions endpoint responds, and (b) the output is code, not conversational text. Catches the chat-vs-completions issue in seconds instead of after a full 4400s benchmark run.

---

## Current State of Files

### Modified (tracked, unstaged)

| File | What changed |
|------|-------------|
| `libs/inference/src/inference/provider.py` | Added `complete_text()` ABC method |
| `libs/inference/src/inference/vllm_provider.py` | Added `complete_text()` using `/v1/completions` |
| `libs/evaluation/src/evaluation/benchmarks/runner.py` | Uses `complete_text()`, retry logic, system prompt |
| `libs/evaluation/src/evaluation/benchmarks/adapter_stack.py` | Type aliases for callables |
| `libs/evaluation/src/evaluation/benchmarks/apps.py` | HF loading fix |
| `libs/evaluation/src/evaluation/benchmarks/codecontests.py` | Minor fix |
| `scripts/paper/run_all_conditions.py` | Full 5-condition orchestrator, CPU state_dict, ImportError handling |
| `scripts/paper/run_gate2.py` | Gate 2 runner |
| `scripts/paper/run_gate3.py` | Gate 3 runner |
| `scripts/paper/run_rag_baseline.py` | RAG baseline runner |
| `scripts/paper/run_ttt_baseline.py` | TTT baseline runner |
| `libs/model-training/src/model_training/d2l_config.py` | Hypernetwork config |
| `libs/model-training/src/model_training/sakana_d2l.py` | Hypernetwork changes |
| `pyproject.toml` | Dependency additions |

### New (untracked)

| File | Purpose |
|------|---------|
| `scripts/paper/collect_paper_data.sh` | Master orchestrator (vLLM lifecycle, phases, idempotency) |
| `scripts/train_hypernet_hpo.py` | HyperLoRA training script |
| `scripts/abridge_extreme_records.py` | Data preprocessing |

### Result files (empty — no successful run yet)

- `evaluation_results/paper/metadata.json` — exists
- `evaluation_results/paper/table2_vllm.json` — does not exist
- `evaluation_results/paper/table2_ttt.json` — does not exist
- `evaluation_results/paper/table2_phase1.json` — does not exist

---

## Outstanding Problems

### Critical (blocking paper results)

1. **No successful evaluation run has completed.** Three attempts were all killed (devpod going down, OOM, or errors). The fixes are in place but unverified end-to-end.

2. **Per-problem result streaming.** `run_all_conditions.py` only writes output after all conditions finish. If the process dies during condition (iii) after condition (i) completed successfully, condition (i) results are lost. Fix: write per-condition results incrementally, or at minimum flush after each condition completes.

3. **MBPP scoring may need review.** Even with the completions API fix, MBPP's scoring logic should be verified — the prompt format and expected output format may differ from HumanEval.

### Important (needed for full Table 2)

4. **Condition (ii) RAG requires `sentence-transformers`.** Install via `uv add sentence-transformers` or accept condition (ii) as "N/A" in the paper.

5. **Condition (iv) TTT CPU clone not verified.** The `.cpu().clone()` fix should resolve OOM but hasn't been tested in a successful run. If system RAM is also tight (~18GB for the clone), this could still fail.

6. **Condition (v) Rune requires a trained hypernetwork checkpoint.** `scripts/train_hypernet_hpo.py` exists but hasn't been run. Needs `HYPERNET_CHECKPOINT` path. This is Phase 2.

7. **LiveCodeBench fixture missing.** If the paper references LiveCodeBench results, someone needs to either download/create `tests/fixtures/livecodebench_mini.parquet` or change the paper to use only HumanEval + MBPP.

### Paper-specific

8. **`instructions/paper.tex` Tables 2 and 3 are still placeholder dashes.** Once a successful run completes, results need to be written into:
   - Table 2 (line 179-183): Pass@1 per condition
   - Table 3 (line 200-207): Gate 2 multi-benchmark results
   - Gate verdicts throughout section 4

9. **Paper references 6 benchmarks for Gate 2 (line 190)** but we trimmed to 2 for speed. Either update the paper to match the trimmed set, or run the full 6 (requires fixing LiveCodeBench, BigCodeBench, DS-1000 loading).

---

## How to Re-Run

```bash
# Clean stale results (if any)
rm -f evaluation_results/paper/table2_*.json

# Run phase 1 (conditions i-iv, ~2-3 hours on humaneval+mbpp)
bash scripts/paper/collect_paper_data.sh 2>&1 | tee .tmp/paper_logs/phase1_full_$(date +%Y%m%d-%H%M%S).log

# Override benchmarks if needed
BENCHMARKS="humaneval mbpp" bash scripts/paper/collect_paper_data.sh

# Phase 2 (requires trained hypernetwork)
HYPERNET_CHECKPOINT=path/to/checkpoint.pt bash scripts/paper/collect_paper_data.sh --phase 2
```

### Key Environment Requirements

- vLLM 0.20.1 with `VLLM_ALLOW_RUNTIME_LORA_UPDATING=1` (set automatically by script)
- GPU with ~22GB VRAM (A10G or better)
- HPO adapter at `hpo_artifacts/best_diffloss_v1/` (auto-fetched from S3 if missing)
- MLflow tracking server at `http://localhost:5000` (optional, for logging)

---

## Recommendation for Next Session

1. Run the script on a **stable machine** (not a devpod that may go down). A bare EC2 instance or a tmux/screen session that survives disconnects.
2. If devpod is the only option, use `nohup` or `screen`:
   ```bash
   screen -S paper
   bash scripts/paper/collect_paper_data.sh 2>&1 | tee .tmp/paper_logs/phase1.log
   # Ctrl-A D to detach
   ```
3. Consider adding per-condition incremental writes to `run_all_conditions.py` so partial results survive crashes.
