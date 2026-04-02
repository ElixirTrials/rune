# Coding Benchmark Evaluation Framework

## Context

Rune's hypernetwork encodes coding trajectories into LoRA adapters for episodic weight-space memory. We're finetuning a LoRA on Qwen 3.5 9B using our trajectory training strategy and need to benchmark how well the augmented model performs on standard coding tasks compared to (a) the base model without the adapter and (b) commercially available LLMs.

The evaluation framework must produce leaderboard-comparable numbers using established benchmarks and support iterative development with quick feedback loops before committing to full benchmark runs.

## Base Model

- **Production target:** Qwen 3.5 9B (fits in float16 on L4 GPU with ~18GB, leaving headroom for LoRA)
- **Local dev:** Gemma 2 2B IT (fits on M4 Pro Mac, used for pipeline validation)
- LoRA adapter produced by Rune's trajectory training pipeline (PEFT-compatible, safetensors format)
- Model and adapter are pluggable via CLI flags — any HuggingFace model + PEFT adapter combo works

## Benchmark Suite

### Primary Benchmarks

| Benchmark | Problems | What it Tests | Metric | Framework |
|-----------|----------|---------------|--------|-----------|
| HumanEval+ | 164 | Function generation from docstrings | pass@1, pass@10 | EvalPlus |
| MBPP+ | 974 | Function generation from NL descriptions | pass@1, pass@10 | EvalPlus |
| BigCodeBench-Complete | 1,140 | Complex multi-library tasks from docstrings | pass@1 | BigCodeBench |
| BigCodeBench-Instruct | 1,140 | Same tasks from natural language instructions | pass@1 | BigCodeBench |

### Comparison Baselines

- **Qwen 3.5 9B base** (own run — the primary "LoRA lift" comparison)
- **GPT-4o, Claude Sonnet** (commercial reference points, published scores)
- **DeepSeek-Coder-V2** (strong open-source competitor, published scores)

## Tiered Execution Strategy

### Tier 1: Quick Smoke Test (~5 min)
- HumanEval+ subset: 20 problems (from `libs/evaluation/data/humaneval_subset.json`)
- pass@1 only (greedy, temperature=0.0)
- Both base and LoRA runs
- **Purpose:** Validate pipeline end-to-end, get rough signal

### Tier 2: Mini Benchmark (~30 min)
- HumanEval+ full: 164 problems, pass@1
- MBPP+ sample: ~100 problems (seed-controlled random subset), pass@1
- Both base and LoRA runs
- **Purpose:** Reliable signal on LoRA lift, decide whether to proceed to full run

### Tier 3: Full Benchmark (hours)
- HumanEval+: 164 problems, pass@1 + pass@10
- MBPP+: 974 problems, pass@1 + pass@10
- BigCodeBench-Complete: 1,140 problems, pass@1
- BigCodeBench-Instruct: 1,140 problems, pass@1
- Both base and LoRA runs
- **Purpose:** Leaderboard-ready numbers, publication-quality results

Graduate to the next tier only when the current one looks promising.

## Architecture

### Three-layer design

```
[Completion Generator] → [Benchmark Runners] → [Results Comparator]
```

### Layer 1: Completion Generator

A script (`scripts/eval/generate_completions.py`) that:

1. Loads benchmark prompts from the appropriate framework
2. Calls vLLM's OpenAI-compatible `/v1/completions` endpoint
3. Writes completions to `samples.jsonl` in each framework's expected format:
   - **EvalPlus format:** `{"task_id": "HumanEval/0", "completion": "..."}`
   - **BigCodeBench format:** `{"task_id": "BigCodeBench/0", "solution": "..."}`

**Two inference backends:**

**vLLM** (cloud GPU — fast batch inference):
```bash
vllm serve Qwen/Qwen3.5-9B --max-model-len 4096
vllm serve Qwen/Qwen3.5-9B --enable-lora --lora-modules rune-adapter=<path> --max-model-len 4096
```

**Transformers** (local Mac — no vLLM needed):
```python
# Loads model + optional PEFT adapter directly via HuggingFace transformers
model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b-it")
model = PeftModel.from_pretrained(model, adapter_path)  # optional
```

**Generation parameters:**

| Setting | pass@1 | pass@10 |
|---------|--------|---------|
| n (samples per task) | 1 | 10 |
| temperature | 0.0 (greedy) | 0.8 |
| top_p | 1.0 | 0.95 |
| max_tokens | 1024 | 1024 |

These match EvalPlus and BigCodeBench leaderboard conventions.

### Layer 2: Benchmark Runners

Thin wrappers (`scripts/eval/run_benchmarks.py`) that invoke each framework's evaluation:

**EvalPlus** (HumanEval+, MBPP+):
```bash
evalplus.evaluate --dataset humaneval --samples samples.jsonl
evalplus.evaluate --dataset mbpp --samples samples.jsonl
```
Runs completions in sandboxed subprocesses, outputs pass@k scores.

**BigCodeBench:**
```bash
bigcodebench.evaluate --split complete --samples samples.jsonl
bigcodebench.evaluate --split instruct --samples samples.jsonl
```
Uses Docker-based sandboxing for library-dependent tasks.

### Layer 3: Results Comparator

A script (`scripts/eval/compare_results.py`) that:

1. Loads evaluation results from both base and LoRA runs
2. Computes aggregate metrics: pass@1 delta, pass@10 delta
3. Per-task delta analysis: fail→pass flips, pass→fail regressions
4. Leaderboard positioning using published scores for commercial models
5. Generates markdown report + JSON results file

Output stored in `evaluation_results/<timestamp>/`.

## File Structure

```
scripts/eval/
  generate_completions.py   # Completion generator (vLLM client)
  run_benchmarks.py          # Orchestrator: generate + evaluate + compare
  compare_results.py         # Results analysis and reporting
  config.py                  # Benchmark configs, generation params, model paths

evaluation_results/           # Output directory (gitignored)
  <timestamp>/
    base/                    # Base model completions + scores
    lora/                    # LoRA model completions + scores
    report.md                # Comparison report
    results.json             # Machine-readable results
```

## Dependencies

```
evalplus        # HumanEval+, MBPP+ evaluation
bigcodebench    # BigCodeBench evaluation
vllm            # Model serving with LoRA support
openai          # vLLM client (OpenAI-compatible API)
```

All invoked via `uv run`. Added as optional eval dependencies in `pyproject.toml`.

## Verification

1. **Tier 1 smoke test passes:** Generate completions for 20 HumanEval problems, score them, see non-zero pass rate for both base and LoRA
2. **LoRA delta is measurable:** Tier 2 shows a statistically meaningful difference between base and LoRA scores
3. **Results match leaderboard format:** EvalPlus output can be submitted to the EvalPlus leaderboard
4. **Report is useful:** Markdown report clearly shows base vs LoRA vs commercial model positioning
