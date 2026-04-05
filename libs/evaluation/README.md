# Evaluation

Adapter benchmarking, fitness scoring, and out-of-distribution testing.

## Key Functions

### Metrics (`metrics.py`)

| Function | Description |
|----------|-------------|
| `calculate_pass_at_k()` | Pass@k metric for code generation |
| `score_adapter_quality()` | Quality score for a single adapter |
| `evaluate_fitness()` | Composite evolutionary fitness score |
| `compare_adapters()` | Head-to-head adapter comparison |
| `test_generalization()` | Generalization across task types |
| `run_humaneval_subset()` | Run evaluation on HumanEval subset |
| `run_kill_switch_gate()` | Phase 1 kill-switch threshold check |

### OOD Benchmark (`ood_benchmark.py`)

| Function | Description |
|----------|-------------|
| `run_ood_benchmark()` | Out-of-distribution benchmark evaluation |
| `compute_generalization_delta()` | Delta between in-distribution and OOD performance |

Task definitions are in `data/ood_tasks.json`.

## Usage

```python
from evaluation import evaluate_fitness, run_ood_benchmark

fitness = evaluate_fitness(adapter_id="adapter-001", task_type="bug-fix")
ood_result = run_ood_benchmark(adapter_path="/adapters/adapter-001")
delta = compute_generalization_delta(in_dist_score=0.85, ood_score=0.72)
```

## Coding Benchmark Framework

Standardized coding benchmarks for evaluating base models and adapted models against established suites.

### Benchmarks

| Benchmark | Source | Tasks |
|-----------|--------|-------|
| HumanEval+ | EvalPlus | Function-level code generation |
| MBPP+ | EvalPlus | Python programming problems |
| BigCodeBench | BigCodeBench | Complex coding tasks |

### Execution Tiers

| Tier | Label | Approximate Time | Description |
|------|-------|-------------------|-------------|
| 1 | smoke | ~5 min | Small subset for quick validation |
| 2 | mini | ~30 min | Medium subset for development |
| 3 | full | ~2 hr | Full benchmark suites |

### Entry Points

| Script | Purpose |
|--------|---------|
| `scripts/eval/run_benchmarks.py` | End-to-end benchmark runner |
| `scripts/eval/generate_completions.py` | Generate model completions for evaluation |
| `scripts/eval/config.py` | Benchmark configuration and tier definitions |
| `scripts/eval/compare_results.py` | Compare results across runs |

Supports Transformers and vLLM backends. Default dev model: `google/gemma-2-2b-it`.

### Usage

```bash
uv run scripts/eval/run_benchmarks.py --tier 1
```
