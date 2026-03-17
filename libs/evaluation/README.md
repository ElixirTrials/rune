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
