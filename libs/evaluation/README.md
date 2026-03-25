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

## Math Olympiad Benchmark

Agentic tool-calling evaluation on [OlymMATH](https://huggingface.co/datasets/RUC-AIBOX/OlymMATH) (easy + hard splits). The model generates Python code, executes it in a sandbox, and iterates up to 15 times per problem.

### Setup

```bash
# 1. Build the qwenv environment (from repo root)
bash scripts/qwenv_setup.sh
source ~/.qwenv/bin/activate   # or the path printed by the script

# 2. Download datasets
python libs/evaluation/src/evaluation/download_data.py --datasets olym_math

# 3. Set your model path in the config
#    Edit: libs/evaluation/src/evaluation/configs/qwen3_5_olym_easy.yaml
#    Update model.model_id to your local model snapshot path

# 4. Run
python scripts/run_benchmark.py \
    --config libs/evaluation/src/evaluation/configs/qwen3_5_olym_easy.yaml
```

### Config

Key settings in `configs/qwen3_5_olym_easy.yaml`:

| Field | Value | Notes |
|---|---|---|
| `provider` | `vllm` | in-process vLLM |
| `tensor_parallel_size` | `2` | dual-GPU |
| `temperature` | `0.6` | |
| `max_new_tokens` | `32768` | |
| `use_tools` | `true` | agentic code loop |
| `max_iterations` | `15` | steps per problem |
| `n_samples` | `3` | majority vote |

### Results — Qwen3.5-9B

Evaluated with `n_samples=3` majority vote, `use_tools=true`, `math_prompt_v2` system prompt.

| Dataset | Correct / Total | Accuracy | Errors |
|---|---|---|---|
| OlymMATH Easy | 42 / 47 | **89.4%** | 0 |
| OlymMATH Hard | 23 / 41 | **56.1%** | 0 |

Full outpus found in results.json files: `benchmark_results/20260325_014054_57627943_math_prompt_v2`