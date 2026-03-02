# Evaluation

## Purpose
This component runs offline evaluation benchmarks against your agents.

## How to Add a Benchmark

1.  **Define Dataset**: Place evaluation datasets (JSONL) in `data/`.
2.  **Create Script**: Write a script in `src/evaluation/benchmarks/` that sends inputs to your agent and compares outputs against ground truth.
3.  **Metrics**: Use `shared` models or standard libraries (sklearn, rouge-score) for scoring.

## Running Evaluation
```bash
uv run python src/evaluation/run_benchmark.py --agent agent-a
```
