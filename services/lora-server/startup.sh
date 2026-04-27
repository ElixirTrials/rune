#!/usr/bin/env bash
set -euo pipefail

# Read settings from config.yaml, with environment variable overrides.
# Each setting can be overridden via the corresponding env var:
#   PIPELINE_PARALLEL_SIZE, TENSOR_PARALLEL_SIZE, QUANTIZATION,
#   MAX_LORAS, MAX_LORA_RANK, GPU_MEMORY_UTILIZATION

_yaml_val() {
    python -c "import yaml; cfg=yaml.safe_load(open('config.yaml')); print(cfg.get('$1', '$2'))" 2>/dev/null || echo "$2"
}

MODEL="${MODEL:-$(_yaml_val model "Qwen/Qwen3.5-9B")}"
PIPELINE_PARALLEL_SIZE="${PIPELINE_PARALLEL_SIZE:-$(_yaml_val pipeline_parallel_size 1)}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-$(_yaml_val tensor_parallel_size 1)}"
QUANTIZATION="${QUANTIZATION:-$(_yaml_val quantization awq)}"
MAX_LORAS="${MAX_LORAS:-$(_yaml_val max_loras 8)}"
MAX_LORA_RANK="${MAX_LORA_RANK:-$(_yaml_val max_lora_rank 64)}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-$(_yaml_val gpu_memory_utilization 0.80)}"
PORT="${PORT:-$(_yaml_val port 8000)}"

# Start health sidecar in background
python health.py &

# Start vLLM OpenAI-compatible API server
exec python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --pipeline-parallel-size "$PIPELINE_PARALLEL_SIZE" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --enable-lora \
    --quantization "$QUANTIZATION" \
    --max-loras "$MAX_LORAS" \
    --max-lora-rank "$MAX_LORA_RANK" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --port "$PORT"
