#!/usr/bin/env bash
set -euo pipefail

# Read model name from config.yaml, fall back to default
MODEL=$(python -c "import yaml; print(yaml.safe_load(open('config.yaml'))['model'])" 2>/dev/null || echo "Qwen/Qwen2.5-Coder-7B-Instruct")

# Start health sidecar in background (port 8001)
python health.py &

# Start vLLM OpenAI-compatible API server
exec python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --pipeline-parallel-size 2 \
    --tensor-parallel-size 1 \
    --enable-lora \
    --quantization awq \
    --max-loras 8 \
    --port 8000
