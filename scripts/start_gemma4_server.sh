#!/usr/bin/env bash
# start_gemma4_server.sh — Serve a Gemma 4 model via vLLM with tool calling enabled.
#
# Uses the .gemma4env created by scripts/setup_gemma4.sh.
# Key differences from start_vllm_server.sh (Qwen3):
#   --tool-call-parser gemma4   (NOT hermes)
#   --limit-mm-per-prompt image=0,audio=0   (skip multimodal profiling → more KV cache)
#   NO --reasoning-parser       (known conflict with tool-call-parser in vLLM ≤0.19;
#                                see gemma4_vllm_math_benchmark.md §11)
#   --default-chat-template-kwargs '{"enable_thinking": false}'
#
# Usage:
#   bash scripts/start_gemma4_server.sh                              # E4B default
#   bash scripts/start_gemma4_server.sh google/gemma-4-E4B-it        # explicit HF id
#   bash scripts/start_gemma4_server.sh ~/models/gemma-4-E4B-it      # local snapshot
#   bash scripts/start_gemma4_server.sh google/gemma-4-31B-it --tp 2
#   bash scripts/start_gemma4_server.sh --port 8001 --eager
#   bash scripts/start_gemma4_server.sh --max-len 32768 --fp8-kv
#
# Run the benchmark after the server is up:
#   uv run python scripts/run_benchmark.py \
#       --config libs/evaluation/src/evaluation/configs/gemma4_olym_hard.yaml
#
# Health check:
#   curl -s http://localhost:8000/health
#   curl -s http://localhost:8000/v1/models | python3 -m json.tool

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$REPO_ROOT/.gemma4env"

# ── helpers ────────────────────────────────────────────────────────────────────
info()  { echo -e "\033[36m[serve]\033[0m $*"; }
ok()    { echo -e "\033[32m[  ok ]\033[0m $*"; }
warn()  { echo -e "\033[33m[ warn]\033[0m $*"; }
die()   { echo -e "\033[31m[error]\033[0m $*" >&2; exit 1; }

# ── CUDA driver preflight ─────────────────────────────────────────────────────
# vLLM 0.19+ uses cuMemcpyBatchAsync which requires CUDA driver >= 12.9
# (NVIDIA driver >= 565.57.01).  Check upfront so the error is actionable.
_driver_cuda_minor() {
    nvidia-smi 2>/dev/null \
        | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+" \
        | awk -F. '{print $1 * 10 + $2}'
}
DRIVER_COMBINED=$(_driver_cuda_minor)
MIN_COMBINED=129  # 12.9
if [[ -n "$DRIVER_COMBINED" ]] && (( DRIVER_COMBINED < MIN_COMBINED )); then
    DRIVER_VER=$(nvidia-smi 2>/dev/null | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+" | head -1)
    die "CUDA driver too old: your driver supports CUDA ${DRIVER_VER}, but vLLM 0.19+ requires CUDA 12.9+.
       Upgrade the host NVIDIA driver to >= 565.57.01 (CUDA 12.9 support) and retry.
       On RunPod/Lambda/vast.ai: select a PyTorch 2.6+ / CUDA 12.9 template."
fi

# ── check venv ────────────────────────────────────────────────────────────────
[[ -d "$VENV_DIR" ]] || die ".gemma4env not found at $VENV_DIR — run scripts/setup_gemma4.sh first"
[[ -f "$VENV_DIR/bin/vllm" ]] || die "vllm binary not found in $VENV_DIR/bin — re-run setup_gemma4.sh"

# ── defaults ──────────────────────────────────────────────────────────────────
# Default model: E4B — smallest Gemma 4, fits a 16–24 GB GPU.
# Override with first positional arg or GEMMA4_MODEL env var.
MODEL="${GEMMA4_MODEL:-google/gemma-4-26B-A4B-it}"
PORT=8000
EAGER=0
TP=1
MAX_LEN=16384
DTYPE=bfloat16       # Gemma 4 is BF16-native
GPU_MEM=0.92
FP8_KV=0             # --fp8-kv flag halves KV cache memory

# ── parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --port)    PORT="$2";    shift 2 ;;
        --tp)      TP="$2";      shift 2 ;;
        --max-len) MAX_LEN="$2"; shift 2 ;;
        --dtype)   DTYPE="$2";   shift 2 ;;
        --gpu-mem) GPU_MEM="$2"; shift 2 ;;
        --eager)   EAGER=1;      shift   ;;
        --fp8-kv)  FP8_KV=1;     shift   ;;
        --help|-h)
            sed -n 's/^# \{0,1\}//p' "$0" | head -30
            exit 0
            ;;
        -*)
            die "Unknown flag: $1  (use --help for usage)"
            ;;
        *)
            MODEL="$1"
            shift
            ;;
    esac
done

# ── free port if occupied ─────────────────────────────────────────────────────
if lsof -ti:"$PORT" >/dev/null 2>&1; then
    warn "Port $PORT is in use — killing existing process…"
    lsof -ti:"$PORT" | xargs -r kill -9 2>/dev/null || true
    sleep 1
fi

# ── build vllm serve command ──────────────────────────────────────────────────
SERVE_CMD=(
    "$VENV_DIR/bin/vllm" serve "$MODEL"
    --port                    "$PORT"
    --tensor-parallel-size    "$TP"
    --max-model-len           "$MAX_LEN"
    --dtype                   "$DTYPE"
    --gpu-memory-utilization  "$GPU_MEM"

    # Gemma 4 is multimodal but we only need text for math benchmarks.
    # Skipping multimodal profiling frees significant KV cache memory.
    --limit-mm-per-prompt     "image=0,audio=0"

    # Tool calling — must use the gemma4 parser (NOT hermes).
    --enable-auto-tool-choice
    --tool-call-parser         gemma4

    # NOTE: --reasoning-parser gemma4 is intentionally omitted.
    # There is a known vLLM ≤0.19 bug where the tool-call parser never fires
    # if the model skips the <|channel> preamble.  For tool-heavy workloads,
    # omit the reasoning parser and control thinking per-request via extra_body.
    # See gemma4_vllm_math_benchmark.md §11 for details.

    # Disable thinking by default to avoid empty-channel overhead.
    # The benchmark config can override this per-request.
    --default-chat-template-kwargs '{"enable_thinking": false}'

    --host 0.0.0.0
)

[[ $EAGER -eq 1 ]] && SERVE_CMD+=(--enforce-eager)
[[ $FP8_KV -eq 1 ]] && SERVE_CMD+=(--kv-cache-dtype fp8)

# Async scheduling improves throughput for batchable workloads
SERVE_CMD+=(--async-scheduling)

# ── print summary ─────────────────────────────────────────────────────────────
echo ""
echo -e "\033[36m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\033[0m"
echo -e "\033[36m  Gemma 4 vLLM server  (tool calling enabled)\033[0m"
echo ""
info "model   : $MODEL"
info "port    : $PORT"
info "tp      : $TP"
info "max-len : $MAX_LEN"
info "dtype   : $DTYPE"
info "gpu-mem : $GPU_MEM"
info "fp8-kv  : $( [[ $FP8_KV -eq 1 ]] && echo "yes" || echo "no" )"
info "eager   : $( [[ $EAGER -eq 1 ]] && echo "yes (fast startup)" || echo "no  (compiled, ~3-4 min cold start)" )"
echo ""
info "command : ${SERVE_CMD[*]}"
echo ""
echo -e "\033[36m  Once healthy, run the benchmark:\033[0m"
echo "    uv run python scripts/run_benchmark.py \\"
echo "        --config libs/evaluation/src/evaluation/configs/gemma4_olym_hard.yaml"
echo ""
echo -e "\033[36m  Health check:\033[0m"
echo "    curl -s http://localhost:${PORT}/health"
echo "    curl -s http://localhost:${PORT}/v1/models | python3 -m json.tool"
echo -e "\033[36m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\033[0m"
echo ""

# ── launch ────────────────────────────────────────────────────────────────────
exec "${SERVE_CMD[@]}"
