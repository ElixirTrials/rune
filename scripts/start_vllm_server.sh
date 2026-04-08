#!/usr/bin/env bash
# start_vllm_server.sh — Serve a local model via vLLM with tool calling enabled.
#
# Uses the .qwenv environment created by scripts/qwenv_setup.sh.
# Tool calling is enabled via --enable-auto-tool-choice --tool-call-parser hermes.
#
# Usage:
#   bash scripts/start_vllm_server.sh                        # auto-detect model from .qwenv.env
#   bash scripts/start_vllm_server.sh /path/to/snapshot      # explicit snapshot path
#   bash scripts/start_vllm_server.sh --port 8001            # custom port (default 8000)
#   bash scripts/start_vllm_server.sh --eager                # skip CUDA graph compilation (~40s vs ~4min)
#   bash scripts/start_vllm_server.sh --tp 2                 # tensor parallel across 2 GPUs
#   bash scripts/start_vllm_server.sh --max-len 32768        # cap context window
#   bash scripts/start_vllm_server.sh /path/snap --eager --port 8001 --tp 2
#
# Then run the benchmark:
#   .qwenv/bin/python scripts/run_benchmark.py \
#       --config libs/evaluation/src/evaluation/configs/qwen3_5_vllm_server.yaml
#
# Health check:
#   curl -s http://localhost:8000/health
#   curl -s http://localhost:8000/v1/models | python3 -m json.tool

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$REPO_ROOT/.qwenv"
ENV_FILE="$REPO_ROOT/.qwenv.env"

# ── helpers ────────────────────────────────────────────────────────────────────
info()  { echo -e "\033[36m[serve]\033[0m $*"; }
ok()    { echo -e "\033[32m[  ok ]\033[0m $*"; }
warn()  { echo -e "\033[33m[ warn]\033[0m $*"; }
die()   { echo -e "\033[31m[error]\033[0m $*" >&2; exit 1; }

# ── check venv ────────────────────────────────────────────────────────────────
[[ -d "$VENV_DIR" ]] || die ".qwenv not found at $VENV_DIR — run scripts/qwenv_setup.sh first"
[[ -f "$VENV_DIR/bin/vllm" ]] || die "vllm binary not found in $VENV_DIR/bin — re-run qwenv_setup.sh"

# ── defaults ──────────────────────────────────────────────────────────────────
MODEL_PATH=""
PORT=8000
EAGER=0
TP=1
MAX_LEN=65536
DTYPE=float16

# ── parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --port)    PORT="$2";    shift 2 ;;
        --tp)      TP="$2";      shift 2 ;;
        --max-len) MAX_LEN="$2"; shift 2 ;;
        --dtype)   DTYPE="$2";   shift 2 ;;
        --eager)   EAGER=1;      shift   ;;
        --help|-h)
            sed -n 's/^# \{0,1\}//p' "$0" | head -25
            exit 0
            ;;
        -*)
            die "Unknown flag: $1  (use --help for usage)"
            ;;
        *)
            MODEL_PATH="$1"
            shift
            ;;
    esac
done

# ── resolve model path ────────────────────────────────────────────────────────
if [[ -z "$MODEL_PATH" ]]; then
    # Try .qwenv.env first
    if [[ -f "$ENV_FILE" ]]; then
        # shellcheck disable=SC1090
        QWENV_SNAPSHOT_PATH=""
        source "$ENV_FILE" 2>/dev/null || true
        MODEL_PATH="${QWENV_SNAPSHOT_PATH:-}"
    fi

    # Fall back to HF cache scan
    if [[ -z "$MODEL_PATH" ]]; then
        MODEL_PATH=$(
            "$VENV_DIR/bin/python" - <<'PYEOF' 2>/dev/null || true
from huggingface_hub import snapshot_download
try:
    print(snapshot_download("Qwen/Qwen3.5-9B", local_files_only=True,
                            ignore_patterns=["*.gguf"]))
except Exception:
    pass
PYEOF
        )
        MODEL_PATH="${MODEL_PATH%%$'\n'*}"
    fi
fi

[[ -n "$MODEL_PATH" ]] || die \
    "Could not find model path.  Pass it as an argument or run qwenv_setup.sh first."
[[ -d "$MODEL_PATH" ]] || die "Model path does not exist: $MODEL_PATH"

# ── free port if occupied ─────────────────────────────────────────────────────
if lsof -ti:"$PORT" >/dev/null 2>&1; then
    warn "Port $PORT is in use — killing existing process…"
    lsof -ti:"$PORT" | xargs -r kill -9 2>/dev/null || true
    sleep 1
fi

# ── build vllm serve command ──────────────────────────────────────────────────
SERVE_CMD=(
    "$VENV_DIR/bin/vllm" serve "$MODEL_PATH"
    --port              "$PORT"
    --tensor-parallel-size "$TP"
    --max-model-len     "$MAX_LEN"
    --dtype             "$DTYPE"
    --language-model-only                 # skip vision encoder for Qwen3.5
    --enable-auto-tool-choice             # accept tools param, format schema into prompt
    --tool-call-parser  hermes            # required by vLLM when auto-tool-choice is on
    # Note: hermes parser logs JSONDecodeError on malformed/empty <tool_call> blocks
    # but those errors are non-fatal (server still returns 200 OK with raw text).
    # parse_tool_calls() in mathbox.py extracts calls from the raw text as fallback.
    --reasoning-parser  qwen3             # extract <think>…</think> when present
)

[[ $EAGER -eq 1 ]] && SERVE_CMD+=(--enforce-eager)

# ── print summary ─────────────────────────────────────────────────────────────
echo ""
echo -e "\033[36m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\033[0m"
echo -e "\033[36m  vLLM server  (tool calling enabled)\033[0m"
echo ""
info "model   : $MODEL_PATH"
info "port    : $PORT"
info "tp      : $TP"
info "max-len : $MAX_LEN"
info "dtype   : $DTYPE"
info "eager   : $( [[ $EAGER -eq 1 ]] && echo "yes (fast startup)" || echo "no  (compiled, ~3-4 min cold start)" )"
echo ""
info "command : ${SERVE_CMD[*]}"
echo ""
echo -e "\033[36m  Once healthy, run the benchmark:\033[0m"
echo "    .qwenv/bin/python scripts/run_benchmark.py \\"
echo "        --config libs/evaluation/src/evaluation/configs/qwen3_5_vllm_server.yaml"
echo ""
echo -e "\033[36m  Health check:\033[0m"
echo "    curl -s http://localhost:${PORT}/health"
echo "    curl -s http://localhost:${PORT}/v1/models | python3 -m json.tool"
echo -e "\033[36m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\033[0m"
echo ""

# ── launch ────────────────────────────────────────────────────────────────────
exec "${SERVE_CMD[@]}"
