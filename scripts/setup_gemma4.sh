#!/usr/bin/env bash
# setup_gemma4.sh — Bootstrap a dedicated .gemma4env for vLLM + Gemma 4.
#
# Creates an isolated Python 3.12 venv at .gemma4env (separate from the main
# rune .venv so the transformers 5.5.0 pin doesn't conflict with the core
# project's pin).  All vLLM serving and Gemma 4 inference runs from this venv.
#
# Usage:
#   bash scripts/setup_gemma4.sh              # install everything (CUDA 12.9)
#   CUDA_VERSION=cu130 bash scripts/setup_gemma4.sh   # Blackwell / CUDA 13.0
#   SKIP_VLLM=1 bash scripts/setup_gemma4.sh  # only base deps, no vLLM wheels
#
# After setup:
#   bash scripts/start_gemma4_server.sh             # serve E4B (default)
#   bash scripts/start_gemma4_server.sh google/gemma-4-27B-it --tp 2
#
# Download a model first (requires HF token + accepted Google terms):
#   huggingface-cli login
#   huggingface-cli download google/gemma-4-E4B-it --local-dir ~/models/gemma-4-E4B-it

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VENV_DIR="$ROOT/.gemma4env"
CUDA_VERSION="${CUDA_VERSION:-cu129}"

info()  { echo -e "\033[36m[setup]\033[0m $*"; }
ok()    { echo -e "\033[32m[  ok ]\033[0m $*"; }
warn()  { echo -e "\033[33m[ warn]\033[0m $*"; }
die()   { echo -e "\033[31m[error]\033[0m $*" >&2; exit 1; }

# ── gcc check (Triton JIT needs a C compiler) ──────────────────────────────────
if ! command -v gcc &>/dev/null; then
  info "gcc not found — installing build-essential..."
  apt-get install -y --no-install-recommends build-essential 2>/dev/null \
    || die "Could not install build-essential; install gcc manually or set CC."
fi

# ── uv check ──────────────────────────────────────────────────────────────────
if ! command -v uv &>/dev/null; then
  info "uv not found — installing via pip..."
  pip install --quiet uv
fi

# ── create venv ───────────────────────────────────────────────────────────────
if [[ ! -d "$VENV_DIR" ]] || [[ "$("$VENV_DIR/bin/python" --version 2>&1)" != *"3.12"* ]]; then
  info "Creating .gemma4env (Python 3.12) …"
  rm -rf "$VENV_DIR"
  uv venv "$VENV_DIR" --python 3.12
fi

# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

# ── base deps (always installed) ──────────────────────────────────────────────
info "Installing base dependencies …"
uv pip install --quiet \
  "openai>=1.0.0" \
  "huggingface_hub>=0.24.0" \
  "sympy>=1.12" \
  "numpy>=1.26" \
  "tqdm"

# ── vLLM + CUDA stack ─────────────────────────────────────────────────────────
if [[ "${SKIP_VLLM:-0}" == "1" ]]; then
  warn "SKIP_VLLM=1 — skipping vLLM install.  Server will not be runnable."
else
  info "Installing vLLM nightly + PyTorch ($CUDA_VERSION) — large download …"
  uv pip install -U \
    "torch>=2.6.0" \
    vllm --pre \
    --extra-index-url "https://wheels.vllm.ai/nightly/${CUDA_VERSION}" \
    --extra-index-url "https://download.pytorch.org/whl/${CUDA_VERSION}" \
    --index-strategy unsafe-best-match

  # vLLM may downgrade transformers; re-assert Gemma 4's minimum.
  # transformers 5.5.0 is required for Gemma 4 AutoProcessor support.
  info "Re-pinning transformers to 5.5.0 (Gemma 4 requirement) …"
  uv pip install --reinstall "transformers==5.5.0"
fi

# ── smoke test ────────────────────────────────────────────────────────────────
ok "Setup complete."
echo ""
echo "  Venv  : $VENV_DIR"
echo ""
info "Quick checks:"
"$VENV_DIR/bin/python" - <<'EOF'
import importlib, sys

ok = "\033[32m  ok\033[0m"
fail = "\033[31m FAIL\033[0m"

checks = {
    "openai":           "openai",
    "huggingface_hub":  "huggingface_hub",
    "transformers":     "transformers",
    "sympy":            "sympy",
    "numpy":            "numpy",
}

for name, mod in checks.items():
    try:
        m = importlib.import_module(mod)
        ver = getattr(m, "__version__", "?")
        print(f"{ok}  {name}=={ver}")
    except ImportError:
        print(f"{fail}  {name} NOT FOUND")

# Check transformers version meets 5.5.0
try:
    import transformers
    from packaging.version import Version
    req = Version("5.5.0")
    got = Version(transformers.__version__)
    if got < req:
        print(f"\033[33m warn\033[0m  transformers {got} < 5.5.0 — Gemma 4 requires >=5.5.0")
    else:
        print(f"{ok}  transformers {got} >= 5.5.0 ✓")
except Exception as e:
    print(f"  (could not verify transformers version: {e})")

try:
    import vllm
    print(f"{ok}  vllm=={vllm.__version__}")
    import torch
    print(f"{ok}  torch=={torch.__version__}  cuda_available={torch.cuda.is_available()}")
except ImportError:
    print("\033[33m warn\033[0m  vllm not installed (run without SKIP_VLLM=1 to install)")

EOF

echo ""
info "Next steps:"
echo "  1. Accept Gemma 4 terms on HuggingFace (one-time, in browser)"
echo "  2. huggingface-cli login"
echo "  3. bash scripts/start_gemma4_server.sh [MODEL_ID_OR_PATH]"
echo "       e.g.  bash scripts/start_gemma4_server.sh google/gemma-4-E4B-it"
echo ""
echo "  Benchmark (after server is up):"
echo "    uv run python scripts/run_benchmark.py \\"
echo "        --config libs/evaluation/src/evaluation/configs/gemma4_olym_hard.yaml"
