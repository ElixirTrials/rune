#!/usr/bin/env bash
# setup_gemma4.sh — Bootstrap a dedicated .gemma4env for vLLM + Gemma 4.
#
# Creates an isolated Python 3.12 venv at .gemma4env (separate from the main
# rune .venv so the transformers 5.5.0 pin doesn't conflict with the core
# project's pin).  All vLLM serving and Gemma 4 inference runs from this venv.
#
# CUDA auto-detection: reads the max supported CUDA version from nvidia-smi
# and picks the matching nightly wheel index automatically.  Override with
# CUDA_VERSION=cu129 if auto-detection is wrong.
#
# Usage:
#   bash scripts/setup_gemma4.sh              # auto-detect CUDA version
#   CUDA_VERSION=cu129 bash scripts/setup_gemma4.sh   # force cu129
#   CUDA_VERSION=cu130 bash scripts/setup_gemma4.sh   # Blackwell / CUDA 13.0
#   SKIP_VLLM=1 bash scripts/setup_gemma4.sh          # only base deps, no vLLM
#
# After setup:
#   bash scripts/start_gemma4_server.sh             # serve E4B (default)
#   bash scripts/start_gemma4_server.sh google/gemma-4-26B-A4B-it
#
# Model download (requires HF token + accepted Google licence terms):
#   huggingface-cli login
#   huggingface-cli download google/gemma-4-E4B-it --local-dir ~/models/gemma-4-E4B-it

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VENV_DIR="$ROOT/.gemma4env"

# ── auto-detect CUDA version from driver ──────────────────────────────────────
# Map the driver's max supported CUDA to the closest available wheel index.
# nvidia-smi reports e.g. "CUDA Version: 12.6" — we pick:
#   >= 12.9  →  cu129
#   >= 12.8  →  cu128  (fallback to cu124 if no cu128 wheel exists)
#   >= 12.0  →  cu124  (safe for 12.4 – 12.8 drivers)
#   >= 13.0  →  cu130  (Blackwell)
_detect_cuda_version() {
    local ver
    ver=$(nvidia-smi 2>/dev/null | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+" | head -1)
    [[ -z "$ver" ]] && { echo "cu124"; return; }   # no GPU visible, default safe
    local major minor
    major=$(echo "$ver" | cut -d. -f1)
    minor=$(echo "$ver" | cut -d. -f2)
    local combined=$(( major * 10 + minor ))   # e.g. 12.6 → 126
    if   (( combined >= 130 )); then echo "cu130"
    elif (( combined >= 129 )); then echo "cu129"
    else                              echo "cu124"   # safe for 12.0 – 12.8
    fi
}

if [[ -z "${CUDA_VERSION:-}" ]]; then
    CUDA_VERSION=$(_detect_cuda_version)
fi

info()  { echo -e "\033[36m[setup]\033[0m $*"; }
ok()    { echo -e "\033[32m[  ok ]\033[0m $*"; }
warn()  { echo -e "\033[33m[ warn]\033[0m $*"; }
die()   { echo -e "\033[31m[error]\033[0m $*" >&2; exit 1; }

# ── CUDA driver preflight ─────────────────────────────────────────────────────
# vLLM 0.19+ (the first nightly with Gemma 4 support) uses cuMemcpyBatchAsync
# which is a CUDA 12.9 driver API call.  The cu124 nightly wheel no longer
# exists — all current nightly builds require CUDA 12.9 driver.
# Required: NVIDIA driver >= 565.57.01 (reports "CUDA Version: 12.9" in nvidia-smi).
_driver_cuda_minor() {
    nvidia-smi 2>/dev/null \
        | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+" \
        | awk -F. '{print $1 * 10 + $2}'
}
if [[ "${SKIP_VLLM:-0}" != "1" ]]; then
    DRIVER_COMBINED=$(_driver_cuda_minor)
    if [[ -n "$DRIVER_COMBINED" ]] && (( DRIVER_COMBINED < 129 )); then
        DRIVER_VER=$(nvidia-smi 2>/dev/null | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+" | head -1)
        die "CUDA driver too old (${DRIVER_VER}).  vLLM 0.19+ requires CUDA 12.9 driver.
       Upgrade the host NVIDIA driver to >= 565.57.01, or switch to a
       cloud instance with a CUDA 12.9 image (RunPod PyTorch 2.6+, etc.)
       Use SKIP_VLLM=1 to install only base deps without this check."
    fi
fi

info "CUDA wheel: ${CUDA_VERSION}  (override with CUDA_VERSION=cu129 if wrong)"

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

# ── base deps (always installed, before vLLM so the solver sees them) ─────────
info "Installing base dependencies …"
uv pip install --quiet \
  "openai>=1.0.0" \
  "huggingface_hub>=0.24.0" \
  "sympy>=1.12" \
  "numpy>=1.26" \
  "tqdm" \
  "mpmath>=1.3"

# ── MathSandbox deps (before vLLM — scipy needs a wheel, not a source build) ──
# scipy and networkx are imported by the MathSandbox kernel prelude.
# --only-binary :all: prevents pip from trying to compile scipy from source
# (which requires gfortran/g95 and will fail on most cloud instances).
# jupyter_client + ipykernel give us the long-lived IPython kernel that
# MathSandbox wraps.
info "Installing MathSandbox kernel dependencies (scipy wheel, jupyter, networkx) …"
uv pip install --quiet \
  --only-binary scipy \
  "scipy>=1.13" \
  "networkx>=3.3" \
  "jupyter_client>=8.6" \
  "ipykernel>=6.29"

# ── rune shared library (mathbox, etc.) ───────────────────────────────────────
# Editable install so the sandbox can import shared.mathbox, shared.math_retriever, etc.
info "Installing rune shared library (editable) …"
uv pip install --quiet -e "$ROOT/libs/shared"

# ── vLLM + CUDA stack ─────────────────────────────────────────────────────────
if [[ "${SKIP_VLLM:-0}" == "1" ]]; then
    warn "SKIP_VLLM=1 — skipping vLLM install.  Server will not be runnable."
else
    info "Installing vLLM nightly ($CUDA_VERSION) + PyTorch — large download …"
    info "CUDA wheel index: https://wheels.vllm.ai/nightly/${CUDA_VERSION}"
    # Always pull from the vLLM nightly index — the stable PyPI vLLM has
    # drifted to requiring cu129+ and lacks older CUDA-compatible builds.
    # The nightly consistently publishes cu124, cu129, and cu130 wheels with
    # Gemma 4 tool/reasoning parser support.
    uv pip install -U \
        vllm --pre \
        --extra-index-url "https://wheels.vllm.ai/nightly/${CUDA_VERSION}" \
        --extra-index-url "https://download.pytorch.org/whl/${CUDA_VERSION}" \
        --index-strategy unsafe-best-match

    # vLLM tends to downgrade transformers; re-assert Gemma 4's minimum.
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
    "scipy":            "scipy",
    "networkx":         "networkx",
    "mpmath":           "mpmath",
    "jupyter_client":   "jupyter_client",
    "ipykernel":        "ipykernel",
    "shared":           "shared",
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
