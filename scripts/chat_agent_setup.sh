#!/usr/bin/env bash
# chat_agent_setup.sh — one-shot environment setup for the Rune chat agents
#
# Creates / updates .venv-tools with all runtime deps needed by
#   scripts/chat_agent.py          (SubprocessExecutor, local bash sandbox)
#   scripts/chat_agent_sandbox.py  (OpenSandboxExecutor, Docker sandbox)
#
# Safe to re-run; uv pip install is idempotent.
#
# Usage:
#   bash scripts/chat_agent_setup.sh
#   bash scripts/chat_agent_setup.sh --with-opensandbox   # also installs opensandbox SDK
#   bash scripts/chat_agent_setup.sh --skip-model-check   # skip HF hub download check
#
# Prerequisites:
#   uv   — https://github.com/astral-sh/uv
#   CUDA toolkit matching your driver (for torch GPU support)

set -euo pipefail

# ── config ─────────────────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$REPO_ROOT/.chat-venv"
PYTHON_VERSION="3.12"       # 3.11 recommended; 3.12 also works
MODEL_ID="Qwen/Qwen3.5-4B"

WITH_OPENSANDBOX=0
SKIP_MODEL_CHECK=0
for arg in "$@"; do
    case "$arg" in
        --with-opensandbox)  WITH_OPENSANDBOX=1 ;;
        --skip-model-check)  SKIP_MODEL_CHECK=1 ;;
        --help|-h)
            grep '^#' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
    esac
done

# ── helpers ────────────────────────────────────────────────────────────────────
info()  { echo -e "\033[36m[setup]\033[0m $*"; }
ok()    { echo -e "\033[32m[  ok ]\033[0m $*"; }
warn()  { echo -e "\033[33m[ warn]\033[0m $*"; }
die()   { echo -e "\033[31m[error]\033[0m $*" >&2; exit 1; }

# ── 0. sanity checks ───────────────────────────────────────────────────────────
info "Checking prerequisites…"

command -v uv  >/dev/null 2>&1 || die "'uv' not found. Install: curl -LsSf https://astral.sh/uv/install.sh | sh"
command -v git >/dev/null 2>&1 || die "'git' not found."

CUDA_AVAIL=$(python3 -c "import subprocess,sys; r=subprocess.run(['nvidia-smi'],capture_output=True); sys.exit(0 if r.returncode==0 else 1)" 2>/dev/null && echo yes || echo no)
if [[ "$CUDA_AVAIL" == "no" ]]; then
    warn "nvidia-smi not found — CPU-only torch will be installed (very slow for inference)"
fi

# ── 1. create / reuse venv ─────────────────────────────────────────────────────
if [[ ! -d "$VENV_DIR" ]]; then
    info "Creating Python $PYTHON_VERSION virtual environment at $VENV_DIR …"
    uv venv "$VENV_DIR" --python "$PYTHON_VERSION"
    ok "venv created"
else
    ok "Reusing existing venv at $VENV_DIR"
fi

PIP="$VENV_DIR/bin/uv pip"

# ── 2. core torch ──────────────────────────────────────────────────────────────
info "Installing PyTorch (CUDA 12)…"
# Use the cu121 index; adjust to cu118 if needed for older drivers
uv pip install \
    --python "$VENV_DIR/bin/python" \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121
ok "torch installed"

# ── 3. transformers from git main (required for Qwen3.5 architecture) ──────────
info "Installing transformers from git main (needed for Qwen3.5 support)…"
uv pip install \
    --python "$VENV_DIR/bin/python" \
    "git+https://github.com/huggingface/transformers.git@main"
ok "transformers installed"

# Check the installed version
TRANSFORMERS_VER=$("$VENV_DIR/bin/python" -c "import transformers; print(transformers.__version__)")
ok "transformers version: $TRANSFORMERS_VER"

# ── 4. runtime deps ────────────────────────────────────────────────────────────
info "Installing runtime dependencies…"
uv pip install \
    --python "$VENV_DIR/bin/python" \
    accelerate \
    huggingface_hub \
    sentencepiece \
    tokenizers \
    safetensors \
    protobuf \
    openai \
    aiohttp \
    requests \
    pandas \
    numpy \
    scipy \
    scikit-learn \
    matplotlib
ok "runtime deps installed"

# ── 5. opensandbox SDK (optional) ─────────────────────────────────────────────
if [[ "$WITH_OPENSANDBOX" -eq 1 ]]; then
    info "Installing opensandbox SDK…"
    uv pip install --python "$VENV_DIR/bin/python" opensandbox
    ok "opensandbox installed"
else
    info "Skipping opensandbox SDK (pass --with-opensandbox to install)"
fi

# ── 6. install libs/shared dependencies (no editable install — avoids >=3.12) ─
info "Installing libs/shared dependencies…"
SHARED_SRC="$REPO_ROOT/libs/shared/src"
[[ -f "$SHARED_SRC/shared/executor.py" ]] || die "Missing $SHARED_SRC/shared/executor.py"
[[ -f "$SHARED_SRC/shared/tools.py"    ]] || die "Missing $SHARED_SRC/shared/tools.py"
# Install the deps declared in libs/shared/pyproject.toml directly so that the
# shared package's __init__.py (which imports sqlmodel, pydantic, jinja2, etc.)
# works at runtime.  We don't do an editable install because pyproject.toml
# declares requires-python>=3.12 while this venv intentionally targets 3.11.
uv pip install \
    --python "$VENV_DIR/bin/python" \
    "pydantic>=2.0.0" \
    "sqlmodel>=0.0.16" \
    "jinja2>=3.1.0" \
    "jupyter_client>=8.0.0" \
    "psutil>=7.2.2" \
    "ipykernel>=7.2.0"
ok "libs/shared deps installed (imported via sys.path from $SHARED_SRC)"

# ── 7. validate model availability ────────────────────────────────────────────
if [[ "$SKIP_MODEL_CHECK" -eq 0 ]]; then
    info "Checking model cache for $MODEL_ID …"
    MODEL_CACHED=$("$VENV_DIR/bin/python" - <<'PYEOF'
import os, sys
try:
    from huggingface_hub import try_to_load_from_cache, HUGGINGFACE_HUB_CACHE
    result = try_to_load_from_cache("Qwen/Qwen3.5-4B", "config.json")
    if result and result != "not-in-cache":
        print("cached")
    else:
        print("missing")
except Exception as e:
    print(f"error: {e}")
PYEOF
)
    if [[ "$MODEL_CACHED" == "cached" ]]; then
        ok "Model found in HuggingFace cache"
    else
        warn "Model not in cache. Downloading $MODEL_ID (~8 GB)…"
        "$VENV_DIR/bin/python" -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3.5-4B', ignore_patterns=['*.gguf'])
print('Download complete')
"
        ok "Model downloaded"
    fi
fi

# ── 8. quick smoke test ────────────────────────────────────────────────────────
info "Running quick smoke test…"
"$VENV_DIR/bin/python" - "$REPO_ROOT" <<'PYEOF'
import sys

repo_root = sys.argv[1]
sys.path.insert(0, f"{repo_root}/libs/shared/src")

errors = []

try:
    import torch
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"  torch {torch.__version__}  —  {gpu}")
except Exception as e:
    errors.append(f"torch: {e}")

try:
    import transformers
    print(f"  transformers {transformers.__version__}")
    from transformers import AutoConfig
    print("  AutoConfig import ok")
except Exception as e:
    errors.append(f"transformers: {e}")

try:
    import accelerate
    print(f"  accelerate {accelerate.__version__}")
except Exception as e:
    errors.append(f"accelerate: {e}")

try:
    from shared.executor import SubprocessExecutor
    from shared.tools import CODE_EXECUTOR_TOOLS
    print(f"  libs/shared ok — {len(CODE_EXECUTOR_TOOLS)} tools defined")
except Exception as e:
    errors.append(f"libs/shared: {e}")

if errors:
    print("\nErrors found:")
    for e in errors:
        print(f"  ✗ {e}")
    sys.exit(1)
else:
    print("\n  All checks passed ✓")
PYEOF

ok "Smoke test passed"

# ── done ───────────────────────────────────────────────────────────────────────
echo ""
echo -e "\033[32m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\033[0m"
echo -e "\033[32m  Setup complete!\033[0m"
echo ""
echo "  Run the subprocess (local bash) agent:"
echo "    $VENV_DIR/bin/python scripts/chat_agent.py"
echo ""
echo "  Run the OpenSandbox (Docker) agent:"
echo "    # First start the sandbox server:"
echo "    opensandbox-server init-config ~/.sandbox.toml --example docker"
echo "    opensandbox-server &"
echo "    # Then:"
echo "    $VENV_DIR/bin/python scripts/chat_agent_sandbox.py"
echo ""
if [[ "$WITH_OPENSANDBOX" -eq 0 ]]; then
echo "  To enable OpenSandbox support:"
echo "    bash scripts/chat_agent_setup.sh --with-opensandbox"
echo ""
fi
echo -e "\033[32m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\033[0m"
