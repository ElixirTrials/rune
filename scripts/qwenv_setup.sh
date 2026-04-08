#!/usr/bin/env bash
# qwenv_setup.sh — Qwen3.5-9B + vllm environment setup
#
# Creates / updates .qwenv with everything needed to run the benchmark suite
# (evaluation / inference / shared libs) and serve Qwen/Qwen3.5-9B via vllm.
#
# INSTALL ORDER (matters — vllm ships its own pinned transformers):
#   1. gcc / build-essential  (triton needs it)
#   2. workspace libs         (shared → inference → evaluation, editable)
#   3. vllm                   (pulls in triton, flash-attn, its own transformers)
#   4. PyTorch CUDA 12        (may be re-pinned by vllm; re-assert after)
#   5. transformers @ git main ← MUST follow vllm; overrides its older pin
#
# On success a .qwenv.env file is written with key variables, and activation
# instructions are printed so the benchmark can be run immediately.
#
# Smoke test: launches `vllm serve`, polls /health, fires a completion request,
# then tears down the server.
#
# Usage:
#   bash scripts/qwenv_setup.sh
#   bash scripts/qwenv_setup.sh --skip-download    # skip weight download
#   bash scripts/qwenv_setup.sh --skip-smoke-test  # skip vllm serve test
#   bash scripts/qwenv_setup.sh --help
#
# Prerequisites:
#   uv        https://github.com/astral-sh/uv
#   gcc       sudo apt-get install -y build-essential
#   CUDA 12.x matching your NVIDIA driver

set -euo pipefail

# ── config ─────────────────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$REPO_ROOT/.qwenv"
PYTHON_VERSION="3.12"
MODEL_ID="Qwen/Qwen3.5-9B"

VLLM_PORT=8000
VLLM_STARTUP_TIMEOUT=600   # seconds to wait for /health — cold start includes torch.compile + CUDA graph capture (~3-4 min)
STALL_TIMEOUT=60            # seconds of zero download progress before restart
MAX_DOWNLOAD_RETRIES=10

SKIP_DOWNLOAD=0
SKIP_SMOKE_TEST=0
for arg in "$@"; do
    case "$arg" in
        --skip-download)    SKIP_DOWNLOAD=1 ;;
        --skip-smoke-test)  SKIP_SMOKE_TEST=1 ;;
        --help|-h)
            sed -n 's/^# \{0,1\}//p' "$0" | head -30
            exit 0
            ;;
    esac
done

# ── helpers ────────────────────────────────────────────────────────────────────
info()  { echo -e "\033[36m[setup]\033[0m $*"; }
ok()    { echo -e "\033[32m[  ok ]\033[0m $*"; }
warn()  { echo -e "\033[33m[ warn]\033[0m $*"; }
die()   { echo -e "\033[31m[error]\033[0m $*" >&2; exit 1; }

# git_clone_with_retry <url> <dest_dir> [max_attempts]
# Clones <url> into <dest_dir>, retrying on network failures.
# Uses --depth 1 to minimise transfer; does a shallow-fetch update on re-runs.
git_clone_with_retry() {
    local url="$1" dest="$2" max="${3:-8}"
    local delay=5
    for attempt in $(seq 1 "$max"); do
        if [[ -d "$dest/.git" ]]; then
            info "  git fetch (attempt $attempt) $url…"
            git -C "$dest" fetch --depth 1 origin HEAD && \
            git -C "$dest" reset --hard FETCH_HEAD && return 0
        else
            rm -rf "$dest"
            info "  git clone (attempt $attempt) $url…"
            git clone --depth 1 "$url" "$dest" && return 0
        fi
        warn "  git attempt $attempt failed — waiting ${delay}s…"
        sleep "$delay"
        delay=$(( delay * 2 > 120 ? 120 : delay * 2 ))
    done
    die "git clone/fetch failed after $max attempts: $url"
}

# ── 0. prerequisites ───────────────────────────────────────────────────────────
info "Checking prerequisites…"
command -v uv  >/dev/null 2>&1 || die "uv not found. Install: curl -LsSf https://astral.sh/uv/install.sh | sh"
command -v git >/dev/null 2>&1 || die "git not found."

if ! command -v gcc >/dev/null 2>&1; then
    warn "gcc not found — required by triton (vllm dependency)"
    info "Attempting: apt-get install -y build-essential"
    apt-get install -y build-essential 2>/dev/null \
        || die "gcc missing. Run: sudo apt-get install -y build-essential"
fi
ok "gcc: $(gcc --version | head -1)"

if command -v nvidia-smi >/dev/null 2>&1; then
    ok "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
else
    warn "nvidia-smi not found — GPU unavailable (inference will be very slow)"
fi

# ── 1. venv ────────────────────────────────────────────────────────────────────
if [[ ! -d "$VENV_DIR" ]]; then
    info "Creating Python $PYTHON_VERSION venv at $VENV_DIR…"
    uv venv "$VENV_DIR" --python "$PYTHON_VERSION"
    ok "venv created"
else
    ok "Reusing existing venv: $VENV_DIR"
fi

# ── 2. workspace libs (editable — BEFORE vllm so its dep-pins don't clobber) ──
# Install in dependency order: shared has no local deps, inference depends on
# shared, evaluation depends on both.  Installing all three in one uv invocation
# lets uv resolve cross-package deps to the local editables rather than PyPI.
info "Installing workspace libs (editable)…"
uv pip install \
    --python "$VENV_DIR/bin/python" \
    --editable "$REPO_ROOT/libs/shared" \
    --editable "$REPO_ROOT/libs/inference" \
    --editable "$REPO_ROOT/libs/evaluation"
ok "shared     $("$VENV_DIR/bin/python" -c 'import importlib.metadata; print(importlib.metadata.version("shared"))')"
ok "inference  $("$VENV_DIR/bin/python" -c 'import importlib.metadata; print(importlib.metadata.version("inference"))')"
ok "evaluation $("$VENV_DIR/bin/python" -c 'import importlib.metadata; print(importlib.metadata.version("evaluation"))')"

# ── 3. vllm (after local libs — brings triton, flash-attn, its own transformers)
info "Installing vllm… (includes triton + flash-attn; takes a while)"
uv pip install \
    --python "$VENV_DIR/bin/python" \
    vllm
VLLM_VER=$("$VENV_DIR/bin/python" -c "import vllm; print(vllm.__version__)")
ok "vllm $VLLM_VER"

# ── 4. PyTorch (CUDA 12.1) ─────────────────────────────────────────────────────
info "Installing / confirming PyTorch (CUDA 12.1)…"
uv pip install \
    --python "$VENV_DIR/bin/python" \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121
ok "torch $("$VENV_DIR/bin/python" -c 'import torch; print(torch.__version__)')"

# ── 5. transformers @ git main  ← MUST come after vllm ────────────────────────
# vllm pins an older transformers; Qwen3.5 needs the latest architecture code.
# We clone to disk first (with retry) then install from the local path — far
# more resilient to GnuTLS / mid-transfer network drops than a bare git+https
# install which has no retry capability.
TRANSFORMERS_CLONE_DIR="/tmp/transformers-git-main"
info "Cloning transformers (git main) to $TRANSFORMERS_CLONE_DIR…"
git_clone_with_retry \
    "https://github.com/huggingface/transformers.git" \
    "$TRANSFORMERS_CLONE_DIR"
info "Installing transformers from local clone (force-reinstall)…"
uv pip install \
    --python "$VENV_DIR/bin/python" \
    --force-reinstall \
    "$TRANSFORMERS_CLONE_DIR"
ok "transformers $("$VENV_DIR/bin/python" -c 'import transformers; print(transformers.__version__)')"

# ── 6. runtime extras ─────────────────────────────────────────────────────────
info "Installing runtime deps…"
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
    requests
ok "runtime deps installed"

# ── 6b. math sandbox extras ───────────────────────────────────────────────────
# scipy and networkx are pre-imported in MathSandbox._PRELUDE so competition
# code can use scipy.optimize and networkx graph algorithms without explicit
# imports.  Install after vllm so its dependency pins don't get clobbered.
info "Installing math sandbox extras (scipy, networkx)…"
uv pip install \
    --python "$VENV_DIR/bin/python" \
    scipy \
    networkx
ok "scipy    $("$VENV_DIR/bin/python" -c 'import scipy; print(scipy.__version__)')"
ok "networkx $("$VENV_DIR/bin/python" -c 'import networkx; print(networkx.__version__)')"

# ── 7. model download with stall-detection + auto-retry ───────────────────────
SNAPSHOT_PATH=""

if [[ "$SKIP_DOWNLOAD" -eq 0 ]]; then
    info "Checking local HuggingFace cache for $MODEL_ID…"

    # Try local_files_only first — no network if already present.
    CACHE_CHECK=$("$VENV_DIR/bin/python" <<'PYEOF'
import sys
try:
    from huggingface_hub import snapshot_download
    p = snapshot_download("Qwen/Qwen3.5-9B",
                          local_files_only=True,
                          ignore_patterns=["*.gguf"])
    print(f"hit:{p}")
except Exception:
    print("miss")
PYEOF
    )

    if [[ "$CACHE_CHECK" == hit:* ]]; then
        SNAPSHOT_PATH="${CACHE_CHECK#hit:}"
        SNAPSHOT_PATH="${SNAPSHOT_PATH%%$'\n'*}"   # first line only
        ok "Model already in cache: $SNAPSHOT_PATH"
    else
        info "Model not cached — downloading $MODEL_ID (~19 GB)…"
        info "Stall watchdog active: restarts if no new bytes for ${STALL_TIMEOUT}s"

        DOWNLOAD_SUCCESS=0
        for attempt in $(seq 1 "$MAX_DOWNLOAD_RETRIES"); do
            [[ $attempt -gt 1 ]] && { warn "Retry $attempt/$MAX_DOWNLOAD_RETRIES — waiting 5s…"; sleep 5; }
            info "Download attempt $attempt…"

            # Unquoted heredoc so $STALL_TIMEOUT is expanded by bash.
            # Python f-string braces (no leading $) are passed through unchanged.
            set +e
            DOWNLOAD_OUT=$("$VENV_DIR/bin/python" <<PYEOF
import sys, os, time, threading, signal
from pathlib import Path

MODEL_ID   = "Qwen/Qwen3.5-9B"
STALL_SECS = $STALL_TIMEOUT
HF_CACHE   = Path(os.environ.get("HF_HOME",
               os.path.expanduser("~/.cache/huggingface"))) / "hub"
MODEL_DIR  = HF_CACHE / "models--Qwen--Qwen3.5-9B"

def _du():
    total = 0
    if MODEL_DIR.exists():
        for f in MODEL_DIR.rglob("*"):
            try:
                if f.is_file():
                    total += f.stat().st_size
            except OSError:
                pass
    return total

last_bytes = [_du()]
last_tick  = [time.time()]
done_flag  = [False]

def watchdog():
    while not done_flag[0]:
        time.sleep(10)
        cur = _du()
        if cur > last_bytes[0]:
            last_bytes[0] = cur
            last_tick[0]  = time.time()
            print(f"  {cur / 1e9:.2f} GB in cache", flush=True)
        elif time.time() - last_tick[0] > STALL_SECS:
            print(f"  no progress for {STALL_SECS}s — interrupting…", flush=True)
            done_flag[0] = True
            signal.raise_signal(signal.SIGINT)

threading.Thread(target=watchdog, daemon=True).start()

try:
    from huggingface_hub import snapshot_download
    p = snapshot_download(MODEL_ID, ignore_patterns=["*.gguf"])
    done_flag[0] = True
    print(f"done:{p}", flush=True)
    sys.exit(0)
except KeyboardInterrupt:
    print("stalled", flush=True)
    sys.exit(1)
except Exception as exc:
    done_flag[0] = True
    print(f"err:{exc}", flush=True)
    sys.exit(2)
PYEOF
            )
            DL_RC=$?
            set -e

            # Extract the snapshot path from the last "done:…" line
            DONE_LINE=$(echo "$DOWNLOAD_OUT" | grep '^done:' | tail -1 || true)
            if [[ $DL_RC -eq 0 && -n "$DONE_LINE" ]]; then
                SNAPSHOT_PATH="${DONE_LINE#done:}"
                SNAPSHOT_PATH="${SNAPSHOT_PATH%%$'\n'*}"
                DOWNLOAD_SUCCESS=1
                break
            else
                warn "Attempt $attempt output:"
                echo "$DOWNLOAD_OUT" | tail -5 | sed 's/^/    /'
            fi
        done

        if [[ $DOWNLOAD_SUCCESS -eq 0 ]]; then
            die "Model download failed after $MAX_DOWNLOAD_RETRIES attempts"
        fi
        ok "Download complete"
    fi

else
    info "Skipping download (--skip-download)"
    SNAPSHOT_PATH=$("$VENV_DIR/bin/python" <<'PYEOF' 2>/dev/null || true
from huggingface_hub import snapshot_download
print(snapshot_download("Qwen/Qwen3.5-9B",
                        local_files_only=True,
                        ignore_patterns=["*.gguf"]))
PYEOF
    )
    SNAPSHOT_PATH="${SNAPSHOT_PATH%%$'\n'*}"
fi

if [[ -n "$SNAPSHOT_PATH" ]]; then
    ok "Snapshot path: $SNAPSHOT_PATH"
else
    warn "No snapshot path resolved — disabling smoke test"
    SKIP_SMOKE_TEST=1
fi

# ── 8. RoPE / vllm compatibility patches ──────────────────────────────────────
# Two patches discovered through iterative testing:
#
#   Patch A — model config.json (rope_scaling type)
#     Qwen3.5's config.json can have rope_scaling types that some vllm builds
#     reject.  Sanitised in-place with a .orig backup.
#
#   Patch B — vllm source file (list vs set bug, confirmed in vllm 0.18.0)
#     vllm/transformers_utils/configs/qwen3_5.py passes
#       kwargs["ignore_keys_at_rope_validation"] = ["mrope_section", ...]
#     as a LIST.  transformers' _check_received_keys does:
#       received_keys -= ignore_keys   # set -= list → TypeError
#     Fix: change [...] to {...} (set literal).

# ── Patch A: config.json ──────────────────────────────────────────────────────
if [[ -n "$SNAPSHOT_PATH" && -f "$SNAPSHOT_PATH/config.json" ]]; then
    info "Patch A: checking config.json for vllm compatibility…"
    export QWENV_SNAPSHOT="$SNAPSHOT_PATH"
    "$VENV_DIR/bin/python" <<'PYEOF'
import os, json, shutil
from pathlib import Path

snapshot = Path(os.environ["QWENV_SNAPSHOT"])
cfg_path  = snapshot / "config.json"
cfg       = json.loads(cfg_path.read_text())
patched   = []

rope_scaling = cfg.get("rope_scaling")
if rope_scaling and isinstance(rope_scaling, dict):
    rope_type = (rope_scaling.get("rope_type") or rope_scaling.get("type", ""))
    SUPPORTED = {"", "default", "linear", "dynamic", "su", "yarn"}
    if rope_type not in SUPPORTED:
        cfg.pop("rope_scaling")
        patched.append(f"removed rope_scaling (type={rope_type!r})")

if patched:
    backup = cfg_path.with_suffix(".json.orig")
    if not backup.exists():
        shutil.copy2(cfg_path, backup)
        print(f"  backed up original → {backup.name}")
    cfg_path.write_text(json.dumps(cfg, indent=2))
    for p in patched:
        print(f"  patched: {p}")
    print("  config.json updated ✓")
else:
    print("  config.json looks clean — no patch needed")
PYEOF
fi

# ── Patch B: vllm qwen3_5.py list → set ──────────────────────────────────────
info "Patch B: fixing vllm qwen3_5.py ignore_keys list→set (vllm bug #VLLM-1)…"
QWENV3_5_PY="$VENV_DIR/lib/python3.12/site-packages/vllm/transformers_utils/configs/qwen3_5.py"
if [[ -f "$QWENV3_5_PY" ]]; then
    "$VENV_DIR/bin/python" - "$QWENV3_5_PY" <<'PYEOF'
import sys, shutil
from pathlib import Path

target = Path(sys.argv[1])
src    = target.read_text()

old = (
    '        kwargs["ignore_keys_at_rope_validation"] = [\n'
    '            "mrope_section",\n'
    '            "mrope_interleaved",\n'
    '        ]'
)
new = (
    '        kwargs["ignore_keys_at_rope_validation"] = {\n'
    '            "mrope_section",\n'
    '            "mrope_interleaved",\n'
    '        }'
)

if old in src:
    backup = target.with_suffix(".py.orig")
    if not backup.exists():
        shutil.copy2(target, backup)
        print(f"  backed up → {backup.name}")
    target.write_text(src.replace(old, new))
    print("  patched: ignore_keys_at_rope_validation list → set ✓")
elif new in src:
    print("  already patched — skipping")
else:
    print("  WARNING: patch target not found — vllm version may differ, check manually")
    print(f"  file: {target}")
PYEOF
else
    warn "  $QWENV3_5_PY not found — skipping Patch B (vllm version may differ)"
fi

# ── 9. vllm smoke test ────────────────────────────────────────────────────────
VLLM_PID=""
VLLM_LOG="/tmp/qwenv_vllm_smoke_$$.log"

cleanup_vllm() {
    if [[ -n "$VLLM_PID" ]]; then
        info "Shutting down vllm (pid $VLLM_PID)…"
        kill "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
        VLLM_PID=""
    fi
    rm -f "$VLLM_LOG"
}
trap cleanup_vllm EXIT INT TERM

if [[ "$SKIP_SMOKE_TEST" -eq 0 ]]; then
    info "Starting vllm smoke test on port $VLLM_PORT…"
    info "  model: $SNAPSHOT_PATH"

    # Free the port if something is already bound to it
    { fuser -k "${VLLM_PORT}/tcp" 2>/dev/null; } \
        || { lsof -ti:"$VLLM_PORT" 2>/dev/null | xargs -r kill -9; } \
        || true
    sleep 1

    # --enforce-eager skips torch.compile + CUDA graph capture for the smoke
    # test, cutting cold-start from ~3-4 min down to ~40 s.  Remove it if you
    # want to verify the full compiled path.
    info "Command: vllm serve $SNAPSHOT_PATH --port $VLLM_PORT --reasoning-parser qwen3 --language-model-only --enforce-eager"
    "$VENV_DIR/bin/vllm" serve "$SNAPSHOT_PATH" \
        --port "$VLLM_PORT" \
        --reasoning-parser qwen3 \
        --language-model-only \
        --enforce-eager \
        >"$VLLM_LOG" 2>&1 &
    VLLM_PID=$!

    info "Waiting for vllm /health (timeout ${VLLM_STARTUP_TIMEOUT}s)…"
    READY=0
    for i in $(seq 1 "$VLLM_STARTUP_TIMEOUT"); do
        # Abort early if the server process died
        if ! kill -0 "$VLLM_PID" 2>/dev/null; then
            warn "vllm exited unexpectedly — last 50 lines of log:"
            tail -50 "$VLLM_LOG" >&2
            die "vllm failed to start"
        fi
        if curl -sf "http://localhost:${VLLM_PORT}/health" >/dev/null 2>&1; then
            READY=1
            break
        fi
        (( i % 15 == 0 )) && printf "  …%ds\n" "$i"
        sleep 1
    done

    if [[ $READY -eq 0 ]]; then
        warn "vllm did not become healthy within ${VLLM_STARTUP_TIMEOUT}s"
        warn "Last 50 lines of vllm log:"
        tail -50 "$VLLM_LOG" >&2
        die "Smoke test failed — startup timeout"
    fi
    ok "vllm is up — running test inference…"

    export QWENV_PORT="$VLLM_PORT"
    INFERENCE_RC=0
    "$VENV_DIR/bin/python" <<'PYEOF' || INFERENCE_RC=$?
import os, sys, json, urllib.request, urllib.error

port = os.environ["QWENV_PORT"]
base = f"http://localhost:{port}"

# Discover the model name registered in vllm (usually the path we passed)
with urllib.request.urlopen(f"{base}/v1/models", timeout=15) as r:
    models = json.loads(r.read())
model_name = models["data"][0]["id"]
print(f"  model id in vllm: {model_name}")

# Fire a minimal completion
payload = json.dumps({
    "model": model_name,
    "prompt": "The capital of France is",
    "max_tokens": 16,
    "temperature": 0.0,
}).encode()
req = urllib.request.Request(
    f"{base}/v1/completions",
    data=payload,
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(req, timeout=300) as r:
    resp = json.loads(r.read())

text = resp["choices"][0]["text"].strip()
print(f"  prompt:   'The capital of France is'")
print(f"  response: '{text}'")
PYEOF

    cleanup_vllm
    trap - EXIT INT TERM

    if [[ $INFERENCE_RC -eq 0 ]]; then
        ok "Smoke test passed ✓"
    else
        die "Smoke test failed — inference returned non-zero"
    fi
else
    info "Skipping smoke test (--skip-smoke-test)"
fi

# ── 10. write .qwenv.env — key variables for benchmark scripts ────────────────
ENV_FILE="$REPO_ROOT/.qwenv.env"
MODEL_ARG="${SNAPSHOT_PATH:-}"

info "Writing environment file: $ENV_FILE"
cat > "$ENV_FILE" <<EOF
# Auto-generated by scripts/qwenv_setup.sh — source this before running benchmarks
# source .qwenv.env

export VIRTUAL_ENV="$VENV_DIR"
export PATH="$VENV_DIR/bin:\$PATH"

export QWENV_VENV="$VENV_DIR"
export QWENV_MODEL_ID="$MODEL_ID"
export QWENV_SNAPSHOT_PATH="${MODEL_ARG}"
export QWENV_VLLM_PORT="$VLLM_PORT"

# Convenience — benchmark runner picks these up
export VLLM_BASE_URL="http://localhost:${VLLM_PORT}/v1"
export BENCHMARK_MODEL_ID="$MODEL_ID"
EOF
ok "Wrote $ENV_FILE"

# ── done ───────────────────────────────────────────────────────────────────────
echo ""
echo -e "\033[32m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\033[0m"
echo -e "\033[32m  qwenv setup complete!\033[0m"
echo ""
echo "  Model:   $MODEL_ID"
echo "  Venv:    $VENV_DIR"
[[ -n "$MODEL_ARG" ]] && echo "  Weights: $MODEL_ARG"
echo ""
echo -e "\033[36m  ── Activate environment ──────────────────────────────────────\033[0m"
echo "  Option A — source the env file (sets PATH + all QWENV_* vars):"
echo "    source .qwenv.env"
echo ""
echo "  Option B — activate the venv directly:"
echo "    source $VENV_DIR/bin/activate"
echo ""
echo -e "\033[36m  ── Serve the model ───────────────────────────────────────────\033[0m"
MODEL_SERVE="${MODEL_ARG:-<snapshot-path>}"
echo "  Eager (fast startup, ~40 s):"
echo "    $VENV_DIR/bin/vllm serve $MODEL_SERVE \\"
echo "      --port $VLLM_PORT --reasoning-parser qwen3 --language-model-only --enforce-eager"
echo ""
echo "  Compiled (full perf, ~3-4 min cold start):"
echo "    $VENV_DIR/bin/vllm serve $MODEL_SERVE \\"
echo "      --port $VLLM_PORT --reasoning-parser qwen3 --language-model-only"
echo ""
echo -e "\033[36m  ── Run benchmarks ────────────────────────────────────────────\033[0m"
echo "  Update the config file first with downloaded snapshot path! "
echo "    source .qwenv.env && python scripts/run_benchmark.py --config libs/evaluation/src/evaluation/configs/qwen3_5_olym_easy.yaml"
echo ""
echo "  Or Chat with Qwen:"
echo "    source .qwenv.env && python scripts/chat_agent.py --model $MODEL_SERVE"
echo ""
echo -e "\033[32m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\033[0m"
