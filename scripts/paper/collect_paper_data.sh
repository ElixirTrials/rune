#!/usr/bin/env bash
# Paper data collection — reproducible evaluation runs.
#
# Launches vLLM, waits for readiness, runs evaluation, shuts down.
# Results go to evaluation_results/ and MLflow.
#
# Usage:
#   bash scripts/paper/collect_paper_data.sh             # phase 1 only (i-iv)
#   bash scripts/paper/collect_paper_data.sh --phase 2   # phase 2 (v + gates)
#   bash scripts/paper/collect_paper_data.sh --phase all  # everything
#   bash scripts/paper/collect_paper_data.sh --fresh --phase all  # wipe results + checkpoints first
set -euo pipefail

# ── Logging ─────────────────────────────────────────────────────────
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
log_section() { echo ""; log "══════════════════════════════════════════════════"; log "$*"; log "══════════════════════════════════════════════════"; }

# ── Locked parameters (from HPO trial 6, fitness=0.8090) ─────────────
MODEL="Qwen/Qwen3.5-9B"
WARM_START="danielcherubini/Qwen3.5-DeltaCoder-9B"
BENCHMARKS="${BENCHMARKS:-humaneval mbpp}"
CORPUS="data/mined/all_unrolled.jsonl"
RAG_TOP_K=5
VLLM_PORT=8100
VLLM_MAX_MODEL_LEN=3072
VLLM_GPU_UTIL=0.95

# HPO best adapter: diffloss-v1 trial 6 (run e9c9760f816f46948197519e1c905273)
ADAPTER_III="${ADAPTER_III:-hpo_artifacts/best_diffloss_v1/}"
HPO_BEST_RUN_ID="e9c9760f816f46948197519e1c905273"

# TTT-E2E parameters (from Sun et al. 2024 defaults)
TTT_LR=1e-4
TTT_STEPS=5
TTT_MLP_FRACTION=0.25

# Hypernetwork checkpoint — S3 URI consumed directly (no local download needed)
HYPERNET_CHECKPOINT="${HYPERNET_CHECKPOINT:-s3://elixirtrials-949678234935-eu-west-2-artifacts/checkpoints/hypernet_hpo/checkpoint.pt}"

# vLLM health check: 200 attempts × 5s = 1000s (~16 min) to cover slow overlay FS loads
MAX_TRIES=200
HEALTH_SLEEP=5

# ── Output ───────────────────────────────────────────────────────────
RESULTS_DIR="${RESULTS_DIR:-evaluation_results/paper}"
LOG_DIR=".tmp/paper_logs"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
VLLM_LOG="${LOG_DIR}/vllm_${TIMESTAMP}.log"
MASTER_LOG="${LOG_DIR}/master_${TIMESTAMP}.log"
mkdir -p "${RESULTS_DIR}" "${LOG_DIR}"

# Mirror all stdout/stderr to the master log
exec > >(tee -a "${MASTER_LOG}") 2>&1

# ── Cleanup on exit ─────────────────────────────────────────────────
VLLM_PID=""
cleanup() {
    if [[ -n "${VLLM_PID}" ]]; then
        log "Shutting down vLLM (PID ${VLLM_PID})..."
        kill "${VLLM_PID}" 2>/dev/null || true
        wait "${VLLM_PID}" 2>/dev/null || true
        log "vLLM stopped."
    fi
    log "Master log: ${MASTER_LOG}"
}
trap cleanup EXIT INT TERM

# ── Record reproducibility metadata ─────────────────────────────────
GIT_COMMIT=$(git rev-parse HEAD)
GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
GIT_DIRTY=$(git diff --stat)

cat > "${RESULTS_DIR}/metadata.json" <<METAEOF
{
  "timestamp": "${TIMESTAMP}",
  "git_commit": "${GIT_COMMIT}",
  "git_branch": "${GIT_BRANCH}",
  "git_dirty": $(GIT_DIRTY="${GIT_DIRTY}" python3 -c "import json, os; print(json.dumps(os.environ['GIT_DIRTY'].strip()))"),
  "model": "${MODEL}",
  "warm_start": "${WARM_START}",
  "benchmarks": "$(echo ${BENCHMARKS})",
  "corpus": "${CORPUS}",
  "adapter_iii": "${ADAPTER_III}",
  "hypernet_checkpoint": "${HYPERNET_CHECKPOINT}",
  "ttt_lr": "${TTT_LR}",
  "ttt_steps": ${TTT_STEPS},
  "ttt_mlp_fraction": ${TTT_MLP_FRACTION},
  "rag_top_k": ${RAG_TOP_K},
  "vllm_max_model_len": ${VLLM_MAX_MODEL_LEN},
  "rune_max_attempts": ${RUNE_MAX_ATTEMPTS:-3},
  "vllm_gpu_util": ${VLLM_GPU_UTIL},
  "cuda_visible_devices": "${CUDA_VISIBLE_DEVICES:-all}",
  "python_version": "$(python3 --version 2>&1)"
}
METAEOF

log_section "Paper Data Collection"
log "Timestamp:   ${TIMESTAMP}"
log "Git commit:  ${GIT_COMMIT} (${GIT_BRANCH})"
log "Model:       ${MODEL}"
log "Benchmarks:  ${BENCHMARKS}"
log "Results dir: ${RESULTS_DIR}/"
log "Master log:  ${MASTER_LOG}"
log "vLLM log:    ${VLLM_LOG}"

# ── System diagnostics ──────────────────────────────────────────────
log "System diagnostics:"
log "  RAM free: $(free -h | awk '/Mem:/ {print $4 "/" $2}')"
log "  GPU: $(nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null || echo 'N/A')"
log "  Disk: $(df -h . | awk 'NR==2 {print $4 " free of " $2}')"
log "  Python: $(python3 --version 2>&1)"
log "  vLLM: $(uv run python -c 'import vllm; print(vllm.__version__)' 2>/dev/null || echo 'N/A')"

# ── Auto-fetch HPO adapter from MLflow/S3 if missing ────────────────
HPO_S3_PREFIX="s3://elixirtrials-949678234935-eu-west-2-artifacts/mlflow/artifacts/3/${HPO_BEST_RUN_ID}/artifacts"
if [[ ! -f "${ADAPTER_III}/adapter_config.json" ]]; then
    log "HPO adapter not found at ${ADAPTER_III} — fetching from S3..."
    mkdir -p "${ADAPTER_III}"
    if aws s3 cp "${HPO_S3_PREFIX}/" "${ADAPTER_III}/" --recursive 2>&1 | tail -5; then
        true  # aws s3 cp succeeded
    else
        log "S3 download failed — trying MLflow CLI (requires tracking server)..."
        MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://localhost:5000}" \
            uv run mlflow artifacts download \
                --run-id "${HPO_BEST_RUN_ID}" \
                --dst-path "${ADAPTER_III}" \
                2>&1 | tail -5
    fi
    if [[ ! -f "${ADAPTER_III}/adapter_config.json" ]]; then
        log "ERROR: Could not fetch HPO adapter. Neither S3 nor MLflow download produced adapter_config.json."
        exit 1
    fi
    log "HPO adapter downloaded to ${ADAPTER_III}"
else
    log "HPO adapter found at ${ADAPTER_III}"
fi

# ── Safety checks ────────────────────────────────────────────────────
if pgrep -f "run_training_hpo\|run_optimization" > /dev/null 2>&1; then
    log "WARNING: HPO process still running. Kill it first to free GPU memory:"
    pgrep -af "run_training_hpo\|run_optimization" | head -3
    read -p "Continue anyway? [y/N] " -n 1 -r
    echo
    [[ $REPLY =~ ^[Yy]$ ]] || exit 1
fi

# ── Parse flags ─────────────────────────────────────────────────────
FRESH=false
PHASE="1"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --fresh) FRESH=true; shift ;;
        --phase) PHASE="${2:-1}"; shift 2 ;;
        *) log "Unknown flag: $1"; exit 1 ;;
    esac
done

_kill_all_gpu_processes() {
    local pids
    pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ')
    if [[ -n "${pids}" ]]; then
        log "  Killing GPU processes: ${pids}"
        echo "${pids}" | xargs -r kill -9 2>/dev/null || true
        sleep 3
    fi
    pkill -9 -f "vllm" 2>/dev/null || true
}

if [[ "${FRESH}" == "true" ]]; then
    log "──── FRESH MODE: wiping previous results and checkpoints ────"
    rm -f "${RESULTS_DIR}"/table2*.json "${RESULTS_DIR}"/gate*.json
    rm -rf "${RESULTS_DIR}"/checkpoints "${RESULTS_DIR}"/rune_adapters_gate2 "${RESULTS_DIR}"/rune_adapters_ood
    log "  Removed result JSONs and checkpoint dir from ${RESULTS_DIR}/"
    _kill_all_gpu_processes
fi
log "Phase: ${PHASE}"

# ── Check if all outputs already exist (skip vLLM entirely) ─────────
NEEDS_WORK=false
if [[ "${PHASE}" == "1" || "${PHASE}" == "all" ]]; then
    [[ -f "${RESULTS_DIR}/table2_phase1.json" ]] || NEEDS_WORK=true
fi
if [[ "${PHASE}" == "2" || "${PHASE}" == "all" ]]; then
    [[ -f "${RESULTS_DIR}/table2_rune.json" ]] || NEEDS_WORK=true
    [[ -f "${RESULTS_DIR}/gate2.json" ]] || NEEDS_WORK=true
    [[ -f "${RESULTS_DIR}/gate3.json" ]] || NEEDS_WORK=true
fi

if [[ "${NEEDS_WORK}" == "false" ]]; then
    log "All requested outputs already exist in ${RESULTS_DIR}/. Delete to re-run."
    exit 0
fi

# Kill any existing GPU processes to avoid port conflicts / OOM
_kill_all_gpu_processes

# ── start_vllm helper ──────────────────────────────────────────────
start_vllm() {
    local label="${1:-vllm}"
    local vllm_log="${LOG_DIR}/${label}_${TIMESTAMP}.log"

    log "Starting vLLM (model=${MODEL}, port=${VLLM_PORT}, max_len=${VLLM_MAX_MODEL_LEN})..."
    log "  vLLM log: ${vllm_log}"
    VLLM_ALLOW_RUNTIME_LORA_UPDATING=1 \
    uv run python -m vllm.entrypoints.openai.api_server \
        --model "${MODEL}" \
        --port "${VLLM_PORT}" \
        --enable-lora \
        --max-lora-rank 32 \
        --dtype float16 \
        --max-model-len "${VLLM_MAX_MODEL_LEN}" \
        --gpu-memory-utilization "${VLLM_GPU_UTIL}" \
        --enforce-eager \
        > "${vllm_log}" 2>&1 &
    VLLM_PID=$!
    log "  vLLM PID: ${VLLM_PID}"

    local health_url="http://localhost:${VLLM_PORT}/health"
    log "  Waiting for ${health_url} (max ${MAX_TRIES}×${HEALTH_SLEEP}s = $((MAX_TRIES*HEALTH_SLEEP))s)..."
    local tries=0
    while ! curl -sf "${health_url}" > /dev/null 2>&1; do
        if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
            log "  ERROR: vLLM exited unexpectedly. Last 30 lines:"
            tail -30 "${vllm_log}"
            exit 1
        fi
        tries=$((tries + 1))
        if [[ $((tries % 20)) -eq 0 ]]; then
            local elapsed=$((tries * HEALTH_SLEEP))
            local last_line
            last_line=$(tail -1 "${vllm_log}" 2>/dev/null | head -c 120)
            log "  Still waiting... (${elapsed}s elapsed, attempt ${tries}/${MAX_TRIES})"
            log "  vLLM last line: ${last_line}"
            log "  GPU mem: $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
        fi
        if [[ ${tries} -ge ${MAX_TRIES} ]]; then
            log "  ERROR: vLLM not ready after ${MAX_TRIES} attempts (~$((MAX_TRIES*HEALTH_SLEEP))s)."
            log "  Last 30 lines of log:"
            tail -30 "${vllm_log}"
            exit 1
        fi
        sleep "${HEALTH_SLEEP}"
    done
    log "  vLLM ready. Startup took ~$((tries * HEALTH_SLEEP))s."
}

# ── Launch vLLM (only if vLLM conditions exist) ──────────────────────
smoke_test_vllm() {
    log "Running smoke test (1 completion)..."
    local smoke_result
    smoke_result=$(curl -sf http://localhost:${VLLM_PORT}/v1/completions \
        -H "Content-Type: application/json" \
        -d "{\"model\": \"${MODEL}\", \"prompt\": \"def add(a, b):\\n    return\", \"max_tokens\": 16}" \
        2>&1) || { log "SMOKE TEST FAILED: completions endpoint not responding"; log "${smoke_result}"; exit 1; }
    local smoke_text
    smoke_text=$(echo "${smoke_result}" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['text'][:80])" 2>/dev/null)
    log "Smoke test output: '${smoke_text}'"
    if echo "${smoke_text}" | grep -qiE 'sorry|cannot|I am|assist|help you|Hello'; then
        log "SMOKE TEST FAILED: model responding conversationally, not completing code"
        exit 1
    fi
    log "Smoke test passed."
}

# ── run_eval helper ─────────────────────────────────────────────────
run_eval() {
    local label="$1"
    local log_file="${LOG_DIR}/${label}_${TIMESTAMP}.log"
    shift
    log "Starting: ${label}"
    log "  Command: uv run python $*"
    log "  Log: ${log_file}"
    local start_time
    start_time=$(date +%s)
    set +e
    uv run python "$@" 2>&1 | tee "${log_file}"
    local exit_code=${PIPESTATUS[0]}
    set -e
    local end_time
    end_time=$(date +%s)
    local duration=$(( end_time - start_time ))
    if [[ ${exit_code} -eq 0 ]]; then
        log "Completed: ${label} (${duration}s, exit 0)"
    else
        log "FAILED: ${label} (${duration}s, exit ${exit_code})"
        log "  Last 10 lines of output:"
        tail -10 "${log_file}" | while IFS= read -r line; do log "    ${line}"; done
    fi
    return ${exit_code}
}

# ── Resolve conditions to run ──────────────────────────────────────
# vLLM conditions (i, ii, iii) run first. Then exclusive-GPU conditions:
# TTT (iv) and Rune iterative (v) both need the full GPU.
VLLM_CONDITIONS=()
TTT_CONDITIONS=()
RUNE_CONDITIONS=()

resolve_conditions() {
    local phase="$1"
    if [[ "${phase}" == "1" || "${phase}" == "all" ]]; then
        VLLM_CONDITIONS+=(i ii iii)
        TTT_CONDITIONS+=(iv)
    fi
    if [[ "${phase}" == "2" || "${phase}" == "all" ]]; then
        RUNE_CONDITIONS+=(v)
    fi
}
resolve_conditions "${PHASE}"

TABLE2_OUT="${RESULTS_DIR}/table2.json"
TABLE2_VLLM="${RESULTS_DIR}/table2_vllm.json"
TABLE2_RUNE="${RESULTS_DIR}/table2_rune.json"
TABLE2_TTT="${RESULTS_DIR}/table2_ttt.json"

ADAPTER_III_FLAG=""
if [[ -d "${ADAPTER_III}" ]] || [[ "${ADAPTER_III}" == s3://* ]]; then
    ADAPTER_III_FLAG="--adapter-iii ${ADAPTER_III}"
fi

# ── Validate hypernet checkpoint for Rune condition ─────────────────
if [[ ${#RUNE_CONDITIONS[@]} -gt 0 ]]; then
    if [[ -z "${HYPERNET_CHECKPOINT}" ]]; then
        log "ERROR: HYPERNET_CHECKPOINT not set."
        exit 1
    fi
    if [[ "${HYPERNET_CHECKPOINT}" != s3://* ]] && [[ ! -f "${HYPERNET_CHECKPOINT}" ]]; then
        log "ERROR: Hypernetwork checkpoint not found: ${HYPERNET_CHECKPOINT}"
        exit 1
    fi
    log "Hypernetwork checkpoint: ${HYPERNET_CHECKPOINT}"
fi

# ── Step 1: vLLM-served conditions (i, ii, iii) ───────────────────
if [[ ${#VLLM_CONDITIONS[@]} -gt 0 ]]; then
    log_section "vLLM conditions: ${VLLM_CONDITIONS[*]}"

    if [[ -f "${TABLE2_VLLM}" ]]; then
        log "Conditions ${VLLM_CONDITIONS[*]} already done: ${TABLE2_VLLM}"
    else
        start_vllm "vllm"
        smoke_test_vllm
        run_eval "table2_vllm" scripts/paper/run_all_conditions.py \
            --conditions ${VLLM_CONDITIONS[*]} \
            --benchmarks ${BENCHMARKS} \
            --model "${MODEL}" \
            --warm-start-adapter "${WARM_START}" \
            --corpus "${CORPUS}" \
            --rag-top-k ${RAG_TOP_K} \
            ${ADAPTER_III_FLAG} \
            --output "${TABLE2_VLLM}" || true
        if [[ -f "${TABLE2_VLLM}" ]]; then
            log "Conditions ${VLLM_CONDITIONS[*]} written to ${TABLE2_VLLM}"
        else
            log "WARNING: Conditions ${VLLM_CONDITIONS[*]} produced no output file"
        fi
    fi
fi

# ── Step 2: Kill vLLM, run exclusive-GPU conditions (TTT + Rune iterative) ──
_stop_vllm_for_exclusive_gpu() {
    if [[ -n "${VLLM_PID}" ]]; then
        log "Stopping vLLM for exclusive GPU access..."
        kill "${VLLM_PID}" 2>/dev/null || true
        wait "${VLLM_PID}" 2>/dev/null || true
        VLLM_PID=""
        sleep 3
    fi
    _kill_all_gpu_processes
    log "GPU memory freed."
    log "  GPU mem: $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
}

if [[ ${#TTT_CONDITIONS[@]} -gt 0 ]]; then
    if [[ -f "${TABLE2_TTT}" ]]; then
        log "Condition (iv) already done: ${TABLE2_TTT}"
    else
        _stop_vllm_for_exclusive_gpu
        run_eval "table2_ttt" scripts/paper/run_all_conditions.py \
            --conditions iv \
            --benchmarks ${BENCHMARKS} \
            --model "${MODEL}" \
            --warm-start-adapter "${WARM_START}" \
            --ttt-lr ${TTT_LR} \
            --ttt-steps ${TTT_STEPS} \
            --ttt-mlp-fraction ${TTT_MLP_FRACTION} \
            --output "${TABLE2_TTT}" || true
        if [[ -f "${TABLE2_TTT}" ]]; then
            log "Condition iv written to ${TABLE2_TTT}"
        else
            log "WARNING: Condition iv produced no output file"
        fi
    fi
fi

# Condition (v): Rune iterative — needs exclusive GPU for hypernetwork
# + base model (4-bit NF4) loaded together for the retry loop.
if [[ ${#RUNE_CONDITIONS[@]} -gt 0 ]]; then
    if [[ -f "${TABLE2_RUNE}" ]]; then
        log "Condition (v) already done: ${TABLE2_RUNE}"
    else
        _stop_vllm_for_exclusive_gpu
        RUNE_MAX_ATTEMPTS="${RUNE_MAX_ATTEMPTS:-3}"
        run_eval "table2_rune" scripts/paper/run_all_conditions.py \
            --conditions v \
            --benchmarks ${BENCHMARKS} \
            --model "${MODEL}" \
            --warm-start-adapter "${WARM_START}" \
            --hypernet-checkpoint "${HYPERNET_CHECKPOINT}" \
            --rune-max-attempts ${RUNE_MAX_ATTEMPTS} \
            --output "${TABLE2_RUNE}" || true
        if [[ -f "${TABLE2_RUNE}" ]]; then
            log "Condition (v) written to ${TABLE2_RUNE}"
        else
            log "WARNING: Condition (v) produced no output file"
        fi
    fi
fi

# ── Step 3: Merge all partial results into final table ──────────────
log "Merging results..."
uv run python -c "
import json, sys, os
partials = ['${TABLE2_VLLM}', '${TABLE2_RUNE}', '${TABLE2_TTT}']
out_path = '${TABLE2_OUT}'
merged = {}
for path in partials:
    if os.path.exists(path):
        data = json.load(open(path))
        for k, v in data.items():
            if k == 'conditions':
                merged.setdefault('conditions', {}).update(v)
            else:
                merged[k] = v
if merged:
    json.dump(merged, open(out_path, 'w'), indent=2)
    n = len(merged.get('conditions', {}))
    print(f'Merged {n} conditions into {out_path}')
    for cond, data in merged.get('conditions', {}).items():
        print(f'  ({cond}) {data.get(\"label\", \"?\")}: {data.get(\"scores\", {})}')
else:
    print('WARNING: No results to merge', file=sys.stderr)
"

# ── Gates (phase 2 only, need vLLM) ────────────────────────────────
if [[ "${PHASE}" == "2" || "${PHASE}" == "all" ]]; then
    GATE2_OUT="${RESULTS_DIR}/gate2.json"
    GATE3_OUT="${RESULTS_DIR}/gate3.json"
    GATE2_ADAPTER_DIR="${RESULTS_DIR}/rune_adapters_gate2"
    GATE3_ADAPTER_DIR="${RESULTS_DIR}/rune_adapters_ood"

    # Pre-generate adapters for gate2 (all 6 benchmarks) and gate3 (OOD tasks).
    # Needs exclusive GPU — stop vLLM first.
    if [[ ! -f "${GATE2_OUT}" && ! -f "${GATE2_ADAPTER_DIR}/manifest.json" ]]; then
        log "Stopping vLLM for gate2 adapter pre-generation..."
        if [[ -n "${VLLM_PID}" ]]; then
            kill "${VLLM_PID}" 2>/dev/null || true
            wait "${VLLM_PID}" 2>/dev/null || true
            VLLM_PID=""
            sleep 3
        fi
        log "  GPU mem: $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"

        run_eval "gate2_pregenerate" scripts/paper/run_all_conditions.py \
            --pregenerate \
            --benchmarks humaneval mbpp apps bigcodebench ds_1000 livecodebench \
            --model "${MODEL}" \
            --hypernet-checkpoint "${HYPERNET_CHECKPOINT}" \
            --rune-adapter-dir "${GATE2_ADAPTER_DIR}" || true
    fi

    if [[ ! -f "${GATE3_OUT}" && ! -f "${GATE3_ADAPTER_DIR}/manifest.json" ]]; then
        if [[ -n "${VLLM_PID}" ]]; then
            log "Stopping vLLM for gate3 adapter pre-generation..."
            kill "${VLLM_PID}" 2>/dev/null || true
            wait "${VLLM_PID}" 2>/dev/null || true
            VLLM_PID=""
            sleep 3
        fi
        log "  GPU mem: $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"

        run_eval "gate3_pregenerate" scripts/paper/run_gate3.py \
            --pregenerate \
            --model "${MODEL}" \
            --hypernet-checkpoint "${HYPERNET_CHECKPOINT}" \
            --rune-adapter-dir "${GATE3_ADAPTER_DIR}" || true
    fi

    # Restart vLLM for evaluation
    if ! pgrep -f "vllm.entrypoints" > /dev/null 2>&1; then
        start_vllm "vllm_gates"
    fi

    # Gate 2: Multi-benchmark robustness
    if [[ -f "${GATE2_OUT}" ]]; then
        log "Gate 2 already done: ${GATE2_OUT} — skipping."
    else
        run_eval "gate2" scripts/paper/run_gate2.py \
            --model "${MODEL}" \
            --warm-start-adapter "${WARM_START}" \
            --rune-adapter-dir "${GATE2_ADAPTER_DIR}" \
            --output "${GATE2_OUT}" || true
    fi

    # Gate 3: OOD procedural encoding
    if [[ -f "${GATE3_OUT}" ]]; then
        log "Gate 3 already done: ${GATE3_OUT} — skipping."
    else
        run_eval "gate3" scripts/paper/run_gate3.py \
            --model "${MODEL}" \
            --warm-start-adapter "${WARM_START}" \
            --rune-adapter-dir "${GATE3_ADAPTER_DIR}" \
            --output "${GATE3_OUT}" || true
    fi
fi

log_section "Collection Complete"
log "Results:"
ls -la "${RESULTS_DIR}/" | while IFS= read -r line; do log "  ${line}"; done
log "Logs: ${LOG_DIR}/"
log "Master log: ${MASTER_LOG}"
log "MLflow experiments: paper-table2, paper-gate2, paper-gate3"
