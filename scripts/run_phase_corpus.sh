#!/usr/bin/env bash
# run_phase_corpus.sh — batch oracle corpus production across all 6 benchmarks.
#
# APPS is subsampled to N=500 with seed=42, stratified by difficulty.
# All other benchmarks use their full split (or capped by --max-problems).
#
# Usage:
#   ./scripts/run_phase_corpus.sh                    # full run
#   ./scripts/run_phase_corpus.sh --dry-run          # dry run (no GPU)
#   ./scripts/run_phase_corpus.sh --skip-training    # emit manifests only
#   OUT_DIR=data/phase_corpus_v2 ./scripts/run_phase_corpus.sh
#
# Environment:
#   OUT_DIR          Output directory (default: data/phase_corpus)
#   BASE_MODEL       HF repo id (default: Qwen/Qwen3.5-9B)
#   PIPELINE_TIMEOUT Per-problem timeout in seconds (default: 300)

set -euo pipefail

OUT_DIR="${OUT_DIR:-data/phase_corpus}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.5-9B}"
PIPELINE_TIMEOUT="${PIPELINE_TIMEOUT:-300}"
EXTRA_FLAGS="${*}"

SCRIPT="uv run scripts/phase_corpus_producer.py"
COMMON_FLAGS="--out-dir ${OUT_DIR} --base-model ${BASE_MODEL} --pipeline-timeout ${PIPELINE_TIMEOUT} ${EXTRA_FLAGS}"

echo "=== Phase Corpus Production ==="
echo "    OUT_DIR=${OUT_DIR}"
echo "    BASE_MODEL=${BASE_MODEL}"
echo "    PIPELINE_TIMEOUT=${PIPELINE_TIMEOUT}s"
echo ""

# HumanEval — 164 problems (elementary)
echo "--- humaneval ---"
${SCRIPT} --benchmark humaneval ${COMMON_FLAGS}

# MBPP — 374 problems (basic)
echo "--- mbpp ---"
${SCRIPT} --benchmark mbpp ${COMMON_FLAGS}

# APPS — subsampled N=500, seed=42, stratified by difficulty
# The corpus producer's _load_problems() calls load_problems(benchmark,
# max_samples=500) which applies stratified sampling internally.
echo "--- apps (N=500, seed=42, stratified) ---"
${SCRIPT} --benchmark apps --max-problems 500 ${COMMON_FLAGS}

# BigCodeBench — 1140 problems (applied)
echo "--- bigcodebench ---"
${SCRIPT} --benchmark bigcodebench ${COMMON_FLAGS}

# DS-1000 — 1000 problems (data science)
echo "--- ds_1000 ---"
${SCRIPT} --benchmark ds_1000 ${COMMON_FLAGS}

# LiveCodeBench — 500 problems (competitive)
echo "--- livecodebench ---"
${SCRIPT} --benchmark livecodebench ${COMMON_FLAGS}

echo ""
echo "=== All benchmarks complete. Manifests in ${OUT_DIR}/manifests/ ==="
