#!/usr/bin/env bash
# ────────────────────────────────────────────────────────────────────────────
#  Rune — batch-mine GitHub PRs and upload to S3
#
#  Wraps mine_github.py --batch + aws s3 sync. Bumps defaults.max_prs in a
#  temp copy of instructions/mining_repos.json so the tracked file stays
#  unchanged.
#
#  Required env: GITHUB_TOKEN, AWS creds (env or instance role).
#
#  Usage:
#    scripts/mine_and_upload.sh                    # max_prs=200, default bucket
#    scripts/mine_and_upload.sh --max-prs 500
#    scripts/mine_and_upload.sh --skip-upload      # mine only
#    scripts/mine_and_upload.sh --bucket s3://my-bucket/pairs/
# ────────────────────────────────────────────────────────────────────────────
set -euo pipefail

MAX_PRS=200
BUCKET_URI="s3://elixirtrials-949678234935-eu-west-2-artifacts/training-data/github-pairs/"
SKIP_UPLOAD=0
CONFIG_SRC="instructions/mining_repos.json"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --max-prs)     MAX_PRS="$2"; shift 2;;
        --bucket)      BUCKET_URI="$2"; shift 2;;
        --config)      CONFIG_SRC="$2"; shift 2;;
        --skip-upload) SKIP_UPLOAD=1; shift;;
        -h|--help)     sed -n '2,17p' "$0"; exit 0;;
        *) echo "unknown flag: $1" >&2; exit 2;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

# ── prereq checks ──────────────────────────────────────────────────────────
for bin in uv jq; do
    command -v "$bin" >/dev/null || { echo "missing: $bin" >&2; exit 127; }
done
[[ -n "${GITHUB_TOKEN:-}" ]] || { echo "GITHUB_TOKEN not set" >&2; exit 1; }
[[ -f "$CONFIG_SRC" ]] || { echo "config not found: $CONFIG_SRC" >&2; exit 1; }

if [[ $SKIP_UPLOAD -eq 0 ]]; then
    command -v aws >/dev/null || { echo "missing: aws cli" >&2; exit 127; }
    aws sts get-caller-identity >/dev/null 2>&1 \
        || { echo "AWS credentials not usable" >&2; exit 1; }
fi

# ── prepare temp config + log dir ─────────────────────────────────────────
mkdir -p .tmp data/pairs
TS="$(date +%Y%m%d-%H%M%S)"
CONFIG_TMP=".tmp/mining_repos_${MAX_PRS}_${TS}.json"
LOG=".tmp/mine_${TS}.log"

jq --argjson n "$MAX_PRS" '.defaults.max_prs = $n' "$CONFIG_SRC" > "$CONFIG_TMP"

echo "Mining: max_prs=${MAX_PRS}, config=${CONFIG_TMP}"
echo "Output: data/pairs/    Log: ${LOG}"
echo

# ── mine (batch mode is PR-only; per-repo failures don't abort) ───────────
uv run python scripts/mine_github.py \
    --batch "$CONFIG_TMP" \
    --output-dir data/pairs/ \
    2>&1 | tee "$LOG"

echo
echo "Mining complete. Per-repo line counts:"
wc -l data/pairs/*.jsonl | sort -n

# ── upload ─────────────────────────────────────────────────────────────────
if [[ $SKIP_UPLOAD -eq 1 ]]; then
    echo "Skipping upload (--skip-upload)."
    exit 0
fi

echo
echo "Uploading to ${BUCKET_URI}"
aws s3 sync data/pairs/ "$BUCKET_URI" \
    --exclude '*' --include '*.jsonl' --size-only

echo "Done."
