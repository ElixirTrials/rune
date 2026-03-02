#!/bin/bash
# Run all code quality checks across the monorepo
# Usage: ./scripts/check-all.sh [--fix] [--with-docs]
#
# Options:
#   --fix        Auto-fix issues where possible (ruff, eslint)
#   --with-docs  Include documentation build check (slower)

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$ROOT_DIR"

FIX_MODE=false
WITH_DOCS=false
for arg in "$@"; do
    case $arg in
        --fix) FIX_MODE=true ;;
        --with-docs) WITH_DOCS=true ;;
    esac
done

echo "=============================================="
echo "Running all code quality checks"
echo "=============================================="
echo ""

FAILED=0

# Function to run a check
run_check() {
    local name="$1"
    local cmd="$2"

    echo -e "${YELLOW}► Running $name...${NC}"
    if eval "$cmd"; then
        echo -e "${GREEN}✓ $name passed${NC}"
        echo ""
    else
        echo -e "${RED}✗ $name failed${NC}"
        echo ""
        FAILED=1
    fi
}

# Python checks
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Python Checks"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if $FIX_MODE; then
    run_check "Ruff (lint + fix)" "uv run ruff check . --fix"
    run_check "Ruff (format)" "uv run ruff format ."
else
    run_check "Ruff (lint)" "uv run ruff check ."
    run_check "Ruff (format check)" "uv run ruff format . --check"
fi

run_check "Mypy (type check)" "uv run mypy libs/shared/src services/api-service/src libs/inference/src services/agent-a-service/src services/agent-b-service/src libs/data-pipeline/src libs/evaluation/src libs/events-py/src libs/model-training/src"

run_check "Pytest" "uv run pytest services/api-service/tests libs/events-py/tests -q"

# Frontend checks
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Frontend Checks (hitl-ui)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

HITL_UI_DIR="$ROOT_DIR/apps/hitl-ui"

if [[ -d "$HITL_UI_DIR/node_modules" ]]; then
    cd "$HITL_UI_DIR"

    if $FIX_MODE; then
        run_check "Biome (fix)" "npm run lint:fix"
    else
        run_check "Biome" "npm run lint"
    fi

    run_check "TypeScript" "npx tsc --noEmit"
    run_check "Vitest" "npm test -- --run"

    cd "$ROOT_DIR"
else
    echo -e "${YELLOW}⚠ Skipping frontend checks - node_modules not installed${NC}"
    echo "  Run: cd apps/hitl-ui && npm install"
    echo ""
fi

# Documentation checks (optional, enabled with --with-docs)
if $WITH_DOCS; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Documentation Checks"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""

    # Generate required doc files first
    echo -e "${YELLOW}► Generating doc prerequisites...${NC}"
    uv run python scripts/generate_components_overview.py > /dev/null 2>&1
    uv run python scripts/update_root_navigation.py > /dev/null 2>&1
    uv run --project services/api-service python services/api-service/scripts/export_openapi.py > /dev/null 2>&1
    echo -e "${GREEN}✓ Doc prerequisites generated${NC}"
    echo ""

    run_check "MkDocs Build (strict)" "uv run python scripts/build_docs.py build -f mkdocs.yml --strict"
else
    echo -e "${YELLOW}⚠ Skipping documentation checks (use --with-docs to include)${NC}"
    echo ""
fi

# Summary
echo "=============================================="
if [[ $FAILED -eq 0 ]]; then
    echo -e "${GREEN}All checks passed!${NC}"
else
    echo -e "${RED}Some checks failed${NC}"
    exit 1
fi
