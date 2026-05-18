#!/usr/bin/env bash
# Wire .githooks/ as the active git hooks directory for this repo.
#
# Idempotent — safe to re-run. Only sets git config; doesn't copy or symlink files.
# This way the hooks stay tracked in the repo (visible, reviewable, sync-able)
# instead of living in untracked .git/hooks/.
#
# After running once: every git commit triggers .githooks/post-commit which
# incrementally refreshes the code-review-graph index (no-op if CRG not installed).
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || {
  echo "Error: not inside a git repo."
  exit 1
}
cd "$REPO_ROOT"

if [[ ! -d .githooks ]]; then
  echo "Error: no .githooks/ directory in $REPO_ROOT — sync from the template first."
  exit 1
fi

# Ensure all hook files are executable
chmod +x .githooks/* 2>/dev/null || true

CURRENT="$(git config --get core.hooksPath || echo '')"
if [[ "$CURRENT" == ".githooks" ]]; then
  echo "✓ git core.hooksPath already set to .githooks"
else
  git config core.hooksPath .githooks
  echo "✓ Set git core.hooksPath = .githooks"
  if [[ -n "$CURRENT" ]]; then
    echo "  (was: $CURRENT)"
  fi
fi

echo
echo "Active hooks:"
ls -1 .githooks/ | sed 's/^/  /'
echo
echo "Done. Future commits will run these hooks."
echo "To disable: git config --unset core.hooksPath"
