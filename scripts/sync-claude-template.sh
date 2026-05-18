#!/usr/bin/env bash
# Sync ElixirTrials Claude infrastructure from the canonical template repo into this repo.
#
# Manifest-aware: only files listed in .claude/.template-manifest are touched.
# Per-repo skills, settings additions, and custom commands SURVIVE every sync.
#
# - Listed files                 → overwritten
# - Listed directories (with /)  → contents synced WITHOUT --delete (additive)
# - .claude/settings.json        → deep-merged (local hooks + MCP servers preserved)
# - PRODUCT.md                   → preserved unless missing or all-stub
# - Anything else in .claude/    → left alone
#
# Usage:
#   scripts/sync-claude-template.sh                  # standard sync
#   scripts/sync-claude-template.sh --init           # also stamps PRODUCT.md if missing/stub
#   scripts/sync-claude-template.sh --dry-run        # preview, no writes
#   TEMPLATE_SRC=path/to/template scripts/sync-claude-template.sh
#   TEMPLATE_GIT=git@github.com:org/template.git scripts/sync-claude-template.sh
set -euo pipefail

DRY_RUN=0
INIT=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    --init)    INIT=1 ;;
    -h|--help) sed -n '2,18p' "$0"; exit 0 ;;
    *) echo "Unknown arg: $arg"; exit 1 ;;
  esac
done

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

# ----- Resolve template source ------------------------------------------------
SRC=""
if [[ -n "${TEMPLATE_SRC:-}" && -d "$TEMPLATE_SRC" ]]; then
  SRC="$TEMPLATE_SRC"
elif [[ -d "../Template" ]]; then
  SRC="$(cd ../Template && pwd)"
elif [[ -n "${TEMPLATE_GIT:-}" ]]; then
  SRC="$(mktemp -d)"
  trap 'rm -rf "$SRC"' EXIT
  echo "Cloning $TEMPLATE_GIT → $SRC"
  git clone --depth 1 "$TEMPLATE_GIT" "$SRC" >/dev/null
else
  echo "Error: set TEMPLATE_SRC=<path> or TEMPLATE_GIT=<url>, or place template at ../Template"
  exit 1
fi

MANIFEST="$SRC/.claude/.template-manifest"
if [[ ! -f "$MANIFEST" ]]; then
  echo "Error: template at $SRC has no .claude/.template-manifest — refusing to sync."
  echo "  (Older template versions wiped local skills via rsync --delete; we now require a manifest.)"
  exit 1
fi

echo "Source:  $SRC"
echo "Target:  $REPO_ROOT"
echo "Manifest:  $(grep -cv '^[[:space:]]*\(#\|$\)' "$MANIFEST") entries"
echo

run() { if [[ $DRY_RUN -eq 1 ]]; then echo "[dry] $*"; else eval "$@"; fi; }

# ----- One-time safety: warn about local-only skills -------------------------
# Compare every .claude/skills/<name>/SKILL.md in the repo against the manifest.
# Anything not listed is a per-repo skill the user added — log it so they know
# we're preserving it (not silently ignoring it).
LOCAL_ONLY_SKILLS=""
if [[ -d "$REPO_ROOT/.claude/skills" ]]; then
  while IFS= read -r local_skill; do
    rel="${local_skill#$REPO_ROOT/}"
    if ! grep -qxF "$rel" "$MANIFEST"; then
      LOCAL_ONLY_SKILLS="$LOCAL_ONLY_SKILLS  $rel\n"
    fi
  done < <(find "$REPO_ROOT/.claude/skills" -mindepth 2 -maxdepth 2 -name SKILL.md 2>/dev/null)
fi
if [[ -n "$LOCAL_ONLY_SKILLS" ]]; then
  echo "Repo-local skills (preserved by manifest, not touched by sync):"
  printf "$LOCAL_ONLY_SKILLS"
  echo
fi

# ----- Walk the manifest -----------------------------------------------------
echo "→ Syncing per manifest"
while IFS= read -r entry; do
  # Strip comments and whitespace
  entry="${entry%%#*}"
  entry="$(echo "$entry" | tr -d '[:space:]')"
  [[ -z "$entry" ]] && continue

  src_path="$SRC/$entry"
  dst_path="$REPO_ROOT/$entry"

  if [[ "$entry" == */ ]]; then
    # Directory entry — sync contents WITHOUT --delete
    if [[ ! -d "$src_path" ]]; then
      echo "  skip (missing in template): $entry"
      continue
    fi
    run "mkdir -p '$dst_path'"
    run "rsync -a '$src_path' '$dst_path'"
    echo "  synced dir: $entry (additive)"
  else
    # File entry — overwrite
    if [[ ! -f "$src_path" ]]; then
      echo "  skip (missing in template): $entry"
      continue
    fi
    # PRODUCT.md special handling
    if [[ "$entry" == "PRODUCT.md" ]]; then
      if [[ ! -f "$dst_path" ]]; then
        run "cp '$src_path' '$dst_path'"
        echo "  created: PRODUCT.md (was missing)"
      elif [[ $INIT -eq 1 ]] && ! grep -qv '<!-- TODO' "$dst_path" 2>/dev/null; then
        run "cp '$src_path' '$dst_path'"
        echo "  refreshed: PRODUCT.md (--init, was all-stub)"
      else
        echo "  preserved: PRODUCT.md (has content)"
      fi
      continue
    fi
    run "mkdir -p '$(dirname "$dst_path")'"
    run "cp '$src_path' '$dst_path'"
    echo "  overwrote: $entry"
  fi
done < "$MANIFEST"

# ----- settings.json deep-merge ---------------------------------------------
echo
echo "→ Merging .claude/settings.json"
SRC_SETTINGS="$SRC/.claude/settings.json"
DST_SETTINGS="$REPO_ROOT/.claude/settings.json"
if [[ -f "$SRC_SETTINGS" && -f "$DST_SETTINGS" ]]; then
  if [[ $DRY_RUN -eq 1 ]]; then
    echo "[dry] would deep-merge settings.json (local hooks + MCP preserved)"
  else
    python3 - "$SRC_SETTINGS" "$DST_SETTINGS" <<'PY'
import json, sys
src = json.load(open(sys.argv[1]))
dst_path = sys.argv[2]
dst = json.load(open(dst_path))

def merge(template, local):
    if isinstance(template, dict) and isinstance(local, dict):
        out = dict(template)
        for k, v in local.items():
            out[k] = merge(template.get(k), v) if k in template else v
        return out
    if isinstance(template, list) and isinstance(local, list):
        seen, out = set(), []
        for item in template + local:
            key = json.dumps(item, sort_keys=True)
            if key not in seen:
                seen.add(key); out.append(item)
        return out
    return local if local is not None else template

merged = merge(src, dst)
with open(dst_path, "w") as f:
    json.dump(merged, f, indent=2)
    f.write("\n")
print("  merged: .claude/settings.json")
PY
  fi
elif [[ -f "$SRC_SETTINGS" ]]; then
  run "mkdir -p '$REPO_ROOT/.claude'"
  run "cp '$SRC_SETTINGS' '$DST_SETTINGS'"
  echo "  copied: .claude/settings.json (was missing locally)"
fi

# ----- Stamp version --------------------------------------------------------
if [[ $DRY_RUN -eq 0 ]]; then
  TPL_SHA="$(git -C "$SRC" rev-parse --short HEAD 2>/dev/null || echo unknown)"
  cat > "$REPO_ROOT/.claude/.template-version" <<EOF
template_sha: $TPL_SHA
synced_at: $(date -u +%Y-%m-%dT%H:%M:%SZ)
synced_from: $SRC
EOF
  echo
  echo "  stamped: .claude/.template-version ($TPL_SHA)"
fi

echo
echo "Done. Review with: git status && git diff"
