#!/bin/bash
# Post-create script for rune devcontainer
set -eo pipefail

export PATH="$HOME/.local/bin:/usr/local/bin:$PATH"

# All interactive shell setup (PATH, env exports, aliases) is written to ONE
# file that BOTH bash and zsh source. The devcontainer installs zsh
# (common-utils) and mounts zsh_history, so the interactive shell may be zsh;
# writing only to ~/.bashrc would leave such shells without aliases or PATH.
DEVPOD_ENV="$HOME/.devpod-env"
touch "$DEVPOD_ENV"
for _rc in "$HOME/.bashrc" "$HOME/.zshrc"; do
  touch "$_rc"
  grep -q '\.devpod-env' "$_rc" 2>/dev/null \
    || echo '[ -f "$HOME/.devpod-env" ] && . "$HOME/.devpod-env"' >> "$_rc"
done

# ---- Durable, portable history persistence (symlinks) ---------------------
# Claude state and shell history live in ONE bind-mounted state dir
# (/opt/devbox-state, backed by /opt/workspace/.devcontainer-state on the host,
# which restic backs up and restores on launch). We symlink the home-dir paths
# into it instead of bind-mounting each file: a single dir mount never orphans
# when contents are rewritten, and the same scheme works on any host/cloud that
# presents the state dir. Result: ~/.claude (all repos' sessions) and shell
# history survive container rebuilds, repo switches, and region/cloud moves.
# Idempotent; migrates any pre-existing real file/dir into the store once.
STATE_DIR="/opt/devbox-state"
if [ -d "$STATE_DIR" ]; then
  mkdir -p "$STATE_DIR/claude" "$STATE_DIR/shell"
  touch "$STATE_DIR/shell/bash_history" "$STATE_DIR/shell/zsh_history"
  link_state() {
    local tgt="$1" link="$2"
    if [ -e "$link" ] && [ ! -L "$link" ]; then
      if [ -d "$link" ]; then cp -an "$link/." "$tgt/" 2>/dev/null || true
      else cp -an "$link" "$tgt" 2>/dev/null || true; fi
      rm -rf "$link"
    fi
    ln -sfn "$tgt" "$link"
  }
  link_state "$STATE_DIR/claude"             "$HOME/.claude"
  link_state "$STATE_DIR/claude.json"        "$HOME/.claude.json"
  link_state "$STATE_DIR/shell/bash_history" "$HOME/.bash_history"
  link_state "$STATE_DIR/shell/zsh_history"  "$HOME/.zsh_history"
  # ~/.claude.json (Claude's main config) is a sibling of ~/.claude, easy to
  # miss. If the persisted copy is empty/missing but Claude left a backup under
  # ~/.claude/backups, restore the newest one so Claude isn't unconfigured
  # ("Claude configuration file not found at /home/vscode/.claude.json").
  if [ ! -s "$STATE_DIR/claude.json" ]; then
    _bk="$(ls -1t "$STATE_DIR"/claude/backups/.claude.json.backup.* 2>/dev/null | head -1)"
    if [ -n "$_bk" ]; then cp "$_bk" "$STATE_DIR/claude.json" && echo "Restored ~/.claude.json from $_bk"; fi
  fi
  echo "Persistence: ~/.claude, ~/.claude.json + shell history -> $STATE_DIR (restic-backed, portable)"
else
  echo "WARNING: $STATE_DIR not mounted — history will NOT persist across rebuilds." >&2
fi

# Install Node.js LTS (for Claude Code CLI)
#
# Justification for the curl-pipe-bash below: official NodeSource setup script
# piped to sudo bash is the documented installation path for devcontainer
# bootstrap. File is a `.example` template — devs review before copying to
# `.devcontainer/post-create.sh`.
if ! command -v node &>/dev/null; then
  echo "Installing Node.js..."
  # nosemgrep: bash.curl.security.curl-pipe-bash.curl-pipe-bash
  curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -
  sudo apt-get install -y nodejs
fi

# Install uv (Python package manager)
if ! command -v uv &>/dev/null; then
  echo "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  grep -q 'HOME/.local/bin' "$DEVPOD_ENV" 2>/dev/null \
    || echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$DEVPOD_ENV"
fi

# npm: give the container user a writable global prefix + cache so global
# installs (and the tools they place on PATH) work WITHOUT sudo and are never
# left root-owned (a sudo `npm i -g` otherwise root-owns ~/.npm and the global
# node_modules, breaking later user installs with EACCES). PATH addition goes in
# ~/.devpod-env (sourced by bash and zsh).
export NPM_CONFIG_PREFIX="$HOME/.npm-global"
mkdir -p "$HOME/.npm-global" "$HOME/.npm"
sudo chown -R "$(id -u):$(id -g)" "$HOME/.npm-global" "$HOME/.npm" 2>/dev/null || true
export PATH="$HOME/.npm-global/bin:$PATH"
grep -q '.npm-global/bin' "$DEVPOD_ENV" 2>/dev/null \
  || echo 'export PATH="$HOME/.npm-global/bin:$PATH"' >> "$DEVPOD_ENV"
# Persist NPM_CONFIG_PREFIX too (not just PATH): at runtime, claude's auto-update
# shells out to npm, which without this falls back to the default root-owned
# global prefix and fails with "npm global folder isn't writable". Sourced by
# bash and zsh via ~/.devpod-env.
grep -q 'NPM_CONFIG_PREFIX' "$DEVPOD_ENV" 2>/dev/null \
  || echo 'export NPM_CONFIG_PREFIX="$HOME/.npm-global"' >> "$DEVPOD_ENV"

# Install Claude Code (user prefix — no sudo, stays user-owned)
if ! command -v claude &>/dev/null; then
  echo "Installing Claude Code..."
  npm install -g @anthropic-ai/claude-code
fi

# Install the infra-login wrapper inside the container so the `login` alias
# resolves. /usr/local/bin/infra-login is created by gpu-userdata on the HOST
# only — the container has a separate filesystem — so connect.sh copies the
# host's (already-substituted) wrapper into .devcontainer/ at launch. Install
# it if present; otherwise drop a stub that prints a clear hint instead of a
# confusing "No such file or directory".
if [ -f .devcontainer/infra-login ]; then
  sudo install -m 0755 .devcontainer/infra-login /usr/local/bin/infra-login
elif [ ! -x /usr/local/bin/infra-login ]; then
  sudo tee /usr/local/bin/infra-login >/dev/null <<'STUB'
#!/bin/bash
echo "infra-login was not injected into this container." >&2
echo "Re-launch with 'make devpod' from the infra repo to install it." >&2
exit 1
STUB
  sudo chmod +x /usr/local/bin/infra-login
fi

# infra-login clones the infra repo into /opt/workspace. On the host that dir
# is pre-created (chmod 775) by gpu-userdata, but the container filesystem is
# separate — create it writable by the container user so the clone step does
# not fail with "mkdir: cannot create directory '/opt/workspace': Permission
# denied".
sudo mkdir -p /opt/workspace
sudo chown "$(id -un)":"$(id -gn)" /opt/workspace

# Generic devpod convenience aliases — wanted in any workspace, not repo-specific.
#   login   → the infra-login bootstrap (AWS/HF/GitHub/Claude auth)
#   eclaude → claude with permission prompts skipped
# Written to ~/.devpod-env (sourced by bash AND zsh). Idempotent.
if ! grep -q "### devpod-aliases ###" "$DEVPOD_ENV" 2>/dev/null; then
  cat >> "$DEVPOD_ENV" <<'ALIASES'

### devpod-aliases ###
alias login='/usr/local/bin/infra-login'
alias eclaude='claude --dangerously-skip-permissions'
### end devpod-aliases ###
ALIASES
fi

# Log in to all dev services (AWS, HuggingFace, GitHub, Claude) via Secrets Manager.
# Finds the unified login script whether the repo is mounted at /workspaces/infra
# (devcontainer default) or elsewhere.
# Compute the repo root defensively — a failed `git rev-parse` under `set -e`
# would otherwise abort the whole post-create hook, and an empty substitution
# would produce a bogus absolute path like `/shared/scripts/login.sh`.
REPO_ROOT="$(git -C "$(pwd)" rev-parse --show-toplevel 2>/dev/null || true)"

CANDIDATES=(
  /workspaces/infra/shared/scripts/login.sh
  ./shared/scripts/login.sh
  ../infra/shared/scripts/login.sh
)
if [ -n "$REPO_ROOT" ]; then
  CANDIDATES+=("$REPO_ROOT/shared/scripts/login.sh")
fi

LOGIN_SCRIPT=""
for candidate in "${CANDIDATES[@]}"; do
  if [ -f "$candidate" ]; then
    LOGIN_SCRIPT="$candidate"
    break
  fi
done

if [ -n "$LOGIN_SCRIPT" ]; then
  bash "$LOGIN_SCRIPT" || echo "WARNING: login.sh failed — run 'make login' manually"
else
  echo "WARNING: login.sh not found — run 'make login' from the infra repo root"
fi

# The GitHub MCP server (claude-plugins-official) reads GITHUB_PERSONAL_ACCESS_TOKEN
# from the environment. Persist a RESOLVER, not the value: each shell resolves it
# from the user's gh login at startup. The token never lands on disk, so it is
# never captured in a golden-AMI snapshot (bake the fetcher, not the secret).
# For a shared, login-free token instead, point this at Secrets Manager:
#   export GITHUB_PERSONAL_ACCESS_TOKEN="$(aws secretsmanager get-secret-value \
#     --secret-id "$DEVBOX_PROJECT/$DEVBOX_ENV_NAME/github-mcp-pat" \
#     --query SecretString --output text --region "$AWS_DEFAULT_REGION" 2>/dev/null)"
# Rewrite any existing line (incl. a previously-baked literal value) to the resolver.
_gh_resolver='export GITHUB_PERSONAL_ACCESS_TOKEN="$(gh auth token 2>/dev/null)"'
if grep -q '^export GITHUB_PERSONAL_ACCESS_TOKEN=' "$DEVPOD_ENV" 2>/dev/null; then
  sed -i 's|^export GITHUB_PERSONAL_ACCESS_TOKEN=.*|'"$_gh_resolver"'|' "$DEVPOD_ENV"
else
  echo "$_gh_resolver" >> "$DEVPOD_ENV"
fi

# Discover artifact bucket and export training data path.
if command -v aws &>/dev/null; then
  ARTIFACT_BUCKET="$(aws s3 ls 2>/dev/null \
    | awk '{print $3}' | grep -- '-artifacts$' | head -1 || true)"
  if [ -n "$ARTIFACT_BUCKET" ]; then
    RUNE_TRAINING_DATA="s3://$ARTIFACT_BUCKET/training-data/github-pairs"
    # Rewrite-or-append to avoid duplicate lines on re-run.
    if grep -q '^export RUNE_TRAINING_DATA=' "$DEVPOD_ENV" 2>/dev/null; then
      sed -i "s|^export RUNE_TRAINING_DATA=.*|export RUNE_TRAINING_DATA=\"$RUNE_TRAINING_DATA\"|" "$DEVPOD_ENV"
    else
      echo "export RUNE_TRAINING_DATA=\"$RUNE_TRAINING_DATA\"" >> "$DEVPOD_ENV"
    fi
    export RUNE_TRAINING_DATA
    echo "Training data path: $RUNE_TRAINING_DATA"
  fi
fi

# Install project dependencies with GPU extras (guarded: only if the workspace
# is a uv project). Generic for any GPU repo; rune pulls flash-attn/bitsandbytes/
# trl + the Mamba fast-path (causal-conv1d, flash-linear-attention) via its extras.
if [ -f pyproject.toml ]; then
  echo "Installing dependencies (with GPU extras)..."
  # Narrow the from-source CUDA build to the host GPU's compute capability:
  # a default build targets ~9 archs (30-45 min); single-arch is 5-10 min.
  if command -v nvidia-smi &>/dev/null; then
    GPU_CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')"
    GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr -d '\n')"
    if [ -n "$GPU_CAP" ]; then
      export TORCH_CUDA_ARCH_LIST="$GPU_CAP"
      echo "Detected $GPU_NAME (compute capability $GPU_CAP) — TORCH_CUDA_ARCH_LIST=$GPU_CAP"
      grep -q '^export TORCH_CUDA_ARCH_LIST=' "$DEVPOD_ENV" 2>/dev/null \
        || echo "export TORCH_CUDA_ARCH_LIST=\"$GPU_CAP\"" >> "$DEVPOD_ENV"
    fi
  fi
  uv sync --extra gpu || { echo "ERROR: uv sync --extra gpu failed"; exit 1; }

  # Verify GPU stack works
  echo "Verifying GPU stack..."
  uv run python -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available'
print(f'GPU OK: {torch.cuda.get_device_name(0)}')
print(f'CUDA: {torch.version.cuda}')
print(f'PyTorch: {torch.__version__}')
try:
    import causal_conv1d_cuda  # noqa: F401
    import fla  # noqa: F401
    print('Mamba fast-path kernels loaded (causal-conv1d + flash-linear-attention)')
except ImportError as e:
    print(f'Mamba fast-path not available — torch fallback will be used: {e}')
" || { echo "WARNING: GPU verification failed — check CUDA drivers and torch installation"; }
fi
