#!/bin/bash
# Post-create script for rune devcontainer
set -e

export PATH="$HOME/.local/bin:/usr/local/bin:$PATH"

# Install Node.js LTS (for Claude Code CLI)
if ! command -v node &>/dev/null; then
  echo "Installing Node.js..."
  curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -
  sudo apt-get install -y nodejs
fi

# Install uv (Python package manager)
if ! command -v uv &>/dev/null; then
  echo "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
fi

# Install Claude Code
if ! command -v claude &>/dev/null; then
  echo "Installing Claude Code..."
  sudo npm install -g @anthropic-ai/claude-code
  echo ""
  echo "=== Claude Code installed ==="
  echo "Run 'claude login' to authenticate (opens a URL to paste in your browser)."
fi

# Install rune dependencies (including GPU extras: flash-attn, bitsandbytes, trl)
# Pull secrets and configure AWS environment
if command -v aws &>/dev/null; then
  HF_TOKEN="$(aws secretsmanager get-secret-value \
    --secret-id "elixirtrials/dev/huggingface-token" \
    --region eu-west-2 \
    --query 'SecretString' --output text 2>/dev/null || true)"
  if [ -n "$HF_TOKEN" ]; then
    echo "export HF_TOKEN=\"$HF_TOKEN\"" >> ~/.bashrc
    export HF_TOKEN
    echo "HuggingFace token loaded from Secrets Manager."
  fi

  # Discover artifact bucket and export training data path
  ARTIFACT_BUCKET="$(aws s3 ls 2>/dev/null \
    | awk '{print $3}' | grep -- '-artifacts$' | head -1 || true)"
  if [ -n "$ARTIFACT_BUCKET" ]; then
    RUNE_TRAINING_DATA="s3://$ARTIFACT_BUCKET/training-data/github-pairs"
    echo "export RUNE_TRAINING_DATA=\"$RUNE_TRAINING_DATA\"" >> ~/.bashrc
    export RUNE_TRAINING_DATA
    echo "Training data path: $RUNE_TRAINING_DATA"
  fi
fi

# -----------------------------------------------------------------------------
# `login` bootstrap wrapper + convenience aliases
#
# Installs /usr/local/bin/infra-login: first invocation runs `gh auth login
# --web` (device flow), clones the private ElixirTrials/infra repo, then runs
# the shared login.sh via `make login` (AWS instance role + Secrets Manager
# for HF, gh SSO for GitHub, SSO for Claude). Later runs `git pull --ff-only`
# then re-run. No stored PAT, no extra Secrets Manager entry.
# -----------------------------------------------------------------------------
sudo tee /usr/local/bin/infra-login >/dev/null <<'LOGIN_STUB'
#!/bin/bash
# infra-login: devpod bootstrap for AWS/HF/GitHub/Claude auth.
set -euo pipefail

PROJECT_NAME="elixirtrials"
ENVIRONMENT="dev"
AWS_REGION="eu-west-2"

# Prefer the shared bind-mounted workspace if writable; fall back to $HOME.
if [ -d /opt/workspace ] && [ -w /opt/workspace ]; then
  INFRA_DIR="/opt/workspace/infra"
else
  INFRA_DIR="$HOME/workspace/infra"
fi
TFVARS_DIR="$INFRA_DIR/providers/aws/environments"
TFVARS_FILE="$TFVARS_DIR/dev.tfvars"

log() { printf '\033[0;36m[infra-login]\033[0m %s\n' "$*"; }

write_tfvars() {
  # dev.tfvars is gitignored; synthesize so login.sh can read
  # project_name/environment/aws_region.
  mkdir -p "$TFVARS_DIR"
  cat > "$TFVARS_FILE" <<TFV
project_name = "$PROJECT_NAME"
environment  = "$ENVIRONMENT"
aws_region   = "$AWS_REGION"
TFV
}

clone_infra() {
  if ! gh auth status >/dev/null 2>&1; then
    log "First run — authenticating with GitHub (device flow)..."
    gh auth login --hostname github.com --git-protocol https --web
  fi
  local token
  token="$(gh auth token)"
  [ -n "$token" ] || { echo "[infra-login] gh auth token empty" >&2; exit 1; }
  log "Cloning ElixirTrials/infra to $INFRA_DIR ..."
  mkdir -p "$(dirname "$INFRA_DIR")"
  git clone --depth=1 \
    "https://x-access-token:$token@github.com/ElixirTrials/infra.git" \
    "$INFRA_DIR"
  # Strip the embedded token — future pulls use gh's credential helper.
  git -C "$INFRA_DIR" remote set-url origin \
    "https://github.com/ElixirTrials/infra.git"
}

if [ ! -d "$INFRA_DIR/.git" ]; then
  clone_infra
else
  git -C "$INFRA_DIR" pull --ff-only >/dev/null 2>&1 || true
fi

write_tfvars
gh auth setup-git >/dev/null 2>&1 || true
exec make -C "$INFRA_DIR" login
LOGIN_STUB
sudo chmod +x /usr/local/bin/infra-login

# Idempotent: append convenience aliases only if marker is absent.
if ! grep -q "### devpod-aliases ###" ~/.bashrc 2>/dev/null; then
  cat >> ~/.bashrc <<'ALIASES'

### devpod-aliases ###
alias login='/usr/local/bin/infra-login'
alias eclaude='claude --dangerously-skip-permissions'
### end devpod-aliases ###
ALIASES
fi

# Persistent shell history — flush after every command, 100k line ring.
# devcontainer.json bind-mounts /home/vscode/.bash_history from the host's
# /home/ubuntu/.bash_history (which is in restic's backup paths), so commands
# survive instance rollovers. Without these settings bash only flushes on
# shell exit, so most session commands are never written before terminate.
if ! grep -q "### persistent-history ###" ~/.bashrc 2>/dev/null; then
  cat >> ~/.bashrc <<'HISTORY'

### persistent-history ###
HISTSIZE=100000
HISTFILESIZE=100000
HISTTIMEFORMAT='%F %T '
HISTCONTROL=ignoredups
shopt -s histappend 2>/dev/null
case "${PROMPT_COMMAND:-}" in
  *"history -a"*) ;;
  *) PROMPT_COMMAND="history -a${PROMPT_COMMAND:+; $PROMPT_COMMAND}" ;;
esac
export HISTSIZE HISTFILESIZE HISTTIMEFORMAT HISTCONTROL PROMPT_COMMAND
### end persistent-history ###
HISTORY
fi

# Install rune dependencies (including GPU extras: flash-attn, bitsandbytes, trl)
if [ -f pyproject.toml ]; then
  echo "Installing rune dependencies (with GPU extras)..."
  uv sync --extra gpu || { echo "ERROR: uv sync --extra gpu failed"; exit 1; }

  # Verify GPU stack works
  echo "Verifying GPU stack..."
  uv run python -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available'
print(f'GPU OK: {torch.cuda.get_device_name(0)}')
print(f'CUDA: {torch.version.cuda}')
print(f'PyTorch: {torch.__version__}')
" || { echo "WARNING: GPU verification failed — check CUDA drivers and torch installation"; }
fi
