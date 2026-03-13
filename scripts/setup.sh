#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
#  Rune — one-command dev environment setup
#
#  Usage:
#    bash setup.sh                          # interactive
#    RUNE_DIR=~/projects/rune bash setup.sh # custom location
#    curl -fsSL <raw-url>/setup.sh | bash   # remote (non-interactive)
# ─────────────────────────────────────────────────────────────
set -euo pipefail

# ── Config ──────────────────────────────────────────────────
REPO="ElixirTrials/rune"
RUNE_DIR="${RUNE_DIR:-$HOME/rune}"
REQUIRED_PYTHON="3.12"

# ── Colors (respect NO_COLOR: https://no-color.org) ─────────
if [[ -z "${NO_COLOR:-}" ]] && [[ -t 2 ]]; then
    BOLD='\033[1m'    RESET='\033[0m'
    RED='\033[0;31m'  GREEN='\033[0;32m'
    YELLOW='\033[0;33m' CYAN='\033[0;36m'
else
    BOLD='' RESET='' RED='' GREEN='' YELLOW='' CYAN=''
fi

# ── Logging ─────────────────────────────────────────────────
info()  { printf "${GREEN}[OK]${RESET}  %s\n" "$*" >&2; }
warn()  { printf "${YELLOW}[!!]${RESET}  %s\n" "$*" >&2; }
fail()  { printf "${RED}[FAIL]${RESET} %s\n" "$*" >&2; exit 1; }
step()  { printf "\n${BOLD}${CYAN}▸ %s${RESET}\n" "$*" >&2; }

# ── Banner ──────────────────────────────────────────────────
printf >&2 "\n${BOLD}${CYAN}"
cat >&2 <<'BANNER'
  ┌─────────────────────────────────────────────┐
  │               R U N E                       │
  │   AI agent + LoRA training + swarm infra    │
  │   github.com/ElixirTrials/rune              │
  └─────────────────────────────────────────────┘
BANNER
printf >&2 "${RESET}"
printf >&2 "  This script will install system deps, authenticate with\n"
printf >&2 "  GitHub, clone the repo, and set up the Python environment.\n\n"
printf >&2 "  Install location: ${BOLD}%s${RESET}\n\n" "$RUNE_DIR"

# Confirm if running interactively
if [[ -t 0 ]]; then
    read -rp "  Continue? [Y/n] " answer
    case "${answer:-Y}" in
        [Yy]*|"") ;;
        *) echo "Aborted." >&2; exit 0 ;;
    esac
fi

# ── OS Detection ────────────────────────────────────────────
step "Detecting operating system"

OS="$(uname -s)"
DISTRO=""

case "$OS" in
    Darwin)
        info "macOS detected"
        ;;
    Linux)
        if [[ -f /etc/os-release ]]; then
            # shellcheck source=/dev/null
            . /etc/os-release
            DISTRO="${ID:-unknown}"
            info "Linux detected ($PRETTY_NAME)"
        else
            DISTRO="unknown"
            warn "Linux detected but /etc/os-release missing — some installs may fail"
        fi
        ;;
    *)
        fail "Unsupported OS: $OS (this script supports macOS and Linux)"
        ;;
esac

# ── Helper: check if a command exists ───────────────────────
has() { command -v "$1" &>/dev/null; }

# ── System Packages ─────────────────────────────────────────
step "Installing system packages"

if [[ "$OS" == "Darwin" ]]; then
    if ! has brew; then
        warn "Homebrew not found — installing..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        # Add brew to PATH for Apple Silicon
        if [[ -f /opt/homebrew/bin/brew ]]; then
            eval "$(/opt/homebrew/bin/brew shellenv)"
        fi
        info "Homebrew installed"
    else
        info "Homebrew already installed"
    fi

    for pkg in git curl; do
        if has "$pkg"; then
            info "$pkg already installed"
        else
            brew install "$pkg"
            info "$pkg installed"
        fi
    done

elif [[ "$DISTRO" == "ubuntu" || "$DISTRO" == "debian" ]]; then
    NEEDED=()
    for pkg in git curl build-essential libssl-dev; do
        if dpkg -s "$pkg" &>/dev/null; then
            info "$pkg already installed"
        else
            NEEDED+=("$pkg")
        fi
    done
    if [[ ${#NEEDED[@]} -gt 0 ]]; then
        sudo apt-get update -qq
        sudo apt-get install -y "${NEEDED[@]}"
        info "Installed: ${NEEDED[*]}"
    fi
else
    warn "Skipping system package install — please ensure git and curl are available"
fi

# ── GitHub CLI ──────────────────────────────────────────────
step "Setting up GitHub CLI"

if has gh; then
    info "gh already installed ($(gh --version | head -1))"
else
    if [[ "$OS" == "Darwin" ]]; then
        brew install gh
        info "gh installed via Homebrew"

    elif [[ "$DISTRO" == "ubuntu" || "$DISTRO" == "debian" ]]; then
        (type -p wget >/dev/null || (sudo apt update && sudo apt-get install wget -y))
        sudo mkdir -p -m 755 /etc/apt/keyrings
        out=$(mktemp)
        wget -qO "$out" https://cli.github.com/packages/githubcli-archive-keyring.gpg
        sudo tee /etc/apt/keyrings/githubcli-archive-keyring.gpg < "$out" >/dev/null
        rm -f "$out"
        sudo chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg
        echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | \
            sudo tee /etc/apt/sources.list.d/github-cli.list >/dev/null
        sudo apt-get update -qq
        sudo apt-get install -y gh
        info "gh installed via apt"
    else
        fail "Cannot auto-install gh on this distro — install manually: https://cli.github.com"
    fi
fi

# ── GitHub Authentication ───────────────────────────────────
step "Authenticating with GitHub"

if gh auth status &>/dev/null; then
    info "Already authenticated with GitHub"
else
    printf >&2 "  You'll be prompted to log in. Choose ${BOLD}HTTPS${RESET} and follow the\n"
    printf >&2 "  browser flow, or paste a personal access token.\n\n"
    gh auth login
    if gh auth status &>/dev/null; then
        info "GitHub authentication successful"
    else
        fail "GitHub authentication failed — re-run this script to try again"
    fi
fi

# ── Clone Repository ───────────────────────────────────────
step "Cloning repository"

if [[ -d "$RUNE_DIR/.git" ]]; then
    info "Repo already exists at $RUNE_DIR — pulling latest..."
    git -C "$RUNE_DIR" pull --ff-only || warn "Pull failed (you may have local changes) — continuing with existing code"
else
    gh repo clone "$REPO" "$RUNE_DIR"
    info "Cloned to $RUNE_DIR"
fi

cd "$RUNE_DIR"

# ── Install uv ─────────────────────────────────────────────
step "Installing uv (Python package manager)"

if has uv; then
    info "uv already installed ($(uv --version))"
else
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Make uv available in this session
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    if has uv; then
        info "uv installed ($(uv --version))"
    else
        fail "uv install succeeded but binary not found on PATH"
    fi
fi

# ── Python + Dependencies ──────────────────────────────────
step "Installing Python $REQUIRED_PYTHON and project dependencies"

uv python install "$REQUIRED_PYTHON"
info "Python $REQUIRED_PYTHON ready"

uv sync
info "All workspace dependencies installed"

# ── Smoke Test ──────────────────────────────────────────────
step "Running smoke tests"

SMOKE_PASS=true

if uv run python -c "from shared.rune_models import TaskStatus; print('  shared lib:', TaskStatus.PENDING.value)" 2>&1; then
    info "shared library OK"
else
    warn "shared library import failed (may need GPU deps)"
    SMOKE_PASS=false
fi

if uv run python -c "from inference.provider import InferenceProvider; print('  inference provider ABC loaded')" 2>&1; then
    info "inference library OK"
else
    warn "inference library import failed (may need GPU deps)"
    SMOKE_PASS=false
fi

if [[ "$SMOKE_PASS" == true ]]; then
    info "All smoke tests passed"
fi

# ── Advisory Checks ────────────────────────────────────────
step "Checking optional tools"

if has docker; then
    info "Docker found ($(docker --version | cut -d' ' -f3 | tr -d ','))"
else
    warn "Docker not found — install it to use the full stack (infra/docker-compose.yml)"
fi

if has nvidia-smi; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true)
    if [[ -n "$GPU_INFO" ]]; then
        info "NVIDIA GPU detected: $GPU_INFO"
    fi
else
    if [[ "$OS" == "Linux" ]]; then
        warn "nvidia-smi not found — GPU training/inference requires NVIDIA drivers + CUDA"
    fi
fi

# ── Done ────────────────────────────────────────────────────
printf >&2 "\n"
printf >&2 "${BOLD}${GREEN}"
cat >&2 <<DONE
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Rune is ready!   ${RUNE_DIR}
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DONE
printf >&2 "${RESET}"
cat >&2 <<DONE
  cd ${RUNE_DIR}

  Run tests:       uv run pytest
  Lint:            uv run ruff check .
  Type check:      uv run mypy libs/ services/ scripts/ --ignore-missing-imports
  Format:          uv run ruff format .
  Full stack:      docker compose -f infra/docker-compose.yml up --build

DONE
