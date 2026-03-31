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
if [ -f pyproject.toml ]; then
  echo "Installing rune dependencies (with GPU extras)..."
  uv sync --extra gpu 2>/dev/null || echo "uv sync failed — run manually: uv sync --extra gpu"
fi
