#!/bin/bash
# Post-create script for rune devcontainer
set -e

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
  npm install -g @anthropic-ai/claude-code
  echo ""
  echo "=== Claude Code installed ==="
  echo "Run 'claude login' to authenticate (opens a URL to paste in your browser)."
fi

# Install rune dependencies
if [ -f pyproject.toml ]; then
  echo "Installing rune dependencies..."
  uv sync --all-extras 2>/dev/null || true
fi
