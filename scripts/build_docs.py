"""Build documentation site using MkDocs.

Usage:
    uv run python scripts/build_docs.py build -f mkdocs.yml
    uv run python scripts/build_docs.py serve -f mkdocs.yml
"""

from __future__ import annotations

import subprocess
import sys


def main() -> None:
    """Proxy to mkdocs CLI with the given arguments."""
    args = sys.argv[1:] or ["build"]
    cmd = [sys.executable, "-m", "mkdocs"] + args
    result = subprocess.run(cmd, check=False)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
