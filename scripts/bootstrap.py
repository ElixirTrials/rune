"""Bootstrap: add all workspace source directories to sys.path.

Import this module at the top of any standalone script that needs to import
from workspace packages without a full ``uv run`` invocation.

Usage::

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from bootstrap import setup_path; setup_path()
"""

from __future__ import annotations

import sys
from pathlib import Path

_WORKSPACE_SRC_DIRS = [
    "services/rune-agent/src",
    "services/training-svc/src",
    "services/evolution-svc/src",
    "services/api-service/src",
    "libs/inference/src",
    "libs/shared/src",
    "libs/model-training/src",
    "libs/adapter-registry/src",
    "libs/evaluation/src",
]


def setup_path() -> None:
    """Insert all workspace src directories into ``sys.path`` (idempotent).

    Determines the workspace root as the parent of the ``scripts/`` directory,
    then prepends each ``src`` dir to ``sys.path`` if not already present.
    """
    root = Path(__file__).resolve().parent.parent
    for rel_src in _WORKSPACE_SRC_DIRS:
        p = str(root / rel_src)
        if p not in sys.path:
            sys.path.insert(0, p)
