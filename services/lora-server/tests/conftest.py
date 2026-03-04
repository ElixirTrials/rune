"""Pytest configuration for services/lora-server.

lora-server is a Dockerfile-only service (not a uv workspace member).
We add its src directory to sys.path so tests can import directly.
"""

import sys
from pathlib import Path

# Allow direct import of lora_server modules without workspace membership
_LORA_SERVER_SRC = Path(__file__).parent.parent
if str(_LORA_SERVER_SRC) not in sys.path:
    sys.path.insert(0, str(_LORA_SERVER_SRC))
