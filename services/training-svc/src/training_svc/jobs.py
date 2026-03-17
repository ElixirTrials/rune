"""In-memory training job status tracking.

Module-level JOB_STORE dict shared across all request handlers.
State is lost on service restart — acceptable for single-user local MVP.

All mutations to JOB_STORE must be made while holding ``_JOB_STORE_LOCK``
to prevent race conditions when background threads update job status while
FastAPI request handlers are reading it.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Optional

JOB_STORE: dict[str, "JobStatus"] = {}
_JOB_STORE_LOCK = threading.Lock()


@dataclass
class JobStatus:
    """Training job status tracker."""

    job_id: str
    status: str  # "queued" | "running" | "completed" | "failed"
    adapter_id: Optional[str] = None
    error: Optional[str] = None
