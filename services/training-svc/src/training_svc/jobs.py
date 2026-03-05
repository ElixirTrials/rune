"""In-memory training job status tracking.

Module-level JOB_STORE dict shared across all request handlers.
State is lost on service restart — acceptable for single-user local MVP.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

JOB_STORE: dict[str, "JobStatus"] = {}


@dataclass
class JobStatus:
    """Training job status tracker."""

    job_id: str
    status: str  # "queued" | "running" | "completed" | "failed"
    adapter_id: Optional[str] = None
    error: Optional[str] = None
