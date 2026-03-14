"""FastAPI health sidecar for lora-server.

Runs on port 8001, separate from the vLLM server on port 8000.
Provides /health (liveness) and /ready (readiness) endpoints.
"""

from __future__ import annotations

import logging

import httpx
from fastapi import FastAPI

health_app = FastAPI(title="lora-server-health")
logger = logging.getLogger(__name__)


async def check_vllm_ready(base_url: str = "http://localhost:8000") -> bool:
    """Check if the vLLM server is responding and ready to serve requests.

    Sends a GET request to the vLLM server's /health endpoint to determine
    if the inference engine is operational.

    Args:
        base_url: Base URL of the vLLM server. Defaults to localhost:8000.

    Returns:
        True if vLLM responded with HTTP 200, False otherwise.

    Raises:
        No exceptions raised -- connection failures return False.

    Example:
        >>> ready = await check_vllm_ready()
        >>> isinstance(ready, bool)
        True
    """
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{base_url}/health", timeout=2.0)
            return resp.status_code == 200
    except Exception as e:
        logger.debug("vLLM health check failed for %s: %s", base_url, e)
        return False


@health_app.get("/health")
async def health() -> dict[str, str]:
    """Liveness probe -- always returns healthy if the sidecar is running."""
    return {"status": "healthy", "service": "lora-server"}


@health_app.get("/ready")
async def ready() -> dict[str, str | bool]:
    """Readiness probe -- checks if vLLM is responding on localhost:8000."""
    vllm_ready = await check_vllm_ready()
    return {
        "service": "lora-server",
        "vllm_ready": vllm_ready,
        "status": "ready" if vllm_ready else "not_ready",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(health_app, host="0.0.0.0", port=8001)
