"""FastAPI health sidecar for lora-server.

Runs on port 8001, separate from the vLLM server on port 8000.
Provides /health (liveness) and /ready (readiness) endpoints.
"""

from __future__ import annotations

import httpx
from fastapi import FastAPI

health_app = FastAPI(title="lora-server-health")


@health_app.get("/health")
async def health() -> dict[str, str]:
    """Liveness probe -- always returns healthy if the sidecar is running."""
    return {"status": "healthy", "service": "lora-server"}


@health_app.get("/ready")
async def ready() -> dict[str, str | bool]:
    """Readiness probe -- checks if vLLM is responding on localhost:8000."""
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get("http://localhost:8000/health", timeout=2.0)
            vllm_ready = resp.status_code == 200
    except Exception:
        vllm_ready = False

    return {
        "service": "lora-server",
        "vllm_ready": vllm_ready,
        "status": "ready" if vllm_ready else "not_ready",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(health_app, host="0.0.0.0", port=8001)
