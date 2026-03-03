"""Evolution router with stub 501 endpoints."""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from evolution_svc.schemas import (
    EvaluationRequest,
    EvolveRequest,
    PromoteRequest,
    PruneRequest,
)

router = APIRouter(tags=["evolution"])


@router.post("/evaluate")
async def evaluate_adapter(request: EvaluationRequest) -> JSONResponse:
    """Evaluate an adapter's performance. Not yet implemented."""
    return JSONResponse(status_code=501, content={"detail": "Not Implemented"})


@router.post("/evolve")
async def evolve_adapters(request: EvolveRequest) -> JSONResponse:
    """Evolve adapters via crossover/mutation. Not yet implemented."""
    return JSONResponse(status_code=501, content={"detail": "Not Implemented"})


@router.post("/promote")
async def promote_adapter(request: PromoteRequest) -> JSONResponse:
    """Promote an adapter to a higher tier. Not yet implemented."""
    return JSONResponse(status_code=501, content={"detail": "Not Implemented"})


@router.post("/prune")
async def prune_adapter(request: PruneRequest) -> JSONResponse:
    """Prune an underperforming adapter. Not yet implemented."""
    return JSONResponse(status_code=501, content={"detail": "Not Implemented"})
