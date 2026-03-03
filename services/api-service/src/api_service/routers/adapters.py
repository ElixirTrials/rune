"""Adapter management router stubs."""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/adapters", tags=["adapters"])


@router.get("")
async def list_adapters():
    """List all stored adapters. Not yet implemented."""
    return JSONResponse(status_code=501, content={"detail": "Not Implemented"})


@router.get("/{adapter_id}")
async def get_adapter(adapter_id: str):
    """Get adapter by ID. Not yet implemented."""
    return JSONResponse(status_code=501, content={"detail": "Not Implemented"})


@router.post("")
async def create_adapter():
    """Store a new adapter. Not yet implemented."""
    return JSONResponse(status_code=501, content={"detail": "Not Implemented"})
