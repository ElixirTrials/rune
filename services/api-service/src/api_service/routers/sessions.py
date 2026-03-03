"""Coding session router stubs."""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.get("")
async def list_sessions():
    """List coding sessions. Not yet implemented."""
    return JSONResponse(status_code=501, content={"detail": "Not Implemented"})


@router.get("/{session_id}")
async def get_session(session_id: str):
    """Get coding session by ID. Not yet implemented."""
    return JSONResponse(status_code=501, content={"detail": "Not Implemented"})


@router.post("")
async def create_session():
    """Create a new coding session. Not yet implemented."""
    return JSONResponse(status_code=501, content={"detail": "Not Implemented"})
