"""Coding session router stubs."""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.get("")
async def list_sessions():
    """List coding sessions.

    Returns:
        JSONResponse with list of coding session records.

    Raises:
        HTTPException: 501 while not yet implemented.

    Example:
        >>> response = client.get("/sessions")
        >>> response.status_code
        200
        >>> isinstance(response.json(), list)
        True
        >>> 'session_id' in response.json()[0]
        True
    """
    return JSONResponse(status_code=501, content={"detail": "Not Implemented"})


@router.get("/{session_id}")
async def get_session(session_id: str):
    """Get coding session by ID.

    Args:
        session_id: Unique identifier for the coding session.

    Returns:
        JSONResponse with a single coding session record dict.

    Raises:
        HTTPException: 404 if session not found.
        HTTPException: 501 while not yet implemented.

    Example:
        >>> response = client.get("/sessions/test-session-456")
        >>> response.status_code
        200
        >>> 'session_id' in response.json()
        True
    """
    return JSONResponse(status_code=501, content={"detail": "Not Implemented"})


@router.post("")
async def create_session():
    """Create a new coding session.

    Returns:
        JSONResponse with the created coding session record.

    Raises:
        HTTPException: 422 if request body is invalid.
        HTTPException: 501 while not yet implemented.

    Example:
        >>> body = {"session_id": "new-session-001", "task_type": "bug-fix"}
        >>> response = client.post("/sessions", json=body)
        >>> response.status_code
        201
        >>> 'session_id' in response.json()
        True
    """
    return JSONResponse(status_code=501, content={"detail": "Not Implemented"})
