"""Adapter management router stubs."""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/adapters", tags=["adapters"])


@router.get("")
async def list_adapters():
    """List all stored adapters.

    Returns:
        JSONResponse with list of adapter records.

    Raises:
        HTTPException: 501 while not yet implemented.

    Example:
        >>> response = client.get("/adapters")
        >>> response.status_code
        200
        >>> isinstance(response.json(), list)
        True
        >>> 'adapter_id' in response.json()[0]
        True
    """
    return JSONResponse(status_code=501, content={"detail": "Not Implemented"})


@router.get("/{adapter_id}")
async def get_adapter(adapter_id: str):
    """Get adapter by ID.

    Args:
        adapter_id: Unique identifier for the adapter.

    Returns:
        JSONResponse with a single adapter record dict.

    Raises:
        HTTPException: 404 if adapter not found.
        HTTPException: 501 while not yet implemented.

    Example:
        >>> response = client.get("/adapters/test-adapter-123")
        >>> response.status_code
        200
        >>> 'adapter_id' in response.json()
        True
    """
    return JSONResponse(status_code=501, content={"detail": "Not Implemented"})


@router.post("")
async def create_adapter():
    """Store a new adapter.

    Returns:
        JSONResponse with the created adapter record.

    Raises:
        HTTPException: 422 if request body is invalid.
        HTTPException: 501 while not yet implemented.

    Example:
        >>> body = {"adapter_id": "new-adapter-001", "task_type": "bug-fix"}
        >>> response = client.post("/adapters", json=body)
        >>> response.status_code
        201
        >>> 'adapter_id' in response.json()
        True
    """
    return JSONResponse(status_code=501, content={"detail": "Not Implemented"})
