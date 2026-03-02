#!/bin/bash

# Kill FastAPI/Uvicorn
pkill -f "uvicorn api_service.main:app" || true

# Kill Python http server (docs)
pkill -f "python -m http.server 8000" || true

# Kill any node processes (vite)
pkill -f "vite" || true

echo "Killed common development processes."
