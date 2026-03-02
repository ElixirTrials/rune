# Developer onboarding

Get the repo running locally and run tests. For architecture and workflows, see [Project Overview](index.md) and the [Testing Guide](testing-guide.md).

## Prerequisites

- **Python 3.12+** and **uv** (recommended: install via [uv](https://docs.astral.sh/uv/))
- **Node.js 20+** (for the HITL UI)
- **Docker & Docker Compose** (optional, for full stack)

## 1. Clone and install

```bash
git clone <repo-url>
cd <repo-name>

# Sync Python dependencies (root workspace)
uv sync --all-extras

# Frontend (HITL UI)
cd apps/hitl-ui && npm ci && cd ../..
```

## 2. Run the API

From repo root:

```bash
uv run uvicorn api_service.main:app --reload
```

API runs with hot reload. Health: `GET /health`, readiness: `GET /ready`. See `services/api-service/README.md` for env and DB setup.

## 3. Run the UI

```bash
cd apps/hitl-ui && npm run dev
```

Set `VITE_API_BASE_URL` if the API is not at the default. See `apps/hitl-ui/README.md`.

## 4. Run agents

Agent services and LangGraph usage are documented in the [Components Overview](components-overview.md) and per-service READMEs under `services/agent-a-service` and `services/agent-b-service`.

## 5. Run tests

**All checks (lint, typecheck, tests):**

```bash
make check
```

**Python only:**

```bash
uv run pytest services/api-service/tests libs/events-py/tests -q
```

**Frontend only:**

```bash
cd apps/hitl-ui && npm test -- --run
```

## 6. Build docs

```bash
make docs-build
make docs-serve   # serve at http://localhost:8000
```

## 7. Full stack with Docker

From repo root:

```bash
docker-compose -f infra/docker-compose.yml up --build
```

See `infra/README.md` for env vars and options.

## Next steps

- [Components Overview](components-overview.md) — services and libraries
- [Testing Guide](testing-guide.md) — testing patterns and examples
- [Documentation index](index.md) — architecture and quick start
