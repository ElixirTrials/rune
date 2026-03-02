# Infra — Deployment aid

This directory holds deployment-related config for local and (optionally) cloud runs.

## Docker Compose

Full stack: API, UI, Postgres, MLflow, Pub/Sub emulator.

**From repo root:**

```bash
docker-compose -f infra/docker-compose.yml up --build
```

Optional env (e.g. in `.env` at repo root or exported):

- `API_PORT`, `UI_PORT`, `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`
- `VITE_API_BASE_URL` (default `http://localhost:8000` for UI)
- `PUBSUB_EMULATOR_PORT`, `PUBSUB_PROJECT_ID`

## Future

- Terraform, K8s manifests, or env examples can live here.
- Keep `.github/` at repo root (GitHub expects it there).
