# evolution-svc

Adapter evaluation, evolution, promotion, and pruning service for Rune.

## Endpoints

| Method | Path | Status | Description |
|--------|------|--------|-------------|
| POST | `/evaluate` | Stub (501) | Evaluate adapter performance |
| POST | `/evolve` | Stub (501) | Evolve adapters via crossover/mutation |
| POST | `/promote` | Stub (501) | Promote high-fitness adapter |
| POST | `/prune` | Stub (501) | Archive low-fitness adapters |

## Current State

The REST endpoints are stub implementations returning 501. The actual evolution logic runs in `scripts/swarm_evolution.py` as part of the swarm orchestrator, performing:

- TIES/DARE merging of top adapters per task type
- Archiving adapters with fitness below threshold
- Generational lineage tracking (`generation = max(parents) + 1`)

## Request Schemas

- `EvaluationRequest` — adapter_id, task_type
- `EvolveRequest` — adapter_ids, task_type
- `PromoteRequest` — adapter_id
- `PruneRequest` — adapter_id or criteria

## Running

```bash
uv run uvicorn evolution_svc.main:app --reload
```
