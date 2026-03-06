---
phase: 19-inference-provider-abstraction
plan: 02
subsystem: infra
tags: [vllm, docker, docker-compose, lora, sqlite]

requires:
  - phase: 18-adapter-registry
    provides: adapter registry with SQLite storage that lora-server will share via rune_data volume

provides:
  - vLLM-native Dockerfile using vllm/vllm-openai:v0.16.0 base image with runtime LoRA updating
  - LoraServerConfig extended with max_lora_rank=64 and gpu_memory_utilization=0.80
  - Port-conflict-free docker-compose (lora-server on host 8100, api on host 8000)
  - Shared rune_data volume for SQLite access between api and lora-server

affects:
  - 19-inference-provider-abstraction
  - 20-agent-loop-trajectory
  - 21-qlora-training

tech-stack:
  added: [vllm/vllm-openai:v0.16.0 base image]
  patterns:
    - vLLM container runs on container port 8000; docker-compose maps to host port 8100 to avoid api conflict
    - Shared SQLite volume (rune_data:/data) mounted in both api and lora-server for cross-service access
    - Runtime LoRA updating enabled via VLLM_ALLOW_RUNTIME_LORA_UPDATING=True ENV in Dockerfile

key-files:
  created: []
  modified:
    - services/lora-server/Dockerfile
    - services/lora-server/config.py
    - infra/docker-compose.yml

key-decisions:
  - "vllm/vllm-openai:v0.16.0 is the base image; openai removed from pip install since it is bundled in the vLLM image"
  - "Container-internal port stays 8000; docker-compose host mapping changed from 8000 to 8100 to eliminate conflict"
  - "max_lora_rank=64 and gpu_memory_utilization=0.80 added to LoraServerConfig per VRAM OOM prevention decisions"

patterns-established:
  - "Port mapping pattern: container always listens on 8000; host port differentiated (api=8000, lora-server=8100)"
  - "Shared volume pattern: rune_data:/data used across services that need SQLite access"

requirements-completed: [INFRA-01, INFRA-02, INFRA-03]

duration: 8min
completed: 2026-03-05
---

# Phase 19 Plan 02: lora-server Infrastructure Update Summary

**vLLM-native Dockerfile (vllm/vllm-openai:v0.16.0) with runtime LoRA env var, VRAM-safe config defaults, and port-conflict-free docker-compose with shared SQLite volume**

## Performance

- **Duration:** 8 min
- **Started:** 2026-03-05T11:15:00Z
- **Completed:** 2026-03-05T11:23:00Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments

- Replaced `python:3.12-slim` base image with `vllm/vllm-openai:v0.16.0` and added `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True` ENV, enabling dynamic LoRA adapter loading at runtime
- Extended `LoraServerConfig` with `max_lora_rank=64` and `gpu_memory_utilization=0.80` per VRAM OOM prevention decisions logged in STATE.md
- Fixed docker-compose port conflict by remapping lora-server host port from 8000 to 8100; added shared `rune_data` volume for cross-service SQLite access

## Task Commits

Each task was committed atomically:

1. **Task 1: Update lora-server Dockerfile and config.py** - `c227e97` (feat)
2. **Task 2: Fix docker-compose port conflict and add shared SQLite volume** - `c812771` (feat)

**Plan metadata:** (docs commit follows)

## Files Created/Modified

- `services/lora-server/Dockerfile` - Base image changed to vllm/vllm-openai:v0.16.0; VLLM_ALLOW_RUNTIME_LORA_UPDATING=True added; openai removed from pip install
- `services/lora-server/config.py` - Added max_lora_rank=64 and gpu_memory_utilization=0.80 fields to LoraServerConfig dataclass
- `infra/docker-compose.yml` - lora-server port changed to 8100:8000; rune_data volume added to api and lora-server; DATABASE_URL updated to sqlite:////data/rune.db

## Decisions Made

- Removed `openai` from pip install in Dockerfile: the vLLM base image bundles it, so re-installing would be redundant
- Container-internal port stays 8000 (startup.sh and vLLM server unchanged); only the host-side port mapping changes to 8100
- `rune_data` volume provides shared filesystem for SQLite (`/data/rune.db`) across api and lora-server — enables the adapter registry (Phase 18) to be visible to lora-server at runtime

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- lora-server infrastructure ready: correct base image, runtime LoRA flag set, port conflict resolved
- Shared rune_data volume provides path for api-service and lora-server to share SQLite adapter registry
- Phase 19 Plan 03+ can proceed with InferenceProvider abstraction, VLLMProvider, and OllamaProvider implementation

## Self-Check: PASSED

- FOUND: services/lora-server/Dockerfile
- FOUND: services/lora-server/config.py
- FOUND: infra/docker-compose.yml
- FOUND: .planning/phases/19-inference-provider-abstraction/19-02-SUMMARY.md
- FOUND commit: c227e97 (Task 1)
- FOUND commit: c812771 (Task 2)

---
*Phase: 19-inference-provider-abstraction*
*Completed: 2026-03-05*
