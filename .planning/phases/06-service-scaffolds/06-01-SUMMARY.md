---
phase: 06-service-scaffolds
plan: 01
subsystem: infra
tags: [vllm, lora, docker, fastapi, openai, multi-gpu]

# Dependency graph
requires:
  - phase: 05-foundation-libraries
    provides: "inference lib with AsyncOpenAI pattern, adapter-registry"
provides:
  - "services/lora-server/ directory with 6 files (Dockerfile-only service)"
  - "LoraServerConfig with PP=2/TP=1 enforcement and TP=2 ValueError guard"
  - "VLLMClient stub wrapping AsyncOpenAI"
  - "FastAPI health sidecar on port 8001"
affects: [06-service-scaffolds, 07-config-quality-gate]

# Tech tracking
tech-stack:
  added: [pyyaml, httpx]
  patterns: [Dockerfile-only service, config-dataclass-with-safety-valve, health-sidecar-pattern, openai-compatible-vllm-client]

key-files:
  created:
    - services/lora-server/Dockerfile
    - services/lora-server/startup.sh
    - services/lora-server/config.yaml
    - services/lora-server/config.py
    - services/lora-server/health.py
    - services/lora-server/vllm_client.py
  modified: []

key-decisions:
  - "lora-server is Dockerfile-only, not a uv workspace member"
  - "LoraServerConfig raises ValueError on TP=2 referencing vLLM bug #21471"
  - "VLLMClient wraps AsyncOpenAI, not direct vllm import"
  - "Health sidecar uses httpx for readiness check against vLLM"

patterns-established:
  - "Dockerfile-only service pattern: services that run in containers get a directory under services/ but NO pyproject.toml, NOT added to workspace members"
  - "Config safety valve: dataclass __post_init__ raises ValueError for known-dangerous configurations"
  - "Health sidecar: separate FastAPI app on distinct port with /health and /ready endpoints"

requirements-completed: [SVC-02]

# Metrics
duration: 2min
completed: 2026-03-03
---

# Phase 6 Plan 01: LoRA Server Scaffold Summary

**Dockerfile-only lora-server with PP=2/TP=1 config safety valve, FastAPI health sidecar, and AsyncOpenAI-based VLLMClient stub**

## Performance

- **Duration:** 2 min
- **Started:** 2026-03-03T07:18:28Z
- **Completed:** 2026-03-03T07:20:22Z
- **Tasks:** 2
- **Files created:** 6

## Accomplishments
- Created complete services/lora-server/ directory with 6 files as a Dockerfile-only service
- LoraServerConfig enforces PP=2/TP=1 with ValueError guard on TP=2 (vLLM bug #21471)
- VLLMClient wraps AsyncOpenAI for lightweight vLLM communication without GPU dependencies
- Health sidecar provides /health and /ready endpoints on port 8001 separate from vLLM on 8000

## Task Commits

Each task was committed atomically:

1. **Task 1: Create lora-server config, Dockerfile, and startup script** - `6d2de8c` (feat)
2. **Task 2: Create lora-server health sidecar and VLLMClient stub** - `7a389e5` (feat)

## Files Created/Modified
- `services/lora-server/config.yaml` - Server configuration: model, PP/TP sizes, quantization, ports
- `services/lora-server/config.py` - LoraServerConfig dataclass with TP=2 ValueError safety valve and from_yaml classmethod
- `services/lora-server/startup.sh` - Entrypoint launching health sidecar + vLLM API server with PP=2/TP=1
- `services/lora-server/Dockerfile` - Container build definition installing health sidecar dependencies
- `services/lora-server/health.py` - FastAPI health sidecar with /health (liveness) and /ready (readiness) endpoints
- `services/lora-server/vllm_client.py` - VLLMClient wrapping AsyncOpenAI with load_adapter and generate stubs

## Decisions Made
- Used httpx (already a project dependency) for health readiness check rather than adding urllib3
- VLLMClient stores api_key="not-needed-for-local-vllm" for AsyncOpenAI compatibility
- startup.sh uses exec for vLLM process to receive signals properly in container
- config.py from_yaml filters unknown keys silently for forward compatibility

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- lora-server scaffold is complete with all 6 files
- Ready for remaining 06-service-scaffolds plans (02-04)
- VLLMClient stubs ready for future implementation when vLLM dynamic LoRA API is integrated

## Self-Check: PASSED

All 7 files verified present. Both task commits (6d2de8c, 7a389e5) verified in git history.

---
*Phase: 06-service-scaffolds*
*Completed: 2026-03-03*
