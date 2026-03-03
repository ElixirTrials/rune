---
phase: 13-test-infrastructure
plan: "02"
subsystem: test-infrastructure
tags: [testing, conftest, fixtures, pytest, fastapi, vllm]
dependency_graph:
  requires: [13-01]
  provides: [component-conftest-files, mock-vllm-client, service-test-clients]
  affects: [14-api-service-wireframes, 15-lib-wireframes, 16-service-wireframes]
tech_stack:
  added: []
  patterns: [pytest-fixture-discovery, fastapi-testclient-db-override, sys-path-manipulation]
key_files:
  created:
    - libs/adapter-registry/tests/conftest.py
    - libs/shared/tests/conftest.py
    - libs/model-training/tests/conftest.py
    - libs/inference/tests/conftest.py
    - libs/events-py/tests/conftest.py
    - libs/evaluation/tests/conftest.py
    - services/rune-agent/tests/conftest.py
    - services/training-svc/tests/conftest.py
    - services/evolution-svc/tests/conftest.py
    - services/lora-server/tests/conftest.py
  modified:
    - services/api-service/tests/conftest.py
    - services/api-service/tests/test_example_integration.py
    - services/api-service/tests/test_example_mocking.py
    - services/api-service/tests/test_example_unit.py
decisions:
  - "lora-server conftest uses parent dir (not src subdir) in sys.path since source files live directly in services/lora-server/"
  - "training-svc and evolution-svc both have dependencies.py with get_db — used full DB override pattern"
  - "api-service three example test files replaced with placeholder comments to avoid fixture-not-found errors"
metrics:
  duration: "4 min"
  completed_date: "2026-03-03"
  tasks_completed: 3
  files_changed: 14
---

# Phase 13 Plan 02: Component Conftest Files Summary

One-liner: 11 per-component conftest.py files created with mock_vllm_client for libs/inference, DB+TestClient fixtures for 3 FastAPI services, and sys.path injection for lora-server.

## What Was Built

Created conftest.py files for all 11 components (6 libs + 5 services), enabling Phases 14-16 TDD tests to use pre-configured fixtures without inline setup.

### Lib Conftest Files (6)

All minimal comment-only placeholders except libs/inference:

- `libs/adapter-registry/tests/conftest.py` — minimal placeholder
- `libs/shared/tests/conftest.py` — minimal placeholder
- `libs/model-training/tests/conftest.py` — minimal placeholder
- `libs/events-py/tests/conftest.py` — minimal placeholder
- `libs/evaluation/tests/conftest.py` — placeholder for Phase 15 module
- `libs/inference/tests/conftest.py` — `mock_vllm_client` fixture returning MagicMock with `.chat.completions.create` pre-configured

### Service Conftest Files (5)

- `services/api-service/tests/conftest.py` — cleaned of template fixtures; kept db_engine, db_session, test_client, async_client, mock_env_vars, reset_singletons
- `services/rune-agent/tests/conftest.py` — minimal (no TestClient; LangGraph graph service)
- `services/training-svc/tests/conftest.py` — db_engine + db_session + test_client with get_db override
- `services/evolution-svc/tests/conftest.py` — db_engine + db_session + test_client with get_db override
- `services/lora-server/tests/conftest.py` — sys.path.insert to services/lora-server/ for direct module import

### Template Cleanup

Three api-service example test files replaced with `# Placeholder — TDD tests added in Phase 16.`:
- `test_example_integration.py`
- `test_example_mocking.py`
- `test_example_unit.py`

## Verification

- `find libs/ services/ -name conftest.py | wc -l` returns **11**
- `uv run pytest -q` exits 0: **23 passed in 2.83s**

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] lora-server sys.path points to parent dir not /src**
- **Found during:** Task 2
- **Issue:** Plan specified `Path(__file__).parent.parent / "src"` but lora-server has no src/ subdirectory — source files (health.py, vllm_client.py, config.py) live directly in services/lora-server/
- **Fix:** Changed path to `Path(__file__).parent.parent` (services/lora-server/)
- **Files modified:** services/lora-server/tests/conftest.py

## Self-Check: PASSED

All 11 conftest.py files confirmed to exist. All commits verified in git log:
- fd7fd62: feat(13-02): create lib conftest.py files for all 6 libs
- ff4cc13: feat(13-02): create service conftest.py files and clean api-service template fixtures
