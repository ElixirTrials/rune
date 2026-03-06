---
phase: quick-1
plan: "01"
subsystem: docs-and-config
tags: [hardware-agnostic, docs, configuration, lora-server]
dependency_graph:
  requires: []
  provides: [hardware-agnostic-docs, configurable-lora-server]
  affects: [README.md, docs/architecture/multi-gpu-strategy.md, docs/implementation-plan.md, services/lora-server]
tech_stack:
  added: []
  patterns: [env-var-override-pattern, yaml-config-reading-in-bash]
key_files:
  created: []
  modified:
    - README.md
    - docs/architecture/multi-gpu-strategy.md
    - docs/implementation-plan.md
    - docs/architecture/monorepo-mapping.md
    - docs/architecture/recursive-loop.md
    - docs/article/methods.md
    - docs/article/background.md
    - docs/article/discussion.md
    - mkdocs.yml
    - services/lora-server/config.py
    - services/lora-server/config.yaml
    - services/lora-server/startup.sh
    - infra/docker-compose.yml
decisions:
  - "lora-server defaults to single-GPU (PP=1) — most universally compatible default"
  - "TP=2 ValueError guard removed; replaced with docstring warning — NVLink-equipped users should be able to choose TP"
  - "startup.sh reads all parallelism settings from config.yaml via Python yaml parsing with env var overrides"
  - "GPU Strategy doc keeps PCIe vs NVLink reasoning — valid general knowledge, framed generically"
metrics:
  duration_seconds: 543
  completed_date: "2026-03-06"
  tasks_completed: 2
  files_modified: 13
---

# Quick Task 1: Remove Hardware Prescriptions, Make Rune Hardware-Agnostic

**One-liner:** Replaced all specific hardware prescriptions (RTX 4090, Threadripper, CXL, CUDA 12.8, PyTorch 2.9) with generic consumer-GPU guidance across 9 documentation files and parameterized lora-server config/startup for user-configurable parallelism.

---

## Tasks Completed

| Task | Name | Commit | Key Files |
|------|------|--------|-----------|
| 1 | Remove hardware prescriptions from all documentation | 6717ae2 | README.md, multi-gpu-strategy.md, implementation-plan.md, monorepo-mapping.md, recursive-loop.md, methods.md, background.md, discussion.md, mkdocs.yml |
| 2 | Parameterize lora-server config and docker-compose | 9ebaeb4 | config.py, config.yaml, startup.sh, docker-compose.yml |

---

## What Was Done

### Task 1: Documentation

**README.md:**
- Replaced "Hardware Requirements" section (specific GPU table) with a "Hardware" section describing Rune as running on any local CUDA-capable GPU
- Updated "Serving Architecture" paragraph to remove hardcoded PP=2 and two-GPU language
- Updated "Current Status" to describe Phase 0 as environment validation, not hardware validation
- Changed VRAM limit language from "filesystem or VRAM limits" to "filesystem limits"
- Updated QLoRA rationale to be generic ("consumer GPUs with limited VRAM") rather than specific ("24 GB RTX 4090")

**docs/architecture/multi-gpu-strategy.md:**
- Retitled to "GPU Strategy"
- Rewrote overview to describe single-GPU default with optional multi-GPU pipeline parallelism
- Removed RTX 4090, CXL, PCIe Gen4, Ada Lovelace, sm_89 hardware references
- Kept the technical reasoning for why pipeline parallelism is preferred over tensor parallelism on PCIe-connected GPUs (valid general knowledge, framed generically)
- Removed "Hardware Reference" table entirely
- Replaced GPU 0 / GPU 1 specifics with "primary GPU" / "secondary GPU" language
- Updated VRAM budget tables to use generic examples

**docs/implementation-plan.md:**
- Phase 0 renamed from "Hardware and Environment Validation" to "Environment Validation"
- Replaced hardware constraints table with environment requirements table (generic)
- Removed Note on tensor parallelism paragraph and specific GPU specs
- Replaced "Hardware note for Phase 1" with "Compute note for Phase 1"
- Removed PP=2 from serving baseline description
- Removed `No --tensor-parallel-size 2 flag anywhere in config` success criterion

**docs/architecture/monorepo-mapping.md:**
- Changed vLLM subprocess description from "PP=2, TP=1, --enable-lora" to "dynamic LoRA loading"
- Updated GPU lease diagram to use "secondary GPU" instead of "GPU 1"
- Updated cross-reference from "Multi-GPU Strategy" to "GPU Strategy"

**docs/architecture/recursive-loop.md:**
- Changed "served via vLLM PP=2" to "served via vLLM"
- Updated cross-references from "Multi-GPU Strategy" to "GPU Strategy"

**docs/article/methods.md:**
- Changed lora-server description from "pipeline parallelism (PP=2)" to "optional pipeline parallelism"
- Rewrote Serving Architecture subsection: removed dual RTX 4090 / 24 GB per GPU language, reframed as consumer-GPU guidance, kept PCIe vs NVLink reasoning generically

**docs/article/background.md:**
- Replaced "For Rune's target hardware (dual RTX 4090, 24 GB per GPU)" with "consumer GPUs with limited VRAM"

**docs/article/discussion.md:**
- Replaced "Phase 0 hardware validation (PP=2 + QLoRA + vLLM compatibility on dual RTX 4090)" with "Phase 0 environment validation (vLLM + QLoRA compatibility)"

**mkdocs.yml:**
- Updated nav label from "Multi-GPU Strategy" to "GPU Strategy"

### Task 2: Config and Startup Parameterization

**services/lora-server/config.py:**
- Changed `pipeline_parallel_size` default from 2 to 1 (single-GPU default)
- Removed `__post_init__` ValueError that raised on `tensor_parallel_size == 2`
- Updated class docstring: removed "Enforces PP=2/TP=1 layout" language; added Warning section documenting vLLM #21471 and when TP is acceptable (NVLink GPUs)
- Updated docstring example to show `pipeline_parallel_size = 1`

**services/lora-server/config.yaml:**
- Changed `pipeline_parallel_size` from 2 to 1
- Added explanatory comments for each setting including how to configure for multi-GPU
- Added `gpu_memory_utilization` field with comment

**services/lora-server/startup.sh:**
- Replaced hardcoded `--pipeline-parallel-size 2 --tensor-parallel-size 1` with dynamic values read from config.yaml
- Added `_yaml_val()` helper function that reads a key from config.yaml with a fallback default
- All parallelism settings (PIPELINE_PARALLEL_SIZE, TENSOR_PARALLEL_SIZE, QUANTIZATION, MAX_LORAS, MAX_LORA_RANK, GPU_MEMORY_UTILIZATION) can be overridden via environment variables
- Added `--gpu-memory-utilization` flag to vLLM invocation (was previously missing)
- Added `--max-lora-rank` flag to vLLM invocation (was previously missing from startup.sh)

**infra/docker-compose.yml:**
- Added comment above deploy block explaining that `count: all` passes through all available GPUs and how to limit it

---

## Verification

```
# Hardware reference check (excluding .venv and .planning):
grep -rn "RTX 4090|Threadripper|CXL|CUDA 12.8|PyTorch 2.9|48 GB total|24 GB VRAM|24 GB per GPU|sm_89|Ada Lovelace|cu128" \
  --include="*.md" --include="*.py" --include="*.yaml" --include="*.yml" --include="*.sh" . \
  | grep -v ".planning/" | grep -v ".venv/" | grep -v "node_modules/"
# Result: 0 matches (PASS)

# lora-server tests:
uv run pytest services/lora-server/tests/ -x -q
# Result: 3 passed (PASS)
```

---

## Deviations from Plan

### Auto-fixed Issues

None. Plan executed exactly as written with one small addition:

**[Rule 2 - Missing functionality] Added --max-lora-rank flag to startup.sh**
- **Found during:** Task 2
- **Issue:** The original startup.sh was missing `--max-lora-rank` even though `max_lora_rank` was a config field and was being read by the config loader
- **Fix:** Added `--max-lora-rank "$MAX_LORA_RANK"` to the vLLM invocation in startup.sh
- **Files modified:** services/lora-server/startup.sh
- **Commit:** 9ebaeb4

---

## Self-Check

**Files exist:**
- README.md: FOUND
- docs/architecture/multi-gpu-strategy.md: FOUND
- services/lora-server/config.py: FOUND
- services/lora-server/config.yaml: FOUND
- services/lora-server/startup.sh: FOUND

**Commits exist:**
- 6717ae2: FOUND (docs task 1)
- 9ebaeb4: FOUND (feat task 2)

## Self-Check: PASSED
