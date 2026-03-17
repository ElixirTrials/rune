# Shared Component

## Purpose
This component holds code that is strictly **common** to multiple components. This usually includes:
- Pydantic/Data classes for domain entities (User, Item, Transaction).
- Utility functions (date parsing, string normalization).
- Shared constants.

## Prompt templates

Canonical Jinja2 prompt templates (`.j2`) live in `src/shared/templates/`. Use them from agents via:

```python
from shared import get_prompts_dir
from inference.factory import create_structured_extractor

extractor = create_structured_extractor(
    ...
    prompts_dir=get_prompts_dir(),
    system_template="guest_system.j2",
    user_template="guest_user.j2",
    ...
)
result = await extractor(prompt_vars={"guest_name": "..."})
```

- **DRY:** One template set in shared; agents pass different `prompt_vars` only. Do not duplicate `.j2` files in services.

## Rune-Specific Modules

| Module | Purpose |
|--------|---------|
| `hardware.py` | `HardwareProbe` for detecting CPU, RAM, GPU resources; `get_best_device()` for auto-selecting compute device |
| `checkpoint_db.py` | `SwarmCheckpointDB` for tracking swarm task execution state in SQLite |
| `sandbox.py` | `SubprocessBackend` for isolated code execution with configurable timeout |
| `template_loader.py` | `render_trajectory()` and `render_prompt()` for Jinja2 template rendering |
| `rune_models.py` | Cross-service data contracts: `CodingSession`, `AdapterRef`, `EvolMetrics`, `SwarmConfig`, `SwarmCheckpoint`, `PipelinePhase`, `TaskStatus` |
| `storage_utils.py` | SQLite WAL mode setup and shared database utilities |

## Templates

Pipeline phase templates in `src/shared/templates/`:

| Template | Phase | Purpose |
|----------|-------|---------|
| `decompose.j2` / `prompt_decompose.j2` | 1: Decompose | Break project into subtasks |
| `plan.j2` / `prompt_plan.j2` | 2: Plan | Architecture plan per subtask |
| `code.j2` / `code_retry.j2` / `prompt_code.j2` | 3: Code | Generate code with retry support |
| `integrate.j2` / `prompt_integrate.j2` | 4: Integrate | Merge subtask outputs |

## Rules
1.  **No Business Logic**: Do not put complex agent logic or API handlers here.
2.  **Minimal Dependencies**: Keep imports light. `shared` is imported by everyone.
3.  **Type Safety**: Everything must be typed.

## Adding a New Model

1.  Open `src/shared/models.py`.
2.  Add your dataclass or Pydantic model.

```python
from pydantic import BaseModel

class AnalysisResult(BaseModel):
    score: float
    reasoning: str
```

3.  Run `uv sync` if you added new dependencies (unlikely for shared).
