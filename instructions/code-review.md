## Summary (non-training only)

| Category | High | Medium | Low | **Total** |
|----------|------|--------|-----|-----------|
| **Orphan** | 2 | 3 | 3 | **8** |
| **No-op / dead path** | 1 | 0 | 3 | **4** |
| **DRY** | 0 | 3 | 2 | **5** |
| **Duplication** | 1 | 2 | 1 | **4** |
| **All** | **4** | **8** | **9** | **21** |

---

## Orphan

### High
| Location | Issue | Fix |
|----------|-------|-----|
| `tools/cont_probe.py` L24–25 | Imports `dedup_code` / `extract_code` from `continuation.py`, which only exports `extract_partial_code` — **import fails** | Re-point to `extract_partial_code` or restore aliases |
| `tools/capacity_sweep.py` L22–27 | Imports broken helpers via `cont_probe` | Fix `cont_probe` first |

### Medium
| Location | Issue | Fix |
|----------|-------|-----|
| `src/rune/engine/state.py` L103 (`current_adapter`) | Written in `graph.py`, never read | Use for lineage/debug or remove |
| `src/rune/model/wrapper.py` L106–124 (`offload_base`) | Never passed as `True` from callers | Expose via config or remove |
| `src/rune/model/inference.py` L83 (`skip_completion_retry`) | Never set `True` from engine/wrapper | Wire from config or remove |

### Low
| Location | Issue | Fix |
|----------|-------|-----|
| `pyproject.toml` L22 (`httpx`) | No usage in repo | Remove or implement HTTP client |
| `pyproject.toml` L14–15 (`tree-sitter*`) | No imports in `src/` or `tests/` | Remove or implement `validate_syntax` |
| `src/rune/config.py` L68–91, L39–52 (`from_env`, `save`) | Only used in unit tests | Document as operator API or drop |
| `tests/` (no `tests/gpu/`) | `@pytest.mark.gpu` defined but no GPU test dir | Add tests or update docs |

---

## No-op / dead path

### High
| Location | Issue | Fix |
|----------|-------|-----|
| `hypernetwork.py` L13–45, L410–470; `graph.py` L177–214 | Agent debug logging to hard-coded `.cursor/debug-*.log` | Remove `#region agent log` blocks or env-gate |

### Low
| Location | Issue | Fix |
|----------|-------|-----|
| `continuation.py` L40–41 | `validate_syntax` raises `NotImplementedError` for non-Python | OK if Python-only; implement or document |
| `parse.py` L117, L141, L184, L217 | Validation failures return `{}` (silent budget burn) | By design; consider surfacing feedback |
| `tools/smoke_test_engine.py` L44–45 | Bare `except Exception: pass` in `_mem()` | Narrow exception type |

---

## DRY

### Medium
| Location | Issue | Fix |
|----------|-------|-----|
| `continuation.py` + `parse.py` | Repeated `CodeResult.model_validate_json` → `extract_code_value` fallback | Shared `extract_code_from_raw()` |
| `cli.py` L68–74 vs `config.py` L94–116 | Two similar YAML loaders | Shared `load_yaml_model(path, Model)` |
| `hypernetwork.py` + `graph.py` + `smoke_test_engine.py` | CUDA memory reporting repeated three ways | Single `cuda_mem_snapshot()` |

### Low
| Location | Issue | Fix |
|----------|-------|-----|
| `policy.py` L43–59 | `code` and `repair` actions differ only by template names | Small factory helper |
| `inference.py` L69–216 vs L293–370 | `generate` / `generate_continuation` share chat-template + sampling setup | Extract shared builders |

---

## Duplication

### High
| Location | Issue | Fix |
|----------|-------|-----|
| `hypernetwork.py` + `graph.py` | Duplicate agent-debug JSONL writers | One env-gated helper |

### Medium
| Location | Issue | Fix |
|----------|-------|-----|
| `bench/runner.py` L139–153 vs `graph.py` L350–357 | Sandbox exec + pass/fail from `exit_code == 0` | Optional shared `evaluate_code()` if bench grows |
| `tools/adapter_probe.py`, `adapter_diag.py`, etc. | Repeated load-config → ModelWrapper → generate pattern | Thin `tools/_model_session.py` |

### Low
| Location | Issue | Fix |
|----------|-------|-----|
| `templates/*.j2` | Paired `*.j2` + `prompt_*.j2` per action | Intentional for hypernet conditioning — don’t merge blindly |

---

## Tests / stale code (non-training)

| Location | Severity | Issue | Fix |
|----------|----------|-------|-----|
| `tests/unit/test_state.py` L24 | Low | `prompt_template="prompt_decompose"` references removed template | Use `prompt_decompose_concise` |
| `pyproject.toml` L112–113 | Low | Mypy overrides for packages not in repo | Remove stale overrides |

---

## Priority (non-training)

1. Remove or env-gate agent debug logging in `hypernetwork.py` / `graph.py`
2. Fix `tools/cont_probe.py` imports (and `capacity_sweep.py`)
3. Extract shared code-extraction + CUDA mem helpers
4. Prune unused deps (`httpx`, `tree-sitter*`) or implement the planned uses
5. Clean up dead config/model params (`offload_base`, `skip_completion_retry`, `current_adapter`)