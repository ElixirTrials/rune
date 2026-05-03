# Plug-and-Play HPO + Champion-Training Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Single-command "clone and run" pipeline that auto-detects GPU hardware, runs HPO, then trains the champion adapter on the full dataset — with parameterised storage so cloners without AWS credentials get a working local-only path and operators with credentials get the full S3+MLflow stack.

**Architecture:** A bash entry point (`scripts/run_pipeline.sh`) sources two YAML profiles — one *hardware* (training knobs: `max_length`, `optim`, `attn_impl`, QLoRA on/off, accelerate launcher) and one *storage* (MLflow URI, artifact root, dataset source) — and exports their values as env vars. Existing `train_qlora` / `_build_trial_kwargs` / `_build_sft_config` are taught to read those env vars (with current behaviour as defaults), so the pipeline composes existing scripts (`run_hpo.sh`, plus a new `run_champion.py` that reads the Optuna study's `best_params` and calls `train_and_register` against the full dataset). The two profile axes are orthogonal so `--hw a100-80gb-multi --storage local` is the no-AWS path.

**Tech Stack:** bash, `uv` (project package manager), Python 3.12, PyYAML (already a transitive dep via mlflow), Optuna (already used by HPO), MLflow (already used), HuggingFace transformers/trl/peft (already used), `nvidia-smi` (for hardware detection).

---

## File Structure

**New files (created):**
- `scripts/run_pipeline.sh` — main entry point. Resolves profiles, brings up MLflow stack if cloud, runs HPO, runs champion.
- `scripts/run_champion.py` — reads Optuna study's `best_params`, calls `train_and_register` on the full dataset with champion-tier knobs.
- `scripts/_lib/detect_hw.py` — parses `nvidia-smi -L` + `nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits`, prints a profile name.
- `scripts/_lib/profile_loader.py` — loads a YAML profile and emits shell `export X=Y` lines on stdout (so bash can `eval` them).
- `infra/hw_profiles/l4-22gb.yaml` — current default L4 single-GPU.
- `infra/hw_profiles/a100-40gb.yaml` — A100 40GB single.
- `infra/hw_profiles/a100-80gb.yaml` — A100 80GB single.
- `infra/hw_profiles/a100-80gb-multi.yaml` — multi-A100 with accelerate.
- `infra/storage_profiles/local.yaml` — sqlite MLflow + local filesystem artifacts + bundled mini-dataset.
- `infra/storage_profiles/cloud.yaml` — http MLflow + S3 artifacts + S3 dataset sync.
- `infra/accelerate/multi_gpu.yaml` — accelerate config for DDP across N GPUs.
- `data/sample/pairs_sample.jsonl` — 50-record subset of `pairs_all.jsonl` (bundled so cloners without S3 still see a green run).
- `docs/QUICKSTART.md` — user-facing one-pager.
- `scripts/optimization/tests/test_run_champion.py` — unit test for the champion driver's best-params-extraction logic.
- `tests/test_profile_loader.py` — unit test for YAML→env-var emission.
- `tests/test_detect_hw.py` — unit test for the GPU→profile mapping.
- `tests/test_run_pipeline_smoke.sh` — end-to-end `--dry-run` smoke test.

**Existing files modified:**
- `libs/model-training/src/model_training/trainer.py` — read new `RUNE_*` env vars (`RUNE_MAX_LENGTH`, `RUNE_OPTIM`, `RUNE_ATTN_IMPL`, `RUNE_USE_QLORA`, `RUNE_GRAD_ACCUM`) inside `train_qlora` / `_build_sft_config` / `_build_bnb_config`, with current values as defaults.
- `libs/model-training/src/model_training/trainer_cli.py` — pass-through CLI flags so `train.sh` users can override without env vars.
- `scripts/optimization/run_training_hpo.py` — `_build_trial_kwargs` honours the new env vars when building per-trial overrides.
- `scripts/run_hpo.sh` — drop the unconditional `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for non-quantised paths (it's L4-specific); read it from the profile env.

---

## Task 1: `RUNE_MAX_LENGTH` env-var support in `train_qlora`

**Files:**
- Modify: `libs/model-training/src/model_training/trainer.py:806` (`train_qlora` signature) and `:1180` (`train_and_register` signature)
- Test: `libs/model-training/tests/test_trainer.py`

- [ ] **Step 1: Write the failing test**

Add to `libs/model-training/tests/test_trainer.py`:

```python
def test_train_qlora_reads_max_length_from_env(monkeypatch) -> None:
    """RUNE_MAX_LENGTH overrides the function default but explicit kwarg wins."""
    from model_training import trainer as trainer_mod

    monkeypatch.setenv("RUNE_MAX_LENGTH", "8192")
    assert trainer_mod._resolve_max_length(None) == 8192
    # Explicit kwarg always wins over env var.
    assert trainer_mod._resolve_max_length(4096) == 4096
    monkeypatch.delenv("RUNE_MAX_LENGTH")
    assert trainer_mod._resolve_max_length(None) == 3072  # default
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest libs/model-training/tests/test_trainer.py::test_train_qlora_reads_max_length_from_env -v`
Expected: FAIL — `module 'model_training.trainer' has no attribute '_resolve_max_length'`

- [ ] **Step 3: Add the resolver helper to `trainer.py`**

Add near the top of `libs/model-training/src/model_training/trainer.py`, just after the existing helpers:

```python
def _resolve_max_length(explicit: int | None) -> int:
    """Resolve the SFT max_length from explicit arg → env var → default.

    Precedence: an explicit ``max_length`` kwarg always wins (so call
    sites that have already plumbed it through don't silently regress
    when the env var is set). When the kwarg is ``None`` we fall back
    to ``RUNE_MAX_LENGTH`` (the hardware-profile lever) and finally to
    ``3072`` — the post-RCA-5-H2 default that fits the p75 of the
    mined-pairs token distribution.
    """
    if explicit is not None:
        return explicit
    raw = os.environ.get("RUNE_MAX_LENGTH")
    if raw is None:
        return 3072
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(
            f"RUNE_MAX_LENGTH must be an integer, got {raw!r}"
        ) from exc
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest libs/model-training/tests/test_trainer.py::test_train_qlora_reads_max_length_from_env -v`
Expected: PASS.

- [ ] **Step 5: Wire `_resolve_max_length` into `train_qlora` and `train_and_register`**

In `libs/model-training/src/model_training/trainer.py`, change the `max_length: int = 3072` default in BOTH `train_qlora` (around line 806) and `train_and_register` (around line 1180) to `max_length: int | None = None`. Inside each function body, immediately resolve:

```python
max_length = _resolve_max_length(max_length)
```

Place this line BEFORE any code that uses `max_length` (search for `max_length=` inside each function and ensure the resolve happens first).

- [ ] **Step 6: Run the full trainer test suite to confirm no regression**

Run: `uv run pytest libs/model-training/tests/test_trainer.py -q`
Expected: all pass (existing tests use the explicit kwarg → resolver returns it unchanged).

- [ ] **Step 7: Commit**

```bash
git add libs/model-training/src/model_training/trainer.py libs/model-training/tests/test_trainer.py
git commit -m "feat(trainer): RUNE_MAX_LENGTH env override for hardware profiles"
```

---

## Task 2: `RUNE_OPTIM` and `RUNE_GRAD_ACCUM` env-var support

**Files:**
- Modify: `libs/model-training/src/model_training/trainer.py:633-702` (`_build_sft_config`)
- Test: `libs/model-training/tests/test_trainer.py`

- [ ] **Step 1: Write the failing tests**

Append to `libs/model-training/tests/test_trainer.py`:

```python
def test_resolve_optim_uses_env_var(monkeypatch) -> None:
    from model_training import trainer as trainer_mod

    monkeypatch.setenv("RUNE_OPTIM", "adamw_torch_fused")
    assert trainer_mod._resolve_optim() == "adamw_torch_fused"
    monkeypatch.delenv("RUNE_OPTIM")
    assert trainer_mod._resolve_optim() == "paged_adamw_8bit"  # L4 default


def test_resolve_optim_rejects_unknown_value(monkeypatch) -> None:
    """Surface typos loudly instead of silently falling back."""
    import pytest

    from model_training import trainer as trainer_mod

    monkeypatch.setenv("RUNE_OPTIM", "adamw_torch_fased")  # typo
    with pytest.raises(ValueError, match="RUNE_OPTIM"):
        trainer_mod._resolve_optim()


def test_resolve_grad_accum_uses_env_var(monkeypatch) -> None:
    from model_training import trainer as trainer_mod

    monkeypatch.setenv("RUNE_GRAD_ACCUM", "4")
    assert trainer_mod._resolve_grad_accum(None) == 4
    # Explicit kwarg wins.
    assert trainer_mod._resolve_grad_accum(8) == 8
    monkeypatch.delenv("RUNE_GRAD_ACCUM")
    assert trainer_mod._resolve_grad_accum(None) == 16  # current default
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest libs/model-training/tests/test_trainer.py::test_resolve_optim_uses_env_var libs/model-training/tests/test_trainer.py::test_resolve_grad_accum_uses_env_var -v`
Expected: FAIL — helpers don't exist.

- [ ] **Step 3: Add the resolvers**

In `libs/model-training/src/model_training/trainer.py`, just below `_resolve_max_length`:

```python
_VALID_OPTIM = frozenset(
    {
        "paged_adamw_8bit",
        "paged_adamw_32bit",
        "adamw_torch",
        "adamw_torch_fused",
        "adamw_bnb_8bit",
    }
)


def _resolve_optim() -> str:
    """Resolve the SFTConfig ``optim`` knob from env var → default.

    Default ``paged_adamw_8bit`` is L4-tuned (it spills optimizer state
    to host RAM so QLoRA + adapter grads fit in 22 GB). On A100 80GB
    set ``RUNE_OPTIM=adamw_torch_fused`` for ~1.5× throughput.
    """
    raw = os.environ.get("RUNE_OPTIM", "paged_adamw_8bit")
    if raw not in _VALID_OPTIM:
        raise ValueError(
            f"RUNE_OPTIM must be one of {sorted(_VALID_OPTIM)}, got {raw!r}"
        )
    return raw


def _resolve_grad_accum(explicit: int | None) -> int:
    """Resolve gradient_accumulation_steps from explicit → env → default."""
    if explicit is not None:
        return explicit
    raw = os.environ.get("RUNE_GRAD_ACCUM")
    if raw is None:
        return 16
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(
            f"RUNE_GRAD_ACCUM must be an integer, got {raw!r}"
        ) from exc
```

- [ ] **Step 4: Wire into `_build_sft_config`**

In `_build_sft_config` (around line 720) replace:

```python
        "optim": "paged_adamw_8bit",
```

with:

```python
        "optim": _resolve_optim(),
```

In `_resolve_training_params` (around line 290) the existing `grad_accum` resolution falls through to a hard-coded `16`. Find that fallback and replace with `_resolve_grad_accum(gradient_accumulation_steps)`. (Only one site — search for `grad_accum`.)

- [ ] **Step 5: Run tests**

Run: `uv run pytest libs/model-training/tests/test_trainer.py -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add libs/model-training/src/model_training/trainer.py libs/model-training/tests/test_trainer.py
git commit -m "feat(trainer): RUNE_OPTIM and RUNE_GRAD_ACCUM env overrides"
```

---

## Task 3: `RUNE_ATTN_IMPL` and `RUNE_USE_QLORA` env-var support

**Files:**
- Modify: `libs/model-training/src/model_training/trainer.py` (`_build_bnb_config`, `_resolve_training_params`)
- Test: `libs/model-training/tests/test_trainer.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_resolve_attn_impl_uses_env_var(monkeypatch) -> None:
    from model_training import trainer as trainer_mod

    monkeypatch.setenv("RUNE_ATTN_IMPL", "flash_attention_2")
    # Env wins when registry returns no preference (None).
    assert trainer_mod._resolve_attn_impl(None) == "flash_attention_2"
    # Registry / explicit value wins over env var.
    assert trainer_mod._resolve_attn_impl("eager") == "eager"
    monkeypatch.delenv("RUNE_ATTN_IMPL")
    assert trainer_mod._resolve_attn_impl(None) is None


def test_resolve_use_qlora_default_true(monkeypatch) -> None:
    from model_training import trainer as trainer_mod

    monkeypatch.delenv("RUNE_USE_QLORA", raising=False)
    assert trainer_mod._resolve_use_qlora() is True
    monkeypatch.setenv("RUNE_USE_QLORA", "0")
    assert trainer_mod._resolve_use_qlora() is False
    monkeypatch.setenv("RUNE_USE_QLORA", "1")
    assert trainer_mod._resolve_use_qlora() is True


def test_build_bnb_config_returns_none_when_qlora_disabled(monkeypatch) -> None:
    """A100 80GB doesn't need 4-bit quantisation; bf16 LoRA is faster."""
    from model_training import trainer as trainer_mod

    monkeypatch.setenv("RUNE_USE_QLORA", "0")
    assert trainer_mod._build_bnb_config() is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest libs/model-training/tests/test_trainer.py::test_resolve_attn_impl_uses_env_var libs/model-training/tests/test_trainer.py::test_resolve_use_qlora_default_true libs/model-training/tests/test_trainer.py::test_build_bnb_config_returns_none_when_qlora_disabled -v`
Expected: FAIL.

- [ ] **Step 3: Add the resolvers**

In `libs/model-training/src/model_training/trainer.py`, alongside the other resolvers:

```python
def _resolve_attn_impl(explicit: str | None) -> str | None:
    """Pick the attention impl: explicit (registry) wins → env → None."""
    if explicit is not None:
        return explicit
    return os.environ.get("RUNE_ATTN_IMPL")


def _resolve_use_qlora() -> bool:
    """Whether to use bitsandbytes 4-bit quantisation.

    Default ``True`` keeps the L4 22GB story working out of the box;
    set ``RUNE_USE_QLORA=0`` on A100 80GB+ for ~2× throughput.
    """
    raw = os.environ.get("RUNE_USE_QLORA", "1")
    return raw not in {"0", "false", "False", ""}
```

Modify `_build_bnb_config` so the first line short-circuits when QLoRA is disabled:

```python
def _build_bnb_config() -> Any | None:
    """Build the BitsAndBytesConfig, or return None when QLoRA is disabled."""
    if not _resolve_use_qlora():
        return None
    # ... existing body unchanged
```

- [ ] **Step 4: Update callers to handle a `None` bnb_config**

Search for `bnb_config = _build_bnb_config()` in `trainer.py` (two sites). Each call site passes `bnb_config` into `_get_or_load_base`. Open `_get_or_load_base` and find the line that does `model_kwargs["quantization_config"] = bnb_config` (or similar). Guard:

```python
if bnb_config is not None:
    model_kwargs["quantization_config"] = bnb_config
else:
    model_kwargs["torch_dtype"] = torch.bfloat16
```

This is the bf16-LoRA path: when not quantising, load the model in bf16 directly so VRAM still fits A100 80GB and LoRA layers attach as usual.

- [ ] **Step 5: Wire `_resolve_attn_impl` into `_resolve_training_params`**

Around line 320, where `resolved["attn_impl"] = mc.attn_implementation` is set: wrap it with `_resolve_attn_impl(...)` so env var fills in when the registry has nothing:

```python
resolved["attn_impl"] = _resolve_attn_impl(mc.attn_implementation)
```

For the `model_config_name is None` branch (no registry lookup), default to `_resolve_attn_impl(None)`.

- [ ] **Step 6: Run tests**

Run: `uv run pytest libs/model-training/tests/test_trainer.py -q`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add libs/model-training/src/model_training/trainer.py libs/model-training/tests/test_trainer.py
git commit -m "feat(trainer): RUNE_ATTN_IMPL and RUNE_USE_QLORA env overrides"
```

---

## Task 4: Honour the new env vars in `_build_trial_kwargs` (HPO)

**Files:**
- Modify: `scripts/optimization/run_training_hpo.py` (`_build_trial_kwargs`, around line 305)
- Test: `scripts/optimization/tests/test_training_hpo.py`

- [ ] **Step 1: Write the failing test**

Append to `scripts/optimization/tests/test_training_hpo.py`:

```python
def test_build_trial_kwargs_passes_through_max_length_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RUNE_MAX_LENGTH must reach train_and_register via trial kwargs.

    Without this, the HW profile would set RUNE_MAX_LENGTH=8192 but
    every HPO trial would still bake max_length=3072 into its
    train_and_register call (because trainer.py's resolver only fires
    on a None kwarg). We forward the env value as an explicit kwarg.
    """
    monkeypatch.setenv("RUNE_MAX_LENGTH", "8192")
    run_args = HPORunArgs(
        dataset="/tmp/x",
        adapter_id_prefix="t",
        model_config_name="qwen3.5-9b",
        warm_start=None,
        subsample=10,
        output_root=Path("/tmp/out"),
        experiment_name="exp",
        keep_top_k=3,
    )
    sampled = {
        "lr": 1e-4,
        "alpha_override": 32,
        "lora_dropout": 0.0,
        "warmup_ratio": 0.05,
        "grad_accum": 16,
        "lr_scheduler": "cosine",
        "diff_aware_loss": True,
        "neftune_noise_alpha": None,
    }
    kwargs = _build_trial_kwargs(
        run_args=run_args,
        sampled=sampled,
        adapter_id="t-001",
        trial_dataset_path="/tmp/d.jsonl",
    )
    assert kwargs["max_length"] == 8192
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest scripts/optimization/tests/test_training_hpo.py::test_build_trial_kwargs_passes_through_max_length_env -v`
Expected: FAIL — `max_length` not in `kwargs`.

- [ ] **Step 3: Modify `_build_trial_kwargs`**

In `scripts/optimization/run_training_hpo.py`, find `_build_trial_kwargs` (around line 305). After the existing `kwargs` dict is built and before `kwargs.update(run_args.extra_train_kwargs)`, add:

```python
    # Hardware profiles set RUNE_MAX_LENGTH so the HPO trials inherit
    # the same sequence cap as champion training. trainer.py also reads
    # it, but forwarding here makes the kwarg explicit on the call site
    # (visible in MLflow params, CI logs, etc.).
    raw_ml = os.environ.get("RUNE_MAX_LENGTH")
    if raw_ml is not None:
        kwargs["max_length"] = int(raw_ml)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest scripts/optimization/tests/test_training_hpo.py::test_build_trial_kwargs_passes_through_max_length_env -v`
Expected: PASS.

- [ ] **Step 5: Run the full HPO test suite**

Run: `uv run pytest scripts/optimization/tests/test_training_hpo.py -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add scripts/optimization/run_training_hpo.py scripts/optimization/tests/test_training_hpo.py
git commit -m "feat(hpo): forward RUNE_MAX_LENGTH into per-trial train kwargs"
```

---

## Task 5: Profile loader (YAML → exported env vars)

**Files:**
- Create: `scripts/_lib/profile_loader.py`
- Test: `tests/test_profile_loader.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_profile_loader.py`:

```python
"""Tests for scripts/_lib/profile_loader.py — bash-friendly YAML loader."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LOADER = REPO_ROOT / "scripts" / "_lib" / "profile_loader.py"


def _run(*args: str) -> tuple[int, str, str]:
    proc = subprocess.run(
        [sys.executable, str(LOADER), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode, proc.stdout, proc.stderr


def test_loader_emits_export_lines(tmp_path: Path) -> None:
    profile = tmp_path / "p.yaml"
    profile.write_text(
        "env:\n"
        "  RUNE_MAX_LENGTH: '8192'\n"
        "  RUNE_OPTIM: adamw_torch_fused\n"
    )
    rc, out, err = _run(str(profile))
    assert rc == 0, err
    # Each line must be a valid bash export the caller can `eval`.
    lines = [ln for ln in out.splitlines() if ln.strip()]
    assert "export RUNE_MAX_LENGTH='8192'" in lines
    assert "export RUNE_OPTIM='adamw_torch_fused'" in lines


def test_loader_quotes_special_chars(tmp_path: Path) -> None:
    """Single quotes in values must be escaped so eval is shell-safe."""
    profile = tmp_path / "p.yaml"
    profile.write_text("env:\n  X: \"a'b\"\n")
    rc, out, _ = _run(str(profile))
    assert rc == 0
    assert "export X='a'\"'\"'b'" in out


def test_loader_emits_hpo_block_as_export_with_prefix(tmp_path: Path) -> None:
    """The hpo: block becomes RUNE_PIPELINE_HPO_<KEY> exports."""
    profile = tmp_path / "p.yaml"
    profile.write_text(
        "hpo:\n"
        "  n_trials: 30\n"
        "  subsample: 500\n"
    )
    rc, out, _ = _run(str(profile))
    assert rc == 0
    assert "export RUNE_PIPELINE_HPO_N_TRIALS='30'" in out
    assert "export RUNE_PIPELINE_HPO_SUBSAMPLE='500'" in out


def test_loader_rejects_missing_file() -> None:
    rc, _, err = _run("/nonexistent/profile.yaml")
    assert rc != 0
    assert "not found" in err.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_profile_loader.py -v`
Expected: FAIL — file does not exist.

- [ ] **Step 3: Implement `scripts/_lib/profile_loader.py`**

```python
"""Translate a HW or storage profile YAML into shell ``export`` lines.

Used by ``scripts/run_pipeline.sh`` like::

    eval "$(uv run python scripts/_lib/profile_loader.py infra/hw_profiles/a100-80gb.yaml)"

so the bash entry point doesn't need its own YAML parser. Top-level
``env:`` keys are emitted verbatim as ``export KEY='VALUE'``; nested
blocks (``hpo:``, ``champion:``, ``accelerate:``, ``dataset:``) are
flattened with an ``RUNE_PIPELINE_<BLOCK>_<KEY>`` prefix so bash can
read them via ``${RUNE_PIPELINE_HPO_N_TRIALS:-30}`` lookups.

Single quotes in values are escaped using the standard bash trick
(``'\''``) so ``eval`` stays safe even on ill-formed profile values.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml


def _shell_quote(value: str) -> str:
    """Single-quote-wrap and escape any literal single quotes."""
    return "'" + value.replace("'", "'\"'\"'") + "'"


def _emit_block(prefix: str, block: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for k, v in block.items():
        if isinstance(v, dict):
            out.extend(_emit_block(f"{prefix}_{k.upper()}", v))
        else:
            out.append(f"export {prefix}_{k.upper()}={_shell_quote(str(v))}")
    return out


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(
            f"usage: {argv[0]} <profile.yaml>",
            file=sys.stderr,
        )
        return 2

    path = Path(argv[1])
    if not path.is_file():
        print(f"profile not found: {path}", file=sys.stderr)
        return 1

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    # env: top-level keys exported verbatim.
    env = data.get("env") or {}
    for k, v in env.items():
        print(f"export {k}={_shell_quote(str(v))}")

    # Other top-level blocks become RUNE_PIPELINE_<BLOCK>_<KEY> exports.
    for block_name, block in data.items():
        if block_name == "env":
            continue
        if not isinstance(block, dict):
            print(
                f"export RUNE_PIPELINE_{block_name.upper()}={_shell_quote(str(block))}"
            )
            continue
        for line in _emit_block(f"RUNE_PIPELINE_{block_name.upper()}", block):
            print(line)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_profile_loader.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/_lib/profile_loader.py tests/test_profile_loader.py
git commit -m "feat(pipeline): YAML profile loader for bash entry point"
```

---

## Task 6: Hardware detector (`nvidia-smi` → profile name)

**Files:**
- Create: `scripts/_lib/detect_hw.py`
- Test: `tests/test_detect_hw.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_detect_hw.py`:

```python
"""Tests for scripts/_lib/detect_hw.py — GPU model+VRAM → profile name."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "_lib"))

from detect_hw import classify_gpus  # noqa: E402


def test_single_l4_returns_l4_profile() -> None:
    assert classify_gpus([("NVIDIA L4", 22_528)]) == "l4-22gb"


def test_single_a100_40gb() -> None:
    assert classify_gpus([("NVIDIA A100-PCIE-40GB", 40_960)]) == "a100-40gb"


def test_single_a100_80gb() -> None:
    assert classify_gpus([("NVIDIA A100-SXM4-80GB", 81_920)]) == "a100-80gb"


def test_multi_a100_80gb() -> None:
    gpus = [("NVIDIA A100-SXM4-80GB", 81_920)] * 4
    assert classify_gpus(gpus) == "a100-80gb-multi"


def test_multi_a100_40gb_treated_as_multi() -> None:
    """Multi-GPU on the smaller card still maps to multi (DDP wins)."""
    gpus = [("NVIDIA A100-PCIE-40GB", 40_960)] * 2
    assert classify_gpus(gpus) == "a100-80gb-multi"


def test_unknown_gpu_falls_back_to_l4_profile() -> None:
    """Unknown SKUs default to the most conservative profile.

    Better to under-utilise a fancy card than to OOM on a smaller one
    we mis-detected.
    """
    assert classify_gpus([("NVIDIA RTX 9999", 24_000)]) == "l4-22gb"


def test_no_gpus_raises() -> None:
    import pytest

    with pytest.raises(RuntimeError, match="no GPUs"):
        classify_gpus([])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_detect_hw.py -v`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Implement `scripts/_lib/detect_hw.py`**

```python
"""Map ``nvidia-smi`` output to a hardware-profile name.

Pure function ``classify_gpus`` is unit-tested; the ``main`` wrapper
shells out to ``nvidia-smi`` so the bash entry point can do
``profile=$(uv run python scripts/_lib/detect_hw.py)``.

Keep the classification rules conservative: better to fall back to
the L4 profile (the only one that's been hammered against the existing
scripts) than to silently pick a too-aggressive profile and OOM mid-trial.
"""

from __future__ import annotations

import subprocess
import sys


def _query_gpus() -> list[tuple[str, int]]:
    """Run ``nvidia-smi`` and return ``[(name, total_mem_mib), ...]``."""
    out = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    rows: list[tuple[str, int]] = []
    for line in out.stdout.strip().splitlines():
        name, mem = line.split(",", 1)
        rows.append((name.strip(), int(mem.strip())))
    return rows


def classify_gpus(gpus: list[tuple[str, int]]) -> str:
    """Pick a profile name from a list of ``(model, total_mem_mib)`` tuples."""
    if not gpus:
        raise RuntimeError("no GPUs detected — refusing to pick a profile")

    if len(gpus) > 1:
        # Multi-GPU: only A100 (or better) is supported in our accelerate
        # config; anything else falls back to the conservative profile so
        # the user gets a working — if slow — single-GPU run.
        names = " ".join(name for name, _ in gpus)
        if "A100" in names or "H100" in names:
            return "a100-80gb-multi"
        return "l4-22gb"

    name, mem = gpus[0]
    if "A100" in name:
        return "a100-80gb" if mem >= 70_000 else "a100-40gb"
    if "L4" in name:
        return "l4-22gb"
    # Unknown SKU: fall back to the L4 profile (most conservative).
    return "l4-22gb"


def main() -> int:
    try:
        gpus = _query_gpus()
    except FileNotFoundError:
        print("nvidia-smi not found", file=sys.stderr)
        return 127
    except subprocess.CalledProcessError as exc:
        print(f"nvidia-smi failed: {exc}", file=sys.stderr)
        return exc.returncode

    print(classify_gpus(gpus))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_detect_hw.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/_lib/detect_hw.py tests/test_detect_hw.py
git commit -m "feat(pipeline): GPU → profile-name detector"
```

---

## Task 7: HW profile YAMLs (4 tiers)

**Files:**
- Create: `infra/hw_profiles/l4-22gb.yaml`
- Create: `infra/hw_profiles/a100-40gb.yaml`
- Create: `infra/hw_profiles/a100-80gb.yaml`
- Create: `infra/hw_profiles/a100-80gb-multi.yaml`
- Create: `infra/accelerate/multi_gpu.yaml`

- [ ] **Step 1: Write `infra/hw_profiles/l4-22gb.yaml`**

```yaml
# Single L4 22GB — the original hardware target. These values match
# what the codebase had hard-coded before the pipeline refactor, so
# this profile is the "do nothing different" baseline.
name: l4-22gb
env:
  RUNE_MAX_LENGTH: '3072'
  RUNE_OPTIM: paged_adamw_8bit
  RUNE_USE_QLORA: '1'
  RUNE_GRAD_ACCUM: '16'
  RUNE_ATTN_IMPL: eager
  PYTORCH_CUDA_ALLOC_CONF: 'expandable_segments:True'
hpo:
  n_trials: 30
  subsample: 500
  keep_top_k: 3
champion:
  epochs: 3
accelerate:
  launcher: ''  # plain `uv run python`, no accelerate
```

- [ ] **Step 2: Write `infra/hw_profiles/a100-40gb.yaml`**

```yaml
# Single A100 40GB. QLoRA still on (the safety margin is worth the
# small throughput hit), but max_length and batch grow.
name: a100-40gb
env:
  RUNE_MAX_LENGTH: '6144'
  RUNE_OPTIM: paged_adamw_8bit
  RUNE_USE_QLORA: '1'
  RUNE_GRAD_ACCUM: '8'
  RUNE_ATTN_IMPL: flash_attention_2
  PYTORCH_CUDA_ALLOC_CONF: 'expandable_segments:True'
hpo:
  n_trials: 30
  subsample: 750
  keep_top_k: 3
champion:
  epochs: 3
accelerate:
  launcher: ''
```

- [ ] **Step 3: Write `infra/hw_profiles/a100-80gb.yaml`**

```yaml
# Single A100 80GB. Drop QLoRA — bf16 LoRA is ~2× faster and fits.
name: a100-80gb
env:
  RUNE_MAX_LENGTH: '8192'
  RUNE_OPTIM: adamw_torch_fused
  RUNE_USE_QLORA: '0'
  RUNE_GRAD_ACCUM: '4'
  RUNE_ATTN_IMPL: flash_attention_2
hpo:
  n_trials: 30
  subsample: 1000
  keep_top_k: 3
champion:
  epochs: 5
accelerate:
  launcher: ''
```

- [ ] **Step 4: Write `infra/hw_profiles/a100-80gb-multi.yaml`**

```yaml
# Multi A100 80GB via accelerate DDP. The ``accelerate.launcher: multi_gpu``
# field tells run_pipeline.sh to wrap the train command in
# ``accelerate launch --config-file infra/accelerate/multi_gpu.yaml``.
name: a100-80gb-multi
env:
  RUNE_MAX_LENGTH: '8192'
  RUNE_OPTIM: adamw_torch_fused
  RUNE_USE_QLORA: '0'
  RUNE_GRAD_ACCUM: '2'
  RUNE_ATTN_IMPL: flash_attention_2
hpo:
  n_trials: 30
  subsample: 1500
  keep_top_k: 3
champion:
  epochs: 5
accelerate:
  launcher: multi_gpu
  config: infra/accelerate/multi_gpu.yaml
```

- [ ] **Step 5: Write `infra/accelerate/multi_gpu.yaml`**

```yaml
# accelerate launch config for multi-GPU DDP. ``num_processes`` is left
# unset so the operator picks it via ``--num-processes N`` on the
# command line (or the run_pipeline.sh wrapper auto-detects via
# nvidia-smi -L | wc -l).
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
mixed_precision: bf16
num_machines: 1
machine_rank: 0
main_training_function: main
rdzv_backend: static
same_network: true
use_cpu: false
```

- [ ] **Step 6: Commit**

```bash
git add infra/hw_profiles/ infra/accelerate/
git commit -m "feat(pipeline): hardware profiles for L4 / A100 / multi-A100"
```

---

## Task 8: Storage profile YAMLs (local + cloud)

**Files:**
- Create: `infra/storage_profiles/local.yaml`
- Create: `infra/storage_profiles/cloud.yaml`

- [ ] **Step 1: Write `infra/storage_profiles/local.yaml`**

```yaml
# Local-only mode: no docker, no S3, no AWS creds required. MLflow
# uses a sqlite tracking store and writes artifacts to ./mlruns. Best
# for "I just cloned the repo and want to verify it works" runs.
name: local
env:
  MLFLOW_TRACKING_URI: 'sqlite:///./mlflow.db'
  RUNE_DATABASE_URL: 'sqlite:///./rune.db'
upload_adapters_to_mlflow: 'true'  # logs to ./mlruns, not S3
cleanup_local_adapters: 'false'    # keep them — that's the only copy
dataset:
  source: local
  path: data/sample/pairs_sample.jsonl
bring_up_docker_stack: 'false'
```

- [ ] **Step 2: Write `infra/storage_profiles/cloud.yaml`**

```yaml
# Full cloud mode: docker compose stack with MLflow + Litestream + S3.
# Requires AWS creds at ~/.aws and reachable internet.
name: cloud
env:
  MLFLOW_TRACKING_URI: 'http://localhost:5000'
upload_adapters_to_mlflow: 'true'
cleanup_local_adapters: 'true'
dataset:
  source: s3
  uri: 's3://elixirtrials-949678234935-eu-west-2-artifacts/training-data/'
bring_up_docker_stack: 'true'
```

- [ ] **Step 3: Commit**

```bash
git add infra/storage_profiles/
git commit -m "feat(pipeline): local + cloud storage profiles"
```

---

## Task 9: Bundled mini-dataset

**Files:**
- Create: `data/sample/pairs_sample.jsonl`

- [ ] **Step 1: Generate the sample**

The bundled sample needs to (a) be small enough to commit, (b) cover the variety needed to exercise the diff-aware path. Take 50 records spanning multiple repos.

Run from the repo root (NOT in CI — this only runs once at planning time):

```bash
uv run python - <<'EOF'
import json
import random

src = "data/github-pairs/_merged/pairs_all.jsonl"
dst = "data/sample/pairs_sample.jsonl"

# Group by source_task_id, pick the first 25 groups, then keep at most
# 2 pairs per group. Result: ~50 rows that span multiple PRs and have
# the multi-turn structure the diff-aware loss exercises.
import os
os.makedirs("data/sample", exist_ok=True)

from collections import defaultdict, OrderedDict
buckets = OrderedDict()
with open(src) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        meta = rec.get("metadata") or {}
        key = meta.get("source_task_id") or rec.get("task_id") or ""
        buckets.setdefault(key, []).append(line)
        if len(buckets) > 25 and len(buckets[next(reversed(buckets))]) >= 2:
            # have enough variety
            pass
        if len(buckets) >= 25 and sum(min(len(v), 2) for v in buckets.values()) >= 50:
            break

written = 0
with open(dst, "w") as out:
    for key, lines in buckets.items():
        for ln in lines[:2]:
            out.write(ln + "\n")
            written += 1
            if written >= 50:
                break
        if written >= 50:
            break

print(f"wrote {written} records to {dst}")
EOF
```

- [ ] **Step 2: Verify the sample is reasonable**

```bash
wc -l data/sample/pairs_sample.jsonl
du -h data/sample/pairs_sample.jsonl
```

Expected: ≤ 50 lines, ≤ 1MB total. If much larger, drop the per-group cap to 1 and re-run.

- [ ] **Step 3: Commit**

```bash
git add data/sample/pairs_sample.jsonl
git commit -m "feat(pipeline): bundle 50-record sample dataset for local mode"
```

---

## Task 10: Champion-training driver

**Files:**
- Create: `scripts/run_champion.py`
- Test: `scripts/optimization/tests/test_run_champion.py`

- [ ] **Step 1: Write the failing test**

Create `scripts/optimization/tests/test_run_champion.py`:

```python
"""Tests for scripts/run_champion.py — Optuna best_params → train_and_register."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from run_champion import (  # noqa: E402
    _best_params_from_study,
    _build_champion_kwargs,
)


class _FakeTrial:
    def __init__(self, number: int, value: float, params: dict[str, Any]) -> None:
        self.number = number
        self.value = value
        self.params = params

        class _State:
            name = "COMPLETE"

        self.state = _State()


class _FakeStudy:
    def __init__(self, trials: list[_FakeTrial]) -> None:
        self._trials = trials

    def get_trials(self, deepcopy: bool = True) -> list[_FakeTrial]:  # noqa: ARG002
        return list(self._trials)


def test_best_params_from_study_picks_highest_value() -> None:
    s = _FakeStudy(
        [
            _FakeTrial(0, 0.4, {"lr": 1e-4, "alpha_override": 32}),
            _FakeTrial(1, 0.7, {"lr": 3e-4, "alpha_override": 64}),
            _FakeTrial(2, 0.5, {"lr": 5e-5, "alpha_override": 16}),
        ]
    )
    best = _best_params_from_study(s)
    assert best["lr"] == 3e-4
    assert best["alpha_override"] == 64


def test_best_params_from_study_skips_failed_trials() -> None:
    failed = _FakeTrial(0, 0.99, {"lr": 1e-4})
    failed.state.name = "FAIL"
    completed = _FakeTrial(1, 0.5, {"lr": 3e-4})
    s = _FakeStudy([failed, completed])
    assert _best_params_from_study(s)["lr"] == 3e-4


def test_best_params_raises_when_no_completed_trials() -> None:
    failed = _FakeTrial(0, 0.99, {"lr": 1e-4})
    failed.state.name = "FAIL"
    s = _FakeStudy([failed])
    with pytest.raises(RuntimeError, match="no completed trials"):
        _best_params_from_study(s)


def test_build_champion_kwargs_translates_param_names() -> None:
    """Optuna param names → train_and_register kwarg names."""
    best = {
        "lr": 3e-4,
        "alpha_override": 64,
        "lora_dropout": 0.05,
        "warmup_ratio": 0.05,
        "grad_accum": 4,
        "lr_scheduler": "cosine",
        "diff_aware_loss": True,
        "neftune_noise_alpha": None,
    }
    kwargs = _build_champion_kwargs(
        best,
        adapter_id="champion",
        dataset_path="/full/data.jsonl",
        epochs=5,
        warm_start="deltacoder",
        model_config_name="qwen3.5-9b",
        experiment_name="rune-champion",
    )
    assert kwargs["learning_rate"] == 3e-4
    assert kwargs["override_lora_alpha"] == 64
    assert kwargs["lr_scheduler_type"] == "cosine"
    assert kwargs["epochs"] == 5
    assert kwargs["adapter_id"] == "champion"
    assert kwargs["dataset_path"] == "/full/data.jsonl"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest scripts/optimization/tests/test_run_champion.py -v`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Implement `scripts/run_champion.py`**

```python
"""Train the champion adapter on the full dataset using HPO best params.

Reads ``study.best_trial.params`` from an Optuna SQLite DB (the one
``run_training_hpo.py`` wrote to) and calls ``train_and_register``
with those hyperparameters + the full dataset path + a longer epoch
count. The hardware profile env vars (``RUNE_MAX_LENGTH``,
``RUNE_OPTIM``, etc.) are respected automatically by the trainer.

Usage::

    uv run python scripts/run_champion.py \\
        --study-db sqlite:///./optuna_training.db \\
        --study-name rune-training-v1 \\
        --dataset data/github-pairs/_merged/pairs_all.jsonl \\
        --adapter-id champion-2026-05-08 \\
        --epochs 5
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run-champion")


def _best_params_from_study(study: Any) -> dict[str, Any]:
    """Pick the highest-fitness COMPLETED trial's params.

    Avoids ``study.best_trial`` because it raises on partially-failed
    studies; we want a graceful "no completed trials" error instead.
    """
    completed = [
        t for t in study.get_trials(deepcopy=False) if t.state.name == "COMPLETE"
    ]
    if not completed:
        raise RuntimeError(
            "Champion training: study has no completed trials — refusing to "
            "synthesise hyperparameters from failures."
        )
    completed.sort(key=lambda t: t.value, reverse=True)
    best = completed[0]
    logger.info(
        "Champion picked trial %d with fitness=%.4f", best.number, best.value
    )
    return dict(best.params)


def _build_champion_kwargs(
    best: dict[str, Any],
    *,
    adapter_id: str,
    dataset_path: str,
    epochs: int,
    warm_start: str | None,
    model_config_name: str,
    experiment_name: str,
) -> dict[str, Any]:
    """Translate Optuna param names → ``train_and_register`` kwarg names."""
    return {
        "session_id": None,
        "adapter_id": adapter_id,
        "dataset_path": dataset_path,
        "encoding_mode": "multi_turn",
        "model_config_name": model_config_name,
        "warm_start_adapter_id": warm_start,
        "epochs": epochs,
        "learning_rate": best["lr"],
        "gradient_accumulation_steps": best["grad_accum"],
        "lr_scheduler_type": best["lr_scheduler"],
        "override_lora_alpha": best["alpha_override"],
        "override_lora_dropout": best["lora_dropout"],
        "diff_aware_loss": best["diff_aware_loss"],
        "warmup_ratio": best["warmup_ratio"],
        "neftune_noise_alpha": best.get("neftune_noise_alpha"),
        "mlflow_experiment": experiment_name,
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_champion",
        description="Train the champion adapter on the full dataset using "
        "HPO best params.",
    )
    p.add_argument("--study-db", required=True, help="Optuna storage URI.")
    p.add_argument("--study-name", required=True)
    p.add_argument("--dataset", required=True, help="Full-dataset JSONL.")
    p.add_argument("--adapter-id", required=True)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--warm-start", default="deltacoder")
    p.add_argument("--model", dest="model_config_name", default="qwen3.5-9b")
    p.add_argument("--experiment-name", default="rune-champion")
    p.add_argument(
        "--print-only",
        action="store_true",
        help="Resolve params and print kwargs as JSON; do not train.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    import optuna  # noqa: PLC0415

    study = optuna.load_study(study_name=args.study_name, storage=args.study_db)
    best = _best_params_from_study(study)

    from model_training.trainer_cli import _resolve_warm_start  # noqa: PLC0415

    kwargs = _build_champion_kwargs(
        best,
        adapter_id=args.adapter_id,
        dataset_path=args.dataset,
        epochs=args.epochs,
        warm_start=_resolve_warm_start(args.warm_start),
        model_config_name=args.model_config_name,
        experiment_name=args.experiment_name,
    )

    if args.print_only:
        print(json.dumps(kwargs, indent=2, sort_keys=True, default=str))
        return 0

    from model_training.trainer import train_and_register  # noqa: PLC0415

    train_and_register(**kwargs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest scripts/optimization/tests/test_run_champion.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_champion.py scripts/optimization/tests/test_run_champion.py
git commit -m "feat(pipeline): champion-training driver from Optuna best_params"
```

---

## Task 11: Pipeline orchestrator (`run_pipeline.sh`)

**Files:**
- Create: `scripts/run_pipeline.sh`
- Test: `tests/test_run_pipeline_smoke.sh`

- [ ] **Step 1: Write the smoke test**

Create `tests/test_run_pipeline_smoke.sh`:

```bash
#!/usr/bin/env bash
# Smoke test for scripts/run_pipeline.sh in --dry-run mode.
#
# --dry-run resolves profiles, prints what it would execute, and exits
# without invoking HPO or training. This lets us assert the wiring in
# CI on a CPU-only runner.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

OUT=$(scripts/run_pipeline.sh --hw l4-22gb --storage local --dry-run 2>&1)
echo "$OUT"

# Sanity assertions on the dry-run output.
echo "$OUT" | grep -q "hw_profile=l4-22gb" \
    || { echo "FAIL: hw profile not echoed"; exit 1; }
echo "$OUT" | grep -q "storage_profile=local" \
    || { echo "FAIL: storage profile not echoed"; exit 1; }
echo "$OUT" | grep -q "RUNE_MAX_LENGTH=3072" \
    || { echo "FAIL: hw env vars not exported"; exit 1; }
echo "$OUT" | grep -q "MLFLOW_TRACKING_URI=sqlite" \
    || { echo "FAIL: storage env vars not exported"; exit 1; }
echo "$OUT" | grep -q "would run: scripts/run_hpo.sh" \
    || { echo "FAIL: HPO step not announced"; exit 1; }
echo "$OUT" | grep -q "would run: uv run python scripts/run_champion.py" \
    || { echo "FAIL: champion step not announced"; exit 1; }

echo "PASS"
```

Make it executable: `chmod +x tests/test_run_pipeline_smoke.sh`.

- [ ] **Step 2: Run smoke test to verify it fails**

Run: `tests/test_run_pipeline_smoke.sh`
Expected: FAIL — script does not exist.

- [ ] **Step 3: Implement `scripts/run_pipeline.sh`**

```bash
#!/usr/bin/env bash
# ────────────────────────────────────────────────────────────────────────────
# Rune — End-to-end HPO + champion-training pipeline
#
# One command for cloners: detect GPU → resolve hw profile → resolve
# storage profile → run HPO → train the champion on the full dataset.
#
# Usage:
#   scripts/run_pipeline.sh                              # auto-detect HW, local storage
#   scripts/run_pipeline.sh --hw a100-80gb-multi         # override HW
#   scripts/run_pipeline.sh --storage cloud              # use docker MLflow + S3
#   scripts/run_pipeline.sh --dry-run                    # print what would run
# ────────────────────────────────────────────────────────────────────────────
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

HW_PROFILE=""
STORAGE_PROFILE="local"
DRY_RUN=0
ADAPTER_ID="champion-$(date +%Y%m%d-%H%M%S)"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --hw)        HW_PROFILE="$2"; shift 2 ;;
        --storage)   STORAGE_PROFILE="$2"; shift 2 ;;
        --dry-run)   DRY_RUN=1; shift ;;
        --adapter-id) ADAPTER_ID="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,18p' "$0"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

# 1. Resolve HW profile (auto-detect when unset).
if [[ -z "$HW_PROFILE" ]]; then
    HW_PROFILE=$(uv run python scripts/_lib/detect_hw.py)
fi
echo "hw_profile=${HW_PROFILE}"
echo "storage_profile=${STORAGE_PROFILE}"

HW_YAML="infra/hw_profiles/${HW_PROFILE}.yaml"
STORAGE_YAML="infra/storage_profiles/${STORAGE_PROFILE}.yaml"

[[ -f "$HW_YAML" ]] || { echo "missing: $HW_YAML" >&2; exit 1; }
[[ -f "$STORAGE_YAML" ]] || { echo "missing: $STORAGE_YAML" >&2; exit 1; }

# 2. Export both profiles into the current shell.
eval "$(uv run python scripts/_lib/profile_loader.py "$HW_YAML")"
eval "$(uv run python scripts/_lib/profile_loader.py "$STORAGE_YAML")"

echo "RUNE_MAX_LENGTH=${RUNE_MAX_LENGTH:-unset}"
echo "MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-unset}"

# 3. Resolve dataset path (local sample vs S3 sync).
DATASET_PATH=""
if [[ "${RUNE_PIPELINE_DATASET_SOURCE:-local}" == "s3" ]]; then
    [[ "${DRY_RUN}" -eq 1 ]] || aws s3 sync "${RUNE_PIPELINE_DATASET_URI}" data/
    DATASET_PATH="data/github-pairs/_merged/pairs_all.jsonl"
else
    DATASET_PATH="${RUNE_PIPELINE_DATASET_PATH:-data/sample/pairs_sample.jsonl}"
fi
echo "dataset=${DATASET_PATH}"

# 4. Optionally bring up the docker MLflow stack.
if [[ "${RUNE_PIPELINE_BRING_UP_DOCKER_STACK:-false}" == "true" ]]; then
    if [[ "${DRY_RUN}" -eq 0 ]]; then
        docker compose -f infra/docker-compose.yml up -d mlflow litestream
        # Wait for healthcheck.
        for _ in $(seq 1 30); do
            if curl -fsS --max-time 2 "${MLFLOW_TRACKING_URI%/}/health" >/dev/null; then
                break
            fi
            sleep 2
        done
    fi
fi

# 5. Run HPO.
HPO_CMD=(scripts/run_hpo.sh \
    --dataset "$DATASET_PATH" \
    --n-trials "${RUNE_PIPELINE_HPO_N_TRIALS:-30}" \
    --subsample "${RUNE_PIPELINE_HPO_SUBSAMPLE:-500}" \
    --keep-top-k "${RUNE_PIPELINE_HPO_KEEP_TOP_K:-3}")
echo "would run: ${HPO_CMD[*]}"
if [[ "${DRY_RUN}" -eq 0 ]]; then
    "${HPO_CMD[@]}"
fi

# 6. Train the champion.
LAUNCHER="${RUNE_PIPELINE_ACCELERATE_LAUNCHER:-}"
if [[ "$LAUNCHER" == "multi_gpu" ]]; then
    CHAMPION_CMD=(uv run accelerate launch \
        --config-file "${RUNE_PIPELINE_ACCELERATE_CONFIG:-infra/accelerate/multi_gpu.yaml}" \
        scripts/run_champion.py)
else
    CHAMPION_CMD=(uv run python scripts/run_champion.py)
fi
CHAMPION_CMD+=(\
    --study-db "sqlite:///./optuna_training.db" \
    --study-name "rune-training-v1" \
    --dataset "$DATASET_PATH" \
    --adapter-id "$ADAPTER_ID" \
    --epochs "${RUNE_PIPELINE_CHAMPION_EPOCHS:-3}")
echo "would run: ${CHAMPION_CMD[*]}"
if [[ "${DRY_RUN}" -eq 0 ]]; then
    "${CHAMPION_CMD[@]}"
fi

echo "Pipeline complete. Champion adapter id: ${ADAPTER_ID}"
echo "Load via: AdapterRegistry.default().get('${ADAPTER_ID}')"
```

Make it executable: `chmod +x scripts/run_pipeline.sh`.

- [ ] **Step 4: Run smoke test to verify it passes**

Run: `tests/test_run_pipeline_smoke.sh`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_pipeline.sh tests/test_run_pipeline_smoke.sh
git commit -m "feat(pipeline): one-command HPO + champion entry point"
```

---

## Task 12: Quickstart documentation

**Files:**
- Create: `docs/QUICKSTART.md`

- [ ] **Step 1: Write `docs/QUICKSTART.md`**

```markdown
# Quickstart

Run the full HPO + champion-training pipeline against any GPU,
including someone who just cloned this repo with no AWS access.

## Local-only (no AWS, no docker)

```bash
git clone https://github.com/ElixirTrials/rune.git
cd rune
uv sync --all-extras
scripts/run_pipeline.sh                         # auto-detects HW, local storage
```

This uses the bundled `data/sample/pairs_sample.jsonl` (50 records),
writes MLflow tracking to `./mlflow.db` (sqlite), and stores adapter
checkpoints under `./mlruns/`. Useful for verifying the pipeline runs
end-to-end on your hardware before committing to a real study.

## Cloud mode (full S3 + docker MLflow)

```bash
# Requires AWS credentials at ~/.aws and Docker.
docker compose -f infra/docker-compose.yml up -d mlflow litestream
scripts/run_pipeline.sh --storage cloud
```

Datasets sync from `s3://elixirtrials-…/training-data/`, MLflow runs
in the in-pod container, top-K adapter checkpoints land in
`s3://…/mlflow/artifacts/`.

## Picking the hardware profile

`run_pipeline.sh` runs `scripts/_lib/detect_hw.py` to map your GPU(s)
to a profile. Override with `--hw <profile>`:

| Profile             | Hardware                | Notes |
|---------------------|-------------------------|-------|
| `l4-22gb`           | Single L4 22GB          | QLoRA NF4, max_length 3072 |
| `a100-40gb`         | Single A100 40GB        | QLoRA, max_length 6144, flash-attn 2 |
| `a100-80gb`         | Single A100 80GB        | bf16 LoRA (no QLoRA), max_length 8192 |
| `a100-80gb-multi`   | Multi A100 80GB         | + accelerate DDP launcher |

To force a profile: `scripts/run_pipeline.sh --hw a100-80gb-multi`.

## What the pipeline produces

1. **HPO study**: `optuna_training.db` (sqlite). Inspect best trial via
   `optuna best-trial --storage sqlite:///./optuna_training.db --study-name rune-training-v1`.
2. **Top-K adapters**: in MLflow under the trial run's `adapter/`
   artifact path (S3 in cloud mode, local in standalone).
3. **Champion adapter**: registered in the AdapterRegistry under the
   `--adapter-id` you passed (defaults to `champion-<timestamp>`).

## Loading the champion

```python
from adapter_registry.registry import AdapterRegistry

reg = AdapterRegistry.default()
record = reg.get("champion-20260508-143000")
# record.adapter_path points to the local on-disk safetensors.
```
```

- [ ] **Step 2: Commit**

```bash
git add docs/QUICKSTART.md
git commit -m "docs: quickstart for plug-and-play pipeline"
```

---

## Task 13: End-to-end verification on real hardware

**Files:** none (operational task)

This task does not touch code — it's a manual gate before declaring the
pipeline shippable.

- [ ] **Step 1: Run the local-only path on the target dev box (L4 or CPU)**

```bash
scripts/run_pipeline.sh --hw l4-22gb --storage local
```

Expected: pipeline completes; champion adapter registered; the warning
`DiffAwareSFTTrainer: all-masked batch (denom=0.000e+00)` does NOT
appear in the log (the new `keep_end` truncation + `RUNE_MAX_LENGTH=3072`
should eliminate it on the sample dataset).

- [ ] **Step 2: Run the cloud path with one trial against full data**

```bash
scripts/run_pipeline.sh --storage cloud --hw l4-22gb
# (override n_trials inline if needed)
```

Verify: top-K adapters land in S3 under `mlflow/artifacts/`; champion
appears as a separate MLflow run; `train/all_masked_batch_frac_mean`
metric is logged.

- [ ] **Step 3: If A100 hardware is available, exercise it**

```bash
scripts/run_pipeline.sh --hw a100-80gb --storage cloud
# or
scripts/run_pipeline.sh --hw a100-80gb-multi --storage cloud
```

Verify: bf16 LoRA path doesn't OOM; `accelerate launch` spawns N
worker processes equal to `nvidia-smi -L | wc -l`; final adapter
quality is at least on par with the L4 baseline (compare hunk_loss
on a held-out set).

- [ ] **Step 4: Open the integration PR**

After all three exercises pass, open a PR with the full plan
implemented. Title: `feat(pipeline): plug-and-play HPO + champion
training across L4 / A100 / multi-A100`. Reference PR #35 (the
RCA-5 H2 / S3 top-K work this builds on).

---

## Self-Review

- **Spec coverage:** Each spec point in the brainstorming round trip is
  covered: hardware-tier yaml (Task 7), storage-tier yaml (Task 8),
  YAML loader (Task 5), HW detector (Task 6), bundled mini-dataset
  (Task 9), champion driver (Task 10), one-command entry point
  (Task 11), docs (Task 12), env-var plumbing (Tasks 1-4),
  end-to-end verification (Task 13).

- **Placeholder scan:** No "TBD"/"add error handling"/"similar to Task
  N" stubs. Every code step shows the full code.

- **Type consistency:** Function names referenced across tasks
  (`_resolve_max_length`, `_resolve_optim`, `_resolve_grad_accum`,
  `_resolve_attn_impl`, `_resolve_use_qlora`, `classify_gpus`,
  `_best_params_from_study`, `_build_champion_kwargs`) are defined in
  the same task they're first invoked. Env-var names
  (`RUNE_MAX_LENGTH`, `RUNE_OPTIM`, `RUNE_USE_QLORA`,
  `RUNE_GRAD_ACCUM`, `RUNE_ATTN_IMPL`) are spelled identically across
  Tasks 1-7.
