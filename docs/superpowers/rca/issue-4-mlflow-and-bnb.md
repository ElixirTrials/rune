# RCA: Issue #4 — MLflow "already active" + BNB fp32 offload warning

**Log line (raw, as captured):**
```
2026-04-27 17:49:34,194 [INFO] model_training.training_common: MLflow already activ these modules in 32-bit, you need to set `llm_int8_enable_fp32_cpu_offload=True` and pass a custom `device_map` to `from_pretrained`.
```

This is two log lines printed to the same terminal position without a newline separator — a race or buffering artifact between the Python `logging` handler and the bitsandbytes C-extension `print()` call. The two problems are independent.

---

## (a) "MLflow already active" — run nesting

### What fires and where

`setup_mlflow()` in `libs/model-training/src/model_training/training_common.py:53–59` calls `mlflow.active_run()`. When a run is already open it emits:

```python
logger.info(
    "MLflow already active under run_id=%s uri=%s; reusing",
    active.info.run_id,
    mlflow.get_tracking_uri(),
)
```

This is level `INFO`, not `WARNING`. The truncated log shows the string up to `"MLflow already activ"` — the rest of the message was overwritten by the bnb `print()` on the same line.

### Run nesting path

`run_training_hpo.py:731` calls `mlflow.start_run()` without a context manager — the run stays open for the duration of the trial. Then `train_and_register()` calls `setup_mlflow()` (`trainer.py:891`) and `mlflow_run()` (`trainer.py:952`). Both check `mlflow.active_run() is not None` and take the "reuse" branch — they attach params to the HPO-owned run rather than opening a nested one. This is the intended design per the comment at `training_common.py:116–130`.

### Is it benign?

Mostly yes: the TRL `MLflowCallback` (wired via `report_to="mlflow"`) attaches to the same active run, so per-step training metrics land in the correct run. However there is a subtle metric-routing risk:

- `mlflow_run()` at `trainer.py:952` enters the "already active" branch and **does not call `mlflow.end_run()`** — it returns immediately after logging params. The HPO harness owns `end_run()` at lines 769/772.
- If training crashes after `mlflow_run.__enter__` but before `train_and_register` bubbles the exception to the HPO harness's `except BaseException` block (`run_training_hpo.py:768`), there is a window where `end_run(status="FAILED")` is never called and the run stays in `RUNNING` state indefinitely. MLflow server-side will eventually time out the run, but metrics logged up to that point may not be flushed.

---

## (b) BNB fp32 offload warning

### The error text

> `you need to set llm_int8_enable_fp32_cpu_offload=True and pass a custom device_map to from_pretrained`

Despite the mention of `llm_int8`, this error fires for NF4 (4-bit) loads too when `accelerate` decides to offload layers to CPU. The bitsandbytes check (`transformers/integrations/bitsandbytes.py`) raises this if any module in the device map targets `"cpu"` and `llm_int8_enable_fp32_cpu_offload` is not set.

### Device map in use

Both `_get_or_load_base()` (`trainer.py:129`) and the eval path (`_evaluate_adapter_on_heldout` → `_get_or_load_base` at `run_training_hpo.py:505`) pass `device_map="auto"`. With `"auto"`, `accelerate` computes a per-layer placement based on available VRAM.

### Does NF4 Qwen3.5-9B fit on a 22 GB GPU?

The base NF4 weights are approximately 4.8–5.2 GB. At the start of Trial 2 the OOM trace from Issue #2 shows 21.49 GB already in use (`training_issues.md:6`). In that state, `device_map="auto"` on a fresh `_get_or_load_base()` call for the eval path — if the base cache is cold or has been evicted — would find only ~0.5 GB free and would begin offloading transformer layers to CPU. The bitsandbytes integration then fires this warning/error because it cannot quantize CPU-resident modules in 4-bit.

### When is this emitted — load time or forward time?

The bnb warning fires at **load time** inside `from_pretrained`, not during the forward pass. The timestamp `17:49:34,194` is 5.459 seconds after Trial 1's OOM at `17:49:28,735` (`training_issues.md:2`). This aligns with Trial 2 starting immediately after Trial 1 fails, the CUDA cache being partially freed, and `_get_or_load_base` attempting a fresh load with `device_map="auto"` finding insufficient VRAM. `llm_int8_enable_fp32_cpu_offload` is not set anywhere in the codebase (confirmed by grep).

The key question is whether the base model cache (`RUNE_PERSIST_BASE_MODEL=1`) was hot at this point. If it was, `_get_or_load_base` returns immediately without calling `from_pretrained` and the bnb error would not fire here. The log line at `17:49:34,190` (`training_issues.md:92`) shows "LoRA alpha override" — which happens *after* model loading — suggesting the cache **miss** path was taken (or the bnb error is coming from `PeftModel.from_pretrained` at `run_training_hpo.py:512` during heldout eval, not the base load).

---

## Connection to Issue #2 (OOM) and Issue #5 ("loss not going down")

**Issue #2 link:** The bnb offload warning is a direct symptom of the same VRAM pressure that caused OOM in Trial 1. If accelerate successfully offloads layers to CPU (fp32), forward passes become CPU↔GPU transfers, inflating memory usage further and making a second OOM more likely. It also degrades throughput to the point where the HPO budget is consumed by slow CPU-offloaded trials rather than real GPU training.

**Issue #5 link (no-learning hypothesis):** There is a plausible metrics-routing failure. When Trial 1 crashes via OOM, `mlflow.end_run(status="FAILED")` is called at `run_training_hpo.py:769`. Trial 2 then opens a **new** `mlflow.start_run()` at line 731. However, if the TRL `MLflowCallback` was initialized inside Trial 1's training run and holds a stale reference to Trial 1's `run_id`, any metrics it flushes after the trial boundary would be written to the orphaned (FAILED) run rather than Trial 2's run. This would make Trial 2's training loss metrics disappear from the active run — the "loss is not going down" in the UI would actually be the absence of metrics, not evidence of non-learning.

---

## Recommended fix directions (no implementation)

**MLflow (a):**
- Confirm `mlflow.end_run()` is always reached even when `train_and_register` raises inside the `mlflow_run` context manager while the HPO outer run is active. Consider whether `mlflow_run` should call `mlflow.end_run()` for the outer run or simply ensure flush via `mlflow.flush()` before re-raising.
- Reinitialize the TRL `MLflowCallback` at the start of each trial rather than relying on it attaching to whatever run is active at callback init time.

**BNB / device_map (b):**
- Set `llm_int8_enable_fp32_cpu_offload=True` in `_get_or_load_base()`'s `model_kwargs` alongside `device_map="auto"` to suppress the error and allow graceful CPU offload.
- Alternatively, gate `device_map="auto"` with a VRAM check: if free VRAM < model_size_estimate, either raise early with a clear message or force `device_map="cuda:0"` and let torch OOM cleanly.
- Ensure `RUNE_PERSIST_BASE_MODEL=1` is set for all HPO runs so the base is loaded once and the eval path (`_evaluate_adapter_on_heldout`) always hits the cache — eliminating the second `from_pretrained` that triggers the offload.
