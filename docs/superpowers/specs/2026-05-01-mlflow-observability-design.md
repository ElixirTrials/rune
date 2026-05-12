# MLflow Observability — Design Spec

**Date:** 2026-05-01
**Status:** Approved (brainstorming session 2026-05-01)
**Owner:** Noah Dolev

## Problem

The Rune project has four training paths logging to MLflow with inconsistent
conventions, no eval-during-training (except `encoder_pretrain`), and no GenAI
tracing for the 5-phase pipeline. The user wants:

1. Live "tensorboard-like" monitoring of fine-tuning to detect underfit/overfit
   in real time.
2. The ability to view loss per-step OR per-epoch without logging the same
   metric under two different keys.
3. GenAI traces of pipeline runs (5 phases × LLM calls) cross-linked to the
   training runs that produced the adapter being evaluated.

## Decisions (confirmed against MLflow docs)

- MLflow per-run charts only support **Step / Wall Time / Relative Time** as
  X-axis. There is no native "epoch" X-axis on the live single-run chart.
- The canonical pattern is **one global step axis**, with `epoch` co-logged as
  its own metric for visual cross-reference.
- The per-step / per-epoch granularity choice is satisfied by logging
  `train/loss` every step **and** `eval/loss` every epoch as **distinct
  signals** (not duplicates). On a single chart they form the canonical
  underfit/overfit view.
- TRL's `report_to="mlflow"` already emits `train/loss`, `train/lr`,
  `train/grad_norm`, `train/epoch`, `train/global_step`, and `eval/loss` for
  the QLoRA path. We **extend** that callback chain rather than replace it.
- `@mlflow.trace` is OpenTelemetry-based and nests naturally — phase functions
  decorated with `@trace_phase` produce parent spans; LLM call sites produce
  child spans.
- HPO via Optuna uses parent-run/child-run nesting (already the project's
  pattern in `run_training_hpo.py`).

## Scope

**In:**
- Unify MLflow metric/param/artifact logging across `trainer.py` (QLoRA),
  `d2l_train.py` (Doc-to-LoRA hypernet),
  `encoder_pretrain/train_encoder.py`, and `run_training_hpo.py`.
- Add eval-cadence logging to all three training paths (currently only
  `encoder_pretrain` has any eval-during-training).
- Add GenAI tracing to the 5-phase pipeline in `scripts/rune_runner.py` and
  the inference provider in `libs/inference/`.
- Plot artifacts: per-run `loss_vs_epoch.png`, HPO parent-run plots
  (parallel coordinate, contour, importance, optimization history).
- Run-tagging strategy that links training runs ↔ pipeline-inference traces
  via `rune.adapter_id`.

**Out:**
- Refactoring training logic (loss functions, schedulers, etc.).
- Replacing TRL's MLflow callback. We extend it.
- Self-managing the MLflow tracking server. Document the recommended
  `mlflow ui` invocation; assume the user runs it.
- Per-layer grad-norm or weight histograms (deferrable).

## Module layout

New code lives in `libs/model-training/src/model_training/observability/`:

```
observability/
├── __init__.py            # public exports
├── conventions.py         # metric-name constants, tag-key constants, run-name builders
├── mlflow_logger.py       # MetricLogger, RunContext, log_figure helpers
├── trainer_callback.py    # RuneMlflowCallback (HF TrainerCallback subclass)
├── tracer.py              # @trace_phase, trace_llm_call, span_type helpers
└── plots.py               # loss_vs_epoch_figure, hpo_summary_figures
```

Public surface (importable from `model_training.observability`):

- `setup_mlflow(experiment, tracking_uri)` — replaces existing
  `training_common.setup_mlflow`.
- `RunContext(...)` — context manager wrapping `mlflow.start_run` with tag and
  param defaults; supports `nested=True` automatically when an MLflow run is
  already active.
- `MetricLogger` — `.log_train(metrics, step)`, `.log_eval(metrics, step)`,
  `.close()`.
- `RuneMlflowCallback` — drop into `trainer.callback_handler`.
- `@trace_phase("name")`, `trace_llm_call(...)` — for pipeline/inference.

`training_common.py` keeps its existing function names as **thin shims** that
import from the new module so call sites can migrate incrementally.

## Metric naming and cadence

Final naming convention (matches TRL's prefixes verbatim):

| Metric | Logged by | When | Used for |
|---|---|---|---|
| `train/loss` | TRL callback (QLoRA), MetricLogger (d2l, encoder) | every step | live underfit detection |
| `train/lr` | same | every step | schedule sanity check |
| `train/grad_norm` | same | every step | gradient health |
| `train/epoch` | same | every step | visual step→epoch mapping |
| `train/<diff_metric>` | `DiffAwareSFTTrainer.log` (already exists) | every step | diff-aware loss diagnostics |
| `eval/loss` | RuneMlflowCallback (QLoRA), MetricLogger (others) | every epoch | overfit detection |
| `eval/<task_metric>` | same | every epoch | e.g. `eval/mrr_at_10` for encoder |
| `wall_time_s` | RuneMlflowCallback / MetricLogger | every step | wall-clock view |

**Step semantics:** every metric uses `step=global_step` where `global_step`
is the trainer's step counter. No metric is ever logged at two different
step encodings.

## Tag keys

| Tag | Value | Purpose |
|---|---|---|
| `rune.adapter_id` | adapter-registry id | links training run ↔ inference traces |
| `rune.model_name` | base model id | filter by model |
| `rune.task_type` | `qlora_sft` / `d2l_round2` / `encoder_pretrain` / `pipeline_inference` / `hpo_study` / `hpo_trial` | experiment-level filter |
| `rune.git_sha` | current commit | reproducibility |
| `rune.host` | hostname / GPU id | debug noisy nodes |

## RuneMlflowCallback (HF TrainerCallback subclass)

Drop into `Trainer.callback_handler` alongside TRL's built-in `MLflowCallback`.
Responsibilities:

- `on_train_begin`: set the five run tags above. Log `wall_time_s=0`.
- `on_step_end`: log `wall_time_s` (monotonic delta from `on_train_begin`).
- `on_evaluate`: receives HF's eval metrics dict, prefixes any unprefixed
  keys with `eval/`, logs at current `state.global_step`.
- `on_train_end`: call `plots.log_loss_vs_epoch_figure(run_id)` to attach the
  static `loss_vs_epoch.png`. Reads its data via
  `MlflowClient.get_metric_history`.

Does **not** log `train/loss`, `train/lr`, `train/epoch`, `train/grad_norm`,
or `eval/loss` itself — TRL's existing callback already emits those when
`report_to="mlflow"`. We layer.

## MetricLogger (for d2l_train + encoder_pretrain)

Plain Python class — no HF Trainer involved in those paths.

```python
class MetricLogger:
    def __init__(self, run_context: RunContext, *, total_steps: int, steps_per_epoch: int): ...
    def log_train(self, metrics: dict[str, float], *, step: int) -> None:
        # adds train/ prefix where missing, co-logs train/epoch and wall_time_s
    def log_eval(self, metrics: dict[str, float], *, step: int) -> None:
        # adds eval/ prefix, same step as the train metrics it's evaluated against
    def close(self) -> None:
        # logs loss_vs_epoch.png artifact
```

Replaces the manual `mlflow.log_metrics(metrics, step=step)` calls in
`d2l_train.py:687` and `encoder_pretrain/train_encoder.py:332`.

## Pipeline tracer

```python
@trace_phase("decompose")  # SpanType.CHAIN, captures inputs/outputs
def decompose(...): ...

# Inside an LLM call site:
def call_model(prompt, adapter_id):
    with mlflow.start_span(name="inference", span_type=SpanType.LLM) as span:
        span.set_inputs({"prompt": prompt, "adapter_id": adapter_id})
        out = provider.generate(prompt)
        span.set_outputs({"text": out, "tokens": ...})
        return out
```

`trace_phase` is a thin wrapper around
`@mlflow.trace(span_type=SpanType.CHAIN, name=...)` that also tags the active
trace with `rune.adapter_id` and `rune.task_type=pipeline_inference` on first
call. One trace per pipeline invocation; phase functions become children;
`call_model` calls become grandchildren.

Tracer is **opt-in** via env var `RUNE_MLFLOW_TRACE=1`. When off,
`@trace_phase` is identity, `start_span` is a `nullcontext`.

## Eval-dataset plumbing per trainer

| Path | Today | Change |
|---|---|---|
| `trainer.py` (QLoRA) | `eval_strategy="no"`, no eval dataset | Accept new `eval_dataset` arg in `train_qlora`; when provided, set `eval_strategy="epoch"`, `per_device_eval_batch_size=1`. Default None preserves current behavior. |
| `d2l_train.py` | None | Add `eval_steps` config field (default disabled); when set, every N steps run a small held-out val pass and `log_eval({"loss": ...}, step=step)`. |
| `encoder_pretrain` | Already evals per-epoch | Rename current keys to `eval/mrr_at_10`, `eval/recall_at_1`. |

Eval datasets come from the same source as training (split off held-out
indices before passing to trainer) — no new loader code.

## Plots

- `log_loss_vs_epoch_figure(run_id)` — pulls `train/loss` and `train/epoch`
  history via `MlflowClient.get_metric_history`, plots loss vs epoch with
  matplotlib, calls `mlflow.log_figure(fig, "plots/loss_vs_epoch.png")`.
- `log_hpo_summary_figures(study, run_id)` — at end of Optuna study,
  generates parallel coordinate / contour / parameter importance /
  optimization history plots using `optuna.visualization.matplotlib`, logs
  each as `plots/hpo_*.png`.

## HPO integration

`run_training_hpo.py` keeps its parent-run/child-run shape but routes through
`RunContext` for tag consistency. Each trial's child run gets
`rune.task_type=hpo_trial`, parent `rune.task_type=hpo_study`. Each trial's
`MetricLogger` is constructed with `nested=True` automatically when an MLflow
run is already active. After `study.optimize` returns,
`log_hpo_summary_figures(study, parent_run_id)` is called.

## Error handling

- `setup_mlflow` returns `False` (silent no-op) when MLflow is missing or
  tracking URI unreachable. All downstream code is gated on `mlflow_enabled`.
- `MetricLogger.log_*` and `RuneMlflowCallback` catch `Exception` from MLflow
  API calls, log at DEBUG, never raise. Training never fails because of
  telemetry.
- Tracer is opt-in via `RUNE_MLFLOW_TRACE=1` to avoid trace-volume blowup in
  swarm runs.

## Testing

New tests in `libs/model-training/tests/`:

- `test_observability_conventions.py` — metric-name constants stable, tag
  keys stable.
- `test_metric_logger.py` — fakes `mlflow` module, asserts correct
  prefix/step/epoch co-logging, no duplicates.
- `test_runeml_callback.py` — instantiate with stub `TrainerState` /
  `TrainingArguments`, assert tag-set + eval-prefix logic.
- `test_tracer.py` — `RUNE_MLFLOW_TRACE=0` → identity; `=1` → spans created
  (mock `mlflow.start_span`).
- `test_plots.py` — generates a tiny figure, asserts `log_figure` called with
  right path.

Existing tests in `test_training_common.py` continue to pass via the shim.

## Migration plan (single PR)

1. Add the `observability/` module + tests (no callers yet).
2. Switch `training_common.setup_mlflow` to a shim importing from
   observability.
3. Update `trainer.py`: register `RuneMlflowCallback`, accept `eval_dataset`
   arg.
4. Update `d2l_train.py`: replace `mlflow.log_metrics` with `MetricLogger`,
   add eval-step config.
5. Update `encoder_pretrain/train_encoder.py`: replace direct
   `mlflow.log_metrics` with `MetricLogger`, rename eval keys.
6. Update `run_training_hpo.py`: route through `RunContext`, add
   `log_hpo_summary_figures` at study end.
7. Add `pipeline_tracer.py`; decorate the 5 phase functions in
   `scripts/rune_runner.py` and the LLM call site in
   `libs/inference/provider.py`.
8. Run full test suite + `ruff` + `mypy`.
9. Update `docs/` with a brief "MLflow observability" page describing the
   conventions and how to launch `mlflow ui`.

## Acceptance criteria

- All four training paths log `train/loss`, `train/lr`, `train/grad_norm`,
  `train/epoch`, `wall_time_s` with identical metric names.
- All four log `eval/loss` (or task-specific `eval/*`) when an eval dataset
  is provided.
- `train/loss` + `eval/loss` plotted on the same MLflow chart shows the
  canonical underfit/overfit view.
- A pipeline run with `RUNE_MLFLOW_TRACE=1` produces a single trace with one
  child span per phase and grandchild spans per LLM call, all tagged with
  `rune.adapter_id`.
- HPO study run shows parent run with attached `plots/hpo_*.png` and N child
  runs (one per trial), each tagged `rune.task_type=hpo_trial`.
- `uv run pytest`, `uv run ruff check`, `uv run mypy libs/ services/` all pass.
