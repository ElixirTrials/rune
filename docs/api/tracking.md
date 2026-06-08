# Tracking

Lightweight MLflow integration providing a one-time configure_mlflow() (URI/experiment/LangChain autolog) and a tracked_run() context manager that starts a run and logs params.

## Planned

- **Prompt logging in Traces** — manual `mlflow.start_span(span_type=LLM)` around engine generation (Transformers path; autolog not supported). Register Jinja templates as Prompt Registry shells; `load_prompt()` inside traced spans for UI linkage. See `instructions/scratchpad.md` block `[2026-06-05 21:20 UTC]`.

::: rune.tracking
