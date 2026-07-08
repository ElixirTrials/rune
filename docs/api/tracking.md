# Tracking

Lightweight MLflow integration providing a one-time configure_mlflow() (URI/experiment/LangChain autolog) and a tracked_run() context manager that starts a run and logs params.

## Planned

- **Prompt logging in Traces** — manual `mlflow.start_span(span_type=LLM)` around engine generation (Transformers path; autolog not supported). Register Jinja templates as Prompt Registry shells; `load_prompt()` inside traced spans for UI linkage. Partially superseded: `graph.py` already logs per-step trajectory/prompt/output text via `mlflow.log_text` plus `adapter_cond_tokens`/`prompt_tokens` metrics.

::: rune.tracking
