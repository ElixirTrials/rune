# Inference

Provider-agnostic inference interface for LLM generation and LoRA adapter management.

## Providers

| Provider | Module | Backend | LoRA Support |
|----------|--------|---------|-------------|
| `TransformersProvider` | `transformers_provider.py` | HuggingFace Transformers + PEFT | Full (load/unload/list) |
| `LlamaCppProvider` | `llamacpp_provider.py` | llama-cpp-python | Model-level |
| `OllamaProvider` | `ollama_provider.py` | Ollama HTTP API | Base model only (adapter ops raise `UnsupportedOperationError`) |
| `VLLMProvider` | `vllm_provider.py` | vLLM OpenAI-compatible API | Full (dynamic loading) |

## Base Interface

`InferenceProvider` (ABC in `provider.py`) defines:

```python
async def generate(prompt, model, adapter_id=None, max_tokens=4096) -> GenerationResult
async def load_adapter(adapter_id, adapter_path) -> None
async def unload_adapter(adapter_id) -> None
async def list_adapters() -> list[str]
```

`GenerationResult` fields: `text`, `model`, `adapter_id`, `token_count`, `finish_reason`.

## Factory

`factory.py` provides:
- `get_provider(config)` — Instantiate a provider from configuration
- `get_provider_for_step(step)` — Get provider configured for a pipeline step

## Lazy Loading

Provider classes are lazily imported via module `__getattr__` to avoid hard failures when backend packages (openai, torch, llama_cpp) are not installed.
