# Phase 19: Inference Provider Abstraction - Research

**Researched:** 2026-03-05
**Domain:** Python ABC async providers, vLLM dynamic LoRA API, Ollama OpenAI-compat, Docker/docker-compose
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Provider Interface Shape**
- Replace existing module-level stubs (completion.py, adapter_loader.py) with an abstract `InferenceProvider` ABC — clean break, no legacy wrappers
- `VLLMClient` in lora-server/ becomes `VLLMProvider` implementing the interface
- All InferenceProvider methods are **async** (awaitable) — both backends are HTTP-based, async is natural
- Keep the provider in `libs/inference/` — refactor in-place, no new workspace lib
- `generate()` returns a **structured result** (dataclass/Pydantic model) with fields: text, model, adapter_id, token_count, finish_reason — agent loop needs metadata

**Configuration & Backend Selection**
- Default provider selected by env var: `INFERENCE_PROVIDER=vllm|ollama` — consistent with `DATABASE_URL` and `VLLM_BASE_URL` patterns
- Per-backend URL env vars: keep `VLLM_BASE_URL`, add `OLLAMA_BASE_URL` (default http://localhost:11434)
- Per-step model/provider config via dict mapping at agent init: `{"generate": {"provider": "vllm", "model": "...", "adapter": "..."}, "reflect": {"provider": "ollama", "model": "..."}}`
- Factory caches provider instances keyed by (provider_type, base_url) — avoids redundant HTTP clients when multiple steps use the same backend

**Ollama Adapter Handling**
- `OllamaProvider.load_adapter()` raises `UnsupportedOperationError` — explicit failure, no silent no-ops
- `OllamaProvider.generate()` fully works for base model completions — Ollama is useful for non-adapter steps (e.g., reflection)
- Use httpx async client directly against Ollama's REST API — no extra ollama-python dependency
- Use Ollama's **OpenAI-compatible endpoint** (`/v1/chat/completions`) — symmetrical with VLLMProvider, could even share the openai SDK

**lora-server Docker & Port Layout**
- Dockerfile base image: `vllm/vllm-openai:v0.16.0` (replaces python:3.12-slim) — per INFRA-01
- `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True` env var set in Dockerfile — per INFRA-02
- lora-server host port: **8100** (container port stays 8000, mapped to 8100 on host). Update `VLLM_BASE_URL` default to `http://localhost:8100/v1`
- api-service stays on port 8000
- Keep existing health sidecar on port 8001 — no change to health.py
- Create minimal `docker-compose.yml` with api-service (8000) and lora-server (8100), shared SQLite volume mount — satisfies INFRA-03

### Claude's Discretion
- Exact InferenceProvider ABC method signatures beyond generate/load_adapter/unload_adapter/list_adapters
- Internal error handling and retry logic within providers
- Structured result dataclass field names and optional fields
- Factory implementation pattern (module-level function vs class)
- Test strategy for providers (mocked HTTP vs integration)

### Deferred Ideas (OUT OF SCOPE)
- HuggingFace model pulling utility — easy download/cache management for base models and adapter weights from HF Hub. vLLM auto-downloads on first use, but a dedicated tool for pre-fetching and browsing available models would be useful. Future phase.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| INF-01 | Abstract InferenceProvider interface with generate(), load_adapter(), unload_adapter(), list_adapters() methods | Python ABC with @abstractmethod + async def; all four methods verified against interface shape from CONTEXT.md |
| INF-02 | VLLMProvider implementation with full LoRA hot-loading support (POST /v1/load_lora_adapter, /v1/unload_lora_adapter) | vLLM dynamic LoRA API verified: POST /v1/load_lora_adapter with {lora_name, lora_path}, POST /v1/unload_lora_adapter with {lora_name}; adapter used by passing lora_name as model parameter in chat completions |
| INF-03 | OllamaProvider implementation for inference via Ollama API (LoRA support where Ollama supports it) | Ollama /v1/chat/completions confirmed; no LoRA support — OllamaProvider.load_adapter() raises UnsupportedOperationError per locked decision |
| INF-04 | Provider factory/registry for selecting backend by configuration (env var or config file) | Factory pattern: module-level function or class; INFERENCE_PROVIDER env var; caches (provider_type, base_url) key |
| INF-05 | User can generate with a specific loaded adapter by passing adapter name as model parameter | vLLM confirmed: adapter_name passed as model parameter to /v1/chat/completions; VLLMProvider.generate(adapter_id=...) maps to model=adapter_id |
| INF-06 | User can load multiple adapters simultaneously for composition (provider-dependent, graceful degradation) | vLLM max_loras=8 already configured; each load_adapter call independent; OllamaProvider raises UnsupportedOperationError |
| INF-07 | Per-step model/provider configuration — agent can use different models or providers for different steps | Factory keyed by (provider_type, base_url) caches instances; per-step dict {"generate": {"provider": "vllm", ...}, "reflect": {"provider": "ollama", ...}} creates correct provider per step |
| INFRA-01 | lora-server Dockerfile uses vllm/vllm-openai:v0.16.0 base image (not python:3.12-slim) | Current Dockerfile uses python:3.12-slim — must be replaced with vllm/vllm-openai:v0.16.0; vllm/vllm-openai image bundles vLLM engine and CUDA runtime |
| INFRA-02 | lora-server sets VLLM_ALLOW_RUNTIME_LORA_UPDATING=True environment variable | Must add ENV VLLM_ALLOW_RUNTIME_LORA_UPDATING=True to Dockerfile; startup.sh will inherit it |
| INFRA-03 | docker-compose resolves port conflict (api-service and lora-server on different host ports) | Existing infra/docker-compose.yml maps both api and lora-server to host port 8000 — conflict. New docker-compose.yml: api-service on 8000, lora-server on 8100, shared SQLite volume |
</phase_requirements>

## Summary

Phase 19 converts the existing stubs in `libs/inference/` and `services/lora-server/` into a working provider-abstraction layer. The stubs are comprehensive but raise `NotImplementedError` throughout — the implementation work is replacing those stubs with real HTTP calls against vLLM and Ollama.

The vLLM dynamic LoRA API is verified: `POST /v1/load_lora_adapter` with `{"lora_name": "...", "lora_path": "..."}` loads an adapter, `POST /v1/unload_lora_adapter` with `{"lora_name": "..."}` removes it, and the loaded adapter is used in generation by passing `lora_name` as the `model` parameter to `/v1/chat/completions`. This maps cleanly to `VLLMProvider.generate(model=..., adapter_id=...)` where `adapter_id` becomes the `model` value when set. Ollama supports `/v1/chat/completions` via the same OpenAI SDK, making both providers structurally identical in their HTTP layer.

The infrastructure work (INFRA-01/02/03) is a discrete 3-file change: update the lora-server Dockerfile base image, add the env var, update LoraServerConfig port default to 8000→8100 (container side) and create a new minimal `docker-compose.yml` in the project root (or a rune/ subfolder) that puts api-service on 8000 and lora-server on 8100 with a shared SQLite volume.

**Primary recommendation:** Implement in this order: (1) `InferenceProvider` ABC + `GenerationResult` dataclass, (2) `VLLMProvider` refactored from existing `VLLMClient`, (3) `OllamaProvider`, (4) provider factory, (5) infrastructure files (Dockerfile, docker-compose), (6) update `__init__.py` exports. Each step is independently testable with mocked HTTP.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| openai | >=1.0.0 | AsyncOpenAI client for both vLLM and Ollama OpenAI-compat APIs | Already in inference pyproject.toml; both backends support OpenAI-compat |
| httpx | >=0.28.1 | Async HTTP for Ollama fallback if OpenAI SDK insufficient | Already in workspace deps; used in health.py |
| pydantic | >=2.0.0 | GenerationResult structured dataclass with validation | Already in inference pyproject.toml |
| abc (stdlib) | stdlib | Abstract base class for InferenceProvider | No install needed; built-in |
| pytest-asyncio | >=1.3.0 | Async test support (asyncio_mode = "auto" in root pyproject.toml) | Already in workspace; root pyproject.toml sets asyncio_mode = "auto" |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| unittest.mock (stdlib) | stdlib | AsyncMock for provider HTTP clients in tests | No install; all HTTP mocking via AsyncMock + patch |
| docker-compose | host tool | Orchestrate api-service + lora-server | Only for INFRA-03; no Python dependency |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| openai SDK for Ollama | httpx direct | OpenAI SDK over Ollama's /v1/chat/completions is cleaner and already installed; httpx only needed if Ollama returns non-OpenAI responses |
| Pydantic model for GenerationResult | stdlib dataclass | Pydantic already in deps and gives validation; stdlib dataclass is simpler but offers no validation — discretionary choice |
| Module-level factory function | Factory class | Module-level function with module-level dict cache is simpler; class needed only if factory needs lifecycle hooks |

**Installation:** No new dependencies required. `openai>=1.0.0`, `pydantic>=2.0.0`, `httpx>=0.28.1` are already declared in workspace.

## Architecture Patterns

### Recommended Project Structure
```
libs/inference/src/inference/
├── __init__.py          # Update exports: InferenceProvider, VLLMProvider, OllamaProvider, get_provider, GenerationResult
├── provider.py          # InferenceProvider ABC + GenerationResult dataclass  (NEW)
├── vllm_provider.py     # VLLMProvider (refactored from adapter_loader.py + completion.py)  (NEW)
├── ollama_provider.py   # OllamaProvider  (NEW)
├── factory.py           # get_provider() factory with instance cache  (NEW)
├── exceptions.py        # UnsupportedOperationError + existing exception types  (NEW or add to existing)
├── adapter_loader.py    # DELETE or keep as deprecated shim (clean break per CONTEXT.md)
└── completion.py        # DELETE or keep as deprecated shim (clean break per CONTEXT.md)

services/lora-server/
├── vllm_client.py       # DELETE or absorb into VLLMProvider in libs/inference
├── config.py            # Update port default: 8000 → stays 8000 (container); update VLLM_BASE_URL default
├── Dockerfile           # Replace base image, add ENV var
└── startup.sh           # Update --max-loras default per existing config

infra/
└── docker-compose.yml   # UPDATE: fix port conflict, add SQLite volume

# OR create new:
docker-compose.yml       # New minimal compose in project root (INFRA-03 says "create")
```

### Pattern 1: InferenceProvider ABC with Async Abstract Methods

**What:** Abstract base class using Python `abc.ABC` with `@abstractmethod` decorating each `async def` method.
**When to use:** All four methods — generate, load_adapter, unload_adapter, list_adapters must be declared abstract and async.

```python
# Source: Python docs — abc module; pattern confirmed for async methods
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class GenerationResult:
    """Structured response from any InferenceProvider.generate() call."""
    text: str
    model: str
    adapter_id: str | None
    token_count: int
    finish_reason: str


class InferenceProvider(ABC):
    """Provider-agnostic interface for LLM inference and adapter management."""

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        model: str,
        adapter_id: str | None = None,
        max_tokens: int = 1024,
    ) -> GenerationResult:
        """Generate a completion, optionally with a LoRA adapter."""
        ...

    @abstractmethod
    async def load_adapter(self, adapter_id: str, adapter_path: str) -> None:
        """Hot-load a LoRA adapter into the running backend."""
        ...

    @abstractmethod
    async def unload_adapter(self, adapter_id: str) -> None:
        """Remove a loaded LoRA adapter from the backend."""
        ...

    @abstractmethod
    async def list_adapters(self) -> list[str]:
        """Return names of all currently loaded adapters."""
        ...
```

**Key points:**
- Concrete subclass that doesn't implement all four methods cannot be instantiated — ABC enforces this.
- Python does NOT enforce that the override is also `async`. Type checkers (mypy strict) will catch sync-override-of-async-abstract, but runtime does not. Tests must verify coroutine semantics explicitly.
- `GenerationResult` as a `dataclass` is the simplest option aligned with project patterns (LoraServerConfig is a dataclass). Use `@dataclass` not Pydantic unless validation is needed.

### Pattern 2: VLLMProvider Using AsyncOpenAI

**What:** Wrap existing `VLLMClient` logic from `lora-server/vllm_client.py` into a proper `InferenceProvider` subclass.
**When to use:** VLLMProvider is the primary adapter-capable backend.

```python
# Source: vLLM docs https://docs.vllm.ai/en/v0.8.1/features/lora.html
#         openai SDK async pattern from existing adapter_loader.py
import os
from openai import AsyncOpenAI

VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8100/v1")


class VLLMProvider(InferenceProvider):
    """InferenceProvider backed by a local vLLM server."""

    def __init__(self, base_url: str | None = None) -> None:
        self._base_url = base_url or VLLM_BASE_URL
        self._client = AsyncOpenAI(
            base_url=self._base_url,
            api_key="not-needed-for-local-vllm",
        )

    async def generate(
        self,
        prompt: str,
        model: str,
        adapter_id: str | None = None,
        max_tokens: int = 1024,
    ) -> GenerationResult:
        # When adapter_id is set, pass it as the model name.
        # vLLM identifies loaded adapters by the lora_name passed to load_adapter.
        effective_model = adapter_id if adapter_id is not None else model
        response = await self._client.chat.completions.create(
            model=effective_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )
        choice = response.choices[0]
        return GenerationResult(
            text=choice.message.content or "",
            model=model,
            adapter_id=adapter_id,
            token_count=response.usage.total_tokens if response.usage else 0,
            finish_reason=choice.finish_reason or "unknown",
        )

    async def load_adapter(self, adapter_id: str, adapter_path: str) -> None:
        # POST /v1/load_lora_adapter with lora_name + lora_path
        import httpx
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self._base_url.rstrip('/v1')}/v1/load_lora_adapter",
                json={"lora_name": adapter_id, "lora_path": adapter_path},
                timeout=30.0,
            )
            resp.raise_for_status()

    async def unload_adapter(self, adapter_id: str) -> None:
        import httpx
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self._base_url.rstrip('/v1')}/v1/unload_lora_adapter",
                json={"lora_name": adapter_id},
                timeout=10.0,
            )
            resp.raise_for_status()

    async def list_adapters(self) -> list[str]:
        # GET /v1/models returns base model + loaded adapters.
        # Known vLLM issue: dynamically-loaded adapters may not appear in /v1/models.
        # Maintain a local set as fallback if /v1/models is insufficient.
        models = await self._client.models.list()
        return [m.id for m in models.data]
```

**Note on list_adapters:** There is a known vLLM bug (issue #11761) where dynamically-loaded adapters do not always appear in GET /v1/models after load_lora_adapter. Consider maintaining an internal set of loaded adapter names in VLLMProvider as a reliable fallback.

### Pattern 3: OllamaProvider Using AsyncOpenAI Against /v1/chat/completions

**What:** OllamaProvider uses the same AsyncOpenAI client pattern but points to Ollama's OpenAI-compat endpoint.
**When to use:** Non-adapter steps (reflection, summarization) where Ollama serves a base model.

```python
# Source: Ollama OpenAI compat docs https://docs.ollama.com/api/openai-compatibility
import os
from openai import AsyncOpenAI
from inference.exceptions import UnsupportedOperationError

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")


class OllamaProvider(InferenceProvider):
    """InferenceProvider backed by Ollama (base model inference only)."""

    def __init__(self, base_url: str | None = None) -> None:
        self._base_url = base_url or OLLAMA_BASE_URL
        self._client = AsyncOpenAI(
            base_url=self._base_url,
            api_key="ollama",  # Ollama requires non-empty but ignores value
        )

    async def generate(
        self,
        prompt: str,
        model: str,
        adapter_id: str | None = None,
        max_tokens: int = 1024,
    ) -> GenerationResult:
        response = await self._client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )
        choice = response.choices[0]
        return GenerationResult(
            text=choice.message.content or "",
            model=model,
            adapter_id=None,  # Ollama does not support adapters
            token_count=response.usage.total_tokens if response.usage else 0,
            finish_reason=choice.finish_reason or "unknown",
        )

    async def load_adapter(self, adapter_id: str, adapter_path: str) -> None:
        raise UnsupportedOperationError(
            "OllamaProvider does not support LoRA adapter loading. "
            "Use VLLMProvider for adapter operations."
        )

    async def unload_adapter(self, adapter_id: str) -> None:
        raise UnsupportedOperationError(
            "OllamaProvider does not support LoRA adapter unloading."
        )

    async def list_adapters(self) -> list[str]:
        return []  # Ollama has no adapter concept; empty list is correct
```

### Pattern 4: Provider Factory with Instance Cache

**What:** Module-level function `get_provider()` reads `INFERENCE_PROVIDER` env var, creates the correct backend, and caches by `(provider_type, base_url)`.
**When to use:** All consumer code uses `get_provider()` rather than instantiating providers directly.

```python
# Source: project patterns from adapter_loader.py (env var + module-level default)
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from inference.provider import InferenceProvider

_provider_cache: dict[tuple[str, str], "InferenceProvider"] = {}

INFERENCE_PROVIDER = os.getenv("INFERENCE_PROVIDER", "vllm")
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8100/v1")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")


def get_provider(
    provider_type: str | None = None,
    base_url: str | None = None,
) -> "InferenceProvider":
    """Return a cached InferenceProvider instance.

    Args:
        provider_type: "vllm" or "ollama". Defaults to INFERENCE_PROVIDER env var.
        base_url: Override base URL. Defaults to per-backend env var.

    Returns:
        Cached InferenceProvider for (provider_type, base_url) key.

    Raises:
        ValueError: If provider_type is not "vllm" or "ollama".
    """
    ptype = (provider_type or INFERENCE_PROVIDER).lower()
    if ptype == "vllm":
        url = base_url or VLLM_BASE_URL
    elif ptype == "ollama":
        url = base_url or OLLAMA_BASE_URL
    else:
        raise ValueError(f"Unknown provider: {ptype!r}. Expected 'vllm' or 'ollama'.")

    cache_key = (ptype, url)
    if cache_key not in _provider_cache:
        if ptype == "vllm":
            from inference.vllm_provider import VLLMProvider
            _provider_cache[cache_key] = VLLMProvider(base_url=url)
        else:
            from inference.ollama_provider import OllamaProvider
            _provider_cache[cache_key] = OllamaProvider(base_url=url)
    return _provider_cache[cache_key]


def get_provider_for_step(step_config: dict) -> "InferenceProvider":
    """Return a provider configured for a specific agent step.

    Args:
        step_config: Dict with keys "provider" (str), "model" (str),
                     and optional "adapter" (str). From per-step config dict.

    Returns:
        InferenceProvider for the step's backend configuration.
    """
    return get_provider(
        provider_type=step_config.get("provider"),
        base_url=step_config.get("base_url"),
    )
```

### Pattern 5: Updated lora-server Dockerfile

**What:** Replace `python:3.12-slim` base image with `vllm/vllm-openai:v0.16.0` and add runtime env var.
**When to use:** Required for INFRA-01 and INFRA-02.

```dockerfile
# Source: vLLM Docker docs https://docs.vllm.ai/en/stable/deployment/docker/
FROM vllm/vllm-openai:v0.16.0
ENV VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
WORKDIR /app
COPY . .
# health sidecar dependencies only — vLLM is already in the base image
RUN pip install --no-cache-dir fastapi uvicorn[standard] pyyaml
RUN chmod +x startup.sh
EXPOSE 8000 8001
CMD ["./startup.sh"]
```

**Key differences from current Dockerfile:**
- Base image: `vllm/vllm-openai:v0.16.0` (not `python:3.12-slim`)
- Remove `openai` from pip install — it's already in the base image
- `ENV VLLM_ALLOW_RUNTIME_LORA_UPDATING=True` at image level
- Container still exposes port 8000 (the vLLM server) and 8001 (health sidecar)

### Pattern 6: Minimal docker-compose.yml for INFRA-03

**What:** A minimal compose file in the project root (or infra/) with api-service on 8000 and lora-server on 8100, sharing a SQLite volume.
**When to use:** INFRA-03 — resolves the existing port conflict in `infra/docker-compose.yml` where both api and lora-server map to host port 8000.

```yaml
# Source: docker-compose docs; infra/docker-compose.yml existing pattern
# Minimal compose for Rune local development — api + lora-server only
services:
  api-service:
    build:
      context: .
      dockerfile: services/api-service/Dockerfile
    ports:
      - "8000:8000"
    volumes:
      - rune_data:/data
    environment:
      - DATABASE_URL=sqlite:////data/rune.db
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  lora-server:
    build:
      context: .
      dockerfile: services/lora-server/Dockerfile
    ports:
      - "8100:8000"   # host 8100 → container 8000 (vLLM)
      - "8001:8001"   # health sidecar unchanged
    volumes:
      - rune_data:/data
      - adapters_data:/adapters
    environment:
      - VLLM_BASE_URL=http://localhost:8000/v1
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8001/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 120s

volumes:
  rune_data:
  adapters_data:
```

**Port layout:**
| Service | Host Port | Container Port | Purpose |
|---------|-----------|----------------|---------|
| api-service | 8000 | 8000 | FastAPI REST API |
| lora-server | 8100 | 8000 | vLLM OpenAI API |
| lora-server | 8001 | 8001 | Health sidecar |

### Anti-Patterns to Avoid
- **Sync override of async abstract method:** Concrete provider implementing `def load_adapter()` (not `async def`) — Python won't catch this at instantiation; tests must verify coroutine semantics.
- **Silent no-op for UnsupportedOperationError:** OllamaProvider must raise explicitly, not return None silently. Agent loop cannot distinguish "operation succeeded" from "operation was silently skipped."
- **Using base_url string manipulation to build load_lora_adapter URL:** The `/v1/load_lora_adapter` endpoint is not under the `/v1/` path prefix conventionally — use `client.base_url` parsed to get host:port, then construct the full URL carefully. Pattern: strip trailing `/v1` and append `/v1/load_lora_adapter`.
- **Sharing AsyncOpenAI client across threads:** `AsyncOpenAI` is not thread-safe; since the factory caches per (provider, base_url) this is fine for async code, but don't share providers across threads.
- **Forgetting to update VLLM_BASE_URL default:** Current default is `http://localhost:8000/v1`. With lora-server now on host port 8100, the default must be updated to `http://localhost:8100/v1` everywhere (adapter_loader.py, vllm_client.py, factory.py).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| OpenAI-compat HTTP client | Custom httpx requests to /v1/chat/completions | AsyncOpenAI with custom base_url | SDK handles auth, retry, response parsing; both vLLM and Ollama expose same interface |
| Provider instance management | New instance per call | Factory cache keyed by (provider_type, base_url) | AsyncOpenAI creates connection pool per instance — shared instance is correct |
| LoRA load/unload HTTP | httpx against vLLM | httpx is correct here since /v1/load_lora_adapter is not in the OpenAI SDK | Exception to AsyncOpenAI rule: SDK doesn't expose load_lora_adapter — direct httpx POST is appropriate |
| Custom ABC enforcement | isinstance checks at call sites | ABC @abstractmethod | Python raises TypeError on instantiation if abstract methods not implemented |
| Docker port conflict fix | Any code change | docker-compose ports mapping | Port conflict is infrastructure-only; solution is `- "8100:8000"` mapping, not code |

**Key insight:** Both vLLM and Ollama expose identical `/v1/chat/completions` endpoints. The `AsyncOpenAI(base_url=..., api_key=...)` pattern unifies both backends with zero additional HTTP code for generation. The only asymmetry is LoRA management, which vLLM exposes as custom endpoints outside the OpenAI spec.

## Common Pitfalls

### Pitfall 1: VLLM_BASE_URL Default Still Points to Port 8000
**What goes wrong:** After updating lora-server to host port 8100, existing `VLLM_BASE_URL` defaults in adapter_loader.py and vllm_client.py still read `http://localhost:8000/v1`. Requests silently hit api-service instead of lora-server.
**Why it happens:** Default is hardcoded in multiple files; easy to update one and miss the others.
**How to avoid:** Update ALL occurrences of the default URL in one task. Search for `localhost:8000` in the codebase before finishing.
**Warning signs:** Generate calls appear to succeed (api-service returns 200) but no LLM output is produced.

### Pitfall 2: AsyncOpenAI base_url Trailing Slash Behavior
**What goes wrong:** `AsyncOpenAI(base_url="http://localhost:8100/v1")` may or may not add a trailing slash — and OpenAI SDK path construction can produce double slashes (`/v1//chat/completions`) or miss the prefix.
**Why it happens:** The SDK normalizes base_url but the behavior depends on whether it ends with `/v1` or `/v1/`.
**How to avoid:** Pass `base_url` with trailing slash: `"http://localhost:8100/v1/"` — this is the pattern that works reliably. Verify with existing adapter_loader.py behavior (tests already pass with the current pattern).
**Warning signs:** 404 errors on chat completions; test `str(client.base_url)` and confirm it matches expected.

### Pitfall 3: vLLM load_lora_adapter Endpoint URL Construction
**What goes wrong:** Provider has `self._base_url = "http://localhost:8100/v1"`. To call `/v1/load_lora_adapter`, code constructs `self._base_url + "/load_lora_adapter"` = `http://localhost:8100/v1/load_lora_adapter`. This is correct. BUT if base_url is `http://localhost:8100/v1/` (trailing slash), concatenation gives `http://localhost:8100/v1//load_lora_adapter`.
**Why it happens:** Inconsistent trailing slash handling.
**How to avoid:** Use `self._base_url.rstrip("/") + "/load_lora_adapter"` or parse with `urllib.parse` for reliable URL construction.
**Warning signs:** 404 from vLLM on adapter operations.

### Pitfall 4: list_adapters Unreliable via GET /v1/models for Dynamic Loaders
**What goes wrong:** After calling `load_adapter()`, `list_adapters()` calls `GET /v1/models` and the newly-loaded adapter doesn't appear.
**Why it happens:** Known vLLM issue #11761 — dynamically-loaded adapters via `/v1/load_lora_adapter` may not be reflected in the `/v1/models` response immediately.
**How to avoid:** Maintain an internal `set[str]` in `VLLMProvider` that tracks loaded adapter names locally. Update on load/unload. Use this as the source of truth for `list_adapters()`.
**Warning signs:** `list_adapters()` returns only the base model even after successful `load_adapter()` call.

### Pitfall 5: Async ABC Method Not Enforced as Async in Subclass
**What goes wrong:** `class VLLMProvider(InferenceProvider)` declares `def generate(...)` (sync) instead of `async def generate(...)`. Python does NOT raise TypeError — the abstract method contract is satisfied as long as the name exists.
**Why it happens:** Python's ABC checks method existence, not coroutine nature.
**How to avoid:** Tests must explicitly check `asyncio.iscoroutinefunction(provider.generate)` OR simply call `await provider.generate(...)` in tests, which will raise `TypeError: object NoneType can't be used in 'await' expression` if non-async.
**Warning signs:** `await provider.generate(...)` raises TypeError at test time.

### Pitfall 6: OllamaProvider api_key Must Be Non-Empty String
**What goes wrong:** `AsyncOpenAI(api_key="")` raises a validation error — the SDK requires a non-empty string.
**Why it happens:** OpenAI SDK validates API key presence at construction time.
**How to avoid:** Use `api_key="ollama"` (Ollama ignores the value but the SDK accepts it). Same pattern used in VLLMProvider with `"not-needed-for-local-vllm"`.
**Warning signs:** `openai.AuthenticationError` or validation error on OllamaProvider instantiation.

### Pitfall 7: Factory Cache Leaks Between Tests
**What goes wrong:** Module-level `_provider_cache` dict in factory.py persists between tests. Test A creates a VLLMProvider; Test B expects a fresh one and gets the cached instance.
**Why it happens:** Module-level mutable state in Python is process-global.
**How to avoid:** In conftest.py, add a fixture that clears `_provider_cache` before/after each test: `from inference import factory; factory._provider_cache.clear()`. Or use `monkeypatch.setattr` to isolate.
**Warning signs:** Tests pass individually but fail when run together; ordering-dependent failures.

## Code Examples

Verified patterns from official sources and existing codebase:

### InferenceProvider ABC (Minimal Complete Example)
```python
# Source: Python abc module docs; async ABC pattern
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class GenerationResult:
    text: str
    model: str
    adapter_id: str | None
    token_count: int
    finish_reason: str


class InferenceProvider(ABC):
    @abstractmethod
    async def generate(
        self, prompt: str, model: str,
        adapter_id: str | None = None, max_tokens: int = 1024
    ) -> GenerationResult: ...

    @abstractmethod
    async def load_adapter(self, adapter_id: str, adapter_path: str) -> None: ...

    @abstractmethod
    async def unload_adapter(self, adapter_id: str) -> None: ...

    @abstractmethod
    async def list_adapters(self) -> list[str]: ...
```

### vLLM Dynamic LoRA API (Verified Request Format)
```python
# Source: https://docs.vllm.ai/en/v0.8.1/features/lora.html
# Load:
# POST /v1/load_lora_adapter
# {"lora_name": "my_adapter", "lora_path": "/adapters/my_adapter"}
# Response: 200 OK

# Unload:
# POST /v1/unload_lora_adapter
# {"lora_name": "my_adapter"}
# Response: 200 OK

# Generate with loaded adapter — pass lora_name as model:
# POST /v1/chat/completions
# {"model": "my_adapter", "messages": [...], "max_tokens": 1024}
```

### Mocked Provider Test (AsyncMock Pattern)
```python
# Source: existing services/lora-server/tests/test_health.py (AsyncMock + patch pattern)
# asyncio_mode = "auto" in root pyproject.toml — no @pytest.mark.asyncio needed
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from inference.vllm_provider import VLLMProvider


async def test_vllm_provider_generate_returns_result():
    provider = VLLMProvider(base_url="http://localhost:8100/v1")
    mock_choice = MagicMock()
    mock_choice.message.content = "def hello(): pass"
    mock_choice.finish_reason = "stop"
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage.total_tokens = 10

    with patch.object(provider._client.chat.completions, "create",
                      new=AsyncMock(return_value=mock_response)):
        result = await provider.generate("Write hello", "Qwen/Qwen2.5-Coder-7B")

    assert result.text == "def hello(): pass"
    assert result.finish_reason == "stop"
    assert result.token_count == 10
```

### Factory Cache Clear Fixture
```python
# In tests/conftest.py or inference/tests/conftest.py
import pytest
from inference import factory


@pytest.fixture(autouse=True)
def clear_provider_cache():
    """Clear factory cache between tests to avoid state leak."""
    factory._provider_cache.clear()
    yield
    factory._provider_cache.clear()
```

### Updated __init__.py Exports
```python
# libs/inference/src/inference/__init__.py after refactor
from inference.provider import InferenceProvider, GenerationResult
from inference.vllm_provider import VLLMProvider
from inference.ollama_provider import OllamaProvider
from inference.factory import get_provider, get_provider_for_step
from inference.exceptions import UnsupportedOperationError

__all__ = [
    "InferenceProvider",
    "GenerationResult",
    "VLLMProvider",
    "OllamaProvider",
    "get_provider",
    "get_provider_for_step",
    "UnsupportedOperationError",
]
```

### Test Commands
```bash
# Unit tests for libs/inference (from project root)
uv run pytest libs/inference/ -x -v

# Lora-server tests
uv run pytest services/lora-server/ -x -v

# Mypy strict check
cd /Users/noahdolevelixir/Code/rune/libs/inference && uv run mypy src/

# Ruff check
cd /Users/noahdolevelixir/Code/rune/libs/inference && uv run ruff check .

# Full phase-relevant suite
uv run pytest libs/inference/ libs/adapter-registry/ services/lora-server/ -x -q
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Module-level functions (current completion.py, adapter_loader.py) | InferenceProvider ABC with concrete subclasses | Phase 19 | Pluggable backends; agent loop backend-agnostic |
| VLLMClient stub in lora-server/ | VLLMProvider in libs/inference/ | Phase 19 | Provider co-located with interface; lora-server becomes Docker-only artifact |
| python:3.12-slim base image | vllm/vllm-openai:v0.16.0 | Phase 19 (INFRA-01) | vLLM engine + CUDA runtime baked in; no manual vLLM install |
| Static adapter config at startup (--lora-modules) | Dynamic loading via /v1/load_lora_adapter | Phase 19 (INF-02) | Adapters hot-swappable without server restart |
| Port 8000 for both api-service and lora-server | api-service 8000, lora-server 8100 | Phase 19 (INFRA-03) | Port conflict resolved; both services can run simultaneously |

**Deprecated/outdated:**
- `get_vllm_client()` function in adapter_loader.py: Replaced by `VLLMProvider.__init__()` — factory pattern is cleaner and cached.
- `generate_completion()`, `generate_with_adapter()`, `batch_generate()` module-level functions in completion.py: Replaced by `VLLMProvider.generate()` and `OllamaProvider.generate()`.
- `VLLMClient` class in lora-server/vllm_client.py: Absorbed into VLLMProvider; lora-server tests for VLLMClient become provider tests.
- `loaders.py` in inference (referenced in `__pycache__` as `loaders.cpython-313.pyc`): Appears to be a cached artifact — verify if the file exists or if it's a stale cache; if it exists, determine if it needs updating.

## Open Questions

1. **VLLMProvider placement: libs/inference or lora-server?**
   - What we know: CONTEXT.md locks "Keep the provider in libs/inference — refactor in-place." VLLMClient currently lives in services/lora-server/.
   - What's unclear: Whether lora-server/vllm_client.py should be deleted, kept as a thin shim, or moved wholesale.
   - Recommendation: Delete `services/lora-server/vllm_client.py` after implementing `libs/inference/vllm_provider.py`. Update `services/lora-server/tests/test_vllm_client.py` to import from `inference.vllm_provider`. The lora-server becomes a Docker service with no Python business logic beyond config.py and health.py.

2. **list_adapters() reliability with vLLM dynamic loading**
   - What we know: vLLM bug #11761 — dynamically-loaded adapters may not appear in GET /v1/models.
   - What's unclear: Whether this is fixed in v0.16.0 (research couldn't confirm fix status for that exact version).
   - Recommendation: Implement `list_adapters()` with a local tracking set (`_loaded_adapters: set[str]`) in VLLMProvider as the primary source of truth. Also call GET /v1/models and merge results. Document the known limitation.

3. **docker-compose.yml location: project root or infra/?**
   - What we know: INFRA-03 says "Create minimal docker-compose.yml." Current `infra/docker-compose.yml` exists but has the port conflict.
   - What's unclear: Whether to fix `infra/docker-compose.yml` in-place or create a separate `docker-compose.yml` in project root.
   - Recommendation: Fix `infra/docker-compose.yml` in-place — it already exists and is the established location. A separate root-level compose is unnecessary.

4. **`loaders.py` stale cache artifact**
   - What we know: `.pyc` file for `loaders.cpython-313.pyc` exists in `libs/inference/src/inference/__pycache__/` but no corresponding `loaders.py` source was found.
   - What's unclear: Whether this is a deleted file that left a cached artifact or something else.
   - Recommendation: Safe to ignore; `.pyc` files without corresponding `.py` are harmless stale artifacts from a deleted file.

## Sources

### Primary (HIGH confidence)
- Direct code inspection: `libs/inference/src/inference/` — adapter_loader.py, completion.py, __init__.py, pyproject.toml
- Direct code inspection: `services/lora-server/` — vllm_client.py, config.py, Dockerfile, startup.sh, health.py
- Direct code inspection: `infra/docker-compose.yml` — existing port conflict confirmed
- Direct code inspection: `pyproject.toml` (root) — asyncio_mode = "auto", workspace members, pythonpath
- `.planning/phases/19-inference-provider-abstraction/19-CONTEXT.md` — locked decisions verbatim
- [vLLM LoRA Adapters docs v0.8.1](https://docs.vllm.ai/en/v0.8.1/features/lora.html) — load/unload endpoint format verified
- [Ollama OpenAI compat docs](https://docs.ollama.com/api/openai-compatibility) — /v1/chat/completions confirmed, api_key="ollama" pattern

### Secondary (MEDIUM confidence)
- [vLLM Docker docs](https://docs.vllm.ai/en/stable/deployment/docker/) — vllm/vllm-openai image confirmed; v0.16.0 specific contents not independently verified
- vLLM issue #11761 (referenced in web search) — list_adapters reliability concern for dynamically-loaded adapters
- Web search results for vLLM LoRA request format — confirmed by multiple sources including official docs

### Tertiary (LOW confidence)
- vLLM v0.16.0 specific behavior (searched but could not confirm exact changelog vs v0.8.x) — LOW confidence on per-version differences; core API shape is stable across recent versions

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all dependencies already in workspace; no new installs needed
- Architecture: HIGH — all patterns derived from locked CONTEXT.md decisions + existing codebase + verified official docs
- vLLM dynamic LoRA API: HIGH — endpoint format (lora_name, lora_path, model=adapter_id for generation) confirmed from official docs
- Ollama OpenAI compat: HIGH — /v1/chat/completions and api_key="ollama" confirmed from official docs
- list_adapters reliability: MEDIUM — known issue documented; workaround (local tracking set) recommended
- Docker/INFRA: HIGH — port conflict confirmed in existing file; solution is straightforward mapping change

**Research date:** 2026-03-05
**Valid until:** 2026-06-05 (vLLM LoRA API is stable; Ollama OpenAI compat is stable; Python ABC patterns are stdlib)
