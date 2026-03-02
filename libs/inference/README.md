# Inference Component

## Purpose
This component acts as the "Standard Library" for AI in this repository. It centralizes model loading, prompt rendering, and agent construction to ensure consistency across all services.

## How to Use

### 1. Model Loading
Do not instantiate `ChatVertexAI` or `ChatOpenAI` directly in your services. Use the loaders here to ensure tracing and config are applied.

```python
# src/inference/loaders.py (create this if needed)
from langchain_google_vertexai import ChatVertexAI

def get_vertex_model(model_name: str = "gemini-1.5-pro"):
    return ChatVertexAI(model_name=model_name, temperature=0)
```

### 2. Creating Agents
Use the factories in `src/inference/factory.py` to create robust agents.

**Structured Extraction Agent:**
```python
from inference.factory import create_structured_extractor
from shared.models import MyOutputSchema

from shared import get_prompts_dir

extractor = create_structured_extractor(
    model_loader=get_vertex_model,
    prompts_dir=get_prompts_dir(),
    response_schema=MyOutputSchema,
    system_template="extract_sys.j2",
    user_template="extract_user.j2"
)
result = await extractor(prompt_vars={"text": "..."})
```

### 3. Prompt Management
- **Canonical templates:** Use `shared.get_prompts_dir()` for shared Jinja2 templates (`.j2`) so all agents use the same templates with different `prompt_vars` (see `libs/shared/README.md`). Do not duplicate template files in services.
- For service-specific prompts only, use a `prompts/` directory in that service and pass its path.
- Do not hardcode prompt strings in Python files. The factory handles rendering automatically.

## Best Practices
- **Lazy Loading**: Models are heavy. Use `@lazy_singleton` to load them only when needed.
- **Retry Logic**: The factories come with built-in retries for transient errors.
- **Tracing**: MLflow or LangSmith tracing is configured at the model loader level.
