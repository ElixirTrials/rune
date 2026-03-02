import json
import sys
from pathlib import Path

# Add src to path for import
sys.path.append(str(Path(__file__).parent.parent / "src"))

try:
    from fastapi.openapi.utils import get_openapi

    from api_service.main import app
except ImportError as e:
    print(f"Error importing app: {e}")
    sys.exit(1)


def export_openapi():
    """Export the FastAPI OpenAPI schema to a local JSON file."""
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        openapi_version=app.openapi_version,
        description=app.description,
        routes=app.routes,
    )

    # Ensure directory exists
    output_path = Path("docs/openapi.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(openapi_schema, f, indent=2)

    print(f"Successfully exported OpenAPI schema to {output_path}")


if __name__ == "__main__":
    export_openapi()
