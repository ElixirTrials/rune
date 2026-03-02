#!/bin/bash
# Create a new service scaffold (Python or TypeScript).
# Usage: create-service.sh [--lang py|ts] [--lib | --app] <SERVICE_NAME>
#
# Options:
#   --lang py|ts   Language (default: py)
#   --lib          Create under libs/ (default: services/)
#   --app          Create under apps/ (default: services/)
set -e

LANG=py
PARENT=services
while [[ $# -gt 0 ]]; do
    case $1 in
        --lang)
            LANG="$2"
            shift 2
            ;;
        --lib)
            PARENT=libs
            shift
            ;;
        --app)
            PARENT=apps
            shift
            ;;
        -*)
            echo "Error: Unknown option $1"
            echo "Usage: $0 [--lang py|ts] [--lib | --app] <SERVICE_NAME>"
            exit 1
            ;;
        *)
            break
            ;;
    esac
done

SERVICE_NAME=$1

if [[ -z "$SERVICE_NAME" ]]; then
    echo "Usage: $0 [--lang py|ts] [--lib | --app] <SERVICE_NAME>"
    exit 1
fi

if ! [[ "$SERVICE_NAME" =~ ^[a-z][a-z0-9-]*$ ]]; then
    echo "Error: Service name must start with a letter and contain only lowercase letters, numbers, and hyphens."
    exit 1
fi

case "$LANG" in
    py)
        if ! command -v uv &> /dev/null; then
            echo "Error: uv is required for Python services but is not installed."
            exit 1
        fi
        SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
        ROOT_DIR="$(dirname "$SCRIPT_DIR")"
        mkdir -p "$ROOT_DIR/$PARENT"
        cd "$ROOT_DIR"
        uv init --name "$SERVICE_NAME" --package --python 3.12 "$PARENT/$SERVICE_NAME"
        cd "$PARENT/$SERVICE_NAME"
        PACKAGE=$(echo "$SERVICE_NAME" | awk '{gsub("-", "_"); print}')

        uv add ruff mypy pytest pytest-cov ipykernel nbqa pre-commit --dev

        mkdir -p "src/$PACKAGE/notebooks"
        mkdir -p "src/$PACKAGE/internal"
        touch "src/$PACKAGE/notebooks/.gitkeep"
        touch "src/$PACKAGE/internal/__init__.py"
        touch "Dockerfile"
        mkdir -p "tests"
        echo '' > "src/$PACKAGE/__init__.py"

        cat <<EOF > "src/$PACKAGE/example.py"
def example_function(a: int, b: int) -> int:
    \"\"\"Add two numbers.\"\"\"
    return a + b
EOF

        cat <<EOF > "src/$PACKAGE/main.py"
from $PACKAGE.example import example_function


def main() -> None:
    \"\"\"Execute the main logic of the component.\"\"\"
    print("Component is running.")
    example_function(1, 2)


if __name__ == "__main__":
    main()
EOF

        cat <<EOF > "tests/test_example.py"
from $PACKAGE.example import example_function


def test_example_function():
    assert example_function(2, 3) == 5
EOF

        cat <<EOF >> "pyproject.toml"
[tool.ruff]
lint.select = ["E", "F", "W", "C", "N", "I", "D"]
lint.ignore = ["E203", "D203", "D213", "D100", "D413", "D104"]
target-version = "py312"
line-length = 88
exclude = ["venv", ".venv", "tests", "docs"]
lint.pydocstyle.convention = "google"

[tool.mypy]
strict = true
exclude = ["venv", ".venv", "tests", "docs"]
plugins = []

[tool.pytest.ini_options]
pythonpath = ["src"]

[tool.uv]
cache-dir = "./.uv_cache"
EOF

        cat <<EOF > "Makefile"
.PHONY: fmt lint type-check run-tests clean check-all
fmt:
	uv run ruff format ./src
lint:
	uv run ruff check .
type-check:
	uv run mypy ./src
run-tests:
	uv run pytest --cov=./src/$PACKAGE .
clean:
	rm -rf .pytest_cache .mypy_cache .ipynb_checkpoints .coverage
	find . -name "__pycache__" -exec rm -rf {} +
check-all: fmt lint type-check run-tests
EOF

        cat <<EOF > "mkdocs.yml"
docs_dir: docs
site_name: $SERVICE_NAME
nav:
  - API Reference: api/index.md
EOF

        mkdir -p "docs/api"
        cat <<EOF > "docs/api/index.md"
# $SERVICE_NAME API Reference

::: $PACKAGE
    handler: python
    selection:
      docstring_style: google
    rendering:
      show_source: true
      show_root_heading: true
      show_category_heading: true
      show_signature_annotations: true
      show_bases: true
      heading_level: 3
      members_order: source
      filters: ["!^_"]
      merge_init_into_class: true
      show_if_no_docstring: false
      separate_signature: true
      signature_crossrefs: true
      show_submodules: true
      show_inheritance_diagram: true
      show_root_toc_entry: true
EOF

        echo "Created Python service: $PARENT/$SERVICE_NAME"
        ;;
    ts)
        SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
        ROOT_DIR="$(dirname "$SCRIPT_DIR")"
        mkdir -p "$ROOT_DIR/$PARENT"
        DIR="$ROOT_DIR/$PARENT/$SERVICE_NAME"
        mkdir -p "$DIR/src"
        PACKAGE=$(echo "$SERVICE_NAME" | awk '{gsub("-", "_"); print}')

        cat <<EOF > "$DIR/package.json"
{
  "name": "$SERVICE_NAME",
  "version": "0.1.0",
  "private": true,
  "type": "module",
  "main": "dist/index.js",
  "types": "dist/index.d.ts",
  "scripts": {
    "build": "tsc",
    "lint": "biome check .",
    "lint:fix": "biome check . --write",
    "format": "biome format --write .",
    "typecheck": "tsc --noEmit"
  }
}
EOF

        cat <<EOF > "$DIR/tsconfig.json"
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "NodeNext",
    "moduleResolution": "NodeNext",
    "outDir": "dist",
    "rootDir": "src",
    "strict": true,
    "declaration": true,
    "declarationMap": true,
    "skipLibCheck": true,
    "esModuleInterop": true
  },
  "include": ["src/**/*"]
}
EOF

        cat <<EOF > "$DIR/src/index.ts"
/**
 * Entry point for $SERVICE_NAME.
 */

export function greet(name: string): string {
  return \`Hello, \${name}\`;
}
EOF

        cat <<EOF > "$DIR/README.md"
# $SERVICE_NAME

TypeScript service scaffold.
EOF

        cat <<EOF > "$DIR/mkdocs.yml"
docs_dir: docs
site_name: $SERVICE_NAME
nav:
  - API Reference: api/index.md
EOF

        mkdir -p "$DIR/docs/api"
        cat <<EOF > "$DIR/docs/api/index.md"
# $SERVICE_NAME API Reference

TypeScript API documentation (placeholder).
EOF

        if command -v npm &> /dev/null; then
            cd "$DIR"
            npm install --save-dev typescript @biomejs/biome
            cd "$ROOT_DIR"
        fi

        echo "Created TypeScript service: $PARENT/$SERVICE_NAME"
        echo "  Run: cd $PARENT/$SERVICE_NAME && npm install && npm run build"
        ;;
    *)
        echo "Error: --lang must be 'py' or 'ts', got: $LANG"
        exit 1
        ;;
esac
