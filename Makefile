SHELL := /bin/bash
.PHONY: help lint lint-fix typecheck test check clean

lint:
	@uv run ruff check .

lint-fix:
	@uv run ruff check . --fix
	@uv run ruff format .

typecheck:
	@uv run mypy src/

test:
	@uv run pytest tests/ -q

test-unit:
	@uv run pytest tests/unit/ -q

test-integration:
	@uv run pytest tests/integration/ -q

test-gpu:
	@uv run pytest tests/gpu/ -q -m gpu

check: lint typecheck test-unit

clean:
	@rm -rf site/ .cache/ .mypy_cache/ .pytest_cache/ .ruff_cache/ htmlcov/

help:
	@echo "Rune v2"
	@echo ""
	@echo "  make check          lint + typecheck + unit tests"
	@echo "  make test           all tests"
	@echo "  make test-unit      unit tests only"
	@echo "  make test-gpu       GPU tests only"
	@echo "  make lint-fix       auto-fix lint issues"
