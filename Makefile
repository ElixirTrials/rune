# Rune Makefile — workspace-aware. Targets iterate services/* / libs/* and
# no-op cleanly when those directories are empty.
SHELL := /bin/bash

.PHONY: help \
        create-component create-service \
        infra-up infra-down \
        docs-components-gen docs-nav-update docs-build docs-serve docs-openapi \
        kill-processes db-migrate db-revision \
        check check-fix check-with-docs lint lint-fix typecheck test clean \
        claude-sync claude-sync-init claude-sync-dry \
        claude-crg-enable claude-crg-disable

# ----- Discovery helpers (workspace-aware) ---------------------------------
PY_SERVICES   := $(shell find services -mindepth 1 -maxdepth 1 -type d -not -name '.gitkeep' 2>/dev/null)
PY_LIBS       := $(shell find libs     -mindepth 1 -maxdepth 1 -type d -not -name '.gitkeep' 2>/dev/null)
PY_PACKAGES   := $(PY_SERVICES) $(PY_LIBS)
ALEMBIC_DIRS  := $(shell find services -mindepth 2 -maxdepth 2 -type d -name alembic 2>/dev/null)
PY_SRC_DIRS   := $(foreach pkg,$(PY_PACKAGES),$(wildcard $(pkg)/src))
PY_TEST_DIRS  := $(foreach pkg,$(PY_PACKAGES),$(wildcard $(pkg)/tests))

# ----- Scaffolding ---------------------------------------------------------
create-component:
	@read -p "Enter component name: " COMPONENT_NAME; \
	./scripts/create-service.sh --lang py "$$COMPONENT_NAME"

create-service:
	@read -p "Enter service name: " NAME; read -p "Language (py|ts) [py]: " LANG; LANG=$${LANG:-py}; \
	./scripts/create-service.sh --lang "$$LANG" "$$NAME"

# ----- Local infra (Postgres, MLflow, litestream) -------------------------
infra-up:
	@docker-compose -f infra/docker-compose.yml up -d

infra-down:
	@docker-compose -f infra/docker-compose.yml down

# ----- Docs ---------------------------------------------------------------
docs-components-gen:
	@echo "Generating components overview page..."
	uv run python scripts/generate_components_overview.py

docs-nav-update:
	@echo "Updating root navigation..."
	uv run python scripts/update_root_navigation.py

docs-build: docs-nav-update docs-components-gen docs-openapi
	@echo "Building documentation site..."
	uv run python scripts/build_docs.py build -f mkdocs.yml

docs-serve:
	@if [ ! -d "site" ]; then echo "No built site. Run 'make docs-build' first."; exit 1; fi
	@echo "Serving from $(PWD)/site at http://localhost:8000"
	@cd site && uv run python -m http.server 8000

# Iterate over any service that defines scripts/export_openapi.py.
docs-openapi:
	@for svc in $(PY_SERVICES); do \
		if [ -f "$$svc/scripts/export_openapi.py" ]; then \
			echo "→ OpenAPI for $$svc"; \
			uv run --project "$$svc" python "$$svc/scripts/export_openapi.py" || exit $$?; \
		fi; \
	done

# ----- Database (Alembic per-service) -------------------------------------
# Iterates every services/<name>/alembic. Migrations are USER-RUN; Claude is
# hook-blocked from db-migrate.
db-revision:
	@if [ -z "$(msg)" ]; then echo "Usage: make db-revision msg=\"...\""; exit 1; fi
	@for svc in $(ALEMBIC_DIRS); do \
		echo "→ revision in $$svc"; \
		(cd "$$(dirname $$svc)" && uv run alembic revision --autogenerate -m "$(msg)") || exit $$?; \
	done

db-migrate:
	@for svc in $(ALEMBIC_DIRS); do \
		echo "→ upgrade head in $$svc"; \
		(cd "$$(dirname $$svc)" && uv run alembic upgrade head) || exit $$?; \
	done

# ----- Code quality (workspace-aware) -------------------------------------
check:
	@./scripts/check-all.sh

check-fix:
	@./scripts/check-all.sh --fix

check-with-docs:
	@./scripts/check-all.sh --with-docs

lint:
	@uv run ruff check .

lint-fix:
	@uv run ruff check . --fix
	@uv run ruff format .

typecheck:
	@if [ -n "$(PY_SRC_DIRS)" ]; then uv run mypy $(PY_SRC_DIRS); else echo "(no python src dirs)"; fi

test:
	@if [ -n "$(PY_TEST_DIRS)" ]; then uv run pytest $(PY_TEST_DIRS) -q; else echo "(no python test dirs)"; fi

# ----- Misc ---------------------------------------------------------------
kill-processes:
	@./scripts/kill-running-processes.sh

clean:
	@rm -rf site/ .cache/ docs/.uv_cache/

# ----- Claude template sync ------------------------------------------------
claude-sync:
	@./scripts/sync-claude-template.sh

claude-sync-init:
	@./scripts/sync-claude-template.sh --init

claude-sync-dry:
	@./scripts/sync-claude-template.sh --dry-run

# ----- code-review-graph (CRG) — OPTIONAL per-repo add-on ------------------
CRG_VERSION := 2.3.2

claude-crg-enable:
	@command -v pipx >/dev/null || { echo "Install pipx first: brew install pipx"; exit 1; }
	@command -v claude >/dev/null || { echo "Claude Code CLI not on PATH"; exit 1; }
	@echo "→ Installing code-review-graph==$(CRG_VERSION) (one-time per machine)…"
	@pipx install code-review-graph==$(CRG_VERSION) 2>/dev/null || pipx upgrade code-review-graph || true
	@echo "→ Building initial CRG index (one-time per repo, ~10s per 500 files, USER-RUN)…"
	@code-review-graph build
	@echo "→ Wiring post-commit auto-update…"
	@./scripts/setup-git-hooks.sh
	@echo "→ Registering as project-scoped MCP server…"
	@claude mcp add code-review-graph --scope project -- code-review-graph serve --repo . 2>&1 || \
		echo "  (already registered — that's fine)"
	@echo
	@echo "✓ CRG installed + indexed + registered."

claude-crg-disable:
	@claude mcp remove code-review-graph 2>/dev/null || echo "  (not registered)"
	@git config --unset core.hooksPath 2>/dev/null || true
	@echo "✓ CRG MCP unregistered, post-commit hook unwired."

help:
	@echo "Rune — Makefile"
	@echo ""
	@echo "Bootstrap / scaffold:"
	@echo "  make create-service          Scaffold a new Python or TypeScript package"
	@echo "  make create-component        Same, Python-only convenience"
	@echo "  make infra-up / infra-down   Local stack (Postgres, MLflow, litestream)"
	@echo ""
	@echo "Code quality:"
	@echo "  make check / check-fix       Run all checks (auto-fix variant available)"
	@echo "  make check-with-docs         Run all checks including doc build"
	@echo "  make lint / lint-fix         Ruff (Python)"
	@echo "  make typecheck               mypy"
	@echo "  make test                    pytest"
	@echo ""
	@echo "Docs (mkdocs):"
	@echo "  make docs-build / docs-serve   Build / serve the doc site"
	@echo "  make docs-openapi              Export OpenAPI from each FastAPI service"
	@echo ""
	@echo "Database (Alembic, USER-RUN):"
	@echo "  make db-revision msg=\"...\"   Generate migration for each services/*/alembic"
	@echo "  make db-migrate                Apply migrations (Claude is hook-blocked from this)"
	@echo ""
	@echo "Claude infra:"
	@echo "  make claude-sync             Sync .claude/ from template (manifest-aware)"
	@echo "  make claude-sync-dry         Preview sync without writing"
	@echo "  make claude-crg-enable       Install + wire code-review-graph MCP"
	@echo "  make claude-crg-disable      Unregister CRG MCP + unhook"
	@echo ""
	@echo "Misc:"
	@echo "  make kill-processes           Kill rune dev/swarm processes"
	@echo "  make clean                    Remove site/, .cache/, docs/.uv_cache/"
