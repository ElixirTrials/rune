# ElixirTrials  Template

This is a comprehensive mono-repo template designed for building production-ready AI applications. It combines Python backend services, LangGraph AI agents, and a React/Vite Human-in-the-Loop (HITL) frontend into a unified, high-performance architecture.

## 🚀 Quick Start

### Prerequisites
- **Python 3.12+**
- **Node.js 20+**
- **uv** (Modern Python package manager)
- **Docker & Docker Compose**

### Installation
Clone the repository and sync dependencies:

```bash
# Sync Python dependencies using uv
uv sync

# Install frontend dependencies
cd apps/hitl-ui && npm install && cd ../..
```

## 🛠️ Template Usage

This template provides several automation commands to streamline development.

### Creating a New Service or Library
To add a new microservice or library (under `services/`, `libs/`, or `apps/`):

```bash
make create-service
```
*This will prompt for a name and language (py|ts) and scaffold the directory structure. Use `scripts/create-service.sh [--lang py|ts] [--lib | --app] <name>` for non-interactive use (default: services/; use --lib for libs/, --app for apps/).*

### Documentation Site
We use MkDocs for comprehensive documentation.

```bash
# Build the documentation site
make docs-build

# Serve the documentation locally (after building)
make docs-serve
```
*The site will be available at [http://localhost:8000](http://localhost:8000).*

### Database Management
We use SQLModel and Alembic for schema migrations.

```bash
# Create a new migration revision
make db-revision msg="Add table name"

# Apply migrations
make db-migrate
```

## 🏗️ Architecture Overview

- **`services/api-service`**: Central FastAPI orchestrator.
- **`services/agent-*-service`**: Specialized AI agent workflows using LangGraph.
- **`libs/inference`**: Shared AI library for model loading and logic.
- **`apps/hitl-ui`**: React/Vite dashboard for human-in-the-loop review.
- **`libs/shared`**: Common models and utility functions.
- **`infra/`**: Deployment aid (Docker Compose, optional Terraform/K8s later).
- **`docs/`**: Markdown-based documentation and system diagrams.

## 📚 Further Reading
For detailed implementation guides and API references, refer to the [Local Documentation Site](docs/index.md) (or run `make docs-serve`).
