# Welcome to ElixirTrials  Template

This is a mono-repo template combining Python backend services, LangGraph AI agents, and a React/Vite HITL frontend.

## 🚀 Quick Start

### Prerequisites
- **Python 3.12+**
- **Node.js 20+**
- **uv** (Python package manager)
- **Docker**

### Installation
Run the following commands to get started:

```bash
# Sync dependencies
uv sync

# Create a new service (interactive)
make create-service
```

## 🎯 Project Goals
- **Robust Template**: A production-ready foundation for AI applications.
- **Unified Tooling**: Standardized patterns for backend, frontend, and AI agents.
- **Rapid Development**: Pre-configured CI/CD, linting, and testing.

## 🏗️ Architecture

- **API Service**: FastAPI orchestrator handling requests and persistence.
- **Agent Services**: Independent LangGraph agents for specific workflows.
- **Inference**: Shared library for model loading and LLM interactions.
- **Data Pipeline**: ETL processes for data ingestion.
- **Evaluation**: Offline metrics and benchmarks.
- **HITL UI**: React/Vite frontend for human review and approval.

## 📚 Documentation Index

- **[Components](components-overview.md)**: Detailed docs for all services and libraries.
- **[Diagrams](diagrams/hitl-flow.md)**: Visual flows of the system.
