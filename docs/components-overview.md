# Components Overview

This page provides an overview of all microservices and shared libraries in this monorepo.

## Available Components

| Component | Description | Documentation |
| :--- | :--- | :--- |
| **adapter-registry** | No description available. | [API Reference](adapter-registry/api/index.md) |
| **api-service** | The API Service is the central orchestrator for the application. It provides HTTP endpoints for the frontend, manages database persistence, and triggers background agent tasks. | [API Reference](api-service/api/index.md) |
| **evaluation** | This component runs offline evaluation benchmarks against your agents. | [API Reference](evaluation/api/index.md) |
| **events-py** | Shared event envelope shapes and helpers used by Python services that publish or consume events. | [API Reference](events-py/api/index.md) |
| **evolution-svc** | Adapter evaluation, evolution, promotion, and pruning service for Rune. | [API Reference](evolution-svc/api/index.md) |
| **inference** | This component acts as the "Standard Library" for AI in this repository. It centralizes model loading, prompt rendering, and agent construction to ensure consistency across all services. | [API Reference](inference/api/index.md) |
| **model-training** | This component handles fine-tuning (LoRA), distillation, or training of custom models. | [API Reference](model-training/api/index.md) |
| **rune-agent** | This component implements a specific AI agent workflow for interacting with guests. It uses LangGraph to define the flow and state. | [API Reference](rune-agent/api/index.md) |
| **shared** | This component holds code that is strictly **common** to multiple components. This usually includes: | [API Reference](shared/api/index.md) |
| **training-svc** | LoRA and hypernetwork training job service for Rune. | [API Reference](training-svc/api/index.md) |
