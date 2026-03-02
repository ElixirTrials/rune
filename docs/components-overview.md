# Components Overview

This page provides an overview of all microservices and shared libraries in this monorepo.

## Available Components

| Component | Description | Documentation |
| :--- | :--- | :--- |
| **agent-a-service** | This component implements a specific AI agent workflow for interacting with guests. It uses LangGraph to define the flow and state. | [API Reference](agent-a-service/api/index.md) |
| **agent-b-service** | This component implements a guardrailing workflow to validate outputs and ensure safety. | [API Reference](agent-b-service/api/index.md) |
| **api-service** | The API Service is the central orchestrator for the application. It provides HTTP endpoints for the frontend, manages database persistence, and triggers background agent tasks. | [API Reference](api-service/api/index.md) |
| **data-pipeline** | This component handles data ingestion (ETL), normalization, and preparation for the API or training. | [API Reference](data-pipeline/api/index.md) |
| **evaluation** | This component runs offline evaluation benchmarks against your agents. | [API Reference](evaluation/api/index.md) |
| **events-py** | Shared event envelope shapes and helpers used by Python services that publish or consume events. | [API Reference](events-py/api/index.md) |
| **events-ts** | Shared event shapes and helpers used by TypeScript services that publish or consume events. | [API Reference](events-ts/api/index.md) |
| **hitl-ui** | React/Vite application for Human-in-the-Loop workflows. This is where users approve, edit, or reject AI-generated content. | [API Reference](hitl-ui/api/index.md) |
| **inference** | This component acts as the "Standard Library" for AI in this repository. It centralizes model loading, prompt rendering, and agent construction to ensure consistency across all services. | [API Reference](inference/api/index.md) |
| **model-training** | This component handles fine-tuning (LoRA), distillation, or training of custom models. | [API Reference](model-training/api/index.md) |
| **shared** | This component holds code that is strictly **common** to multiple components. This usually includes: | [API Reference](shared/api/index.md) |
| **shared-ts** | This library provides common types (e.g. `Result`) and small utilities used across TypeScript services and apps. | [API Reference](shared-ts/api/index.md) |
