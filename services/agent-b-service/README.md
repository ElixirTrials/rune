# Guardrail Agent

## Purpose
This component implements a guardrailing workflow to validate outputs and ensure safety.

## Implementation Guide

### 1. Define State
Create a typed state in `src/agent_b_service/state.py`:
```python
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    context: dict
```

### 2. Define Nodes
Create node functions in `src/agent_b_service/nodes.py`.

### 3. Build Graph
Assemble the graph in `src/agent_b_service/graph.py`.

## Running the Agent
You can test this agent in isolation using the notebook at `src/agent_b_service/notebooks/playground.ipynb` or via `pytest`.
