# Guest Interaction Agent

## Purpose
This component implements a specific AI agent workflow for interacting with guests. It uses LangGraph to define the flow and state.

## Implementation Guide

### 1. Define State
Create a typed state in `src/rune_agent/state.py`:
```python
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    context: dict
```

### 2. Define Nodes
Create node functions in `src/rune_agent/nodes.py`.
```python
async def extraction_node(state: AgentState):
    # Call model via inference component
    result = await extractor(state["context"])
    return {"extracted": result}
```

### 3. Build Graph
Assemble the graph in `src/rune_agent/graph.py`.
```python
from langgraph.graph import StateGraph, START, END

def create_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("extract", extraction_node)
    workflow.add_edge(START, "extract")
    workflow.add_edge("extract", END)
    return workflow.compile()
```

## Running the Agent
You can test this agent in isolation using the notebook at `src/rune_agent/notebooks/playground.ipynb` or via `pytest`.
