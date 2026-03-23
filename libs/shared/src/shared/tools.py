"""OpenAI-format tool definitions for the code execution sandbox.

``CODE_EXECUTOR_TOOLS`` is passed directly to the ``tools`` parameter of any
OpenAI-compatible chat completion endpoint (vLLM, OpenAI API, etc.).

The three tools cover all sandbox interactions a coding agent needs:

- ``execute``    — run Python code, get stdout/stderr back
- ``write_file`` — persist data or scripts to the sandbox filesystem
- ``read_file``  — load data back into context

Installing packages is just normal Python: the model writes
``import subprocess; subprocess.run(["pip", "install", "pandas"], ...)`` or
``!pip install pandas`` when it needs something.

Example::

    from openai import OpenAI
    from tools import CODE_EXECUTOR_TOOLS

    client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")
    response = client.chat.completions.create(
        model="Qwen/Qwen3.5-4B",
        messages=[{"role": "user", "content": "Print hello world"}],
        tools=CODE_EXECUTOR_TOOLS,
        tool_choice="auto",
    )
"""

from __future__ import annotations

from typing import Any

CODE_EXECUTOR_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "execute",
            "description": (
                "Execute a bash script in a persistent, isolated sandbox. "
                "The sandbox runs bash, so ANY shell command works: "
                "ls, grep, curl, pip install, python3, apt-get, df -h, etc. "
                "The working directory and installed packages persist across "
                "calls within the same session. "
                "To run Python, use: python3 -c 'your code' or write a .py "
                "file and run it with python3 script.py. "
                "To install packages: pip install pandas numpy -q "
                "(packages are immediately available in the same call or any "
                "subsequent call)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": (
                            "One or more bash commands / a bash script to run. "
                            "Examples: 'ls -la', 'pip install pandas -q && python3 -c \"import pandas; print(pandas.__version__)\"', "
                            "'python3 script.py'"
                        ),
                    }
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Write a file to the sandbox filesystem. "
                "Use this to create Python scripts, CSV data, config files, "
                "etc. that you can then run or reference in execute calls."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "File path, e.g. /tmp/data.csv or script.py "
                            "(relative paths are in the sandbox working dir)."
                        ),
                    },
                    "content": {
                        "type": "string",
                        "description": "File content as a UTF-8 string.",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read a file from the sandbox filesystem. "
                "Returns the file content as a string."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute or relative path to the file.",
                    }
                },
                "required": ["path"],
            },
        },
    },
]

TOOL_NAMES: frozenset[str] = frozenset(
    t["function"]["name"] for t in CODE_EXECUTOR_TOOLS
)
