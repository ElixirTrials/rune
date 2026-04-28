"""Demo: challenge the model with a full web app project.

Asks the model to build a complete single-file FastAPI REST API
(URL shortener) with multiple endpoints, tested via TestClient.

Usage:
    uv run scripts/demo_project.py
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bootstrap import setup_path

setup_path()  # noqa: E402

os.environ.setdefault("INFERENCE_PROVIDER", "ollama")
os.environ.setdefault("RUNE_MODEL", "Qwen/Qwen3.5-9B")
os.environ.setdefault("RUNE_EXEC_TIMEOUT", "30")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("demo")

from rune_agent.graph import create_graph  # noqa: E402
from rune_agent.state import RuneState  # noqa: E402

TASK = {
    "task_description": """Build a complete URL shortener REST API as a single Python file using FastAPI.

Requirements:
1. A FastAPI app stored in a variable called `app`
2. An in-memory dict called `url_store` mapping short codes to URLs
3. A `POST /shorten` endpoint that accepts JSON `{"url": "..."}` and returns `{"short_code": "...", "url": "..."}`. Generate a random 6-character alphanumeric short code.
4. A `GET /{short_code}` endpoint that returns `{"short_code": "...", "url": "..."}` for a valid code, or raises 404 with `{"detail": "Not found"}` for invalid codes.
5. A `GET /stats/count` endpoint that returns `{"count": N}` with the total number of stored URLs.
6. A `DELETE /{short_code}` endpoint that deletes a URL and returns `{"deleted": true}`, or raises 404.

Use pydantic BaseModel for request/response schemas. Import FastAPI, BaseModel, HTTPException from fastapi and pydantic.""",
    "test_suite": """
# --- Tests using FastAPI TestClient ---
from fastapi.testclient import TestClient

client = TestClient(app)

# Test 1: POST /shorten creates a short URL
resp = client.post("/shorten", json={"url": "https://example.com"})
assert resp.status_code == 200, f"POST /shorten failed: {resp.status_code} {resp.text}"
data = resp.json()
assert "short_code" in data, f"Missing short_code in response: {data}"
assert data["url"] == "https://example.com"
assert len(data["short_code"]) == 6
code1 = data["short_code"]

# Test 2: GET /{short_code} retrieves the URL
resp = client.get(f"/{code1}")
assert resp.status_code == 200, f"GET /{code1} failed: {resp.status_code}"
assert resp.json()["url"] == "https://example.com"

# Test 3: GET with invalid code returns 404
resp = client.get("/ZZZZZZ")
assert resp.status_code == 404, f"Expected 404, got {resp.status_code}"

# Test 4: POST another URL and check stats
resp = client.post("/shorten", json={"url": "https://python.org"})
assert resp.status_code == 200
code2 = resp.json()["short_code"]

resp = client.get("/stats/count")
assert resp.status_code == 200, f"GET /stats/count failed: {resp.status_code}"
assert resp.json()["count"] == 2, f"Expected count 2, got {resp.json()}"

# Test 5: DELETE removes a URL
resp = client.delete(f"/{code1}")
assert resp.status_code == 200
assert resp.json()["deleted"] is True

# Verify it's gone
resp = client.get(f"/{code1}")
assert resp.status_code == 404

# Verify count decreased
resp = client.get("/stats/count")
assert resp.json()["count"] == 1

# Test 6: DELETE invalid code returns 404
resp = client.delete("/ZZZZZZ")
assert resp.status_code == 404

print("All 6 tests passed! URL shortener API works end-to-end.")
""",
}


async def main() -> dict:
    model = os.environ["RUNE_MODEL"]
    logger.info("Model: %s — Challenge: URL Shortener REST API", model)

    state: RuneState = {
        "task_description": TASK["task_description"],
        "task_type": "project",
        "test_suite": TASK["test_suite"],
        "adapter_ids": [],
        "session_id": str(uuid.uuid4()),
        "attempt_count": 0,
        "max_attempts": 5,
        "generated_code": "",
        "stdout": "",
        "stderr": "",
        "exit_code": 0,
        "tests_passed": False,
        "test_count": 0,
        "tests_ran": False,
        "trajectory": [],
        "phase": None,
        "prompt_context": None,
        "finish_reason": None,
        "outcome": None,
    }

    logger.info("Running agent loop (max %d attempts)...", state["max_attempts"])
    print()

    graph = create_graph()
    final_state = await graph.ainvoke(state)

    outcome = final_state["outcome"]
    attempts = final_state["attempt_count"]

    print("\n" + "=" * 70)
    print(f"  OUTCOME: {outcome}")
    print(f"  ATTEMPTS: {attempts} / {state['max_attempts']}")
    print("=" * 70)

    for i, step in enumerate(final_state["trajectory"]):
        status = "PASS" if step["tests_passed"] else "FAIL"
        print(f"\n--- Attempt {i + 1} [{status}] ---")
        print(f"Code ({len(step['generated_code'])} chars):")
        print(step["generated_code"])
        if step["stderr"]:
            print(f"\nStderr:\n{step['stderr'][:800]}")
        if step["stdout"]:
            print(f"\nStdout:\n{step['stdout'][:500]}")

    print("\n" + "=" * 70)
    if outcome == "success":
        print("  The 1.5B model built a working URL shortener API!")
    else:
        print("  The model couldn't solve it in the allotted attempts.")
    print("=" * 70)

    return final_state


if __name__ == "__main__":
    result = asyncio.run(main())
    sys.exit(0 if result["outcome"] == "success" else 1)
