"""Experiment harness: test trajectory/prompt designs in isolation.

Loads model + checkpoint once, then runs individual experiments with
different trajectory/prompt combinations. Each experiment takes ~10-15s.

Usage:
    uv run python scripts/experiment_harness.py
"""

from __future__ import annotations

import gc
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("INFERENCE_PROVIDER", "transformers")
os.environ.setdefault("TRANSFORMERS_MODEL_NAME", "google/gemma-2-2b-it")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bootstrap import setup_path

setup_path()

import torch
from model_training.sakana_d2l import (
    download_checkpoint,
    generate_adapter_from_sakana,
)
from peft import PeftModel
from shared.hardware import get_best_device
from shared.template_loader import render_trajectory
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "google/gemma-2-2b-it"
DEVICE = get_best_device()


# ---------------------------------------------------------------------------
# Harness setup — call once
# ---------------------------------------------------------------------------

_model = None
_tokenizer = None
_checkpoint_path = None


def setup() -> None:
    """Load model, tokenizer, and checkpoint once."""
    global _model, _tokenizer, _checkpoint_path

    if _model is not None:
        return

    print(f"Device: {DEVICE}")
    print(f"Model: {MODEL_NAME}")

    _checkpoint_path = str(download_checkpoint(variant="gemma_demo"))
    print(f"Checkpoint: {_checkpoint_path}")

    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    from shared.hardware import resolve_model_dtype

    config = AutoModelForCausalLM.from_pretrained(MODEL_NAME).config
    h = getattr(config, "hidden_size", 2048)
    v = getattr(config, "vocab_size", 32000)
    n = getattr(config, "num_hidden_layers", 24)
    param_count = v * h + n * 12 * h * h
    dtype = resolve_model_dtype(param_count=param_count, device=DEVICE)
    print(f"Inference dtype: {dtype}")

    _model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=dtype
    )
    _model.to(DEVICE)
    _model.eval()
    print("Model loaded.\n")


def generate_adapter(
    trajectory: str, label: str = "exp", max_length: int = 2048
) -> str:
    """Generate a PEFT adapter from a trajectory string."""
    gc.collect()
    torch.cuda.empty_cache()

    tmpdir = tempfile.mkdtemp()
    adapter_dir = str(Path(tmpdir) / label)

    return generate_adapter_from_sakana(
        text=trajectory,
        output_dir=adapter_dir,
        checkpoint_path=_checkpoint_path,
        base_model_name=MODEL_NAME,
        device=DEVICE,
        max_length=max_length,
    )


def generate_text(
    prompt: str,
    system_prompt: str | None = None,
    adapter_path: str | None = None,
    adapter_name: str = "exp",
    max_tokens: int = 512,
    temperature: float = 0.7,
) -> str:
    """Generate text, optionally with an adapter."""
    model = _model

    if adapter_path:
        model = PeftModel.from_pretrained(
            _model, adapter_path, adapter_name=adapter_name
        )
        model.to(DEVICE)
        model.eval()

    content = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
    messages = [{"role": "user", "content": content}]
    formatted = _tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = _tokenizer(
        formatted, return_tensors="pt", truncation=True, max_length=8192
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 0.01),
            top_p=0.9,
            pad_token_id=_tokenizer.pad_token_id,
        )

    new_tokens = outputs[0][input_len:]
    text = _tokenizer.decode(new_tokens, skip_special_tokens=True)

    if adapter_path:
        del model
        gc.collect()
        torch.cuda.empty_cache()

    return text


def run_experiment(
    name: str,
    trajectory: str,
    prompt: str,
    system_prompt: str,
    max_tokens: int = 512,
) -> str:
    """Run one experiment: trajectory → adapter → generate."""
    print(f"--- Experiment: {name} ---")
    print(f"  Trajectory: {len(trajectory)} chars")
    print(f"  Prompt: {prompt[:80]}...")

    adapter_path = generate_adapter(trajectory, label=name)
    print(f"  Adapter: {adapter_path}")

    output = generate_text(
        prompt=prompt,
        system_prompt=system_prompt,
        adapter_path=adapter_path,
        adapter_name=name,
        max_tokens=max_tokens,
    )

    print(f"  Output ({len(output)} chars):")
    for line in output.splitlines()[:15]:
        print(f"    {line}")
    if len(output.splitlines()) > 15:
        print(f"    ... ({len(output.splitlines()) - 15} more lines)")
    print()
    return output


# ---------------------------------------------------------------------------
# Test data — real outputs from our pipeline run
# ---------------------------------------------------------------------------

PROJECT_PROMPT = (
    "Build an event-sourced bank ledger in Python. "
    "LedgerEvent dataclass with fields: event_id (uuid), account_id, "
    "event_type (credit/debit/transfer), amount (Decimal), timestamp, metadata dict. "
    "EventStore class backed by sqlite3 with append-only writes, "
    "replay_events(account_id) returning ordered list. "
    "Ledger class with create_account, credit, debit, transfer (atomic — "
    "debit source + credit dest in one transaction, raise on insufficient funds), "
    "get_balance (replay events to compute), get_balance_at (point-in-time replay). "
    "All monetary amounts must use decimal.Decimal. "
    "Include 15+ unittest tests."
)

PROJECT_LABEL = "Build an event-sourced bank ledger in Python."

# Simulated subtasks and code from a real run
SUBTASKS = [
    {"name": "Data Model Design", "description": "LedgerEvent dataclass"},
    {"name": "EventStore Implementation", "description": "sqlite3 append-only store"},
    {"name": "Ledger Class", "description": "credit/debit/transfer operations"},
]

# Simulated code outputs (realistic failures)
CODE_OUTPUTS = {
    "Data Model Design": (
        "from dataclasses import dataclass\n"
        "from decimal import Decimal\n"
        "from uuid import uuid4\n"
        "from datetime import datetime\n\n"
        "@dataclass\n"
        "class LedgerEvent:\n"
        "    event_id: str\n"
        "    account_id: str\n"
        "    event_type: str\n"
        "    amount: Decimal\n"
        "    timestamp: datetime\n"
        "    metadata: dict = None\n\n"
        "if __name__ == '__main__':\n"
        "    unittest.main()\n"
    ),
    "EventStore Implementation": (
        "from typing import List\n\n"
        "class EventStore:\n"
        "    def __init__(self):\n"
        "        self.events = []\n\n"
        "    def append(self, event):\n"
        "        self.events.append(event)\n\n"
        "    def replay_events(self, account_id):\n"
        "        return [e for e in self.events if e.account_id == account_id]\n\n"
        "if __name__ == '__main__':\n"
        "    unittest.main()\n"
    ),
    "Ledger Class": (
        "class Ledger:\n"
        "    def __init__(self):\n"
        "        self.balance = 0\n\n"
        "    def credit(self, amount):\n"
        "        self.balance += amount\n\n"
        "    def debit(self, amount):\n"
        "        self.balance -= amount\n\n"
        "if __name__ == '__main__':\n"
        "    unittest.main()\n"
    ),
}

INTEGRATION_ERROR = (
    "NameError: name 'unittest' is not defined\n"
    "Failed: FAIL: test_credit (TestLedger); FAIL: test_transfer (TestLedger)\n"
    "AssertionError: Expected Decimal('100') but got 100"
)

DIAGNOSE_SYSTEM = (
    "You are a code diagnostician. Identify which subtasks have bugs. "
    "Output ONLY a numbered list, never code."
)

CODE_SYSTEM = (
    "You are a Python code generator. Output only code, no explanation."
)


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------


def experiment_1a_diagnose_grounding():
    """Does telling the model 'you know the code' help?"""
    trajectory = render_trajectory(
        "diagnose",
        project=PROJECT_PROMPT,
        code_outputs=CODE_OUTPUTS,
        integration_error=INTEGRATION_ERROR,
        repair_history=[],
    )

    prompt = (
        "You have been given the full project code and integration errors.\n"
        "Based on what you know, which subtasks need repair?\n"
        "Output ONLY a numbered list:\n"
        "1. subtask name — what is wrong and how to fix it"
    )

    return run_experiment("1a_grounding", trajectory, prompt, DIAGNOSE_SYSTEM)


def experiment_1b_diagnose_with_example():
    """Add an example diagnosis to the trajectory."""
    base_traj = render_trajectory(
        "diagnose",
        project=PROJECT_PROMPT,
        code_outputs=CODE_OUTPUTS,
        integration_error=INTEGRATION_ERROR,
        repair_history=[],
    )

    # Add example to trajectory so adapter encodes the pattern
    example = (
        "\n\nEXAMPLE DIAGNOSIS (for a different project):\n"
        "1. Auth Module — missing import for hashlib, password_hash() undefined\n"
        "2. Database Layer — uses sqlite3.connect() but never closes connection\n"
        "3. API Routes — references User class from auth module but imports wrong name\n"
    )
    trajectory = base_traj + example

    prompt = (
        "Which subtasks need repair? Output a numbered list:\n"
        "1. subtask name — what is wrong and how to fix it"
    )

    return run_experiment("1b_example", trajectory, prompt, DIAGNOSE_SYSTEM)


def experiment_1c_diagnose_error_in_prompt():
    """Put the specific error in the prompt as anchor."""
    trajectory = render_trajectory(
        "diagnose",
        project=PROJECT_PROMPT,
        code_outputs=CODE_OUTPUTS,
        integration_error=INTEGRATION_ERROR,
        repair_history=[],
    )

    prompt = (
        "The integration tests failed with this error:\n"
        f"  {INTEGRATION_ERROR.splitlines()[0]}\n\n"
        "Which subtask produced this error? Output a numbered list:\n"
        "1. subtask name — what is wrong and how to fix it"
    )

    return run_experiment("1c_error_anchor", trajectory, prompt, DIAGNOSE_SYSTEM)


def experiment_1d_diagnose_simple_case():
    """Can the model diagnose an obvious single error?"""
    simple_code = {
        "Math Utils": (
            "def add(a, b):\n"
            "    return a + b\n\n"
            "def multiply(a, b):\n"
            "    return a * b\n"
        ),
        "String Utils": (
            "def reverse(s):\n"
            "    return s[::-1]\n\n"
            "def uppercase(s):\n"
            "    return s.upper()\n"
        ),
        "Main App": (
            "from math_utils import add\n"
            "from string_utils import reverse\n\n"
            "result = add(1, 2)\n"
            "print(reverse('hello'))\n"
        ),
    }

    trajectory = render_trajectory(
        "diagnose",
        project="Build a utility library with math, string, and main app modules.",
        code_outputs=simple_code,
        integration_error="ModuleNotFoundError: No module named 'math_utils'",
        repair_history=[],
    )

    prompt = (
        "The integration failed with: ModuleNotFoundError: No module named 'math_utils'\n"
        "Which subtask needs repair? Output a numbered list:\n"
        "1. subtask name — what is wrong"
    )

    return run_experiment("1d_simple", trajectory, prompt, DIAGNOSE_SYSTEM)


def experiment_2a_code_with_prior_subtask():
    """Does feeding prior subtask code via adapter produce compatible code?"""
    # First subtask's code (the Data Model)
    data_model_code = CODE_OUTPUTS["Data Model Design"]

    # Build trajectory for EventStore that includes the data model
    trajectory = render_trajectory(
        "code",
        subtask=SUBTASKS[1],  # EventStore
        subtask_index=2,
        total_subtasks=3,
        plan="Implement EventStore backed by sqlite3. Use LedgerEvent from Data Model.",
        existing_code="",
        project=PROJECT_PROMPT,
    )
    # Append prior subtask code to trajectory
    trajectory += (
        "\n\nPRIOR SUBTASK CODE (Data Model Design — already implemented):\n"
        f"{data_model_code}"
    )

    prompt = (
        "You are implementing the subtask: EventStore Implementation\n"
        f"Project: {PROJECT_LABEL}\n"
        "Follow the architecture plan in your context.\n"
        "Use the LedgerEvent class from the prior subtask.\n"
        "Write tests FIRST, then implement to pass them.\n"
        "Include unittest.TestCase tests. End with:\n"
        "if __name__ == '__main__': unittest.main()"
    )

    return run_experiment("2a_prior_context", trajectory, prompt, CODE_SYSTEM, max_tokens=1024)


def experiment_2b_sequential_generation():
    """Generate Data Model first, then feed it into EventStore adapter."""
    # Step 1: Generate Data Model code
    print("=== Step 1: Generate Data Model ===")
    dm_traj = render_trajectory(
        "code",
        subtask=SUBTASKS[0],  # Data Model
        subtask_index=1,
        total_subtasks=3,
        plan=(
            "Define LedgerEvent dataclass with fields: event_id (uuid4), "
            "account_id (str), event_type (str: credit/debit/transfer), "
            "amount (Decimal), timestamp (datetime), metadata (dict). "
            "Use dataclasses and decimal.Decimal."
        ),
        existing_code="",
        project=PROJECT_PROMPT,
    )

    dm_prompt = (
        "You are implementing the subtask: Data Model Design\n"
        f"Project: {PROJECT_LABEL}\n"
        "Follow the architecture plan in your context.\n"
        "Write tests FIRST, then implement to pass them.\n"
        "Include unittest.TestCase tests. End with:\n"
        "if __name__ == '__main__': unittest.main()"
    )

    dm_code = run_experiment(
        "2b_step1_datamodel", dm_traj, dm_prompt, CODE_SYSTEM, max_tokens=1024
    )

    # Step 2: Generate EventStore using Data Model code as context
    print("=== Step 2: Generate EventStore (with Data Model context) ===")
    es_traj = render_trajectory(
        "code",
        subtask=SUBTASKS[1],  # EventStore
        subtask_index=2,
        total_subtasks=3,
        plan=(
            "Implement EventStore class backed by sqlite3. "
            "append_event(event: LedgerEvent), "
            "replay_events(account_id: str) -> list[LedgerEvent]. "
            "Use the LedgerEvent dataclass defined in prior subtask."
        ),
        existing_code="",
        project=PROJECT_PROMPT,
    )
    # Inject prior subtask code into trajectory
    es_traj += (
        "\n\nCOMPLETED DEPENDENCY (Data Model Design):\n"
        f"{dm_code[:800]}"
    )

    es_prompt = (
        "You are implementing the subtask: EventStore Implementation\n"
        f"Project: {PROJECT_LABEL}\n"
        "Follow the architecture plan in your context.\n"
        "The Data Model subtask is already complete — use its classes.\n"
        "Write tests FIRST, then implement to pass them.\n"
        "Include unittest.TestCase tests. End with:\n"
        "if __name__ == '__main__': unittest.main()"
    )

    es_code = run_experiment(
        "2b_step2_eventstore", es_traj, es_prompt, CODE_SYSTEM, max_tokens=1024
    )

    return dm_code, es_code


def experiment_2c_shared_interface_spec():
    """Does embedding a shared interface spec in trajectory help?"""
    interface_spec = (
        "SHARED INTERFACES (all subtasks must use these exact definitions):\n"
        "  @dataclass\n"
        "  class LedgerEvent:\n"
        "      event_id: str        # uuid4\n"
        "      account_id: str\n"
        "      event_type: str      # 'credit' | 'debit' | 'transfer'\n"
        "      amount: Decimal\n"
        "      timestamp: datetime\n"
        "      metadata: dict\n\n"
        "  class EventStore:\n"
        "      def append_event(self, event: LedgerEvent) -> None: ...\n"
        "      def replay_events(self, account_id: str) -> list[LedgerEvent]: ...\n\n"
        "  class Ledger:\n"
        "      def credit(self, account_id: str, amount: Decimal) -> LedgerEvent: ...\n"
        "      def debit(self, account_id: str, amount: Decimal) -> LedgerEvent: ...\n"
        "      def get_balance(self, account_id: str) -> Decimal: ...\n"
    )

    trajectory = render_trajectory(
        "code",
        subtask=SUBTASKS[1],  # EventStore
        subtask_index=2,
        total_subtasks=3,
        plan="Implement EventStore per the shared interface spec.",
        existing_code="",
        project=PROJECT_PROMPT,
    )
    trajectory += f"\n\n{interface_spec}"

    prompt = (
        "You are implementing the subtask: EventStore Implementation\n"
        f"Project: {PROJECT_LABEL}\n"
        "Follow the shared interface spec in your context.\n"
        "Write tests FIRST, then implement to pass them.\n"
        "Include unittest.TestCase tests. End with:\n"
        "if __name__ == '__main__': unittest.main()"
    )

    return run_experiment(
        "2c_interface_spec", trajectory, prompt, CODE_SYSTEM, max_tokens=1024
    )


# ---------------------------------------------------------------------------
# Round 2: Informed by adapter mechanics (512 token limit, rank 8, down_proj)
# ---------------------------------------------------------------------------


def experiment_1e_subtask_names_in_prompt():
    """Put subtask names + error in prompt. Adapter just carries project topic."""
    # Short trajectory — just the project context for topic bias
    trajectory = f"PROJECT: {PROJECT_PROMPT[:400]}"

    prompt = (
        "These subtasks were built for the bank ledger project:\n"
        "  1. Data Model Design\n"
        "  2. EventStore Implementation\n"
        "  3. Ledger Class\n\n"
        "Integration error: NameError: name 'unittest' is not defined\n"
        "  (multiple subtasks call unittest.main() without importing unittest)\n\n"
        "Which subtask(s) need repair? Output a numbered list:\n"
        "1. subtask name — what is wrong and how to fix it"
    )

    return run_experiment("1e_names_in_prompt", trajectory, prompt, DIAGNOSE_SYSTEM)


def experiment_1f_no_adapter_diagnose():
    """No adapter at all — pure prompt diagnosis. Baseline."""
    prompt = (
        "These subtasks were built for an event-sourced bank ledger:\n"
        "  1. Data Model Design — defines LedgerEvent dataclass\n"
        "  2. EventStore Implementation — sqlite3 event store\n"
        "  3. Ledger Class — credit/debit/transfer operations\n\n"
        "Integration error: NameError: name 'unittest' is not defined\n"
        "  Three subtasks call unittest.main() at the end but never import unittest.\n\n"
        "Which subtask(s) need repair? Output a numbered list:\n"
        "1. subtask name — what is wrong and how to fix it"
    )

    # No adapter — generate with base model only
    print("--- Experiment: 1f_no_adapter ---")
    print(f"  Prompt: {prompt[:80]}...")
    output = generate_text(
        prompt=prompt,
        system_prompt=DIAGNOSE_SYSTEM,
        adapter_path=None,
        max_tokens=512,
    )
    print(f"  Output ({len(output)} chars):")
    for line in output.splitlines()[:15]:
        print(f"    {line}")
    print()
    return output


def experiment_2d_frontloaded_trajectory():
    """Put dependency signatures FIRST in trajectory to survive 512 truncation."""
    # Front-load: dependency code first, then role/plan
    trajectory = (
        "DEPENDENCY (already implemented, use these exact classes):\n"
        "@dataclass\n"
        "class LedgerEvent:\n"
        "    event_id: str\n"
        "    account_id: str\n"
        "    event_type: str  # credit/debit/transfer\n"
        "    amount: Decimal\n"
        "    timestamp: datetime\n"
        "    metadata: dict = None\n\n"
        "ROLE: python-coder\n"
        f"PROJECT: {PROJECT_PROMPT[:200]}\n"
        "SUBTASK: EventStore Implementation (2/3)\n"
        "PLAN: Implement EventStore with sqlite3. "
        "append_event(LedgerEvent), replay_events(account_id) -> list[LedgerEvent].\n"
        "PRACTICES: Self-contained file. Copy dependency classes into your code.\n"
        "Include unittest.TestCase tests.\n"
    )

    prompt = (
        "You are implementing the subtask: EventStore Implementation\n"
        f"Project: {PROJECT_LABEL}\n"
        "The LedgerEvent dataclass is defined in your context — "
        "copy it into your code.\n"
        "Write tests FIRST, then implement to pass them.\n"
        "Include unittest.TestCase tests. End with:\n"
        "if __name__ == '__main__': unittest.main()"
    )

    return run_experiment(
        "2d_frontloaded", trajectory, prompt, CODE_SYSTEM, max_tokens=1024
    )


def experiment_2e_dependency_in_prompt():
    """Put dependency class definition directly in the prompt."""
    # Minimal trajectory — just project topic
    trajectory = (
        "ROLE: python-coder\n"
        f"PROJECT: {PROJECT_PROMPT[:300]}\n"
        "SUBTASK: EventStore Implementation (2/3)\n"
        "PLAN: sqlite3-backed event store with append and replay.\n"
    )

    prompt = (
        "You are implementing the subtask: EventStore Implementation\n"
        f"Project: {PROJECT_LABEL}\n\n"
        "The LedgerEvent dataclass (already implemented by prior subtask):\n"
        "```\n"
        "@dataclass\n"
        "class LedgerEvent:\n"
        "    event_id: str\n"
        "    account_id: str\n"
        "    event_type: str\n"
        "    amount: Decimal\n"
        "    timestamp: datetime\n"
        "    metadata: dict = None\n"
        "```\n"
        "Copy this class definition into your file, then implement EventStore.\n"
        "Use sqlite3. Include unittest.TestCase tests. End with:\n"
        "if __name__ == '__main__': unittest.main()"
    )

    return run_experiment(
        "2e_dep_in_prompt", trajectory, prompt, CODE_SYSTEM, max_tokens=1024
    )


def experiment_2f_hybrid():
    """Adapter carries topic/style, prompt carries interface contract."""
    # Trajectory focused on the "how" — coding style, project context
    trajectory = (
        "ROLE: python-coder\n"
        f"PROJECT: {PROJECT_PROMPT[:300]}\n"
        "SUBTASK: EventStore Implementation (2/3)\n"
        "PLAN: Implement EventStore backed by sqlite3 with append-only writes.\n"
        "Store LedgerEvent objects. Support replay_events(account_id).\n"
        "Use decimal.Decimal for amounts, dataclass for models.\n"
        "PRACTICES: Self-contained file, all definitions inline.\n"
        "Test-first: write TestCase tests before implementation.\n"
        "import unittest at top of file.\n"
    )

    prompt = (
        "You are implementing the subtask: EventStore Implementation\n"
        f"Project: {PROJECT_LABEL}\n\n"
        "Interface to implement:\n"
        "  class EventStore:\n"
        "      def __init__(self, db_path=':memory:')\n"
        "      def append_event(self, event: LedgerEvent) -> None\n"
        "      def replay_events(self, account_id: str) -> list[LedgerEvent]\n\n"
        "Copy the LedgerEvent dataclass into your code (event_id, account_id, "
        "event_type, amount as Decimal, timestamp, metadata).\n"
        "Use sqlite3. Write tests FIRST. End with:\n"
        "if __name__ == '__main__': unittest.main()"
    )

    return run_experiment(
        "2f_hybrid", trajectory, prompt, CODE_SYSTEM, max_tokens=1024
    )


def experiment_2g_short_trajectory():
    """Minimal 256-token trajectory to reduce noise. All detail in prompt."""
    # Very short trajectory — just enough for topic bias
    trajectory = (
        "ROLE: python-coder\n"
        "PROJECT: Event-sourced bank ledger\n"
        "SUBTASK: EventStore with sqlite3\n"
    )

    prompt = (
        "You are implementing EventStore for an event-sourced bank ledger.\n\n"
        "Requirements:\n"
        "- EventStore class backed by sqlite3 (in-memory default)\n"
        "- append_event(event: LedgerEvent) stores events\n"
        "- replay_events(account_id: str) returns events in order\n\n"
        "LedgerEvent dataclass (copy into your code):\n"
        "  event_id: str, account_id: str, event_type: str,\n"
        "  amount: Decimal, timestamp: datetime, metadata: dict\n\n"
        "Write self-contained Python with unittest.TestCase tests.\n"
        "import unittest and from decimal import Decimal at top.\n"
        "End with: if __name__ == '__main__': unittest.main()"
    )

    return run_experiment(
        "2g_short_traj", trajectory, prompt, CODE_SYSTEM, max_tokens=1024
    )


def experiment_2h_long_trajectory():
    """Use max_length=2048 for activation extraction. Embed full context."""
    # Rich trajectory with full dependency code, plan, AND project spec
    trajectory = (
        f"PROJECT: {PROJECT_PROMPT}\n\n"
        "COMPLETED SUBTASK 1 — Data Model Design:\n"
        "```python\n"
        "import unittest\n"
        "from dataclasses import dataclass, field\n"
        "from decimal import Decimal\n"
        "from datetime import datetime\n"
        "from uuid import uuid4\n\n"
        "@dataclass\n"
        "class LedgerEvent:\n"
        "    event_id: str\n"
        "    account_id: str\n"
        "    event_type: str  # 'credit', 'debit', 'transfer'\n"
        "    amount: Decimal\n"
        "    timestamp: datetime\n"
        "    metadata: dict = field(default_factory=dict)\n\n"
        "    @staticmethod\n"
        "    def create(account_id, event_type, amount, metadata=None):\n"
        "        return LedgerEvent(\n"
        "            event_id=str(uuid4()),\n"
        "            account_id=account_id,\n"
        "            event_type=event_type,\n"
        "            amount=Decimal(str(amount)),\n"
        "            timestamp=datetime.now(),\n"
        "            metadata=metadata or {},\n"
        "        )\n\n"
        "class TestLedgerEvent(unittest.TestCase):\n"
        "    def test_create_credit(self):\n"
        "        event = LedgerEvent.create('acct1', 'credit', 100)\n"
        "        self.assertEqual(event.account_id, 'acct1')\n"
        "        self.assertEqual(event.event_type, 'credit')\n"
        "        self.assertEqual(event.amount, Decimal('100'))\n\n"
        "    def test_metadata_default(self):\n"
        "        event = LedgerEvent.create('acct1', 'debit', 50)\n"
        "        self.assertEqual(event.metadata, {})\n\n"
        "if __name__ == '__main__':\n"
        "    unittest.main()\n"
        "```\n\n"
        "YOUR SUBTASK (2/3): EventStore Implementation\n"
        "PLAN: Implement EventStore backed by sqlite3.\n"
        "- __init__(db_path=':memory:') creates table if not exists\n"
        "- append_event(event: LedgerEvent) inserts into sqlite\n"
        "- replay_events(account_id: str) -> list[LedgerEvent] ordered by timestamp\n"
        "- All code self-contained in one file\n"
        "- Copy LedgerEvent definition into your file\n"
        "- Write tests first, then implement\n"
    )

    prompt = (
        "You are implementing the subtask: EventStore Implementation\n"
        f"Project: {PROJECT_LABEL}\n"
        "Use the LedgerEvent class from the completed dependency in your context.\n"
        "Copy it into your code. Use sqlite3.\n"
        "Write tests FIRST, then implement to pass them.\n"
        "End with: if __name__ == '__main__': unittest.main()"
    )

    return run_experiment(
        "2h_long_traj", trajectory, prompt, CODE_SYSTEM, max_tokens=1024
    )


def experiment_1g_diagnose_long_trajectory():
    """Embed full code in trajectory with max_length=2048 for diagnosis."""
    # Full code for all subtasks in the trajectory
    full_code_traj = f"PROJECT: {PROJECT_PROMPT}\n\n"
    for name, code in CODE_OUTPUTS.items():
        full_code_traj += f"SUBTASK CODE — {name}:\n```python\n{code}```\n\n"
    full_code_traj += (
        f"INTEGRATION ERROR:\n{INTEGRATION_ERROR}\n\n"
        "TASK: Identify which subtask(s) have bugs that caused the error.\n"
    )

    prompt = (
        "The integration tests failed with:\n"
        f"  {INTEGRATION_ERROR.splitlines()[0]}\n\n"
        "Subtasks: Data Model Design, EventStore Implementation, Ledger Class\n"
        "Which subtask(s) need repair?\n"
        "1. subtask name — what is wrong and how to fix it"
    )

    return run_experiment(
        "1g_long_traj_diagnose", full_code_traj, prompt, DIAGNOSE_SYSTEM
    )


# ---------------------------------------------------------------------------
# Round 3: Maximizing adapter reliance
# Key insight: rank-8 down_proj adapts MLP output gating (style/topic),
# not attention or specific tokens. We need to find what the adapter CAN
# carry vs what must be in the prompt.
# ---------------------------------------------------------------------------


def _score_code(code: str) -> dict[str, bool]:
    """Score code output for key quality signals."""
    return {
        "LedgerEvent": "LedgerEvent" in code,
        "sqlite3": "sqlite3" in code,
        "Decimal": "Decimal" in code,
        "import_ut": "import unittest" in code,
        "TestCase": "TestCase" in code,
        "dataclass": "dataclass" in code,
        "inline_def": (
            "class LedgerEvent" in code and "from " not in code.split("class LedgerEvent")[0][-100:]
        ),
        "replay": "replay" in code.lower(),
    }


def experiment_3a_adapter_influence_measurement():
    """Measure: same prompt, with vs without adapter. How much does it shift?"""
    # Rich trajectory
    trajectory = (
        f"PROJECT: {PROJECT_PROMPT}\n\n"
        "COMPLETED DEPENDENCY:\n"
        "```python\n"
        "from dataclasses import dataclass, field\n"
        "from decimal import Decimal\n"
        "from datetime import datetime\n"
        "from uuid import uuid4\n\n"
        "@dataclass\n"
        "class LedgerEvent:\n"
        "    event_id: str\n"
        "    account_id: str\n"
        "    event_type: str  # credit/debit/transfer\n"
        "    amount: Decimal\n"
        "    timestamp: datetime\n"
        "    metadata: dict = field(default_factory=dict)\n"
        "```\n\n"
        "YOUR TASK: Implement EventStore backed by sqlite3.\n"
        "- append_event(event: LedgerEvent)\n"
        "- replay_events(account_id: str) -> list[LedgerEvent]\n"
        "Copy the LedgerEvent class into your file.\n"
        "Use decimal.Decimal for amounts.\n"
    )

    # Same prompt for both
    prompt = (
        "You are implementing EventStore.\n"
        f"Project: {PROJECT_LABEL}\n"
        "Write complete, self-contained Python with tests.\n"
        "End with: if __name__ == '__main__': unittest.main()"
    )

    print("=== 3a: Adapter influence measurement ===")

    # Without adapter
    print("  [no adapter]")
    no_adapter = generate_text(
        prompt=prompt, system_prompt=CODE_SYSTEM,
        adapter_path=None, max_tokens=1024, temperature=0.3,
    )
    print(f"    Score: {_score_code(no_adapter)}")
    for line in no_adapter.splitlines()[:8]:
        print(f"    {line}")

    # With adapter
    adapter_path = generate_adapter(trajectory, label="3a")
    print("  [with adapter]")
    with_adapter = generate_text(
        prompt=prompt, system_prompt=CODE_SYSTEM,
        adapter_path=adapter_path, adapter_name="3a",
        max_tokens=1024, temperature=0.3,
    )
    print(f"    Score: {_score_code(with_adapter)}")
    for line in with_adapter.splitlines()[:8]:
        print(f"    {line}")

    print()
    return {"no_adapter": _score_code(no_adapter), "with_adapter": _score_code(with_adapter)}


def experiment_3b_trajectory_as_exemplar_code():
    """Put complete working example code in trajectory (not description).
    The adapter should absorb the code pattern, not prose about it."""
    # Trajectory IS working code — like a code document
    trajectory = (
        "# Event-sourced bank ledger — EventStore implementation\n"
        "import unittest\n"
        "import sqlite3\n"
        "from dataclasses import dataclass, field\n"
        "from decimal import Decimal\n"
        "from datetime import datetime\n"
        "from uuid import uuid4\n\n"
        "@dataclass\n"
        "class LedgerEvent:\n"
        "    event_id: str\n"
        "    account_id: str\n"
        "    event_type: str\n"
        "    amount: Decimal\n"
        "    timestamp: datetime\n"
        "    metadata: dict = field(default_factory=dict)\n\n"
        "class EventStore:\n"
        "    def __init__(self, db_path=':memory:'):\n"
        "        self.conn = sqlite3.connect(db_path)\n"
        "        self.conn.execute(\n"
        "            'CREATE TABLE IF NOT EXISTS events '\n"
        "            '(event_id TEXT, account_id TEXT, event_type TEXT, '\n"
        "            'amount TEXT, timestamp TEXT, metadata TEXT)'\n"
        "        )\n\n"
        "    def append_event(self, event: LedgerEvent) -> None:\n"
        "        self.conn.execute(\n"
        "            'INSERT INTO events VALUES (?,?,?,?,?,?)',\n"
        "            (event.event_id, event.account_id, event.event_type,\n"
        "             str(event.amount), event.timestamp.isoformat(),\n"
        "             str(event.metadata))\n"
        "        )\n"
        "        self.conn.commit()\n\n"
        "    def replay_events(self, account_id: str):\n"
        "        rows = self.conn.execute(\n"
        "            'SELECT * FROM events WHERE account_id=? ORDER BY timestamp',\n"
        "            (account_id,)\n"
        "        ).fetchall()\n"
        "        return [LedgerEvent(*r) for r in rows]\n\n"
        "class TestEventStore(unittest.TestCase):\n"
        "    def test_append_and_replay(self):\n"
        "        store = EventStore()\n"
        "        event = LedgerEvent(\n"
        "            str(uuid4()), 'acct1', 'credit',\n"
        "            Decimal('100'), datetime.now(), {}\n"
        "        )\n"
        "        store.append_event(event)\n"
        "        events = store.replay_events('acct1')\n"
        "        self.assertEqual(len(events), 1)\n\n"
        "if __name__ == '__main__':\n"
        "    unittest.main()\n"
    )

    # Minimal prompt — force adapter to carry the pattern
    prompt = (
        "You are implementing EventStore.\n"
        f"Project: {PROJECT_LABEL}\n"
        "Write code following the pattern in your context.\n"
        "End with: if __name__ == '__main__': unittest.main()"
    )

    return run_experiment(
        "3b_exemplar_code", trajectory, prompt, CODE_SYSTEM, max_tokens=1024
    )


def experiment_3c_repetition_in_trajectory():
    """Repeat key terms/patterns multiple times to strengthen adapter signal."""
    trajectory = (
        f"PROJECT: {PROJECT_PROMPT[:300]}\n\n"
        # Repeat the key interface 3 times in different forms
        "INTERFACE:\n"
        "EventStore uses sqlite3. LedgerEvent uses Decimal.\n"
        "EventStore.append_event(LedgerEvent) stores to sqlite3.\n"
        "EventStore.replay_events(account_id) returns list[LedgerEvent].\n\n"
        "REQUIREMENTS:\n"
        "- EventStore backed by sqlite3 (not in-memory dict)\n"
        "- LedgerEvent with Decimal amounts (not float)\n"
        "- append_event stores LedgerEvent to sqlite3 database\n"
        "- replay_events reads from sqlite3, returns LedgerEvent list\n\n"
        "KEY PATTERNS:\n"
        "import sqlite3\n"
        "from decimal import Decimal\n"
        "from dataclasses import dataclass\n"
        "@dataclass\n"
        "class LedgerEvent: ...\n"
        "class EventStore:\n"
        "    def __init__(self, db=':memory:'): self.conn = sqlite3.connect(db)\n"
        "    def append_event(self, event: LedgerEvent): ...\n"
        "    def replay_events(self, account_id: str) -> list: ...\n\n"
        "REMEMBER:\n"
        "- import unittest\n"
        "- import sqlite3\n"
        "- from decimal import Decimal\n"
        "- @dataclass class LedgerEvent\n"
        "- class EventStore with sqlite3.connect()\n"
        "- class TestEventStore(unittest.TestCase)\n"
    )

    prompt = (
        "You are implementing EventStore.\n"
        f"Project: {PROJECT_LABEL}\n"
        "Write complete, self-contained Python.\n"
        "End with: if __name__ == '__main__': unittest.main()"
    )

    return run_experiment(
        "3c_repetition", trajectory, prompt, CODE_SYSTEM, max_tokens=1024
    )


def experiment_3d_minimal_prompt_max_adapter():
    """Bare minimum prompt. Everything via adapter. Tests adapter reliance."""
    # Rich, code-heavy trajectory
    trajectory = (
        f"PROJECT: {PROJECT_PROMPT}\n\n"
        "import unittest\n"
        "import sqlite3\n"
        "from dataclasses import dataclass, field\n"
        "from decimal import Decimal\n"
        "from datetime import datetime\n"
        "from uuid import uuid4\n\n"
        "@dataclass\n"
        "class LedgerEvent:\n"
        "    event_id: str\n"
        "    account_id: str\n"
        "    event_type: str\n"
        "    amount: Decimal\n"
        "    timestamp: datetime\n"
        "    metadata: dict = field(default_factory=dict)\n\n"
        "class EventStore:\n"
        "    '''sqlite3-backed append-only event store.'''\n"
        "    def __init__(self, db_path=':memory:'): ...\n"
        "    def append_event(self, event: LedgerEvent) -> None: ...\n"
        "    def replay_events(self, account_id: str) -> list[LedgerEvent]: ...\n\n"
        "class TestEventStore(unittest.TestCase):\n"
        "    def test_append_and_replay(self): ...\n"
        "    def test_replay_empty(self): ...\n"
        "    def test_multiple_accounts(self): ...\n\n"
        "if __name__ == '__main__':\n"
        "    unittest.main()\n"
    )

    # Bare prompt
    prompt = "Write code."

    return run_experiment(
        "3d_minimal_prompt", trajectory, prompt, CODE_SYSTEM, max_tokens=1024
    )


def experiment_3e_ab_test_different_adapters():
    """Same prompt, two different adapters. Measures adapter discrimination."""
    prompt = (
        "Write a Python class with tests.\n"
        "End with: if __name__ == '__main__': unittest.main()"
    )

    # Adapter A: bank ledger context
    traj_a = (
        "Event-sourced bank ledger with sqlite3.\n"
        "LedgerEvent dataclass, EventStore, Decimal amounts.\n"
        "import sqlite3, from decimal import Decimal.\n"
        "class EventStore with append_event and replay_events.\n"
    )

    # Adapter B: completely different project
    traj_b = (
        "HTTP request router with regex pattern matching.\n"
        "Route dataclass, Router class, path parameters.\n"
        "import re, from dataclasses import dataclass.\n"
        "class Router with add_route and match methods.\n"
    )

    print("=== 3e: A/B adapter discrimination ===")

    adapter_a = generate_adapter(traj_a, label="3e_bank")
    print("  [Adapter A: bank ledger]")
    out_a = generate_text(
        prompt=prompt, system_prompt=CODE_SYSTEM,
        adapter_path=adapter_a, adapter_name="3e_bank",
        max_tokens=512, temperature=0.3,
    )
    for line in out_a.splitlines()[:10]:
        print(f"    {line}")

    adapter_b = generate_adapter(traj_b, label="3e_router")
    print("  [Adapter B: HTTP router]")
    out_b = generate_text(
        prompt=prompt, system_prompt=CODE_SYSTEM,
        adapter_path=adapter_b, adapter_name="3e_router",
        max_tokens=512, temperature=0.3,
    )
    for line in out_b.splitlines()[:10]:
        print(f"    {line}")

    # Check what each produced
    a_has_bank = any(w in out_a.lower() for w in ["ledger", "bank", "event", "transaction", "sqlite"])
    b_has_router = any(w in out_b.lower() for w in ["router", "route", "path", "http", "url", "match"])
    print(f"\n  A talks about banking: {a_has_bank}")
    print(f"  B talks about routing: {b_has_router}")
    print(f"  Adapters discriminate: {a_has_bank and b_has_router}")
    print()

    return {"a_banking": a_has_bank, "b_routing": b_has_router}


def experiment_3f_low_temperature():
    """Lower temperature to reduce sampling noise and amplify adapter signal."""
    trajectory = (
        f"PROJECT: {PROJECT_PROMPT[:300]}\n\n"
        "import unittest\n"
        "import sqlite3\n"
        "from dataclasses import dataclass, field\n"
        "from decimal import Decimal\n"
        "from datetime import datetime\n\n"
        "@dataclass\n"
        "class LedgerEvent:\n"
        "    event_id: str\n"
        "    account_id: str\n"
        "    event_type: str\n"
        "    amount: Decimal\n"
        "    timestamp: datetime\n"
        "    metadata: dict = field(default_factory=dict)\n\n"
        "class EventStore:\n"
        "    def __init__(self, db=':memory:'): self.conn = sqlite3.connect(db)\n"
        "    def append_event(self, event: LedgerEvent): ...\n"
        "    def replay_events(self, account_id: str) -> list: ...\n"
    )

    prompt = (
        "You are implementing EventStore.\n"
        f"Project: {PROJECT_LABEL}\n"
        "Write complete, self-contained Python with sqlite3.\n"
        "Define LedgerEvent inline. Use Decimal.\n"
        "End with: if __name__ == '__main__': unittest.main()"
    )

    print("=== 3f: Temperature comparison ===")

    adapter_path = generate_adapter(trajectory, label="3f")

    for temp in [0.1, 0.3, 0.7]:
        print(f"  [temp={temp}]")
        out = generate_text(
            prompt=prompt, system_prompt=CODE_SYSTEM,
            adapter_path=adapter_path, adapter_name="3f",
            max_tokens=1024, temperature=temp,
        )
        scores = _score_code(out)
        hits = sum(1 for v in scores.values() if v)
        print(f"    Score: {hits}/{len(scores)} — {scores}")
        for line in out.splitlines()[:5]:
            print(f"    {line}")
        print()

    return "see output"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    setup()

    print("=" * 70)
    print("  ROUND 2: Adapter-aware experiments")
    print("=" * 70)
    print()

    results_3: dict[str, Any] = {}

    print("=" * 70)
    print("  ROUND 3: Adapter reliance experiments (2048 tokens)")
    print("=" * 70)
    print()

    results_3["3a"] = experiment_3a_adapter_influence_measurement()
    results_3["3b"] = experiment_3b_trajectory_as_exemplar_code()
    results_3["3c"] = experiment_3c_repetition_in_trajectory()
    results_3["3d"] = experiment_3d_minimal_prompt_max_adapter()
    results_3["3e"] = experiment_3e_ab_test_different_adapters()
    results_3["3f"] = experiment_3f_low_temperature()

    # Summary
    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print()

    print("Round 3 — Adapter reliance:")
    for key, result in results_3.items():
        if isinstance(result, str) and result != "see output":
            scores = _score_code(result)
            hits = sum(1 for v in scores.values() if v)
            print(f"  {key}: {hits}/{len(scores)} — {scores}")
        elif isinstance(result, dict):
            print(f"  {key}: {result}")


if __name__ == "__main__":
    main()
