#!/usr/bin/env python3
"""Verify XGrammar constrained decoding works with Qwen3.5-9B.

Auto-detects device: NF4 on CUDA, float32 on CPU/MPS.

    uv run python scripts/verify_xgrammar_nf4.py

Exits 0 if all 5 generations parse as valid JSON matching the schema.
Exits 1 on any failure.
"""

from __future__ import annotations

import json
import sys

from pydantic import BaseModel, Field


class Subtask(BaseModel):
    name: str = Field(min_length=1)
    description: str = ""
    depends_on: list[str] = []


class DecomposeResult(BaseModel):
    subtasks: list[Subtask] = Field(min_length=1, max_length=8)


PROMPT = (
    "Decompose this coding task into subtasks.\n"
    "Task: Write a Python function that takes a list of integers and returns "
    "the second largest unique value.\n\n"
    'Output ONLY valid JSON matching this schema:\n'
    '{"subtasks": [{"name": "...", "description": "...", "depends_on": ["..."]}]}'
)

N_TRIALS = 5


def main() -> None:
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    model_id = "Qwen/Qwen3.5-9B"

    if torch.cuda.is_available():
        device = "cuda"
        from transformers import BitsAndBytesConfig

        print(f"Loading {model_id} in NF4 (CUDA)...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        print(f"Loading {model_id} on MPS (Apple Silicon)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float32,
        )
    else:
        device = "cpu"
        print(f"Loading {model_id} on CPU...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    print(f"Model loaded on {device}")

    import xgrammar as xgr

    config = AutoConfig.from_pretrained(model_id)
    tokenizer_info = xgr.TokenizerInfo.from_huggingface(
        tokenizer, vocab_size=config.vocab_size
    )
    compiler = xgr.GrammarCompiler(tokenizer_info)
    compiled_grammar = compiler.compile_json_schema(DecomposeResult)
    processor = xgr.contrib.hf.LogitsProcessor(compiled_grammar)
    print(
        "XGrammar compiled for schema:\n"
        f"{json.dumps(DecomposeResult.model_json_schema(), indent=2)}"
    )

    messages = [
        {"role": "system", "content": "You are a coding assistant."},
        {"role": "user", "content": PROMPT},
    ]
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    successes = 0
    for i in range(N_TRIALS):
        print(f"\n--- Trial {i + 1}/{N_TRIALS} ---")
        inputs = tokenizer(formatted, return_tensors="pt").to(device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.8,
                top_k=20,
                logits_processor=[processor],
            )

        new_tokens = output_ids[0, inputs["input_ids"].shape[1] :]
        raw_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        print(f"Raw output: {raw_text[:500]}")

        try:
            result = DecomposeResult.model_validate_json(raw_text)
            print(f"Parsed OK: {len(result.subtasks)} subtasks")
            for st in result.subtasks:
                print(f"  - {st.name}: {st.description[:80]}")
            successes += 1
        except Exception as e:
            print(f"PARSE FAILED: {e}")

    print(f"\n{'=' * 40}")
    print(f"Results: {successes}/{N_TRIALS} passed")

    if successes == N_TRIALS:
        print(f"PASS — XGrammar + Qwen3.5-9B verified on {device}.")
        sys.exit(0)
    else:
        print(f"FAIL — XGrammar incompatibility on {device}.")
        sys.exit(1)


if __name__ == "__main__":
    main()
