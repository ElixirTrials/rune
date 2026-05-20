#!/usr/bin/env python3
"""Verify XGrammar constrained decoding works with NF4-quantized Qwen3.5-9B.

Run on a GPU machine (L4 / A10G / A100):
    pip install xgrammar
    python scripts/verify_xgrammar_nf4.py

Exits 0 if all 5 generations parse as valid JSON matching the schema.
Exits 1 on any failure — logs the error and raw output for debugging.
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
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    model_id = "Qwen/Qwen3.5-9B"

    print(f"Loading {model_id} in NF4...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    print(f"Model loaded. Device map: {model.hf_device_map}")

    try:
        import xgrammar as xgr
    except ImportError:
        print("ERROR: xgrammar not installed. pip install xgrammar")
        sys.exit(1)

    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_id)
    tokenizer_info = xgr.TokenizerInfo.from_huggingface(
        tokenizer, vocab_size=config.vocab_size
    )
    compiler = xgr.GrammarCompiler(tokenizer_info)
    compiled_grammar = compiler.compile_json_schema(DecomposeResult)
    processor = xgr.contrib.hf.LogitsProcessor(compiled_grammar)
    print(f"XGrammar compiled for schema: {json.dumps(DecomposeResult.model_json_schema(), indent=2)}")

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
        inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

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
        print("PASS — XGrammar + NF4 + Qwen3.5-9B verified.")
        sys.exit(0)
    else:
        print("FAIL — XGrammar + NF4 incompatibility detected.")
        sys.exit(1)


if __name__ == "__main__":
    main()
