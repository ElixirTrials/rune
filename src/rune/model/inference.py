from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    text: str
    thinking: str
    tokens_used: int


async def generate(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    system_prompt: str = "",
    output_schema: type[Any] | None = None,
    max_tokens: int = 2048,
    temperature: float = 0.3,
    thinking_budget: int = 1024,
) -> GenerationResult:
    import torch  # noqa: PLC0415

    if output_schema is not None:
        return await _generate_structured(
            model, tokenizer, prompt,
            system_prompt=system_prompt,
            schema=output_schema,
            max_tokens=max_tokens,
            thinking_budget=thinking_budget,
        )
    return await _generate_freeform(
        model, tokenizer, prompt,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
    )


async def _generate_freeform(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    system_prompt: str,
    max_tokens: int,
    temperature: float,
) -> GenerationResult:
    import asyncio  # noqa: PLC0415

    def _run() -> GenerationResult:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)
        import torch  # noqa: PLC0415
        with torch.no_grad():
            output = model.generate(input_ids, max_new_tokens=max_tokens, temperature=temperature, do_sample=True)
        text = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
        return GenerationResult(text=text, thinking="", tokens_used=output.shape[1])

    return await asyncio.to_thread(_run)


async def _generate_structured(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    system_prompt: str,
    schema: type[Any],
    max_tokens: int,
    thinking_budget: int,
) -> GenerationResult:
    import asyncio  # noqa: PLC0415

    def _run() -> GenerationResult:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)

        think_token_id = tokenizer.encode("</think>", add_special_tokens=False)
        import torch  # noqa: PLC0415
        with torch.no_grad():
            thinking_output = model.generate(
                input_ids,
                max_new_tokens=thinking_budget,
                eos_token_id=think_token_id,
                do_sample=False,
            )
        thinking_text = tokenizer.decode(thinking_output[0][input_ids.shape[1]:], skip_special_tokens=False)

        import outlines  # noqa: PLC0415
        generator = outlines.generate.json(model, schema)
        full_prefix = prompt + thinking_text
        if not full_prefix.endswith("</think>\n"):
            full_prefix += "</think>\n"
        structured_text = generator(full_prefix)
        result_json = structured_text if isinstance(structured_text, str) else structured_text.model_dump_json()
        return GenerationResult(text=result_json, thinking=thinking_text, tokens_used=len(thinking_text.split()))

    return await asyncio.to_thread(_run)
