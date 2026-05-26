"""Model inference: freeform and structured (outlines) generation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """Output from a single model generation call.

    Attributes:
        text: Final decoded text (JSON string for structured outputs).
        thinking: Chain-of-thought text produced before the answer, if any.
        tokens_used: Approximate token count for the generation.
    """

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
    """Dispatch to structured or freeform generation based on output_schema.

    Args:
        model: PEFT-wrapped language model.
        tokenizer: Paired tokenizer.
        prompt: User prompt text.
        system_prompt: Optional system role text.
        output_schema: Pydantic model for JSON-constrained output; None for freeform.
        max_tokens: Maximum new tokens to generate.
        temperature: Sampling temperature (freeform only).
        thinking_budget: Max tokens for chain-of-thought (structured only).

    Returns:
        GenerationResult with text, thinking, and token count.
    """
    if output_schema is not None:
        return await _generate_structured(
            model,
            tokenizer,
            prompt,
            system_prompt=system_prompt,
            schema=output_schema,
            max_tokens=max_tokens,
            thinking_budget=thinking_budget,
        )
    return await _generate_freeform(
        model,
        tokenizer,
        prompt,
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
    """Run greedy/sampled generation without output constraints.

    Args:
        model: Language model.
        tokenizer: Paired tokenizer.
        prompt: User prompt.
        system_prompt: System role text.
        max_tokens: Maximum new tokens.
        temperature: Sampling temperature.

    Returns:
        GenerationResult with decoded text and empty thinking field.
    """
    import asyncio  # noqa: PLC0415

    def _run() -> GenerationResult:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        encoded = tokenizer.apply_chat_template(messages, return_tensors="pt")
        if hasattr(encoded, "input_ids"):
            input_ids = encoded["input_ids"].to(model.device)
        else:
            input_ids = encoded.to(model.device)
        import torch  # noqa: PLC0415

        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            output = model.generate(
                input_ids,
                attention_mask=attention_mask,
                pad_token_id=tokenizer.eos_token_id,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
            )
        text = tokenizer.decode(
            output[0][input_ids.shape[1] :], skip_special_tokens=True
        )
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
    """Run thinking-then-structured generation using xgrammar JSON constraints.

    Args:
        model: Language model (may be PEFT-wrapped).
        tokenizer: Paired tokenizer.
        prompt: User prompt.
        system_prompt: System role text.
        schema: Pydantic model class used as the JSON schema.
        max_tokens: Maximum tokens for the structured answer phase.
        thinking_budget: Maximum tokens for the chain-of-thought phase.

    Returns:
        GenerationResult with JSON text, thinking text, and token count.
    """
    import asyncio  # noqa: PLC0415

    def _run() -> GenerationResult:
        import torch  # noqa: PLC0415
        import xgrammar as xgr  # noqa: PLC0415

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        encoded = tokenizer.apply_chat_template(messages, return_tensors="pt")
        if hasattr(encoded, "input_ids"):
            input_ids = encoded["input_ids"].to(model.device)
        else:
            input_ids = encoded.to(model.device)

        # Phase 1: thinking (unconstrained until </think>)
        think_token_id = tokenizer.encode("</think>", add_special_tokens=False)
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            thinking_output = model.generate(
                input_ids,
                attention_mask=attention_mask,
                pad_token_id=tokenizer.eos_token_id,
                max_new_tokens=thinking_budget,
                eos_token_id=think_token_id,
                do_sample=False,
            )
        new_tokens = thinking_output[0][input_ids.shape[1] :]
        thinking_text = tokenizer.decode(new_tokens, skip_special_tokens=False)

        # Phase 2: JSON-constrained generation with xgrammar
        base_model = getattr(model, "base_model", model)
        model_config = getattr(base_model, "config", None)
        text_cfg = getattr(model_config, "text_config", model_config)
        vocab_size = getattr(text_cfg, "vocab_size", None) or tokenizer.vocab_size

        tokenizer_info = xgr.TokenizerInfo.from_huggingface(
            tokenizer, vocab_size=vocab_size
        )
        compiler = xgr.GrammarCompiler(tokenizer_info)
        compiled = compiler.compile_json_schema(schema)
        logits_processor = xgr.contrib.hf.LogitsProcessor(compiled)

        # Build prefix: original prompt + thinking output + closing tag
        if not thinking_text.rstrip().endswith("</think>"):
            suffix_ids = tokenizer.encode(
                "</think>\n", add_special_tokens=False, return_tensors="pt"
            ).to(model.device)
            prefix_ids = torch.cat([thinking_output, suffix_ids], dim=-1)
        else:
            prefix_ids = thinking_output

        prefix_mask = torch.ones_like(prefix_ids)
        with torch.no_grad():
            structured_output = model.generate(
                prefix_ids,
                attention_mask=prefix_mask,
                pad_token_id=tokenizer.eos_token_id,
                max_new_tokens=max_tokens,
                do_sample=False,
                logits_processor=[logits_processor],
            )
        json_tokens = structured_output[0][prefix_ids.shape[1] :]
        result_json = tokenizer.decode(json_tokens, skip_special_tokens=True)
        total_tokens = thinking_output.shape[1] + len(json_tokens)
        return GenerationResult(
            text=result_json,
            thinking=thinking_text,
            tokens_used=total_tokens,
        )

    return await asyncio.to_thread(_run)
