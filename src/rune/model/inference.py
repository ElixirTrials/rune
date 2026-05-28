"""Model inference: structured (xgrammar) generation with thinking phase."""

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
    truncated: bool = False


async def generate(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    system_prompt: str = "",
    output_schema: type[Any] | None = None,
    max_tokens: int = 2048,
    temperature: float = 0.3,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    thinking_budget: int = 1024,
    no_repeat_ngram_size: int = 0,
    skip_completion_retry: bool = False,
) -> GenerationResult:
    import asyncio  # noqa: PLC0415

    def _sampling_kwargs() -> dict[str, Any]:
        if temperature > 0:
            return {"do_sample": True, "temperature": temperature, "top_p": top_p}
        return {"do_sample": False}

    def _run() -> GenerationResult:
        import torch  # noqa: PLC0415
        import xgrammar as xgr  # noqa: PLC0415

        sampling = _sampling_kwargs()

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        if thinking_budget > 0:
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
                    repetition_penalty=repetition_penalty,
                    **sampling,
                )
            new_tokens = thinking_output[0][input_ids.shape[1] :]
            thinking_text = tokenizer.decode(new_tokens, skip_special_tokens=False)

            if not thinking_text.rstrip().endswith("</think>"):
                suffix_ids = tokenizer.encode(
                    "</think>\n", add_special_tokens=False, return_tensors="pt"
                ).to(model.device)
                prefix_ids = torch.cat([thinking_output, suffix_ids], dim=-1)
            else:
                prefix_ids = thinking_output
        else:
            encoded = tokenizer.apply_chat_template(
                messages, return_tensors="pt",
                enable_thinking=False, add_generation_prompt=True,
            )
            if hasattr(encoded, "input_ids"):
                prefix_ids = encoded["input_ids"].to(model.device)
            else:
                prefix_ids = encoded.to(model.device)
            thinking_text = ""

        # Phase 2: generation
        gen_kwargs: dict[str, Any] = {
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "max_new_tokens": max_tokens,
            "repetition_penalty": repetition_penalty,
            **sampling,
        }

        compiled = None
        if output_schema is None:
            # Raw mode: continuation rounds use this so the model can EOS naturally.
            # no_repeat_ngram_size is only safe here — it conflicts with xgrammar.
            if no_repeat_ngram_size > 0:
                gen_kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size
        else:
            base_model = getattr(model, "base_model", model)
            model_config = getattr(base_model, "config", None)
            text_cfg = getattr(model_config, "text_config", model_config)
            vocab_size = (
                getattr(text_cfg, "vocab_size", None) or tokenizer.vocab_size
            )
            tokenizer_info = xgr.TokenizerInfo.from_huggingface(
                tokenizer, vocab_size=vocab_size
            )
            compiler = xgr.GrammarCompiler(tokenizer_info)
            compiled = compiler.compile_json_schema(
                output_schema,
                any_whitespace=False,
            )
            gen_kwargs["logits_processor"] = [
                xgr.contrib.hf.LogitsProcessor(compiled)
            ]

        prefix_mask = torch.ones_like(prefix_ids)
        with torch.no_grad():
            structured_output = model.generate(
                prefix_ids,
                attention_mask=prefix_mask,
                **gen_kwargs,
            )
        result_tokens = structured_output[0][prefix_ids.shape[1] :]
        result_text = tokenizer.decode(result_tokens, skip_special_tokens=True)
        total_tokens = prefix_ids.shape[1] + len(result_tokens)

        truncated = len(result_tokens) >= max_tokens
        if truncated and not skip_completion_retry and compiled is not None:
            result_text, extra, completed = _try_completion(
                model, tokenizer, result_text, structured_output,
                compiled, max_tokens, repetition_penalty, sampling,
            )
            total_tokens += extra
            truncated = not completed

        return GenerationResult(
            text=result_text,
            thinking=thinking_text,
            tokens_used=total_tokens,
            truncated=truncated,
        )

    return await asyncio.to_thread(_run)


def _try_completion(
    model: Any,
    tokenizer: Any,
    partial_json: str,
    prior_output: Any,
    compiled: Any,
    max_tokens: int,
    repetition_penalty: float,
    sampling: dict[str, Any],
) -> tuple[str, int, bool]:
    """Continue a truncated generation from the model's own output sequence.

    Uses the full prior_output tensor (prompt + thinking + partial JSON) so the
    model — with its adapter still loaded — continues from its own context
    rather than a cold re-prompt.  The xgrammar matcher is advanced past
    partial_json so the grammar constraint picks up where it left off.

    Returns (full_json, extra_tokens_used, completed).
    """
    import torch  # noqa: PLC0415
    import xgrammar as xgr  # noqa: PLC0415

    logger.warning(
        "JSON output truncated, attempting continuation (%d chars so far)",
        len(partial_json),
    )

    matcher = xgr.GrammarMatcher(compiled)
    if not matcher.accept_string(partial_json):
        logger.error("Cannot advance grammar for continuation — partial JSON invalid")
        return partial_json, 0, False

    adv_processor = xgr.contrib.hf.LogitsProcessor(compiled)
    adv_processor.matchers = [matcher]
    adv_processor.token_bitmask = xgr.allocate_token_bitmask(  # type: ignore[assignment]
        1, adv_processor.full_vocab_size
    )
    adv_processor.prefilled = False
    adv_processor.batch_size = 1

    cont_mask = torch.ones_like(prior_output)
    with torch.no_grad():
        cont_output = model.generate(
            prior_output,
            attention_mask=cont_mask,
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=max_tokens,
            repetition_penalty=repetition_penalty,
            logits_processor=[adv_processor],
            **sampling,
        )
    cont_tokens = cont_output[0][prior_output.shape[1] :]
    continuation = tokenizer.decode(cont_tokens, skip_special_tokens=True)
    completed = matcher.is_completed()
    logger.info(
        "Continuation produced %d extra tokens (completed=%s)",
        len(cont_tokens), completed,
    )
    return partial_json + continuation, len(cont_tokens), completed


async def generate_continuation(
    model: Any,
    tokenizer: Any,
    *,
    system_prompt: str,
    user_prompt: str,
    assistant_prefix: str,
    max_tokens: int = 2048,
    temperature: float = 0.3,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    no_repeat_ngram_size: int = 0,
) -> GenerationResult:
    import asyncio  # noqa: PLC0415

    def _run() -> GenerationResult:
        import torch  # noqa: PLC0415

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        messages.append({"role": "assistant", "content": assistant_prefix})

        template_ids = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            continue_final_message=True,
            enable_thinking=False,
        )
        if hasattr(template_ids, "input_ids"):
            template_ids = template_ids["input_ids"]
        template_ids = template_ids.to(model.device)

        sampling: dict[str, Any] = (
            {"do_sample": True, "temperature": temperature, "top_p": top_p}
            if temperature > 0
            else {"do_sample": False}
        )

        gen_kwargs: dict[str, Any] = {
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "max_new_tokens": max_tokens,
            "repetition_penalty": repetition_penalty,
            **sampling,
        }
        if no_repeat_ngram_size > 0:
            gen_kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size

        attention_mask = torch.ones_like(template_ids)
        with torch.no_grad():
            output = model.generate(
                template_ids,
                attention_mask=attention_mask,
                **gen_kwargs,
            )

        new_tokens = output[0][template_ids.shape[1] :]
        new_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        truncated = len(new_tokens) >= max_tokens

        return GenerationResult(
            text=new_text,
            thinking="",
            tokens_used=template_ids.shape[1] + len(new_tokens),
            truncated=truncated,
        )

    return await asyncio.to_thread(_run)
