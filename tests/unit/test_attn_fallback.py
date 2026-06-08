"""flash-attn -> sdpa graceful fallback in model loading (CPU-only)."""

from __future__ import annotations

import importlib.util

from rune.model.wrapper import resolve_attn_implementation


def test_flash_attn_falls_back_to_sdpa_when_absent(monkeypatch) -> None:
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    assert resolve_attn_implementation("flash_attention_2") == "sdpa"


def test_flash_attn_kept_when_installed(monkeypatch) -> None:
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    assert resolve_attn_implementation("flash_attention_2") == "flash_attention_2"


def test_other_impls_pass_through(monkeypatch) -> None:
    # sdpa/eager must never be rewritten regardless of flash_attn availability.
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    assert resolve_attn_implementation("sdpa") == "sdpa"
    assert resolve_attn_implementation("eager") == "eager"
