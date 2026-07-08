"""C4 I5 — composition and bundling primitives for the capacity curve.

Rank-stacking is exact: [B1 B2] @ [A1; A2] = B1@A1 + B2@A2, so composing K
per-fact adapters as concatenated rank slices reproduces the sum of their LoRA
deltas; zero-padding extra rank slices contributes exactly zero. PEFT layout
(hypernetwork._to_peft_state_dict): lora_A.weight is (r, in), lora_B.weight is
(out, r); with use_bias, ranks 0..ctx-1 are context and ctx.. are the
conditioning-independent head bias (wrapper.py:40-50) — so composition keeps
ONE bias slice (from the first adapter) or the bias would be applied K times.
"""

from __future__ import annotations

from typing import Any


def make_bundles(n_rows: int, k: int) -> list[list[int]]:
    """Disjoint consecutive index bundles of size k; remainder rows dropped."""
    if k < 1:
        raise ValueError("k must be >= 1")
    return [list(range(i * k, (i + 1) * k)) for i in range(n_rows // k)]


def compose_rank_stacked(
    state_dicts: list[dict[str, Any]], ctx_rank: int
) -> dict[str, Any]:
    """Compose K adapters: concat context rank slices; keep the first bias."""
    import torch  # noqa: PLC0415

    if not state_dicts:
        raise ValueError("no adapters to compose")
    keys = state_dicts[0].keys()
    if any(sd.keys() != keys for sd in state_dicts[1:]):
        raise ValueError("adapter key sets differ")
    out: dict[str, Any] = {}
    for key in keys:
        if "lora_A" in key:
            parts = [sd[key][:ctx_rank] for sd in state_dicts]
            parts.append(state_dicts[0][key][ctx_rank:])
            out[key] = torch.cat(parts, dim=0)
        elif "lora_B" in key:
            parts = [sd[key][:, :ctx_rank] for sd in state_dicts]
            parts.append(state_dicts[0][key][:, ctx_rank:])
            out[key] = torch.cat(parts, dim=1)
        else:
            out[key] = state_dicts[0][key]
    return out


def pad_adapter_rank(state_dict: dict[str, Any], target_rank: int) -> dict[str, Any]:
    """Zero-pad every lora_A/lora_B pair to target_rank (numerically inert)."""
    import torch  # noqa: PLC0415

    out: dict[str, Any] = {}
    for key, w in state_dict.items():
        if "lora_A" in key:
            r = w.shape[0]
            if r > target_rank:
                raise ValueError(f"{key}: rank {r} > target {target_rank}")
            pad = torch.zeros(
                target_rank - r, w.shape[1], dtype=w.dtype, device=w.device
            )
            out[key] = torch.cat([w, pad], dim=0)
        elif "lora_B" in key:
            r = w.shape[1]
            if r > target_rank:
                raise ValueError(f"{key}: rank {r} > target {target_rank}")
            pad = torch.zeros(
                w.shape[0], target_rank - r, dtype=w.dtype, device=w.device
            )
            out[key] = torch.cat([w, pad], dim=1)
        else:
            out[key] = w
    return out


def multi_cond_text(conds: list[str]) -> str:
    """Mode-(a) conditioning: the K per-row episodic blocks, joined."""
    return "\n\n".join(conds)


def campaign_rank(ctx_rank: int, bias_rank: int, k_max: int) -> int:
    """PEFT rank sized for K_max context slices plus one bias slice."""
    return k_max * ctx_rank + bias_rank
