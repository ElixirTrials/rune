"""Per-layer Frobenius norm of adapter delta-W.

Confirms non-trivial weight changes across trajectory depths.

Usage:
    uv run python scripts/paper/frobenius_norm.py --adapter path/to/adapter.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def compute_frobenius_norms(state_dict: dict[str, torch.Tensor]) -> dict[str, float]:
    """Compute ||ΔW||_F for each layer in an adapter state dict.

    Args:
        state_dict: Mapping of layer_name -> weight tensor.

    Returns:
        Dict mapping layer_name -> Frobenius norm (float).
    """
    return {name: tensor.float().norm().item() for name, tensor in state_dict.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Frobenius norm sentinel")
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    state = torch.load(args.adapter, map_location="cpu", weights_only=True)
    norms = compute_frobenius_norms(state)

    result = {"norms": norms, "all_nonzero": all(v > 0 for v in norms.values())}
    output = json.dumps(result, indent=2)
    print(output)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output)


if __name__ == "__main__":
    main()
