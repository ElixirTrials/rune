"""Inter-adapter cosine diversity metric per paper Eq. 3.

Diversity = 1 - mean(cos_sim(flatten(A_i), flatten(A_j))) for all i<j.

Usage:
    uv run python scripts/paper/cosine_diversity.py \
        --adapter-dir checkpoints/adapters/ \
        --output evaluation_results/diversity.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as f_nn


def compute_cosine_diversity(adapters: list[torch.Tensor]) -> float:
    """Compute 1 - mean pairwise cosine similarity across flattened adapters.

    Args:
        adapters: List of adapter weight tensors (any shape, will be flattened).

    Returns:
        Diversity score in [0, 1]. 0 = all identical, 1 = all orthogonal.
    """
    if len(adapters) < 2:
        return 0.0

    flat = torch.stack([a.flatten().float() for a in adapters])
    flat = f_nn.normalize(flat, dim=1)
    sim_matrix = flat @ flat.T

    n = len(adapters)
    mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
    mean_sim = sim_matrix[mask].mean().item()
    return 1.0 - mean_sim


def main() -> None:
    parser = argparse.ArgumentParser(description="Cosine diversity (Eq. 3)")
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    adapters: list[torch.Tensor] = []
    for pt_file in sorted(args.adapter_dir.glob("*.pt")):
        state = torch.load(pt_file, map_location="cpu", weights_only=True)
        combined = torch.cat([v.flatten() for v in state.values()])
        adapters.append(combined)

    diversity = compute_cosine_diversity(adapters)
    result = {"diversity": diversity, "n_adapters": len(adapters)}

    output = json.dumps(result, indent=2)
    print(output)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output)


if __name__ == "__main__":
    main()
