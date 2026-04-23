"""Retrieval eval and downstream cluster probe for encoder pretraining.

Retrieval eval: given the held-out test split, embed each anchor
(encoder_input) and each positive (post_code), then rank positives by
cosine similarity to each anchor. Reports MRR@10 and Recall@1.

Cluster probe: embed a sample of HumanEval + MBPP problem descriptions
(loaded via HuggingFace ``datasets``), compute intra-cluster vs inter-cluster
mean cosine similarity as a simple cohesion score. Used as a sanity check
that the encoder groups semantically similar problems; does NOT require the
full benchmark harness.

All GPU-dependent imports are deferred (INFRA-05).
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _embed_texts(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    max_length: int,
    batch_size: int,
    device: str,
) -> Any:
    """Embed a list of texts using mean pooling.

    Args:
        model: HuggingFace AutoModel (encoder-only); already on ``device``.
        tokenizer: Matching HuggingFace tokenizer.
        texts: Texts to embed.
        max_length: Truncation length.
        batch_size: Inference batch size.
        device: Device string (``"cpu"`` or ``"cuda:0"``).

    Returns:
        L2-normalized embeddings ``(N, hidden_dim)`` on CPU as a torch.Tensor.
    """
    import torch  # noqa: PLC0415
    import torch.nn.functional as F  # noqa: PLC0415, N812

    from model_training.encoder_pretrain.train_encoder import (  # noqa: PLC0415
        _encode_batch,
    )

    all_embs: list[Any] = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        enc = tokenizer(
            chunk,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        with torch.no_grad():
            emb = _encode_batch(model, input_ids, attention_mask)
        all_embs.append(emb.cpu())

    if not all_embs:
        return torch.empty(0, dtype=torch.float32)

    stacked = torch.cat(all_embs, dim=0)
    return F.normalize(stacked, dim=-1)


def run_retrieval_eval(
    *,
    model: Any,
    tokenizer: Any,
    test_rows: list[dict[str, Any]],
    max_length: int,
    batch_size: int,
    device: str,
) -> dict[str, float]:
    """Compute MRR@10 and Recall@1 on a held-out test split.

    For each anchor embedding, rank all positive embeddings by cosine
    similarity and check the rank of the true positive (diagonal entry).

    Args:
        model: HuggingFace AutoModel; already on ``device``.
        tokenizer: Matching HuggingFace tokenizer.
        test_rows: Augmented pair dicts (must have ``encoder_input`` and
            ``post_code`` keys).
        max_length: Truncation length.
        batch_size: Inference batch size.
        device: Device string.

    Returns:
        Dict with ``"mrr_at_10"`` and ``"recall_at_1"`` float metrics.
    """
    import torch  # noqa: PLC0415

    if not test_rows:
        return {"mrr_at_10": 0.0, "recall_at_1": 0.0}

    anchors_text = [r["encoder_input"] for r in test_rows]
    positives_text = [r["post_code"] for r in test_rows]

    model.eval()
    anchor_embs = _embed_texts(
        model, tokenizer, anchors_text, max_length, batch_size, device
    )
    positive_embs = _embed_texts(
        model, tokenizer, positives_text, max_length, batch_size, device
    )

    # (N, N) similarity matrix
    sim_matrix = torch.matmul(anchor_embs, positive_embs.T)
    n = sim_matrix.shape[0]

    mrr_sum = 0.0
    recall_at_1_sum = 0.0

    for i in range(n):
        scores = sim_matrix[i]  # (N,)
        # Rank is 1-indexed; higher score = rank 1
        rank = int((scores > scores[i]).sum().item()) + 1
        if rank <= 10:
            mrr_sum += 1.0 / rank
        if rank == 1:
            recall_at_1_sum += 1.0

    mrr_at_10 = mrr_sum / n
    recall_at_1 = recall_at_1_sum / n

    logger.info(
        "retrieval_eval: n=%d MRR@10=%.4f Recall@1=%.4f",
        n,
        mrr_at_10,
        recall_at_1,
    )
    return {"mrr_at_10": mrr_at_10, "recall_at_1": recall_at_1}


def run_cluster_probe(
    *,
    model: Any,
    tokenizer: Any,
    max_length: int,
    batch_size: int,
    device: str,
    n_humaneval: int = 50,
    n_mbpp: int = 50,
) -> dict[str, float]:
    """Cohesion probe: do HumanEval and MBPP cluster separately?

    Loads ``n_humaneval`` problems from HumanEval (via ``datasets``) and
    ``n_mbpp`` from MBPP, embeds their prompts, and computes mean intra-cluster
    cosine similarity vs mean inter-cluster cosine similarity. A positive
    ``cohesion_delta`` indicates the encoder groups similar problems together.

    Does NOT require the full benchmark harness — uses ``datasets`` directly.

    Args:
        model: HuggingFace AutoModel; already on ``device``.
        tokenizer: Matching HuggingFace tokenizer.
        max_length: Truncation length.
        batch_size: Inference batch size.
        device: Device string.
        n_humaneval: Number of HumanEval problems to sample.
        n_mbpp: Number of MBPP problems to sample.

    Returns:
        Dict with ``"intra_sim"``, ``"inter_sim"``, and ``"cohesion_delta"``
        float metrics.
    """
    import torch  # noqa: PLC0415
    from datasets import load_dataset  # noqa: PLC0415

    he_ds = load_dataset("openai_humaneval", split="test", trust_remote_code=True)
    mbpp_ds = load_dataset("mbpp", split="train", trust_remote_code=True)

    he_texts = [
        str(ex["prompt"]) for ex in he_ds.select(range(min(n_humaneval, len(he_ds))))
    ]
    mbpp_texts = [
        str(ex["text"]) for ex in mbpp_ds.select(range(min(n_mbpp, len(mbpp_ds))))
    ]

    model.eval()
    he_embs = _embed_texts(model, tokenizer, he_texts, max_length, batch_size, device)
    mbpp_embs = _embed_texts(
        model, tokenizer, mbpp_texts, max_length, batch_size, device
    )

    def _mean_off_diagonal(sim: Any) -> float:
        n = sim.shape[0]
        if n <= 1:
            return 0.0
        mask = 1 - torch.eye(n)
        return float((sim * mask).sum() / mask.sum())

    he_sim = torch.matmul(he_embs, he_embs.T)
    mbpp_sim = torch.matmul(mbpp_embs, mbpp_embs.T)
    inter_sim = torch.matmul(he_embs, mbpp_embs.T)

    intra = (_mean_off_diagonal(he_sim) + _mean_off_diagonal(mbpp_sim)) / 2.0
    inter = float(inter_sim.mean().item())
    delta = intra - inter

    logger.info(
        "cluster_probe: intra_sim=%.4f inter_sim=%.4f cohesion_delta=%.4f",
        intra,
        inter,
        delta,
    )
    return {"intra_sim": intra, "inter_sim": inter, "cohesion_delta": delta}
