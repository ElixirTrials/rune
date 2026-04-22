"""Round-trip test: save encoder checkpoint -> load via SentenceTransformer -> encode.

This test module contains:
- test_checkpoint_layout_without_training: CPU-only, mock-based, always runs.
- test_encoder_roundtrip_shape: requires torch + sentence_transformers, skipped
  if not installed.
- test_encoder_loadable_via_load_default_encoder: same optional deps.

Model weights are loaded from local HF cache — no network required when cached.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest


def _write_augmented_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )


def _make_augmented_rows(n: int = 8) -> list[dict[str, Any]]:
    """Produce n synthetic augmented pair rows for a minimal training run."""
    rows = []
    for i in range(n):
        rows.append(
            {
                "task_id": f"pr_{i:03d}",
                "pre_code": f"def compute_{i}(x): pass",
                "post_code": f"def compute_{i}(x): return x * {i}",
                "task_desc": f"Implement compute_{i} that multiplies by {i}",
                "task_desc_source": "explicit_field",
                "encoder_input": (
                    f"Implement compute_{i} that multiplies by {i}\n\n"
                    f"def compute_{i}(x): pass"
                ),
                "metadata": {"source_task_id": f"pr_{i:03d}", "step_index": 0},
            }
        )
    return rows


def test_checkpoint_layout_without_training(tmp_path: Path) -> None:
    """_save_sentence_transformer_checkpoint writes correct directory layout.

    CPU-only, mock-based — no torch download or GPU required.
    """
    from model_training.encoder_pretrain.train_encoder import (
        _save_sentence_transformer_checkpoint,
    )

    class _MockModel:
        def save_pretrained(self, path: str) -> None:
            p = Path(path)
            p.mkdir(parents=True, exist_ok=True)
            cfg = {"model_type": "bert", "hidden_size": 768}
            (p / "config.json").write_text(json.dumps(cfg))

    class _MockTokenizer:
        def save_pretrained(self, path: str) -> None:
            p = Path(path)
            p.mkdir(parents=True, exist_ok=True)
            tok_cfg = {"tokenizer_class": "BertTokenizer"}
            (p / "tokenizer_config.json").write_text(json.dumps(tok_cfg))

    out_dir = tmp_path / "ckpt"
    _save_sentence_transformer_checkpoint(
        model=_MockModel(),
        tokenizer=_MockTokenizer(),
        output_dir=out_dir,
        max_length=512,
    )

    assert (out_dir / "modules.json").exists()
    assert (out_dir / "sentence_transformers_config.json").exists()
    assert (out_dir / "1_Pooling" / "config.json").exists()

    modules = json.loads((out_dir / "modules.json").read_text())
    assert len(modules) == 2
    assert modules[0]["type"] == "sentence_transformers.models.Transformer"
    assert modules[1]["type"] == "sentence_transformers.models.Pooling"

    pooling_cfg = json.loads((out_dir / "1_Pooling" / "config.json").read_text())
    assert pooling_cfg["pooling_mode_mean_tokens"] is True
    assert pooling_cfg["word_embedding_dimension"] == 768


@pytest.mark.slow
def test_encoder_roundtrip_shape(tmp_path: Path) -> None:
    """Train 1 epoch, save, load via SentenceTransformer, assert shape (2, 768)."""
    # Skip when optional heavy deps are not installed (CI without GPU extras)
    pytest.importorskip("torch")
    pytest.importorskip("sentence_transformers")

    import numpy as np  # noqa: PLC0415, I001
    from sentence_transformers import SentenceTransformer  # noqa: PLC0415
    from model_training.encoder_pretrain.train_encoder import (  # noqa: PLC0415
        EncoderTrainingConfig,
        run_training,
    )

    aug_dir = tmp_path / "pairs_augmented"
    aug_dir.mkdir()
    out_dir = tmp_path / "encoder_checkpoint"

    rows = _make_augmented_rows(n=8)
    _write_augmented_jsonl(aug_dir / "test_repo.jsonl", rows)

    config = EncoderTrainingConfig(
        augmented_dir=aug_dir,
        output_dir=out_dir,
        epochs=1,
        batch_size=4,
        max_length=64,
        warmup_steps=0,
        test_fraction=0.25,
        mlflow_experiment="rune-encoder-pretrain-test",
    )

    saved_path = run_training(config)

    assert saved_path == out_dir
    assert out_dir.exists()
    assert (out_dir / "modules.json").exists()
    assert (out_dir / "1_Pooling" / "config.json").exists()

    encoder = SentenceTransformer(str(out_dir))
    texts = ["Fix the off-by-one error", "Add type annotations to function"]
    embeddings = encoder.encode(texts, convert_to_tensor=False)

    assert embeddings.shape == (2, 768), (
        f"Expected shape (2, 768), got {embeddings.shape}. "
        "The checkpoint must produce 768-d embeddings to match the builder contract."
    )
    assert not np.isnan(embeddings).any()
    assert not (embeddings == 0).all()


@pytest.mark.slow
def test_encoder_loadable_via_load_default_encoder(tmp_path: Path) -> None:
    """The saved checkpoint is loadable by task_embeddings.load_default_encoder()."""
    pytest.importorskip("torch")
    pytest.importorskip("sentence_transformers")

    from model_training.encoder_pretrain.train_encoder import (  # noqa: PLC0415
        EncoderTrainingConfig,
        run_training,
    )
    from model_training.reconstruction.task_embeddings import (  # noqa: PLC0415
        compute_task_embeddings,
        load_default_encoder,
    )

    aug_dir = tmp_path / "pairs_augmented"
    aug_dir.mkdir()
    out_dir = tmp_path / "encoder_checkpoint_2"

    rows = _make_augmented_rows(n=8)
    _write_augmented_jsonl(aug_dir / "test_repo.jsonl", rows)

    config = EncoderTrainingConfig(
        augmented_dir=aug_dir,
        output_dir=out_dir,
        epochs=1,
        batch_size=4,
        max_length=64,
        warmup_steps=0,
        test_fraction=0.25,
    )
    run_training(config)

    encoder = load_default_encoder(model_id=str(out_dir), device="cpu")

    descriptions = {
        "task_a": "Implement binary search",
        "task_b": "Write a function to parse JSON",
    }
    embeddings = compute_task_embeddings(descriptions, model=encoder)

    assert set(embeddings) == {"task_a", "task_b"}
    for key, emb in embeddings.items():
        assert emb.shape == (1, 768), (
            f"{key}: expected shape (1, 768), got {emb.shape}"
        )
