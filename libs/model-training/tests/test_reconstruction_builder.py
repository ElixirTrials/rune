"""Integration tests for build_reconstruction_dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import pytest
from adapter_registry.models import AdapterRecord
from adapter_registry.registry import AdapterRegistry
from sqlalchemy import create_engine

torch = pytest.importorskip("torch")


def _write_fake_adapter(
    adapter_dir: Path,
    *,
    base_model_name_or_path: str,
    rank: int,
    target_modules: list[str],
    layer_indices: list[int],
    in_features: int = 16,
    out_features: int = 16,
) -> None:
    from safetensors.torch import save_file

    sd: dict[str, torch.Tensor] = {}
    for mod in target_modules:
        attn_mods = {"q_proj", "k_proj", "v_proj", "o_proj"}
        prefix = "self_attn" if mod in attn_mods else "mlp"
        for layer in layer_indices:
            sd[
                f"base_model.model.model.layers.{layer}.{prefix}.{mod}.lora_A.weight"
            ] = torch.randn(rank, in_features)
            sd[
                f"base_model.model.model.layers.{layer}.{prefix}.{mod}.lora_B.weight"
            ] = torch.randn(out_features, rank)
    adapter_dir.mkdir(parents=True)
    save_file(sd, str(adapter_dir / "adapter_model.safetensors"))
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": base_model_name_or_path,
                "r": rank,
                "target_modules": target_modules,
                "task_type": "CAUSAL_LM",
            }
        )
    )


@pytest.fixture
def populated_registry(
    tmp_path: Path, make_adapter_record: Callable[..., AdapterRecord]
) -> tuple[AdapterRegistry, Path]:
    engine = create_engine(f"sqlite:///{tmp_path / 'reg.db'}")
    registry = AdapterRegistry(engine=engine)
    for i in range(2):
        adapter_dir = tmp_path / f"adapter-{i}"
        _write_fake_adapter(
            adapter_dir,
            base_model_name_or_path="danielcherubini/Qwen3.5-DeltaCoder-9B",
            rank=4,
            target_modules=["q_proj", "v_proj"],
            layer_indices=[0, 1],
        )
        registry.store(
            make_adapter_record(
                id=f"adapter-{i}",
                rank=4,
                file_path=str(adapter_dir),
                base_model_id="Qwen/Qwen3.5-9B",
                fitness_score=0.5 + 0.1 * i,
            )
        )
    return registry, tmp_path


def test_builds_manifest_plus_embeddings_plus_stats(
    populated_registry: tuple[AdapterRegistry, Path],
) -> None:
    from model_training.reconstruction.builder import build_reconstruction_dataset
    from model_training.reconstruction.manifest import ReconstructionManifest

    registry, base_dir = populated_registry
    out_dir = base_dir / "recon_ds"

    def describe(rec: AdapterRecord) -> str:
        return f"task-description-for-{rec.id}"

    build_reconstruction_dataset(
        registry=registry,
        out_dir=out_dir,
        task_description_fn=describe,
        warm_start_adapter="danielcherubini/Qwen3.5-DeltaCoder-9B",
        base_model_id_override="Qwen/Qwen3.5-9B",
        emb_model=None,  # one-hot fallback
        compute_zscore=True,
    )

    manifest_path = out_dir / "manifest.json"
    embeddings_path = out_dir / "task_embeddings.pt"
    stats_path = out_dir / "zscore_stats.pt"
    assert manifest_path.is_file()
    assert embeddings_path.is_file()
    assert stats_path.is_file()

    manifest = ReconstructionManifest.load(manifest_path)
    assert manifest.base_model_id == "Qwen/Qwen3.5-9B"
    assert manifest.warm_start_adapter == "danielcherubini/Qwen3.5-DeltaCoder-9B"
    assert manifest.rank == 4
    assert manifest.target_modules == ("q_proj", "v_proj")
    assert manifest.layer_indices == (0, 1)
    assert manifest.task_embedding_model is None
    assert manifest.task_embedding_dim == 2  # one-hot over 2 tasks
    assert {r.task_id for r in manifest.records} == {"adapter-0", "adapter-1"}
    assert manifest.zscore_stats_path == str(stats_path.resolve())


def test_skips_zscore_when_disabled(
    populated_registry: tuple[AdapterRegistry, Path],
) -> None:
    from model_training.reconstruction.builder import build_reconstruction_dataset
    from model_training.reconstruction.manifest import ReconstructionManifest

    registry, base_dir = populated_registry
    out_dir = base_dir / "recon_ds_no_stats"

    build_reconstruction_dataset(
        registry=registry,
        out_dir=out_dir,
        task_description_fn=lambda rec: rec.id,
        warm_start_adapter=None,
        base_model_id_override="Qwen/Qwen3.5-9B",
        emb_model=None,
        compute_zscore=False,
    )

    manifest = ReconstructionManifest.load(out_dir / "manifest.json")
    assert manifest.zscore_stats_path is None
    assert not (out_dir / "zscore_stats.pt").exists()


def test_raises_on_heterogeneous_ranks(
    tmp_path: Path, make_adapter_record: Callable[..., AdapterRecord]
) -> None:
    from model_training.reconstruction.builder import build_reconstruction_dataset

    engine = create_engine(f"sqlite:///{tmp_path / 'reg.db'}")
    registry = AdapterRegistry(engine=engine)
    for i, rank in enumerate((4, 8)):
        adapter_dir = tmp_path / f"adapter-{i}"
        _write_fake_adapter(
            adapter_dir,
            base_model_name_or_path="danielcherubini/Qwen3.5-DeltaCoder-9B",
            rank=rank,
            target_modules=["q_proj"],
            layer_indices=[0, 1],
        )
        registry.store(
            make_adapter_record(
                id=f"adapter-{i}", rank=rank, file_path=str(adapter_dir)
            )
        )
    with pytest.raises(ValueError, match="rank"):
        build_reconstruction_dataset(
            registry=registry,
            out_dir=tmp_path / "out",
            task_description_fn=lambda rec: rec.id,
            warm_start_adapter=None,
            base_model_id_override="Qwen/Qwen3.5-9B",
            emb_model=None,
            compute_zscore=False,
        )
    # out_dir must not exist — we fail before emitting.
    assert not (tmp_path / "out").exists()


def test_raises_on_no_candidates(
    tmp_path: Path,
) -> None:
    from model_training.reconstruction.builder import build_reconstruction_dataset

    engine = create_engine(f"sqlite:///{tmp_path / 'reg.db'}")
    registry = AdapterRegistry(engine=engine)
    with pytest.raises(ValueError, match="no candidates"):
        build_reconstruction_dataset(
            registry=registry,
            out_dir=tmp_path / "out",
            task_description_fn=lambda rec: rec.id,
            warm_start_adapter=None,
            base_model_id_override="Qwen/Qwen3.5-9B",
            emb_model=None,
            compute_zscore=False,
        )


def test_builder_module_is_cpu_importable() -> None:
    import importlib

    mod = importlib.import_module("model_training.reconstruction.builder")
    assert hasattr(mod, "build_reconstruction_dataset")
