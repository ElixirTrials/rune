import os
from pathlib import Path

import pytest

from rune.config import (
    DEFAULT_MODEL_ID,
    PipelineConfig,
    load_config,
    load_rune_config,
)


class TestPipelineConfig:
    def test_defaults(self) -> None:
        cfg = PipelineConfig()
        assert cfg.model_id == DEFAULT_MODEL_ID == "Qwen/Qwen3-4B-Instruct-2507"
        assert cfg.adapter_scaling == 1.0
        assert cfg.temperature == 0.3
        assert cfg.max_tokens == 2048
        # model generation profile: non-thinking instruct defaults (#52 fix)
        assert cfg.thinking_budget == 0  # 0 = non-thinking path
        assert cfg.presence_penalty == 0.0  # flat presence penalty harms codegen
        assert cfg.dtype == "bfloat16"
        assert cfg.attn_implementation == "flash_attention_2"
        assert cfg.max_phase_iterations == 16
        assert cfg.advisory_requirement_kinds == ("constraint_scale",)
        assert cfg.constraint_scale_pass_quality is True
        assert cfg.ship_best_on_exhaustion is True
        assert cfg.merge_spec_public_checks is True
        assert cfg.complexity_probe_max_n == 1200
        assert cfg.complexity_probe_n_repeats == 3
        assert cfg.complexity_empirical_timeout_s == 15.0
        assert cfg.complexity_judge_enabled is True
        assert cfg.complexity_judge_max_tokens == 384

    def test_frozen(self) -> None:
        cfg = PipelineConfig()
        with pytest.raises(AttributeError):
            cfg.temperature = 0.5  # type: ignore[misc]

    def test_to_dict_roundtrip(self) -> None:
        cfg = PipelineConfig(temperature=0.5, adapter_scaling=0.1)
        d = cfg.to_dict()
        assert d["temperature"] == 0.5
        assert d["adapter_scaling"] == 0.1

    def test_override(self) -> None:
        cfg = PipelineConfig()
        new = cfg.override(temperature=0.8, max_tokens=4096)
        assert new.temperature == 0.8
        assert new.max_tokens == 4096
        assert cfg.temperature == 0.3  # original unchanged

    def test_save_and_load(self, tmp_path: Path) -> None:
        cfg = PipelineConfig(temperature=0.42)
        path = cfg.save(tmp_path / "config.json")
        loaded = load_config(path)
        assert loaded.temperature == 0.42

    def test_load_empty_file_returns_defaults(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.yaml"
        path.write_text("   \n")
        loaded = load_config(path)  # must not crash on PipelineConfig(**None)
        assert loaded.temperature == PipelineConfig().temperature

    def test_load_non_mapping_yaml_raises_valueerror(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.yaml"
        path.write_text("- just\n- a\n- list\n")
        with pytest.raises(ValueError, match="YAML mapping"):
            load_config(path)

    def test_load_normalizes_advisory_kinds_list(self, tmp_path: Path) -> None:
        path = tmp_path / "config.yaml"
        path.write_text(
            "advisory_requirement_kinds:\n  - constraint_scale\n  - custom_probe\n"
        )
        cfg = load_config(path)
        assert cfg.advisory_requirement_kinds == ("constraint_scale", "custom_probe")

    def test_load_ignores_training_section(self, tmp_path: Path) -> None:
        # One config.yaml holds both surfaces; PipelineConfig ignores `training:`.
        path = tmp_path / "config.yaml"
        path.write_text(
            'model_id: "Org/M"\ntemperature: 0.5\ntraining:\n  learning_rate: 1.0e-3\n'
        )
        cfg = load_config(path)  # must not crash on the unknown `training` key
        assert cfg.model_id == "Org/M"
        assert cfg.temperature == 0.5

    def test_env_var_override(self, monkeypatch: object) -> None:
        os.environ["RUNE_TEMPERATURE"] = "0.99"
        try:
            cfg = PipelineConfig.from_env()
            assert cfg.temperature == 0.99
        finally:
            del os.environ["RUNE_TEMPERATURE"]

    def test_base_model_env_override(self) -> None:
        os.environ["RUNE_BASE_MODEL"] = "Org/Custom-Model"
        try:
            assert PipelineConfig.from_env().model_id == "Org/Custom-Model"
        finally:
            del os.environ["RUNE_BASE_MODEL"]


class TestLoadRuneConfig:
    def test_reads_yaml_file(self, tmp_path: Path) -> None:

        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text('model_id: "Org/From-File"\ntemperature: 0.7\n')
        os.environ["RUNE_CONFIG"] = str(cfg_file)
        try:
            cfg = load_rune_config()
            assert cfg.model_id == "Org/From-File"
            assert cfg.temperature == 0.7
        finally:
            del os.environ["RUNE_CONFIG"]

    def test_env_overrides_file(self, tmp_path: Path) -> None:

        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text('model_id: "Org/From-File"\n')
        os.environ["RUNE_CONFIG"] = str(cfg_file)
        os.environ["RUNE_BASE_MODEL"] = "Org/From-Env"
        try:
            # env wins over the file
            assert load_rune_config().model_id == "Org/From-Env"
        finally:
            del os.environ["RUNE_CONFIG"]
            del os.environ["RUNE_BASE_MODEL"]

    def test_env_applies_with_explicit_path(self, tmp_path: Path) -> None:
        # Env override must win even when an explicit path is given (matches the
        # CLI's --config path; no silent no-op).
        cfg_file = tmp_path / "experiment.yaml"
        cfg_file.write_text('model_id: "Org/From-File"\n')
        os.environ["RUNE_BASE_MODEL"] = "Org/From-Env"
        try:
            assert load_rune_config(cfg_file).model_id == "Org/From-Env"
        finally:
            del os.environ["RUNE_BASE_MODEL"]
