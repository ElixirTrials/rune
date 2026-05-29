import os
from pathlib import Path

import pytest

from rune.config import PipelineConfig, load_config


class TestPipelineConfig:
    def test_defaults(self) -> None:
        cfg = PipelineConfig()
        assert cfg.model_id == "Qwen/Qwen3.5-9B"
        assert cfg.adapter_scaling == 1.0
        assert cfg.temperature == 0.3
        assert cfg.max_tokens == 2048
        assert cfg.thinking_budget == 1024
        assert cfg.max_phase_iterations == 10

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

    def test_env_var_override(self, monkeypatch: object) -> None:
        os.environ["RUNE_TEMPERATURE"] = "0.99"
        try:
            cfg = PipelineConfig.from_env()
            assert cfg.temperature == 0.99
        finally:
            del os.environ["RUNE_TEMPERATURE"]
