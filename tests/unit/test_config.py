import json
import os
from pathlib import Path

from rune.config import PipelineConfig, load_config


class TestPipelineConfig:
    def test_defaults(self) -> None:
        cfg = PipelineConfig()
        assert cfg.model_id == "Qwen/Qwen3.5-9B"
        assert cfg.adapter_scaling == 0.075
        assert cfg.temperature == 0.3
        assert cfg.max_tokens == 2048
        assert cfg.thinking_budget == 1024
        assert cfg.max_phase_iterations == 5
        assert cfg.prompt_style == "skeleton"
        assert cfg.trajectory_style == "prose"
        assert cfg.adapter_ttl_days == 7

    def test_frozen(self) -> None:
        cfg = PipelineConfig()
        try:
            cfg.temperature = 0.5  # type: ignore[misc]
            assert False, "Should raise"
        except AttributeError:
            pass

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

    def test_env_var_override(self, monkeypatch: object) -> None:
        os.environ["RUNE_TEMPERATURE"] = "0.99"
        try:
            cfg = PipelineConfig.from_env()
            assert cfg.temperature == 0.99
        finally:
            del os.environ["RUNE_TEMPERATURE"]

    def test_phase_max_tokens(self) -> None:
        cfg = PipelineConfig(phase_max_tokens={"plan": 512, "code": 2048})
        assert cfg.phase_max_tokens["plan"] == 512
