import time
from pathlib import Path

from rune.registry.store import AdapterRecord, AdapterRegistry


class TestAdapterRegistry:
    def test_register_and_get(self) -> None:
        reg = AdapterRegistry.create(":memory:")
        reg.register(
            adapter_id="a1",
            disk_path="/tmp/a1.safetensors",
            parent_id=None,
            action="decompose",
            session_id="s1",
            generation=0,
        )
        record = reg.get("a1")
        assert record is not None
        assert record.adapter_id == "a1"
        assert record.action == "decompose"
        assert record.parent_id is None

    def test_get_missing_returns_none(self) -> None:
        reg = AdapterRegistry.create(":memory:")
        assert reg.get("nonexistent") is None

    def test_lineage(self) -> None:
        reg = AdapterRegistry.create(":memory:")
        reg.register("a1", "/tmp/a1", None, "decompose", "s1", 0)
        reg.register("a2", "/tmp/a2", "a1", "plan", "s1", 1)
        reg.register("a3", "/tmp/a3", "a2", "code", "s1", 2)
        lineage = reg.lineage("a3")
        assert [r.adapter_id for r in lineage] == ["a3", "a2", "a1"]

    def test_list_by_session(self) -> None:
        reg = AdapterRegistry.create(":memory:")
        reg.register("a1", "/tmp/a1", None, "decompose", "s1", 0)
        reg.register("a2", "/tmp/a2", None, "decompose", "s2", 0)
        records = reg.list_by_session("s1")
        assert len(records) == 1
        assert records[0].adapter_id == "a1"

    def test_prune_by_age(self, tmp_path: Path) -> None:
        reg = AdapterRegistry.create(":memory:")
        reg.register("old", str(tmp_path / "old.st"), None, "code", "s1", 0)
        reg._backdate("old", days=10)
        reg.register("new", str(tmp_path / "new.st"), None, "code", "s1", 1)
        (tmp_path / "old.st").write_bytes(b"\x00")
        pruned = reg.prune(max_age_days=7)
        assert pruned == 1
        assert reg.get("old") is None
        assert reg.get("new") is not None
        assert not (tmp_path / "old.st").exists()
