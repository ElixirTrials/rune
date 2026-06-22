from rune.bench.repobench import RepoBenchRow, order_context, render_xfile_adapter


def _row(import_statement: str = "from m import tool") -> RepoBenchRow:
    return RepoBenchRow(
        task_id="cross_file_first/0",
        cropped_code="x = 1",
        import_statement=import_statement,
        context=({"identifier": "tool", "path": "m.py", "snippet": "def tool():\n    ..."},),
        gold_snippet_index=0,
        next_line="tool()",
        level="2k",
        token_num=10,
        repo_name="r",
        file_path="f.py",
    )


def test_gold_identifier() -> None:
    assert _row().gold_identifier == "tool"


def test_gold_identifier_out_of_range() -> None:
    row = RepoBenchRow(
        task_id="t", cropped_code="", import_statement="", context=(),
        gold_snippet_index=3, next_line="", level="2k", token_num=0,
        repo_name="r", file_path="f",
    )
    assert row.gold_identifier == ""


def test_gold_in_import_confound() -> None:
    # 'tool' is in 'from m import tool' -> confounded
    assert _row().gold_in_import
    # not in an unrelated import -> clean
    assert not _row(import_statement="import os").gold_in_import


def test_render_modes() -> None:
    r = _row()
    assert "def tool()" in render_xfile_adapter(r, "raw")
    structured = render_xfile_adapter(r, "structured")
    assert "## File: m.py" in structured
    assert "def tool()" in structured


def test_order_context_gold_first() -> None:
    ctx = (
        {"identifier": "a", "path": "a.py", "snippet": "A"},
        {"identifier": "b", "path": "b.py", "snippet": "B"},
        {"identifier": "c", "path": "c.py", "snippet": "C"},
    )
    row = RepoBenchRow(
        task_id="t", cropped_code="", import_statement="", context=ctx,
        gold_snippet_index=2, next_line="", level="8k", token_num=0,
        repo_name="r", file_path="f",
    )
    assert [c["identifier"] for c in order_context(row)] == ["a", "b", "c"]
    # gold (index 2 = "c") moves to front, others keep order
    assert [c["identifier"] for c in order_context(row, gold_first=True)] == ["c", "a", "b"]
    # gold snippet leads the rendered conditioning
    assert render_xfile_adapter(row, "raw", gold_first=True).startswith("C")


def test_render_training_mode() -> None:
    training = render_xfile_adapter(_row(), "training")
    # hypernet distillation surface, with the cross-file snippet as the Task
    assert training.startswith("## Task")
    assert "## Current Code" in training
    assert "## Review Feedback" in training
    assert "def tool()" in training
