from rune.bench.identifier_match import (
    edit_similarity,
    exact_match,
    extract_identifiers,
    gold_id_recovery,
    identifier_f1,
)


def test_extract_identifiers_filters_keywords() -> None:
    assert extract_identifiers("class Foo(Bar):") == ["Foo", "Bar"]


def test_gold_id_recovery_whole_token() -> None:
    assert gold_id_recovery("rs = tool.runffmpeg(params)", "tool")
    # substring must NOT count as recovery
    assert not gold_id_recovery("x = tooltip()", "tool")


def test_gold_id_recovery_member_call() -> None:
    assert gold_id_recovery("self.layers = _get_clones(layer, n)", "_get_clones")


def test_gold_id_recovery_empty_gold() -> None:
    assert not gold_id_recovery("anything", "")


def test_identifier_f1_exact() -> None:
    assert identifier_f1("a = foo(b)", "a = foo(b)") == 1.0


def test_identifier_f1_partial() -> None:
    # pred {a, foo}, gold {a, bar} -> overlap 1, p=1/2, r=1/2 -> f1 0.5
    assert abs(identifier_f1("a = foo()", "a = bar()") - 0.5) < 1e-9


def test_identifier_f1_disjoint() -> None:
    assert identifier_f1("foo()", "bar()") == 0.0


def test_exact_match_strips() -> None:
    assert exact_match("  x = 1  ", "x = 1")
    assert not exact_match("x = 1", "x = 2")


def test_edit_similarity_bounds() -> None:
    assert edit_similarity("abc", "abc") == 1.0
    assert 0.0 <= edit_similarity("abc", "xyz") < 1.0
