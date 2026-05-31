"""Corpus mapping + answer-preserving truncation contracts (issue #49 reviewer)."""
from rune.training.hypernet_distill import _map_record, _prepare_ids


def test_map_record_synthetic_passthrough() -> None:
    assert _map_record({"context": "c", "answer": "a"}) == {"context": "c", "answer": "a"}


def test_map_record_s3_strips_activation_prefix() -> None:
    at = "## Task\nfix it\n## Current Code\nx=1"
    tt = at + "\n\n## Revision\nx=2"
    m = _map_record({"activation_text": at, "teacher_text": tt})
    assert m is not None
    assert m["context"] == at
    assert m["answer"] == "## Revision\nx=2"  # leading whitespace stripped


def test_map_record_fallback_when_teacher_not_prefixed() -> None:
    m = _map_record({"activation_text": "ctx", "teacher_text": "unrelated answer"})
    assert m == {"context": "ctx", "answer": "unrelated answer"}


def test_map_record_none_on_empty_or_missing() -> None:
    assert _map_record({"activation_text": "ctx"}) is None  # no teacher_text
    assert _map_record({"context": "c", "answer": "   "}) is None  # empty answer
    assert _map_record({"activation_text": "x", "teacher_text": "x"}) is None  # empty revision


class _FakeTok:
    """Whitespace tokenizer: one token id per word."""

    def __call__(self, text, add_special_tokens=False):
        toks = text.split()
        return {"input_ids": [hash(t) % 1000 for t in toks]}


def test_prepare_ids_preserves_full_answer_and_front_truncates_context() -> None:
    tok = _FakeTok()
    context = " ".join(f"c{i}" for i in range(100))  # 100 ctx tokens
    answer = "a0 a1 a2 a3 a4"  # 5 answer tokens
    full_ids, ans_ids = _prepare_ids(tok, context, answer, max_length=10)
    # answer fully preserved and is the suffix of full_ids
    assert len(ans_ids) == 5
    assert full_ids[-5:] == ans_ids
    # total respects budget; context was front-truncated to its END (5 tokens kept)
    assert len(full_ids) == 10


def test_prepare_ids_answer_longer_than_max_keeps_answer_head() -> None:
    tok = _FakeTok()
    answer = " ".join(f"a{i}" for i in range(20))
    full_ids, ans_ids = _prepare_ids(tok, "ctx ctx ctx", answer, max_length=8)
    assert len(ans_ids) == 8
    assert full_ids == ans_ids  # no room for context
