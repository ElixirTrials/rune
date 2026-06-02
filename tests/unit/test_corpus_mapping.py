"""Corpus mapping + answer-preserving truncation contracts (issue #49 reviewer)."""

import json

from rune.training.hypernet_distill import _corpus_stats, _map_record, _prepare_ids


def test_map_record_synthetic_passthrough() -> None:
    assert _map_record({"context": "c", "answer": "a"}) == {
        "context": "c",
        "answer": "a",
    }


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
    assert (
        _map_record({"activation_text": "x", "teacher_text": "x"}) is None
    )  # empty revision


def test_corpus_stats_reports_prefix_and_fallback_rates(tmp_path) -> None:
    at = "## Task\nctx"
    rows = [
        {
            "activation_text": at,
            "teacher_text": at + "\n\n## Revision\nfix1",
            "task_id": "ok1",
        },
        {
            "activation_text": at,
            "teacher_text": at + "\n\n## Revision\nlonger fix here",
            "task_id": "ok2",
        },
        {
            "activation_text": "ctxX",
            "teacher_text": "re-rendered different",
            "task_id": "fb1",
        },
        {"activation_text": "only_ctx"},  # missing teacher_text -> skipped
    ]
    p = tmp_path / "corpus.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    st = _corpus_stats(str(p))
    assert st["raw"] == 4
    assert st["s3_rows"] == 3  # last row lacks teacher_text, not counted as s3 row
    assert st["exact_prefix"] == 2
    assert st["fallback"] == 1
    assert st["fallback_task_ids"] == ["fb1"]
    assert st["mapped"] == 3  # 2 exact + 1 fallback map; missing-teacher skipped
    assert st["answer_char_len"]["min"] >= 1


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
