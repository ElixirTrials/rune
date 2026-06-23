from rune.model.wrapper import _tail_ids


def test_tail_ids_keeps_last_n() -> None:
    assert _tail_ids([1, 2, 3, 4, 5], 2) == [4, 5]


def test_tail_ids_shorter_than_n() -> None:
    assert _tail_ids([1, 2], 5) == [1, 2]


def test_tail_ids_zero_or_negative() -> None:
    assert _tail_ids([1, 2, 3], 0) == []
    assert _tail_ids([1, 2, 3], -1) == []
