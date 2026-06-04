from rune.engine.graph import render_training_format_trajectory


def test_inference_trajectory_uses_training_headers() -> None:
    txt = render_training_format_trajectory(
        task="implement find_tuples",
        current_code="def f(): pass",
        feedback="use all() not any()",
    )
    assert "## Task" in txt and "## Current Code" in txt and "## Review Feedback" in txt


def test_attempt1_byte_identical_to_distillation_surface() -> None:
    # No prior attempts => unchanged 3-section format (on c3's training surface).
    txt = render_training_format_trajectory(task="implement f", current_code="", feedback="")
    assert txt == "## Task\nimplement f\n\n## Current Code\n\n\n## Review Feedback\n"
    assert "## Previous Attempts" not in txt


def test_prior_attempts_become_episode_history() -> None:
    # R2: with prior failing attempts, the adapter carries "what's been tried".
    attempts = [
        {"code": "def f(): return 1", "error": "AssertionError", "passed": False},
        {"code": "def f(): return 2", "error": "NameError: x", "passed": False},
    ]
    txt = render_training_format_trajectory(
        task="implement f", current_code="def f(): return 3",
        feedback="still wrong", attempts=attempts,
    )
    assert "## Previous Attempts" in txt
    assert "def f(): return 1" in txt and "AssertionError" in txt
    assert "def f(): return 2" in txt and "NameError" in txt
    # current attempt stays in ## Current Code, not duplicated into history
    assert "def f(): return 3" in txt.split("## Previous Attempts")[0]
