"""Public-example oracle: in-loop correctness signal from the spec's doctest.

The engine's only in-loop check was "does the module import" (a bare def exits 0),
so logic errors never triggered repair. The oracle appends the spec's *public*
example asserts (already shown to the model in the prompt -> no held-out leakage)
to the candidate, so a wrong/crashing implementation fails the sandbox and routes
to diagnose->repair with an actual-vs-expected message.
"""

from __future__ import annotations

from rune.engine.oracle import build_probe, defines_function, extract_public_checks

SPEC = (
    '"""\nImplement decode_string(s: str) -> str. k[encoded] repeats k times.\n\n'
    '>>> assert decode_string("3[a]2[bc]") == "aaabcbc"\n"""'
)


class TestExtractPublicChecks:
    def test_extracts_assert_example_for_entry_point(self) -> None:
        checks = extract_public_checks(SPEC, "decode_string")
        assert "decode_string" in checks
        assert "aaabcbc" in checks

    def test_passing_impl_runs_clean(self) -> None:
        checks = extract_public_checks(SPEC, "decode_string")
        good = (
            "def decode_string(s):\n"
            "    stack=[]; cur=''; k=0\n"
            "    for c in s:\n"
            "        if c.isdigit(): k=k*10+int(c)\n"
            "        elif c=='[': stack.append((cur,k)); cur,k='',0\n"
            "        elif c==']': p,r=stack.pop(); cur=p+cur*r\n"
            "        else: cur+=c\n"
            "    return cur"
        )
        ns: dict = {}
        exec(good, ns)  # noqa: S102 - test fixture
        exec(checks, ns)  # noqa: S102 - asserts must pass for a correct impl

    def test_wrong_impl_fails_with_actual_vs_expected(self) -> None:
        checks = extract_public_checks(SPEC, "decode_string")
        bad = "def decode_string(s):\n    return s"
        ns: dict = {}
        exec(bad, ns)  # noqa: S102 - test fixture
        try:
            exec(checks, ns)  # noqa: S102 - expected to raise
        except AssertionError as e:
            msg = str(e)
            assert "want" in msg and "aaabcbc" in msg  # actual-vs-expected message
        else:
            raise AssertionError("oracle did not fail on a wrong implementation")

    def test_no_doctest_returns_empty(self) -> None:
        assert extract_public_checks('"""no examples here"""', "foo") == ""

    def test_only_examples_calling_entry_point(self) -> None:
        spec = (
            '"""f\n>>> assert helper(1) == 2\n'
            '>>> assert target(3) == 9\n"""'
        )
        checks = extract_public_checks(spec, "target")
        assert "target(3)" in checks
        assert "helper" not in checks


class TestDefinesFunction:
    def test_top_level_def_detected(self) -> None:
        assert defines_function("def calculate(x):\n    return x", "calculate")

    def test_substring_in_comment_not_matched(self) -> None:
        # 'calculate' appears only in a comment / helper, not as a top-level def
        assert not defines_function("# calculate something\ndef other():\n    pass", "calculate")

    def test_syntax_error_is_false(self) -> None:
        assert not defines_function("def broken(:", "broken")


class TestBuildProbe:
    def test_appends_checks_when_entry_point_defined(self) -> None:
        code = "def decode_string(s):\n    return s"
        probe, fired = build_probe(code, SPEC, "decode_string")
        assert fired is True
        assert "decode_string" in probe
        assert "want" in probe  # the assert message

    def test_falls_back_to_bare_when_entry_point_absent(self) -> None:
        code = "def helper(s):\n    return s"  # entry_point not defined here
        probe, fired = build_probe(code, SPEC, "decode_string")
        assert fired is False
        assert probe == code

    def test_falls_back_when_no_checks(self) -> None:
        code = "def foo():\n    return 1"
        probe, fired = build_probe(code, '"""no examples"""', "foo")
        assert fired is False
        assert probe == code
