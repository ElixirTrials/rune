from rune.engine.json_repair import extract_code_value


class TestExtractCodeValue:
    def test_complete_json(self) -> None:
        raw = '{"code": "def f():\\n    return 1"}'
        assert extract_code_value(raw) == "def f():\n    return 1"

    def test_truncated_mid_string(self) -> None:
        raw = '{"code": "import os\\ndef main():\\n    print(\\"hello'
        result = extract_code_value(raw)
        assert result == 'import os\ndef main():\n    print("hello'

    def test_truncated_mid_escape(self) -> None:
        raw = '{"code": "line1\\nline2\\'
        result = extract_code_value(raw)
        assert result == "line1\nline2"

    def test_no_code_key_returns_empty(self) -> None:
        assert extract_code_value('{"cod') == ""
        assert extract_code_value("") == ""
        assert extract_code_value("garbage") == ""

    def test_unicode_escape(self) -> None:
        raw = '{"code": "x = \\u0041\\u0042"}'
        assert extract_code_value(raw) == "x = AB"

    def test_truncated_unicode_escape(self) -> None:
        raw = '{"code": "x = \\u00"}'
        result = extract_code_value(raw)
        assert "x = " in result

    def test_tab_and_cr_escapes(self) -> None:
        raw = '{"code": "a\\tb\\rc"}'
        assert extract_code_value(raw) == "a\tb\rc"

    def test_escaped_quotes_in_code(self) -> None:
        raw = '{"code": "x = \\"hello\\""}'
        assert extract_code_value(raw) == 'x = "hello"'

    def test_forward_slash_escape(self) -> None:
        raw = '{"code": "a\\/b"}'
        assert extract_code_value(raw) == "a/b"
