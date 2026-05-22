from rune.engine.interfaces import extract_interfaces


class TestExtractInterfaces:
    def test_extract_function_signature(self) -> None:
        code = "def add(a: int, b: int) -> int:\n    return a + b\n"
        interfaces = extract_interfaces(code)
        assert "def add(a: int, b: int) -> int" in interfaces

    def test_extract_class_definition(self) -> None:
        code = "class Calculator:\n    def __init__(self) -> None:\n        self.value = 0\n    def add(self, x: int) -> None:\n        self.value += x\n"
        interfaces = extract_interfaces(code)
        assert "class Calculator" in interfaces
        assert "def add" in interfaces

    def test_empty_code(self) -> None:
        assert extract_interfaces("") == ""

    def test_no_definitions(self) -> None:
        code = "x = 1\ny = 2\nprint(x + y)\n"
        interfaces = extract_interfaces(code)
        assert "def " not in interfaces
