from typer.testing import CliRunner

from rune.cli import app

runner = CliRunner()


class TestCLI:
    def test_help(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "run" in result.output
        assert "train" in result.output
        assert "bench" in result.output
        assert "mine" in result.output

    def test_run_requires_task(self) -> None:
        result = runner.invoke(app, ["run"])
        assert result.exit_code != 0

    def test_bench_help(self) -> None:
        result = runner.invoke(app, ["bench", "--help"])
        assert result.exit_code == 0
        assert "hpo" in result.output.lower() or "n-trials" in result.output.lower()
