"""Load .env into os.environ."""

from __future__ import annotations

from pathlib import Path

from sparkles.env import load_dotenv


def test_load_dotenv_sets_missing_keys(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TWELVEDATA_API_KEY", raising=False)
    (tmp_path / ".env").write_text(
        'TWELVEDATA_API_KEY=secret-from-file\n# comment\nFOO="bar"\n',
        encoding="utf-8",
    )
    assert load_dotenv() is True
    assert __import__("os").environ["TWELVEDATA_API_KEY"] == "secret-from-file"
    assert __import__("os").environ["FOO"] == "bar"


def test_load_dotenv_does_not_override_existing(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TWELVEDATA_API_KEY", "from-shell")
    (tmp_path / ".env").write_text("TWELVEDATA_API_KEY=from-file\n", encoding="utf-8")
    load_dotenv()
    assert __import__("os").environ["TWELVEDATA_API_KEY"] == "from-shell"
