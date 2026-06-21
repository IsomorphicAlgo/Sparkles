"""Load gitignored ``.env`` into ``os.environ`` (does not override existing vars)."""

from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: Path | None = None) -> bool:
    """Parse a simple ``KEY=VALUE`` file into the environment.

    Returns True if the file existed and was read. Quotes around values are stripped.
    Existing environment variables are never overwritten.
    """
    env_path = Path.cwd() / ".env" if path is None else path
    if not env_path.is_file():
        return False
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").strip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key or key in os.environ:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ[key] = value
    return True
