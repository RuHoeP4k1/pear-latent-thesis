"""Machine-specific configuration.

Each machine (laptop, school PC, lab workstation) has its own `config/local.toml`,
which is ignored by git. Code never hard-codes a path to data.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path


def repo_root() -> Path:
    """Return the repository root (the folder that contains pyproject.toml)."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("Could not find pyproject.toml above " + str(here))


@dataclass(frozen=True)
class Config:
    raw: dict
    root: Path

    def path(self, key: str) -> Path:
        """Resolve a path from the [paths] table; relative paths are taken from the repo root."""
        value = self.raw.get("paths", {}).get(key)
        if value is None:
            raise KeyError(
                f"'{key}' is missing from [paths] in config/local.toml. "
                "Copy config/local.example.toml and fill it in."
            )
        p = Path(value).expanduser()
        return p if p.is_absolute() else self.root / p

    def compute(self, key: str, default=None):
        return self.raw.get("compute", {}).get(key, default)


def load_config(path: str | Path | None = None) -> Config:
    """Load config/local.toml, or the file named by the PEARLATENT_CONFIG environment variable."""
    root = repo_root()
    candidate = Path(
        path or os.environ.get("PEARLATENT_CONFIG", root / "config" / "local.toml")
    )
    if not candidate.exists():
        raise FileNotFoundError(
            f"{candidate} does not exist. On this machine run:\n"
            "  copy config\\local.example.toml config\\local.toml   (Windows)\n"
            "  cp config/local.example.toml config/local.toml       (Linux)\n"
            "and edit the paths."
        )
    # utf-8-sig drops the byte-order mark that Windows PowerShell 5 and Notepad may write;
    # tomllib refuses a file that starts with one.
    text = candidate.read_text(encoding="utf-8-sig")
    return Config(raw=tomllib.loads(text), root=root)
