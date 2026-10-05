"""Run records: every product in derived/ or results/ is traceable to code and parameters."""

from __future__ import annotations

import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path


def git_state(root: Path | None = None) -> dict:
    """Commit hash and whether the working tree has uncommitted changes."""

    def _git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True, check=False
        ).stdout.strip()

    return {
        "commit": _git("rev-parse", "HEAD") or None,
        "dirty": bool(_git("status", "--porcelain")),
    }


def new_run_dir(base: Path, name: str) -> Path:
    """Create `base/<UTC timestamp>_<name>/` and return it."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(base) / f"{stamp}_{name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_run_record(run_dir: Path, params: dict, root: Path | None = None) -> Path:
    """Write run.json with parameters, git state, host and library versions."""
    import numpy as np
    import polars as pl

    try:
        import torch

        torch_info = {"torch": torch.__version__, "cuda": torch.version.cuda}
    except ImportError:
        torch_info = {}

    record = {
        "created_utc": datetime.now(UTC).isoformat(),
        "host": platform.node(),
        "python": platform.python_version(),
        "git": git_state(root),
        "versions": {"numpy": np.__version__, "polars": pl.__version__, **torch_info},
        "params": params,
    }
    out = Path(run_dir) / "run.json"
    out.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    return out


def latest_run_dir(base: Path, name: str) -> Path | None:
    """The newest `base/<timestamp>_<name>/` that contains a run.json, or None."""
    candidates = sorted(
        p
        for p in Path(base).glob(f"*_{name}")
        if p.is_dir() and (p / "run.json").exists()
    )
    return candidates[-1] if candidates else None
