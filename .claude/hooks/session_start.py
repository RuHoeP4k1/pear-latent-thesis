"""SessionStart hook: print the current state from HANDOFF.md and the git status.

Standard output of a SessionStart hook is added to Claude's context.
"""

import subprocess
import sys
from pathlib import Path


def section(text: str, heading: str) -> str:
    lines = text.splitlines()
    out, inside = [], False
    for line in lines:
        if line.startswith("## "):
            if inside:
                break
            inside = line.strip() == f"## {heading}"
            continue
        if inside:
            out.append(line)
    return "\n".join(out).strip()


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    handoff = root / "HANDOFF.md"
    if handoff.exists():
        text = handoff.read_text(encoding="utf-8")
        print("== HANDOFF.md: Current state ==")
        print(section(text, "Current state"))
        print("\n== HANDOFF.md: Open questions ==")
        print(section(text, "Open questions (must be answered from Tier 1 or by Hugo)"))
    status = subprocess.run(
        ["git", "status", "--short", "--branch"],
        cwd=root,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        print("\n== git status ==")
        print(status)
    if not (root / "config" / "local.toml").exists():
        print(
            "\nNote: config/local.toml is missing on this machine; data paths are not set."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
