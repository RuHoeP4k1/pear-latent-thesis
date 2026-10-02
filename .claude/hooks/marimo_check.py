"""PostToolUse hook: run `marimo check` after Claude edits a marimo notebook.

Exit code 2 sends stderr back to Claude, which then fixes the reported problems.
Standard library only, so it works on Windows and Linux without the project environment.
"""

import json
import subprocess
import sys
from pathlib import Path


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError:
        return 0
    tool_input = payload.get("tool_input") or {}
    file_path = tool_input.get("file_path") or tool_input.get("notebook_path")
    if not file_path:
        return 0
    path = Path(file_path)
    if path.suffix != ".py" or not path.exists():
        return 0
    text = path.read_text(encoding="utf-8", errors="ignore")
    if "import marimo" not in text or "marimo.App(" not in text:
        return 0

    result = subprocess.run(
        ["uv", "run", "--quiet", "marimo", "check", str(path)],
        capture_output=True,
        text=True,
        cwd=payload.get("cwd") or None,
    )
    output = (result.stdout + result.stderr).strip()
    if result.returncode != 0:
        print(
            f"marimo check failed for {path.name}:\n{output}\n\n"
            "Fix these problems now, without asking the user, then continue.",
            file=sys.stderr,
        )
        return 2
    if "warning[" in output:
        # marimo lint warnings (not uv messages): show them to Claude so it fixes them
        print(f"marimo check warnings for {path.name}:\n{output}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
