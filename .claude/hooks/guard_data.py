"""PreToolUse hook: refuse edits that would put data into the repository or alter raw data.

Blocks Write/Edit to data/, and to data-like files (volumes, weights, arrays) anywhere.
Exit code 2 blocks the tool call and tells Claude why.
"""

import json
import sys
from pathlib import Path

BLOCKED_DIRS = {"data"}
BLOCKED_SUFFIXES = {
    ".nii",
    ".gz",
    ".mha",
    ".mhd",
    ".raw",
    ".tif",
    ".tiff",
    ".h5",
    ".hdf5",
    ".npy",
    ".npz",
    ".pt",
    ".pth",
    ".ckpt",
    ".safetensors",
    ".parquet",
}
ALLOWED_FILES = {"README.md"}


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
    root = Path(payload.get("cwd") or ".").resolve()
    try:
        rel = path.resolve().relative_to(root)
    except ValueError:
        rel = path

    if rel.parts and rel.parts[0] in BLOCKED_DIRS and rel.name not in ALLOWED_FILES:
        print(
            f"Blocked: {rel} is inside data/, which is read-only raw data. "
            "Write products to derived/ or results/ from code instead.",
            file=sys.stderr,
        )
        return 2
    if path.suffix.lower() in BLOCKED_SUFFIXES:
        print(
            f"Blocked: {rel} is a data or weights file. These are produced by running code "
            "on the data, never written by hand.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
