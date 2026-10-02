---
description: Create a new numbered marimo notebook from the template for one research question
argument-hint: <short_name> <one-sentence question>
allowed-tools: Read, Write, Glob, Bash(uv run marimo check:*)
---

Existing notebooks:
!`git ls-files notebooks/`

Create a new notebook for: $ARGUMENTS

1. Load the `thesis-notebook` skill.
2. Pick the next free two-digit number. File name `notebooks/NN_<short_name>.py`.
3. Copy `notebooks/_template.py`, fill in the header cell (question, inputs, outputs, status:
   draft) and the app title.
4. Before writing analysis cells, list what the notebook needs from the data and which of those
   facts are unverified (see HANDOFF.md). Ask Ruben if an unverified fact decides the design.
5. Run `uv run marimo check` on the file.
