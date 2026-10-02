---
description: Lint marimo notebooks and fix any errors
allowed-tools: Bash(uv run marimo check:*), Read, Edit
---

## Context

Output of `uv run marimo check --fix $ARGUMENTS`:

!`uv run marimo check --fix ${ARGUMENTS:-notebooks/} || true`

## Your task

Only if the output above reports errors or warnings: read the affected notebooks and fix them,
following the `thesis-notebook` skill. If there are no issues, say so and do nothing else.
