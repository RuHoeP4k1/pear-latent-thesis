# Latent space is all you need

Code for the master's thesis on classifying internal quality of pears (browning, cavity, rot) from
X-ray CT using the latent space of a self-supervised encoder. KU Leuven, MeBioS, 2026–2027.

No data is stored in this repository.

## Quick start

```bash
uv sync                                            # Python 3.12 + all packages from uv.lock
cp config/local.example.toml config/local.toml     # then set the data paths for this machine
uv run marimo edit notebooks/00_environment_check.py
```

Windows: `scripts\setup.ps1`. Linux workstation: `bash scripts/setup.sh`.

## Layout

```
notebooks/        marimo notebooks, one question each (NN_name.py)
src/pearlatent/   shared code imported by notebooks (config, device, splits, labels, runs)
tests/            pytest tests for src/pearlatent
config/           local.example.toml (committed), local.toml (per machine, ignored)
data/             raw data, read-only, ignored by git
derived/          rebuildable intermediates such as latents, ignored by git
results/          per-run outputs with run.json, ignored by git
reports/figures/  figures that go into the thesis (committed)
docs/workflow.md  machines, setup, GPU, remote marimo
CLAUDE.md         rulebook for Claude Code
HANDOFF.md        current state and next step, updated each session
.claude/          hooks, skills and commands for Claude Code
```

## Working with Claude Code

Start `claude` in the repository root. It reads `CLAUDE.md`, and a hook prints the current state
from `HANDOFF.md`. Useful commands:

- `/new-notebook <short_name> <question>` — new notebook from the template
- `/marimo-check` — lint all notebooks and fix problems
- `/handoff` — update `HANDOFF.md` at the end of a session

Every edit to a notebook triggers `marimo check` automatically, and edits inside `data/` are blocked.
