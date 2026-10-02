# HANDOFF.md

The living state of the code work. Claude reads this at the start of every session (a hook prints
the "Current state" section) and updates it with `/handoff` at the end. Ruben may edit it freely.

## Current state

- **Date:** 2026-10-02
- **Phase:** setup — repository skeleton created, not yet run against real data.
- **Last done:** created repository structure, rulebook (CLAUDE.md), hooks, skills, environment
  check notebook, box-level split and label utilities with tests.
- **Verified (2026-10-02, Linux test copy, marimo 0.25.1, torch 2.14.1):** `marimo check`
  clean, 7 tests pass, ruff clean, both notebooks run as scripts, all three hooks behave.
  `uv.lock` is not yet committed: download.pytorch.org was not reachable from the build
  machine. The first `uv sync` on the laptop creates it; commit it straight away.
- **Next step:** on each machine: `uv sync`, copy `config/local.example.toml` to
  `config/local.toml`, run `notebooks/00_environment_check.py`. Then write
  `01_label_inventory.py`: load both label CSV files, confirm column names, run
  `validate_ranges` and `check_derived_labels`, count fruit per box and per harvest year.
- **Blocked on:** see "Open questions".

## Open questions (must be answered from Tier 1 or by Hugo)

1. Exact file-name pattern of the CT volumes, and where the box letter sits in it.
2. File format of the volumes (NIfTI, MHA, TIFF stack, HDF5?) and whether they are already 128³ or
   need resampling.
3. Column names in the two label CSV files; whether `defective`, `binary_123`, `binary_23` exist as
   columns or must be derived.
4. Whether the encoder weights and decoder are released in the Synthetic_CT_pear repository, and how
   the encoder expects input (intensity scaling is min-max to [0,1] per volume — confirm in code).
5. Whether the data may be uploaded to cloud services (molab). Until answered: no.
6. Lab workstation: GPU model, NVIDIA driver version (`nvidia-smi`), operating system, whether `uv`
   is installed, whether tmux is available.

## Unverified assumptions in code

| Where | Assumption | Verify against |
|---|---|---|
| `src/pearlatent/labels.py` | binary_123 = defective ≥ 1, binary_23 = defective ≥ 2 | label CSV via `check_derived_labels` |
| `pyproject.toml` | workstation driver supports CUDA 12.8 wheels | `nvidia-smi` on the workstation |
| `uv.lock` | missing; first `uv sync` on the laptop resolves it | commit `uv.lock` after the first sync |

## Session log

Newest first. One entry per session, three to six lines.

### 2026-10-02 — setup
- Created the repository skeleton and agent rulebook.
- Decided: uv project (not per-notebook sandboxes), Polars, PyTorch, marimo with project config,
  data outside git, molab not used for data.
