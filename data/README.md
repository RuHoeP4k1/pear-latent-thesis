# data/

This folder is ignored by git, except this file. Nothing placed here is ever committed.

On each machine, either copy the data here or point `config/local.toml` to where the data already lives (recommended on the workstation, to avoid a second copy).

Expected contents, once Hugo confirms the formats:

1. The CT volumes for the 2024, 2025 and 2026 harvests.
2. The two label CSV files (browning 0–3, cavity 0–3, rot 0–1).
3. The grading rubric table.

Raw files are read-only. Code never writes into this folder; it writes into `derived/` or `results/`.
