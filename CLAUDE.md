# CLAUDE.md — rulebook for this repository

Master's thesis of Ruben (KU Leuven, Bioscience Engineering), supervisor Hugo (Junyan) Li, MeBioS.
Topic: whether the latent space of Hugo's self-supervised encoder holds enough information to
classify internal quality (browning, cavity, rot) of pears from X-ray CT, compared with 3DINO.

Read `HANDOFF.md` at the start of every session. It holds the current state and the next step.

## 1. Non-negotiable rules

1. **Data never enters git.** No CT volumes, label CSV files, weights, latents or result tables in a
   commit. Paths come from `config/local.toml` via `pearlatent.config.load_config()`. Never write an
   absolute path to data into code.
2. **Raw data is read-only.** Nothing writes into `data/`. Products go to `derived/` (rebuildable
   intermediates) or `results/<run_id>/`, each with a `run.json` from `pearlatent.runs`.
3. **Split by box, never by fruit.** Every model fitted on labels uses `pearlatent.splits`
   (`group_kfold` or an explicit box hold-out) and calls `assert_no_group_leakage` before fitting.
   The frozen encoder is not affected.
4. **Nothing is trained from scratch.** At most: frozen-encoder features, linear or statistical
   models on them, fine-tuning, staged pretraining.
5. **Claims about our data trace to Tier 1** (Hugo's preprint SSRN 6433761 and its supplement,
   github.com/Hugo-Li-Junyan/Synthetic_CT_pear, the two label CSV files, the grading rubric).
   If code depends on an unverified fact (file-name pattern, column name, preprocessing step),
   mark it `# UNVERIFIED:` and list it in `HANDOFF.md`. Numbers in the thesis come from a run, not
   from memory.
6. **No cloud for data.** molab notebooks are public-but-unlisted by default. Until Hugo confirms in
   writing that the data may leave university systems, molab is used only for toy or public data.

## 2. Stack

| Purpose | Tool | Not |
|---|---|---|
| Environment | `uv` (`uv sync`, `uv run`, `uv add`) | pip, conda, `uv pip install` into the project |
| Notebooks | marimo `.py` files in `notebooks/` | Jupyter `.ipynb` |
| Tables | Polars (lazy `scan_*` for files) | pandas, except at a library boundary (`.to_pandas()` on the last line before the call) |
| Statistics | scikit-learn (PCA, PLS-DA, penalised regression), statsmodels (`OrderedModel` for ordinal grades) | ad-hoc reimplementations |
| Deep learning | PyTorch | TensorFlow, Lightning (not needed at this scale) |
| Plots | Altair inside notebooks; Matplotlib for thesis figures in `reports/figures/` | |

Add packages only with `uv add <pkg>` and commit `pyproject.toml` and `uv.lock` together.

## 3. marimo conventions

Load the `thesis-notebook` skill before writing or editing any notebook. In short:

1. One notebook answers one question. Name `NN_short_name.py` (two-digit order). Start from
   `notebooks/_template.py`; keep its header (question, inputs, outputs, status).
2. Imports only in the `with app.setup:` block.
3. A variable is defined in exactly one cell and never mutated in another cell. Temporaries start
   with `_`.
4. Logic reused by two notebooks, or worth testing, moves to `src/pearlatent/` with a test in `tests/`.
   Notebooks stay thin.
5. UI element created in one cell, its `.value` read in another.
6. Anything slower than a few seconds sits behind `mo.ui.run_button` + `mo.stop`, and its result is
   cached: `mo.persistent_cache` for small results, files in `derived/` for large arrays (latents).
7. GPU tensors live inside functions so memory is freed; use `torch.inference_mode()` for encoding.
8. Every notebook must also run as a script: `uv run python notebooks/NN_name.py`. Long runs on the
   workstation are scripts, not open browser tabs.
9. After every edit, `marimo check` runs automatically (hook). Fix every error it reports before
   moving on.

## 4. Machines

- **Laptop (Windows)** and **school PC (Windows, no admin)**: editing, statistics on saved latents,
  plots. CPU PyTorch.
- **Lab workstation (Linux, GPU, via SSH)**: encoding volumes, fine-tuning. Load the
  `workstation-run` skill before preparing anything that runs there.
- Sync only through git (`git pull` at start, `git push` at end). Data is synced separately.

Details: `docs/workflow.md`.

## 5. How to work with Ruben

1. Ask before a decision that changes the analysis plan; do not change the plan silently.
2. Prefer few well-built paths over many options. Say plainly when an idea is weak.
3. Flag contradictions between documents instead of working around them.
4. Plain literal language in comments, markdown cells and commit messages: no metaphors, write
   abbreviations out on first use.
5. Commit small, with messages in the imperative ("Add box-level split for browning model").
6. At the end of a session, run `/handoff`.

## 6. Commands

```bash
uv sync                                         # install exactly what uv.lock says
uv run marimo edit notebooks/                   # open the notebook browser
uv run marimo edit --watch notebooks/NN_x.py    # editor reloads when Claude edits the file
uv run marimo check notebooks/                  # lint all notebooks
uv run pytest                                   # tests for src/pearlatent
uv run ruff check src tests && uv run ruff format src tests
```
