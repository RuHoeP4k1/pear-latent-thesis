---
name: thesis-notebook
description: House conventions for marimo notebooks in this thesis repository. Load before creating, editing or reviewing any file in notebooks/, or when turning notebook code into src/pearlatent functions.
---

# Writing thesis notebooks in marimo

These rules add to the official marimo skills (`marimo-notebook`, `marimo-pair`). Where they
differ, these rules win for this repository.

## File layout of a notebook

```python
import marimo

__generated_with = "<version>"
app = marimo.App(width="medium", app_title="NN Title")

with app.setup:
    import marimo as mo
    import polars as pl
    from pearlatent.config import load_config

@app.cell(hide_code=True)
def _():
    mo.md(r"""# NN Title ... question, inputs, outputs, status""")
    return

# cells ...

if __name__ == "__main__":
    app.run()
```

Always start from `notebooks/_template.py`. Never hand-edit `__generated_with`.

## Reactivity rules (these are what `marimo check` enforces, plus ours)

1. Each global name is defined in exactly one cell. Reusing a name causes a "multiple definitions" error.
2. Never mutate a global from another cell (`df = df.with_columns(...)` in a second cell is wrong;
   name the result `df_scored`). Polars expressions return new frames, which makes this natural.
3. Temporaries start with `_` (`_tmp`, `_fig`); they are local to the cell.
4. A cell's last expression is its output. Put display calls last.
5. A UI element is created in one cell and read (`.value`) in a different cell. A cell that reads
   `.value` reruns when the user changes the element.
6. Only `return` the names other cells need.

## Expensive and GPU work

1. Gate with a run button:
   ```python
   run = mo.ui.run_button(label="Encode volumes")
   run
   ```
   ```python
   mo.stop(not run.value, mo.md("_Press the button to start._"))
   ```
   In script mode the button value is False, so a cell that must also run as a script uses
   `mo.stop(mo.running_in_notebook() and not run.value)` and reads parameters with
   `mo.cli_args()` (see `workstation-run`).
2. Cache:
   - Small results (metrics, fitted scikit-learn models, tables under ~100 MB):
     `@mo.persistent_cache` on a function in its own cell. Cache lives in
     `notebooks/__marimo__/cache/` (ignored by git).
   - Large arrays (latents for all fruit): write `derived/latents/<encoder>/<run_id>/latents.npy`
     plus `manifest.parquet` (one row per fruit: file, box, year, row index) plus `run.json`.
     Notebooks load these files; they do not recompute them.
3. GPU memory: create tensors inside a function, return only CPU NumPy arrays or Polars frames.
   Use `torch.inference_mode()` and `torch.autocast("cuda", dtype=torch.bfloat16)` for encoding
   when `describe_device().bf16_supported`.
4. Show progress with `mo.status.progress_bar(iterable, title=...)`.

## Data handling with Polars

1. Read files lazily: `pl.scan_csv(cfg.path("labels_2024"))`, then `.collect()` once.
2. Declare schemas for label files (`schema_overrides={"browning": pl.Int8, ...}`) and run
   `pearlatent.labels.validate_ranges` right after loading.
3. Join on explicit keys with `validate="1:1"` so duplicated fruit identifiers fail immediately.
4. Convert to NumPy at the model boundary only: `X = latents_df.select(feature_cols).to_numpy()`.
5. `mo.ui.table(df)` or `mo.ui.dataframe(df)` for inspection; `mo.ui.data_explorer(df)` for quick
   visual checks.

## Statistics and models

1. Splits: `pearlatent.splits.group_kfold(df, group_col="box", stratify_col=...)`, then
   `assert_no_group_leakage`. Fit preprocessing (scaling, PCA) inside the training fold only:
   use a scikit-learn `Pipeline`.
2. Ordinal grades (browning 0–3): `statsmodels.miscmodels.ordinal_model.OrderedModel`, compared
   with a binary model on `binary_23`. Report per-fold metrics and their spread, not one number.
3. Set `random_state` everywhere and record it in `run.json`.

## Plots

- Exploration: Altair (`mo.ui.altair_chart` makes selections reactive).
- Thesis figures: Matplotlib, saved to `reports/figures/NN_name.pdf` by a cell that only runs when a
  "Save figure" run button is pressed.

## When to move code to `src/pearlatent/`

Move a function when a second notebook needs it, when it has a branch worth testing, or when it is
longer than about 30 lines. Add a test in `tests/`. The notebook then imports it; module
autoreload (configured in pyproject.toml) picks up edits without a kernel restart.

## Writing style inside notebooks

Plain literal sentences, abbreviations written out on first use, no metaphors. Every markdown
header cell states the question, the inputs, the outputs and the status.

## Finish

Run `uv run marimo check notebooks/<file>.py` (the hook does this after each edit) and
`uv run python notebooks/<file>.py` to prove the notebook also runs as a script.
