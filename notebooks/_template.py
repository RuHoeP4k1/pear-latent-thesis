import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title="NN Title")

with app.setup:
    import marimo as mo
    import polars as pl

    from pearlatent.config import load_config


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # NN Title

    **Question.** One sentence: what this notebook answers.

    **Inputs.** Which files or derived products it reads (by config key, never by absolute path).

    **Outputs.** What it writes, and where (`derived/` or `results/<run_id>/`).

    **Status.** draft / in use / superseded by NN
    """)
    return


@app.cell
def _():
    cfg = load_config()
    return


@app.cell
def _():
    # Parameters as UI elements, so the notebook can be explored interactively.
    # When the notebook runs as a script, the default values are used.
    seed = mo.ui.number(0, label="Random seed")
    seed
    return


if __name__ == "__main__":
    app.run()
