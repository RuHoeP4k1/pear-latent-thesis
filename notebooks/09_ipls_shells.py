import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title="09 iPLS on shells")

with app.setup(hide_code=True):
    import altair as alt
    import marimo as mo
    import numpy as np
    import polars as pl

    from pearlatent.config import load_config
    from pearlatent.feature_sets import load_features
    from pearlatent.fruit_table import drop_faulty_volumes, open_fruit
    from pearlatent.ipls import (
        MAX_RADIUS,
        N_SHELLS,
        forward_ipls,
        interval_rmsecv,
        shell_labels,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 09 Interval PLS on shells around the stem–calyx axis

    **Question.** Which shells of the fruit, from the axis outwards, carry the information for
    PLS-DA on `binary_23`, and which combination of shells gives the lowest cross-validation
    error?

    **Method** (Ruben, 2 November 2026; `pearlatent.ipls`). The 32³ grid is cut into hollow
    cylinders around the grid axis (the stem–calyx axis after alignment), each over the full
    height. The 10 shells inside a radius of 10 cells hold about the same number of cells
    (about 1,000 each); shell 10 is everything further out (mostly outside the fruit). For
    every shell alone and for every step of forward iPLS: PLS on those cells only, the same
    20 box partitions as notebook 08, centring only, RMSECV at the lowest point of the curve
    (1 to 15 LVs). Forward iPLS adds, at each step, the shell that lowers RMSECV most.

    **Caveat.** The RMSECV of the chosen combination is slightly optimistic: the shells are
    chosen on the same cross-validation. The sealed test boxes give the independent check.

    **Inputs.** As notebook 08. **Outputs.** Figures and tables only.

    **Status.** draft (first run, 2 November 2026)
    """)
    return


@app.cell
def _():
    cfg = load_config()
    shells, boundaries = shell_labels()
    shell_names = [
        f"{_k}: {boundaries[_k]:.1f}–{boundaries[_k + 1]:.1f}" for _k in range(N_SHELLS)
    ] + [f"{N_SHELLS}: > {MAX_RADIUS:.0f}"]
    print("shell boundaries (cells from the axis):", np.round(boundaries, 2))
    print("cells per shell:", np.bincount(shells.ravel()).tolist())
    return cfg, shell_names, shells


@app.cell(hide_code=True)
def _(cfg):
    _open_2526 = open_fruit(cfg).filter(pl.col("season") == "2526")
    fruit = drop_faulty_volumes(
        _open_2526, cfg.path("derived_dir") / "volume_decisions.csv"
    ).with_columns(binary_23=(pl.col("defective") >= 2).cast(pl.Int8))
    print(f"open 2526 fruit {_open_2526.height}, after faulty volumes {fruit.height}")
    return (fruit,)


@app.cell(hide_code=True)
def _():
    # LVs of the full-grid models chosen in notebook 08 (Ruben, 2 November 2026).
    FULL_GRID_LV = {"hugo_vae": 4, "ct_32": 5}
    # In script mode the representation comes from the command line:
    # uv run python notebooks/09_ipls_shells.py -- --representation ct_32
    representation_choice = mo.ui.dropdown(
        options={"Hugo's latent (mu)": "hugo_vae", "CT averaged to 32³": "ct_32"},
        value="CT averaged to 32³"
        if mo.cli_args().get("representation") == "ct_32"
        else "Hugo's latent (mu)",
        label="Representation",
    )
    run_ipls = mo.ui.run_button(label="Run iPLS (about 5 minutes the first time)")
    mo.hstack([representation_choice, run_ipls], justify="start")
    return FULL_GRID_LV, representation_choice, run_ipls


@app.function
@mo.persistent_cache
def run_shells(representation: str, fruit_ids: tuple) -> dict:
    """Each shell alone, the forward path, and the full-grid curve, for one representation.

    In: the feature set name and the fruit (in table order). Out: a dict with the tables
    `alone` and `path` of `forward_ipls` and `full_curve` (RMSECV for 1 to 15 LVs on all
    cells). Cached on disk.
    """
    cfg = load_config()
    fruit = (
        drop_faulty_volumes(
            open_fruit(cfg).filter(pl.col("season") == "2526"),
            cfg.path("derived_dir") / "volume_decisions.csv",
        )
        .with_columns(binary_23=(pl.col("defective") >= 2).cast(pl.Int8))
        .filter(pl.col("fruit_id").is_in(list(fruit_ids)))
    )
    assert tuple(fruit["fruit_id"].to_list()) == fruit_ids, "fruit order changed"
    x = load_features(cfg, representation, list(fruit_ids))
    y = fruit["binary_23"].to_numpy()
    boxes = fruit["box"].to_numpy()
    folds = pl.read_csv(cfg.path("derived_dir") / "outer_folds_20" / "outer_folds.csv")
    labels, _ = shell_labels()
    alone, path = forward_ipls(x, y, boxes, folds, labels)
    full_curve = interval_rmsecv(x, y, boxes, folds)
    return {"alone": alone, "path": path, "full_curve": full_curve}


@app.cell(hide_code=True)
def _(fruit, representation_choice, run_ipls):
    mo.stop(
        mo.running_in_notebook() and not run_ipls.value,
        mo.md("_Press the button to run iPLS._"),
    )
    result = run_shells(representation_choice.value, tuple(fruit["fruit_id"].to_list()))
    return (result,)


@app.cell(hide_code=True)
def _(FULL_GRID_LV, representation_choice, result):
    # The step of the forward path with the lowest RMSECV is the selected combination.
    path = result["path"]
    best_step = path.sort("rmsecv").row(0, named=True)
    full_lv = FULL_GRID_LV[representation_choice.value]
    full_rmsecv = float(result["full_curve"][full_lv - 1])
    # Printed too, so the script run shows the result.
    print(result["alone"])
    print(path)
    print(f"full grid {full_lv} LVs: {full_rmsecv:.4f}; lowest: {best_step}")
    mo.md(
        f"**Full grid** ({full_lv} LVs): RMSECV {full_rmsecv:.4f}. "
        f"**Lowest combination**: shells {best_step['shells']} "
        f"({best_step['n_lv']} LVs), RMSECV {best_step['rmsecv']:.4f}."
    )
    return best_step, full_lv, full_rmsecv, path


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1. Each shell alone

    Bar height: RMSECV of PLS on that shell only, at its own lowest point (the number in the
    bar is its number of LVs). Green: shells in the combination with the lowest RMSECV; red:
    not. Red dashed line: the full grid at the LVs chosen in notebook 08. Green dashed line:
    the lowest combination.
    """)
    return


@app.cell(hide_code=True)
def _(best_step, full_rmsecv, result, shell_names):
    _alone = result["alone"].with_columns(
        shell_name=pl.Series([shell_names[_s] for _s in result["alone"]["shell"]]),
        in_selection=pl.col("shell").is_in(best_step["shells"]),
    )
    _x = alt.X("shell_name:N", sort=shell_names, title="shell (radius from the axis, in cells)")
    _bars = (
        alt.Chart(_alone)
        .mark_bar()
        .encode(
            x=_x,
            y=alt.Y("rmsecv:Q", title="RMSECV", scale=alt.Scale(zero=True)),
            color=alt.Color(
                "in_selection:N",
                scale=alt.Scale(domain=[True, False], range=["#2ca02c", "#b00000"]),
                title="in lowest combination",
            ),
            tooltip=["shell_name", "n_cells", "n_lv", alt.Tooltip("rmsecv:Q", format=".4f")],
        )
    )
    _labels = (
        alt.Chart(_alone)
        .mark_text(dy=-8, fontSize=11)
        .encode(x=_x, y=alt.value(300), text="n_lv:Q")
    )
    _lines = pl.DataFrame(
        {
            "rmsecv": [full_rmsecv, best_step["rmsecv"]],
            "line": ["full grid", "lowest combination"],
        }
    )
    _rules = (
        alt.Chart(_lines)
        .mark_rule(strokeDash=[6, 4], strokeWidth=2)
        .encode(
            y="rmsecv:Q",
            color=alt.Color(
                "line:N",
                scale=alt.Scale(
                    domain=["full grid", "lowest combination"], range=["#ff0000", "#2ca02c"]
                ),
                title="",
            ),
        )
    )
    (_bars + _labels + _rules).resolve_scale(color="independent").properties(
        width=620, height=320
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2. Forward path

    RMSECV after each step; the label is the shell added at that step. The last step holds
    every shell, so it equals the full grid at its own lowest point.
    """)
    return


@app.cell(hide_code=True)
def _(path, shell_names):
    _path = path.with_columns(
        added_name=pl.Series([shell_names[_s] for _s in path["added"]]),
        shells_text=pl.col("shells").cast(pl.List(pl.String)).list.join(", "),
    )
    _line = (
        alt.Chart(_path)
        .mark_line(point=True)
        .encode(
            x=alt.X("step:O", title="number of shells in the model"),
            y=alt.Y("rmsecv:Q", title="RMSECV", scale=alt.Scale(zero=False)),
            tooltip=["step", "added_name", "shells_text", "n_lv", alt.Tooltip("rmsecv:Q", format=".4f")],
        )
    )
    _text = _line.mark_text(dy=-10, fontSize=10).encode(text="added:N")
    mo.vstack([(_line + _text).properties(width=620, height=260), _path.drop("added_name")])
    return


if __name__ == "__main__":
    app.run()
