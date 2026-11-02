import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title="09 iPLS on layers")

with app.setup(hide_code=True):
    import altair as alt
    import marimo as mo
    import numpy as np
    import polars as pl

    from pearlatent.config import load_config
    from pearlatent.feature_sets import load_features
    from pearlatent.fruit_table import drop_faulty_volumes, open_fruit
    from pearlatent.ipls import (
        forward_ipls,
        height_bin_labels,
        interval_rmsecv,
        square_ring_labels,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 09 Interval PLS on layers around the stem–calyx axis, then on height

    **Question.** Which layers of the fruit, from the outside of the grid to the axis, carry
    the information for PLS-DA on `binary_23`; and within those layers, which heights?

    **Method** (Ruben, 2 November 2026; `pearlatent.ipls`). No computed radius: the layers are
    square rings of whole cells around the grid axis (the stem–calyx axis after alignment),
    one cell thick, over the full height. Layer 0 is the outer edge of the 32 × 32
    cross-section (124 columns), layer 1 the edge of the 30 × 30 square inside it, and so on
    to layer 15, the 2 × 2 columns at the centre. One cell is about 2.5 mm.

    - **Stage 1, layers.** Every layer alone, and forward iPLS over the 16 layers: at each
      step the layer that lowers RMSECV most is added. The combination with the lowest
      RMSECV is kept.
    - **Stage 2, height.** The cells of the kept layers are cut into 16 height bins of 2
      slices (slice 0 at the calyx end); every bin alone and forward iPLS over the bins.

    Every model: PLS on those cells only, the same 20 box partitions as notebook 08, centring
    only, RMSECV at the lowest point of its curve (1 to 15 LVs).

    **Caveat.** The RMSECV of a chosen combination is optimistic: the layers and bins are
    chosen on the same cross-validation that scores them, and the more intervals are tried,
    the larger that effect. The sealed test boxes give the independent check.

    **Inputs.** As notebook 08. **Outputs.** Figures and tables only.

    **Status.** draft (2 November 2026). The first version of 2 November used circular shells
    of equal cell count (`shell_labels`); replaced by these layers at Ruben's request.
    """)
    return


@app.cell
def _():
    cfg = load_config()
    layers = square_ring_labels()
    _columns = np.bincount(layers[:, :, 0].ravel())
    layer_names = [
        f"{_k}: {32 - 2 * _k}×{32 - 2 * _k} edge ({_columns[_k]} col.)" for _k in range(16)
    ]
    print("columns per layer, outside to centre:", _columns.tolist())
    return cfg, layer_names, layers


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
    # uv run python notebooks/09_ipls_layers.py -- --representation ct_32
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
def run_layers(representation: str, fruit_ids: tuple) -> dict:
    """Stage 1 (layers) and stage 2 (height bins of 2 slices in the kept layers).

    In: the feature set name and the fruit (in table order). Out: a dict with the tables
    `layers_alone`, `layers_path`, `heights_alone`, `heights_path` of `forward_ipls`, the
    kept layers, and `full_curve` (RMSECV for 1 to 15 LVs on all cells). Cached on disk.
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

    layers = square_ring_labels()
    layers_alone, layers_path = forward_ipls(x, y, boxes, folds, layers)
    kept_layers = layers_path.sort("rmsecv")["shells"][0].to_list()

    heights = height_bin_labels(np.isin(layers, kept_layers), slices_per_bin=2)
    heights_alone, heights_path = forward_ipls(x, y, boxes, folds, heights)
    return {
        "layers_alone": layers_alone,
        "layers_path": layers_path,
        "kept_layers": kept_layers,
        "heights_alone": heights_alone,
        "heights_path": heights_path,
        "full_curve": interval_rmsecv(x, y, boxes, folds),
    }


@app.cell(hide_code=True)
def _(fruit, representation_choice, run_ipls):
    mo.stop(
        mo.running_in_notebook() and not run_ipls.value,
        mo.md("_Press the button to run iPLS._"),
    )
    result = run_layers(representation_choice.value, tuple(fruit["fruit_id"].to_list()))
    return (result,)


@app.cell(hide_code=True)
def _(FULL_GRID_LV, representation_choice, result):
    full_lv = FULL_GRID_LV[representation_choice.value]
    full_rmsecv = float(result["full_curve"][full_lv - 1])
    best_layers = result["layers_path"].sort("rmsecv").row(0, named=True)
    best_heights = result["heights_path"].sort("rmsecv").row(0, named=True)
    # Printed too, so the script run shows the result (all rows).
    pl.Config.set_tbl_rows(-1)
    pl.Config.set_fmt_table_cell_list_len(-1)
    print(f"full grid {full_lv} LVs: {full_rmsecv:.4f}")
    print(result["layers_alone"])
    print(result["layers_path"])
    print(f"lowest layers: {best_layers}")
    print(result["heights_alone"])
    print(result["heights_path"])
    print(f"lowest height bins: {best_heights}")
    mo.md(
        f"**Full grid** ({full_lv} LVs): RMSECV {full_rmsecv:.4f}. "
        f"**Stage 1, lowest layers** {best_layers['shells']} ({best_layers['n_lv']} LVs): "
        f"{best_layers['rmsecv']:.4f}. "
        f"**Stage 2, lowest height bins** {best_heights['shells']} in those layers "
        f"({best_heights['n_lv']} LVs): {best_heights['rmsecv']:.4f}."
    )
    return best_heights, best_layers, full_rmsecv


@app.function
def ipls_bar_chart(
    alone: pl.DataFrame,
    names: list[str],
    selected: list[int],
    full_rmsecv: float,
    best_rmsecv: float,
    title: str,
):
    """Bar per interval: RMSECV alone, LVs as text; green if in the lowest combination.

    Dashed lines: red for the full grid at the LVs of notebook 08, green for the lowest
    combination. The layout follows the forward iPLS plot of SOLO.
    """
    data = alone.with_columns(
        name=pl.Series([names[s] for s in alone["shell"]]),
        in_selection=pl.col("shell").is_in(selected),
    )
    x = alt.X("name:N", sort=names, title=title, axis=alt.Axis(labelAngle=-60))
    bars = alt.Chart(data).mark_bar().encode(
        x=x,
        y=alt.Y("rmsecv:Q", title="RMSECV", scale=alt.Scale(zero=True)),
        color=alt.Color(
            "in_selection:N",
            scale=alt.Scale(domain=[True, False], range=["#2ca02c", "#b00000"]),
            title="in lowest combination",
        ),
        tooltip=["name", "n_cells", "n_lv", alt.Tooltip("rmsecv:Q", format=".4f")],
    )
    labels = alt.Chart(data).mark_text(dy=-8, fontSize=11, color="white").encode(
        x=x, y=alt.value(300), text="n_lv:Q"
    )
    lines = pl.DataFrame(
        {"rmsecv": [full_rmsecv, best_rmsecv], "line": ["full grid", "lowest combination"]}
    )
    rules = alt.Chart(lines).mark_rule(strokeDash=[6, 4], strokeWidth=2).encode(
        y="rmsecv:Q",
        color=alt.Color(
            "line:N",
            scale=alt.Scale(
                domain=["full grid", "lowest combination"], range=["#ff0000", "#2ca02c"]
            ),
            title="",
        ),
    )
    return (
        (bars + labels + rules)
        .resolve_scale(color="independent")
        .properties(width=640, height=320)
    )


@app.function
def forward_path_chart(path: pl.DataFrame):
    """RMSECV after each forward step, labelled with the interval added."""
    data = path.with_columns(
        shells_text=pl.col("shells").cast(pl.List(pl.String)).list.join(", ")
    )
    line = alt.Chart(data).mark_line(point=True).encode(
        x=alt.X("step:O", title="number of intervals in the model"),
        y=alt.Y("rmsecv:Q", title="RMSECV", scale=alt.Scale(zero=False)),
        tooltip=["step", "added", "shells_text", "n_lv", alt.Tooltip("rmsecv:Q", format=".4f")],
    )
    text = line.mark_text(dy=-10, fontSize=10).encode(text="added:N")
    return (line + text).properties(width=640, height=240)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1. Stage 1: layers, outside (0) to centre (15)

    Bar: RMSECV of PLS on that layer only, at its own lowest point (number in the bar: its
    LVs). Green: layers in the combination with the lowest RMSECV. Below: the forward path.
    """)
    return


@app.cell(hide_code=True)
def _(best_layers, full_rmsecv, layer_names, result):
    mo.vstack(
        [
            ipls_bar_chart(
                result["layers_alone"],
                layer_names,
                best_layers["shells"],
                full_rmsecv,
                best_layers["rmsecv"],
                "layer (0 = outer edge of the grid, 15 = 2 × 2 columns at the axis)",
            ),
            forward_path_chart(result["layers_path"]),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2. Stage 2: height bins of 2 slices within the kept layers

    Bin 0 holds slices 0–1 at the calyx end, bin 15 slices 30–31 at the stalk end. Green
    dashed line: the lowest combination of bins; red: the full grid.
    """)
    return


@app.cell(hide_code=True)
def _(best_heights, full_rmsecv, result):
    _names = [f"{_k}: slices {2 * _k}–{2 * _k + 1}" for _k in range(16)]
    mo.vstack(
        [
            mo.md(f"Kept layers from stage 1: {result['kept_layers']}"),
            ipls_bar_chart(
                result["heights_alone"],
                _names,
                best_heights["shells"],
                full_rmsecv,
                best_heights["rmsecv"],
                "height bin (0 = calyx end, 15 = stalk end)",
            ),
            forward_path_chart(result["heights_path"]),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
