import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title="13 iPLS on the height-radius bins")

with app.setup(hide_code=True):
    import altair as alt
    import marimo as mo
    import numpy as np
    import polars as pl

    from pearlatent.config import load_config
    from pearlatent.feature_sets import load_features
    from pearlatent.fruit_table import drop_faulty_volumes, open_fruit
    from pearlatent.ipls import forward_ipls, interval_rmsecv
    from pearlatent.step0 import N_LATENT_HEIGHT_BINS, N_LATENT_RADIUS_BINS


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 13 Interval PLS on the height-radius bins

    **Question.** Which heights of the fruit (and, as a second view, which depths below the
    skin) carry the information of the height-radius PLS-DA models on `binary_23`?

    **Method** (Ruben, 7 November 2026; forward iPLS after Nørgaard et al. 2000, as in SOLO and
    notebook 09, `pearlatent.ipls.forward_ipls`). The 160 features of a height-radius row
    (20 height bins × 8 radius bins, `docs/methods_plsda.md` section 10) are cut into
    non-overlapping intervals:

    - **Heights:** 20 intervals, one per height bin (its 8 radius bins). Bin 0 is the calyx
      end of the body, bin 19 the stalk end.
    - **Depths:** 8 intervals, one per radius bin (its 20 height bins). Bin 0 is the axis, bin
      7 the skin.

    Every interval alone gets its RMSECV curve (1 to 15 LVs, or fewer when the interval has
    fewer features), with the same 20 box partitions as notebook 08, centring only. Forward
    mode: the interval with the lowest RMSECV is kept; then each remaining interval is tried
    beside it and the best pair is kept; and so on until all intervals are in. The
    combination with the lowest RMSECV along this path is the selection. Each model takes the
    LVs at the lowest point of its own curve (as notebook 09).

    One difference from SOLO: SOLO (`numintervals = inf`) stops at the first step that does
    not lower RMSECV; here the path runs to the end so it can be drawn, and the lowest point of
    the whole path is taken.

    **Caveat.** The RMSECV of a chosen combination is optimistic: the intervals are chosen on
    the same cross-validation that scores them. Read the bars as where the information sits,
    not as a better model. The sealed test boxes give the independent check.

    **Inputs.** As notebook 08 (feature sets `hugo_vae_hr`, `ct_32_hr`). **Outputs.** Figures
    and tables only.

    **Status.** draft, 7 November 2026.
    """)
    return


@app.cell
def _():
    cfg = load_config()
    ROWS = {
        "hugo_vae_hr": "Hugo's latent, height-radius",
        "ct_32_hr": "CT at 32³, height-radius",
    }
    return ROWS, cfg


@app.cell(hide_code=True)
def _(ROWS):
    row_choice = mo.ui.dropdown(
        options={_v: _k for _k, _v in ROWS.items()},
        value=ROWS["hugo_vae_hr"],
        label="Row",
    )
    interval_choice = mo.ui.dropdown(
        options=["height", "depth"], value="height", label="Intervals"
    )
    mo.hstack([row_choice, interval_choice], justify="start")
    return interval_choice, row_choice


@app.cell(hide_code=True)
def _(cfg):
    _open_2526 = open_fruit(cfg).filter(pl.col("season") == "2526")
    fruit = drop_faulty_volumes(
        _open_2526, cfg.path("derived_dir") / "volume_decisions.csv"
    ).with_columns(binary_23=(pl.col("defective") >= 2).cast(pl.Int8))
    print(f"open 2526 fruit {_open_2526.height}, after faulty volumes {fruit.height}")
    return (fruit,)


@app.function(hide_code=True)
@mo.persistent_cache
def run_ipls(row: str, interval_kind: str, fruit_ids: tuple) -> dict:
    """All-feature RMSECV curve, each interval alone, and the forward path.

    In: the feature set, "height" or "depth", the fruit in table order. Out: a dict with the
    curve of all 160 features and the two tables of `forward_ipls`. Cached on disk.
    """
    cfg = load_config()
    fruit = drop_faulty_volumes(
        open_fruit(cfg).filter(pl.col("season") == "2526"),
        cfg.path("derived_dir") / "volume_decisions.csv",
    ).filter(pl.col("fruit_id").is_in(list(fruit_ids)))
    assert tuple(fruit["fruit_id"].to_list()) == fruit_ids, "fruit order changed"
    folds = pl.read_csv(cfg.path("derived_dir") / "outer_folds_20" / "outer_folds.csv")
    x = load_features(cfg, row, list(fruit_ids))
    y = (fruit["defective"] >= 2).cast(pl.Float64).to_numpy()
    boxes = fruit["box"].to_numpy()
    feature = np.arange(x.shape[1])
    labels = (
        feature // N_LATENT_RADIUS_BINS
        if interval_kind == "height"
        else feature % N_LATENT_RADIUS_BINS
    )
    alone, path = forward_ipls(x, y, boxes, folds, labels)
    return {"all": interval_rmsecv(x, y, boxes, folds), "alone": alone, "path": path}


@app.cell(hide_code=True)
def _(fruit, interval_choice, row_choice):
    result = run_ipls(
        row_choice.value, interval_choice.value, tuple(fruit["fruit_id"].to_list())
    )
    _all = result["all"]
    print(
        f"all 160 features: lowest RMSECV {_all.min():.3f} at {int(np.argmin(_all)) + 1} LVs;"
        f" at 3 LVs (Ruben's choice in notebook 08) {_all[2]:.3f}"
    )
    return (result,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1. Each interval alone

    Bar height: the lowest RMSECV of PLS on that interval only; the number on the bar is its
    LVs. Dashed line: all 160 features at 3 LVs (Ruben's choice). Dotted line: no model
    (predicting the share of non-consumable fruit for every fruit). A low bar is an interval
    that alone carries much of the information.
    """)
    return


@app.cell(hide_code=True)
def _(fruit, interval_choice, result):
    _no_model = float((fruit["defective"] >= 2).cast(pl.Float64).std(ddof=0))
    _alone = result["alone"].rename({"shell": "interval"})
    _bars = (
        alt.Chart(_alone)
        .mark_bar()
        .encode(
            x=alt.X("interval:O", title=f"{interval_choice.value} bin"),
            y=alt.Y("rmsecv:Q", title="RMSECV", scale=alt.Scale(zero=False)),
            tooltip=[
                "interval",
                "n_cells",
                "n_lv",
                alt.Tooltip("rmsecv:Q", format=".3f"),
            ],
        )
    )
    _labels = _bars.mark_text(dy=-6).encode(text="n_lv:Q")
    _lines = pl.DataFrame(
        {
            "y": [float(result["all"][2]), _no_model],
            "line": ["all features, 3 LVs", "no model"],
        }
    )
    _rules = (
        alt.Chart(_lines)
        .mark_rule()
        .encode(
            y="y:Q",
            strokeDash=alt.StrokeDash("line:N", title=None),
        )
    )
    mo.vstack(
        [
            alt.layer(_bars, _labels, _rules).properties(width=560, height=280),
            _alone.with_columns(pl.col("rmsecv").round(3)),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2. Forward path

    Step k: the k intervals in the model after the k-th addition and their RMSECV. The lowest
    point of the path is the selection.
    """)
    return


@app.cell(hide_code=True)
def _(result):
    _path = result["path"].rename({"shells": "intervals"})
    _best = int(_path["rmsecv"].arg_min())
    _chart = (
        alt.Chart(
            _path.with_columns(pl.col("added").cast(pl.Utf8).alias("added_label"))
        )
        .mark_line(point=True)
        .encode(
            x=alt.X("step:O", title="number of intervals in the model"),
            y=alt.Y("rmsecv:Q", title="RMSECV", scale=alt.Scale(zero=False)),
            tooltip=["step", "added", "n_lv", alt.Tooltip("rmsecv:Q", format=".3f")],
        )
    )
    mo.vstack(
        [
            _chart.properties(width=560, height=260),
            mo.md(
                f"**Selection:** intervals {_path['intervals'][_best].to_list()},"
                f" {_path['n_lv'][_best]} LVs, RMSECV {_path['rmsecv'][_best]:.3f}"
                f" (optimistic, see the caveat above)."
            ),
            _path.with_columns(pl.col("rmsecv").round(3)),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
