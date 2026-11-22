import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title="13 iPLS on the height-radius bins")

with app.setup(hide_code=True):
    import json

    import altair as alt
    import marimo as mo
    import numpy as np
    import polars as pl

    from pearlatent.config import load_config
    from pearlatent.feature_sets import load_features
    from pearlatent.fruit_table import drop_faulty_volumes, open_fruit
    from pearlatent.ipls import forward_ipls, interval_rmsecv
    from pearlatent.plsda import (
        box_bootstrap_auc,
        cross_validated_predictions,
        paired_box_bootstrap_auc_difference,
    )
    from pearlatent.runs import PLSDA_BINARY23_RUN_DIRS
    from sklearn.metrics import roc_auc_score
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

    **Mean + SD rows** (22 November 2026, Ruben): with `hugo_vae_hr_sd` and `ct_32_hr_sd` an
    interval holds the means and the SDs of its bins (a height interval 16 numbers, a depth
    interval 40).

    **Section 3** (22 November 2026, Ruben): build one model on a chosen selection. First the
    number of height intervals along the forward path, then one forward pass over the 8 rings
    inside those heights, then the model on the chosen bins, scored as every row.

    **Inputs.** As notebook 08 (feature sets `hugo_vae_hr`, `ct_32_hr`, `hugo_vae_hr_sd`,
    `ct_32_hr_sd`) and the saved `binary_23` runs of the same rows (`PLSDA_BINARY23_RUN_DIRS`).
    **Outputs.** Figures and tables only.

    **Status.** draft, 7 November 2026; section 3 added 22 November 2026.
    """)
    return


@app.cell
def _():
    cfg = load_config()
    ROWS = {
        "hugo_vae_hr_sd": "Hugo's latent, height-radius mean + SD",
        "ct_32_hr_sd": "CT at 32³, height-radius mean + SD",
        "hugo_vae_hr": "Hugo's latent, height-radius",
        "ct_32_hr": "CT at 32³, height-radius",
    }
    return ROWS, cfg


@app.cell(hide_code=True)
def _(ROWS):
    row_choice = mo.ui.dropdown(
        options={_v: _k for _k, _v in ROWS.items()},
        value=ROWS["hugo_vae_hr_sd"],
        label="Row",
    )
    interval_choice = mo.ui.dropdown(
        options=["height", "depth"], value="height", label="Intervals"
    )
    run_button = mo.ui.run_button(label="Run iPLS (slow the first time, then cached)")
    mo.hstack([row_choice, interval_choice, run_button], justify="start")
    return interval_choice, row_choice, run_button


@app.cell(hide_code=True)
def _(cfg):
    _open_2526 = open_fruit(cfg).filter(pl.col("season") == "2526")
    fruit = drop_faulty_volumes(
        _open_2526, cfg.path("derived_dir") / "volume_decisions.csv"
    ).with_columns(binary_23=(pl.col("defective") >= 2).cast(pl.Int8))
    print(f"open 2526 fruit {_open_2526.height}, after faulty volumes {fruit.height}")
    return (fruit,)


@app.function(hide_code=True)
def bin_positions(n_features: int) -> tuple[np.ndarray, np.ndarray]:
    """Height bin and ring of every feature of a height-radius row.

    In: the number of features, 160 (means) or 320 (means, then SDs in the same bin order;
    `pearlatent.height_radius.height_radius_features`). Out: two integer arrays of that length.
    Feature index of a mean = 8 × height bin + ring; an SD sits 160 places further.
    """
    n_bins = N_LATENT_HEIGHT_BINS * N_LATENT_RADIUS_BINS
    bin_index = np.arange(n_features) % n_bins
    return bin_index // N_LATENT_RADIUS_BINS, bin_index % N_LATENT_RADIUS_BINS


@app.function(hide_code=True)
@mo.persistent_cache
def run_ipls(row: str, interval_kind: str, fruit_ids: tuple) -> dict:
    """All-feature RMSECV curve, each interval alone, and the forward path.

    In: the feature set, "height" or "depth", the fruit in table order. Out: a dict with the
    curve of all features and the two tables of `forward_ipls`. Cached on disk.
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
    height, ring = bin_positions(x.shape[1])
    labels = height if interval_kind == "height" else ring
    alone, path = forward_ipls(x, y, boxes, folds, labels)
    return {"all": interval_rmsecv(x, y, boxes, folds), "alone": alone, "path": path}


@app.cell(hide_code=True)
def _(fruit, interval_choice, row_choice, run_button):
    # In script mode the button is never pressed; iPLS then runs directly.
    mo.stop(
        mo.running_in_notebook() and not run_button.value,
        mo.md("_Press the button to run iPLS._"),
    )
    result = run_ipls(
        row_choice.value, interval_choice.value, tuple(fruit["fruit_id"].to_list())
    )
    _all = result["all"]
    print(
        f"all features: lowest RMSECV {_all.min():.3f} at {int(np.argmin(_all)) + 1} LVs;"
        f" at 3 LVs (Ruben's choice in notebook 08) {_all[2]:.3f}"
    )
    return (result,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1. Each interval alone

    Bar height: the lowest RMSECV of PLS on that interval only; the number on the bar is its
    LVs. Dashed line: all features of the row at 3 LVs (Ruben's choice). Dotted line: no model
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


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 3. A model on a chosen selection: heights, then rings

    Three steps, each chosen by Ruben (22 November 2026):

    1. **Heights.** The forward path over the 20 height intervals (section 2 with
       "height"); choose how many intervals to keep. The first k intervals of the path are
       kept.
    2. **Rings.** Inside the kept heights, one forward pass over the 8 rings (each ring =
       its bins at the kept heights, means and SDs); choose how many rings to keep.
    3. **Model.** PLS-DA on the kept bins (kept heights × kept rings), the same 20 × 4 box
       folds, centring only; RMSECV curve, LVs chosen at the elbow, AUC with the box
       interval and the paired difference against the saved row with all features.

    **Caveat.** Heights, rings and LVs are all chosen on the cross-validation that scores
    them, so the AUC here is optimistic, more so than for a single row. It shows how much
    of the information a small region carries, not a better model; only the sealed boxes can
    test a selection.
    """)
    return


@app.cell(hide_code=True)
def _(fruit, row_choice, run_button):
    mo.stop(
        mo.running_in_notebook() and not run_button.value,
        mo.md("_Press the button above to run iPLS._"),
    )
    fruit_ids = tuple(fruit["fruit_id"].to_list())
    height_path = run_ipls(row_choice.value, "height", fruit_ids)["path"]
    _best = int(height_path["rmsecv"].arg_min()) + 1
    n_heights_choice = mo.ui.dropdown(
        options={
            f"{_k} (RMSECV {_r:.3f})": _k
            for _k, _r in zip(height_path["step"], height_path["rmsecv"])
        },
        value=f"{_best} (RMSECV {height_path['rmsecv'][_best - 1]:.3f})",
        label="Step 1: number of height intervals (default: lowest point of the path)",
    )
    n_heights_choice
    return fruit_ids, height_path, n_heights_choice


@app.function(hide_code=True)
@mo.persistent_cache
def run_ring_stage(row: str, heights: tuple, fruit_ids: tuple) -> dict:
    """Forward iPLS over the 8 rings, using only the bins at the kept heights.

    In: the feature set, the kept height bins, the fruit in table order. Out: the two
    tables of `forward_ipls` (each ring alone, the forward path). Cached on disk.
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
    height, ring = bin_positions(x.shape[1])
    # Features at other heights get label -1: `forward_ipls` never uses them.
    labels = np.where(np.isin(height, heights), ring, -1)
    alone, path = forward_ipls(x, y, fruit["box"].to_numpy(), folds, labels)
    return {"alone": alone, "path": path}


@app.cell(hide_code=True)
def _(fruit_ids, height_path, n_heights_choice, row_choice):
    kept_heights = tuple(
        sorted(height_path["shells"][n_heights_choice.value - 1].to_list())
    )
    ring_stage = run_ring_stage(row_choice.value, kept_heights, fruit_ids)
    _path = ring_stage["path"]
    _best = int(_path["rmsecv"].arg_min()) + 1
    n_rings_choice = mo.ui.dropdown(
        options={
            f"{_k} (RMSECV {_r:.3f})": _k
            for _k, _r in zip(_path["step"], _path["rmsecv"])
        },
        value=f"{_best} (RMSECV {_path['rmsecv'][_best - 1]:.3f})",
        label="Step 2: number of rings (default: lowest point of the path)",
    )
    mo.vstack(
        [
            mo.md(
                f"Kept heights (0 = calyx end): **{list(kept_heights)}**. Ring path inside"
                " them (0 = axis, 7 = skin):"
            ),
            _path.rename({"shells": "rings"}).with_columns(pl.col("rmsecv").round(3)),
            n_rings_choice,
        ]
    )
    return kept_heights, n_rings_choice, ring_stage


@app.function(hide_code=True)
@mo.persistent_cache
def run_selection_cv(row: str, heights: tuple, rings: tuple, fruit_ids: tuple) -> dict:
    """Out-of-fold predictions of PLS-DA on the kept bins (means and SDs).

    In: the feature set, kept heights and rings, the fruit in table order. Out: the
    predictions (repeats, fruit, LVs) and the number of features. Cached on disk.
    """
    cfg = load_config()
    fruit = drop_faulty_volumes(
        open_fruit(cfg).filter(pl.col("season") == "2526"),
        cfg.path("derived_dir") / "volume_decisions.csv",
    ).filter(pl.col("fruit_id").is_in(list(fruit_ids)))
    assert tuple(fruit["fruit_id"].to_list()) == fruit_ids, "fruit order changed"
    folds = pl.read_csv(cfg.path("derived_dir") / "outer_folds_20" / "outer_folds.csv")
    x = load_features(cfg, row, list(fruit_ids))
    height, ring = bin_positions(x.shape[1])
    kept = np.isin(height, heights) & np.isin(ring, rings)
    y = (fruit["defective"] >= 2).cast(pl.Float64).to_numpy()
    # No more LVs than features (a small selection can have fewer than 15).
    predictions = cross_validated_predictions(
        x[:, kept],
        y,
        fruit["box"].to_numpy(),
        folds,
        scale=False,
        max_components=min(15, int(kept.sum())),
    )
    return {"predictions": predictions, "n_features": int(kept.sum())}


@app.cell(hide_code=True)
def _(fruit, fruit_ids, kept_heights, n_rings_choice, ring_stage, row_choice):
    kept_rings = tuple(
        sorted(ring_stage["path"]["shells"][n_rings_choice.value - 1].to_list())
    )
    selection = run_selection_cv(row_choice.value, kept_heights, kept_rings, fruit_ids)
    _y = (fruit["defective"] >= 2).cast(pl.Float64).to_numpy()
    _errors = (selection["predictions"] - _y[None, :, None]) ** 2
    # RMSECV per LV: square root of the PRESS averaged over the repeats, divided by n.
    _rmsecv = np.sqrt(_errors.sum(axis=1).mean(axis=0) / len(_y))
    selection_curve = pl.DataFrame(
        {"n_lv": np.arange(1, len(_rmsecv) + 1), "rmsecv": _rmsecv}
    )
    mo.vstack(
        [
            mo.md(
                f"Kept rings: **{list(kept_rings)}**; {selection['n_features']} features"
                f" ({len(kept_heights)} heights × {len(kept_rings)} rings"
                f" × {selection['n_features'] // (len(kept_heights) * len(kept_rings))}"
                " numbers per bin)."
            ),
            alt.Chart(selection_curve)
            .mark_line(point=True)
            .encode(
                x=alt.X("n_lv:O", title="LVs"),
                y=alt.Y("rmsecv:Q", title="RMSECV", scale=alt.Scale(zero=False)),
                tooltip=["n_lv", alt.Tooltip("rmsecv:Q", format=".3f")],
            )
            .properties(width=480, height=220),
        ]
    )
    return kept_rings, selection, selection_curve


@app.cell(hide_code=True)
def _(selection_curve):
    lv_choice = mo.ui.dropdown(
        options=["not chosen"] + [str(_k) for _k in selection_curve["n_lv"]],
        value="not chosen",
        label="Step 3: number of LVs (Ruben's choice at the elbow)",
    )
    lv_choice
    return (lv_choice,)


@app.cell(hide_code=True)
def _(cfg, fruit, kept_heights, kept_rings, lv_choice, row_choice, selection):
    mo.stop(lv_choice.value == "not chosen", mo.md("_Choose the number of LVs._"))
    _n_lv = int(lv_choice.value)
    _y = (fruit["defective"] >= 2).cast(pl.Int8).to_numpy()
    _boxes = fruit["box"].to_numpy()
    _selected = selection["predictions"][:, :, _n_lv - 1].mean(axis=0)
    # The saved row with all features, same fruit in the same order.
    _run = cfg.path("results_dir") / PLSDA_BINARY23_RUN_DIRS[row_choice.value]
    _all_lv = json.loads((_run / "run.json").read_text())["params"]["n_lv_chosen"]
    _all = (
        fruit.select("fruit_id")
        .join(
            pl.read_parquet(_run / "predictions.parquet").select(
                "fruit_id", "prediction"
            ),
            on="fruit_id",
            how="left",
            validate="1:1",
        )["prediction"]
        .to_numpy()
    )
    _rows = []
    for _name, _p, _lv in [
        (
            f"selection: heights {list(kept_heights)}, rings {list(kept_rings)}",
            _selected,
            _n_lv,
        ),
        (f"all features (saved run {_run.name})", _all, _all_lv),
    ]:
        _low, _high, _ = box_bootstrap_auc(_p, _y, _boxes)
        _rows.append(
            {
                "model": _name,
                "n_lv": _lv,
                "auc": round(float(roc_auc_score(_y, _p)), 3),
                "auc_95": f"{_low:.3f} to {_high:.3f}",
            }
        )
    _paired = paired_box_bootstrap_auc_difference(_selected, _all, _y, _boxes)
    mo.vstack(
        [
            pl.DataFrame(_rows),
            mo.md(
                f"AUC selection − all features: **{_paired['difference']:+.3f}**"
                f" ({_paired['lower_95']:+.3f} to {_paired['upper_95']:+.3f}), paired on"
                f" the same {len(_y)} fruit and box draws. Optimistic (see the caveat)."
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
