import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title="13 iPLS on the height-radius bins")

with app.setup(hide_code=True):
    import json

    import altair as alt
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl

    from pearlatent.config import load_config
    from pearlatent.feature_sets import load_features
    from pearlatent.height_radius import RADIUS_METHOD
    from pearlatent.nuisance import box_bootstrap, healthy_fruit_correlations
    from pearlatent.fruit_table import drop_faulty_volumes, open_fruit
    from pearlatent.ipls import forward_ipls, interval_rmsecv
    from pearlatent.plsda import (
        CUTOFF,
        box_bootstrap_auc,
        classification_scores,
        cross_validated_predictions,
        paired_box_bootstrap_auc_difference,
        youden_cutoff,
    )
    from pearlatent.runs import (
        ENCODING_RUN_DIR,
        PLSDA_BINARY23_RUN_DIRS,
        new_run_dir,
        write_run_record,
    )
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.metrics import roc_auc_score
    from pearlatent.step0 import N_LATENT_HEIGHT_BINS, N_LATENT_RADIUS_BINS, cell_bins


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
    chosen_lv = int(lv_choice.value)
    _y = (fruit["defective"] >= 2).cast(pl.Int8).to_numpy()
    _boxes = fruit["box"].to_numpy()
    # Mean of the 20 repeats at the chosen LVs, one value per fruit.
    selected_prediction = selection["predictions"][:, :, chosen_lv - 1].mean(axis=0)
    # The saved row with all features, same fruit in the same order.
    all_run = cfg.path("results_dir") / PLSDA_BINARY23_RUN_DIRS[row_choice.value]
    all_lv = json.loads((all_run / "run.json").read_text())["params"]["n_lv_chosen"]
    all_prediction = (
        fruit.select("fruit_id")
        .join(
            pl.read_parquet(all_run / "predictions.parquet").select(
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
            selected_prediction,
            chosen_lv,
        ),
        (f"all features (saved run {all_run.name})", all_prediction, all_lv),
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
    paired_selection = paired_box_bootstrap_auc_difference(
        selected_prediction, all_prediction, _y, _boxes
    )
    mo.vstack(
        [
            pl.DataFrame(_rows),
            mo.md(
                f"AUC selection − all features: **{paired_selection['difference']:+.3f}**"
                f" ({paired_selection['lower_95']:+.3f} to"
                f" {paired_selection['upper_95']:+.3f}), paired on the same {len(_y)} fruit"
                " and box draws. Optimistic (see the caveat)."
            ),
        ]
    )
    return (
        all_lv,
        all_prediction,
        all_run,
        chosen_lv,
        paired_selection,
        selected_prediction,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 4. More measures of the selection, beside the row with all features

    As notebooks 10 and 21: recall and precision at the fixed cut-off 0.5 and at the Youden
    cut-off (chosen on these predictions, so slightly optimistic); AUC of grade 0 against
    each defect group (charter rule 5); check C, the Spearman correlation of the score (minus
    its box median) with stalk length and body size on the healthy fruit, 1,000 box draws.
    """)
    return


@app.cell(hide_code=True)
def _(all_prediction, fruit, selected_prediction):
    _y = (fruit["defective"] >= 2).cast(pl.Int8).to_numpy()
    _grade = fruit["defective"].to_numpy()
    _rows = []
    selection_scores = {}
    for _model, _p in (
        ("selection", selected_prediction),
        ("all features", all_prediction),
    ):
        _youden = youden_cutoff(_p, _y)
        for _rule, _cut in (("0.5", CUTOFF), ("Youden", _youden)):
            _s = classification_scores(_p, _y, _grade, cutoff=_cut)
            selection_scores[f"{_model}, {_rule}"] = {"cutoff": _cut, **_s}
            _rows.append(
                {
                    "model": _model,
                    "cut-off": f"{_rule} ({_cut:.3f})",
                    "recall": round(_s["recall"], 3),
                    "precision": round(_s["precision"], 3),
                }
            )
    pl.DataFrame(_rows)
    return (selection_scores,)


@app.cell(hide_code=True)
def _(all_prediction, fruit, selected_prediction):
    _groups = {
        "browning only 1": (pl.col("browning") == 1) & (pl.col("cavity") == 0),
        "browning only 2": (pl.col("browning") == 2) & (pl.col("cavity") == 0),
        "browning only 3": (pl.col("browning") == 3) & (pl.col("cavity") == 0),
        "cavity 2 or 3": pl.col("cavity") >= 2,
        "any grade 1": pl.col("defective") == 1,
    }
    _table = fruit.with_columns(
        selection=selected_prediction, all_features=all_prediction
    )
    _healthy = _table.filter(pl.col("defective") == 0)
    _rows = []
    for _name, _condition in _groups.items():
        _pair = pl.concat(
            [
                _healthy.with_columns(_label=pl.lit(0)),
                _table.filter(_condition).with_columns(_label=pl.lit(1)),
            ]
        )
        _labels = _pair["_label"].to_numpy()
        for _model in ("selection", "all_features"):
            _values = _pair[_model].to_numpy()
            _low, _high, _ = box_bootstrap_auc(
                _values, _labels, _pair["box"].to_numpy()
            )
            _rows.append(
                {
                    "group": _name,
                    "model": _model,
                    "n_group": int(_labels.sum()),
                    "auc": roc_auc_score(_labels, _values),
                    "lower_95": _low,
                    "upper_95": _high,
                }
            )
    selection_auc_by_group = pl.DataFrame(_rows)
    selection_auc_by_group.with_columns(pl.col("auc", "lower_95", "upper_95").round(3))
    return (selection_auc_by_group,)


@app.cell(hide_code=True)
def _(all_prediction, fruit, selected_prediction):
    _rows = []
    for _model, _p in (
        ("selection", selected_prediction),
        ("all features", all_prediction),
    ):
        _scored = fruit.with_columns(score=_p)
        for _measure in ("stalk_length", "body_voxels"):
            _key = f"spearman_{_measure}"
            _low, _high = box_bootstrap(
                lambda t, k=_key: healthy_fruit_correlations(t)[k],
                _scored,
                n_resamples=1000,
            )
            _rows.append(
                {
                    "model": _model,
                    "measure": _measure,
                    "spearman": healthy_fruit_correlations(_scored)[_key],
                    "lower_95": _low,
                    "upper_95": _high,
                }
            )
    selection_check_c = pl.DataFrame(_rows)
    selection_check_c.with_columns(pl.col("spearman", "lower_95", "upper_95").round(3))
    return (selection_check_c,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 5. The weights, mapped back onto the height-radius map and onto the cells

    One PLS-DA fit on all 315 fruit of the 12 open boxes, on the kept bins, at the chosen
    LVs (as every final model). Its regression vector has one weight per kept feature: one
    for the mean and, in a mean + SD row, one for the SD of each kept bin. Left: the weights
    on the 20 × 8 map (height 0 = calyx end at the bottom, ring 0 = axis on the left); grey =
    bin not kept. Right: the same weights drawn on the 32³ body cells of one fruit (long
    section through its body centre), each cell coloured by the weight of the bin it falls in
    (`pearlatent.step0.cell_bins`, the rule of the features). Red = a higher value in that bin
    raises the score (towards non-consumable), blue = lowers it.

    Mean and SD features have different units, so each has its own colour scale. A weight
    says how the model uses a bin given all other kept bins; it is not a test of that bin.
    """)
    return


@app.cell(hide_code=True)
def _(examples_for_cells):
    cell_fruit_choice = mo.ui.dropdown(
        options=examples_for_cells,
        value=examples_for_cells[0],
        label="Fruit for the cells",
    )
    cell_fruit_choice
    return (cell_fruit_choice,)


@app.cell(hide_code=True)
def _(fruit):
    # The four counterfactual fruit first (notebook 17), then the others.
    _first = ["2526_C23", "2526_N17", "2526_K07", "2526_A03"]
    examples_for_cells = [_f for _f in _first if _f in fruit["fruit_id"].to_list()] + [
        _f for _f in fruit["fruit_id"].to_list() if _f not in _first
    ]
    return (examples_for_cells,)


@app.cell(hide_code=True)
def _(
    cell_fruit_choice,
    cfg,
    chosen_lv,
    fruit,
    kept_heights,
    kept_rings,
    row_choice,
):
    _x = load_features(cfg, row_choice.value, fruit["fruit_id"].to_list()).astype(float)
    _height, _ring = bin_positions(_x.shape[1])
    _kept = np.isin(_height, kept_heights) & np.isin(_ring, kept_rings)
    _y = (fruit["defective"] >= 2).cast(pl.Float64).to_numpy()
    _model = PLSRegression(n_components=chosen_lv, scale=False).fit(_x[:, _kept], _y)
    _n_bins = N_LATENT_HEIGHT_BINS * N_LATENT_RADIUS_BINS
    # Weights back on the full feature list (0 for features not kept), then as maps.
    _full = np.zeros(_x.shape[1])
    _full[_kept] = _model.coef_.ravel()
    _kept_map = np.isin(
        np.arange(N_LATENT_HEIGHT_BINS)[:, None], kept_heights
    ) & np.isin(np.arange(N_LATENT_RADIUS_BINS)[None, :], kept_rings)
    weight_maps = {
        "mean": _full[:_n_bins].reshape(N_LATENT_HEIGHT_BINS, N_LATENT_RADIUS_BINS)
    }
    if _x.shape[1] == 2 * _n_bins:
        weight_maps["SD"] = _full[_n_bins:].reshape(
            N_LATENT_HEIGHT_BINS, N_LATENT_RADIUS_BINS
        )

    # The body cells of the chosen fruit, and the bin of every cell.
    _manifest = pl.read_parquet(
        cfg.path("derived_dir") / ENCODING_RUN_DIR / "manifest.parquet"
    ).with_row_index("position")
    _position = _manifest.filter(pl.col("fruit_id") == cell_fruit_choice.value)[
        "position"
    ].item()
    _body = np.asarray(
        np.load(
            cfg.path("derived_dir") / ENCODING_RUN_DIR / "masks.npy", mmap_mode="r"
        )[_position],
        dtype=bool,
    )
    _h_bins, _r_bins = cell_bins(_body, RADIUS_METHOD)
    _centre = int(round(np.nonzero(_body)[0].mean()))  # long section: first axis fixed

    weights_figure, _axes = plt.subplots(
        1,
        2 * len(weight_maps),
        figsize=(4.2 * len(weight_maps) * 2, 4.6),
        squeeze=False,
    )
    for _k, (_name, _map) in enumerate(weight_maps.items()):
        _limit = float(np.abs(_map[_kept_map]).max()) or 1.0
        _shown = np.ma.masked_where(~_kept_map, _map)
        _ax = _axes[0, 2 * _k]
        _ax.set_facecolor("0.85")
        _image = _ax.imshow(
            _shown,
            cmap="RdBu_r",
            vmin=-_limit,
            vmax=_limit,
            origin="lower",
            aspect="auto",
        )
        _ax.set_xlabel("ring (0 = axis, 7 = skin)")
        _ax.set_ylabel("height bin (0 = calyx end)")
        _ax.set_title(f"weights of the bin {_name}", fontsize=10)
        weights_figure.colorbar(_image, ax=_ax, shrink=0.8)

        # Cells: weight of the bin of every body cell in the long section; grey = not kept.
        _section_h = _h_bins[_centre]
        _section_r = _r_bins[_centre]
        _in_body = _section_h >= 0
        _cell_weight = np.full(_section_h.shape, np.nan)
        _cell_weight[_in_body] = _map[_section_h[_in_body], _section_r[_in_body]]
        _cell_kept = np.zeros(_section_h.shape, dtype=bool)
        _cell_kept[_in_body] = _kept_map[_section_h[_in_body], _section_r[_in_body]]
        _ax = _axes[0, 2 * _k + 1]
        _background = np.where(_in_body, 0.85, 0.0)
        _ax.imshow(_background.T, cmap="gray", vmin=0, vmax=1, origin="lower")
        _ax.imshow(
            np.ma.masked_where(~_cell_kept, _cell_weight).T,
            cmap="RdBu_r",
            vmin=-_limit,
            vmax=_limit,
            origin="lower",
        )
        _ax.set_title(f"{cell_fruit_choice.value}: cells, bin {_name}", fontsize=10)
        _ax.set_xticks([])
        _ax.set_yticks([])
    weights_figure.suptitle(
        f"{row_choice.value}: heights {list(kept_heights)}, rings {list(kept_rings)},"
        f" {chosen_lv} LVs, fit on {len(_y)} fruit",
        fontsize=10,
    )
    weights_figure.tight_layout()
    weights_figure
    return weight_maps, weights_figure


@app.cell(hide_code=True)
def _():
    save_button = mo.ui.run_button(
        label="Save this selection to results/ and the figure"
    )
    save_button
    return (save_button,)


@app.cell(hide_code=True)
def _(
    all_lv,
    all_run,
    cfg,
    chosen_lv,
    fruit,
    kept_heights,
    kept_rings,
    paired_selection,
    row_choice,
    save_button,
    selected_prediction,
    selection_auc_by_group,
    selection_check_c,
    selection_scores,
    weight_maps,
    weights_figure,
):
    mo.stop(not save_button.value, mo.md("_Not saved._"))
    _run_dir = new_run_dir(
        cfg.path("results_dir"), f"ipls_selection_{row_choice.value}"
    )
    fruit.select(
        "fruit_id", "box", "defective", "browning", "cavity", "cause"
    ).with_columns(prediction=selected_prediction).write_parquet(
        _run_dir / "predictions.parquet"
    )
    selection_auc_by_group.write_parquet(_run_dir / "auc_by_group.parquet")
    selection_check_c.write_parquet(_run_dir / "check_c_healthy_correlations.parquet")
    np.savez(_run_dir / "weight_maps.npz", **weight_maps)
    (_run_dir / "scores.json").write_text(
        json.dumps(
            {
                # The 2,000 single differences are left out; the summary is kept.
                "paired": {
                    _k: _v
                    for _k, _v in paired_selection.items()
                    if _k != "differences"
                },
                **selection_scores,
            },
            indent=2,
            default=float,
        )
    )
    _figure = (
        cfg.path("derived_dir") / "figures" / f"13_selection_{row_choice.value}.png"
    )
    weights_figure.savefig(_figure, dpi=130, bbox_inches="tight")
    write_run_record(
        _run_dir,
        {
            "what": "PLS-DA on an iPLS selection: heights along the forward path, then rings",
            "row": row_choice.value,
            "heights": list(kept_heights),
            "rings": list(kept_rings),
            "n_lv_chosen": chosen_lv,
            "selection_rule": "number of heights and of rings chosen by Ruben on the paths",
            "lv_rule": "elbow of the mean RMSECV curve, chosen by Ruben by eye",
            "compared_with": all_run.name,
            "compared_n_lv": all_lv,
            "folds": "derived/outer_folds_20/outer_folds.csv",
            "bootstrap_seed": 369,
            "figure": str(_figure.relative_to(cfg.root)),
        },
        root=cfg.root,
    )
    mo.md(f"Saved to `{_run_dir}` and `{_figure}`.")
    return


if __name__ == "__main__":
    app.run()
