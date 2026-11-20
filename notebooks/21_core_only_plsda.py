import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title="21 Core-only PLS-DA")

with app.setup(hide_code=True):
    import json

    import altair as alt
    import marimo as mo
    import numpy as np
    import polars as pl
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.metrics import roc_auc_score

    from pearlatent.config import load_config
    from pearlatent.core_region import (
        HEIGHT_HIGH,
        HEIGHT_LOW,
        apply_fill,
        core_regions,
        fit_fill,
    )
    from pearlatent.feature_sets import load_features
    from pearlatent.fruit_table import CAUSES, drop_faulty_volumes, open_fruit
    from pearlatent.nuisance import box_bootstrap, healthy_fruit_correlations
    from pearlatent.plsda import (
        CUTOFF,
        MAX_COMPONENTS,
        box_bootstrap_auc,
        classification_scores,
        cross_validated_predictions,
        mean_squared_errors,
        paired_box_bootstrap_auc_difference,
        predict_each_h,
        youden_cutoff,
    )
    from pearlatent.runs import (
        ENCODING_RUN_DIR,
        PLSDA_BINARY23_RUN_DIRS,
        new_run_dir,
        write_run_record,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 21 Core-only PLS-DA on binary_23

    **Question.** Does PLS-DA on the core region only (the lower part of the body, without the
    neck and the stalk) separate consumable from non-consumable fruit (`binary_23`) as well as
    the full grid, and does its score still follow stalk length and fruit size on healthy fruit
    (check C)? Step 1 of the plan of 20 November 2026 (HANDOFF, story section 9).

    **Region** (`pearlatent.core_region`, approved by Ruben on 20 November 2026). Per fruit:
    the body cells of its 32³ grid (`masks.npy` of the encoding run: stalk removed, holes
    filled) with height 0.05 to 0.60 over the body (0 = calyx end), all radii. Chosen from the
    rubric and pear biology, before any model on it was fitted.

    **Full grid with a region per fruit.** All 32,768 cells stay columns. Cells outside a
    fruit's region get the mean of that cell over the training fruit that have it in their
    region (`fit_fill`, computed in every training fold only), so after centring they are 0 and
    add nothing to the fruit's score. A large fruit still has more region cells, so size is not
    removed; check C measures what remains.

    **Design.** As notebook 08: design A, the 12 open 2526 boxes, 20 repeats of 4-fold box
    cross-validation (`derived/outer_folds_20/`), 1 to 15 latent variables (LVs), centring
    only. Ruben chooses the number of LVs at the elbow of the mean RMSECV curve. Compared,
    paired on the same fruit, with the full-grid row of the same representation
    (`PLSDA_BINARY23_RUN_DIRS`).

    **Inputs.** The fruit table, `derived/volume_decisions.csv`, the body cells of the
    encoding run, `derived/outer_folds_20/outer_folds.csv`, the feature sets `hugo_vae` and
    `ct_32`. The sealed test boxes F, J and M are never loaded.

    **Outputs.** `results/<run_id>_plsda_binary23_<representation>_core/` (only when the save
    button is pressed): per-fruit predictions, curves, scores, check C, the regression vector
    and fill of the fit on all 12 boxes (for the counterfactuals), and `run.json`.

    **Status.** draft (20 November 2026)
    """)
    return


@app.cell
def _():
    cfg = load_config()
    return (cfg,)


@app.cell(hide_code=True)
def _(cfg):
    # The open 2526 fruit without the volumes judged faulty, as notebook 08.
    _open_2526 = open_fruit(cfg).filter(pl.col("season") == "2526")
    fruit = drop_faulty_volumes(
        _open_2526, cfg.path("derived_dir") / "volume_decisions.csv"
    ).with_columns(binary_23=(pl.col("defective") >= 2).cast(pl.Int8))
    print(f"open 2526 fruit: {_open_2526.height}")
    print(f"after removing faulty volumes: {fruit.height}")
    print(f"non-consumable (binary_23 = 1): {fruit['binary_23'].sum()}")
    return (fruit,)


@app.cell(hide_code=True)
def _(cfg, fruit):
    folds = pl.read_csv(cfg.path("derived_dir") / "outer_folds_20" / "outer_folds.csv")
    if set(folds["box"].unique().to_list()) != set(fruit["box"].unique().to_list()):
        raise ValueError("the boxes of the folds and of the fruit differ")
    print(f"{folds['repeat'].n_unique()} repeats over {folds['box'].n_unique()} boxes")
    return (folds,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 0. The region of every fruit

    Number of region cells per fruit and their share of the body. The region is the same for
    the latent and the CT row: both are 32³ grids on the same blocks of 4³ voxels.
    """)
    return


@app.cell(hide_code=True)
def _(cfg, fruit):
    # `position` is the row of the fruit in `masks.npy` of the encoding run.
    _bodies = np.load(
        cfg.path("derived_dir") / ENCODING_RUN_DIR / "masks.npy", mmap_mode="r"
    )[fruit["position"].to_numpy()]
    _region_grids = core_regions(np.asarray(_bodies))
    regions = _region_grids.reshape(fruit.height, -1)  # (fruit, 32,768), as the features
    region_sizes = fruit.select("fruit_id", "box", "defective", "cause").with_columns(
        region_cells=regions.sum(axis=1),
        body_cells=np.asarray(_bodies).reshape(fruit.height, -1).sum(axis=1),
    ).with_columns(share_of_body=pl.col("region_cells") / pl.col("body_cells"))
    print(
        f"height {HEIGHT_LOW} to {HEIGHT_HIGH} of the body, all radii; {fruit.height} fruit;"
        f" cells in the region of at least one fruit: {int(regions.any(axis=0).sum())}"
        f" of {regions.shape[1]}"
    )
    region_sizes.group_by("defective").agg(
        n=pl.len(),
        region_cells_median=pl.col("region_cells").median(),
        region_cells_min=pl.col("region_cells").min(),
        region_cells_max=pl.col("region_cells").max(),
        share_of_body_median=pl.col("share_of_body").median().round(3),
    ).sort("defective")
    return (regions,)


@app.cell(hide_code=True)
def _():
    REPRESENTATIONS = {
        "hugo_vae": "Hugo's latent (mu), core region",
        "ct_32": "CT averaged to 32³, core region",
    }
    representation_choice = mo.ui.dropdown(
        options={_label: _key for _key, _label in REPRESENTATIONS.items()},
        value=REPRESENTATIONS["hugo_vae"],
        label="Representation",
    )
    run_cv = mo.ui.run_button(label="Run the cross-validation (about 10 seconds)")
    mo.hstack([representation_choice, run_cv], justify="start")
    return REPRESENTATIONS, representation_choice, run_cv


@app.function(hide_code=True)
@mo.persistent_cache
def cross_validate_core(representation: str, fruit_ids: tuple) -> dict:
    """Out-of-fold predictions and RMSEC for one representation on the core region.

    In: the feature set name and the fruit (in table order). Out: a dict with `predictions`
    (repeats × fruit × LVs) and `rmsec` (one per number of LVs, from one fit on all fruit).
    The fill is fitted inside every training fold (`cross_validated_predictions(...,
    regions=...)`). Cached on disk, because the inputs are fixed.
    """
    cfg = load_config()
    derived = cfg.path("derived_dir")
    fruit = (
        drop_faulty_volumes(
            open_fruit(cfg).filter(pl.col("season") == "2526"),
            derived / "volume_decisions.csv",
        )
        .with_columns(binary_23=(pl.col("defective") >= 2).cast(pl.Int8))
        .filter(pl.col("fruit_id").is_in(list(fruit_ids)))
    )
    assert tuple(fruit["fruit_id"].to_list()) == fruit_ids, "fruit order changed"
    bodies = np.load(derived / ENCODING_RUN_DIR / "masks.npy", mmap_mode="r")
    regions = core_regions(np.asarray(bodies[fruit["position"].to_numpy()]))
    regions = regions.reshape(fruit.height, -1)
    x = load_features(cfg, representation, list(fruit_ids))
    y = fruit["binary_23"].to_numpy()
    folds = pl.read_csv(derived / "outer_folds_20" / "outer_folds.csv")
    # Cells are centred only (CLAUDE.md, section 7), as the full-grid rows.
    predictions = cross_validated_predictions(
        x, y, fruit["box"].to_numpy(), folds, scale=False, regions=regions
    )
    # RMSEC: one model on all 12 boxes (fill fitted on all of them), error on the same
    # fruit. Plotted for reference only; it always goes down with more LVs.
    fill = fit_fill(x, regions)
    x_filled = apply_fill(x, regions, fill)
    full_model = PLSRegression(n_components=MAX_COMPONENTS, scale=False).fit(
        x_filled, y.astype(float)
    )
    fitted = predict_each_h(full_model, x_filled)
    rmsec = np.sqrt(((fitted - y[:, None]) ** 2).mean(axis=0))
    return {"predictions": predictions, "rmsec": rmsec}


@app.cell(hide_code=True)
def _(fruit, representation_choice, run_cv):
    # In script mode the button is never pressed; the cross-validation then runs directly.
    mo.stop(
        mo.running_in_notebook() and not run_cv.value,
        mo.md("_Press the button to run the cross-validation._"),
    )
    cv = cross_validate_core(
        representation_choice.value, tuple(fruit["fruit_id"].to_list())
    )
    y = fruit["binary_23"].to_numpy()
    boxes = fruit["box"].to_numpy()
    print(
        f"{representation_choice.value}, core: predictions for"
        f" {cv['predictions'].shape[0]} repeats × {cv['predictions'].shape[1]} fruit ×"
        f" {cv['predictions'].shape[2]} LVs"
    )
    return boxes, cv, y


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1. Error curves: choose the number of latent variables

    RMSECV(h) = √(PRESS(h) / n) for every repeat (grey) and for PRESS averaged over the 20
    repeats (black); RMSEC (blue) from one fit on all fruit. The choice is the elbow of the
    black curve.
    """)
    return


@app.cell(hide_code=True)
def _(cv, y):
    _squared = (cv["predictions"] - y[None, :, None].astype(float)) ** 2
    _per_repeat = np.sqrt(_squared.mean(axis=1))  # (repeats, LVs)
    _mean_press = mean_squared_errors(cv["predictions"], y).sum(axis=0)
    _n_lv = np.arange(1, cv["predictions"].shape[2] + 1)
    curves = pl.concat(
        [
            pl.DataFrame(
                {
                    "n_lv": np.tile(_n_lv, _per_repeat.shape[0]),
                    "error": _per_repeat.ravel(),
                    "curve": "RMSECV, one repeat",
                    "repeat": np.repeat(
                        np.arange(1, _per_repeat.shape[0] + 1), len(_n_lv)
                    ),
                }
            ),
            pl.DataFrame(
                {
                    "n_lv": _n_lv,
                    "error": np.sqrt(_mean_press / len(y)),
                    "curve": "RMSECV, mean PRESS",
                    "repeat": 0,
                }
            ),
            pl.DataFrame(
                {"n_lv": _n_lv, "error": cv["rmsec"], "curve": "RMSEC", "repeat": -1}
            ),
        ],
        how="vertical_relaxed",
    )
    _colours = alt.Scale(
        domain=["RMSECV, one repeat", "RMSECV, mean PRESS", "RMSEC"],
        range=["#bbbbbb", "#000000", "#1f77b4"],
    )
    _chart = (
        alt.Chart(curves)
        .mark_line(point=True)
        .encode(
            x=alt.X("n_lv:O", title="number of latent variables"),
            y=alt.Y(
                "error:Q", title="root mean squared error", scale=alt.Scale(zero=False)
            ),
            color=alt.Color("curve:N", scale=_colours),
            detail="repeat:N",
            tooltip=["curve", "repeat", "n_lv", alt.Tooltip("error:Q", format=".4f")],
        )
        .properties(width=560, height=320)
    )
    _table = curves.filter(pl.col("repeat") <= 0).pivot(
        on="curve", index="n_lv", values="error"
    )
    mo.vstack([_chart, _table])
    return (curves,)


@app.cell(hide_code=True)
def _():
    lv_choice = mo.ui.dropdown(
        options=["not chosen"] + [str(_h) for _h in range(1, MAX_COMPONENTS + 1)],
        value="not chosen",
        label="Number of latent variables (Ruben's choice at the elbow)",
    )
    lv_choice
    return (lv_choice,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2. Scores at the chosen number of latent variables, against the full grid

    Per-fruit average of the 20 out-of-fold predictions; 95 % interval from 2,000 draws of
    the 12 boxes (seed 369). The full-grid row of the same representation is read from its
    saved run, and the difference is paired on the same fruit and the same box draws.
    """)
    return


@app.cell(hide_code=True)
def _(boxes, cfg, cv, fruit, lv_choice, representation_choice, y):
    mo.stop(lv_choice.value == "not chosen", mo.md("_Choose the number of LVs first._"))
    chosen_lv = int(lv_choice.value)
    _at_h = cv["predictions"][:, :, chosen_lv - 1]  # (repeats, fruit)
    mean_prediction = _at_h.mean(axis=0)
    _grade = fruit["defective"].to_numpy()

    cutoff_youden = youden_cutoff(mean_prediction, y)
    _main = classification_scores(mean_prediction, y, _grade, cutoff=CUTOFF)
    _youden = classification_scores(mean_prediction, y, _grade, cutoff=cutoff_youden)
    _low, _high, _ = box_bootstrap_auc(mean_prediction, y, boxes)

    # The full-grid row of the same representation (notebook 08, saved 3 November 2026).
    _full_run = PLSDA_BINARY23_RUN_DIRS[representation_choice.value]
    _full = pl.read_parquet(
        cfg.path("results_dir") / _full_run / "predictions.parquet"
    ).select("fruit_id", full_prediction="prediction")
    full_table = fruit.select("fruit_id").join(
        _full, on="fruit_id", how="left", validate="1:1"
    )
    if full_table["full_prediction"].null_count():
        raise ValueError("some fruit have no saved full-grid prediction")
    _paired = paired_box_bootstrap_auc_difference(
        mean_prediction, full_table["full_prediction"].to_numpy(), y, boxes
    )
    scores = {
        **_main,
        "youden": {"cutoff": cutoff_youden, **_youden},
        "auc_box_bootstrap_95": [_low, _high],
        "n_lv": chosen_lv,
        "full_grid_run": _full_run,
        "full_grid_auc": _paired["auc_b"],
        "core_minus_full": _paired["difference"],
        "core_minus_full_95": [_paired["lower_95"], _paired["upper_95"]],
        "share_core_better": _paired["share_a_better"],
    }
    mo.md(
        f"""
    **{chosen_lv} LVs.** Core-only AUC {_main['auc']:.3f} (95 % box bootstrap {_low:.3f}
    to {_high:.3f}); full grid {_paired['auc_b']:.3f}. Paired difference core − full:
    {_paired['difference']:+.3f} ({_paired['lower_95']:+.3f} to {_paired['upper_95']:+.3f});
    the core is higher in {100 * _paired['share_a_better']:.0f} % of the box draws.

    At cut-off {CUTOFF}: recall {_main['recall']:.3f}, precision {_main['precision']:.3f}.
    At the Youden cut-off {cutoff_youden:.3f} (chosen on these predictions): recall
    {_youden['recall']:.3f}, precision {_youden['precision']:.3f}.
    """
    )
    return chosen_lv, cutoff_youden, full_table, mean_prediction, scores


@app.cell(hide_code=True)
def _(cutoff_youden, fruit, full_table, mean_prediction):
    # Charter rule 5: results split by cause. For grades 0 and 1 (label 0) the share called
    # positive is the false-positive rate; for grades 2 and 3 it is the recall.
    predictions_table = fruit.select(
        "fruit_id", "box", "defective", "browning", "cavity", "cause", "binary_23"
    ).with_columns(
        prediction=mean_prediction,
        full_prediction=full_table["full_prediction"],
        called_positive=mean_prediction >= CUTOFF,
        called_positive_youden=mean_prediction >= cutoff_youden,
    )
    _cause_order = {_c: _k for _k, _c in enumerate(CAUSES)}
    (
        predictions_table.group_by("cause", "defective")
        .agg(
            n=pl.len(),
            called_at_0_5=pl.col("called_positive").mean().round(2),
            called_at_youden=pl.col("called_positive_youden").mean().round(2),
        )
        .with_columns(_order=pl.col("cause").replace_strict(_cause_order))
        .sort("_order", "defective")
        .drop("_order")
    )
    return (predictions_table,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 3. Grade 0 against each defect group

    AUC of grade 0 against one group, for the core-only and the full-grid row, with a 95 %
    interval over box draws (2,000, seed 369). Both models were trained on `binary_23`.
    """)
    return


@app.cell(hide_code=True)
def _(predictions_table):
    _groups = {
        "browning only 1": (pl.col("browning") == 1) & (pl.col("cavity") == 0),
        "browning only 2": (pl.col("browning") == 2) & (pl.col("cavity") == 0),
        "browning only 3": (pl.col("browning") == 3) & (pl.col("cavity") == 0),
        "cavity 2 or 3": pl.col("cavity") >= 2,
        "any grade 1": pl.col("defective") == 1,
    }
    _healthy = predictions_table.filter(pl.col("defective") == 0)
    _rows = []
    for _name, _condition in _groups.items():
        _pair = pl.concat(
            [
                _healthy.with_columns(_label=pl.lit(0)),
                predictions_table.filter(_condition).with_columns(_label=pl.lit(1)),
            ]
        )
        _labels = _pair["_label"].to_numpy()
        for _model, _column in (("core", "prediction"), ("full grid", "full_prediction")):
            _values = _pair[_column].to_numpy()
            _low, _high, _ = box_bootstrap_auc(_values, _labels, _pair["box"].to_numpy())
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
    auc_by_group = pl.DataFrame(_rows)
    auc_by_group.with_columns(pl.col("auc", "lower_95", "upper_95").round(3))
    return (auc_by_group,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 4. Check C: does the core-only score still follow stalk and size on healthy fruit?

    Grade 0 fruit only; Spearman correlation of the score (minus its box median) with stalk
    length and body size, 1,000 box draws, as notebooks 10 and 19. The full-grid row is
    computed the same way beside it.
    """)
    return


@app.cell(hide_code=True)
def _(fruit, predictions_table):
    _table = predictions_table.join(
        fruit.select("fruit_id", "stalk_length", "body_voxels", "stalk_present"),
        on="fruit_id",
        how="left",
        validate="1:1",
    )
    _rows = []
    for _model, _column in (("core", "prediction"), ("full grid", "full_prediction")):
        _scored = _table.with_columns(score=pl.col(_column))
        _c = healthy_fruit_correlations(_scored)
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
                    "n_healthy": _c["n_healthy"],
                    "spearman": _c[_key],
                    "lower_95": _low,
                    "upper_95": _high,
                }
            )
    check_c = pl.DataFrame(_rows)
    check_c.with_columns(pl.col("spearman", "lower_95", "upper_95").round(3))
    return (check_c,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 5. Final fit on all 12 open boxes (for the counterfactuals)

    One model at the chosen number of LVs on all 315 fruit, with the fill fitted on all of
    them. Its regression vector (32,768 cells) and the fill are saved with the run; the
    counterfactual step moves only the cells of a fruit's own region along this vector.
    """)
    return


@app.cell(hide_code=True)
def _(cfg, chosen_lv, fruit, regions, representation_choice, y):
    _x = load_features(cfg, representation_choice.value, fruit["fruit_id"].to_list())
    final_fill = fit_fill(_x, regions)
    _model = PLSRegression(n_components=chosen_lv, scale=False).fit(
        apply_fill(_x, regions, final_fill), y.astype(float)
    )
    regression_vector = _model.coef_.ravel()  # one weight per cell
    _grid = regression_vector.reshape(32, 32, 32)
    _outside = ~regions.any(axis=0)
    print(
        f"regression vector: {regression_vector.size} cells; largest absolute weight"
        f" outside every region: {np.abs(regression_vector[_outside]).max():.2e}"
        " (should be 0: those columns are constant)"
    )
    # Mean absolute weight per height slice (third axis), to see where the vector sits.
    pl.DataFrame(
        {
            "slice": np.arange(32),
            "mean_abs_weight": np.abs(_grid).mean(axis=(0, 1)),
        }
    ).filter(pl.col("mean_abs_weight") > 0)
    return final_fill, regression_vector


@app.cell(hide_code=True)
def _():
    save_button = mo.ui.run_button(label="Save this run to results/")
    save_button
    return (save_button,)


@app.cell(hide_code=True)
def _(
    REPRESENTATIONS,
    auc_by_group,
    cfg,
    check_c,
    chosen_lv,
    curves,
    cutoff_youden,
    final_fill,
    predictions_table,
    regression_vector,
    representation_choice,
    save_button,
    scores,
):
    mo.stop(not save_button.value, mo.md("_Not saved._"))
    _run_dir = new_run_dir(
        cfg.path("results_dir"), f"plsda_binary23_{representation_choice.value}_core"
    )
    predictions_table.write_parquet(_run_dir / "predictions.parquet")
    curves.write_parquet(_run_dir / "curves.parquet")
    auc_by_group.write_parquet(_run_dir / "auc_by_group.parquet")
    check_c.write_parquet(_run_dir / "check_c_healthy_correlations.parquet")
    np.save(_run_dir / "regression_vector.npy", regression_vector)
    np.save(_run_dir / "fill.npy", final_fill)
    (_run_dir / "scores.json").write_text(json.dumps(scores, indent=2))
    write_run_record(
        _run_dir,
        {
            "what": "PLS-DA on binary_23, core region only, design A, 12 open 2526 boxes",
            "representation": representation_choice.value,
            "representation_label": REPRESENTATIONS[representation_choice.value],
            "region": {
                "module": "pearlatent.core_region",
                "height_low": HEIGHT_LOW,
                "height_high": HEIGHT_HIGH,
                "radius": "all",
                "body_cells": f"{ENCODING_RUN_DIR}/masks.npy",
                "outside_region": "training-fold mean of the cell over fruit with it in their region",
            },
            "n_fruit": predictions_table.height,
            "n_lv_chosen": chosen_lv,
            "lv_rule": "elbow of the mean RMSECV curve, chosen by Ruben by eye",
            "scale": False,
            "max_components": MAX_COMPONENTS,
            "cutoff": CUTOFF,
            "cutoff_youden": cutoff_youden,
            "compared_with": PLSDA_BINARY23_RUN_DIRS[representation_choice.value],
            "folds": "derived/outer_folds_20/outer_folds.csv",
            "bootstrap_seed": 369,
        },
        root=cfg.root,
    )
    mo.md(f"Saved to `{_run_dir}`.")
    return


if __name__ == "__main__":
    app.run()
