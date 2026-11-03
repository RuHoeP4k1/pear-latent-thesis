import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title="08 PLS-DA binary_23")

with app.setup(hide_code=True):
    import json

    import altair as alt
    import marimo as mo
    import numpy as np
    import polars as pl
    from sklearn.cross_decomposition import PLSRegression

    from pearlatent.config import load_config
    from pearlatent.feature_sets import load_features
    from pearlatent.fruit_table import CAUSES, drop_faulty_volumes, open_fruit
    from pearlatent.plsda import (
        CUTOFF,
        MAX_COMPONENTS,
        box_bootstrap_auc,
        classification_scores,
        cross_validated_predictions,
        mean_squared_errors,
        predict_each_h,
        youden_cutoff,
    )
    from pearlatent.runs import new_run_dir, write_run_record


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 08 PLS-DA on binary_23: can consumable and non-consumable fruit be told apart?

    **Question.** How well does partial least squares discriminant analysis (PLS-DA) separate
    consumable from non-consumable fruit (`binary_23`: defective grade ≥ 2) in the open 2526
    boxes, on Hugo's latent and on the CT averaged to 32³? Charter rule 5: the first target.

    **Design.** Design A (`docs/cv_explained.md`, sections 4 to 7): the 12 open boxes, 20
    repeats of a 4-fold box cross-validation (`derived/outer_folds_20/`), 1 to 15 latent
    variables (LVs), centring only (`scale=False`). The number of LVs is chosen by Ruben at
    the elbow of the mean RMSECV curve (2 November 2026), before any test box is opened. The
    score: AUC (area under the ROC curve), recall per grade and precision at two cut-offs,
    split by cause; 95 % interval by a bootstrap over boxes. Cut-offs (Ruben, 2 November
    2026): 0.5, fixed in advance, and the Youden cut-off (largest recall − false-positive
    rate) on the out-of-fold predictions, saved in `run.json` and applied unchanged to the
    test boxes.

    **Inputs.** The fruit table (`pearlatent.fruit_table`), `derived/volume_decisions.csv`,
    `derived/outer_folds_20/outer_folds.csv`, the feature sets of `pearlatent.feature_sets`.
    The sealed test boxes F, J and M are never loaded.

    **Outputs.** `results/<run_id>_plsda_binary23_<representation>/` (only when the save
    button is pressed): per-fruit predictions, the curves, the scores and `run.json`.

    **Status.** draft (first run, 2 November 2026)
    """)
    return


@app.cell
def _():
    cfg = load_config()
    return (cfg,)


@app.cell(hide_code=True)
def _(cfg):
    # The open 2526 fruit (the sealed boxes are not in the encoding run's open part),
    # without the volumes judged faulty (`docs/outlier_rule.md`, sections 9 and 10).
    _all_open = open_fruit(cfg)
    _open_2526 = _all_open.filter(pl.col("season") == "2526")
    fruit = drop_faulty_volumes(
        _open_2526, cfg.path("derived_dir") / "volume_decisions.csv"
    ).with_columns(binary_23=(pl.col("defective") >= 2).cast(pl.Int8))
    print(f"open fruit, both seasons: {_all_open.height}")
    print(f"open 2526 fruit: {_open_2526.height}")
    print(f"after removing faulty volumes: {fruit.height}")
    print(f"non-consumable (binary_23 = 1): {fruit['binary_23'].sum()}")
    fruit.group_by("box").agg(
        n=pl.len(), non_consumable=pl.col("binary_23").sum()
    ).sort("box")
    return (fruit,)


@app.cell(hide_code=True)
def _(cfg, fruit):
    # The 20 partitions of the 12 open boxes into 4 folds of 3, drawn on 13 and 21 October.
    folds = pl.read_csv(cfg.path("derived_dir") / "outer_folds_20" / "outer_folds.csv")
    _fold_boxes = set(folds["box"].unique().to_list())
    _fruit_boxes = set(fruit["box"].unique().to_list())
    if _fold_boxes != _fruit_boxes:
        raise ValueError(
            f"boxes differ: folds {sorted(_fold_boxes)}, fruit {sorted(_fruit_boxes)}"
        )
    print(f"{folds['repeat'].n_unique()} repeats over the {len(_fold_boxes)} boxes")
    return


@app.cell(hide_code=True)
def _():
    # Random encoder dropped (Hugo, 29 October 2026).
    REPRESENTATIONS = {
        "hugo_vae": "Hugo's latent (mu)",
        "ct_32": "CT averaged to 32³ (mean of 4 × 4 × 4 voxel blocks)",
        "tissue_features": "10 tissue features of Van De Looverbosch et al. 2020 (128³)",
        "med3d": "Med3D ResNet-18, 512 channels averaged over the grid (128³)",
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
def cross_validate(representation: str, fruit_ids: tuple) -> dict:
    """Out-of-fold predictions and the calibration curve for one representation.

    In: the feature set name and the fruit (in table order). Out: a dict with `predictions` (repeats × fruit × LVs) and `rmsec` (one per
    number of LVs, from one fit on all fruit). Cached on disk, because the inputs are fixed.
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
    # Cells are centred only (CLAUDE.md, section 7). The 10 tissue features are in different
    # units (a volume share, means, SDs, a test statistic), so they are also scaled to unit
    # variance; PLSRegression does that on the training fruit of each fold only. The same for
    # the 512 Med3D channels: different variables, not neighbouring cells (Ruben, 2 November).
    scale = representation in ("tissue_features", "med3d")
    # PLS has at most as many LVs as features: 10 for the tissue features, 15 otherwise.
    n_components = min(MAX_COMPONENTS, x.shape[1])
    predictions = cross_validated_predictions(
        x, y, boxes, folds, scale=scale, max_components=n_components
    )
    # RMSEC: one model on all 12 boxes, error on the same fruit. It always goes down with
    # more LVs, so it is plotted for reference and does not decide.
    full_model = PLSRegression(n_components=n_components, scale=scale).fit(
        x, y.astype(float)
    )
    fitted = predict_each_h(full_model, x)
    rmsec = np.sqrt(((fitted - y[:, None]) ** 2).mean(axis=0))
    return {"predictions": predictions, "rmsec": rmsec}


@app.cell(hide_code=True)
def _(fruit, representation_choice, run_cv):
    # In script mode the button is never pressed; the cross-validation then runs directly.
    mo.stop(
        mo.running_in_notebook() and not run_cv.value,
        mo.md("_Press the button to run the cross-validation._"),
    )
    cv = cross_validate(
        representation_choice.value,
        tuple(fruit["fruit_id"].to_list()),
    )
    y = fruit["binary_23"].to_numpy()
    print(
        f"{representation_choice.value}: predictions for {cv['predictions'].shape[0]}"
        f" repeats × {cv['predictions'].shape[1]} fruit × {cv['predictions'].shape[2]} LVs"
    )
    return cv, y


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1. Error curves: choose the number of latent variables

    RMSECV(h) = √(PRESS(h) / n) for every repeat (grey) and for PRESS averaged over the 20
    repeats (black); RMSEC (blue) from one fit on all fruit. The choice is the elbow: the
    number of LVs after which the black curve stops going down. Only these 12 boxes are
    looked at.
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
                    "repeat": np.repeat(np.arange(1, _per_repeat.shape[0] + 1), len(_n_lv)),
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
        how="vertical_relaxed",  # the repeat column is int32 in one frame, int64 in others
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
            y=alt.Y("error:Q", title="root mean squared error", scale=alt.Scale(zero=False)),
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
    ## 2. Scores at the chosen number of latent variables

    The per-fruit average of the 20 out-of-fold predictions gives the main row; the score of
    each repeat shows the spread caused by the partition (not a confidence interval). The
    interval comes from 2,000 draws of 12 boxes with replacement (seed 369).
    """)
    return


@app.cell(hide_code=True)
def _(cv, fruit, lv_choice, y):
    mo.stop(lv_choice.value == "not chosen", mo.md("_Choose the number of LVs first._"))
    chosen_lv = int(lv_choice.value)
    mo.stop(
        chosen_lv > cv["predictions"].shape[2],
        mo.md(f"_This representation has at most {cv['predictions'].shape[2]} LVs._"),
    )
    _at_h = cv["predictions"][:, :, chosen_lv - 1]  # (repeats, fruit)
    mean_prediction = _at_h.mean(axis=0)
    _grade = fruit["defective"].to_numpy()
    _boxes = fruit["box"].to_numpy()

    # Youden cut-off from the averaged out-of-fold predictions of the open boxes only; the
    # repeats are scored at this same cut-off, not each at its own.
    cutoff_youden = youden_cutoff(mean_prediction, y)
    _cutoffs = {"0.5": CUTOFF, "Youden": cutoff_youden}
    _main = {
        _name: classification_scores(mean_prediction, y, _grade, cutoff=_c)
        for _name, _c in _cutoffs.items()
    }
    _per_repeat = {
        _name: pl.DataFrame(
            [
                classification_scores(_at_h[_r], y, _grade, cutoff=_c)
                for _r in range(_at_h.shape[0])
            ]
        )
        for _name, _c in _cutoffs.items()
    }
    _low, _high, _ = box_bootstrap_auc(mean_prediction, y, _boxes)
    scores = {
        **_main["0.5"],
        "youden": {"cutoff": cutoff_youden, **_main["Youden"]},
        "auc_box_bootstrap_95": [_low, _high],
        "n_lv": chosen_lv,
    }
    _rows = [("AUC", "0.5", "auc")] + [
        (_score, _name, _key)
        for _name in _cutoffs
        for _score, _key in (("recall", "recall"), ("precision", "precision"))
    ]
    _summary = pl.DataFrame(
        {
            "score": [_r[0] for _r in _rows],
            "cut-off": ["none" if _r[0] == "AUC" else _r[1] for _r in _rows],
            "average of 20 predictions": [_main[_n][_k] for _, _n, _k in _rows],
            "repeats: min": [_per_repeat[_n][_k].min() for _, _n, _k in _rows],
            "repeats: max": [_per_repeat[_n][_k].max() for _, _n, _k in _rows],
        }
    )
    mo.vstack(
        [
            mo.md(
                f"**{chosen_lv} LVs.** AUC {_main['0.5']['auc']:.3f}, 95 % box-bootstrap"
                f" interval {_low:.3f} to {_high:.3f}. Cut-offs: {CUTOFF} (fixed in advance)"
                f" and Youden {cutoff_youden:.3f} (chosen on these predictions)."
            ),
            _summary,
        ]
    )
    return chosen_lv, cutoff_youden, mean_prediction, scores


@app.cell(hide_code=True)
def _(cutoff_youden, fruit, mean_prediction):
    # Charter rule 5: results on `defective` split by cause. For grades 0 and 1 (label 0) the
    # share called positive is the false-positive rate; for grades 2 and 3 it is the recall.
    predictions_table = fruit.select(
        "fruit_id", "box", "defective", "browning", "cavity", "cause", "binary_23"
    ).with_columns(
        prediction=mean_prediction,
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
def _(predictions_table):
    mo.md("""Out-of-fold prediction per fruit (average of 20 repeats), by defective grade and cause.""")
    alt.Chart(predictions_table).mark_tick(thickness=2).encode(
        x=alt.X("prediction:Q", title="PLS-DA output"),
        y=alt.Y("defective:O", title="defective grade"),
        color=alt.Color("cause:N", sort=list(CAUSES)),
        tooltip=["fruit_id", "box", "browning", "cavity", alt.Tooltip("prediction:Q", format=".3f")],
    ).properties(width=560, height=200)
    return


@app.cell(hide_code=True)
def _():
    save_button = mo.ui.run_button(label="Save this run to results/")
    save_button
    return (save_button,)


@app.cell(hide_code=True)
def _(
    REPRESENTATIONS,
    cfg,
    chosen_lv,
    curves,
    cutoff_youden,
    predictions_table,
    representation_choice,
    save_button,
    scores,
):
    mo.stop(not save_button.value, mo.md("_Not saved._"))
    _run_dir = new_run_dir(
        cfg.path("results_dir"), f"plsda_binary23_{representation_choice.value}"
    )
    predictions_table.write_parquet(_run_dir / "predictions.parquet")
    curves.write_parquet(_run_dir / "curves.parquet")
    (_run_dir / "scores.json").write_text(json.dumps(scores, indent=2))
    write_run_record(
        _run_dir,
        {
            "what": "PLS-DA on binary_23, design A, 12 open 2526 boxes",
            "representation": representation_choice.value,
            "representation_label": REPRESENTATIONS[representation_choice.value],
            "n_fruit": predictions_table.height,
            "n_lv_chosen": chosen_lv,
            "lv_rule": "elbow of the mean RMSECV curve, chosen by Ruben by eye",
            # Must match `cross_validate`: unit-variance scaling inside each fold for the
            # tissue features and Med3D, centring only for the cell grids.
            "scale": representation_choice.value in ("tissue_features", "med3d"),
            "max_components": MAX_COMPONENTS,
            "cutoff": CUTOFF,
            # Chosen on the out-of-fold predictions above; apply unchanged to the test boxes.
            "cutoff_youden": cutoff_youden,
            "folds": "derived/outer_folds_20/outer_folds.csv",
            "bootstrap_seed": 369,
        },
        root=cfg.root,
    )
    mo.md(f"Saved to `{_run_dir}`.")
    return


if __name__ == "__main__":
    app.run()
