import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title="12 Ordinal model on the PLS-DA prediction")

with app.setup:
    import altair as alt
    import marimo as mo
    import numpy as np
    import polars as pl
    from scipy.stats import spearmanr
    from sklearn.metrics import cohen_kappa_score

    from pearlatent.config import load_config
    from pearlatent.ordinal import (
        cross_validated_ordinal,
        fit_ordinal,
        slope_and_cutpoints,
    )
    from pearlatent.runs import PLSDA_BINARY23_RUN_DIRS, new_run_dir, write_run_record


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 12 Ordinal model on the PLS-DA prediction

    **Question.** The binary PLS-DA model (`binary_23`, defective grade 2 or higher, notebook
    08) gives one out-of-fold prediction per fruit. Does that one number predict the defective
    grade 0 to 3, and which neighbouring grades can it tell apart?

    **Why this model.** One PLS model per grade (notebook 11) fails for grades 1 and 2: their
    RMSECV never drops below that of no model. On the binary prediction the four grades lie in
    order (section 1), so a grade in the middle cannot be "high" on a linear score while both
    neighbours are "low". A cumulative logit model keeps the order:

    P(grade ≤ j) = logistic(c_j − β · score),  j = 0, 1, 2,

    with one slope β and three cut-points c_0 < c_1 < c_2. On the score axis the boundary
    between grade j and j + 1 lies at c_j / β. No new PLS model and no new choice of LVs: the
    prediction comes from the saved runs at Ruben's LVs (latent 4, CT at 32³ 5, tissue
    features 7, Med3D 7).

    **Inputs.** `results/<run>/predictions.parquet` of the four runs in
    `PLSDA_BINARY23_RUN_DIRS`; `derived/outer_folds_20/outer_folds.csv`.

    **Outputs.** On request: `results/<run_id>_ordinal_defective/` with the class
    probabilities per fruit, the summary per representation, the confusion tables and
    `run.json`.

    **Status.** draft, 7 November 2026.
    """)
    return


@app.cell
def _():
    cfg = load_config()
    CLASSES = (0, 1, 2, 3)
    return CLASSES, cfg


@app.cell(hide_code=True)
def _(cfg):
    # The saved out-of-fold prediction of every representation (mean over the 20 repeats
    # of the box cross-validation), one row per fruit and representation.
    _frames = []
    for _rep, _dir in PLSDA_BINARY23_RUN_DIRS.items():
        _p = pl.read_parquet(cfg.path("results_dir") / _dir / "predictions.parquet")
        print(f"{_rep}: {_p.height} fruit from {_dir}")
        _frames.append(
            _p.select(
                "fruit_id", "box", "defective", "browning", "cavity", "prediction"
            ).with_columns(representation=pl.lit(_rep))
        )
    scores = pl.concat(_frames)
    # Every representation must hold the same fruit, or the comparison is not paired.
    _sets = scores.group_by("representation").agg(pl.col("fruit_id").sort())
    assert all(
        _s.to_list() == _sets["fruit_id"][0].to_list() for _s in _sets["fruit_id"]
    )
    REPRESENTATIONS = list(PLSDA_BINARY23_RUN_DIRS)
    folds = pl.read_csv(cfg.path("derived_dir") / "outer_folds_20" / "outer_folds.csv")
    scores.filter(pl.col("representation") == REPRESENTATIONS[0]).group_by(
        "defective"
    ).agg(n=pl.len()).sort("defective")
    return REPRESENTATIONS, folds, scores


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1. The binary prediction per defective grade

    If the grades are steps on one scale, the boxes should climb from grade 0 to grade 3. How
    much two neighbouring boxes overlap is how hard those two grades are to tell apart.
    """)
    return


@app.cell(hide_code=True)
def _(scores):
    _chart = (
        alt.Chart(scores)
        .mark_boxplot()
        .encode(
            x=alt.X("defective:O", title="defective grade"),
            y=alt.Y("prediction:Q", title="out-of-fold PLS-DA prediction"),
            color=alt.Color("defective:O", legend=None),
            column=alt.Column("representation:N", title=None),
        )
        .properties(width=150, height=260)
    )
    _medians = scores.group_by("representation", "defective").agg(
        median=pl.col("prediction").median().round(3)
    )
    mo.vstack(
        [
            _chart,
            _medians.pivot(on="defective", index="representation", values="median")
            .rename({str(_k): f"median grade {_k}" for _k in range(4)})
            .sort("representation"),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2. The model fitted on all 315 fruit: where the boundaries fall

    Only to read the model, not to score it (section 3 scores it out of fold). The boundary
    between two grades on the prediction axis is c_j / β: a fruit with a prediction above it
    is more likely to be in the higher grade group than in the lower one.
    """)
    return


@app.cell(hide_code=True)
def _(CLASSES, REPRESENTATIONS, scores):
    _rows = []
    for _rep in REPRESENTATIONS:
        _s = scores.filter(pl.col("representation") == _rep)
        _result = fit_ordinal(
            _s["prediction"].to_numpy(), _s["defective"].to_numpy(), CLASSES
        )
        _slope, _cuts = slope_and_cutpoints(_result)
        _row = {"representation": _rep, "slope_beta": round(_slope, 2)}
        for _j in range(3):
            _row[f"boundary_{_j}_{_j + 1}"] = round(float(_cuts[_j] / _slope), 3)
        _rows.append(_row)
    boundaries = pl.DataFrame(_rows)
    boundaries
    return (boundaries,)


@app.cell(hide_code=True)
def _(CLASSES, REPRESENTATIONS, scores):
    # Worked example for checking by hand: the first fruit of the first representation.
    _s = scores.filter(pl.col("representation") == REPRESENTATIONS[0])
    _result = fit_ordinal(
        _s["prediction"].to_numpy(), _s["defective"].to_numpy(), CLASSES
    )
    _slope, _cuts = slope_and_cutpoints(_result)
    _x = float(_s["prediction"][0])
    _cumulative = 1 / (1 + np.exp(-(_cuts - _slope * _x)))  # P(grade ≤ 0, 1, 2)
    _by_hand = np.diff(np.concatenate([[0.0], _cumulative, [1.0]]))
    _library = _result.predict(exog=np.array([[_x]]))[0]
    assert np.allclose(_by_hand, _library)
    mo.md(
        f"""
    **By hand, fruit {_s["fruit_id"][0]} ({REPRESENTATIONS[0]}, true grade
    {_s["defective"][0]}), prediction x = {_x:.3f}, β = {_slope:.2f},
    cut-points c = {", ".join(f"{_c:.2f}" for _c in _cuts)}.**

    P(grade ≤ j) = 1 / (1 + exp(−(c_j − β·x))) = {", ".join(f"{_v:.3f}" for _v in _cumulative)}

    P(grade = 0, 1, 2, 3) = differences = {", ".join(f"{_v:.3f}" for _v in _by_hand)}
    (statsmodels gives {", ".join(f"{_v:.3f}" for _v in _library)}).
    """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 3. Out of fold: probabilities, assignment and agreement

    For each of the 20 repeats of the 4-fold box cross-validation, the model is fitted on the
    prediction and grade of the fruit in the training boxes and gives the four class
    probabilities of the fruit in the held-out boxes; the probabilities are then averaged over
    the repeats. Assignment: the class with the highest probability.

    Limit: the saved prediction of a training fruit is itself an average of fold models, some
    of which saw the held-out boxes. The ordinal step has only four parameters, so the effect
    is small, but the scores here are not fully nested.

    Reported beside the error rate: quadratic weighted kappa (agreement on an ordered scale,
    a miss by two grades counts four times a miss by one; 0 is chance, 1 is perfect),
    Spearman correlation of the expected grade Σ k·P(k) with the true grade, the share
    assigned within one grade, and the recall of each grade (share of its fruit assigned to
    exactly that grade).
    """)
    return


@app.cell(hide_code=True)
def _(CLASSES, REPRESENTATIONS, folds, scores):
    _frames = []
    for _rep in REPRESENTATIONS:
        _s = scores.filter(pl.col("representation") == _rep)
        _p = cross_validated_ordinal(
            _s["prediction"].to_numpy(),
            _s["defective"].to_numpy(),
            _s["box"].to_numpy(),
            folds,
            CLASSES,
        )
        _frames.append(
            _s.with_columns(
                *[
                    pl.Series(f"p_grade_{_k}", _p[:, _i])
                    for _i, _k in enumerate(CLASSES)
                ],
                assigned=pl.Series(np.array(CLASSES)[_p.argmax(axis=1)]),
                expected_grade=pl.Series(_p @ np.array(CLASSES, dtype=float)),
            )
        )
    probabilities = pl.concat(_frames)

    _rows = []
    for _rep in REPRESENTATIONS:
        _s = probabilities.filter(pl.col("representation") == _rep)
        _true = _s["defective"].to_numpy()
        _assigned = _s["assigned"].to_numpy()
        _row = {
            "representation": _rep,
            "n": _s.height,
            "error_rate": round(float(np.mean(_assigned != _true)), 3),
            "largest_class_error": round(
                1 - float(np.bincount(_true).max() / _true.size), 3
            ),
            "weighted_kappa": round(
                cohen_kappa_score(_true, _assigned, weights="quadratic"), 3
            ),
            "spearman_expected": round(
                float(spearmanr(_s["expected_grade"], _true)[0]), 3
            ),
            "within_one_grade": round(
                float(np.mean(np.abs(_assigned - _true) <= 1)), 3
            ),
        }
        for _k in CLASSES:
            _row[f"recall_grade_{_k}"] = round(
                float(np.mean(_assigned[_true == _k] == _k)), 3
            )
        _rows.append(_row)
    summary = pl.DataFrame(_rows)
    summary
    return probabilities, summary


@app.cell(hide_code=True)
def _(REPRESENTATIONS):
    confusion_choice = mo.ui.dropdown(
        options=REPRESENTATIONS, value=REPRESENTATIONS[0], label="Representation"
    )
    confusion_choice
    return (confusion_choice,)


@app.cell(hide_code=True)
def _(CLASSES, confusion_choice, probabilities):
    _s = probabilities.filter(pl.col("representation") == confusion_choice.value)
    _counts = (
        _s.group_by("defective", "assigned")
        .agg(n=pl.len())
        .join(
            pl.DataFrame({"defective": list(CLASSES)}).join(
                pl.DataFrame({"assigned": list(CLASSES)}), how="cross"
            ),
            on=["defective", "assigned"],
            how="right",
        )
        .with_columns(pl.col("n").fill_null(0))
    )
    _base = alt.Chart(_counts).encode(
        x=alt.X("assigned:O", title="assigned grade"),
        y=alt.Y("defective:O", title="true defective grade"),
    )
    _heat = _base.mark_rect().encode(
        color=alt.Color("n:Q", scale=alt.Scale(scheme="blues"))
    )
    _text = _base.mark_text().encode(text="n:Q")
    # Split by cause (charter rule 5): which defect the fruit has, and how it is assigned.
    _cause = (
        _s.with_columns(
            cause=pl.when((pl.col("browning") >= 1) & (pl.col("cavity") >= 1))
            .then(pl.lit("both"))
            .when(pl.col("browning") >= 1)
            .then(pl.lit("browning only"))
            .when(pl.col("cavity") >= 1)
            .then(pl.lit("cavity only"))
            .otherwise(pl.lit("healthy"))
        )
        .group_by("cause", "defective")
        .agg(
            n=pl.len(),
            assigned_exact=(pl.col("assigned") == pl.col("defective")).mean().round(3),
            assigned_2_or_higher=(pl.col("assigned") >= 2).mean().round(3),
        )
        .sort("cause", "defective")
    )
    mo.vstack([(_heat + _text).properties(width=260, height=260), _cause])
    return


@app.cell(hide_code=True)
def _():
    save = mo.ui.run_button(label="Save this run to results/")
    save
    return (save,)


@app.cell(hide_code=True)
def _(boundaries, cfg, probabilities, save, summary):
    mo.stop(not save.value, mo.md("_Press save to write the run._"))
    _run_dir = new_run_dir(cfg.path("results_dir"), "ordinal_defective")
    probabilities.write_parquet(_run_dir / "probabilities.parquet")
    summary.write_parquet(_run_dir / "summary.parquet")
    boundaries.write_parquet(_run_dir / "boundaries_all_fruit.parquet")
    write_run_record(
        _run_dir,
        {
            "target": "defective",
            "input_runs": PLSDA_BINARY23_RUN_DIRS,
            "model": "statsmodels OrderedModel, distr logit, no constant, bfgs",
            "folds": "derived/outer_folds_20/outer_folds.csv",
            "assignment": "highest class probability, averaged over 20 repeats",
        },
    )
    mo.md(f"Saved to `{_run_dir}`.")
    return


if __name__ == "__main__":
    app.run()
