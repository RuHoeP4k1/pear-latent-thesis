import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title="11 PLS on the defective grade")

with app.setup(hide_code=True):
    import json

    import altair as alt
    import marimo as mo
    import numpy as np
    import polars as pl
    from scipy.stats import spearmanr
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.decomposition import PCA
    from sklearn.metrics import cohen_kappa_score, roc_auc_score

    from pearlatent.config import load_config
    from pearlatent.feature_sets import load_features
    from pearlatent.fruit_table import CAUSES, drop_faulty_volumes, open_fruit
    from pearlatent.nuisance import box_bootstrap
    from pearlatent.plsda import (
        BOOTSTRAP_SEED,
        MAX_COMPONENTS,
        cross_validated_predictions,
        mean_squared_errors,
        out_of_fold_scores,
        predict_each_h,
    )
    from pearlatent.runs import new_run_dir, write_run_record


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 11 PLS on the defective grade: does the prediction follow the severity?

    **Question.** Can PLS regression on the defective grade (0 to 3, the maximum of browning
    and cavity) order the fruit by severity, on Hugo's latent, on CT at 32³, on the tissue
    features and on Med3D? Charter rule 5 (`defective` first, split by cause) and rule 6
    (report per grade).

    **Why not the PLS-DA scores of notebook 08.** PLS-DA on `binary_23` gives grade 0 and
    grade 1 the same target (0) and grades 2 and 3 the same target (1). Its components are
    never asked to separate 0 from 1, or 2 from 3. Here the target is the grade itself, so
    the components are fitted to the order of the grades.

    **Design.** As notebook 08 (design A, `docs/cv_explained.md`, sections 4 to 7): the 315
    open 2526 fruit after the faulty volumes, the 12 open boxes, the same 20 repeats of a
    4-fold box cross-validation, 1 to 15 latent variables (LVs), centring only for the cell
    grids and unit-variance scaling inside each fold for the tissue features and Med3D.
    Target `defective` as a number (0, 1, 2, 3): the distance between two neighbouring
    grades counts as 1. Number of LVs: chosen by Ruben at the elbow of the mean RMSECV curve.

    **Scores** (fixed before looking):
    - Spearman correlation between the out-of-fold prediction and the grade, with a 95 %
      interval from 2,000 draws of boxes (seed 369).
    - Quadratic weighted kappa between the grade and the prediction rounded to the nearest
      grade and clipped to 0 to 3 (rounding rule fixed in advance; the grade cut-points are
      not tuned).
    - AUC of the same prediction for `binary_23`, to compare with notebook 08.
    - Per true grade, split by cause: the mean prediction and the share assigned to each
      rounded grade.

    **Inputs.** As notebook 08. The sealed test boxes F, J and M are never loaded.

    **Outputs.** `results/<run_id>_pls_grade_<representation>/` (only when the save button
    is pressed): per-fruit predictions, the curves, the scores and `run.json`.

    **Status.** draft (7 November 2026)
    """)
    return


@app.cell(hide_code=True)
def _():
    cfg = load_config()
    return (cfg,)


@app.cell(hide_code=True)
def _(cfg):
    # The same fruit as notebook 08: open 2526 fruit without the faulty volumes.
    _open_2526 = open_fruit(cfg).filter(pl.col("season") == "2526")
    fruit = drop_faulty_volumes(
        _open_2526, cfg.path("derived_dir") / "volume_decisions.csv"
    ).with_columns(binary_23=(pl.col("defective") >= 2).cast(pl.Int8))
    print(f"open 2526 fruit {_open_2526.height}, after faulty volumes {fruit.height}")
    _cause_order = {_c: _k for _k, _c in enumerate(CAUSES)}
    (
        fruit.group_by("defective", "cause")
        .agg(n=pl.len())
        .with_columns(_order=pl.col("cause").replace_strict(_cause_order))
        .sort("defective", "_order")
        .drop("_order")
    )
    return (fruit,)


@app.cell(hide_code=True)
def _(cfg, fruit):
    # The 20 partitions of the 12 open boxes into 4 folds of 3 (as notebook 08).
    folds = pl.read_csv(cfg.path("derived_dir") / "outer_folds_20" / "outer_folds.csv")
    if set(folds["box"].to_list()) != set(fruit["box"].to_list()):
        raise ValueError("the boxes of the folds and of the fruit differ")
    print(f"{folds['repeat'].n_unique()} repeats over {folds['box'].n_unique()} boxes")
    return (folds,)


@app.cell(hide_code=True)
def _():
    REPRESENTATIONS = {
        "hugo_vae": "Hugo's latent (mu, full 32³ grid)",
        "ct_32": "CT averaged to 32³",
        "tissue_features": "10 tissue features (Van De Looverbosch et al. 2020)",
        "med3d": "Med3D ResNet-18, 512 channels",
    }
    # Scaling as in notebook 08: cells centred only; features in different units scaled.
    SCALED = ("tissue_features", "med3d")
    representation_choice = mo.ui.dropdown(
        options={_label: _key for _key, _label in REPRESENTATIONS.items()},
        value=REPRESENTATIONS["hugo_vae"],
        label="Representation",
    )
    run_cv = mo.ui.run_button(label="Run the cross-validation (about 10 seconds)")
    mo.hstack([representation_choice, run_cv], justify="start")
    return REPRESENTATIONS, SCALED, representation_choice, run_cv


@app.function(hide_code=True)
@mo.persistent_cache
def cross_validate_grade(representation: str, fruit_ids: tuple, scale: bool) -> dict:
    """Out-of-fold predictions of the grade and the calibration curve for one representation.

    In: the feature set name, the fruit (in table order), the scaling choice. Out: a dict
    with `predictions` (repeats × fruit × LVs) and `rmsec` (one per number of LVs, from one
    fit on all fruit). Cached on disk, because the inputs are fixed.
    """
    cfg = load_config()
    fruit = drop_faulty_volumes(
        open_fruit(cfg).filter(pl.col("season") == "2526"),
        cfg.path("derived_dir") / "volume_decisions.csv",
    ).filter(pl.col("fruit_id").is_in(list(fruit_ids)))
    assert tuple(fruit["fruit_id"].to_list()) == fruit_ids, "fruit order changed"
    x = load_features(cfg, representation, list(fruit_ids))
    y = fruit["defective"].to_numpy().astype(float)
    folds = pl.read_csv(cfg.path("derived_dir") / "outer_folds_20" / "outer_folds.csv")
    # PLS has at most as many LVs as features: 10 for the tissue features, 15 otherwise.
    n_components = min(MAX_COMPONENTS, x.shape[1])
    predictions = cross_validated_predictions(
        x, y, fruit["box"].to_numpy(), folds, scale=scale, max_components=n_components
    )
    # RMSEC: one model on all 12 boxes, error on the same fruit. It always goes down with
    # more LVs, so it is plotted for reference and does not decide.
    fitted = predict_each_h(
        PLSRegression(n_components=n_components, scale=scale).fit(x, y), x
    )
    rmsec = np.sqrt(((fitted - y[:, None]) ** 2).mean(axis=0))
    return {"predictions": predictions, "rmsec": rmsec}


@app.cell(hide_code=True)
def _(SCALED, fruit, representation_choice, run_cv):
    # In script mode the button is never pressed; the cross-validation then runs directly.
    mo.stop(
        mo.running_in_notebook() and not run_cv.value,
        mo.md("_Press the button to run the cross-validation._"),
    )
    cv = cross_validate_grade(
        representation_choice.value,
        tuple(fruit["fruit_id"].to_list()),
        representation_choice.value in SCALED,
    )
    grade = fruit["defective"].to_numpy()
    print(
        f"{representation_choice.value}: predictions for {cv['predictions'].shape[0]}"
        f" repeats × {cv['predictions'].shape[1]} fruit × {cv['predictions'].shape[2]} LVs"
    )
    return cv, grade


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1. Error curves: choose the number of latent variables

    RMSECV(h) = √(PRESS(h) / n), in grades, for every repeat (grey) and for PRESS averaged
    over the 20 repeats (black); RMSEC (blue) from one fit on all fruit. The choice is the
    elbow: the number of LVs after which the black curve stops going down. For reference:
    predicting the mean grade for every fruit gives an RMSE equal to the standard deviation
    of the grade (printed under the table).
    """)
    return


@app.cell(hide_code=True)
def _(cv, grade):
    _squared = (cv["predictions"] - grade[None, :, None].astype(float)) ** 2
    _per_repeat = np.sqrt(_squared.mean(axis=1))  # (repeats, LVs)
    _mean_press = mean_squared_errors(cv["predictions"], grade).sum(axis=0)
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
                    "error": np.sqrt(_mean_press / len(grade)),
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
            y=alt.Y(
                "error:Q",
                title="root mean squared error (grades)",
                scale=alt.Scale(zero=False),
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
    mo.vstack(
        [
            _chart,
            _table,
            mo.md(
                f"Standard deviation of the grade (RMSE of predicting the mean): {grade.std():.4f}"
            ),
        ]
    )
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
    each repeat shows the spread caused by the partition (not a confidence interval).
    """)
    return


@app.function(hide_code=True)
def rounded_grade(prediction: np.ndarray) -> np.ndarray:
    """Prediction rounded to the nearest whole grade and clipped to 0 to 3.

    Example: 1.49 → 1, 1.51 → 2, −0.3 → 0, 3.4 → 3. Fixed in advance; no tuned cut-points.
    """
    return np.clip(np.rint(prediction), 0, 3).astype(int)


@app.function(hide_code=True)
def grade_scores(prediction: np.ndarray, grade: np.ndarray) -> dict:
    """In: predictions and true grades of the same fruit. Out: Spearman, kappa, AUC, RMSE."""
    return {
        "spearman": float(spearmanr(prediction, grade).statistic),
        "quadratic_weighted_kappa": float(
            cohen_kappa_score(grade, rounded_grade(prediction), weights="quadratic")
        ),
        "auc_binary_23": float(roc_auc_score(grade >= 2, prediction)),
        "rmse": float(np.sqrt(((prediction - grade) ** 2).mean())),
    }


@app.cell(hide_code=True)
def _(cv, fruit, grade, lv_choice):
    mo.stop(lv_choice.value == "not chosen", mo.md("_Choose the number of LVs first._"))
    chosen_lv = int(lv_choice.value)
    mo.stop(
        chosen_lv > cv["predictions"].shape[2],
        mo.md(f"_This representation has at most {cv['predictions'].shape[2]} LVs._"),
    )
    _at_h = cv["predictions"][:, :, chosen_lv - 1]  # (repeats, fruit)
    mean_prediction = _at_h.mean(axis=0)
    _main = grade_scores(mean_prediction, grade)
    _per_repeat = pl.DataFrame(
        [grade_scores(_at_h[_r], grade) for _r in range(_at_h.shape[0])]
    )

    # Box bootstrap of the Spearman correlation: whole boxes drawn with replacement.
    _table = fruit.select("box", "defective").with_columns(prediction=mean_prediction)
    _low, _high = box_bootstrap(
        lambda t: spearmanr(t["prediction"], t["defective"]).statistic, _table
    )
    scores = {**_main, "spearman_box_bootstrap_95": [_low, _high], "n_lv": chosen_lv}
    _summary = pl.DataFrame(
        {
            "score": list(_main),
            "average of 20 predictions": list(_main.values()),
            "repeats: min": [_per_repeat[_k].min() for _k in _main],
            "repeats: max": [_per_repeat[_k].max() for _k in _main],
        }
    )
    mo.vstack(
        [
            mo.md(
                f"**{chosen_lv} LVs.** Spearman {_main['spearman']:.3f}, 95 % box-bootstrap"
                f" interval {_low:.3f} to {_high:.3f} (2,000 draws, seed {BOOTSTRAP_SEED})."
            ),
            _summary,
        ]
    )
    return chosen_lv, mean_prediction, scores


@app.cell(hide_code=True)
def _(fruit, mean_prediction):
    # Charter rule 5: per true grade, split by cause. Columns `as_0` to `as_3`: the share of
    # the fruit whose rounded prediction is that grade (each row sums to 1).
    predictions_table = fruit.select(
        "fruit_id", "box", "defective", "browning", "cavity", "cause", "binary_23"
    ).with_columns(prediction=mean_prediction, rounded=rounded_grade(mean_prediction))
    _cause_order = {_c: _k for _k, _c in enumerate(CAUSES)}
    (
        predictions_table.group_by("cause", "defective")
        .agg(
            n=pl.len(),
            mean_prediction=pl.col("prediction").mean().round(2),
            sd_prediction=pl.col("prediction").std().round(2),
            **{
                f"as_{_g}": (pl.col("rounded") == _g).mean().round(2) for _g in range(4)
            },
        )
        .with_columns(_order=pl.col("cause").replace_strict(_cause_order))
        .sort("_order", "defective")
        .drop("_order")
    )
    return (predictions_table,)


@app.cell(hide_code=True)
def _(predictions_table):
    _ticks = (
        alt.Chart(predictions_table)
        .mark_tick(thickness=2)
        .encode(
            x=alt.X("prediction:Q", title="out-of-fold predicted grade"),
            y=alt.Y("defective:O", title="defective grade"),
            color=alt.Color("cause:N", sort=list(CAUSES)),
            tooltip=[
                "fruit_id",
                "box",
                "browning",
                "cavity",
                alt.Tooltip("prediction:Q", format=".2f"),
            ],
        )
        .properties(width=560, height=200)
    )
    _rules = (
        alt.Chart(pl.DataFrame({"x": [0.5, 1.5, 2.5]}))
        .mark_rule(strokeDash=[4, 4], color="grey")
        .encode(x="x:Q")
    )
    mo.vstack(
        [
            mo.md(
                "Out-of-fold prediction per fruit (average of 20 repeats). Dashed lines: the"
                " rounding boundaries 0.5, 1.5, 2.5."
            ),
            _ticks + _rules,
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 3. Score plots: out-of-fold PLS scores beside PCA

    **Left, PLS.** The scores t1 and t2 of every fruit, from the model of the fold in which
    its box was held out (`pearlatent.plsda.out_of_fold_scores`, repeat 1, at the chosen
    number of LVs). Not the scores of one model fitted on all fruit: with 32,768 cells and
    315 fruit, those separate the grades almost by construction. A sign flip of a component
    between folds is undone by comparing each fold's weights with one model fitted on all
    fruit.

    **Right, PCA.** PC1 and PC2 of the same representation on the same 315 fruit (no labels
    used, so no fold is needed), centred, and scaled for the tissue features and Med3D, as
    in the PLS. This is the view of notebook 07 (last week) for these fruit: the components
    that explain most variance, whatever their relation to the grade.
    """)
    return


@app.cell(hide_code=True)
def _():
    colour_choice = mo.ui.dropdown(
        options=["defective", "cause", "box", "browning", "cavity"],
        value="defective",
        label="Colour by",
    )
    colour_choice
    return (colour_choice,)


@app.cell(hide_code=True)
def _(SCALED, cfg, chosen_lv, folds, fruit, representation_choice):
    _x = load_features(cfg, representation_choice.value, fruit["fruit_id"].to_list())
    _scale = representation_choice.value in SCALED
    # At least 2 components, so that t2 exists even when 1 LV was chosen.
    _pls = out_of_fold_scores(
        _x,
        fruit["defective"].to_numpy(),
        fruit["box"].to_numpy(),
        folds,
        repeat=1,
        scale=_scale,
        n_components=max(chosen_lv, 2),
    )
    _x_pca = (_x - _x.mean(axis=0)) / (_x.std(axis=0) if _scale else 1.0)
    _pca = PCA(n_components=2, random_state=0).fit(_x_pca)
    _pcs = _pca.transform(_x_pca)
    print(
        "PCA explained variance PC1, PC2:",
        np.round(_pca.explained_variance_ratio_, 3).tolist(),
    )
    score_table = fruit.select(
        "fruit_id", "box", "defective", "cause", "browning", "cavity"
    ).with_columns(t1=_pls[:, 0], t2=_pls[:, 1], pc1=_pcs[:, 0], pc2=_pcs[:, 1])
    return (score_table,)


@app.cell(hide_code=True)
def _(colour_choice, score_table):
    _is_grade = colour_choice.value in ("defective", "browning", "cavity")
    _colour = alt.Color(
        f"{colour_choice.value}:{'O' if _is_grade else 'N'}",
        scale=alt.Scale(scheme="viridis") if _is_grade else alt.Undefined,
    )
    _tooltip = ["fruit_id", "box", "defective", "cause"]

    def _panel(x, y, title):
        return (
            alt.Chart(score_table)
            .mark_circle(size=30, opacity=0.8)
            .encode(x=f"{x}:Q", y=f"{y}:Q", color=_colour, tooltip=_tooltip)
            .properties(width=320, height=300, title=title)
        )

    _panel("t1", "t2", "PLS on the grade, out-of-fold scores") | _panel(
        "pc1", "pc2", "PCA, same fruit"
    )
    return


@app.cell(hide_code=True)
def _():
    save_button = mo.ui.run_button(label="Save this run to results/")
    save_button
    return (save_button,)


@app.cell(hide_code=True)
def _(
    REPRESENTATIONS,
    SCALED,
    cfg,
    chosen_lv,
    curves,
    predictions_table,
    representation_choice,
    save_button,
    score_table,
    scores,
):
    mo.stop(not save_button.value, mo.md("_Not saved._"))
    _run_dir = new_run_dir(
        cfg.path("results_dir"), f"pls_grade_{representation_choice.value}"
    )
    predictions_table.write_parquet(_run_dir / "predictions.parquet")
    score_table.write_parquet(_run_dir / "score_plot_repeat1.parquet")
    curves.write_parquet(_run_dir / "curves.parquet")
    (_run_dir / "scores.json").write_text(json.dumps(scores, indent=2))
    write_run_record(
        _run_dir,
        {
            "what": "PLS regression on the defective grade (0 to 3), design A, 12 open 2526 boxes",
            "representation": representation_choice.value,
            "representation_label": REPRESENTATIONS[representation_choice.value],
            "n_fruit": predictions_table.height,
            "n_lv_chosen": chosen_lv,
            "lv_rule": "elbow of the mean RMSECV curve, chosen by Ruben by eye",
            "scale": representation_choice.value in SCALED,
            "max_components": MAX_COMPONENTS,
            "rounding": "nearest grade, clipped to 0 to 3, for the kappa",
            "folds": "derived/outer_folds_20/outer_folds.csv",
            "score_plot_repeat": 1,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        root=cfg.root,
    )
    mo.md(f"Saved to `{_run_dir}`.")
    return


if __name__ == "__main__":
    app.run()
