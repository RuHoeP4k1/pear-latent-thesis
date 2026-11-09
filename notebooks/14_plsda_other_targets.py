import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title="14 PLS-DA on the other targets")

with app.setup(hide_code=True):
    import json

    import altair as alt
    import marimo as mo
    import numpy as np
    import polars as pl
    from sklearn.metrics import roc_auc_score

    from pearlatent.config import load_config
    from pearlatent.feature_sets import load_features
    from pearlatent.fruit_table import drop_faulty_volumes, open_fruit
    from pearlatent.plsda import (
        CUTOFF,
        MAX_COMPONENTS,
        box_bootstrap_auc,
        classification_scores,
        cross_validated_predictions,
        youden_cutoff,
    )
    from pearlatent.runs import new_run_dir, write_run_record
    from pearlatent.targets import TARGETS, target_table


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 14 PLS-DA on the other targets of charter rule 6

    **Question.** Does the picture of notebook 08 hold on the other targets of charter rule 6:
    browning without cavity, and a severity score fitted on grade 0 against 3 only? (`binary_123`
    left out, Ruben, 9 November 2026.) The cavity-free targets show what is left when cavity cannot carry the score.

    **Design.** As notebook 08: the 12 open boxes, 20 repeats of the 4-fold box cross-validation,
    1 to 15 latent variables (LVs), centring only for cell grids and height-radius bins, scaling
    inside the fold for tissue features and Med3D. The LVs are chosen by Ruben at the elbow of the
    mean RMSECV curve.

    **Severity targets.** A fold model is fitted only on the grade 0 and grade 3 fruit of its
    training boxes (`pearlatent.targets`) and predicts every fruit of the held-out boxes. RMSECV
    and the main AUC use the grade 0 and 3 fruit only, the only fruit with a defined label. The
    AUC between neighbouring grades shows where grades 1 and 2 land.

    **Inputs.** As notebook 08. The sealed test boxes F, J and M are never loaded.

    **Outputs.** `results/<run_id>_plsda_<target>_<representation>/` (only when saved):
    per-fruit predictions, curves, scores and `run.json`.

    In script mode: `uv run python notebooks/14_plsda_other_targets.py -- --target severity_defective
    --representation hugo_vae --lv 3 --save yes`.

    **Status.** draft, 9 November 2026.
    """)
    return


@app.cell(hide_code=True)
def _():
    cfg = load_config()
    args = mo.cli_args()
    REPRESENTATIONS = {
        "hugo_vae": "Hugo's latent (mu, full 32³ grid)",
        "ct_32": "CT averaged to 32³",
        "tissue_features": "10 tissue features (Van De Looverbosch et al. 2020)",
        "med3d": "Med3D ResNet-18, 512 channels",
        "hugo_vae_hr": "Hugo's latent, 20 height × 8 radius bins",
        "ct_32_hr": "CT at 32³, 20 height × 8 radius bins",
        "hugo_vae_hr_sd": "Hugo's latent, 20 height × 8 radius bins, mean and SD",
        "ct_32_hr_sd": "CT at 32³, 20 height × 8 radius bins, mean and SD",
    }
    # Scaling as in notebook 08: cells and bins centred only; features in different units scaled.
    SCALED = ("tissue_features", "med3d")
    return REPRESENTATIONS, SCALED, args, cfg


@app.cell(hide_code=True)
def _(REPRESENTATIONS, args):
    target_choice = mo.ui.dropdown(
        options={_t.label: _k for _k, _t in TARGETS.items()},
        value=TARGETS[args.get("target", "browning_2plus_cavity_free")].label,
        label="Target",
    )
    representation_choice = mo.ui.dropdown(
        options={_label: _k for _k, _label in REPRESENTATIONS.items()},
        value=REPRESENTATIONS[args.get("representation", "hugo_vae")],
        label="Representation",
    )
    run_cv = mo.ui.run_button(label="Run the cross-validation")
    mo.hstack([target_choice, representation_choice, run_cv], justify="start")
    return representation_choice, run_cv, target_choice


@app.cell(hide_code=True)
def _(cfg, target_choice):
    # Open 2526 fruit without the faulty volumes (as notebook 08), then the target's own fruit.
    _open_2526 = open_fruit(cfg).filter(pl.col("season") == "2526")
    _kept = drop_faulty_volumes(_open_2526, cfg.path("derived_dir") / "volume_decisions.csv")
    print(f"open 2526 fruit {_open_2526.height}, after faulty volumes {_kept.height}")
    fruit = target_table(_kept, target_choice.value)
    print(
        f"label 1: {fruit['y'].sum()}; fruit a model may be fitted on: {fruit['fit_on'].sum()}"
    )
    fruit.group_by("grade").agg(n=pl.len()).sort("grade")
    return (fruit,)


@app.function(hide_code=True)
@mo.persistent_cache
def cross_validate_target(
    representation: str, target: str, fruit_ids: tuple, scale: bool
) -> np.ndarray:
    """Out-of-fold predictions (repeats, fruit, LVs) of one target and representation.

    In: the feature set, the target name, the fruit in table order, the scaling choice. Cached on
    disk. Delete `notebooks/__marimo__/cache` after changing the code in `src/pearlatent`.
    """
    cfg = load_config()
    kept = drop_faulty_volumes(
        open_fruit(cfg).filter(pl.col("season") == "2526"),
        cfg.path("derived_dir") / "volume_decisions.csv",
    )
    fruit = target_table(kept, target)
    assert tuple(fruit["fruit_id"].to_list()) == fruit_ids, "fruit order changed"
    x = load_features(cfg, representation, list(fruit_ids))
    folds = pl.read_csv(cfg.path("derived_dir") / "outer_folds_20" / "outer_folds.csv")
    return cross_validated_predictions(
        x,
        fruit["y"].to_numpy(),
        fruit["box"].to_numpy(),
        folds,
        scale=scale,
        # PLS has at most as many LVs as features: 10 for the tissue features.
        max_components=min(MAX_COMPONENTS, x.shape[1]),
        fit_on=fruit["fit_on"].to_numpy(),
    )


@app.cell(hide_code=True)
def _(SCALED, fruit, representation_choice, run_cv, target_choice):
    # In script mode the button is never pressed; the cross-validation then runs directly.
    mo.stop(
        mo.running_in_notebook() and not run_cv.value,
        mo.md("_Press the button to run the cross-validation._"),
    )
    predictions = cross_validate_target(
        representation_choice.value,
        target_choice.value,
        tuple(fruit["fruit_id"].to_list()),
        representation_choice.value in SCALED,
    )
    print(f"predictions: {predictions.shape} (repeats, fruit, LVs)")
    return (predictions,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1. Error curve: choose the number of latent variables

    RMSECV(h) per repeat (grey) and from PRESS averaged over the repeats (black), over the fruit
    with a defined label, divided by nothing: the dashed line is the RMSE of no model. Only these
    12 boxes are looked at.
    """)
    return


@app.cell(hide_code=True)
def _(fruit, predictions):
    # Severity: only grades 0 and 3 have a label, so only they enter the curve.
    _scored = fruit["fit_on"].to_numpy()
    _y = fruit["y"].to_numpy()[_scored].astype(float)
    _squared = (predictions[:, _scored, :] - _y[None, :, None]) ** 2
    _per_repeat = np.sqrt(_squared.mean(axis=1))
    _mean_press = _squared.mean(axis=0).sum(axis=0)
    _n_lv = np.arange(1, predictions.shape[2] + 1)
    no_model_rmse = float(_y.std())
    curves = pl.concat(
        [
            pl.DataFrame(
                {
                    "n_lv": np.tile(_n_lv, _per_repeat.shape[0]),
                    "error": _per_repeat.ravel(),
                    "curve": "RMSECV, one repeat",
                    "repeat": np.repeat(np.arange(1, _per_repeat.shape[0] + 1), _n_lv.size),
                }
            ),
            pl.DataFrame(
                {
                    "n_lv": _n_lv,
                    "error": np.sqrt(_mean_press / _y.size),
                    "curve": "RMSECV, mean PRESS",
                    "repeat": 0,
                }
            ),
        ],
        how="vertical_relaxed",
    )
    _lines = (
        alt.Chart(curves)
        .mark_line(point=True)
        .encode(
            x=alt.X("n_lv:O", title="number of latent variables"),
            y=alt.Y("error:Q", title="root mean squared error", scale=alt.Scale(zero=False)),
            color=alt.Color(
                "curve:N",
                scale=alt.Scale(
                    domain=["RMSECV, one repeat", "RMSECV, mean PRESS"],
                    range=["#bbbbbb", "#000000"],
                ),
            ),
            detail="repeat:N",
        )
    )
    _none = (
        alt.Chart(pl.DataFrame({"y": [no_model_rmse]}))
        .mark_rule(strokeDash=[4, 4])
        .encode(y="y:Q")
    )
    print(f"fruit in the curve: {_y.size}; RMSE of no model: {no_model_rmse:.3f}")
    alt.layer(_lines, _none).properties(width=560, height=320)
    return curves, no_model_rmse


@app.cell(hide_code=True)
def _(args):
    _options = ["not chosen"] + [str(_h) for _h in range(1, MAX_COMPONENTS + 1)]
    lv_choice = mo.ui.dropdown(
        options=_options,
        value=str(args.get("lv", "not chosen")),
        label="Number of latent variables (Ruben's choice at the elbow)",
    )
    lv_choice
    return (lv_choice,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2. Scores at the chosen number of latent variables

    The per-fruit average of the 20 out-of-fold predictions, as notebook 08. Beside it the mean
    AUC of the single repeats, which runs slightly lower. AUC between neighbouring grades on the
    same score. For a severity target the cut-offs mean little, and recall is left out.
    """)
    return


@app.cell(hide_code=True)
def _(fruit, lv_choice, predictions, target_choice):
    mo.stop(lv_choice.value == "not chosen", mo.md("_Choose the number of LVs first._"))
    chosen_lv = int(lv_choice.value)
    mo.stop(
        chosen_lv > predictions.shape[2],
        mo.md(f"_This representation has at most {predictions.shape[2]} LVs._"),
    )
    _at_h = predictions[:, :, chosen_lv - 1]
    mean_prediction = _at_h.mean(axis=0)
    _scored = fruit["fit_on"].to_numpy()
    _y = fruit["y"].to_numpy()
    _grade = fruit["grade"].to_numpy()
    _boxes = fruit["box"].to_numpy()
    _low, _high, _ = box_bootstrap_auc(mean_prediction[_scored], _y[_scored], _boxes[_scored])
    _neighbours = {}
    for _a, _b in [(0, 1), (1, 2), (2, 3), (0, 3)]:
        _pair = np.isin(_grade, [_a, _b])
        _neighbours[f"{_a}|{_b}"] = float(
            roc_auc_score(_grade[_pair] == _b, mean_prediction[_pair])
        )
    scores = {
        "auc": float(roc_auc_score(_y[_scored], mean_prediction[_scored])),
        "auc_box_bootstrap_95": [_low, _high],
        "auc_mean_of_repeats": float(
            np.mean([roc_auc_score(_y[_scored], _r[_scored]) for _r in _at_h])
        ),
        "auc_between_grades": _neighbours,
        "median_per_grade": {
            str(_g): float(np.median(mean_prediction[_grade == _g])) for _g in range(4)
        },
        "n_lv": chosen_lv,
    }
    cutoff_youden = None
    if TARGETS[target_choice.value].fit_grades is None:
        cutoff_youden = youden_cutoff(mean_prediction, _y)
        scores["at_0_5"] = classification_scores(mean_prediction, _y, _grade, CUTOFF)
        scores["youden"] = {
            "cutoff": cutoff_youden,
            **classification_scores(mean_prediction, _y, _grade, cutoff_youden),
        }
    print(json.dumps(scores, indent=1))
    mo.md(
        f"**{chosen_lv} LVs.** AUC {scores['auc']:.3f}, 95 % box-bootstrap interval {_low:.3f} to"
        f" {_high:.3f}; mean of the single repeats {scores['auc_mean_of_repeats']:.3f}."
    )
    return chosen_lv, cutoff_youden, mean_prediction, scores


@app.cell(hide_code=True)
def _(fruit, mean_prediction):
    predictions_table = fruit.select(
        "fruit_id", "box", "grade", "defective", "browning", "cavity", "cause", "y", "fit_on"
    ).with_columns(prediction=mean_prediction)
    # Charter rule 5: split by cause, where a target holds more than one cause.
    (
        predictions_table.group_by("cause", "grade")
        .agg(n=pl.len(), median_score=pl.col("prediction").median().round(3))
        .sort("cause", "grade")
    )
    return (predictions_table,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 3. Where every grade lands on the score

    One dot per fruit (out-of-fold, average of the 20 repeats), coloured by cause; the box shows
    the median and quartiles of the grade. Dashed lines: the median of grade 0 and of grade 3.
    For a severity score this is the main picture: the model only saw grades 0 and 3, so a grade
    1 box between the two lines, apart from grade 0, means grade 1 is seen; a grade 1 box on top
    of grade 0 means it is not.
    """)
    return


@app.cell(hide_code=True)
def _(predictions_table):
    _base = alt.Chart(predictions_table).encode(
        x=alt.X("grade:O", title="grade of the target"),
    )
    _boxes = _base.mark_boxplot(opacity=0.35, size=40, outliers=False).encode(
        y=alt.Y("prediction:Q", title="out-of-fold score")
    )
    _dots = (
        _base.mark_circle(size=22, opacity=0.7)
        .encode(
            y="prediction:Q",
            xOffset=alt.XOffset("jitter:Q"),
            color=alt.Color("cause:N", title="cause"),
            tooltip=["fruit_id", "box", "browning", "cavity", "prediction"],
        )
        .transform_calculate(jitter="random()")
    )
    _medians = pl.DataFrame(
        {
            "y": [
                predictions_table.filter(pl.col("grade") == _g)["prediction"].median()
                for _g in (0, 3)
            ],
            "line": ["median grade 0", "median grade 3"],
        }
    )
    _lines = (
        alt.Chart(_medians)
        .mark_rule(strokeDash=[4, 4])
        .encode(y="y:Q", tooltip=["line", "y"])
    )
    alt.layer(_boxes, _dots, _lines).properties(width=420, height=320)
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
    args,
    cfg,
    chosen_lv,
    curves,
    cutoff_youden,
    no_model_rmse,
    predictions_table,
    representation_choice,
    save_button,
    scores,
    target_choice,
):
    mo.stop(
        not (save_button.value or args.get("save") == "yes"), mo.md("_Not saved._")
    )
    _run_dir = new_run_dir(
        cfg.path("results_dir"),
        f"plsda_{target_choice.value}_{representation_choice.value}",
    )
    predictions_table.write_parquet(_run_dir / "predictions.parquet")
    curves.write_parquet(_run_dir / "curves.parquet")
    (_run_dir / "scores.json").write_text(json.dumps(scores, indent=2))
    write_run_record(
        _run_dir,
        {
            "what": "PLS-DA on another target of charter rule 6, design A, 12 open 2526 boxes",
            "target": target_choice.value,
            "target_label": TARGETS[target_choice.value].label,
            "representation": representation_choice.value,
            "representation_label": REPRESENTATIONS[representation_choice.value],
            "n_fruit": predictions_table.height,
            "n_fruit_fitted_on": int(predictions_table["fit_on"].sum()),
            "n_lv_chosen": chosen_lv,
            "lv_rule": "elbow of the mean RMSECV curve, chosen by Ruben by eye",
            "scale": representation_choice.value in SCALED,
            "max_components": MAX_COMPONENTS,
            "no_model_rmse": no_model_rmse,
            "cutoff": CUTOFF,
            "cutoff_youden": cutoff_youden,
            "folds": "derived/outer_folds_20/outer_folds.csv",
            "bootstrap_seed": 369,
        },
        root=cfg.root,
    )
    print(f"saved to {_run_dir}")
    return


if __name__ == "__main__":
    app.run()
