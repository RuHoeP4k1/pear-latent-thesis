import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title="11 PLS-DA per grade")

with app.setup(hide_code=True):
    import json

    import altair as alt
    import marimo as mo
    import numpy as np
    import polars as pl
    from sklearn.metrics import roc_auc_score

    from pearlatent.config import load_config
    from pearlatent.feature_sets import load_features
    from pearlatent.fruit_table import CAUSES, drop_faulty_volumes, open_fruit
    from pearlatent.plsda import (
        MAX_COMPONENTS,
        cross_validated_predictions,
        mean_squared_errors,
    )
    from pearlatent.runs import new_run_dir, write_run_record


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 11 PLS-DA with one model per grade

    **Question.** Can the four grades be told apart with PLS-DA, one class per grade, on
    Hugo's latent, CT at 32³, the tissue features and Med3D? Charter rules 5 and 6.

    **Method** (Ruben, 7 November 2026; chemometrics course, chapter 4, "Discrimination via
    regression"). One 0/1 dummy variable per class: for grade k, 1 when the fruit has grade
    k, 0 otherwise. One PLS model per dummy (one-versus-rest), each with its own number of
    latent variables (LVs), chosen by Ruben at the elbow of its mean RMSECV curve. A fruit
    is placed in the class whose model gives the highest predicted membership. Every
    prediction is out-of-fold: the same 20 repeats of the 4-fold box cross-validation as
    notebook 08 (design A, `docs/cv_explained.md`), centring only for the cell grids,
    unit-variance scaling inside each fold for the tissue features and Med3D.

    **Targets** (first dropdown): the browning grade on the cavity-free fruit (cavity grade
    0), or the defective grade on all fruit (results split by cause).

    **Scores** (course chapter 4, slide "Validation of classification rules"): the error
    rate per class and in total, the 4 × 4 table of true against assigned class, and per
    class the AUC of its own model (that class against the rest).

    **Inputs.** As notebook 08. The sealed test boxes F, J and M are never loaded.

    **Outputs.** `results/<run_id>_plsda_classes_<target>_<representation>/` (only when the
    save button is pressed): per-fruit predictions of every class model, the curves, the
    scores and `run.json`.

    **Status.** draft (7 November 2026)
    """)
    return


@app.cell(hide_code=True)
def _():
    cfg = load_config()
    CLASSES = [0, 1, 2, 3]
    return CLASSES, cfg


@app.cell(hide_code=True)
def _():
    TARGETS = {
        "browning": "browning grade, cavity-free fruit",
        "defective": "defective grade, all fruit",
    }
    REPRESENTATIONS = {
        "hugo_vae": "Hugo's latent (mu, full 32³ grid)",
        "ct_32": "CT averaged to 32³",
        "tissue_features": "10 tissue features (Van De Looverbosch et al. 2020)",
        "med3d": "Med3D ResNet-18, 512 channels",
    }
    # Scaling as in notebook 08: cells centred only; features in different units scaled.
    SCALED = ("tissue_features", "med3d")
    target_choice = mo.ui.dropdown(
        options={_label: _key for _key, _label in TARGETS.items()},
        value=TARGETS["browning"],
        label="Target",
    )
    representation_choice = mo.ui.dropdown(
        options={_label: _key for _key, _label in REPRESENTATIONS.items()},
        value=REPRESENTATIONS["hugo_vae"],
        label="Representation",
    )
    run_cv = mo.ui.run_button(label="Run the cross-validation (four models)")
    mo.hstack([target_choice, representation_choice, run_cv], justify="start")
    return (
        REPRESENTATIONS,
        SCALED,
        TARGETS,
        representation_choice,
        run_cv,
        target_choice,
    )


@app.cell(hide_code=True)
def _(cfg, target_choice):
    # Open 2526 fruit without the faulty volumes (as notebook 08). For the browning target,
    # only fruit with cavity grade 0. `target` is the grade that defines the classes.
    _open_2526 = open_fruit(cfg).filter(pl.col("season") == "2526")
    _all = drop_faulty_volumes(
        _open_2526, cfg.path("derived_dir") / "volume_decisions.csv"
    )
    _kept = (
        _all.filter(pl.col("cavity") == 0)
        if target_choice.value == "browning"
        else _all
    )
    fruit = _kept.with_columns(target=pl.col(target_choice.value))
    print(f"open 2526 fruit {_open_2526.height}, after faulty volumes {_all.height}")
    print(
        f"target {target_choice.value}: {fruit.height} fruit, {fruit['box'].n_unique()} boxes"
    )
    fruit.group_by("target").agg(n=pl.len()).sort("target")
    return (fruit,)


@app.function(hide_code=True)
@mo.persistent_cache
def cross_validate_classes(
    representation: str, fruit_ids: tuple, scale: bool, target: str, classes: tuple
) -> dict:
    """Out-of-fold predictions of every one-versus-rest class model.

    In: the feature set, the fruit (in table order), the scaling choice, the grade column
    that defines the classes, the classes. Out: {class: array (repeats, fruit, LVs)}: for
    every repeat of the box cross-validation, every fruit and every number of LVs, the
    predicted membership of that class. Each class has its own PLS model, fitted on the
    0/1 dummy of that class. Cached on disk, because the inputs are fixed.
    """
    cfg = load_config()
    fruit = drop_faulty_volumes(
        open_fruit(cfg).filter(pl.col("season") == "2526"),
        cfg.path("derived_dir") / "volume_decisions.csv",
    ).filter(pl.col("fruit_id").is_in(list(fruit_ids)))
    assert tuple(fruit["fruit_id"].to_list()) == fruit_ids, "fruit order changed"
    x = load_features(cfg, representation, list(fruit_ids))
    folds = pl.read_csv(cfg.path("derived_dir") / "outer_folds_20" / "outer_folds.csv")
    # PLS has at most as many LVs as features: 10 for the tissue features, 15 otherwise.
    n_components = min(MAX_COMPONENTS, x.shape[1])
    grade = fruit[target].to_numpy()
    return {
        k: cross_validated_predictions(
            x,
            (grade == k).astype(float),
            fruit["box"].to_numpy(),
            folds,
            scale=scale,
            max_components=n_components,
        )
        for k in classes
    }


@app.cell(hide_code=True)
def _(CLASSES, SCALED, fruit, representation_choice, run_cv, target_choice):
    # In script mode the button is never pressed; the cross-validation then runs directly.
    mo.stop(
        mo.running_in_notebook() and not run_cv.value,
        mo.md("_Press the button to run the cross-validation._"),
    )
    class_predictions = cross_validate_classes(
        representation_choice.value,
        tuple(fruit["fruit_id"].to_list()),
        representation_choice.value in SCALED,
        target_choice.value,
        tuple(CLASSES),
    )
    grade = fruit["target"].to_numpy()
    _shape = class_predictions[CLASSES[0]].shape
    print(
        f"{len(CLASSES)} class models, each: {_shape[0]} repeats × {_shape[1]} fruit × {_shape[2]} LVs"
    )
    return class_predictions, grade


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1. One error curve per class: choose the number of LVs of each class model

    Per class: RMSECV(h) of its 0/1 dummy, for PRESS averaged over the 20 repeats (black,
    left axis), and, as a check on the elbow, the error rate of that model alone at a
    cut-off of 0.5 (orange, right axis: share of fruit on the wrong side of 0.5 for "this
    class or not"). The RMSECV curve decides (as in notebook 08); the error-rate curve shows
    whether the classification would choose differently.
    """)
    return


@app.cell(hide_code=True)
def _(CLASSES, class_predictions, grade):
    _rows = []
    for _k in CLASSES:
        _dummy = (grade == _k).astype(float)
        _press = mean_squared_errors(class_predictions[_k], _dummy).sum(axis=0)
        _mean_prediction = class_predictions[_k].mean(axis=0)  # (fruit, LVs)
        _wrong = (_mean_prediction >= 0.5) != (_dummy[:, None] == 1)
        for _h in range(_press.size):
            _rows.append(
                {
                    "class": _k,
                    "n_lv": _h + 1,
                    "rmsecv": float(np.sqrt(_press[_h] / _dummy.size)),
                    "error_rate_at_0_5": float(_wrong[:, _h].mean()),
                    "share_in_class": float(_dummy.mean()),
                }
            )
    class_curves = pl.DataFrame(_rows)

    def _panel(k):
        _data = class_curves.filter(pl.col("class") == k)
        _base = alt.Chart(_data).encode(x=alt.X("n_lv:O", title="number of LVs"))
        _rmse = _base.mark_line(point=True, color="black").encode(
            y=alt.Y("rmsecv:Q", title="RMSECV", scale=alt.Scale(zero=False))
        )
        _error = _base.mark_line(point=True, color="#e6550d").encode(
            y=alt.Y(
                "error_rate_at_0_5:Q",
                title="error rate at 0.5",
                scale=alt.Scale(zero=False),
            )
        )
        _share = _data["share_in_class"][0]
        return (
            alt.layer(_rmse, _error)
            .resolve_scale(y="independent")
            .properties(
                width=260,
                height=200,
                title=f"grade {k} against the rest ({_share:.0%} in class)",
            )
        )

    mo.vstack(
        [
            alt.vconcat(
                alt.hconcat(_panel(CLASSES[0]), _panel(CLASSES[1])),
                alt.hconcat(_panel(CLASSES[2]), _panel(CLASSES[3])),
            ),
            class_curves.pivot(on="class", index="n_lv", values="rmsecv").rename(
                {str(_k): f"rmsecv grade {_k}" for _k in CLASSES}
            ),
        ]
    )
    return (class_curves,)


@app.cell(hide_code=True)
def _(CLASSES):
    _options = ["not chosen"] + [str(_h) for _h in range(1, MAX_COMPONENTS + 1)]
    lv_choices = mo.ui.dictionary(
        {
            f"grade {_k}": mo.ui.dropdown(options=_options, value="not chosen")
            for _k in CLASSES
        }
    )
    mo.vstack(
        [
            mo.md("**Number of LVs per class model** (Ruben's choice at the elbow)"),
            lv_choices,
        ]
    )
    return (lv_choices,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2. Assignment to the class with the highest predicted membership

    For every fruit: the prediction of each class model at its chosen number of LVs,
    averaged over the 20 repeats; the fruit is placed in the class with the highest of the
    four. The four predictions need not sum to 1 (four separate models).
    """)
    return


@app.cell(hide_code=True)
def _(CLASSES, class_predictions, fruit, grade, lv_choices):
    mo.stop(
        any(_v == "not chosen" for _v in lv_choices.value.values()),
        mo.md("_Choose the number of LVs of every class model first._"),
    )
    chosen_lvs = {_k: int(lv_choices.value[f"grade {_k}"]) for _k in CLASSES}
    _max = class_predictions[CLASSES[0]].shape[2]
    mo.stop(
        any(_h > _max for _h in chosen_lvs.values()),
        mo.md(f"_This representation has at most {_max} LVs._"),
    )
    # (fruit, class): each class model at its own LVs, averaged over the repeats.
    _membership = np.column_stack(
        [class_predictions[_k][:, :, chosen_lvs[_k] - 1].mean(axis=0) for _k in CLASSES]
    )
    _assigned = np.array(CLASSES)[_membership.argmax(axis=1)]
    predictions_table = fruit.select(
        "fruit_id", "box", "target", "defective", "browning", "cavity", "cause"
    ).with_columns(
        assigned=_assigned,
        **{f"membership_{_k}": _membership[:, _i] for _i, _k in enumerate(CLASSES)},
    )
    per_class = pl.DataFrame(
        [
            {
                "class": _k,
                "n": int((grade == _k).sum()),
                "n_lv": chosen_lvs[_k],
                "error_rate": float((_assigned[grade == _k] != _k).mean()),
                "auc_own_model": float(roc_auc_score(grade == _k, _membership[:, _i])),
            }
            for _i, _k in enumerate(CLASSES)
        ]
    )
    total_error_rate = float((_assigned != grade).mean())
    # Error rate of always assigning the largest class, for reference.
    _largest_share = max((grade == _k).mean() for _k in CLASSES)
    mo.vstack(
        [
            mo.md(
                f"**Total error rate {total_error_rate:.3f}** (always assigning the largest"
                f" class would give {1 - _largest_share:.3f})."
            ),
            per_class,
        ]
    )
    return chosen_lvs, per_class, predictions_table, total_error_rate


@app.cell(hide_code=True)
def _(CLASSES, predictions_table):
    # Rows: true class; columns: assigned class; each row as counts and as a share.
    confusion = (
        predictions_table.group_by("target", "assigned")
        .agg(n=pl.len())
        .with_columns(share=(pl.col("n") / pl.col("n").sum().over("target")).round(2))
    )
    _grid = pl.DataFrame(
        {
            "target": np.repeat(CLASSES, len(CLASSES)),
            "assigned": np.tile(CLASSES, len(CLASSES)),
        }
    ).cast(pl.Int64)
    _full = _grid.join(
        confusion.cast({"target": pl.Int64, "assigned": pl.Int64}),
        on=["target", "assigned"],
        how="left",
    ).fill_null(0)
    _base = alt.Chart(_full).encode(
        x=alt.X("assigned:O", title="assigned class"),
        y=alt.Y("target:O", title="true class"),
    )
    _cells = _base.mark_rect().encode(
        color=alt.Color(
            "share:Q",
            scale=alt.Scale(scheme="blues", domain=[0, 1]),
            title="share of row",
        )
    )
    _text = _base.mark_text().encode(text="n:Q")
    mo.vstack(
        [
            mo.md(
                "True against assigned class (number of fruit; colour: share of the row)."
            ),
            (_cells + _text).properties(width=260, height=260),
        ]
    )
    return (confusion,)


@app.cell(hide_code=True)
def _(predictions_table):
    # Charter rule 5: for the defective target, the same split by cause.
    _cause_order = {_c: _k for _k, _c in enumerate(CAUSES)}
    (
        predictions_table.group_by("cause", "target")
        .agg(
            n=pl.len(),
            correct=(pl.col("assigned") == pl.col("target")).mean().round(2),
            **{
                f"as_{_g}": (pl.col("assigned") == _g).mean().round(2)
                for _g in range(4)
            },
        )
        .with_columns(_order=pl.col("cause").replace_strict(_cause_order))
        .sort("_order", "target")
        .drop("_order")
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
    TARGETS,
    cfg,
    chosen_lvs,
    class_curves,
    confusion,
    per_class,
    predictions_table,
    representation_choice,
    save_button,
    target_choice,
    total_error_rate,
):
    mo.stop(not save_button.value, mo.md("_Not saved._"))
    _run_dir = new_run_dir(
        cfg.path("results_dir"),
        f"plsda_classes_{target_choice.value}_{representation_choice.value}",
    )
    predictions_table.write_parquet(_run_dir / "predictions.parquet")
    class_curves.write_parquet(_run_dir / "curves.parquet")
    confusion.write_parquet(_run_dir / "confusion.parquet")
    _scores = {"total_error_rate": total_error_rate, "per_class": per_class.to_dicts()}
    (_run_dir / "scores.json").write_text(json.dumps(_scores, indent=2))
    write_run_record(
        _run_dir,
        {
            "what": "PLS-DA one model per grade (one-versus-rest), design A, 12 open 2526 boxes",
            "target": target_choice.value,
            "target_label": TARGETS[target_choice.value],
            "fruit_filter": "cavity == 0"
            if target_choice.value == "browning"
            else "none",
            "representation": representation_choice.value,
            "representation_label": REPRESENTATIONS[representation_choice.value],
            "n_fruit": predictions_table.height,
            "n_lv_per_class": {str(_k): _h for _k, _h in chosen_lvs.items()},
            "lv_rule": "elbow of the mean RMSECV curve of each class model, chosen by Ruben by eye",
            "assignment": "class with the highest predicted membership",
            "scale": representation_choice.value in SCALED,
            "max_components": MAX_COMPONENTS,
            "folds": "derived/outer_folds_20/outer_folds.csv",
        },
        root=cfg.root,
    )
    mo.md(f"Saved to `{_run_dir}`.")
    return


if __name__ == "__main__":
    app.run()
