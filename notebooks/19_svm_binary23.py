import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title="19 SVM-DA check on binary_23")

with app.setup(hide_code=True):
    import json

    import marimo as mo
    import numpy as np
    import polars as pl
    from sklearn.metrics import roc_auc_score

    from pearlatent.config import load_config
    from pearlatent.feature_sets import load_features
    from pearlatent.fruit_table import drop_faulty_volumes, open_fruit
    from pearlatent.nuisance import box_bootstrap, healthy_fruit_correlations
    from pearlatent.plsda import box_bootstrap_auc, paired_box_bootstrap_auc_difference
    from pearlatent.runs import PLSDA_BINARY23_RUN_DIRS, new_run_dir, write_run_record
    from pearlatent.svm import (
        C_VALUES,
        GAMMA_FACTORS,
        cross_validated_decisions,
        settings_grid,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 19 SVM-DA check on binary_23

    **Question.** Does a non-linear boundary (support vector machine with a radial basis
    function kernel, SVM-DA) find more than linear PLS-DA on the same features? Reason (Ruben,
    20 November 2026): one number per fruit, the SD of the interior CT cells, separates grade 0
    from grade 1 (AUC 0.82), while every linear PLS-DA row does not (0.59 to 0.72). A weighted
    sum of cells cannot form a spread.

    **Inputs.** The fruit table, `derived/volume_decisions.csv`,
    `derived/outer_folds_20/outer_folds.csv`, the feature set `hugo_vae_hr_sd` (Hugo's
    latent, 20 height × 8 radius bins, mean and SD of each bin, 320 numbers per fruit), and the
    saved PLS-DA run on the same features (3 LVs) for the paired comparison.

    **Outputs.** With the save button: `results/<run_id>_svm_binary23_hugo_vae_hr_sd/`
    (`predictions.parquet`, `grid.parquet`, `scores.json`, `run.json`).

    **Status.** draft. One check row, not a new main line (`pearlatent.svm`, module docstring).

    Reading guide: same 315 fruit, same 20 × 4-fold box partitions and same bootstrap seed as
    the PLS-DA rows. Only the 12 open boxes are used.
    """)
    return


@app.cell
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
    print(f"open 2526 fruit: {_open_2526.height}")
    print(f"after removing faulty volumes: {fruit.height}")
    print(f"non-consumable (binary_23 = 1): {fruit['binary_23'].sum()}")
    folds = pl.read_csv(cfg.path("derived_dir") / "outer_folds_20" / "outer_folds.csv")
    if set(folds["box"].unique()) != set(fruit["box"].unique()):
        raise ValueError("the boxes of the folds and of the fruit differ")
    print(f"{folds['repeat'].n_unique()} repeats over {folds['box'].n_unique()} boxes")
    return fruit, folds


@app.cell(hide_code=True)
def _():
    run_cv = mo.ui.run_button(
        label="Run the SVM cross-validation (12 settings × 80 fits)"
    )
    run_cv
    return (run_cv,)


@app.function(hide_code=True)
@mo.persistent_cache
def cross_validate_svm(fruit_ids: tuple) -> np.ndarray:
    """Out-of-fold decision values (repeats × fruit × settings) for the 12 grid settings.

    In: the fruit, in table order. Out: the array of `cross_validated_decisions`. Cached on
    disk, because the inputs are fixed.
    """
    cfg = load_config()
    fruit = drop_faulty_volumes(
        open_fruit(cfg).filter(pl.col("season") == "2526"),
        cfg.path("derived_dir") / "volume_decisions.csv",
    ).with_columns(binary_23=(pl.col("defective") >= 2).cast(pl.Int8))
    assert tuple(fruit["fruit_id"].to_list()) == fruit_ids, "fruit order changed"
    x = load_features(cfg, "hugo_vae_hr_sd", list(fruit_ids))
    folds = pl.read_csv(cfg.path("derived_dir") / "outer_folds_20" / "outer_folds.csv")
    return cross_validated_decisions(
        x,
        fruit["binary_23"].to_numpy(),
        fruit["box"].to_numpy(),
        folds,
        settings_grid(),
    )


@app.cell(hide_code=True)
def _(fruit, run_cv):
    # In script mode the button is never pressed; the cross-validation then runs directly.
    mo.stop(
        mo.running_in_notebook() and not run_cv.value,
        mo.md("_Press the button to run the cross-validation._"),
    )
    decisions = cross_validate_svm(tuple(fruit["fruit_id"].to_list()))
    y = fruit["binary_23"].to_numpy()
    boxes = fruit["box"].to_numpy()
    print(
        f"decision values for {decisions.shape[0]} repeats × {decisions.shape[1]} fruit"
        f" × {decisions.shape[2]} settings"
    )
    return boxes, decisions, y


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1. The grid: choose C and gamma

    For each setting: the AUC of the decision value averaged over the 20 repeats (as for the
    PLS-DA rows), and the lowest and highest AUC of a single repeat. C: how hard a
    misclassified training fruit is penalised. Gamma factor: how local the boundary is
    (1 = scikit-learn's default, 1 / 320 on standardised features; small = almost linear).

    Rule (Ruben, 20 November 2026): the setting with the highest AUC of the mean. Caveat:
    the best of 12 settings on these boxes makes the chosen AUC slightly optimistic, more so
    than the one choice of LVs for PLS-DA.
    """)
    return


@app.cell(hide_code=True)
def _(decisions, y):
    _rows = []
    for _k, (_c, _g) in enumerate(settings_grid()):
        _per_repeat = [roc_auc_score(y, decisions[_r, :, _k]) for _r in range(decisions.shape[0])]
        _rows.append(
            {
                "C": _c,
                "gamma_factor": _g,
                "auc_of_mean": roc_auc_score(y, decisions[:, :, _k].mean(axis=0)),
                "repeats_min": min(_per_repeat),
                "repeats_max": max(_per_repeat),
            }
        )
    grid = pl.DataFrame(_rows)
    mo.vstack(
        [
            mo.md("AUC of the mean decision value, rows C, columns gamma factor:"),
            grid.pivot(on="gamma_factor", index="C", values="auc_of_mean").with_columns(
                pl.exclude("C").round(3)
            ),
            grid.with_columns(pl.col("auc_of_mean", "repeats_min", "repeats_max").round(3)),
        ]
    )
    return (grid,)


@app.cell(hide_code=True)
def _(grid):
    # Rule (Ruben, 20 November 2026): the setting with the highest AUC of the mean decision
    # value over the 20 repeats (a grid search). The chosen AUC is therefore slightly
    # optimistic: it is the best of 12 tries on the same boxes.
    _best = grid.sort("auc_of_mean", descending=True).row(0, named=True)
    chosen = (_best["C"], _best["gamma_factor"])
    mo.md(
        f"Chosen by the rule: **C = {chosen[0]}, gamma factor {chosen[1]}**"
        f" (AUC of the mean {_best['auc_of_mean']:.3f})."
    )
    return (chosen,)


@app.cell(hide_code=True)
def _(cfg, fruit):
    # The saved PLS-DA row on the same features (3 LVs, run of 9 November), joined one to one.
    _run = cfg.path("results_dir") / PLSDA_BINARY23_RUN_DIRS["hugo_vae_hr_sd"]
    _pls = pl.read_parquet(_run / "predictions.parquet").select(
        "fruit_id", pls_prediction="prediction"
    )
    pls_table = fruit.select("fruit_id").join(
        _pls, on="fruit_id", how="left", validate="1:1"
    )
    if pls_table["pls_prediction"].null_count():
        raise ValueError("some fruit have no saved PLS-DA prediction")
    print(f"PLS-DA predictions joined for {pls_table.height} fruit")
    return (pls_table,)


@app.cell(hide_code=True)
def _(boxes, chosen, decisions, fruit, pls_table, y):
    _k = settings_grid().index(chosen)
    svm_score = decisions[:, :, _k].mean(axis=0)
    _low, _high, _ = box_bootstrap_auc(svm_score, y, boxes)
    _paired = paired_box_bootstrap_auc_difference(
        svm_score, pls_table["pls_prediction"].to_numpy(), y, boxes
    )
    scores = {
        "C": chosen[0],
        "gamma_factor": chosen[1],
        "auc": roc_auc_score(y, svm_score),
        "auc_box_bootstrap_95": [_low, _high],
        "pls_auc": _paired["auc_b"],
        "svm_minus_pls": _paired["difference"],
        "svm_minus_pls_95": [_paired["lower_95"], _paired["upper_95"]],
        "share_svm_better": _paired["share_a_better"],
    }
    predictions_table = fruit.select(
        "fruit_id", "box", "defective", "browning", "cavity", "cause", "binary_23"
    ).with_columns(prediction=svm_score, pls_prediction=pls_table["pls_prediction"])
    mo.md(
        f"""
    ## 2. SVM-DA against PLS-DA on the same features

    **C = {chosen[0]}, gamma factor {chosen[1]}.** SVM AUC {scores['auc']:.3f}
    (95 % box bootstrap {_low:.3f} to {_high:.3f}); PLS-DA (3 LVs) {_paired['auc_b']:.3f}.
    Paired difference SVM − PLS-DA: {_paired['difference']:+.3f}
    ({_paired['lower_95']:+.3f} to {_paired['upper_95']:+.3f}); the SVM is higher in
    {100 * _paired['share_a_better']:.0f} % of the box draws.
    """
    )
    return predictions_table, scores


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 3. Which defects, and grade 0 against grade 1

    AUC of grade 0 against one group (as in notebook 18), for both models. The question of
    this notebook is mostly the row "browning only 1": does the non-linear boundary see
    grade 1 where the linear one does not? Both models were trained on `binary_23`, where
    grade 1 counts as consumable.
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
    _rows = []
    for _group, _condition in _groups.items():
        _pair = predictions_table.filter(
            (pl.col("defective") == 0) | _condition
        ).with_columns(in_group=_condition.cast(pl.Int8))
        _label = _pair["in_group"].to_numpy()
        for _model, _column in (("SVM", "prediction"), ("PLS-DA", "pls_prediction")):

            def _auc(t, c=_column):
                return roc_auc_score(t["in_group"], t[c]) if t["in_group"].n_unique() == 2 else np.nan

            _low, _high = box_bootstrap(_auc, _pair)
            _rows.append(
                {
                    "group": _group,
                    "model": _model,
                    "n_group": int(_label.sum()),
                    "n_grade_0": int((1 - _label).sum()),
                    "auc": _auc(_pair),
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
    ## 4. Check C: does the SVM score follow stalk and size on healthy fruit?

    Grade 0 fruit only; Spearman correlation of the score (minus its box median) with stalk
    length and body size, 1,000 box draws, as in notebook 10. The PLS-DA row on these features
    gave −0.29 (stalk) and +0.30 (size) on 9 November.
    """)
    return


@app.cell(hide_code=True)
def _(cfg, predictions_table):
    _descriptors = drop_faulty_volumes(
        open_fruit(cfg).filter(pl.col("season") == "2526"),
        cfg.path("derived_dir") / "volume_decisions.csv",
    ).select("fruit_id", "stalk_length", "body_voxels", "stalk_present")
    _table = predictions_table.join(
        _descriptors, on="fruit_id", how="left", validate="1:1"
    )
    _rows = []
    for _model, _column in (("SVM", "prediction"), ("PLS-DA", "pls_prediction")):
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
    save_button = mo.ui.run_button(label="Save this run to results/")
    save_button
    return (save_button,)


@app.cell(hide_code=True)
def _(
    auc_by_group,
    cfg,
    check_c,
    chosen,
    grid,
    predictions_table,
    save_button,
    scores,
):
    mo.stop(not save_button.value, mo.md("_Not saved._"))
    _run_dir = new_run_dir(cfg.path("results_dir"), "svm_binary23_hugo_vae_hr_sd")
    predictions_table.write_parquet(_run_dir / "predictions.parquet")
    grid.write_parquet(_run_dir / "grid.parquet")
    auc_by_group.write_parquet(_run_dir / "auc_by_group.parquet")
    check_c.write_parquet(_run_dir / "check_c_healthy_correlations.parquet")
    (_run_dir / "scores.json").write_text(json.dumps(scores, indent=2))
    write_run_record(
        _run_dir,
        {
            "what": "SVM-DA (RBF kernel) on binary_23, design A, 12 open 2526 boxes",
            "representation": "hugo_vae_hr_sd",
            "n_fruit": predictions_table.height,
            "c_values": list(C_VALUES),
            "gamma_factors": list(GAMMA_FACTORS),
            "gamma": "factor / number of features (320), on standardised features",
            "standardisation": "StandardScaler inside each training fold",
            "chosen": {"C": chosen[0], "gamma_factor": chosen[1]},
            "choice_rule": "highest AUC of the mean decision value over the 20 repeats (Ruben, 20 November 2026)",
            "score": "SVM decision value averaged over the 20 repeats",
            "compared_with": PLSDA_BINARY23_RUN_DIRS["hugo_vae_hr_sd"],
            "folds": "derived/outer_folds_20/outer_folds.csv",
            "bootstrap_seed": 369,
        },
        root=cfg.root,
    )
    mo.md(f"Saved to `{_run_dir}`.")
    return


if __name__ == "__main__":
    app.run()
