import marimo

__generated_with = "0.25.1"
app = marimo.App(
    width="medium",
    app_title="22 EPO model for the counterfactuals",
)

with app.setup(hide_code=True):
    import json

    import altair as alt
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score

    from pearlatent.config import load_config
    from pearlatent.counterfactual import plsda_direction
    from pearlatent.counterfactual_run import DIRECTIONS
    from pearlatent.epo import fit_epo_directions, remove_directions
    from pearlatent.feature_sets import load_features
    from pearlatent.fruit_table import drop_faulty_volumes, open_fruit
    from pearlatent.nuisance import box_bootstrap, healthy_fruit_correlations
    from pearlatent.plsda import (
        MAX_COMPONENTS,
        box_bootstrap_auc,
        cross_validated_predictions,
        mean_squared_errors,
        paired_box_bootstrap_auc_difference,
        youden_cutoff,
    )
    from pearlatent.runs import new_run_dir, write_run_record
    from pearlatent.targets import target_table


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 22 EPO model for the counterfactuals

    **Question.** With the size axis removed from Hugo's latent by external parameter
    orthogonalisation (EPO, `pearlatent.epo`), how many latent variables (LVs) does the
    full-grid PLS-DA model need, does it keep its AUC, and is its direction b free of the size
    axis? This model then gives the new counterfactuals (question 55: the old direction added a
    stalk to 2526_C23).

    **EPO.** In every training fold: the first principal component of the healthy (grade 0)
    training fruit is removed from all fruit before PLS. On 20 November it followed body size
    at r = 0.97 and was the same direction as the latent's covariance with size (cosine 0.99).
    The final model's b is orthogonal to it, so a counterfactual step cannot move a fruit along
    the size axis.

    **Design.** As notebooks 08 and 14: the 12 open 2526 boxes, 20 repeats of 4-fold box
    cross-validation, centring only, the LVs chosen by Ruben at the elbow of the mean RMSECV
    curve. The model without EPO is computed beside it on the same folds.

    **Inputs.** The fruit table, `derived/volume_decisions.csv`, `derived/outer_folds_20/`,
    the feature set `hugo_vae`. The sealed boxes F, J and M are never loaded.

    **Outputs.** `results/<run_id>_epo_<direction>/` when the save button is pressed, and the
    commands for `pearlatent.counterfactual_run`.

    **Status.** draft (20 November 2026)
    """)
    return


@app.cell
def _():
    cfg = load_config()
    return (cfg,)


@app.cell(hide_code=True)
def _():
    # The two model directions of `pearlatent.counterfactual_run`; the old counterfactuals
    # used "cavity_free" (browning ≥ 2 on cavity-free fruit, 5 LVs).
    direction_choice = mo.ui.dropdown(
        options={
            "cavity-free browning ≥ 2 (the model of the old counterfactuals)": "cavity_free",
            "binary_23, all fruit": "general",
        },
        value="cavity-free browning ≥ 2 (the model of the old counterfactuals)",
        label="Model",
    )
    run_cv = mo.ui.run_button(label="Run the cross-validation (about 1 minute)")
    mo.hstack([direction_choice, run_cv], justify="start")
    return direction_choice, run_cv


@app.function(hide_code=True)
def model_fruit(cfg, direction: str) -> pl.DataFrame:
    """The fruit of one model direction with `y` and `healthy`, as `counterfactual_run`."""
    kept = drop_faulty_volumes(
        open_fruit(cfg).filter(pl.col("season") == "2526"),
        cfg.path("derived_dir") / "volume_decisions.csv",
    )
    target = DIRECTIONS[direction]["target"]
    if target is None:
        fruit = kept.with_columns(y=(pl.col("defective") >= 2).cast(pl.Int8))
    else:
        fruit = target_table(kept, target)
    # Healthy = defective grade 0, the fruit EPO is fitted on.
    return fruit.with_columns(healthy=pl.col("defective") == 0)


@app.cell(hide_code=True)
def _(cfg, direction_choice):
    fruit = model_fruit(cfg, direction_choice.value)
    print(
        f"{direction_choice.value}: {fruit.height} fruit, label 1: {fruit['y'].sum()},"
        f" healthy (EPO set): {fruit['healthy'].sum()}"
    )
    return (fruit,)


@app.function(hide_code=True)
@mo.persistent_cache
def cross_validate_epo(direction: str, fruit_ids: tuple) -> dict:
    """Out-of-fold predictions (repeats × fruit × LVs) without and with EPO, same folds.

    In: the direction name and the fruit (in table order). Cached on disk.
    """
    cfg = load_config()
    fruit = model_fruit(cfg, direction)
    assert tuple(fruit["fruit_id"].to_list()) == fruit_ids, "fruit order changed"
    x = load_features(cfg, "hugo_vae", list(fruit_ids))
    y = fruit["y"].to_numpy()
    boxes = fruit["box"].to_numpy()
    folds = pl.read_csv(cfg.path("derived_dir") / "outer_folds_20" / "outer_folds.csv")
    return {
        "none": cross_validated_predictions(x, y, boxes, folds, scale=False),
        "epo": cross_validated_predictions(
            x, y, boxes, folds, scale=False, epo_healthy=fruit["healthy"].to_numpy()
        ),
    }


@app.cell(hide_code=True)
def _(direction_choice, fruit, run_cv):
    mo.stop(
        mo.running_in_notebook() and not run_cv.value,
        mo.md("_Press the button to run the cross-validation._"),
    )
    cv = cross_validate_epo(direction_choice.value, tuple(fruit["fruit_id"].to_list()))
    y = fruit["y"].to_numpy()
    boxes = fruit["box"].to_numpy()
    print(f"predictions: {cv['epo'].shape[0]} repeats × {cv['epo'].shape[1]} fruit")
    return boxes, cv, y


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1. Choose the number of latent variables (EPO curve, black)

    RMSECV(h) = √(mean PRESS(h) / n) over the 20 repeats, with EPO (black) and without (grey,
    for reference). Choose at the elbow of the black curve.
    """)
    return


@app.cell(hide_code=True)
def _(cv, y):
    _rows = []
    for _name, _label in (("epo", "with EPO"), ("none", "without EPO")):
        _press = mean_squared_errors(cv[_name], y).sum(axis=0)
        for _h, _e in enumerate(np.sqrt(_press / len(y)), start=1):
            _rows.append({"n_lv": _h, "rmsecv": float(_e), "model": _label})
    curves = pl.DataFrame(_rows)
    _chart = (
        alt.Chart(curves)
        .mark_line(point=True)
        .encode(
            x=alt.X("n_lv:O", title="number of latent variables"),
            y=alt.Y("rmsecv:Q", title="RMSECV, mean PRESS", scale=alt.Scale(zero=False)),
            color=alt.Color(
                "model:N",
                scale=alt.Scale(
                    domain=["with EPO", "without EPO"], range=["#000000", "#bbbbbb"]
                ),
            ),
            tooltip=["model", "n_lv", alt.Tooltip("rmsecv:Q", format=".4f")],
        )
        .properties(width=560, height=300)
    )
    mo.vstack([_chart, curves.pivot(on="model", index="n_lv", values="rmsecv")])
    return (curves,)


@app.cell(hide_code=True)
def _():
    lv_choice = mo.ui.dropdown(
        options=["not chosen"] + [str(_h) for _h in range(1, MAX_COMPONENTS + 1)],
        value="not chosen",
        label="Number of LVs of the EPO model (Ruben's choice at the elbow)",
    )
    lv_choice
    return (lv_choice,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2. EPO model against the model without EPO

    The model without EPO is scored at the number of LVs of its counterfactual direction
    (`DIRECTIONS`: 5 for cavity-free, 4 for binary_23). AUC with a 95 % interval over 2,000 box
    draws, the paired difference on the same draws, and check C on the healthy fruit (Spearman
    of the score minus its box median with stalk length and body size, 1,000 box draws).
    """)
    return


@app.cell(hide_code=True)
def _(boxes, cv, direction_choice, fruit, lv_choice, y):
    mo.stop(lv_choice.value == "not chosen", mo.md("_Choose the number of LVs first._"))
    chosen_lv = int(lv_choice.value)
    _old_lv = DIRECTIONS[direction_choice.value]["n_lv"]
    predictions = {
        "with EPO": cv["epo"][:, :, chosen_lv - 1].mean(axis=0),
        "without EPO": cv["none"][:, :, _old_lv - 1].mean(axis=0),
    }
    _base = fruit.select(
        "box", "defective", "stalk_length", "body_voxels", "stalk_present"
    )
    _rows = []
    for _model, _p in predictions.items():
        _low, _high, _ = box_bootstrap_auc(_p, y, boxes)
        _table = _base.with_columns(score=_p)
        _row = {
            "model": _model,
            "n_lv": chosen_lv if _model == "with EPO" else _old_lv,
            "auc": roc_auc_score(y, _p),
            "auc_95": f"{_low:.3f} to {_high:.3f}",
        }
        for _measure in ("stalk_length", "body_voxels"):
            _key = f"spearman_{_measure}"
            _l, _h = box_bootstrap(
                lambda t, k=_key: healthy_fruit_correlations(t)[k],
                _table,
                n_resamples=1000,
            )
            _row[_measure] = round(healthy_fruit_correlations(_table)[_key], 2)
            _row[f"{_measure}_95"] = f"{_l:+.2f} to {_h:+.2f}"
        _rows.append(_row)
    comparison = pl.DataFrame(_rows)
    _paired = paired_box_bootstrap_auc_difference(
        predictions["with EPO"], predictions["without EPO"], y, boxes
    )
    paired = {
        "difference": _paired["difference"],
        "lower_95": _paired["lower_95"],
        "upper_95": _paired["upper_95"],
    }
    # Youden cut-off of the EPO model on its out-of-fold predictions (methods section 3);
    # saved in run.json and used by `counterfactual_run --flip-at` (22 November 2026).
    cutoff_youden = youden_cutoff(predictions["with EPO"], y)
    # One row per fruit: the out-of-fold prediction of the EPO model (mean of 20 repeats).
    epo_predictions = fruit.select("fruit_id", "box", "defective", "y").with_columns(
        prediction=predictions["with EPO"]
    )
    mo.vstack(
        [
            comparison,
            mo.md(
                f"AUC with EPO − without: **{paired['difference']:+.3f}**"
                f" ({paired['lower_95']:+.3f} to {paired['upper_95']:+.3f})."
                f" Youden cut-off of the EPO model: **{cutoff_youden:.3f}**."
            ),
        ]
    )
    return chosen_lv, comparison, cutoff_youden, epo_predictions, paired


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 3. The final direction: what EPO removed, and is b free of it?

    One fit on all fruit of the 12 boxes, as `counterfactual_run` does. The removed direction is
    the first principal component of the healthy fruit; its score on the healthy fruit is
    compared with body size and stalk length. |b · v| / |b| should be about 0.
    """)
    return


@app.cell(hide_code=True)
def _(cfg, chosen_lv, fruit, y):
    _x = load_features(cfg, "hugo_vae", fruit["fruit_id"].to_list()).astype(float)
    _healthy = fruit["healthy"].to_numpy()
    final_direction = plsda_direction(_x, y, chosen_lv, epo_healthy=_healthy)
    _v = final_direction.removed[:, 0]
    _b = final_direction.coefficients
    _deviations = _x[_healthy] - _x[_healthy].mean(axis=0)
    _score = _deviations @ _v
    _share = float((_score**2).sum() / (_deviations**2).sum())
    _check = fit_epo_directions(_x, _healthy)[:, 0]
    mo.md(
        f"""
    Removed direction: {_share:.0%} of the healthy fruit's variation; its score follows body
    size at r = {spearmanr(_score, fruit.filter(pl.col("healthy"))["body_voxels"]).statistic:+.2f}
    and stalk length at r = {spearmanr(_score, fruit.filter(pl.col("healthy"))["stalk_length"]).statistic:+.2f}
    (Spearman, {int(_healthy.sum())} healthy fruit).

    |b · v| / |b| = {abs(_b @ _v) / np.linalg.norm(_b):.1e} (0 by construction).
    Same direction recomputed: |cosine| = {abs(_check @ _v):.3f}.
    """
    )
    return (final_direction,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 4. Commands for the new counterfactuals

    The same fruit, steps and plain reconstructions as the counterfactual runs of 9 November
    (notebook 17). One decode takes about 5 minutes and 18 GB on the Mac; run them one after
    the other in a terminal, with the marimo editor closed. Each run writes
    `derived/counterfactuals/<timestamp>_<fruit>_<direction>_epo/`; notebook 17 shows them.
    """)
    return


@app.cell(hide_code=True)
def _(chosen_lv, direction_choice):
    # From the run.json of the four runs of 9 November (cavity-free direction).
    _old_runs = [
        ("2526_C23", "0.5 0.0", "derived/step0_examples/20261109T174632Z_examples"),
        ("2526_K07", "0.5 1.0", "derived/step0_examples/20261109T190424Z_examples"),
        ("2526_A03", "0.5 1.0", "derived/step0_examples/20261109T190424Z_examples"),
        ("2526_N17", "0.5 0.0", "derived/step0_examples/20261109T190424Z_examples"),
    ]
    _lines = [
        f"uv run python -m pearlatent.counterfactual_run --fruit {_f}"
        f" --direction {direction_choice.value}_epo --n-lv {chosen_lv}"
        f" --t {_t} --plain {_p}"
        for _f, _t, _p in _old_runs
    ]
    mo.md("```bash\n" + " && \\\n".join(_lines) + "\n```")
    return


@app.cell(hide_code=True)
def _():
    save_button = mo.ui.run_button(label="Save this run to results/")
    save_button
    return (save_button,)


@app.cell(hide_code=True)
def _(
    cfg,
    chosen_lv,
    comparison,
    curves,
    cutoff_youden,
    direction_choice,
    final_direction,
    paired,
    save_button,
):
    mo.stop(not save_button.value, mo.md("_Not saved._"))
    _run_dir = new_run_dir(cfg.path("results_dir"), f"epo_{direction_choice.value}")
    curves.write_parquet(_run_dir / "curves.parquet")
    comparison.write_parquet(_run_dir / "comparison.parquet")
    np.save(_run_dir / "regression_vector.npy", final_direction.coefficients)
    np.save(_run_dir / "epo_directions.npy", final_direction.removed)
    (_run_dir / "paired.json").write_text(json.dumps(paired, indent=2))
    write_run_record(
        _run_dir,
        {
            "what": "PLS-DA with EPO (healthy PC 1 removed), latent full grid, 12 open 2526 boxes",
            "direction": direction_choice.value,
            "target": DIRECTIONS[direction_choice.value]["target"] or "binary_23",
            "n_lv_chosen": chosen_lv,
            "cutoff": 0.5,
            "cutoff_youden": cutoff_youden,
            "lv_rule": "elbow of the mean RMSECV curve, chosen by Ruben by eye",
            "epo": "first principal component of the grade 0 training fruit, per fold",
            "scale": False,
            "folds": "derived/outer_folds_20/outer_folds.csv",
            "bootstrap_seed": 369,
        },
        root=cfg.root,
    )
    mo.md(f"Saved to `{_run_dir}`.")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 5. PCA of the latent before and after EPO (description, 23 November 2026)

    All 315 open 2526 fruit (the `binary_23` fruit), latent full grid. Before: PCA of the
    plain latent. After: the first principal component of the 145 healthy fruit is removed from
    every fruit (the EPO step of the models), then PCA again. Here the direction is fitted once
    on all healthy fruit, not per fold: this section describes the data, it scores no model.
    Question: does the main structure stop being size and stalk, and does the grade show up?
    """)
    return


@app.function(hide_code=True)
def principal_scores(x: np.ndarray, n_components: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """PCA scores and explained variance shares of the first components.

    In: features (fruit, features). Out: scores (fruit, n_components) and the share of the
    total variance of each component. Centred, not scaled (CLAUDE.md section 7).
    """
    deviations = x - x.mean(axis=0)
    u, s, _ = np.linalg.svd(deviations, full_matrices=False)
    shares = s**2 / (s**2).sum()
    return u[:, :n_components] * s[:n_components], shares[:n_components]


@app.cell(hide_code=True)
def _(cfg):
    _fruit = model_fruit(cfg, "general")
    _x = load_features(cfg, "hugo_vae", _fruit["fruit_id"].to_list()).astype(float)
    _healthy = _fruit["healthy"].to_numpy()
    # One direction, fitted on all 145 healthy fruit: the same rule as the models
    # (N_DIRECTIONS = 1), without folds because nothing is scored here.
    _removed = fit_epo_directions(_x, _healthy)
    pca_views = {
        "before EPO": principal_scores(_x),
        "after EPO": principal_scores(remove_directions(_x, _removed)),
    }
    pca_fruit = _fruit
    _rows = []
    for _view, (_scores, _shares) in pca_views.items():
        for _k in range(2):
            _row = {"view": _view, "PC": _k + 1, "variance share": round(float(_shares[_k]), 3)}
            # Spearman: the descriptors are counts with long tails.
            for _name in ("body_voxels", "stalk_length", "defective"):
                _row[f"r {_name}"] = round(
                    float(spearmanr(_scores[:, _k], _fruit[_name].to_numpy()).statistic), 2
                )
            _rows.append(_row)
    pca_correlations = pl.DataFrame(_rows)
    print(f"{_fruit.height} fruit, {int(_healthy.sum())} healthy (EPO set)")
    pca_correlations
    return pca_correlations, pca_fruit, pca_views


@app.cell(hide_code=True)
def _(cfg, pca_fruit, pca_views):
    pca_figure, _axes = plt.subplots(2, 2, figsize=(10, 9))
    _colourings = [
        ("body_voxels", "viridis", "body size (voxels in the 128³ volume)"),
        ("defective", "plasma", "defective grade"),
    ]
    for _r, (_view, (_scores, _shares)) in enumerate(pca_views.items()):
        for _c, (_name, _map, _label) in enumerate(_colourings):
            _ax = _axes[_r, _c]
            _shown = _ax.scatter(
                _scores[:, 0], _scores[:, 1], c=pca_fruit[_name].to_numpy(), cmap=_map, s=12
            )
            _ax.set_xlabel(f"PC 1 ({_shares[0]:.0%})")
            _ax.set_ylabel(f"PC 2 ({_shares[1]:.0%})")
            _ax.set_title(f"{_view}, coloured by {_name}")
            pca_figure.colorbar(_shown, ax=_ax, label=_label)
    pca_figure.suptitle(
        f"Latent full grid, {pca_fruit.height} open 2526 fruit: PCA before and after removing "
        "the healthy PC 1"
    )
    pca_figure.tight_layout()
    _out = cfg.path("derived_dir") / "figures" / "22_pca_before_after_epo.png"
    pca_figure.savefig(_out, dpi=130, bbox_inches="tight")
    print(f"saved {_out}")
    pca_figure
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 6. Check C before and after EPO, latent height-radius mean + SD (23 November 2026)

    The row with the strongest size and stalk link (`hugo_vae_hr_sd`, 320 features, 3 LVs as
    saved on 9 November), on the 315 `binary_23` fruit and the 20 × 4 box folds. Without EPO
    and with EPO (healthy PC 1 removed in every fold, fitted in these 320 features). Each fruit's
    score is the mean of its 20 out-of-fold predictions. The figure shows the 145 healthy fruit:
    score minus its box median (as check C) against body size and stalk length. Repeats the
    scratch numbers of methods section 17 as a figure.
    """)
    return


@app.function(hide_code=True)
@mo.persistent_cache
def cross_validate_hr_sd(fruit_ids: tuple) -> dict:
    """Out-of-fold predictions (repeats × fruit × LVs) of `hugo_vae_hr_sd`, without and with
    EPO, on the same folds. In: the `binary_23` fruit in table order. Cached on disk.
    """
    cfg = load_config()
    fruit = model_fruit(cfg, "general")
    assert tuple(fruit["fruit_id"].to_list()) == fruit_ids, "fruit order changed"
    x = load_features(cfg, "hugo_vae_hr_sd", list(fruit_ids))
    y = fruit["y"].to_numpy()
    boxes = fruit["box"].to_numpy()
    folds = pl.read_csv(cfg.path("derived_dir") / "outer_folds_20" / "outer_folds.csv")
    return {
        "without EPO": cross_validated_predictions(x, y, boxes, folds, scale=False),
        "with EPO": cross_validated_predictions(
            x, y, boxes, folds, scale=False, epo_healthy=fruit["healthy"].to_numpy()
        ),
    }


@app.cell(hide_code=True)
def _(cfg):
    # 3 LVs: the number Ruben chose for this row on 9 November (methods section 13).
    HR_SD_LV = 3
    _fruit = model_fruit(cfg, "general")
    _cv = cross_validate_hr_sd(tuple(_fruit["fruit_id"].to_list()))
    _y = _fruit["y"].to_numpy()
    _base = _fruit.select("box", "defective", "stalk_length", "body_voxels", "stalk_present")
    hr_sd_healthy = {}
    _rows = []
    for _model, _p in _cv.items():
        _score = _p[:, :, HR_SD_LV - 1].mean(axis=0)
        _table = _base.with_columns(score=_score)
        _check = healthy_fruit_correlations(_table)
        _rows.append(
            {
                "model": _model,
                "auc": round(roc_auc_score(_y, _score), 3),
                "stalk_length": round(_check["spearman_stalk_length"], 2),
                "body_voxels": round(_check["spearman_body_voxels"], 2),
            }
        )
        # The healthy fruit with the score minus its box median, as check C uses it.
        hr_sd_healthy[_model] = _table.filter(pl.col("defective") == 0).with_columns(
            score_in_box=pl.col("score") - pl.col("score").median().over("box")
        )
    hr_sd_check_c = pl.DataFrame(_rows)
    hr_sd_check_c
    return hr_sd_check_c, hr_sd_healthy


@app.cell(hide_code=True)
def _(cfg, hr_sd_check_c, hr_sd_healthy):
    check_c_figure, _axes = plt.subplots(2, 2, figsize=(10, 8), sharey="row")
    _measures = [
        ("body_voxels", "body size (voxels in the 128³ volume)"),
        ("stalk_length", "stalk length (voxels)"),
    ]
    for _c, (_model, _healthy) in enumerate(hr_sd_healthy.items()):
        _row = hr_sd_check_c.filter(pl.col("model") == _model).row(0, named=True)
        for _r, (_name, _label) in enumerate(_measures):
            _ax = _axes[_r, _c]
            _ax.scatter(_healthy[_name], _healthy["score_in_box"], s=12, color="#3b6ea5")
            _ax.axhline(0, color="grey", linewidth=0.8)
            _ax.set_xlabel(_label)
            _ax.set_title(f"{_model} (AUC {_row['auc']:.3f}): r = {_row[_name]:+.2f}")
        _axes[0, 0].set_ylabel("score − box median")
        _axes[1, 0].set_ylabel("score − box median")
    check_c_figure.suptitle(
        "Latent height-radius mean + SD, 3 LVs: score of the 145 healthy fruit against size "
        "and stalk, without and with EPO (Spearman r)"
    )
    check_c_figure.tight_layout()
    _out = cfg.path("derived_dir") / "figures" / "22_check_c_before_after_epo.png"
    check_c_figure.savefig(_out, dpi=130, bbox_inches="tight")
    print(f"saved {_out}")
    check_c_figure
    return


if __name__ == "__main__":
    app.run()
