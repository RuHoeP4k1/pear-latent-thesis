import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title="16 All volumes and the reverse check")

with app.setup(hide_code=True):
    import json

    import marimo as mo
    import numpy as np
    import polars as pl
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.metrics import roc_auc_score

    from pearlatent.config import load_config
    from pearlatent.feature_sets import load_features
    from pearlatent.fruit_table import drop_faulty_volumes, open_fruit
    from pearlatent.nuisance import box_bootstrap
    from pearlatent.plsda import cross_validated_predictions
    from pearlatent.runs import PLSDA_BINARY23_RUN_DIRS, new_run_dir, write_run_record


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 16 All volumes and the reverse check

    **Question.** Do the `binary_23` results of notebook 08 hold (1) when the 31 faulty 2526
    volumes are kept, and (2) on fruit of another season, Hugo's 2425 validation fruit?

    **1. All volumes** (promised at the meeting of 29 October, because the excluded cuts are
    uneven over the grades). The same design A as notebook 08 on all 346 open 2526 fruit, with
    each row's number of latent variables (LVs) and scaling taken unchanged from its saved run.
    Reported: AUC on all 346, on the 315 kept fruit and on the 31 faulty volumes, each with
    the same predictions.

    **2. Reverse check** (charter rule 2). Each row's general model (one fit on the 315 open
    2526 fruit, its saved LVs) predicts Hugo's 2425 validation fruit, minus the faulty volumes.
    Another harvest, other orchards, and volumes without stalk. Read with care: in these fruit
    the defect follows the storage group, and every non-consumable validation fruit has a
    cavity, so this is an easy test of the season change, not of browning.

    **Inputs.** The fruit table, the feature sets of `pearlatent.feature_sets`, the saved runs
    of `PLSDA_BINARY23_RUN_DIRS` (LVs, scaling, Youden cut-off), the folds of design A,
    Hugo's split (`derived/hugo_vae_split_2425/`). Sealed boxes F, J, M and Hugo's test fruit
    are not loaded.

    **Outputs.** `results/<run_id>_all_volumes_reverse_check/` (save button or
    `-- --save yes`): the two tables and `run.json`.

    **Status.** draft (9 November 2026)
    """)
    return


@app.cell(hide_code=True)
def _():
    cfg = load_config()
    args = mo.cli_args()
    _results = cfg.path("results_dir")
    rows = {}
    for _name, _folder in PLSDA_BINARY23_RUN_DIRS.items():
        _params = json.loads((_results / _folder / "run.json").read_text())["params"]
        rows[_name] = {
            "n_lv": _params["n_lv_chosen"],
            "scale": _params["scale"],
            "cutoff_youden": _params["cutoff_youden"],
            "run": _folder,
        }
    pl.DataFrame([{"row": _k, **_v} for _k, _v in rows.items()])
    return args, cfg, rows


@app.cell(hide_code=True)
def _(cfg):
    all_2526 = open_fruit(cfg).filter(pl.col("season") == "2526")
    kept_ids = set(
        drop_faulty_volumes(all_2526, cfg.path("derived_dir") / "volume_decisions.csv")[
            "fruit_id"
        ].to_list()
    )
    folds = pl.read_csv(cfg.path("derived_dir") / "outer_folds_20" / "outer_folds.csv")

    _split = pl.read_csv(
        cfg.path("derived_dir") / "hugo_vae_split_2425" / "hugo_vae_split_2425.csv"
    )
    _validation_all = (
        open_fruit(cfg)
        .filter(pl.col("season") == "2425")
        .join(_split.select("fruit_id", "hugo_split"), on="fruit_id", validate="1:1")
        .filter(pl.col("hugo_split") == "validation")
    )
    validation = drop_faulty_volumes(
        _validation_all, cfg.path("derived_dir") / "volume_decisions.csv"
    )
    print(
        f"open 2526 fruit: {all_2526.height}; kept after faulty volumes: {len(kept_ids)}"
    )
    print(
        f"Hugo's validation fruit: {_validation_all.height} in, {validation.height} after the"
        f" faulty volumes; non-consumable {validation.filter(pl.col('defective') >= 2).height},"
        f" of which with cavity {validation.filter((pl.col('defective') >= 2) & (pl.col('cavity') > 0)).height}"
    )
    validation.group_by("storage", "defective").len().sort("storage", "defective")
    return all_2526, folds, kept_ids, validation


@app.function(hide_code=True)
def auc_and_interval(table: pl.DataFrame) -> tuple[float, float, float]:
    """AUC of `score` for `binary_23` and its 95 % interval over draws of whole boxes."""

    def _auc(t: pl.DataFrame) -> float:
        if t["binary_23"].n_unique() < 2:
            return np.nan
        return roc_auc_score(t["binary_23"].to_numpy(), t["score"].to_numpy())

    low, high = box_bootstrap(_auc, table)
    return _auc(table), low, high


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1. All 346 volumes, design A

    Every row is fitted again under design A, now with the faulty volumes in the training folds
    as well. The score of a fruit is the mean of its 20 out-of-fold predictions at the row's
    saved LVs.
    """)
    return


@app.cell(hide_code=True)
def _(all_2526, cfg, folds, kept_ids, rows):
    _ids = all_2526["fruit_id"].to_list()
    _y = (all_2526["defective"] >= 2).cast(pl.Int8).to_numpy()
    _boxes = all_2526["box"].to_numpy()
    _kept = np.array([_f in kept_ids for _f in _ids])
    _rows = []
    for _name, _row in rows.items():
        _x = load_features(cfg, _name, _ids).astype(np.float64)
        _predictions = cross_validated_predictions(
            _x, _y, _boxes, folds, _row["scale"], max_components=_row["n_lv"]
        )
        _table = all_2526.select("fruit_id", "box").with_columns(
            binary_23=pl.Series(_y),
            score=pl.Series(_predictions[:, :, -1].mean(axis=0)),
        )
        for _part, _mask in (
            ("all 346", np.ones(len(_ids), bool)),
            ("315 kept", _kept),
            ("31 faulty", ~_kept),
        ):
            _auc, _low, _high = auc_and_interval(_table.filter(pl.Series(_mask)))
            _rows.append(
                {
                    "row": _name,
                    "fruit": _part,
                    "n": int(_mask.sum()),
                    "auc": _auc,
                    "lower_95": _low,
                    "upper_95": _high,
                }
            )
    all_volumes = pl.DataFrame(_rows)
    all_volumes.with_columns(pl.col("auc", "lower_95", "upper_95").round(3))
    return (all_volumes,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2. Reverse check on Hugo's 2425 validation fruit

    One fit per row on the 315 open 2526 fruit, then the validation fruit are predicted. Recall
    and false-positive rate at the fixed cut-off 0.5 and at the row's Youden cut-off (both from
    the 2526 cross-validation, applied unchanged). No interval: the 2425 boxes are orchards that
    also supplied Hugo's training fruit, and 2 storage groups hold all the defects.
    """)
    return


@app.cell(hide_code=True)
def _(all_2526, cfg, kept_ids, rows, validation):
    _train = all_2526.filter(pl.col("fruit_id").is_in(list(kept_ids)))
    _y_train = (_train["defective"] >= 2).cast(pl.Int8).to_numpy().astype(float)
    _y = (validation["defective"] >= 2).cast(pl.Int8).to_numpy()
    _storage = validation["storage"].to_numpy()
    _rows = []
    for _name, _row in rows.items():
        _model = PLSRegression(n_components=_row["n_lv"], scale=_row["scale"])
        _model.fit(load_features(cfg, _name, _train["fruit_id"].to_list()), _y_train)
        _score = _model.predict(
            load_features(cfg, _name, validation["fruit_id"].to_list())
        ).ravel()
        _result = {"row": _name, "n": len(_y), "auc": roc_auc_score(_y, _score)}
        for _cut_name, _cut in (("0.5", 0.5), ("youden", _row["cutoff_youden"])):
            _result[f"recall_{_cut_name}"] = float((_score[_y == 1] >= _cut).mean())
            _result[f"false_positive_rate_{_cut_name}"] = float(
                (_score[_y == 0] >= _cut).mean()
            )
        for _group in np.unique(_storage):
            _result[f"median_score_{_group}"] = float(
                np.median(_score[_storage == _group])
            )
        _rows.append(_result)
    reverse_check = pl.DataFrame(_rows)
    reverse_check.with_columns(pl.selectors.float().round(3))
    return (reverse_check,)


@app.cell(hide_code=True)
def _():
    save_button = mo.ui.run_button(label="Save this run to results/")
    save_button
    return (save_button,)


@app.cell(hide_code=True)
def _(all_volumes, args, cfg, reverse_check, rows, save_button, validation):
    mo.stop(not (save_button.value or args.get("save") == "yes"), mo.md("_Not saved._"))
    _run_dir = new_run_dir(cfg.path("results_dir"), "all_volumes_reverse_check")
    all_volumes.write_parquet(_run_dir / "all_volumes.parquet")
    reverse_check.write_parquet(_run_dir / "reverse_check.parquet")
    write_run_record(
        _run_dir,
        {
            "what": "binary_23 PLS-DA rows on all 346 open 2526 volumes (design A) and the"
            " reverse check on Hugo's 2425 validation fruit",
            "rows": rows,
            "n_validation_fruit": validation.height,
            "folds": "derived/outer_folds_20/outer_folds.csv",
            "bootstrap": "2,000 draws of whole boxes, seed 369",
        },
        root=cfg.root,
    )
    mo.md(f"Saved to `{_run_dir}`.")
    return


if __name__ == "__main__":
    app.run()
