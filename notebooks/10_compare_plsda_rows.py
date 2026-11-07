import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title="10 Compare the PLS-DA rows")

with app.setup(hide_code=True):
    import itertools
    import json

    import altair as alt
    import marimo as mo
    import numpy as np
    import polars as pl
    from sklearn.metrics import roc_auc_score

    from pearlatent.config import load_config
    from pearlatent.fruit_table import drop_faulty_volumes, open_fruit
    from pearlatent.nuisance import (
        box_bootstrap,
        healthy_fruit_correlations,
        score_icc,
        within_between_logit,
        within_box_auc,
    )
    from pearlatent.plsda import (
        BOOTSTRAP_RESAMPLES,
        BOOTSTRAP_SEED,
        paired_box_bootstrap_auc_difference,
    )
    from pearlatent.runs import PLSDA_BINARY23_RUN_DIRS, new_run_dir, write_run_record


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 10 Compare the four PLS-DA rows on binary_23

    **Question.** Is one representation better than another at telling consumable from
    non-consumable fruit, by more than the variation between orchards (boxes)? And does any
    row owe its score to the box rather than to the fruit (nuisance checks A to C)?

    **What is compared.** The four PLS-DA runs of notebook 08 (saved 3 November 2026):
    Hugo's latent (full 32³ grid, 4 latent variables), CT averaged to 32³ (5), tissue
    features (7), Med3D (7). Nothing is refitted here: the notebook only reads the
    out-of-fold predictions (one number per fruit: the prediction averaged over the 20
    repeats of the box cross-validation).

    **Method.**
    - Paired box bootstrap of the difference in AUC (area under the ROC curve), for all six
      pairs of rows (`docs/cv_explained.md`, section 7;
      `pearlatent.plsda.paired_box_bootstrap_auc_difference`).
    - Nuisance checks A to C on each row (`docs/methods_plsda.md`, section 6;
      `pearlatent.nuisance`).

    **Inputs.** `results/<run>/predictions.parquet`, `run.json` and `scores.json` of the four
    runs in `pearlatent.runs.PLSDA_BINARY23_RUN_DIRS`; the fruit table with the descriptors
    (`pearlatent.fruit_table`) for check C. Only the 12 open 2526 boxes; the sealed test
    boxes are not in these runs.

    **Outputs.** `results/<run_id>_compare_plsda_binary23/` (only when the save button is
    pressed): the pair table, the bootstrap differences, the nuisance tables and `run.json`.

    **Status.** draft (7 November 2026)
    """)
    return


@app.cell
def _():
    cfg = load_config()
    ROW_LABELS = {
        "hugo_vae": "Hugo's latent",
        "ct_32": "CT at 32³",
        "tissue_features": "Tissue features",
        "med3d": "Med3D",
    }
    return ROW_LABELS, cfg


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1. Load the four runs and check that they cover the same fruit

    The paired comparison only makes sense when both rows predicted the same fruit with the
    same labels. The cell below joins the four prediction tables on `fruit_id` (one-to-one)
    and stops if any fruit is missing in one run or if a label differs between runs.
    """)
    return


@app.cell
def _(ROW_LABELS, cfg):
    _results = cfg.path("results_dir")
    _label_columns = ["fruit_id", "box", "defective", "browning", "cavity", "cause", "binary_23"]

    run_records = {}
    stored_scores = {}
    _tables = []
    for _row, _folder in PLSDA_BINARY23_RUN_DIRS.items():
        _dir = _results / _folder
        run_records[_row] = json.loads((_dir / "run.json").read_text())
        stored_scores[_row] = json.loads((_dir / "scores.json").read_text())
        _table = pl.read_parquet(_dir / "predictions.parquet")
        print(f"{ROW_LABELS[_row]:16s} {_folder}: {_table.height} fruit")
        _tables.append(
            _table.select(_label_columns + ["prediction"]).rename(
                {"prediction": f"prediction_{_row}"}
            )
        )

    # Join the four tables. The labels are part of the join key, so a fruit whose label
    # differs between two runs would not match and the count check below would fail.
    predictions = _tables[0]
    for _table in _tables[1:]:
        predictions = predictions.join(_table, on=_label_columns, how="inner", validate="1:1")
    if not all(_t.height == predictions.height for _t in _tables):
        raise ValueError("the four runs do not hold the same fruit with the same labels")
    print(f"\nfruit in all four runs, same labels: {predictions.height}")
    print(f"boxes: {predictions['box'].n_unique()}")
    print(f"non-consumable (binary_23 = 1): {predictions['binary_23'].sum()}")
    return predictions, run_records, stored_scores


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2. AUC of each row, recomputed

    The AUC is computed again from the saved predictions and set beside the value that
    notebook 08 stored in `scores.json`. The two columns must agree; if they do not, the
    prediction file and the score file belong to different computations.
    """)
    return


@app.cell
def _(ROW_LABELS, predictions, run_records, stored_scores):
    _y = predictions["binary_23"].to_numpy()
    auc_check = pl.DataFrame(
        [
            {
                "row": ROW_LABELS[_row],
                "latent_variables": run_records[_row]["params"]["n_lv_chosen"],
                "auc_recomputed": roc_auc_score(_y, predictions[f"prediction_{_row}"]),
                "auc_in_scores_json": stored_scores[_row]["auc"],
                "box_bootstrap_lower": stored_scores[_row]["auc_box_bootstrap_95"][0],
                "box_bootstrap_upper": stored_scores[_row]["auc_box_bootstrap_95"][1],
            }
            for _row in ROW_LABELS
        ]
    )
    _largest_gap = (auc_check["auc_recomputed"] - auc_check["auc_in_scores_json"]).abs().max()
    print(f"largest difference recomputed vs stored: {_largest_gap:.2e}")
    auc_check
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 3. The paired box bootstrap, one draw worked out

    One draw: pick 12 boxes out of the 12 open boxes **with replacement** (some boxes come
    twice, some not at all), take all fruit of the drawn boxes, compute the AUC of row A and
    of row B on exactly those fruit, keep A minus B. Repeat 2,000 times (seed 369). The
    middle 95 % of the 2,000 differences is the interval. If it does not contain 0, the
    difference between the rows is larger than what changing the set of orchards does.

    The cell below shows the **first** draw for Med3D (A) against Hugo's latent (B), computed
    by hand. The last line checks that the library function gets the same first difference,
    so you can verify that the function does what this text says.
    """)
    return


@app.cell
def _(predictions):
    _y = predictions["binary_23"].to_numpy()
    _boxes = predictions["box"].to_numpy()
    _a = predictions["prediction_med3d"].to_numpy()
    _b = predictions["prediction_hugo_vae"].to_numpy()

    # The same random generator and the same call as in the library function, so the first
    # draw here is the first draw there.
    _names = np.unique(_boxes)
    _drawn = np.random.default_rng(BOOTSTRAP_SEED).choice(
        _names, size=len(_names), replace=True
    )
    _index = np.concatenate([np.flatnonzero(_boxes == _box) for _box in _drawn])
    _auc_a = roc_auc_score(_y[_index], _a[_index])
    _auc_b = roc_auc_score(_y[_index], _b[_index])

    _counts = {str(_box): int((_drawn == _box).sum()) for _box in _names}
    print("times each box was drawn:", _counts)
    print(f"fruit in this draw: {_index.size} (a box drawn twice counts its fruit twice)")
    print(f"non-consumable in this draw: {int(_y[_index].sum())}")
    print(f"AUC Med3D {_auc_a:.4f}, AUC Hugo's latent {_auc_b:.4f}, difference {_auc_a - _auc_b:+.4f}")

    _check = paired_box_bootstrap_auc_difference(_a, _b, _y, _boxes, n_resamples=1)
    print(f"first difference from the library function: {_check['differences'][0]:+.4f}")
    return


@app.cell
def _():
    run_bootstrap = mo.ui.run_button(
        label="Run the bootstrap and the nuisance checks (about one minute)"
    )
    run_bootstrap
    return (run_bootstrap,)


@app.cell
def _(ROW_LABELS, predictions, run_bootstrap):
    # In script mode the button is never pressed, so the cell runs without it.
    mo.stop(mo.running_in_notebook() and not run_bootstrap.value)
    _y = predictions["binary_23"].to_numpy()
    _boxes = predictions["box"].to_numpy()

    _rows = []
    pair_differences = {}
    for _a, _b in itertools.combinations(ROW_LABELS, 2):
        _result = paired_box_bootstrap_auc_difference(
            predictions[f"prediction_{_a}"].to_numpy(),
            predictions[f"prediction_{_b}"].to_numpy(),
            _y,
            _boxes,
        )
        pair_differences[f"{_a}_minus_{_b}"] = _result.pop("differences")
        _rows.append({"row_a": ROW_LABELS[_a], "row_b": ROW_LABELS[_b], **_result})
    pair_table = pl.DataFrame(_rows).with_columns(
        interval_excludes_0=(pl.col("lower_95") > 0) | (pl.col("upper_95") < 0)
    )
    print(f"draws per pair: {BOOTSTRAP_RESAMPLES}, seed {BOOTSTRAP_SEED}")
    print(
        "draws kept per pair (a draw without both classes is skipped):",
        {_k: _v.size for _k, _v in pair_differences.items()},
    )
    pair_table
    return pair_differences, pair_table


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    How to read the table: `difference` is AUC of row A minus AUC of row B on all 315
    fruit; `lower_95` and `upper_95` bound the middle 95 % of the 2,000 paired differences;
    `share_a_better` is the share of draws in which row A had the higher AUC. The histogram
    below shows the 2,000 differences of one pair; the red line is 0.
    """)
    return


@app.cell
def _(pair_differences):
    pair_choice = mo.ui.dropdown(
        options=list(pair_differences), value="hugo_vae_minus_med3d", label="Pair"
    )
    pair_choice
    return (pair_choice,)


@app.cell
def _(pair_choice, pair_differences):
    _values = pl.DataFrame({"difference": pair_differences[pair_choice.value]})
    _bars = (
        alt.Chart(_values)
        .mark_bar()
        .encode(
            alt.X("difference:Q", bin=alt.Bin(maxbins=50), title="AUC of A minus AUC of B"),
            alt.Y("count():Q", title="draws"),
        )
    )
    _zero = alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color="red").encode(x="x:Q")
    (_bars + _zero).properties(width=500, height=220, title=pair_choice.value)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 4. Nuisance checks A to C

    The "score" of a fruit is its out-of-fold prediction. Same computation as on 3 November
    (`docs/methods_plsda.md`, section 6), now saved as a run:

    - **A. Within-box AUC.** The AUC using only pairs (one non-consumable, one consumable
      fruit) from the same box. Close to the overall AUC: the row does not need differences
      between boxes to rank the fruit. Interval: 2,000 box draws, seed 369.
    - **B. Box mean against within-box deviation.** Logistic regression of `binary_23` on
      the box mean of the score and the fruit's deviation from it, both per standard
      deviation of the score, standard errors clustered by box. The within-box coefficient
      should be clearly above 0. ICC: share of the score's variation between boxes given the
      grade (linear mixed model, random intercept per box).
    - **C. Healthy fruit only (grade 0).** Spearman correlation of the score (minus its box
      median) with stalk length and with body size. Should be near 0: the score should not
      follow the stalk or the size. Interval: 1,000 box draws.
    """)
    return


@app.cell
def _(cfg, predictions):
    # Stalk length, body size and stalk presence from the descriptors run, joined one to
    # one on fruit_id. Only fruit in the predictions are kept (inner join on the right side).
    _descriptors = drop_faulty_volumes(
        open_fruit(cfg).filter(pl.col("season") == "2526"),
        cfg.path("derived_dir") / "volume_decisions.csv",
    ).select("fruit_id", "stalk_length", "body_voxels", "stalk_present")
    fruit_with_descriptors = predictions.join(
        _descriptors, on="fruit_id", how="left", validate="1:1"
    )
    _missing = fruit_with_descriptors["stalk_length"].null_count()
    print(f"fruit {predictions.height}, without descriptors: {_missing}")
    print(f"grade 0 fruit (used in check C): {(predictions['defective'] == 0).sum()}")
    return (fruit_with_descriptors,)


@app.cell
def _(ROW_LABELS, fruit_with_descriptors, run_bootstrap):
    mo.stop(mo.running_in_notebook() and not run_bootstrap.value)
    # Check C interval with 1,000 draws (as on 3 November): it refits nothing, but the
    # box median is recomputed in every draw, which is slower than an AUC.
    _check_c_resamples = 1000

    _rows_a, _rows_b, _rows_c = [], [], []
    for _row, _label in ROW_LABELS.items():
        _table = fruit_with_descriptors.with_columns(score=pl.col(f"prediction_{_row}"))

        # Check A
        def _within(t):
            return within_box_auc(t["score"], t["binary_23"], t["box"])

        _lower, _upper = box_bootstrap(_within, _table)
        _rows_a.append(
            {
                "row": _label,
                "overall_auc": roc_auc_score(_table["binary_23"], _table["score"]),
                "within_box_auc": _within(_table),
                "lower_95": _lower,
                "upper_95": _upper,
            }
        )

        # Check B
        _rows_b.append(
            within_between_logit(_table).with_columns(
                row=pl.lit(_label), icc_given_grade=pl.lit(score_icc(_table))
            )
        )

        # Check C
        _c = healthy_fruit_correlations(_table)
        for _measure in ["stalk_length", "body_voxels"]:
            _key = f"spearman_{_measure}"
            _lower, _upper = box_bootstrap(
                lambda t, k=_key: healthy_fruit_correlations(t)[k],
                _table,
                n_resamples=_check_c_resamples,
            )
            _rows_c.append(
                {
                    "row": _label,
                    "measure": _measure,
                    "n_healthy": _c["n_healthy"],
                    "spearman": _c[_key],
                    "lower_95": _lower,
                    "upper_95": _upper,
                }
            )

    check_a = pl.DataFrame(_rows_a)
    check_b = pl.concat(_rows_b).select("row", pl.exclude("row"))
    check_c = pl.DataFrame(_rows_c)
    mo.vstack(
        [
            mo.md("**A. Within-box AUC**"),
            check_a,
            mo.md("**B. Box mean against within-box deviation (log-odds per SD)**"),
            check_b,
            mo.md("**C. Grade 0 fruit: Spearman correlation with stalk and size**"),
            check_c,
        ]
    )
    return check_a, check_b, check_c


@app.cell
def _():
    save_button = mo.ui.run_button(label="Save this comparison to results/")
    save_button
    return (save_button,)


@app.cell
def _(
    cfg,
    check_a,
    check_b,
    check_c,
    pair_differences,
    pair_table,
    predictions,
    run_records,
    save_button,
):
    mo.stop(not save_button.value, mo.md("_Press the save button to write the run._"))
    _dir = new_run_dir(cfg.path("results_dir"), "compare_plsda_binary23")
    pair_table.write_parquet(_dir / "pair_differences.parquet")
    np.savez(_dir / "bootstrap_differences.npz", **pair_differences)
    check_a.write_parquet(_dir / "check_a_within_box_auc.parquet")
    check_b.write_parquet(_dir / "check_b_box_mean_vs_within.parquet")
    check_c.write_parquet(_dir / "check_c_healthy_correlations.parquet")
    write_run_record(
        _dir,
        {
            "what": "paired box bootstrap of AUC differences and nuisance checks A to C "
            "for the four PLS-DA rows on binary_23 (open 2526 boxes)",
            "plsda_runs": PLSDA_BINARY23_RUN_DIRS,
            "plsda_run_commits": {
                _row: _record["git"]["commit"] for _row, _record in run_records.items()
            },
            "n_fruit": predictions.height,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_resamples_check_c": 1000,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
    )
    mo.md(f"Saved to `{_dir.name}`.")
    return


if __name__ == "__main__":
    app.run()
