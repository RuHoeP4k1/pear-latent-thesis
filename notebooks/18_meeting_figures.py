import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title="18 Figures for the meeting of 17 November")

with app.setup(hide_code=True):
    import json

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
    from sklearn.metrics import roc_auc_score

    from pearlatent.config import load_config
    from pearlatent.nuisance import box_bootstrap
    from pearlatent.runs import PLSDA_BINARY23_RUN_DIRS, new_run_dir, write_run_record


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 18 Figures for the meeting of 17 November

    **Question.** Two pictures of the saved `binary_23` PLS-DA rows: (1) how well each row
    separates consumable from non-consumable fruit, on all fruit and on cavity-free fruit;
    (2) which defects each row detects: browning alone at grades 1, 2, 3 against cavity.

    **Nothing is fitted here.** The notebook reads the out-of-fold score of every fruit (mean of
    the 20 repeats of the box cross-validation) from the saved runs of
    `PLSDA_BINARY23_RUN_DIRS`, the same runs as notebook 10. Intervals: 2,000 draws of whole
    boxes, seed 369 (`pearlatent.nuisance.box_bootstrap`). The third meeting figure, the spread
    of the fruit interior, is made by notebook 15.

    **Inputs.** `results/<run>/predictions.parquet` of the eight runs.

    **Outputs.** `derived/figures/18_auc_rows.png`, `derived/figures/18_auc_by_cause.png` and
    `results/<run_id>_meeting_figures/` with the two tables (save button or `-- --save yes`).

    **Status.** draft (9 November 2026)
    """)
    return


@app.cell(hide_code=True)
def _():
    cfg = load_config()
    args = mo.cli_args()
    LABELS = {
        "hugo_vae": "Latent, grid",
        "hugo_vae_hr": "Latent, rings: mean",
        "hugo_vae_hr_sd": "Latent, rings: mean + SD",
        "ct_32": "CT 32³, grid",
        "ct_32_hr": "CT 32³, rings: mean",
        "ct_32_hr_sd": "CT 32³, rings: mean + SD",
        "tissue_features": "Tissue features (128³)",
        "med3d": "Med3D (128³)",
    }
    _tables = []
    for _name, _folder in PLSDA_BINARY23_RUN_DIRS.items():
        _tables.append(
            pl.read_parquet(cfg.path("results_dir") / _folder / "predictions.parquet")
            .select(
                "fruit_id",
                "box",
                "defective",
                "browning",
                "cavity",
                "binary_23",
                "prediction",
            )
            .with_columns(row=pl.lit(_name))
        )
    scores = pl.concat(_tables)
    print(scores.group_by("row").len().sort("row"))
    return LABELS, args, cfg, scores


@app.function(hide_code=True)
def auc_interval(table: pl.DataFrame, label: str) -> dict:
    """AUC of `prediction` for the 0/1 column `label`, with its 95 % box-bootstrap interval."""

    def _auc(t: pl.DataFrame) -> float:
        if t[label].n_unique() < 2:
            return np.nan
        return roc_auc_score(t[label].to_numpy(), t["prediction"].to_numpy())

    low, high = box_bootstrap(_auc, table)
    return {"auc": _auc(table), "lower_95": low, "upper_95": high, "n": table.height}


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1. AUC per row: all fruit and cavity-free fruit

    All fruit: the 315 open 2526 fruit. Cavity-free: the 233 fruit with cavity 0, so only
    browning can make a fruit non-consumable.
    """)
    return


@app.cell(hide_code=True)
def _(LABELS, scores):
    _rows = []
    for _name in LABELS:
        _t = scores.filter(pl.col("row") == _name)
        _rows.append({"row": _name, "fruit": "all", **auc_interval(_t, "binary_23")})
        _rows.append(
            {
                "row": _name,
                "fruit": "cavity-free",
                **auc_interval(_t.filter(pl.col("cavity") == 0), "binary_23"),
            }
        )
    auc_rows = pl.DataFrame(_rows)
    auc_rows.with_columns(pl.col("auc", "lower_95", "upper_95").round(3))
    return (auc_rows,)


@app.cell(hide_code=True)
def _(LABELS, auc_rows):
    figure_rows, _ax = plt.subplots(figsize=(7, 4))
    _order = list(LABELS)[::-1]  # first row at the top
    for _offset, _fruit, _colour in ((0.15, "all", "C0"), (-0.15, "cavity-free", "C1")):
        _t = auc_rows.filter(pl.col("fruit") == _fruit)
        _t = _t.join(pl.DataFrame({"row": _order, "y": range(len(_order))}), on="row")
        _ax.errorbar(
            _t["auc"],
            _t["y"] + _offset,
            xerr=[_t["auc"] - _t["lower_95"], _t["upper_95"] - _t["auc"]],
            fmt="o",
            color=_colour,
            capsize=2,
            label=f"{_fruit} (n = {_t['n'][0]})",
        )
    _ax.set_yticks(range(len(_order)), [LABELS[_k] for _k in _order])
    _ax.set_xlabel("AUC, consumable against non-consumable (binary_23)")
    _ax.set_xlim(0.7, 1.0)
    _ax.grid(axis="x", alpha=0.3)
    _ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=8)
    _ax.set_title(
        "PLS-DA, 12 open 2526 boxes, 20 × 4-fold box cross-validation;"
        " 95 % interval: 2,000 box draws",
        fontsize=8,
    )
    figure_rows.tight_layout()
    figure_rows
    return (figure_rows,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2. Which defects does each row detect?

    AUC of grade 0 against one group of defective fruit: browning only (cavity 0) at grade 1,
    2 and 3, and fruit with cavity 2 or 3 (any browning). 0.5 = the group scores like healthy
    fruit. Four rows, to keep the figure readable; all eight are in the saved table.
    """)
    return


@app.cell(hide_code=True)
def _(scores):
    _groups = {
        "browning only 1": (pl.col("browning") == 1) & (pl.col("cavity") == 0),
        "browning only 2": (pl.col("browning") == 2) & (pl.col("cavity") == 0),
        "browning only 3": (pl.col("browning") == 3) & (pl.col("cavity") == 0),
        "cavity 2 or 3": pl.col("cavity") >= 2,
    }
    _rows = []
    for _name in scores["row"].unique(maintain_order=True):
        _t = scores.filter(pl.col("row") == _name)
        for _group, _condition in _groups.items():
            _pair = _t.filter((pl.col("defective") == 0) | _condition).with_columns(
                in_group=_condition.cast(pl.Int8)
            )
            _rows.append(
                {"row": _name, "group": _group, **auc_interval(_pair, "in_group")}
            )
    auc_by_cause = pl.DataFrame(_rows)
    auc_by_cause.with_columns(pl.col("auc", "lower_95", "upper_95").round(3))
    return (auc_by_cause,)


@app.cell(hide_code=True)
def _(LABELS, auc_by_cause, scores):
    _shown = ["hugo_vae", "hugo_vae_hr_sd", "ct_32_hr_sd", "med3d"]
    _groups = auc_by_cause["group"].unique(maintain_order=True).to_list()
    _healthy = scores.filter(
        (pl.col("row") == "hugo_vae") & (pl.col("defective") == 0)
    ).height
    figure_cause, _ax = plt.subplots(figsize=(7, 3.8))
    for _i, _name in enumerate(_shown):
        _t = auc_by_cause.filter(pl.col("row") == _name)
        _x = np.arange(len(_groups)) + (_i - 1.5) * 0.18
        _ax.errorbar(
            _x,
            _t["auc"],
            yerr=[_t["auc"] - _t["lower_95"], _t["upper_95"] - _t["auc"]],
            fmt="o",
            capsize=2,
            label=LABELS[_name],
        )
    _n = auc_by_cause.filter(pl.col("row") == "hugo_vae")["n"].to_list()
    _ax.set_xticks(
        range(len(_groups)),
        [f"{_g}\n(n = {_m - _healthy})" for _g, _m in zip(_groups, _n)],
    )
    _ax.axhline(0.5, color="black", lw=0.6, ls=":")
    _ax.set_ylim(0.4, 1.0)
    _ax.set_ylabel(f"AUC against grade 0 (n = {_healthy})")
    _ax.grid(axis="y", alpha=0.3)
    _ax.legend(fontsize=7, loc="lower right")
    _ax.set_title(
        "Same saved scores as figure 1; 95 % interval: 2,000 box draws", fontsize=8
    )
    figure_cause.tight_layout()
    figure_cause
    return (figure_cause,)


@app.cell(hide_code=True)
def _():
    save_button = mo.ui.run_button(label="Save the figures and tables")
    save_button
    return (save_button,)


@app.cell(hide_code=True)
def _(args, auc_by_cause, auc_rows, cfg, figure_cause, figure_rows, save_button):
    mo.stop(not (save_button.value or args.get("save") == "yes"), mo.md("_Not saved._"))
    _run_dir = new_run_dir(cfg.path("results_dir"), "meeting_figures")
    auc_rows.write_parquet(_run_dir / "auc_rows.parquet")
    auc_by_cause.write_parquet(_run_dir / "auc_by_cause.parquet")
    write_run_record(
        _run_dir,
        {
            "what": "AUC per binary_23 row (all and cavity-free fruit) and per defect group",
            "runs": PLSDA_BINARY23_RUN_DIRS,
            "bootstrap": "2,000 draws of whole boxes, seed 369",
        },
        root=cfg.root,
    )
    _figures = cfg.path("derived_dir") / "figures"
    figure_rows.savefig(_figures / "18_auc_rows.png", dpi=150)
    figure_cause.savefig(_figures / "18_auc_by_cause.png", dpi=150)
    (_run_dir / "figures.json").write_text(
        json.dumps(
            ["derived/figures/18_auc_rows.png", "derived/figures/18_auc_by_cause.png"]
        )
    )
    mo.md(f"Saved to `{_run_dir}` and `{_figures}`.")
    return


if __name__ == "__main__":
    app.run()
