import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title="15 Spread of the fruit interior")

with app.setup(hide_code=True):
    import json

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score

    from pearlatent.config import load_config
    from pearlatent.feature_sets import load_features
    from pearlatent.fruit_table import drop_faulty_volumes, open_fruit
    from pearlatent.interior import N_EDGE_LAYERS, interior_cells, interior_sd
    from pearlatent.nuisance import box_bootstrap, within_box_auc
    from pearlatent.runs import (
        ENCODING_RUN_DIR,
        LOGVAR_RUN_DIR,
        new_run_dir,
        write_run_record,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 15 Spread of the fruit interior: where is grade 1 lost?

    **Question.** Grade 0 and grade 1 hardly separate in any PLS-DA row (AUC 0.59 to 0.72).
    Is grade 1 not visible in the CT at all, or visible in the CT but lost in the latent? One
    number per fruit, no model: how uneven the interior of the fruit is.

    **Three measures per fruit**, each over the interior body cells of the 32³ grid (the body
    without its outer layer of cells, `pearlatent.interior`):

    1. CT at 32³: SD of the cell values (blocks of 4³ voxels).
    2. Hugo's latent `mu`: SD of the cell values.
    3. The encoder's own uncertainty: median of exp(logvar / 2) over the same cells. This is
       the SD the encoder gives each cell, a second output beside `mu` that the encoding run
       did not save (`pearlatent.logvar_run`).

    **Why without the outer layer, and a caveat.** The outer cells straddle skin and air and
    vary in every fruit. Chosen after looking: in a scratch run (9 November 2026) the CT SD over
    all body cells did not separate grade 0 from 1 (AUC 0.53), without one layer it did (0.82),
    and without 2 or 3 layers the same (0.82). So this notebook confirms a measure found by
    looking; the sealed test boxes decide.

    **Fruit.** The 315 open 2526 fruit (faulty volumes removed). The grade comparisons use the
    cavity-free fruit (cavity 0), so that a gas void cannot carry the result. No model is fitted,
    so there is no cross-validation; intervals come from 2,000 draws of whole boxes (seed 369).

    **Inputs.** The fruit table, `ct_32` and `hugo_vae` (`pearlatent.feature_sets`), the 32³ body
    cells of the encoding run, the logvar run. Sealed boxes F, J, M are not loaded.

    **Outputs.** `results/<run_id>_interior_spread/` (save button or `-- --save yes`): one row
    per fruit, the summary table, `run.json`; the figure `derived/figures/15_interior_spread.png`.

    **Status.** draft (9 November 2026)
    """)
    return


@app.cell(hide_code=True)
def _():
    cfg = load_config()
    args = mo.cli_args()
    _all = open_fruit(cfg).filter(pl.col("season") == "2526")
    fruit = drop_faulty_volumes(_all, cfg.path("derived_dir") / "volume_decisions.csv")
    print(f"open 2526 fruit: {_all.height} in, {fruit.height} after the faulty volumes")
    print(f"cavity-free: {fruit.filter(pl.col('cavity') == 0).height}")
    return args, cfg, fruit


@app.cell(hide_code=True)
def _(cfg, fruit):
    _ids = fruit["fruit_id"].to_list()
    # Body cells from the encoding run, in the row order of its manifest.
    _run = cfg.path("derived_dir") / ENCODING_RUN_DIR
    _manifest = pl.read_parquet(_run / "manifest.parquet").with_row_index("position")
    _positions = (
        pl.DataFrame({"fruit_id": _ids})
        .join(
            _manifest.select("fruit_id", "position"),
            on="fruit_id",
            how="left",
            validate="1:1",
        )["position"]
        .to_numpy()
    )
    body = np.asarray(
        np.load(_run / "masks.npy", mmap_mode="r")[_positions], dtype=bool
    )

    _ct = load_features(cfg, "ct_32", _ids).reshape(-1, 32, 32, 32)
    _latent = load_features(cfg, "hugo_vae", _ids).reshape(-1, 32, 32, 32)

    # The encoder SD: logvar run, joined on fruit_id.
    _lv_run = cfg.path("derived_dir") / LOGVAR_RUN_DIR
    _lv_manifest = pl.read_parquet(_lv_run / "manifest.parquet")
    _lv_rows = (
        pl.DataFrame({"fruit_id": _ids})
        .join(_lv_manifest, on="fruit_id", how="left", validate="1:1")["row"]
        .to_numpy()
    )
    _logvar = np.load(_lv_run / "logvar.npy", mmap_mode="r")
    _encoder_sd = np.array(
        [
            np.median(np.exp(0.5 * _logvar[_r][interior_cells(body[_k])]))
            for _k, _r in enumerate(_lv_rows)
        ]
    )

    measures = fruit.select(
        "fruit_id",
        "box",
        "browning",
        "cavity",
        "defective",
        "cause",
        "stalk_length",
        "body_voxels",
    ).with_columns(
        ct_interior_sd=pl.Series(interior_sd(_ct, body)),
        latent_interior_sd=pl.Series(interior_sd(_latent, body)),
        encoder_sd=pl.Series(_encoder_sd),
    )
    print(f"{measures.height} fruit, {N_EDGE_LAYERS} outer layer of cells removed")
    measures.head(5)
    return (measures,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1. How well does each number separate the grades?

    AUC = the share of pairs (one fruit of the higher grade, one of the lower) in which the
    higher grade has the larger value; 0.5 is no separation. Neighbouring grades on the
    cavity-free fruit; `binary_23` on all fruit as the reference task of notebook 08.
    Within-box AUC of grade 0 against 1: only pairs from the same box, so a difference between
    orchards cannot produce it. Size and stalk: Spearman correlation with body size and stalk
    length on the grade 0 fruit, each value minus its box median (check C of notebook 10).
    """)
    return


@app.function(hide_code=True)
def auc_with_interval(
    table: pl.DataFrame, column: str, label: str
) -> tuple[float, float, float]:
    """AUC of `column` for the 0/1 column `label`, with its 95 % box-bootstrap interval."""

    def _auc(t: pl.DataFrame) -> float:
        if t[label].n_unique() < 2:
            return np.nan
        return roc_auc_score(t[label].to_numpy(), t[column].to_numpy())

    low, high = box_bootstrap(_auc, table)
    return _auc(table), low, high


@app.cell(hide_code=True)
def _(measures):
    _cavity_free = measures.filter(pl.col("cavity") == 0)
    _comparisons = {
        "grade 0 against 1 (cavity-free)": _cavity_free.filter(
            pl.col("defective") <= 1
        ).with_columns(label=(pl.col("defective") == 1).cast(pl.Int8)),
        "grade 1 against 2 (cavity-free)": _cavity_free.filter(
            pl.col("defective").is_in([1, 2])
        ).with_columns(label=(pl.col("defective") == 2).cast(pl.Int8)),
        "grade 2 against 3 (cavity-free)": _cavity_free.filter(
            pl.col("defective") >= 2
        ).with_columns(label=(pl.col("defective") == 3).cast(pl.Int8)),
        "binary_23 (all fruit)": measures.with_columns(
            label=(pl.col("defective") >= 2).cast(pl.Int8)
        ),
    }
    _healthy = measures.filter(pl.col("defective") == 0)
    _rows = []
    for _column in ("ct_interior_sd", "latent_interior_sd", "encoder_sd"):
        for _name, _table in _comparisons.items():
            _auc, _low, _high = auc_with_interval(_table, _column, "label")
            _rows.append(
                {
                    "measure": _column,
                    "comparison": _name,
                    "n": _table.height,
                    "auc": _auc,
                    "lower_95": _low,
                    "upper_95": _high,
                }
            )
        _zero_one = _comparisons["grade 0 against 1 (cavity-free)"]
        _in_box = _healthy.with_columns(
            _r=pl.col(_column) - pl.col(_column).median().over("box")
        )
        _rows.append(
            {
                "measure": _column,
                "comparison": "grade 0 against 1, within box",
                "n": _zero_one.height,
                "auc": within_box_auc(
                    _zero_one[_column].to_numpy(),
                    _zero_one["label"].to_numpy(),
                    _zero_one["box"].to_numpy(),
                ),
                "lower_95": None,
                "upper_95": None,
            }
        )
        for _descriptor in ("stalk_length", "body_voxels"):
            _rows.append(
                {
                    "measure": _column,
                    "comparison": f"grade 0: Spearman with {_descriptor}",
                    "n": _in_box.height,
                    "auc": float(
                        spearmanr(_in_box["_r"], _in_box[_descriptor]).statistic
                    ),
                    "lower_95": None,
                    "upper_95": None,
                }
            )
    summary = pl.DataFrame(_rows)
    summary.with_columns(pl.col("auc", "lower_95", "upper_95").round(3))
    return (summary,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2. The three numbers by grade (cavity-free fruit)

    Each dot is one fruit, the line is the median of its grade. The left panel is the CT, the
    middle panel the latent of the same fruit, the right panel the encoder's own SD.
    """)
    return


@app.cell(hide_code=True)
def _(measures, summary):
    _cavity_free = measures.filter(pl.col("cavity") == 0)
    _titles = {
        "ct_interior_sd": "CT at 32³: SD of the interior",
        "latent_interior_sd": "Latent mu: SD of the interior",
        "encoder_sd": "Encoder SD (median over the interior)",
    }
    figure, _axes = plt.subplots(1, 3, figsize=(11, 3.6))
    _rng = np.random.default_rng(369)  # horizontal jitter of the dots only
    for _ax, (_column, _title) in zip(_axes, _titles.items(), strict=True):
        for _grade in range(4):
            _v = _cavity_free.filter(pl.col("defective") == _grade)[_column].to_numpy()
            _ax.scatter(_grade + _rng.uniform(-0.15, 0.15, len(_v)), _v, s=8, alpha=0.5)
            _ax.hlines(np.median(_v), _grade - 0.3, _grade + 0.3, color="black")
        _row = summary.filter(
            (pl.col("measure") == _column)
            & (pl.col("comparison") == "grade 0 against 1 (cavity-free)")
        ).row(0, named=True)
        _ax.set_title(
            f"{_title}\ngrade 0 vs 1: AUC {_row['auc']:.2f}"
            f" ({_row['lower_95']:.2f} to {_row['upper_95']:.2f})",
            fontsize=9,
        )
        _ax.set_xticks(range(4))
        _ax.set_xlabel("browning grade (cavity 0)")
    _counts = [_cavity_free.filter(pl.col("defective") == _g).height for _g in range(4)]
    figure.suptitle(
        f"{_cavity_free.height} cavity-free open 2526 fruit (grade 0 to 3: {_counts});"
        " interval: 2,000 box draws",
        fontsize=9,
    )
    figure.tight_layout()
    figure
    return (figure,)


@app.cell(hide_code=True)
def _():
    save_button = mo.ui.run_button(label="Save this run to results/ and the figure")
    save_button
    return (save_button,)


@app.cell(hide_code=True)
def _(args, cfg, figure, measures, save_button, summary):
    mo.stop(not (save_button.value or args.get("save") == "yes"), mo.md("_Not saved._"))
    _run_dir = new_run_dir(cfg.path("results_dir"), "interior_spread")
    measures.write_parquet(_run_dir / "measures.parquet")
    summary.write_parquet(_run_dir / "summary.parquet")
    write_run_record(
        _run_dir,
        {
            "what": "SD of the fruit interior per fruit (CT 32³, latent mu, encoder SD)",
            "n_fruit": measures.height,
            "n_edge_layers": N_EDGE_LAYERS,
            "encoding_run": ENCODING_RUN_DIR,
            "logvar_run": LOGVAR_RUN_DIR,
            "bootstrap": "2,000 draws of whole boxes, seed 369",
        },
        root=cfg.root,
    )
    _figures = cfg.path("derived_dir") / "figures"
    figure.savefig(_figures / "15_interior_spread.png", dpi=150)
    (_run_dir / "summary.json").write_text(json.dumps(summary.to_dicts(), indent=2))
    mo.md(f"Saved to `{_run_dir}` and `{_figures / '15_interior_spread.png'}`.")
    return


if __name__ == "__main__":
    app.run()
