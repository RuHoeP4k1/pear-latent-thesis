import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title="17 Counterfactual images")

with app.setup(hide_code=True):
    import marimo as mo
    import matplotlib.pyplot as plt
    import nibabel as nib
    import numpy as np
    import polars as pl

    from pearlatent.config import load_config
    from pearlatent.counterfactual import DRAW_SEED, draw_example_fruit
    from pearlatent.fruit_table import drop_faulty_volumes, open_fruit
    from pearlatent.pooling import scale_minmax
    from pearlatent.reconstruction import body_centre, body_voxel_mask, central_planes
    from pearlatent.runs import (
        ENCODING_RUN_DIR,
        PLSDA_BINARY23_RUN_DIRS,
        latest_run_dir,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 17 Counterfactual images

    **Question.** What does a fruit look like when its latent `mu` is moved along the direction
    that a PLS-DA model uses to call a fruit non-consumable? Charter, Extensions.

    **Plan** (Ruben, 7 and 9 November 2026). Two directions, each the regression vector of one
    PLS-DA model on the full 32³ latent grid, fitted once on all 12 open boxes: the cavity-free
    browning ≥ 2 model (5 LVs, out-of-fold AUC 0.848) and the general `binary_23` model (4 LVs,
    0.887). Height-radius rows cannot be decoded, so they give no direction. Both are applied to
    the same four fruit.

    **This version: section 1 only.** Which four fruit, and their CT, for Ruben to check before
    any decode (one decode took about 6 minutes on the Mac CPU).

    **Inputs.** The fruit table, `derived/volume_decisions.csv`, the CT volumes of the four
    fruit, the saved out-of-fold predictions of the two latent runs. The sealed test boxes are
    not loaded.

    **Outputs.** `derived/figures/17_counterfactual_fruit.png`.

    **Status.** draft, 9 November 2026.
    """)
    return


@app.cell(hide_code=True)
def _():
    cfg = load_config()
    return (cfg,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1. The four fruit

    Rule (`pearlatent.counterfactual.draw_example_fruit`, seed 369): among the 315 open fruit,
    2 with browning 3 and cavity 0, then 2 with defective grade 0, every fruit from its own box.
    A fruit Ruben rejects on its CT is skipped and the next one of the same shuffled list takes
    its place (rejected: 2526_I09, browning not visible). These are examples only.
    Beside each fruit: its out-of-fold prediction in the two latent models whose direction will
    be used (average of the 20 repeats; a cavity-free prediction exists for cavity-free fruit only).
    """)
    return


@app.cell(hide_code=True)
def _(cfg):
    _open_2526 = open_fruit(cfg).filter(pl.col("season") == "2526")
    _kept = drop_faulty_volumes(
        _open_2526, cfg.path("derived_dir") / "volume_decisions.csv"
    )
    print(f"open 2526 fruit {_open_2526.height}, after faulty volumes {_kept.height}")
    # Rejected by Ruben on 9 November 2026 after seeing the CT: 2526_I09 (grade 3, but no
    # visible browning on the sections; the neck end also looked cut).
    REJECTED = ("2526_I09",)
    _drawn = draw_example_fruit(_kept, seed=DRAW_SEED, exclude=REJECTED)

    _results = cfg.path("results_dir")
    _general = pl.read_parquet(
        _results / PLSDA_BINARY23_RUN_DIRS["hugo_vae"] / "predictions.parquet"
    ).select("fruit_id", prediction_binary_23="prediction")
    _cavity_free_run = latest_run_dir(
        _results, "plsda_browning_2plus_cavity_free_hugo_vae"
    )
    print(f"cavity-free run: {_cavity_free_run.name}")
    _cavity_free = pl.read_parquet(_cavity_free_run / "predictions.parquet").select(
        "fruit_id", prediction_cavity_free="prediction"
    )
    examples = _drawn.join(_general, on="fruit_id", how="left", validate="1:1").join(
        _cavity_free, on="fruit_id", how="left", validate="1:1"
    )
    examples.select(
        "group",
        "fruit_id",
        "box",
        "browning",
        "cavity",
        "position",
        pl.col("prediction_binary_23").round(3),
        pl.col("prediction_cavity_free").round(3),
        "stalk_length",
        "body_voxels",
    )
    return REJECTED, examples


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2. Their CT: is anything wrong with these fruit?

    Per fruit (one row): the two long sections through the centre of the body (they contain the
    stem-calyx axis; calyx end at the bottom) and the cross-section at the body centre. Min-max
    scaled as the encoder sees it, grey scale 0 to 1. Look for: a fruit that does not show its
    grade (browning 3 should be visible as darker tissue around the core), a cut or a stray
    object that the faulty-volume check missed.
    """)
    return


@app.cell(hide_code=True)
def _(cfg, examples):
    _cells = np.load(cfg.path("derived_dir") / ENCODING_RUN_DIR / "masks.npy", mmap_mode="r")
    _titles = ["long section, first axis fixed", "long section, second axis fixed", "cross-section"]
    fruit_figure, _axes = plt.subplots(
        examples.height, 3, figsize=(9, 3.1 * examples.height), squeeze=False
    )
    for _i, _row in enumerate(examples.iter_rows(named=True)):
        _path = cfg.path(f"ct_dir_{_row['season']}") / f"{_row['file_stem']}.nii"
        _scaled = scale_minmax(np.asarray(nib.load(_path).dataobj))
        _body = body_voxel_mask(_scaled, np.asarray(_cells[_row["position"]]))
        _planes = central_planes(_scaled, body_centre(_body))
        for _j in range(3):
            _ax = _axes[_i, _j]
            # Long sections: the stem-calyx axis (third array axis) runs upwards.
            _image = _planes[_j].T if _j < 2 else _planes[_j]
            _ax.imshow(_image, cmap="gray", vmin=0, vmax=1, origin="lower")
            _ax.set_xticks([])
            _ax.set_yticks([])
            if _i == 0:
                _ax.set_title(_titles[_j], fontsize=9)
        _axes[_i, 0].set_ylabel(
            f"{_row['fruit_id']}\n{_row['group']} (browning {_row['browning']},"
            f" cavity {_row['cavity']})",
            fontsize=9,
        )
    fruit_figure.tight_layout()
    _out = cfg.path("derived_dir") / "figures" / "17_counterfactual_fruit.png"
    fruit_figure.savefig(_out, dpi=120, bbox_inches="tight")
    print(f"saved {_out}")
    fruit_figure
    return


if __name__ == "__main__":
    app.run()
