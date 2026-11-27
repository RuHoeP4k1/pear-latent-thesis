import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title="23 Stretch and real size")

with app.setup(hide_code=True):
    import json

    import marimo as mo
    import matplotlib.pyplot as plt
    import nibabel as nib
    import numpy as np
    import polars as pl

    from pearlatent.config import load_config
    from pearlatent.epo import fit_stretch_direction
    from pearlatent.epo_stretch_run import model_fruit
    from pearlatent.feature_sets import load_features
    from pearlatent.runs import (
        EPO_STRETCH_RUN_DIRS,
        PLSDA_BINARY23_RUN_DIRS,
        STRETCH_CHECK_RUN_DIR,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 23 Stretch and real size

    **Question.** The latent score of healthy fruit follows "size" (check C). Is that the
    rescaling of every fruit to 128 voxels in length (an artefact), or the real size of the
    fruit (biology)? And what does removing the artefact do to the latent rows?

    **Background.** Hugo rescaled every fruit, stalk included, to 128 voxels in length, to fill
    the 128³ input of his encoder. A long fruit, or a fruit with a long stalk, is shrunk more.
    On 27 November 2026 Hugo gave the 2526 volumes before that rescaling. From them:
    stretch factor s = 128 / original length (stalk included), and real body volume in cm³
    (`pearlatent.real_size`, `docs/methods_plsda.md` sections 20 to 23).

    **Inputs.** `ct_orig_dir_2526` and `ct_dir_2526` (two fruit, figure 1); the real-size run
    (`REAL_SIZE_RUN_DIR`); the stretch check (`STRETCH_CHECK_RUN_DIR`); the saved PLS-DA runs
    without EPO (`PLSDA_BINARY23_RUN_DIRS`) and with EPO on the stretch factor
    (`EPO_STRETCH_RUN_DIRS`); the latent height-radius features (figure 5).

    **Outputs.** Figures in `derived/figures/23_*.png`. No model is fitted here except one
    illustration of the EPO direction on all healthy fruit (figure 5).

    **Status.** draft. Open: the number of latent variables (LVs) of the final latent rows
    with EPO (section 7, Ruben).

    All numbers: the 315 open 2526 fruit (12 boxes, faulty volumes removed); healthy = defective
    grade 0 (145 fruit). Test boxes F, J, M are not used.
    """)
    return


@app.cell(hide_code=True)
def _():
    cfg = load_config()
    figure_dir = cfg.path("derived_dir") / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    return cfg, figure_dir


@app.cell(hide_code=True)
def _(cfg):
    # The model fruit with stretch factor, real body volume and stalk length (mm).
    fruit = model_fruit(cfg)
    healthy = fruit.filter(pl.col("defective") == 0)
    print(f"model fruit {fruit.height}, healthy {healthy.height}")
    return fruit, healthy


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1. What the rescaling does

    The least and the most shrunk fruit of the open boxes. Top: the original volumes, on one
    millimetre scale (voxel 0.6267 mm). Bottom: Hugo's 128³ volumes. Both end up 128 voxels
    long, so in the 128³ grid the long fruit looks as large as the short one.
    """)
    return


@app.cell(hide_code=True)
def _(cfg, figure_dir, fruit):
    _sorted = fruit.sort("stretch_factor")
    _pair = [_sorted.row(-1, named=True), _sorted.row(0, named=True)]
    _voxel = 0.6267
    rescaling_figure, _axes = plt.subplots(2, 2, figsize=(8, 8))
    for _c, _f in enumerate(_pair):
        _stem = _f["fruit_id"].split("_", 1)[1]
        _orig = np.asarray(nib.load(cfg.path("ct_orig_dir_2526") / f"{_stem}.nii").dataobj)
        _resc = np.asarray(nib.load(cfg.path("ct_dir_2526") / f"{_stem}.nii").dataobj)
        _o = _orig[_orig.shape[0] // 2].T  # slice through the middle; stem–calyx axis upwards
        _axes[0, _c].imshow(
            _o,
            origin="lower",
            cmap="gray",
            extent=(0, _o.shape[1] * _voxel, 0, _o.shape[0] * _voxel),
        )
        _axes[0, _c].set_xlim(0, 120)
        _axes[0, _c].set_ylim(0, 180)
        _axes[0, _c].set_title(
            f"{_f['fruit_id']} original: {_f['fruit_length_mm']:.0f} mm, "
            f"{_f['body_volume_cm3']:.0f} cm³"
        )
        _axes[0, _c].set_xlabel("mm")
        _filled = np.flatnonzero((_resc > 0).any(axis=(1, 2)))
        _middle = (_filled[0] + _filled[-1]) // 2
        _axes[1, _c].imshow(_resc[_middle].T, origin="lower", cmap="gray")
        _axes[1, _c].set_title(f"128³ volume: s = {_f['stretch_factor']:.2f}")
        _axes[1, _c].set_xlabel("voxel")
    rescaling_figure.tight_layout()
    rescaling_figure.savefig(figure_dir / "23_1_rescaling.png", dpi=120, bbox_inches="tight")
    rescaling_figure
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2. The stalk sets the stretch, not the size of the fruit

    Left: stretch factor against real body volume, each fruit coloured by its stalk length. At
    equal volume, a long stalk (yellow) means a small s (more shrinking). Right: the "size"
    used in check C until now (body voxels in the 128³ grid) against real volume: it hardly
    follows real volume, because it is mostly the stretch.
    """)
    return


@app.cell(hide_code=True)
def _(figure_dir, fruit):
    _corr = lambda a, b: float(np.corrcoef(fruit[a], fruit[b])[0, 1])  # noqa: E731
    stalk_figure, _axes = plt.subplots(1, 2, figsize=(11, 4.2))
    _shown = _axes[0].scatter(
        fruit["body_volume_cm3"],
        fruit["stretch_factor"],
        c=fruit["stalk_length_mm"],
        cmap="viridis",
        s=14,
    )
    stalk_figure.colorbar(_shown, ax=_axes[0], label="stalk length (mm)")
    _axes[0].set_xlabel("real body volume (cm³)")
    _axes[0].set_ylabel("stretch factor s = 128 / original length")
    _axes[0].set_title(
        f"r(s, volume) = {_corr('stretch_factor', 'body_volume_cm3'):+.2f}, "
        f"r(s, stalk) = {_corr('stretch_factor', 'stalk_length_mm'):+.2f}"
    )
    _shown = _axes[1].scatter(
        fruit["body_volume_cm3"],
        fruit["body_voxels"],
        c=fruit["stretch_factor"],
        cmap="magma",
        s=14,
    )
    stalk_figure.colorbar(_shown, ax=_axes[1], label="stretch factor s")
    _axes[1].set_xlabel("real body volume (cm³)")
    _axes[1].set_ylabel("body voxels in the 128³ grid")
    _axes[1].set_title(
        f"r(128³ voxels, volume) = {_corr('body_voxels', 'body_volume_cm3'):+.2f}, "
        f"r(128³ voxels, s) = {_corr('body_voxels', 'stretch_factor'):+.2f}"
    )
    stalk_figure.suptitle(f"{fruit.height} open 2526 fruit (Pearson r)")
    stalk_figure.tight_layout()
    stalk_figure.savefig(figure_dir / "23_2_stalk_and_stretch.png", dpi=120)
    stalk_figure
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 3. Larger fruit are more often defective, and the models see more than size

    Left: real body volume per defective grade. Right: AUC of each model's out-of-fold score
    for `binary_23`, inside each third of real volume (cut-offs fixed before looking), with
    the AUC of size alone as control. Size alone separates 0.79 over all fruit; inside a third
    it drops, the models stay high.
    """)
    return


@app.cell(hide_code=True)
def _(cfg, figure_dir, fruit):
    _check_dir = cfg.path("results_dir") / STRETCH_CHECK_RUN_DIR
    size_auc = pl.read_parquet(_check_dir / "size_auc.parquet")
    thirds = pl.read_parquet(_check_dir / "auc_by_size_third.parquet")
    _shown_rows = {
        "size alone": "size alone",
        "hugo_vae_hr_sd": "latent height-radius mean + SD",
        "med3d": "Med3D",
        "ct_32_hr_sd": "CT 32³ height-radius mean + SD",
    }
    size_figure, _axes = plt.subplots(1, 2, figsize=(11, 4.2))
    _grades = [0, 1, 2, 3]
    _axes[0].boxplot(
        [fruit.filter(pl.col("defective") == g)["body_volume_cm3"].to_numpy() for g in _grades],
        tick_labels=[
            f"{g}\n(n={fruit.filter(pl.col('defective') == g).height})" for g in _grades
        ],
    )
    _axes[0].set_xlabel("defective grade")
    _axes[0].set_ylabel("real body volume (cm³)")
    _auc = size_auc.filter(pl.col("measure") == "body_volume_cm3").row(0, named=True)
    _axes[0].set_title(
        f"size alone for binary_23: AUC {_auc['auc']:.2f} ({_auc['low']:.2f} to {_auc['high']:.2f})"
    )
    for _k, (_row, _label) in enumerate(_shown_rows.items()):
        _part = thirds.filter(pl.col("row") == _row).sort("size_group")
        _x = np.arange(3) + (_k - 1.5) * 0.2
        _axes[1].errorbar(
            _x,
            _part["auc"],
            yerr=[_part["auc"] - _part["low"], _part["high"] - _part["auc"]],
            fmt="o",
            capsize=3,
            label=_label,
        )
    _limits = thirds.filter(pl.col("row") == "size alone").sort("size_group")
    _axes[1].set_xticks(
        range(3),
        [
            f"{lo:.0f}–{hi:.0f} cm³"
            for lo, hi in zip(_limits["volume_min"], _limits["volume_max"], strict=True)
        ],
    )
    _axes[1].axhline(0.5, color="grey", lw=0.8, ls=":")
    _axes[1].set_ylabel("AUC inside the third (95 % box bootstrap)")
    _axes[1].set_xlabel("third of real body volume")
    _axes[1].legend(fontsize=8, loc="lower right")
    size_figure.tight_layout()
    size_figure.savefig(figure_dir / "23_3_size_and_defect.png", dpi=120)
    size_figure
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 4. Which model scores follow the stretch, which follow real size

    On the healthy fruit, the score (minus its box median) against the stretch factor with real
    volume held equal, and against real volume with the stretch held equal (partial Spearman
    correlations, 95 % box bootstrap). Stretch and volume correlate (−0.52), so a plain
    correlation with one would also pick up the other. Only the latent rows follow the stretch.
    """)
    return


@app.cell(hide_code=True)
def _(cfg, figure_dir):
    _check = pl.read_parquet(
        cfg.path("results_dir") / STRETCH_CHECK_RUN_DIR / "stretch_check.parquet"
    )
    _labels = {
        "hugo_vae": "latent grid",
        "hugo_vae_hr": "latent height-radius",
        "hugo_vae_hr_sd": "latent height-radius mean + SD",
        "ct_32": "CT 32³ grid",
        "ct_32_hr": "CT 32³ height-radius",
        "ct_32_hr_sd": "CT 32³ height-radius mean + SD",
        "tissue_features": "tissue features",
        "med3d": "Med3D",
    }
    _check = _check.filter(pl.col("row").is_in(list(_labels))).with_columns(
        label=pl.col("row").replace_strict(_labels)
    )
    partial_figure, _ax = plt.subplots(figsize=(8, 4.5))
    _y = np.arange(_check.height)
    for _key, _offset, _name in (
        ("partial_stretch_given_volume", -0.15, "stretch, volume held equal"),
        ("partial_volume_given_stretch", 0.15, "real volume, stretch held equal"),
    ):
        _ax.errorbar(
            _check[_key],
            _y + _offset,
            xerr=[_check[_key] - _check[f"{_key}_low"], _check[f"{_key}_high"] - _check[_key]],
            fmt="o",
            capsize=3,
            label=_name,
        )
    _ax.set_yticks(_y, _check["label"])
    _ax.axvline(0, color="grey", lw=0.8)
    _ax.set_xlabel("partial Spearman with the score, 145 healthy fruit")
    _ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
    partial_figure.tight_layout()
    partial_figure.savefig(
        figure_dir / "23_4_partial_per_row.png", dpi=120, bbox_inches="tight"
    )
    partial_figure
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 5. What EPO on the stretch factor removes

    External parameter orthogonalisation (EPO), as done for the latent rows
    (`pearlatent.epo.fit_stretch_direction`), here once on all 145 healthy fruit so the
    direction can be drawn (in the cross-validation it is refitted in every fold on the
    healthy training fruit):

    1. Each of the 320 features (20 heights × 8 rings, mean and SD) is regressed on the
       stretch factor and real volume together. The 320 stretch coefficients form the
       direction p: how the latent changes per unit of stretch, at equal real size.
    2. Each fruit gets one score on it, t = x · p.
    3. t is set equal for every fruit (x* = x − t p). One of 320 dimensions is removed.

    Maps: p, and the defect direction (mean of the non-consumable fruit minus mean of the
    healthy fruit), each split into its mean and SD features. Rows: height, calyx end at the
    bottom; columns: ring, axis on the left, skin on the right.
    """)
    return


@app.cell(hide_code=True)
def _(cfg, fruit):
    # Feature layout (methods section 11): mean of height h, ring r at 8 h + r; SD at 160 + 8 h + r.
    features = load_features(cfg, "hugo_vae_hr_sd", fruit["fruit_id"].to_list()).astype(float)
    _healthy = (fruit["defective"] == 0).to_numpy()
    stretch_direction = fit_stretch_direction(
        features,
        _healthy,
        fruit["stretch_factor"].to_numpy(),
        fruit["body_volume_cm3"].to_numpy(),
    )[:, 0]
    _y = fruit["binary_23"].to_numpy() == 1
    _defect = features[_y].mean(axis=0) - features[_healthy].mean(axis=0)
    defect_direction = _defect / np.linalg.norm(_defect)
    _centred = features[_healthy] - features[_healthy].mean(axis=0)
    epo_numbers = {
        "share_removed": float(
            ((_centred @ stretch_direction) ** 2).sum() / (_centred**2).sum()
        ),
        "cosine_defect": float(abs(stretch_direction @ defect_direction)),
        "share_on_means": float((stretch_direction[:160] ** 2).sum()),
    }
    print(
        f"share of healthy variation along p: {epo_numbers['share_removed']:.1%}; "
        f"share of p on the mean features: {epo_numbers['share_on_means']:.0%}; "
        f"cosine p with the defect direction: {epo_numbers['cosine_defect']:.2f}"
    )
    return defect_direction, epo_numbers, stretch_direction


@app.cell(hide_code=True)
def _(defect_direction, epo_numbers, figure_dir, stretch_direction):
    direction_figure, _axes = plt.subplots(1, 4, figsize=(12, 5))
    _panels = [
        (stretch_direction[:160], "stretch direction p: means"),
        (stretch_direction[160:], "stretch direction p: SDs"),
        (defect_direction[:160], "defect direction: means"),
        (defect_direction[160:], "defect direction: SDs"),
    ]
    _limit = max(np.abs(stretch_direction).max(), np.abs(defect_direction).max())
    for _ax, (_values, _title) in zip(_axes, _panels, strict=True):
        _shown = _ax.imshow(
            _values.reshape(20, 8),
            origin="lower",
            cmap="RdBu_r",
            vmin=-_limit,
            vmax=_limit,
            aspect="auto",
        )
        _ax.set_title(_title, fontsize=9)
        _ax.set_xlabel("ring (0 = axis)")
    _axes[0].set_ylabel("height bin (0 = calyx end)")
    direction_figure.colorbar(_shown, ax=_axes, shrink=0.8, label="weight")
    direction_figure.suptitle(
        f"p holds {epo_numbers['share_removed']:.0%} of the healthy variation, "
        f"{epo_numbers['share_on_means']:.0%} of it on the means; "
        f"cosine with the defect direction {epo_numbers['cosine_defect']:.2f}"
    )
    direction_figure.savefig(figure_dir / "23_5_directions.png", dpi=120, bbox_inches="tight")
    direction_figure
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 6. Before and after EPO, latent height-radius mean + SD

    The healthy fruit: score (minus its box median) against real volume, coloured by the
    stretch factor. Before: at equal volume the colours sort the score (stretch link). After:
    they no longer do, and the rise with real volume stays.
    """)
    return


@app.cell(hide_code=True)
def _(cfg, figure_dir, fruit):
    _results = cfg.path("results_dir")
    _sizes = fruit.select("fruit_id", "stretch_factor", "body_volume_cm3")
    _runs = {
        "without EPO": _results / PLSDA_BINARY23_RUN_DIRS["hugo_vae_hr_sd"],
        "with EPO on the stretch factor": _results / EPO_STRETCH_RUN_DIRS["hugo_vae_hr_sd"],
    }
    _links = json.loads((_runs["with EPO on the stretch factor"] / "scores.json").read_text())[
        "healthy_links"
    ]
    before_after_figure, _axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for _ax, (_name, _dir), _key in zip(
        _axes, _runs.items(), ("without", "with"), strict=True
    ):
        _t = (
            pl.read_parquet(_dir / "predictions.parquet")
            .join(_sizes, on="fruit_id")
            .filter(pl.col("defective") == 0)
            .with_columns(
                score_in_box=pl.col("prediction") - pl.col("prediction").median().over("box")
            )
        )
        _auc = json.loads((_dir / "scores.json").read_text())["auc"]
        _shown = _ax.scatter(
            _t["body_volume_cm3"],
            _t["score_in_box"],
            c=_t["stretch_factor"],
            cmap="viridis",
            s=16,
        )
        _l = _links[_key]
        _ax.set_title(
            f"{_name} (AUC {_auc:.3f})\n"
            f"stretch, volume equal: {_l['stretch_given_volume']:+.2f}; "
            f"stalk, volume equal: {_l['stalk_given_volume']:+.2f}\n"
            f"real volume, stretch equal: {_l['volume_given_stretch']:+.2f}",
            fontsize=9,
        )
        _ax.set_xlabel("real body volume (cm³)")
    _axes[0].set_ylabel("score minus box median (healthy fruit)")
    before_after_figure.colorbar(_shown, ax=_axes, label="stretch factor s")
    before_after_figure.savefig(
        figure_dir / "23_6_before_after_epo.png", dpi=120, bbox_inches="tight"
    )
    before_after_figure
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 7. Number of latent variables for the final latent rows (Ruben)

    The mean RMSECV curve (20 repeats of the 4-fold box cross-validation) of each latent row,
    without and with EPO. The runs with EPO kept the LVs chosen without EPO (grid 4,
    height-radius 3, height-radius mean + SD 3), so that before and after compare the same
    model size. For the final rows the LVs are chosen again, by eye at the elbow, before the
    test boxes are opened.
    """)
    return


@app.cell(hide_code=True)
def _(cfg, figure_dir):
    _results = cfg.path("results_dir")
    _labels = {
        "hugo_vae": "latent grid",
        "hugo_vae_hr": "latent height-radius",
        "hugo_vae_hr_sd": "latent height-radius mean + SD",
    }
    curve_figure, _axes = plt.subplots(1, 3, figsize=(13, 3.8))
    _rows = []
    for _ax, (_row, _label) in zip(_axes, _labels.items(), strict=True):
        _base = (
            pl.read_parquet(_results / PLSDA_BINARY23_RUN_DIRS[_row] / "curves.parquet")
            .filter(pl.col("curve") == "RMSECV, mean PRESS")
            .sort("n_lv")
        )
        _epo = pl.read_parquet(
            _results / EPO_STRETCH_RUN_DIRS[_row] / "curves.parquet"
        ).sort("n_lv")
        _ax.plot(_base["n_lv"], _base["error"], "o-", label="without EPO")
        _ax.plot(_epo["n_lv"], _epo["rmsecv"], "s-", label="with EPO")
        _ax.set_title(_label, fontsize=10)
        _ax.set_xlabel("latent variables")
        _ax.set_xticks(range(1, 16, 2))
        for _h in range(1, 7):
            _rows.append(
                {
                    "row": _label,
                    "LVs": _h,
                    "without EPO": round(_base["error"][_h - 1], 3),
                    "with EPO": round(_epo["rmsecv"][_h - 1], 3),
                }
            )
    _axes[0].set_ylabel("RMSECV (mean PRESS)")
    _axes[0].legend()
    curve_figure.tight_layout()
    curve_figure.savefig(figure_dir / "23_7_rmsecv_curves.png", dpi=120)
    curve_table = pl.DataFrame(_rows)
    mo.vstack([curve_figure, curve_table])
    return


if __name__ == "__main__":
    app.run()
