import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title="06 Step 0 reconstruction")

with app.setup:
    import altair as alt
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl

    from pearlatent.config import load_config
    from pearlatent.exclusions import read_volume_decisions
    from pearlatent.labels import load_labels
    from pearlatent.step0 import (
        MAX_RADIUS,
        N_HEIGHT_BINS,
        N_RADIUS_BINS,
        contrast_retention,
        plane_measures,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 06 Step 0: is browning kept in the reconstruction?

    **Question.** Charter question 1: does Hugo's VAE reconstruct the browned core, or does the
    reconstruction look like a healthy core? Nothing is fitted. The reconstruction is
    `decode(mu)` (never `vae(x)`, which samples).

    **Inputs.** Encoding run `20261013T171306Z_hugo_vae`: `reconstruction.parquet` (one row per
    decoded fruit: errors over the whole volume, the body and the core),
    `planes.npy` (CT and reconstruction in the three planes through the body centre),
    `masks.npy` (32³ body cells), `manifest.parquet`; both label files; the fruit descriptors
    `derived/descriptors/20261023T151348Z_descriptors/`; `derived/volume_decisions.csv` when
    it exists (faulty volumes, `docs/outlier_rule.md`).

    **Fruit.** 478 decoded fruit: the 346 open 2526 fruit, Hugo's 66 validation fruit and a
    sample of 66 of his 528 training fruit (seed 369). The training sample is a separate group
    and is never pooled with the others (charter rule 2): it shows whether the VAE reconstructs
    fruit it was trained on better than unseen fruit.

    **Outputs.** Figures only; saved to `derived/figures/` with the save buttons (outside git
    until Hugo agrees that CT images may be shown publicly).

    **Definitions** (fixed on 23 October 2026 before the numbers were computed): see the
    docstring of `pearlatent.step0` and `pearlatent.reconstruction`.

    **Status.** draft
    """)
    return


@app.cell
def _():
    cfg = load_config()
    run_dir = (
        cfg.path("derived_dir") / "latents" / "hugo_vae" / "20261013T171306Z_hugo_vae"
    )
    descriptor_dir = (
        cfg.path("derived_dir") / "descriptors" / "20261023T151348Z_descriptors"
    )
    return cfg, descriptor_dir, run_dir


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1. The decoded fruit, their labels and groups

    `position` is the place of a fruit in `latents.npy` and `masks.npy`; `recon_row` its place
    in `reconstruction.parquet` and `planes.npy`. The cause of the defect is attached
    (charter rule 5: results on `defective` are split by cause).
    """)
    return


@app.cell
def _(cfg, descriptor_dir, run_dir):
    manifest = pl.read_parquet(run_dir / "manifest.parquet").with_row_index("position")
    reconstruction = pl.read_parquet(run_dir / "reconstruction.parquet")
    labels = load_labels(cfg).select("fruit_id", "browning", "cavity")
    descriptors = pl.read_parquet(descriptor_dir / "descriptors.parquet")

    decoded = (
        reconstruction.join(
            manifest.select(
                "fruit_id",
                "position",
                "recon_row",
                "season",
                "box",
                "storage",
                "hugo_split",
            ),
            on="fruit_id",
            validate="1:1",
        )
        .join(labels, on="fruit_id", how="left", validate="1:1")
        .join(descriptors, on="fruit_id", how="left", validate="1:1")
        .with_columns(defective=pl.max_horizontal("browning", "cavity"))
        .with_columns(
            group=pl.when(pl.col("season") == "2526")
            .then(pl.lit("2526"))
            .when(pl.col("hugo_split") == "train")
            .then(pl.lit("2425 training sample"))
            .otherwise(pl.lit("2425 validation")),
            cause=pl.when(pl.col("defective") == 0)
            .then(pl.lit("none"))
            .when(pl.col("cavity") == 0)
            .then(pl.lit("browning only"))
            .when(pl.col("browning") == 0)
            .then(pl.lit("cavity only"))
            .otherwise(pl.lit("both")),
        )
    )
    print(f"decoded fruit: {reconstruction.height}; after the joins: {decoded.height}")
    print(f"without a grade: {decoded['browning'].null_count()}")
    decoded.group_by("group").len().sort("group")
    return (decoded,)


@app.cell
def _():
    # Faulty volumes (docs/outlier_rule.md) are dropped once Ruben's decisions exist.
    drop_faulty = mo.ui.checkbox(
        value=True, label="Drop faulty volumes (derived/volume_decisions.csv)"
    )
    drop_faulty
    return (drop_faulty,)


@app.cell
def _(cfg, decoded, drop_faulty):
    _file = cfg.path("derived_dir") / "volume_decisions.csv"
    if drop_faulty.value and _file.exists():
        _faulty = read_volume_decisions(_file).filter(pl.col("decision") != "keep")
        fruit = decoded.join(_faulty.select("fruit_id"), on="fruit_id", how="anti")
        _note = f"{decoded.height - fruit.height} faulty volumes dropped"
    else:
        fruit = decoded
        _note = "no faulty volume dropped" + (
            "" if _file.exists() else " (no decisions file yet)"
        )
    print(f"{_note}; {fruit.height} fruit remain")
    return (fruit,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2. The table of reconstruction errors

    Median and quartiles (25 % and 75 %) per group. Mean absolute error (MAE), structural
    similarity (SSIM) and peak signal-to-noise ratio (PSNR) over the whole volume are comparable
    with the preprint (section 3.1.1: MAE 0.0022, SSIM 0.9915, PSNR 36.9; Table D.1: MAE
    0.0025), but only roughly: Hugo's numbers were computed on augmented volumes with `vae(x)`
    and PSNR averaged per batch of 4 (HANDOFF, "Weights, received 13 October"). The whole
    volume is mostly background, which is easy to reconstruct, so MAE over the body and over the
    core say more about the fruit.
    """)
    return


@app.cell
def _(fruit):
    def quartiles(column: str) -> pl.Expr:
        """'median (25 % to 75 %)' of a column, as text, 4 significant digits."""
        return pl.format(
            "{} ({} to {})",
            pl.col(column).median().round_sig_figs(4),
            pl.col(column).quantile(0.25).round_sig_figs(4),
            pl.col(column).quantile(0.75).round_sig_figs(4),
        ).alias(column)

    _metrics = ["mae_volume", "ssim_volume", "psnr_volume", "mae_body", "mae_core"]

    def error_table(df: pl.DataFrame, by: list[str]) -> pl.DataFrame:
        """One row per combination of `by`, with n and the quartiles of every metric."""
        return (
            df.group_by(by)
            .agg(pl.len().alias("n"), *[quartiles(m) for m in _metrics])
            .sort(by)
        )

    mo.vstack(
        [
            mo.md("**By group and storage**"),
            error_table(fruit, ["group", "storage"]),
            mo.md("**2526 by defective grade and cause**"),
            error_table(
                fruit.filter(pl.col("group") == "2526"), ["defective", "cause"]
            ),
            mo.md("**2425 by group, storage and defective grade**"),
            error_table(
                fruit.filter(pl.col("season") == "2425"),
                ["group", "storage", "defective"],
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 3. What else goes with the error? (confounds)

    The 2526 bodies are shorter in the grid than the 2425 bodies (median stalk 16 against 3
    voxels, `pearlatent.descriptors`), so a 2526 fruit is drawn with fewer voxels. If the error
    depends on size, stalk or intensity, a difference between groups or grades can come from
    those and not from the defect. Spearman rank correlations within 2526, all grades together
    and within grade 0 (so the defect cannot drive them).
    """)
    return


@app.cell
def _(fruit):
    _descriptors = [
        "body_voxels",
        "stalk_length",
        "diameter_to_length",
        "tissue_intensity_median",
        "cavity_fraction",
    ]
    _errors = ["mae_body", "mae_core"]

    def spearman_table(df: pl.DataFrame) -> pl.DataFrame:
        """Spearman correlation of every error with every descriptor, one row per descriptor."""
        return pl.DataFrame(
            {
                "descriptor": _descriptors,
                **{
                    f"rho_{e}": [
                        df.select(pl.corr(e, d, method="spearman")).item()
                        for d in _descriptors
                    ]
                    for e in _errors
                },
            }
        ).with_columns(pl.selectors.starts_with("rho_").round(2))

    _in_2526 = fruit.filter(pl.col("group") == "2526")
    mo.vstack(
        [
            mo.md(f"**2526, all grades ({_in_2526.height} fruit)**"),
            spearman_table(_in_2526),
            mo.md(
                f"**2526, grade 0 only ({_in_2526.filter(pl.col('defective') == 0).height}"
                " fruit)**"
            ),
            spearman_table(_in_2526.filter(pl.col("defective") == 0)),
        ]
    )
    return


@app.cell
def _(fruit):
    # Core error per fruit, by grade and cause; every point is one fruit.
    _points = (
        alt.Chart(fruit.filter(pl.col("group") == "2526"))
        .mark_circle(size=30, opacity=0.6)
        .encode(
            x=alt.X("defective:O", title="Defective grade"),
            xOffset=alt.XOffset("jitter:Q"),
            y=alt.Y("mae_core:Q", title="MAE in the core"),
            color=alt.Color("cause:N"),
            tooltip=["fruit_id", "browning", "cavity", "mae_core", "stalk_length"],
        )
        .transform_calculate(jitter="random()")
    )
    _points.properties(width=500, height=300, title="2526: core error by grade")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 4. Contrast retention: is the darkening of the core kept?

    Per fruit: the mean CT value and the mean reconstruction value over the core voxels of the
    two long sections (`pearlatent.step0.plane_measures`). Per grade: the difference from the
    grade 0 fruit (cause "none") in the CT and in the reconstruction, and their ratio.
    A ratio of 1 means the reconstruction keeps the full difference; 0 means the reconstructed
    core looks like a healthy core.

    Groups, 2526 only: browning only (cavity 0), cavity only (browning 0), and any cause
    (defective grade). Interval: 95 % percentile bootstrap over boxes (the 12 open boxes drawn
    with replacement, 2,000 times, seed 369), as for every interval in this thesis (charter
    rule 4).
    """)
    return


@app.cell
def _(fruit, run_dir):
    _planes = np.load(run_dir / "planes.npy", mmap_mode="r")
    _cells = np.load(run_dir / "masks.npy", mmap_mode="r")
    _numbers = []
    _sums = []
    for _row in fruit.iter_rows(named=True):
        _n, _s = plane_measures(
            np.asarray(_planes[_row["recon_row"]]),
            np.asarray(_cells[_row["position"]]),
            _row,
        )
        _numbers.append({"fruit_id": _row["fruit_id"], **_n})
        _sums.append(_s)
    measured = fruit.join(pl.DataFrame(_numbers), on="fruit_id", validate="1:1")
    # Height-radius sums per fruit, in the row order of `measured`.
    map_sums = np.stack(_sums)
    print(f"measured {measured.height} fruit; map sums {map_sums.shape}")
    print(
        f"fruit without core voxels on the planes: {measured['core_ct'].is_nan().sum()}"
    )
    return map_sums, measured


@app.cell
def _(measured):
    # The comparison groups of 2526. Grade 0 is the reference in every row.
    RETENTION_GROUPS = {
        "browning only": (pl.col("cavity") == 0, "browning"),
        "cavity only": (pl.col("browning") == 0, "cavity"),
        "any cause": (pl.lit(True), "defective"),
    }
    BOOTSTRAP_RESAMPLES = 2000
    BOOTSTRAP_SEED = 369

    in_2526 = measured.filter(pl.col("group") == "2526")
    _boxes = in_2526["box"].unique().sort().to_list()
    _box_index = np.array([_boxes.index(b) for b in in_2526["box"].to_list()])
    _rng = np.random.default_rng(BOOTSTRAP_SEED)
    # One draw = 12 boxes drawn with replacement. `_box_weights[k, b]` says how often box b
    # was drawn in draw k; a box drawn twice counts its fruit twice. The same draws serve
    # every row, so the rows can be compared.
    _box_weights = np.stack(
        [
            np.bincount(
                _rng.integers(0, len(_boxes), size=len(_boxes)), minlength=len(_boxes)
            )
            for _ in range(BOOTSTRAP_RESAMPLES)
        ]
    )

    def per_box_sums(selected: np.ndarray) -> np.ndarray:
        """Per box (columns): number of selected fruit, sum of core CT, sum of core
        reconstruction (rows). `selected` is 1 for a fruit in the group, 0 otherwise."""
        return np.stack(
            [
                np.bincount(_box_index, weights=selected * w, minlength=len(_boxes))
                for w in (
                    np.ones(in_2526.height),
                    in_2526["core_ct"].to_numpy(),
                    in_2526["core_reconstruction"].to_numpy(),
                )
            ]
        )

    _core_ct = in_2526["core_ct"].to_numpy()
    _core_rec = in_2526["core_reconstruction"].to_numpy()
    _reference = (in_2526["defective"] == 0).to_numpy()
    _reference_sums = per_box_sums(_reference.astype(float))
    _rows = []
    for _name, (_condition, _column) in RETENTION_GROUPS.items():
        for _g in (1, 2, 3):
            _target = (
                in_2526.select(
                    _condition & (pl.col(_column) == _g) & (pl.col("defective") > 0)
                )
                .to_series()
                .to_numpy()
            )
            if _target.sum() == 0:
                continue
            # Grade label per fruit for contrast_retention: g for the group, 0 for the
            # reference, -1 for all other fruit (not used).
            _labels = np.where(_target, _g, np.where(_reference, 0, -1))
            _point = contrast_retention(_core_ct, _core_rec, _labels, _g)
            # Bootstrap: group means from the weighted per-box sums of every draw.
            _t = _box_weights @ per_box_sums(_target.astype(float)).T  # (draws, 3)
            _r = _box_weights @ _reference_sums.T
            _usable = (_t[:, 0] > 0) & (_r[:, 0] > 0)
            _t, _r = _t[_usable], _r[_usable]
            _ct_difference = _t[:, 1] / _t[:, 0] - _r[:, 1] / _r[:, 0]
            _rec_difference = _t[:, 2] / _t[:, 0] - _r[:, 2] / _r[:, 0]
            _boot = _rec_difference / _ct_difference
            _rows.append(
                {
                    "group": _name,
                    "grade": _g,
                    "n_fruit": int(_target.sum()),
                    "n_boxes": in_2526.filter(pl.Series(_target))["box"].n_unique(),
                    **{k: round(v, 4) for k, v in _point.items()},
                    "retention_2.5%": round(float(np.percentile(_boot, 2.5)), 3),
                    "retention_97.5%": round(float(np.percentile(_boot, 97.5)), 3),
                    "draws_used": int(_usable.sum()),
                }
            )
    retention_table = pl.DataFrame(_rows)
    print(
        f"reference: {int(_reference.sum())} grade 0 fruit;"
        f" mean core CT {_core_ct[_reference].mean():.4f},"
        f" mean core reconstruction {_core_rec[_reference].mean():.4f}"
    )
    retention_table
    return in_2526, retention_table


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 5. Height-radius maps per grade

    Every cell of a map is a group mean over all body voxels of the two long sections that fall
    in that height and radius bin (height 0 = calyx end of the body, 1 = stalk end; radius in
    units of the widest radius). Rows: mean CT, mean reconstruction, mean signed residual
    (reconstruction minus CT; red = reconstruction brighter than the CT). If the decoder
    "repairs" browning, the residual is red near the axis (left edge) and grows with grade.
    Rows 2 and 3 show the difference from the grade 0 fruit, in the CT and in the
    reconstruction: where the two look alike, the reconstruction keeps the darkening (contrast
    retention, section 4, drawn in space). Columns: grade 0 (defective 0) and browning-only
    grades 1 to 3, 2526 fruit; the last column cavity 2 or 3 with any browning (cavity only has
    3 fruit in the open boxes).
    """)
    return


@app.cell
def _(map_sums, measured):
    # Columns of the figure: (title, condition on the rows of `measured`). Cavity only has 3
    # fruit in the open 2526 boxes, too few for a map; the last column shows cavity 2 or 3
    # with any browning instead.
    MAP_COLUMNS = [
        ("grade 0", (pl.col("defective") == 0)),
        ("browning 1 only", (pl.col("browning") == 1) & (pl.col("cavity") == 0)),
        ("browning 2 only", (pl.col("browning") == 2) & (pl.col("cavity") == 0)),
        ("browning 3 only", (pl.col("browning") == 3) & (pl.col("cavity") == 0)),
        ("cavity 2-3, any browning", (pl.col("cavity") >= 2)),
    ]
    # Bins with fewer voxels than this over all fruit of a group are left empty (too noisy).
    MIN_VOXELS_PER_BIN = 50
    _is_2526 = (measured["group"] == "2526").to_numpy()

    def group_map(condition: pl.Expr) -> tuple[np.ndarray, int]:
        """Mean CT, reconstruction and residual per bin (3, height, radius), and the fruit count."""
        _mask = measured.select(condition).to_series().to_numpy() & _is_2526
        _total = map_sums[_mask].sum(axis=0)
        _count = _total[0]
        with np.errstate(invalid="ignore", divide="ignore"):
            _means = np.where(_count >= MIN_VOXELS_PER_BIN, _total[1:] / _count, np.nan)
        return _means, int(_mask.sum())

    _healthy, _ = group_map(MAP_COLUMNS[0][1])
    # Rows: (name, picture from the group means, colour map, colour limits).
    _rows = [
        ("CT", lambda m: m[0], "gray", (0.5, 1.0)),
        ("CT minus\ngrade 0 CT", lambda m: m[0] - _healthy[0], "RdBu_r", (-0.06, 0.06)),
        (
            "reconstruction minus\ngrade 0 reconstruction",
            lambda m: m[1] - _healthy[1],
            "RdBu_r",
            (-0.06, 0.06),
        ),
        (
            "residual\n(reconstruction minus CT)",
            lambda m: m[2],
            "RdBu_r",
            (-0.06, 0.06),
        ),
    ]
    _fig, _axes = plt.subplots(
        len(_rows),
        len(MAP_COLUMNS),
        figsize=(2.9 * len(MAP_COLUMNS) + 1.2, 2.6 * len(_rows)),
        squeeze=False,
        constrained_layout=True,
    )
    _extent = [0, MAX_RADIUS, 0, 1]  # radius on the horizontal axis, height vertical
    _images = {}
    for _c, (_title, _condition) in enumerate(MAP_COLUMNS):
        _means, _n = group_map(_condition)
        for _r, (_name, _picture, _cmap, _limits) in enumerate(_rows):
            _ax = _axes[_r, _c]
            _images[_r] = _ax.imshow(
                _picture(_means),
                origin="lower",
                extent=_extent,
                aspect="auto",
                cmap=_cmap,
                vmin=_limits[0],
                vmax=_limits[1],
            )
            if _r == 0:
                _ax.set_title(f"{_title}\n(n = {_n})", fontsize=10)
            if _c == 0:
                _ax.set_ylabel(f"{_name}\n\nheight (calyx 0, stalk 1)", fontsize=9)
            if _r == len(_rows) - 1:
                _ax.set_xlabel("radius / widest radius")
    # One colour bar per row, at the right.
    for _r, (_name, *_rest) in enumerate(_rows):
        _fig.colorbar(_images[_r], ax=_axes[_r, :].tolist(), shrink=0.9)
    _fig.suptitle(
        "2526 fruit, two long sections per fruit; red = brighter, blue = darker"
    )
    hr_figure = _fig
    hr_figure
    return (hr_figure,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 6. One example fruit per grade

    The fruit whose core MAE is closest to the median core MAE of its group (2526), so the
    example is typical and not chosen by eye. Long section 1 (plane 0) and the cross-section:
    CT, reconstruction, and signed residual (red = reconstruction brighter).
    """)
    return


@app.cell
def _(in_2526, run_dir):
    EXAMPLE_GROUPS = [
        ("grade 0", (pl.col("defective") == 0)),
        ("browning 1 only", (pl.col("browning") == 1) & (pl.col("cavity") == 0)),
        ("browning 2 only", (pl.col("browning") == 2) & (pl.col("cavity") == 0)),
        ("browning 3 only", (pl.col("browning") == 3) & (pl.col("cavity") == 0)),
        ("cavity 3 only", (pl.col("cavity") == 3) & (pl.col("browning") == 0)),
    ]
    _rows = []
    for _title, _condition in EXAMPLE_GROUPS:
        _group = in_2526.filter(_condition)
        if _group.height == 0:
            continue
        _median = _group["mae_core"].median()
        _pick = _group.with_columns(
            _distance=(pl.col("mae_core") - _median).abs()
        ).sort("_distance", "fruit_id")[0]
        _rows.append(
            {
                "example": _title,
                "n_in_group": _group.height,
                **_pick.drop("_distance").row(0, named=True),
            }
        )
    examples = pl.DataFrame(_rows).select(
        "example",
        "n_in_group",
        "fruit_id",
        "browning",
        "cavity",
        "mae_core",
        "recon_row",
    )

    _planes = np.load(run_dir / "planes.npy", mmap_mode="r")
    _fig, _axes = plt.subplots(
        len(_rows), 6, figsize=(15, 2.7 * len(_rows)), squeeze=False
    )
    for _k, _row in enumerate(examples.iter_rows(named=True)):
        _p = np.asarray(_planes[_row["recon_row"]])
        _pictures = [
            (_p[0, 0].T, "CT, long", "gray", (0, 1)),
            (_p[1, 0].T, "reconstruction, long", "gray", (0, 1)),
            (_p[1, 0].T - _p[0, 0].T, "residual, long", "RdBu_r", (-0.3, 0.3)),
            (_p[0, 2], "CT, cross", "gray", (0, 1)),
            (_p[1, 2], "reconstruction, cross", "gray", (0, 1)),
            (_p[1, 2] - _p[0, 2], "residual, cross", "RdBu_r", (-0.3, 0.3)),
        ]
        for _j, (_picture, _name, _cmap, _lim) in enumerate(_pictures):
            _ax = _axes[_k, _j]
            _ax.imshow(_picture, origin="lower", cmap=_cmap, vmin=_lim[0], vmax=_lim[1])
            _ax.set_xticks([])
            _ax.set_yticks([])
            if _k == 0:
                _ax.set_title(_name, fontsize=9)
        _axes[_k, 0].set_ylabel(f"{_row['example']}\n{_row['fruit_id']}", fontsize=9)
    _fig.tight_layout()
    example_figure = _fig
    mo.vstack([examples, example_figure])
    return example_figure, examples


@app.cell
def _():
    save_step0_figures = mo.ui.run_button(label="Save the figures to derived/figures/")
    save_step0_figures
    return (save_step0_figures,)


@app.cell
def _(cfg, example_figure, hr_figure, save_step0_figures):
    # Saved under derived/ (outside git): CT images stay off GitHub until Hugo agrees.
    mo.stop(mo.running_in_notebook() and not save_step0_figures.value)
    _folder = cfg.path("derived_dir") / "figures"
    _folder.mkdir(parents=True, exist_ok=True)
    hr_figure.savefig(
        _folder / "06_height_radius_maps.png", dpi=130, bbox_inches="tight"
    )
    example_figure.savefig(_folder / "06_examples.png", dpi=130, bbox_inches="tight")
    print(f"saved 06_height_radius_maps.png and 06_examples.png to {_folder}")
    return


if __name__ == "__main__":
    app.run()
