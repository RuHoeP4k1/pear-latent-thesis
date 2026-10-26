import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title="06 Step 0 reconstruction")

with app.setup:
    import altair as alt
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import plotly.graph_objects as go
    import polars as pl
    from matplotlib.animation import FuncAnimation, PillowWriter

    from pearlatent.config import load_config
    from pearlatent.exclusions import read_volume_decisions
    from pearlatent.labels import load_labels
    from pearlatent.step0 import (
        MAX_RADIUS,
        N_HEIGHT_BINS,
        N_RADIUS_BINS,
        contrast_retention,
        latent_map_sums,
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
    `derived/descriptors/20261023T155219Z_descriptors/`; `derived/volume_decisions.csv` when
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


@app.cell(hide_code=True)
def _():
    cfg = load_config()
    run_dir = (
        cfg.path("derived_dir") / "latents" / "hugo_vae" / "20261013T171306Z_hugo_vae"
    )
    descriptor_dir = (
        cfg.path("derived_dir") / "descriptors" / "20261023T155219Z_descriptors"
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


@app.cell(hide_code=True)
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
    and PSNR averaged per batch of 4 . The whole
    volume is mostly background, which is easy to reconstruct, so MAE over the body and over the
    core say more about the fruit.
    """)
    return


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
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
    return (in_2526,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 5. Height-radius maps per grade

    Every fruit is folded onto one half long section: **height** runs from 0 at the calyx end of
    the body to 1 at the stalk end (stalk removed), **radius** is the distance to the stem-calyx
    axis divided by the fruit's widest radius (0 = on the axis, about 1 = skin at the widest
    point). Both sides of the axis land on the same side. A bin shows the mean over all
    voxels (or latent cells) of all fruit of the column that fall in it. A bin is drawn only
    when at least half of the fruit of the column reach it: bins near the skin that only some
    fruit reach show differences in fruit shape, not in tissue.

    Columns: grade 0 (defective 0) and browning-only grades 1 to 3, 2526 fruit; the last
    column cavity 2 or 3 with any browning (cavity only has 3 fruit in the open boxes).

    **View "CT and reconstruction"** (two long sections per fruit, `planes.npy`). Row 1: mean
    CT. Rows 2 and 3: difference from the grade 0 fruit, in the CT and in the reconstruction
    (blue = darker than healthy); where the two look alike, the reconstruction keeps the
    darkening (contrast retention, section 4, drawn in space). Row 4: residual,
    reconstruction minus CT, on its own tighter scale (red = reconstruction brighter). If the
    decoder "repaired" browning, row 4 would turn red near the axis with rising grade.

    **View "latent"** (all body cells of the 32³ latent `mu`, so the full rotation, not two
    sections). Row 1: CT minus grade 0 CT as in the other view, for reference. Row 2: mean
    latent value. Row 3: latent minus the grade 0 latent. A latent value is not an intensity:
    red means a higher latent value, not a brighter tissue. Each latent cell sees about 31
    voxels around it (about 19 mm), so the latent map is smoother than the CT map.
    """)
    return


@app.cell(hide_code=True)
def _(measured):
    # Columns of both maps: (title, condition on the rows of `measured`). Cavity only has 3
    # fruit in the open 2526 boxes, too few for a map; the last column shows cavity 2 or 3
    # with any browning instead.
    MAP_COLUMNS = [
        ("grade 0", (pl.col("defective") == 0)),
        ("browning 1 only", (pl.col("browning") == 1) & (pl.col("cavity") == 0)),
        ("browning 2 only", (pl.col("browning") == 2) & (pl.col("cavity") == 0)),
        ("browning 3 only", (pl.col("browning") == 3) & (pl.col("cavity") == 0)),
        ("cavity 2-3, any browning", (pl.col("cavity") >= 2)),
    ]
    # A bin is drawn only when at least this share of the fruit of the column reach it.
    MIN_FRUIT_SHARE_PER_BIN = 0.5
    _is_2526 = (measured["group"] == "2526").to_numpy()

    def column_fruit(condition: pl.Expr) -> np.ndarray:
        """Boolean over the rows of `measured`: 2526 fruit that meet `condition`."""
        return measured.select(condition).to_series().to_numpy() & _is_2526

    def group_means(
        counts: np.ndarray, sums: np.ndarray, selected: np.ndarray
    ) -> np.ndarray:
        """Group mean per bin, NaN where fewer than MIN_FRUIT_SHARE_PER_BIN of the fruit
        reach the bin.

        In: per-fruit counts (fruit, height, radius), per-fruit sums (fruit, k, height,
        radius) and the selected fruit. Out: means (k, height, radius).
        """
        _counts = counts[selected]
        _fruit_reaching = (_counts > 0).sum(axis=0)
        _enough = _fruit_reaching >= MIN_FRUIT_SHARE_PER_BIN * selected.sum()
        _total = _counts.sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            _means = sums[selected].sum(axis=0) / _total
        return np.where(_enough, _means, np.nan)

    return MAP_COLUMNS, column_fruit, group_means


@app.cell(hide_code=True)
def _(measured, run_dir):
    # Latent height-radius sums of every measured fruit, in the row order of `measured`.
    # `position` is the row of the fruit in latents.npy and masks.npy.
    _latents = np.load(run_dir / "latents.npy", mmap_mode="r")
    _cells = np.load(run_dir / "masks.npy", mmap_mode="r")
    _counts, _sums = [], []
    for _position in measured["position"].to_list():
        _c, _s = latent_map_sums(
            np.asarray(_latents[_position]), np.asarray(_cells[_position])
        )
        _counts.append(_c)
        _sums.append(_s)
    latent_counts = np.stack(_counts)
    latent_sums = np.stack(_sums)[:, None]  # (fruit, 1, height, radius)
    print(
        f"latent maps of {latent_counts.shape[0]} fruit, bins {latent_counts.shape[1:]}"
    )
    return latent_counts, latent_sums


@app.function(hide_code=True)
def draw_maps(columns, rows, title):
    """One figure: a panel per (row, column), one colour bar per row.

    `columns`: list of (title, n fruit, means). `rows`: list of (name, picture function
    of (means, column index) returning an array or None for an empty panel, colour map,
    colour limits). Radius on the horizontal axis, height vertical.
    """
    _fig, _axes = plt.subplots(
        len(rows),
        len(columns),
        figsize=(2.9 * len(columns) + 1.2, 2.6 * len(rows)),
        squeeze=False,
        constrained_layout=True,
    )
    _extent = [0, MAX_RADIUS, 0, 1]
    _images = {}
    for _c, (_title, _n, _means) in enumerate(columns):
        for _r, (_name, _picture, _cmap, _limits) in enumerate(rows):
            _ax = _axes[_r, _c]
            _array = _picture(_means, _c)
            if _r == 0:
                _ax.set_title(f"{_title}\n(n = {_n})", fontsize=10)
            if _array is None:
                # Grade 0 minus grade 0 is zero by definition: no panel.
                _ax.axis("off")
                _ax.text(0.5, 0.5, "reference\n(grade 0)", ha="center", va="center")
                continue
            _images[_r] = _ax.imshow(
                _array,
                origin="lower",
                extent=_extent,
                aspect="auto",
                cmap=_cmap,
                vmin=_limits[0],
                vmax=_limits[1],
            )
            if _c == 0 or (_c == 1 and _axes[_r, 0].axison is False):
                _ax.set_ylabel(f"{_name}\n\nheight (calyx 0, stalk 1)", fontsize=9)
            if _r == len(rows) - 1:
                _ax.set_xlabel("radius / widest radius")
    for _r in _images:
        _fig.colorbar(_images[_r], ax=_axes[_r, :].tolist(), shrink=0.9)
    _fig.suptitle(title)
    return _fig


@app.cell(hide_code=True)
def _(MAP_COLUMNS, column_fruit, group_means, map_sums):
    # Voxel maps from the two long sections: map_sums is (fruit, 4, height, radius) with
    # counts, CT sums, reconstruction sums and residual sums.
    _columns = []
    for _title, _condition in MAP_COLUMNS:
        _selected = column_fruit(_condition)
        _means = group_means(map_sums[:, 0], map_sums[:, 1:], _selected)
        _columns.append((_title, int(_selected.sum()), _means))
    _healthy = _columns[0][2]

    def _difference(index):
        return lambda m, c: None if c == 0 else m[index] - _healthy[index]

    hr_figure = draw_maps(
        _columns,
        [
            ("CT", lambda m, c: m[0], "gray", (0.5, 1.0)),
            ("CT minus\ngrade 0 CT", _difference(0), "RdBu_r", (-0.06, 0.06)),
            (
                "reconstruction minus\ngrade 0 reconstruction",
                _difference(1),
                "RdBu_r",
                (-0.06, 0.06),
            ),
            # Own scale: the residual is about ten times smaller than the differences above.
            (
                "residual\n(reconstruction minus CT)",
                lambda m, c: m[2],
                "RdBu_r",
                (-0.02, 0.02),
            ),
        ],
        "2526 fruit, two long sections per fruit; red = brighter, blue = darker",
    )
    return (hr_figure,)


@app.cell(hide_code=True)
def _(
    MAP_COLUMNS,
    column_fruit,
    group_means,
    latent_counts,
    latent_sums,
    map_sums,
):
    _columns = []
    for _title, _condition in MAP_COLUMNS:
        _selected = column_fruit(_condition)
        _voxel = group_means(map_sums[:, 0], map_sums[:, 1:2], _selected)[0]
        _latent = group_means(latent_counts, latent_sums, _selected)[0]
        _columns.append((_title, int(_selected.sum()), (_voxel, _latent)))
    _healthy_ct, _healthy_latent = _columns[0][2]

    # Colour limits of the latent rows from the data (latent values have no fixed range):
    # the 1st to 99th percentile of the mean latent, and a symmetric limit at the 99th
    # percentile of the absolute differences from grade 0.
    _mean_values = np.concatenate([c[2][1].ravel() for c in _columns])
    _mean_limits = tuple(np.nanpercentile(_mean_values, [1, 99]))
    _differences = np.concatenate(
        [(c[2][1] - _healthy_latent).ravel() for c in _columns[1:]]
    )
    _d = float(np.nanpercentile(np.abs(_differences), 99))

    latent_figure = draw_maps(
        _columns,
        [
            (
                "CT minus\ngrade 0 CT",
                lambda m, c: None if c == 0 else m[0] - _healthy_ct,
                "RdBu_r",
                (-0.06, 0.06),
            ),
            ("mean latent", lambda m, c: m[1], "viridis", _mean_limits),
            (
                "latent minus\ngrade 0 latent",
                lambda m, c: None if c == 0 else m[1] - _healthy_latent,
                "RdBu_r",
                (-_d, _d),
            ),
        ],
        "2526 fruit; row 1 from two CT long sections, rows 2 and 3 from all body cells of"
        " the latent; red = higher value",
    )
    return (latent_figure,)


@app.cell(hide_code=True)
def _():
    map_view = mo.ui.dropdown(
        options=["CT and reconstruction", "latent"],
        value="CT and reconstruction",
        label="Map of",
    )
    map_view
    return (map_view,)


@app.cell(hide_code=True)
def _(hr_figure, latent_figure, map_view):
    hr_figure if map_view.value == "CT and reconstruction" else latent_figure
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **Check of one oddity (26 October 2026).** In row 2 of the CT view, browning 1 is darker
    than grade 0 at the seed cavity (height about 0.2 to 0.35, radius below 0.25), more than
    browning 2. Before reading anything into it: which fruit cause it? The table gives, per
    fruit, the mean CT in that region (from the two long sections), for grade 0 and browning
    1 only. If a few browning 1 fruit are far darker than the rest, they make the spot; look
    at them in section 7 or in notebook 05.

    Result (26 October 2026): not a few fruit; the whole browning 1 group is darker there
    (median 0.724 against 0.789 for grade 0). Explained by the grading rubric as stated by
    Ruben (the rubric table itself is not in the repository): browning grade 1 is a light
    discoloration of the kernel membranes, which line the seed cavity. Grades 2 and 3 spread
    further from the core.
    """)
    return


@app.cell(hide_code=True)
def _(map_sums, measured):
    # The region of the spot, in bins of the voxel map: height 0.2 to 0.35, radius 0 to 0.25.
    _h = slice(int(0.2 * N_HEIGHT_BINS), int(0.35 * N_HEIGHT_BINS) + 1)
    _r = slice(0, int(0.25 / MAX_RADIUS * N_RADIUS_BINS) + 1)
    _count = map_sums[:, 0, _h, _r].sum(axis=(1, 2))
    _ct = map_sums[:, 1, _h, _r].sum(axis=(1, 2))
    _with_region = measured.select(
        "fruit_id", "box", "group", "browning", "cavity", "defective"
    ).with_columns(
        seed_region_ct=pl.Series(
            np.where(_count > 0, _ct / np.maximum(_count, 1), np.nan)
        )
    )
    _grade_0 = _with_region.filter(
        (pl.col("group") == "2526") & (pl.col("defective") == 0)
    )
    _browning_1 = _with_region.filter(
        (pl.col("group") == "2526")
        & (pl.col("browning") == 1)
        & (pl.col("cavity") == 0)
    )
    print(
        "mean CT in the seed-cavity region, median (25 % to 75 %):"
        f" grade 0 {_grade_0['seed_region_ct'].median():.3f}"
        f" ({_grade_0['seed_region_ct'].quantile(0.25):.3f} to"
        f" {_grade_0['seed_region_ct'].quantile(0.75):.3f});"
        f" browning 1 {_browning_1['seed_region_ct'].median():.3f}"
        f" ({_browning_1['seed_region_ct'].quantile(0.25):.3f} to"
        f" {_browning_1['seed_region_ct'].quantile(0.75):.3f})"
    )
    _browning_1.sort("seed_region_ct").select("fruit_id", "box", "seed_region_ct").head(
        10
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 6. One example fruit per grade

    The fruit whose core MAE is closest to the median core MAE of its group (2526), so the
    example is typical and not chosen by eye. Long section 1 (plane 0) and the cross-section:
    CT, reconstruction, and signed residual (red = reconstruction brighter).
    """)
    return


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
def _(cfg, example_figure, hr_figure, latent_figure, save_step0_figures):
    # Saved under derived/ (outside git): CT images stay off GitHub until Hugo agrees.
    mo.stop(mo.running_in_notebook() and not save_step0_figures.value)
    _folder = cfg.path("derived_dir") / "figures"
    _folder.mkdir(parents=True, exist_ok=True)
    hr_figure.savefig(
        _folder / "06_height_radius_maps.png", dpi=130, bbox_inches="tight"
    )
    latent_figure.savefig(
        _folder / "06_height_radius_maps_latent.png", dpi=130, bbox_inches="tight"
    )
    example_figure.savefig(_folder / "06_examples.png", dpi=130, bbox_inches="tight")
    print(
        "saved 06_height_radius_maps.png, 06_height_radius_maps_latent.png and"
        f" 06_examples.png to {_folder}"
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 7. Scroll through an example fruit (3D)

    The example fruit of section 6, decoded in full by
    `python -m pearlatent.decode_examples` (about one minute per fruit on the CPU; run only
    after Ruben's approval). Three orthogonal slices through the chosen point, for CT,
    reconstruction and residual; a 3D view of the three CT planes; and a GIF that scrolls along
    the stem-calyx axis, for a presentation.
    """)
    return


@app.cell(hide_code=True)
def _(cfg, examples):
    # Every decoding run writes its own folder. Combine all of them, newest first, so fruit
    # decoded on different days are all available; keep only the current example fruit of
    # section 6 (so a fruit that was an example before the exclusions, such as the cut
    # 2526_A07, is not offered).
    _wanted = examples["fruit_id"].to_list()
    _runs = sorted(
        (cfg.path("derived_dir") / "step0_examples").glob("*_examples/volumes.npz"),
        reverse=True,
    )
    example_volumes = {}
    for _file in _runs:
        _saved = np.load(_file)
        for _k, _id in enumerate(_saved["fruit_id"]):
            _id = str(_id)
            if _id in _wanted and _id not in example_volumes:
                example_volumes[_id] = (
                    _saved["ct"][_k].astype(np.float32),
                    _saved["reconstruction"][_k].astype(np.float32),
                )
    _missing = [f for f in _wanted if f not in example_volumes]
    mo.stop(
        not example_volumes,
        mo.md(
            "_No decoded examples yet. After approval, run:_\n\n"
            f"`uv run python -m pearlatent.decode_examples {' '.join(_wanted)}`"
        ),
    )
    # Keep the order of section 6 (grade 0 first).
    example_volumes = {f: example_volumes[f] for f in _wanted if f in example_volumes}
    print(f"decoded examples: {list(example_volumes)}; not decoded yet: {_missing}")
    return (example_volumes,)


@app.cell
def _(example_volumes):
    volume_choice = mo.ui.dropdown(
        options=list(example_volumes), value=list(example_volumes)[0], label="Fruit"
    )
    x_slice = mo.ui.slider(0, 127, value=64, label="x", show_value=True)
    y_slice = mo.ui.slider(0, 127, value=64, label="y", show_value=True)
    z_slice = mo.ui.slider(0, 127, value=50, label="z (stem-calyx)", show_value=True)
    mo.hstack([volume_choice, x_slice, y_slice, z_slice], justify="start")
    return volume_choice, x_slice, y_slice, z_slice


@app.cell(hide_code=True)
def _(example_volumes, volume_choice, x_slice, y_slice, z_slice):
    _ct, _rec = example_volumes[volume_choice.value]
    _residual = _rec - _ct
    _i, _j, _k = x_slice.value, y_slice.value, z_slice.value
    _fig, _axes = plt.subplots(3, 3, figsize=(10, 10))
    for _r, (_volume, _name, _cmap, _lim) in enumerate(
        [
            (_ct, "CT", "gray", (0, 1)),
            (_rec, "reconstruction", "gray", (0, 1)),
            (_residual, "residual", "RdBu_r", (-0.3, 0.3)),
        ]
    ):
        # Long sections transposed so the stem-calyx axis is vertical, calyx at the bottom.
        for _c, (_picture, _view) in enumerate(
            [
                (_volume[_i, :, :].T, f"x = {_i}"),
                (_volume[:, _j, :].T, f"y = {_j}"),
                (_volume[:, :, _k], f"z = {_k}"),
            ]
        ):
            _ax = _axes[_r, _c]
            _ax.imshow(_picture, origin="lower", cmap=_cmap, vmin=_lim[0], vmax=_lim[1])
            _ax.set_xticks([])
            _ax.set_yticks([])
            _ax.set_title(f"{_name}, {_view}", fontsize=9)
    _fig.suptitle(volume_choice.value)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(example_volumes, volume_choice, x_slice, y_slice, z_slice):
    _ct, _ = example_volumes[volume_choice.value]
    # Every second voxel, so the 3D view stays fast (64 × 64 points per plane).
    _step = 2
    _grid = np.arange(0, 128, _step)
    _a, _b = np.meshgrid(_grid, _grid, indexing="ij")
    _i, _j, _k = x_slice.value, y_slice.value, z_slice.value
    _planes = [
        # (x, y, z coordinates of the plane, colour values)
        (np.full_like(_a, _i), _a, _b, _ct[_i, ::_step, ::_step]),
        (_a, np.full_like(_a, _j), _b, _ct[::_step, _j, ::_step]),
        (_a, _b, np.full_like(_a, _k), _ct[::_step, ::_step, _k]),
    ]
    _figure = go.Figure(
        [
            go.Surface(
                x=_x,
                y=_y,
                z=_z,
                surfacecolor=_colour,
                colorscale="gray",
                cmin=0,
                cmax=1,
                showscale=False,
            )
            for _x, _y, _z, _colour in _planes
        ]
    )
    _figure.update_layout(
        height=550,
        scene=dict(aspectmode="cube", zaxis_title="z (stem-calyx)"),
        title=f"{volume_choice.value}: three CT planes",
    )
    _figure
    return


@app.cell
def _():
    # Long sections (first axis) show the pear shape and the core over its whole height;
    # cross-sections (third axis) go from the calyx to the stalk.
    gif_direction = mo.ui.dropdown(
        options=["long sections (through x)", "cross-sections (through z)"],
        value="long sections (through x)",
        label="Scroll through",
    )
    save_gif = mo.ui.run_button(label="Save a GIF (derived/figures/)")
    mo.hstack([gif_direction, save_gif], justify="start")
    return gif_direction, save_gif


@app.cell(hide_code=True)
def _(cfg, example_volumes, gif_direction, save_gif, volume_choice):
    mo.stop(not save_gif.value)
    _ct, _rec = example_volumes[volume_choice.value]
    _long = gif_direction.value.startswith("long")

    def section(volume: np.ndarray, index: int) -> np.ndarray:
        """Long section at x = index (transposed: stem-calyx axis vertical, calyx at the
        bottom), or cross-section at z = index."""
        return volume[index, :, :].T if _long else volume[:, :, index]

    _fig, _axes = plt.subplots(1, 3, figsize=(9, 3.4))
    _images = []
    for _ax, (_volume, _name, _cmap, _lim) in zip(
        _axes,
        [
            (_ct, "CT", "gray", (0, 1)),
            (_rec, "reconstruction", "gray", (0, 1)),
            (_rec - _ct, "residual", "RdBu_r", (-0.3, 0.3)),
        ],
    ):
        _images.append(
            (
                _ax.imshow(
                    section(_volume, 0),
                    cmap=_cmap,
                    vmin=_lim[0],
                    vmax=_lim[1],
                    origin="lower",
                ),
                _volume,
            )
        )
        _ax.set_title(_name)
        _ax.set_xticks([])
        _ax.set_yticks([])
    _label = _fig.suptitle("")

    def draw(index: int):
        """Show section `index` in the three panels."""
        for _image, _volume in _images:
            _image.set_data(section(_volume, index))
        _where = f"x = {index}" if _long else f"z = {index} (calyx 0, stalk 127)"
        _label.set_text(f"{volume_choice.value}, {_where}")
        return [_image for _image, _ in _images]

    # Every second slice, 12 frames per second: about 5 seconds per fruit.
    _animation = FuncAnimation(_fig, draw, frames=range(0, 128, 2))
    _out = (
        cfg.path("derived_dir")
        / "figures"
        / f"06_{volume_choice.value}_scroll_{'x' if _long else 'z'}.gif"
    )
    _out.parent.mkdir(parents=True, exist_ok=True)
    _animation.save(_out, writer=PillowWriter(fps=12))
    plt.close(_fig)
    print(f"saved {_out}")
    return


if __name__ == "__main__":
    app.run()
