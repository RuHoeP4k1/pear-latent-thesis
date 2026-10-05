import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title="03 Raw CT at latent resolution")

with app.setup:
    import altair as alt
    import marimo as mo
    import nibabel as nib
    import numpy as np
    import polars as pl

    from pearlatent.config import load_config
    from pearlatent.pooling import (
        N_HEIGHT,
        N_RADIUS,
        block_mean,
        body_mask,
        fruit_mask,
        pool_grids,
        scale_minmax,
    )
    from pearlatent.runs import latest_run_dir, new_run_dir, write_run_record
    from pearlatent.volumes import list_volumes


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 03 Raw CT at latent resolution

    **Question.** What does the raw computed tomography (CT) look like when averaged to the
    latent grid of 32³, and does the pooling code (fruit mask, mean, 99th percentile,
    cylindrical profile) give sensible values on real fruit? The averaged raw CT is also a
    reference point (rule 7 of CHARTER.md): the latent has to carry more usable information
    than this to be worth using.

    **Inputs.** `ct_dir_2425`, `ct_dir_2526` (all volumes), from `config/local.toml`. No labels.

    **Outputs.** `derived/raw_ct_32/<run_id>/`: `grids.npy` (fruit × 32³, float32, mean of each
    4³ block of the volume scaled to [0, 1]), `masks.npy` (fruit × 32³, fruit body cells, stalk removed),
    `manifest.parquet` (fruit_id, season, box, storage, file_stem, row), `features.parquet`
    (pooled features per fruit), `run.json`.

    **Status.** draft
    """)
    return


@app.cell
def _():
    cfg = load_config()
    out_base = cfg.path("derived_dir") / "raw_ct_32"
    return cfg, out_base


@app.cell
def _(cfg):
    volumes = pl.concat(
        [
            list_volumes(cfg.path("ct_dir_2425"), "2425"),
            list_volumes(cfg.path("ct_dir_2526"), "2526"),
        ]
    ).with_row_index("row")
    return (volumes,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1. Grids, masks and features

    Each volume is min-max scaled to [0, 1] (as in Hugo's loader) and averaged over blocks of
    4³ voxels, which matches the 4-voxel spacing of the latent cells. A cell is fruit when at
    least half of its voxels are above 0.1 (the threshold of notebook 02). Of these cells only the
    fruit body is kept: the largest connected part, without slices of fewer than 4 cells along
    the third axis (the stalk, present in the 2526 volumes; see section 3). Features are computed
    inside the mask only, so they do not mix in fruit shape through the number of background
    cells. The cylindrical profile has 8 height bins (bin 0 at the calyx end) and 4 radius bins
    (bin 0 the core, bin 3 the skin), averaged over angle.

    Reading all volumes takes about a minute. The latest saved run is reloaded; press the button
    to compute a new one.
    """)
    return


@app.cell
def _():
    recompute = mo.ui.run_button(label="Compute grids again")
    recompute
    return (recompute,)


@app.cell
def _(out_base, recompute, volumes):
    _existing = latest_run_dir(out_base, "raw_ct_32")
    if _existing is not None and not recompute.value:
        run_dir = _existing
    else:
        run_dir = new_run_dir(out_base, "raw_ct_32")
        write_run_record(
            run_dir,
            {
                "step": "raw_ct_32",
                "factor": 4,
                "mask_threshold": 0.1,
                "mask_min_fraction": 0.5,
                "mask": "body_mask(fruit_mask), min 4 cells per slice",
                "n_height": N_HEIGHT,
                "n_radius": N_RADIUS,
                "n_volumes": volumes.height,
            },
        )
        _grids = np.empty((volumes.height, 32, 32, 32), dtype=np.float32)
        _masks = np.empty((volumes.height, 32, 32, 32), dtype=bool)
        for _row in mo.status.progress_bar(
            list(volumes.iter_rows(named=True)), title="Reading volumes"
        ):
            _scaled = scale_minmax(np.asarray(nib.load(_row["path"]).dataobj))
            _grids[_row["row"]] = block_mean(_scaled)
            _masks[_row["row"]] = body_mask(fruit_mask(_scaled))
        np.save(run_dir / "grids.npy", _grids)
        np.save(run_dir / "masks.npy", _masks)
        volumes.drop("path").write_parquet(run_dir / "manifest.parquet")
        pool_grids(_grids, _masks, volumes["fruit_id"].to_list()).write_parquet(
            run_dir / "features.parquet"
        )
    return (run_dir,)


@app.cell
def _(run_dir):
    manifest = pl.read_parquet(run_dir / "manifest.parquet")
    raw_features = manifest.join(
        pl.read_parquet(run_dir / "features.parquet"), on="fruit_id", validate="1:1"
    )
    mo.vstack(
        [
            mo.md(f"Run folder: `{run_dir.name}`, **{raw_features.height}** fruit."),
            mo.ui.table(raw_features, selection=None, label="Pooled raw CT features"),
        ]
    )
    return (raw_features,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2. Sanity checks

    Number of fruit cells per season (fruit size at the latent resolution), mean and 99th
    percentile inside the fruit, and the cylindrical profile averaged over all fruit of a season.
    The profile should be symmetric in nothing in particular, but it should be smooth and show
    the core (radius bin 0) and the calyx and stem ends (height bins 0 and 7) differing from the
    flesh. Empty bins would mean the mask or the binning is wrong.

    **Result (5 October 2026).** Median fruit body 3,382 cells in 2425 and 2,591 in 2526; median
    scaled CT inside the body 0.79 in both seasons. The profile is smooth and similar in both
    seasons: lowest in the core at mid height (the core and seed cavity, 0.68 to 0.75 in radius
    bin 0 of height bins 2 to 4) and in the skin bin (0.60 to 0.77, partial-volume cells at the
    surface). Two fruit (2526_K14, 2526_N05) have one empty bin, the core of the top height bin:
    the narrowest slices of the neck have no cell close enough to the centre. Every other bin of
    every fruit has values. Later models must handle these two missing values inside the
    training fold.
    """)
    return


@app.cell
def _(raw_features):
    _cyl = [c for c in raw_features.columns if c.startswith("cyl_")]
    _summary = raw_features.group_by("season").agg(
        pl.len().alias("n_fruit"),
        pl.col("n_mask_cells").min().alias("mask_cells_min"),
        pl.col("n_mask_cells").median().alias("mask_cells_median"),
        pl.col("mean").median().round(4).alias("mean_median"),
        pl.col("p99").median().round(4).alias("p99_median"),
        pl.sum_horizontal([pl.col(c).is_nan() for c in _cyl]).sum().alias("empty_bins"),
    )
    _profile = (
        raw_features.group_by("season")
        .agg([pl.col(c).mean() for c in _cyl])
        .unpivot(index="season", variable_name="bin", value_name="value")
        .with_columns(
            pl.col("bin").str.extract(r"h(\d+)").cast(pl.Int32).alias("height_bin"),
            pl.col("bin").str.extract(r"r(\d+)").cast(pl.Int32).alias("radius_bin"),
        )
    )
    _heat = (
        alt.Chart(_profile.to_pandas())
        .mark_rect()
        .encode(
            alt.X("radius_bin:O", title="Radius bin (0 core, 3 skin)"),
            alt.Y("height_bin:O", title="Height bin (0 calyx end)"),
            alt.Color("value:Q", title="Mean scaled CT", scale=alt.Scale(zero=False)),
            alt.Column("season:N"),
        )
        .properties(width=160, height=240)
    )
    mo.vstack(
        [mo.ui.table(_summary.sort("season"), selection=None, label="Per season"), _heat]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 3. Body length: the stalk in the 2526 volumes

    `n_mask_slices` is the number of slices of 4 voxels along the third axis that contain fruit
    body. A fruit whose body fills the full 128 voxels has 32.

    **Result (5 October 2026).** Most 2526 volumes include the stalk (8 of 12 in a random sample of
    projections, seed 1; in 2425 at most a short stub), and the stalk is counted in
    the 128-voxel length: projections of 2526_B11, 2526_K03 and 2526_N10 show a stalk of 20 to 35
    voxels at the high end of the third axis, while 2425_A40 shows none (images looked at on
    5 October, not saved). So in 2526 the fruit with its stalk is rescaled to 128 voxels, not the
    fruit body. Body length in slices: 2425 median 31 (range 27 to 32); 2526 median 28 (range 21
    to 32), with box medians from 25 (K) to 31 (G, I). Consequences: (1) fruit scale differs
    between fruit and between orchards in 2526, depending on stalk length; (2) the encoder was
    trained on 2024 fruit without stalks, so the 2526 volumes differ from its training data;
    (3) height must be measured over the body, which `body_mask` does.
    """)
    return


@app.cell
def _(raw_features):
    mo.ui.table(
        raw_features.group_by("season", "box")
        .agg(
            pl.len().alias("n_fruit"),
            pl.col("n_mask_slices").min().alias("body_slices_min"),
            pl.col("n_mask_slices").median().alias("body_slices_median"),
            pl.col("n_mask_slices").max().alias("body_slices_max"),
        )
        .sort("season", "box"),
        selection=None,
        label="Body length in slices of 4 voxels, per box",
    )
    return


if __name__ == "__main__":
    app.run()
