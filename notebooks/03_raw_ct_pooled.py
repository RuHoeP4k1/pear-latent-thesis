import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title="03 Raw CT at latent resolution")

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt
    import nibabel as nib
    import numpy as np
    import polars as pl

    from pearlatent.config import load_config
    from pearlatent.pooling import (
        MASK_THRESHOLD,
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
    latent grid of 32³, and does the pooling code (fruit mask, mean, 99th percentile) give
    sensible values on real fruit? The averaged raw CT is also a
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
    cells. The cylindrical profile (height × radius bins, averaged over angle) was dropped on
    10 October 2026 (CHARTER.md); its columns are removed from the feature table.

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
        # Runs before 10 October 2026 also hold the cylindrical profile (cyl_*); dropped.
        pl.read_parquet(run_dir / "features.parquet").select(pl.exclude("^cyl_.*$")),
        on="fruit_id",
        validate="1:1",
    )
    mo.vstack(
        [
            mo.md(f"Run folder: `{run_dir.name}`, **{raw_features.height}** fruit."),
            mo.ui.table(raw_features, selection=None, label="Pooled raw CT features"),
        ]
    )
    return manifest, raw_features


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2. Sanity checks

    Number of fruit cells per season (fruit size at the latent resolution), and the mean and 99th
    percentile of the scaled CT inside the fruit body.

    **Result (5 October 2026).** Median fruit body 3,382 cells in 2425 and 2,591 in 2526; median
    scaled CT inside the body 0.79 in both seasons.
    """)
    return


@app.cell
def _(raw_features):
    mo.ui.table(
        raw_features.group_by("season")
        .agg(
            pl.len().alias("n_fruit"),
            pl.col("n_mask_cells").min().alias("mask_cells_min"),
            pl.col("n_mask_cells").median().alias("mask_cells_median"),
            pl.col("mean").median().round(4).alias("mean_median"),
            pl.col("p99").median().round(4).alias("p99_median"),
        )
        .sort("season"),
        selection=None,
        label="Per season",
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


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Stalk viewer

    One row of fruit per season, 2526 above 2425, drawn at random (choose a box to restrict the
    draw; the button draws other fruit). Within a row the fruit are sorted by body length, shortest
    first. Each panel is the maximum projection of the scaled volume along the first array axis:
    the second axis runs left to right, the third axis bottom (calyx end) to top. The two orange
    lines mark the bottom and top of the fruit body (`body_mask`, at the 4-voxel resolution of
    the grid); fruit above the upper line is stalk that `body_mask` removed. The panel title gives
    the body length in slices of 4 voxels and the stalk length above the body in voxels. Each
    panel reads one volume from disk, so a redraw takes a few seconds.
    """)
    return


@app.cell(hide_code=True)
def _(raw_features):
    def _box_dropdown(season: str) -> mo.ui.dropdown:
        _boxes = raw_features.filter(pl.col("season") == season)["box"].unique().sort()
        return mo.ui.dropdown(
            ["all boxes", *_boxes.to_list()], value="all boxes", label=f"Box {season}"
        )

    stalk_box = {"2526": _box_dropdown("2526"), "2425": _box_dropdown("2425")}
    stalk_n = mo.ui.slider(4, 16, value=10, label="Fruit per season", show_value=True)
    stalk_redraw = mo.ui.button(
        value=0, on_click=lambda v: v + 1, label="Draw other fruit"
    )
    mo.hstack(
        [stalk_box["2526"], stalk_box["2425"], stalk_n, stalk_redraw],
        justify="start",
        gap=2,
    )
    return stalk_box, stalk_n, stalk_redraw


@app.cell(hide_code=True)
def _(manifest, run_dir, volumes):
    _body_masks = np.load(run_dir / "masks.npy", mmap_mode="r")
    _rows = dict(manifest.select("fruit_id", "row").iter_rows())
    _paths = dict(volumes.select("fruit_id", "path").iter_rows())

    def stalk_view(fruit_id: str, ax) -> None:
        """Draw the maximum projection of one volume with the fruit body extent on `ax`."""
        _scaled = scale_minmax(np.asarray(nib.load(_paths[fruit_id]).dataobj))
        _slices = np.flatnonzero(
            np.asarray(_body_masks[_rows[fruit_id]]).any(axis=(0, 1))
        )
        _bottom, _top = 4 * int(_slices[0]), 4 * int(_slices[-1] + 1)
        _fruit = np.flatnonzero((_scaled > MASK_THRESHOLD).any(axis=(0, 1)))
        _stalk = max(int(_fruit[-1]) + 1 - _top, 0)
        # Image rows: third axis; columns: second axis; origin lower puts the calyx end at the bottom
        ax.imshow(_scaled.max(axis=0).T, origin="lower", cmap="gray", vmin=0, vmax=1)
        for _k in (_bottom, _top):
            ax.axhline(_k - 0.5, color="orange", linewidth=1)
        ax.set_xticks([])
        ax.set_yticks([])
        _title = [fruit_id, f"body {len(_slices)}, stalk {_stalk}"]
        ax.set_title(chr(10).join(_title), fontsize=8)

    return (stalk_view,)


@app.cell
def _(raw_features, stalk_box, stalk_n, stalk_redraw, stalk_view):
    _fig, _axes = plt.subplots(
        2, stalk_n.value, figsize=(1.5 * stalk_n.value, 4.4), squeeze=False
    )
    for _axes_row, _season in zip(_axes, ["2526", "2425"], strict=True):
        _pool = raw_features.filter(pl.col("season") == _season)
        if stalk_box[_season].value != "all boxes":
            _pool = _pool.filter(pl.col("box") == stalk_box[_season].value)
        _drawn = _pool.sample(
            min(stalk_n.value, _pool.height), seed=stalk_redraw.value
        ).sort("n_mask_slices", "fruit_id")
        for _ax in _axes_row:
            _ax.set_axis_off()
        for _ax, _fruit_id in zip(_axes_row, _drawn["fruit_id"], strict=False):
            _ax.set_axis_on()
            stalk_view(_fruit_id, _ax)
    _fig.tight_layout()
    _fig
    return


if __name__ == "__main__":
    app.run()
