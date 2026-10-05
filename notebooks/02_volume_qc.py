import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title="02 Volume quality check")

with app.setup:
    import altair as alt
    import marimo as mo
    import polars as pl

    from pearlatent.config import load_config
    from pearlatent.exclusions import apply_exclusions, build_exclusions
    from pearlatent.labels import load_labels
    from pearlatent.runs import write_run_record
    from pearlatent.volume_qc import (
        DEFAULT_THRESHOLD,
        MIN_LONG_EXTENT,
        flag_volumes,
        qc_table,
    )
    from pearlatent.volumes import list_volumes


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 02 Volume quality check

    **Question.** Is every computed tomography (CT) volume of both seasons usable: 128³, not empty,
    the fruit rescaled to about 128 voxels in length, all fruit pointing the same way, no
    duplicates? Which fruit are excluded or flagged for later analyses?

    **Inputs.** `ct_dir_2425`, `ct_dir_2526` (all volumes are opened), `labels_2425`,
    `labels_2526`, from `config/local.toml`. No label is used for the volume measurements; labels
    are used only to build the exclusion table.

    **Outputs.** `derived/volume_qc/volume_qc.parquet` with `run.json` (one row per volume),
    `derived/exclusions.csv` (fruit_id, season, reason, action; rule 10 of CHARTER.md).
    Figure `reports/figures/02_volume_qc.pdf` when the save button is pressed.

    **Status.** draft
    """)
    return


@app.cell
def _():
    cfg = load_config()
    qc_dir = cfg.path("derived_dir") / "volume_qc"
    qc_file = qc_dir / "volume_qc.parquet"
    return cfg, qc_dir, qc_file


@app.cell
def _(cfg):
    volumes = pl.concat(
        [
            list_volumes(cfg.path("ct_dir_2425"), "2425"),
            list_volumes(cfg.path("ct_dir_2526"), "2526"),
        ]
    )
    return (volumes,)


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    ## 1. Measurements per volume

    Each volume is min-max scaled to [0, 1] as in Hugo's loader. A voxel counts as fruit when its
    scaled value is above **{DEFAULT_THRESHOLD}**. The background is exactly 0 and fruit tissue
    lies mostly above 0.7, so the mask hardly depends on this value. Per volume: shape, data type,
    minimum and maximum, the voxel size in the NIfTI header, fruit volume in voxels, the extent of
    the fruit in voxels along each array axis, the position of the widest cross-section along the
    third axis, intensity inside the fruit (mean, 5th and 95th percentile of the scaled values),
    and a checksum of the voxel data to find duplicates.

    Reading all volumes takes a few minutes. The table is saved and reloaded; press the button to
    compute it again.
    """)
    return


@app.cell
def _():
    recompute = mo.ui.run_button(label="Compute the quality check again")
    recompute
    return (recompute,)


@app.cell
def _(qc_dir, qc_file, recompute, volumes):
    if qc_file.exists() and not recompute.value:
        qc = pl.read_parquet(qc_file)
    else:
        qc = qc_table(
            volumes,
            DEFAULT_THRESHOLD,
            progress=lambda rows: mo.status.progress_bar(rows, title="Reading volumes"),
        )
        qc_dir.mkdir(parents=True, exist_ok=True)
        qc.write_parquet(qc_file)
        write_run_record(
            qc_dir,
            {"step": "volume_qc", "threshold": DEFAULT_THRESHOLD, "n_volumes": qc.height},
        )
    return (qc,)


@app.cell
def _(qc):
    qc_flagged = flag_volumes(qc, MIN_LONG_EXTENT)
    mo.ui.table(qc_flagged, selection=None, label="One row per volume")
    return (qc_flagged,)


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    ## 2. Flags and summaries

    A volume is flagged when its shape is not 128³, when it contains no fruit, when the fruit
    extent along its longest axis is below {MIN_LONG_EXTENT} voxels, or when its voxel data are
    identical to another volume.

    **Result (5 October 2026).** All 1093 volumes (660 in 2425, 433 in 2526) are 128³ uint16
    with minimum 0 and maximum 65535; the header voxel size is 1.0 on every axis in every file,
    so the header does not carry the real voxel size. The longest fruit axis is the third array
    axis in every volume, with extent 125 to 128 voxels (2425: median 126, 37 volumes below 126;
    2526: median 128, minimum 126). No volume is flagged and all 1093 checksums differ. This
    confirms on every fruit that each is rescaled to about 128 voxels in length.
    """)
    return


@app.cell
def _(qc_flagged):
    _per_season = qc_flagged.group_by("season").agg(
        pl.len().alias("n_volumes"),
        (pl.col("qc_flag") != "").sum().alias("n_flagged"),
        pl.col("shape").unique().str.join(", ").alias("shapes"),
        pl.col("dtype").unique().str.join(", ").alias("dtypes"),
        pl.col("min").min().alias("min_of_min"),
        pl.col("max").max().alias("max_of_max"),
        pl.col("voxel_size_0").unique().cast(pl.String).str.join(", ").alias("header_voxel_sizes"),
        pl.col("long_axis").unique().cast(pl.String).str.join(", ").alias("long_axes"),
        pl.col("long_extent").min().alias("long_extent_min"),
        pl.col("long_extent").median().alias("long_extent_median"),
        pl.col("fruit_voxels").median().alias("fruit_voxels_median"),
        pl.col("widest_slice_axis2").median().alias("widest_slice_median"),
        (pl.col("widest_slice_axis2") < 64).sum().alias("widest_slice_below_64"),
    )
    mo.vstack(
        [
            mo.ui.table(_per_season.sort("season"), selection=None, label="Per season"),
            mo.ui.table(
                qc_flagged.filter(pl.col("qc_flag") != ""),
                selection=None,
                label="Flagged volumes",
            ),
        ]
    )
    return


@app.cell
def _(qc_flagged):
    qc_per_box = (
        qc_flagged.group_by("season", "box")
        .agg(
            pl.len().alias("n_volumes"),
            (pl.col("qc_flag") != "").sum().alias("n_flagged"),
            pl.col("long_extent").min().alias("long_extent_min"),
            pl.col("fruit_voxels").median().alias("fruit_voxels_median"),
            pl.col("mean_in_fruit").median().alias("mean_in_fruit_median"),
            pl.col("p05_in_fruit").median().alias("p05_in_fruit_median"),
            (pl.col("widest_slice_axis2") < 64).sum().alias("widest_slice_below_64"),
        )
        .sort("season", "box")
    )
    mo.ui.table(qc_per_box, selection=None, label="Per box")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 3. Distributions

    Fruit length along the longest axis, position of the widest cross-section along the third
    axis (the wide end of a pear is the calyx end, so a single cluster means all fruit point the
    same way), and mean scaled intensity inside the fruit per box. A box or season that stands
    apart in intensity points to a difference in scanning that a model could use instead of the
    defect.

    **Result (5 October 2026).** The widest cross-section lies at index 27 to 43 along the third
    axis in 90 % of volumes (median 34 in both seasons) and below index 64 in all 1093, so every
    fruit points the same way, with the wide end at the low end of the third axis. Median scaled
    intensity in the fruit is 0.80 to 0.84 in every box; the 95th percentile is slightly lower in
    2526 (0.87 to 0.91) than in 2425 (0.90 to 0.92). Fruit volume in voxels differs strongly by
    box (2526 medians from 120,805 in box K to 228,988 in box I): with every fruit rescaled to
    the same length, this measures how slender the fruit are, and slenderness is an orchard
    property. Box K is also the box with the fewest defects, so shape is a candidate shortcut to
    check with the nuisance test (rule 8 of CHARTER.md).
    """)
    return


@app.cell
def _(qc_flagged):
    _base = alt.Chart(qc_flagged.to_pandas())
    _extent = (
        _base.mark_bar()
        .encode(
            alt.X("long_extent:Q", bin=alt.Bin(step=1), title="Fruit extent, longest axis (voxels)"),
            alt.Y("count():Q", title="Volumes"),
            alt.Row("season:N"),
        )
        .properties(width=300, height=120)
    )
    _widest = (
        _base.mark_bar()
        .encode(
            alt.X(
                "widest_slice_axis2:Q",
                bin=alt.Bin(step=4),
                title="Widest cross-section, index along third axis",
            ),
            alt.Y("count():Q", title="Volumes"),
            alt.Row("season:N"),
        )
        .properties(width=300, height=120)
    )
    _intensity = (
        _base.mark_boxplot()
        .encode(
            alt.X("box:N", title="Box"),
            alt.Y("mean_in_fruit:Q", title="Mean scaled intensity in fruit", scale=alt.Scale(zero=False)),
            alt.Color("season:N"),
        )
        .properties(width=650, height=200)
    )
    mo.vstack([mo.hstack([_extent, _widest]), _intensity])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 4. Exclusion table

    Rule 10 of CHARTER.md. Excluded: the 17 unscanned 2526 fruit (Hugo's advice). Flagged: 2526
    fruit with browning 3 and cavity 3 (possibly rotten, until Hugo identifies the rotten fruit)
    and every volume flagged above. Later notebooks call `apply_exclusions(df, exclusions, mode)`:
    mode `"primary"` drops excluded fruit, mode `"sensitivity"` also drops flagged fruit.

    **Result (5 October 2026).** 17 fruit excluded and 24 flagged (browning 3 and cavity 3), all
    in 2526; no volume flagged by the quality check. One flagged fruit (2526_H24) is also
    unscanned, so the sensitivity analysis drops 23 scanned fruit. In 2425 only 11 of the 175
    fruit with browning 3 and cavity 3 have rot = 1 (notebook 01, section 5), so most of the 24
    flagged 2526 fruit are probably real severe cases, not rot.
    """)
    return


@app.cell
def _(cfg, qc_flagged, volumes):
    labels = load_labels(cfg)
    exclusions = build_exclusions(labels, volumes, qc_flagged)
    exclusions.write_csv(cfg.path("derived_dir") / "exclusions.csv")
    return exclusions, labels


@app.cell
def _(exclusions, labels):
    _remaining = pl.DataFrame(
        {
            "mode": ["none", "primary", "sensitivity"],
            "fruit_2425": [
                _df.filter(pl.col("season") == "2425").height
                for _df in (
                    labels,
                    apply_exclusions(labels, exclusions, "primary"),
                    apply_exclusions(labels, exclusions, "sensitivity"),
                )
            ],
            "fruit_2526": [
                _df.filter(pl.col("season") == "2526").height
                for _df in (
                    labels,
                    apply_exclusions(labels, exclusions, "primary"),
                    apply_exclusions(labels, exclusions, "sensitivity"),
                )
            ],
        }
    )
    mo.vstack(
        [
            mo.ui.table(
                exclusions.group_by("season", "action", "reason").len("n_fruit").sort("season", "action"),
                selection=None,
                label="Exclusion table by reason",
            ),
            mo.ui.table(_remaining, selection=None, label="Labelled fruit left per mode"),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 5. Figure for the thesis
    """)
    return


@app.cell
def _():
    save_figure = mo.ui.run_button(label="Save figure to reports/figures/")
    save_figure
    return (save_figure,)


@app.cell
def _(cfg, qc_flagged, save_figure):
    mo.stop(not save_figure.value)
    import matplotlib.pyplot as plt

    _fig, _axes = plt.subplots(1, 2, figsize=(8, 3), constrained_layout=True)
    for _season in ("2425", "2526"):
        _d = qc_flagged.filter(pl.col("season") == _season)
        _axes[0].hist(_d["long_extent"], bins=range(100, 130), alpha=0.6, label=_season)
        _axes[1].hist(_d["widest_slice_axis2"], bins=range(0, 129, 4), alpha=0.6, label=_season)
    _axes[0].set_xlabel("Fruit extent along the longest axis (voxels)")
    _axes[1].set_xlabel("Widest cross-section, index along the third axis")
    for _ax in _axes:
        _ax.set_ylabel("Volumes")
        _ax.legend(title="Season")
    _out = cfg.root / "reports" / "figures" / "02_volume_qc.pdf"
    _out.parent.mkdir(parents=True, exist_ok=True)
    _fig.savefig(_out)
    mo.md(f"Saved `{_out.relative_to(cfg.root)}`.")
    return


if __name__ == "__main__":
    app.run()
