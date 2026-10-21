import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title="05 Latent sanity checks")

with app.setup:
    import altair as alt
    import marimo as mo
    import matplotlib.pyplot as plt
    import nibabel as nib
    import numpy as np
    import polars as pl
    from sklearn.decomposition import PCA

    from pearlatent.config import load_config
    from pearlatent.labels import load_labels


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 05 Latent sanity checks

    **Question.** Do the saved latents look sensible, and what are the main directions of
    variation between fruit? Nothing is fitted on labels; the labels are only used as colours.

    **Inputs.** `derived/latents/hugo_vae/20261013T171306Z_hugo_vae/` (`latents.npy`,
    `manifest.parquet`: the 940 open fruit, the sealed test fruit are not in it) and both label
    files (`labels_2425`, `labels_2526`).

    **Outputs.** None. Figures only.

    **Status.** draft
    """)
    return


@app.cell
def _():
    cfg = load_config()
    run_dir = (
        cfg.path("derived_dir") / "latents" / "hugo_vae" / "20261013T171306Z_hugo_vae"
    )
    return cfg, run_dir


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1. Load the latents and attach the labels

    The latent of one fruit is a 32 × 32 × 32 grid of numbers (the mean `mu` of Hugo's encoder).
    Row `i` of `latents.npy` belongs to row `i` of `manifest.parquet`.
    """)
    return


@app.cell
def _(cfg, run_dir):
    latents = np.load(run_dir / "latents.npy")
    manifest = pl.read_parquet(run_dir / "manifest.parquet")

    labels = load_labels(cfg).select("fruit_id", "browning", "cavity")

    # Attach the grades to every fruit. validate="1:1" stops if a fruit occurs twice.
    fruit = manifest.join(
        labels, on="fruit_id", how="left", validate="1:1"
    ).with_columns(defective=pl.max_horizontal("browning", "cavity"))

    print(f"latents: {latents.shape[0]} fruit, grid {latents.shape[1:]}")
    print(f"manifest: {manifest.height} fruit; after the join: {fruit.height} fruit")
    print(f"fruit without a grade: {fruit['browning'].null_count()}")
    return fruit, latents


@app.cell
def _(fruit):
    # How many fruit per season and per group, to know what we are looking at.
    fruit.group_by("season", "storage", "hugo_split").len().sort(
        "season", "storage", "hugo_split"
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2. Basic numbers per fruit

    For each fruit: the mean, standard deviation, minimum and maximum of its 32,768 latent values.
    A fruit that was read or encoded wrongly would stand out here (all zeros, a huge range, or a
    value far from its box).
    """)
    return


@app.cell
def _(fruit, latents):
    # One row of 32,768 numbers per fruit.
    flat = latents.reshape(latents.shape[0], -1)

    per_fruit = fruit.select("fruit_id", "season", "box", "storage").with_columns(
        latent_mean=flat.mean(axis=1),
        latent_sd=flat.std(axis=1),
        latent_min=flat.min(axis=1),
        latent_max=flat.max(axis=1),
    )

    print(f"all values finite: {np.isfinite(flat).all()}")
    per_fruit.describe()
    return flat, per_fruit


@app.cell
def _():
    summary_choice = mo.ui.dropdown(
        options=["latent_mean", "latent_sd", "latent_min", "latent_max"],
        value="latent_sd",
        label="Number per fruit",
    )
    summary_choice
    return (summary_choice,)


@app.cell
def _(per_fruit, summary_choice):
    # One box plot per box; each point inside is one fruit.
    alt.Chart(per_fruit).mark_boxplot(extent="min-max").encode(
        x=alt.X("box:N", title="Box"),
        y=alt.Y(f"{summary_choice.value}:Q", scale=alt.Scale(zero=False)),
        color=alt.Color("season:N"),
    ).properties(width=700, height=250)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 3. Where does the latent vary between fruit?

    For every cell: the mean over all fruit and the standard deviation over all fruit. Shown as
    two slices through the middle of the grid. The third array axis is the stem–calyx axis
    (calyx at the low end), so the long section shows the fruit standing up.
    """)
    return


@app.cell
def _(latents):
    mean_per_cell = latents.mean(axis=0)
    sd_per_cell = latents.std(axis=0)

    middle = 16  # the middle of 32 cells

    _fig, _axes = plt.subplots(2, 2, figsize=(8, 8))
    _pictures = [
        (mean_per_cell[middle, :, :].T, "Mean, long section"),
        (mean_per_cell[:, :, middle], "Mean, cross-section"),
        (sd_per_cell[middle, :, :].T, "Standard deviation, long section"),
        (sd_per_cell[:, :, middle], "Standard deviation, cross-section"),
    ]
    for _ax, (_picture, _title) in zip(_axes.flat, _pictures):
        # origin="lower" puts the calyx (low end of the axis) at the bottom.
        _image = _ax.imshow(_picture, origin="lower", cmap="viridis")
        _ax.set_title(_title)
        _fig.colorbar(_image, ax=_ax, shrink=0.8)
    _fig.tight_layout()
    _fig
    return mean_per_cell, middle


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 4. Principal component analysis (PCA)

    PCA on the 32,768 cells, mean-centred, **not** scaled to unit variance per cell. Reason:
    cells outside the fruit hardly vary; scaling them to unit variance would blow their noise
    up to the same weight as the cells inside the fruit.

    Choose which fruit the PCA is fitted on. The 2025–26 fruit are the main data set. "All open
    fruit" also includes the 2024–25 fruit, among them the 528 Hugo trained his VAE on.
    """)
    return


@app.cell
def _():
    pca_fruit_choice = mo.ui.dropdown(
        options=["2025-26 fruit", "all open fruit"],
        value="2025-26 fruit",
        label="Fit the PCA on",
    )
    pca_fruit_choice
    return (pca_fruit_choice,)


@app.cell
def _(flat, fruit, pca_fruit_choice):
    if pca_fruit_choice.value == "2025-26 fruit":
        keep = (fruit["season"] == "2526").to_numpy()
    else:
        keep = np.ones(fruit.height, dtype=bool)

    n_components = 10  # enough to look at; this is exploration, not a model
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(flat[keep])

    pc_names = [f"PC{i + 1}" for i in range(n_components)]
    scores_df = fruit.filter(pl.Series(keep)).with_columns(
        pl.Series(name, scores[:, i]) for i, name in enumerate(pc_names)
    )
    print(f"PCA fitted on {keep.sum()} of {fruit.height} fruit")
    return pc_names, pca, scores_df


@app.cell
def _(pc_names, pca):
    # Scree plot: how much of the variation each component explains.
    explained = pl.DataFrame(
        {
            "component": pc_names,
            "explained_percent": 100 * pca.explained_variance_ratio_,
        }
    ).with_columns(cumulative_percent=pl.col("explained_percent").cum_sum())

    alt.Chart(explained).mark_bar().encode(
        x=alt.X("component:N", sort=pc_names, title=None),
        y=alt.Y("explained_percent:Q", title="Explained variance (%)"),
        tooltip=["component", "explained_percent", "cumulative_percent"],
    ).properties(width=500, height=200)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 5. Score plot and parallel coordinates

    Drag a rectangle in the score plot to select fruit. The parallel-coordinates plot below shows
    every fruit as one line over PC1 to PC10: grey for all fruit, coloured for the selection.
    """)
    return


@app.cell
def _(pc_names):
    colour_choice = mo.ui.dropdown(
        options=["box", "storage", "hugo_split", "defective", "browning", "cavity"],
        value="box",
        label="Colour by",
    )
    x_choice = mo.ui.dropdown(options=pc_names, value="PC1", label="x axis")
    y_choice = mo.ui.dropdown(options=pc_names, value="PC2", label="y axis")
    mo.hstack([colour_choice, x_choice, y_choice], justify="start")
    return colour_choice, x_choice, y_choice


@app.cell
def _(colour_choice, scores_df, x_choice, y_choice):
    _chart = (
        alt.Chart(scores_df)
        .mark_point(filled=True, size=40)
        .encode(
            x=alt.X(f"{x_choice.value}:Q"),
            y=alt.Y(f"{y_choice.value}:Q"),
            color=alt.Color(f"{colour_choice.value}:N"),
            tooltip=["fruit_id", "box", "storage", "browning", "cavity"],
        )
        .properties(width=500, height=400)
    )
    score_plot = mo.ui.altair_chart(_chart)
    score_plot
    return (score_plot,)


@app.cell
def _(colour_choice, pc_names, score_plot, scores_df):
    selected_ids = score_plot.value["fruit_id"].to_list()

    # Long format: one row per fruit and component, which is what a line chart needs.
    lines = scores_df.select("fruit_id", colour_choice.value, *pc_names).unpivot(
        index=["fruit_id", colour_choice.value],
        on=pc_names,
        variable_name="component",
        value_name="score",
    )
    _x = alt.X("component:N", sort=pc_names, title=None)

    _all_fruit = (
        alt.Chart(lines)
        .mark_line(color="lightgrey", opacity=0.4)
        .encode(x=_x, y=alt.Y("score:Q"), detail="fruit_id:N")
    )
    _selection = (
        alt.Chart(lines.filter(pl.col("fruit_id").is_in(selected_ids)))
        .mark_line(opacity=0.9)
        .encode(
            x=_x,
            y="score:Q",
            detail="fruit_id:N",
            color=alt.Color(f"{colour_choice.value}:N"),
            tooltip=["fruit_id", "component", "score"],
        )
    )
    mo.vstack(
        [
            mo.md(f"**{len(selected_ids)} fruit selected.**"),
            (_all_fruit + _selection).properties(width=700, height=300),
        ]
    )
    return (selected_ids,)


@app.cell
def _(scores_df, selected_ids):
    # The selected fruit as a table, to look them up.
    scores_df.filter(pl.col("fruit_id").is_in(selected_ids)).select(
        "fruit_id",
        "box",
        "storage",
        "hugo_split",
        "browning",
        "cavity",
        "PC1",
        "PC2",
        "PC3",
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 6. Look at one fruit

    Pick a fruit to see its CT volume, its latent, and how its latent differs from the mean latent
    of all fruit. If fruit are selected in the score plot, only those are offered; otherwise all
    fruit in the PCA. Type in the box to search by name.
    """)
    return


@app.cell
def _(scores_df, selected_ids):
    if selected_ids:
        offered_ids = selected_ids
    else:
        offered_ids = scores_df["fruit_id"].to_list()

    fruit_picker = mo.ui.dropdown(
        options=offered_ids, value=offered_ids[0], searchable=True, label="Fruit"
    )
    fruit_picker
    return (fruit_picker,)


@app.cell
def _(fruit_picker, per_fruit, scores_df):
    picked_id = fruit_picker.value

    # Everything we know about this fruit in one row.
    picked_row = (
        scores_df.filter(pl.col("fruit_id") == picked_id)
        .select(
            "row",
            "fruit_id",
            "season",
            "box",
            "storage",
            "hugo_split",
            "file_stem",
            "browning",
            "cavity",
            "PC1",
            "PC2",
            "PC3",
        )
        .join(
            per_fruit.select(
                "fruit_id", "latent_mean", "latent_sd", "latent_min", "latent_max"
            ),
            on="fruit_id",
        )
    )
    picked_row
    return (picked_row,)


@app.cell
def _(cfg, latents, mean_per_cell, middle, picked_row):
    _season = picked_row["season"][0]
    _file = cfg.path(f"ct_dir_{_season}") / f"{picked_row['file_stem'][0]}.nii"
    ct = np.asarray(nib.load(_file).dataobj, dtype=np.float32)
    ct_middle = ct.shape[0] // 2  # 64: the middle of 128 voxels

    # `row` is the position of this fruit in latents.npy (see the manifest).
    picked_latent = latents[picked_row["row"][0]]
    difference = picked_latent - mean_per_cell

    # The difference map uses one colour scale centred on zero, so that red means
    # "higher than the mean fruit" and blue "lower".
    _limit = np.abs(difference).max()

    _fig, _axes = plt.subplots(3, 2, figsize=(8, 12))
    _rows = [
        (ct[ct_middle, :, :].T, ct[:, :, ct_middle], "CT", "gray", None),
        (
            picked_latent[middle, :, :].T,
            picked_latent[:, :, middle],
            "Latent",
            "viridis",
            None,
        ),
        (
            difference[middle, :, :].T,
            difference[:, :, middle],
            "Latent minus mean",
            "RdBu_r",
            _limit,
        ),
    ]
    for _row_axes, (_long, _cross, _name, _cmap, _lim) in zip(_axes, _rows):
        for _ax, _picture, _view in zip(
            _row_axes, [_long, _cross], ["long section", "cross-section"]
        ):
            if _lim is None:
                _image = _ax.imshow(_picture, origin="lower", cmap=_cmap)
            else:
                _image = _ax.imshow(
                    _picture, origin="lower", cmap=_cmap, vmin=-_lim, vmax=_lim
                )
            _ax.set_title(f"{_name}, {_view}")
            _fig.colorbar(_image, ax=_ax, shrink=0.8)
    _fig.suptitle(picked_row["fruit_id"][0])
    _fig.tight_layout()
    _fig
    return


if __name__ == "__main__":
    app.run()
