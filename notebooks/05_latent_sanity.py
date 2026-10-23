import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title="05 Latent sanity checks")

with app.setup:
    import altair as alt
    import marimo as mo
    import matplotlib.pyplot as plt
    import nibabel as nib
    import numpy as np
    import plotly.graph_objects as go
    import polars as pl
    from scipy import stats
    from sklearn.decomposition import PCA

    from pearlatent.config import load_config
    from pearlatent.labels import load_labels
    from pearlatent.pooling import scale_minmax
    from pearlatent.reconstruction import body_centre, body_voxel_mask, central_planes


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
    # Shape, stalk and intensity per fruit (python -m pearlatent.descriptors).
    descriptor_dir = (
        cfg.path("derived_dir") / "descriptors" / "20261023T155219Z_descriptors"
    )
    return cfg, descriptor_dir, run_dir


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1. Load the latents and attach the labels

    The latent of one fruit is a 32 × 32 × 32 grid of numbers (the mean `mu` of Hugo's encoder).
    The n-th latent in `latents.npy` belongs to the n-th row of `manifest.parquet`; the column
    `position` below holds that n.
    """)
    return


@app.cell(hide_code=True)
def _(cfg, descriptor_dir, run_dir):
    latents = np.load(run_dir / "latents.npy")
    # The n-th row of the manifest belongs to the n-th latent. That place is stored as
    # `position`. (The manifest's own column `row` is the row in the list of all 1,093
    # volumes, so it is NOT the place in latents.npy.)
    manifest = pl.read_parquet(run_dir / "manifest.parquet").with_row_index("position")

    labels = load_labels(cfg).select("fruit_id", "browning", "cavity")

    # Shape, stalk and intensity of every fruit (definitions in pearlatent.descriptors).
    descriptors = pl.read_parquet(descriptor_dir / "descriptors.parquet")

    # Attach the grades and the descriptors to every fruit. validate="1:1" stops if a fruit
    # occurs twice.
    fruit = (
        manifest.join(labels, on="fruit_id", how="left", validate="1:1")
        .join(descriptors, on="fruit_id", how="left", validate="1:1")
        .with_columns(defective=pl.max_horizontal("browning", "cavity"))
    )

    # One readable group name per fruit, to filter and colour the plots.
    # 2526: all fruit were scanned after storage. 2425: Hugo's split and the storage group,
    # for example "2425 validation, harvest" (fresh) or "2425 validation, suboptimal_storage".
    fruit = fruit.with_columns(
        group=pl.when(pl.col("season") == "2526")
        .then(pl.lit("2526"))
        .otherwise(pl.format("2425 {}, {}", "hugo_split", "storage"))
    )

    # The cause of the defect (charter rule 5: results on `defective` are split by cause).
    fruit = fruit.with_columns(
        cause=pl.when((pl.col("browning") == 0) & (pl.col("cavity") == 0))
        .then(pl.lit("none"))
        .when(pl.col("cavity") == 0)
        .then(pl.lit("browning only"))
        .when(pl.col("browning") == 0)
        .then(pl.lit("cavity only"))
        .otherwise(pl.lit("both"))
    )

    print(f"latents: {latents.shape[0]} fruit, grid {latents.shape[1:]}")
    print(f"manifest: {manifest.height} fruit; after the join: {fruit.height} fruit")
    print(f"fruit without a grade: {fruit['browning'].null_count()}")
    print(
        f"descriptors from {descriptor_dir.name}; fruit without them: "
        f"{fruit['body_voxels'].null_count()}"
    )
    return descriptors, fruit, latents


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


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
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
    Fruit that are not in the fit still get scores: they are projected onto the same
    components, so 2024–25 fruit can be shown in a PCA fitted on 2025–26.

    Note: the variance is spread over many components (on the 2025–26 fruit: PC1 22 %, 10
    components about 50 %, 50 components about 74 %, 100 components about 84 %; computed on
    21 October 2026). Hugo's VAE has a very small KL weight (1e-6), so its latent is a
    compression of the volume, not a few smooth factors; each cell carries its own local
    detail.
    """)
    return


@app.cell(hide_code=True)
def _():
    pca_fruit_choice = mo.ui.dropdown(
        options=["2025-26 fruit", "all open fruit"],
        value="2025-26 fruit",
        label="Fit the PCA on",
    )
    pca_fruit_choice
    return (pca_fruit_choice,)


@app.cell(hide_code=True)
def _(flat, fruit, pca_fruit_choice):
    if pca_fruit_choice.value == "2025-26 fruit":
        keep = (fruit["season"] == "2526").to_numpy()
    else:
        keep = np.ones(fruit.height, dtype=bool)

    # 50 components are fitted for the scree plot and for section 7 (Hotelling T² and Q).
    # Only the first 10 are shown in the score plot and the parallel coordinates.
    # svd_solver="full": the exact solution, so every run gives the same numbers
    # (the default may pick a randomised solver for a matrix this wide).
    pca = PCA(n_components=50, svd_solver="full")
    pca.fit(flat[keep])
    scores = pca.transform(flat)  # all 940 fruit, also those not in the fit

    pc_names = [f"PC{i + 1}" for i in range(10)]
    scores_df = fruit.with_columns(
        pl.Series(name, scores[:, i]) for i, name in enumerate(pc_names)
    ).with_columns(in_pca_fit=pl.Series(keep))
    print(f"PCA fitted on {keep.sum()} of {fruit.height} fruit")
    return keep, pc_names, pca, scores, scores_df


@app.cell(hide_code=True)
def _(pca):
    # Scree plot: bars for each component, line for the running total.
    explained = pl.DataFrame(
        {
            "component": np.arange(1, pca.n_components_ + 1),
            "explained_percent": 100 * pca.explained_variance_ratio_,
        }
    ).with_columns(cumulative_percent=pl.col("explained_percent").cum_sum())

    _x = alt.X("component:Q", title="Component")
    _bars = (
        alt.Chart(explained)
        .mark_bar()
        .encode(
            x=_x,
            y=alt.Y("explained_percent:Q", title="Explained variance (%)"),
            tooltip=["component", "explained_percent", "cumulative_percent"],
        )
    )
    _total = (
        alt.Chart(explained)
        .mark_line(color="black", point=True)
        .encode(x=_x, y=alt.Y("cumulative_percent:Q", title="Cumulative (%)"))
    )
    alt.layer(_bars, _total).resolve_scale(y="independent").properties(
        width=600, height=250
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 5. Score plot and parallel coordinates

    Choose which groups to show, then drag a rectangle in the score plot to select fruit.

    The parallel-coordinates plot below shows every shown fruit as one line over the group,
    the defective grade and PC1 to PC10. Every axis has its own scale. Drag along an axis to
    keep only the lines that pass through that range; this works on several axes at once.
    Click next to a selected range on the axis to remove it. Fruit selected in the score plot
    are drawn in red.
    """)
    return


@app.cell(hide_code=True)
def _(fruit, pc_names):
    all_groups = sorted(fruit["group"].unique().to_list())
    # Default: everything except Hugo's 528 training fruit, which the encoder has seen.
    group_choice = mo.ui.multiselect(
        options=all_groups,
        value=[g for g in all_groups if "train" not in g],
        label="Show groups",
    )
    grade_choice = mo.ui.multiselect(
        options=["0", "1", "2", "3"],
        value=["0", "1", "2", "3"],
        label="Show defective grades",
    )
    cause_choice = mo.ui.multiselect(
        options=["none", "browning only", "cavity only", "both"],
        value=["none", "browning only", "cavity only", "both"],
        label="Show causes",
    )

    _label_columns = [
        "group",
        "box",
        "storage",
        "defective",
        "cause",
        "browning",
        "cavity",
        "stalk_present",
    ]
    # Measured numbers, drawn with a continuous colour scale (pearlatent.descriptors).
    continuous_columns = [
        "body_voxels",
        "body_length",
        "stalk_length",
        "diameter_to_length",
        "tissue_intensity_median",
        "cavity_fraction",
        "flat_face_share",
    ]
    colour_choice = mo.ui.dropdown(
        options=_label_columns + continuous_columns, value="group", label="Colour by"
    )
    shape_choice = mo.ui.dropdown(
        options=["nothing", *_label_columns], value="defective", label="Symbol by"
    )
    x_choice = mo.ui.dropdown(options=pc_names, value="PC1", label="x axis")
    y_choice = mo.ui.dropdown(options=pc_names, value="PC2", label="y axis")
    mo.vstack(
        [
            group_choice,
            mo.hstack([grade_choice, cause_choice], justify="start"),
            mo.hstack(
                [colour_choice, shape_choice, x_choice, y_choice], justify="start"
            ),
        ]
    )
    return (
        cause_choice,
        colour_choice,
        continuous_columns,
        grade_choice,
        group_choice,
        shape_choice,
        x_choice,
        y_choice,
    )


@app.cell(hide_code=True)
def _(cause_choice, grade_choice, group_choice, scores_df):
    # The grades are chosen as text in the multiselect; turn them back into numbers.
    _grades = [int(g) for g in grade_choice.value]

    shown_df = scores_df.filter(
        pl.col("group").is_in(group_choice.value),
        pl.col("defective").is_in(_grades),
        pl.col("cause").is_in(cause_choice.value),
    )
    print(f"showing {shown_df.height} of {scores_df.height} fruit")
    return (shown_df,)


@app.cell(hide_code=True)
def _(
    colour_choice,
    continuous_columns,
    shape_choice,
    shown_df,
    x_choice,
    y_choice,
):
    # A measured number gets a continuous scale (viridis), a label one colour per value.
    if colour_choice.value in continuous_columns:
        _colour = alt.Color(
            f"{colour_choice.value}:Q", scale=alt.Scale(scheme="viridis")
        )
    else:
        _colour = alt.Color(f"{colour_choice.value}:N")
    _encodings = dict(
        x=alt.X(f"{x_choice.value}:Q"),
        y=alt.Y(f"{y_choice.value}:Q"),
        color=_colour,
        tooltip=[
            "fruit_id",
            "group",
            "box",
            "defective",
            "cause",
            "browning",
            "cavity",
            "stalk_length",
            "body_voxels",
        ],
    )
    if shape_choice.value != "nothing":
        _encodings["shape"] = alt.Shape(f"{shape_choice.value}:N")

    _chart = (
        alt.Chart(shown_df)
        .mark_point(filled=True, size=50)
        .encode(**_encodings)
        .properties(width=500, height=400)
    )
    score_plot = mo.ui.altair_chart(_chart)
    score_plot
    return (score_plot,)


@app.cell
def _(score_plot):
    selected_ids = score_plot.value["fruit_id"].to_list()
    mo.md(f"**{len(selected_ids)} fruit selected in the score plot.**")
    return (selected_ids,)


@app.cell(hide_code=True)
def _(colour_choice, continuous_columns, pc_names, selected_ids, shown_df):
    def as_codes(values: pl.Series) -> tuple[list[int], list[str]]:
        """Turn a column of categories into numbers 0, 1, 2, ... for plotting.

        Plotly's parallel coordinates only take numbers. Returns the number of every fruit
        and the list of category names, so the axis can show the names again.
        """
        names = sorted(str(v) for v in values.unique().to_list())
        codes = [names.index(str(v)) for v in values.to_list()]
        return codes, names

    # Two label axes first, so they can be used as filters, then one axis per component.
    _group_codes, _group_names = as_codes(shown_df["group"])
    _axes = [
        dict(
            label="group",
            values=_group_codes,
            tickvals=list(range(len(_group_names))),
            ticktext=_group_names,
        ),
        dict(
            label="defective",
            values=shown_df["defective"].to_list(),
            tickvals=[0, 1, 2, 3],
        ),
    ]
    _axes += [dict(label=name, values=shown_df[name].to_list()) for name in pc_names]

    # Line colour: if fruit are selected in the score plot, red for those and grey for the
    # rest; otherwise the "Colour by" choice.
    if selected_ids:
        _colour_codes = shown_df["fruit_id"].is_in(selected_ids).cast(pl.Int8).to_list()
        _colour_names = ["not selected", "selected"]
        _colourscale = [[0, "lightgrey"], [1, "crimson"]]
    elif colour_choice.value in continuous_columns:
        # A measured number: the line colour is the number itself.
        _colour_codes = shown_df[colour_choice.value].to_list()
        _colour_names = None
        _colourscale = "Viridis"
    else:
        _colour_codes, _colour_names = as_codes(shown_df[colour_choice.value])
        _colourscale = "Turbo"

    _figure = go.Figure(
        go.Parcoords(
            dimensions=_axes,
            line=dict(
                color=_colour_codes,
                colorscale=_colourscale,
                showscale=True,
                colorbar=dict(title=colour_choice.value)
                if _colour_names is None
                else dict(
                    tickvals=list(range(len(_colour_names))), ticktext=_colour_names
                ),
            ),
        )
    )
    _figure.update_layout(height=450, margin=dict(l=80, r=80, t=60, b=30))
    _figure
    return


@app.cell(hide_code=True)
def _(scores_df, selected_ids):
    # The selected fruit as a table, to look them up.
    scores_df.filter(pl.col("fruit_id").is_in(selected_ids)).select(
        "fruit_id",
        "group",
        "box",
        "defective",
        "cause",
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
    fruit of the groups shown in section 5. Type in the box to search by name.
    """)
    return


@app.cell(hide_code=True)
def _(selected_ids, shown_df):
    if selected_ids:
        offered_ids = selected_ids
    else:
        offered_ids = shown_df["fruit_id"].to_list()
    mo.stop(
        not offered_ids,
        mo.md("_No fruit shown: choose at least one group in section 5._"),
    )

    fruit_picker = mo.ui.dropdown(
        options=offered_ids, value=offered_ids[0], searchable=True, label="Fruit"
    )
    fruit_picker
    return (fruit_picker,)


@app.cell(hide_code=True)
def _(fruit_picker, per_fruit, scores_df):
    picked_id = fruit_picker.value

    # Everything we know about this fruit in one row.
    picked_row = (
        scores_df.filter(pl.col("fruit_id") == picked_id)
        .select(
            "position",
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


@app.cell(hide_code=True)
def _(cfg, latents, mean_per_cell, middle, picked_row):
    _season = picked_row["season"][0]
    _file = cfg.path(f"ct_dir_{_season}") / f"{picked_row['file_stem'][0]}.nii"
    ct = np.asarray(nib.load(_file).dataobj, dtype=np.float32)
    ct_middle = ct.shape[0] // 2  # 64: the middle of 128 voxels

    # `position` is the place of this fruit in latents.npy (see section 1).
    picked_latent = latents[picked_row["position"][0]]
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


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 7. Outlier screen: Hotelling T² and Q residuals

    Two numbers per fruit, from the PCA of section 4 with the first A components:

    - **Hotelling T²**: how far the fruit lies from the centre *within* the model (unusually large
      scores). T² = Σ (score²ₐ / varianceₐ) over the A components.
    - **Q residual** (squared prediction error): how much of the fruit the model does *not*
      describe. Q = Σ (centred value − value rebuilt from A components)² over the 32,768 cells.

    The limits are computed from the fruit the PCA was fitted on, at 95 % and 99 %:
    T² limit = A(n − 1)/(n − A) · F(A, n − A); Q limit from the approximation of Box (1954),
    Q ≈ g·χ²(h) with g = variance(Q) / (2·mean(Q)) and h = 2·mean(Q)² / variance(Q)
    (as in Nomikos and MacGregor 1995). Fruit not in the fit are only compared with those limits.

    **This is a screen, not a rule.** Nothing is removed here. The rule (number of components,
    limit, and what happens to a flagged fruit) is written down before any fruit is removed.
    With only about 50 % of the variance in 10 components, Q is large for every fruit; the
    question is which fruit are large compared with the others.
    """)
    return


@app.cell
def _():
    n_model_components = mo.ui.slider(
        start=1,
        stop=50,
        # 13: the elbow of the scree plot read by Ruben on 21 October 2026.
        value=13,
        label="Components A in the outlier model",
        show_value=True,
    )
    n_model_components
    return (n_model_components,)


@app.cell(hide_code=True)
def _(flat, keep, n_model_components, pca, scores, shown_df):
    _A = n_model_components.value
    _n = int(keep.sum())  # fruit the PCA was fitted on

    # Hotelling T²: squared scores, each divided by the variance of its component.
    _t_scores = scores[:, :_A]
    t_squared = (_t_scores**2 / pca.explained_variance_[:_A]).sum(axis=1)

    # Q: rebuild every fruit from A components and sum the squared differences.
    _centred = flat - pca.mean_
    _rebuilt = _t_scores @ pca.components_[:_A]
    q_residual = ((_centred - _rebuilt) ** 2).sum(axis=1)

    # Limits from the fruit in the fit only.
    def t_squared_limit(level):
        return _A * (_n - 1) / (_n - _A) * stats.f.ppf(level, _A, _n - _A)

    def q_limit(level):
        _q_fit = q_residual[keep]
        _g = _q_fit.var() / (2 * _q_fit.mean())
        _h = 2 * _q_fit.mean() ** 2 / _q_fit.var()
        return _g * stats.chi2.ppf(level, _h)

    limits = {
        "T² 95 %": t_squared_limit(0.95),
        "T² 99 %": t_squared_limit(0.99),
        "Q 95 %": q_limit(0.95),
        "Q 99 %": q_limit(0.99),
    }

    # Attach to the fruit table, keep only the fruit shown (filters of section 5).
    outlier_df = (
        pl.DataFrame(
            {
                "position": np.arange(len(flat)),
                "t_squared": t_squared,
                "q_residual": q_residual,
            }
        )
        .with_columns(pl.col("position").cast(pl.UInt32))
        .join(
            shown_df.select(
                "position", "fruit_id", "group", "box", "defective", "cause"
            ),
            on="position",
        )
        .with_columns(
            _t_beyond=pl.col("t_squared") > limits["T² 99 %"],
            _q_beyond=pl.col("q_residual") > limits["Q 99 %"],
        )
        .with_columns(
            beyond_99=pl.col("_t_beyond") | pl.col("_q_beyond"),
            # Which 99 % limit the fruit lies beyond: T² (unusual within the model), Q (not
            # described by the model), or both.
            limit_hit=pl.when(pl.col("_t_beyond") & pl.col("_q_beyond"))
            .then(pl.lit("T² and Q"))
            .when(pl.col("_t_beyond"))
            .then(pl.lit("T²"))
            .when(pl.col("_q_beyond"))
            .then(pl.lit("Q"))
            .otherwise(pl.lit("none")),
        )
        .drop("_t_beyond", "_q_beyond")
    )
    print({name: round(float(value), 1) for name, value in limits.items()})
    return limits, outlier_df


@app.cell(hide_code=True)
def _(colour_choice, limits, outlier_df):
    _colour = (
        colour_choice.value if colour_choice.value in outlier_df.columns else "group"
    )
    _points = (
        alt.Chart(outlier_df)
        .mark_point(filled=True, size=40)
        .encode(
            x=alt.X("t_squared:Q", title="Hotelling T²"),
            y=alt.Y("q_residual:Q", title="Q residual"),
            color=alt.Color(f"{_colour}:N"),
            tooltip=[
                "fruit_id",
                "group",
                "box",
                "defective",
                "cause",
                "t_squared",
                "q_residual",
            ],
        )
    )
    # Dashed lines: 95 % limits; solid lines: 99 % limits.
    _limit_lines = [
        alt.Chart().mark_rule(strokeDash=[4, 4]).encode(x=alt.datum(limits["T² 95 %"])),
        alt.Chart().mark_rule().encode(x=alt.datum(limits["T² 99 %"])),
        alt.Chart().mark_rule(strokeDash=[4, 4]).encode(y=alt.datum(limits["Q 95 %"])),
        alt.Chart().mark_rule().encode(y=alt.datum(limits["Q 99 %"])),
    ]
    alt.layer(_points, *_limit_lines).properties(width=550, height=400)
    return


@app.cell
def _(outlier_df):
    # Fruit beyond a 99 % limit, by group. With 1 % false alarms, about 1 in 100 fruit
    # lies beyond each limit by chance alone.
    mo.vstack(
        [
            outlier_df.group_by("group")
            .agg(n_fruit=pl.len(), beyond_99=pl.col("beyond_99").sum())
            .sort("group"),
            outlier_df.filter(pl.col("beyond_99")).sort("q_residual", descending=True),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 8. The flagged fruit, three slices each

    Every shown fruit beyond **either** 99 % limit of section 7, tagged with the limit it lies
    beyond (T², Q, or both). For each: three slices of the CT volume through the centre of the
    fruit body (the mean position of the body voxels, the same centre as `planes.npy` uses):
    two long sections (stem–calyx axis vertical, calyx at the bottom) and one cross-section.

    **Purpose.** Ruben decides by eye, per fruit, whether the volume is faulty (scan artefact,
    wrong preprocessing) or the fruit is deformed in a way a sorting line would reject. The
    decision is about the volume, never about the defect grade; the grade table below shows
    whether the flagged fruit are mostly of one grade. Nothing is removed in this notebook.
    """)
    return


@app.cell
def _(fruit, outlier_df):
    # Grades of the flagged fruit next to the grades of all shown fruit of the same group.
    # If the flagged fruit were mostly grade 3, removing them would remove mostly defects.
    _shown = outlier_df.select("fruit_id", "group", "beyond_99").join(
        fruit.select("fruit_id", "defective"), on="fruit_id", validate="1:1"
    )
    _shown.group_by("group", "defective").agg(
        all_shown=pl.len(), flagged=pl.col("beyond_99").sum()
    ).sort("group", "defective")
    return


@app.cell
def _(cfg, fruit, outlier_df, run_dir):
    flagged = (
        outlier_df.filter(pl.col("beyond_99"))
        .join(
            fruit.select("fruit_id", "season", "file_stem", "browning", "cavity"),
            on="fruit_id",
            validate="1:1",
        )
        .sort("group", "fruit_id")
    )
    # Body masks at 32³, one per fruit, in the same order as latents.npy (`position`).
    _body_cells = np.load(run_dir / "masks.npy", mmap_mode="r")

    def ct_planes(row: dict) -> np.ndarray:
        """Three CT planes (3, 128, 128) through the centre of the fruit body.

        In: one row of `flagged`. The volume is min-max scaled as for the encoder, so the
        grey values match what the encoder saw. Out: planes with axis 0, 1 and 2 held fixed.
        """
        _file = cfg.path(f"ct_dir_{row['season']}") / f"{row['file_stem']}.nii"
        _scaled = scale_minmax(np.asarray(nib.load(_file).dataobj))
        _body = body_voxel_mask(_scaled, np.asarray(_body_cells[row["position"]]))
        return central_planes(_scaled, body_centre(_body))

    flagged_planes = {
        row["fruit_id"]: ct_planes(row) for row in flagged.iter_rows(named=True)
    }
    print(f"flagged fruit (beyond either 99 % limit): {flagged.height}")
    return flagged, flagged_planes


@app.cell
def _():
    # Off by default: the decision on a volume is made without seeing its grades
    # (docs/outlier_rule.md, section 3).
    show_grades = mo.ui.checkbox(value=False, label="Show grades in the panel titles")
    show_grades
    return (show_grades,)


@app.cell(hide_code=True)
def _(flagged, flagged_planes, n_model_components, show_grades):
    # Two fruit per row, three slices per fruit.
    _per_row = 2
    _n_rows = max(1, -(-flagged.height // _per_row))  # rounded up
    _fig, _axes = plt.subplots(
        _n_rows, 3 * _per_row, figsize=(15, 2.8 * _n_rows), squeeze=False
    )
    for _ax in _axes.flat:
        _ax.axis("off")
    _views = [
        (0, True, "long section 1"),  # (y, z): transposed so the z axis is vertical
        (1, True, "long section 2"),  # (x, z)
        (2, False, "cross-section"),  # (x, y)
    ]
    for _k, _row in enumerate(flagged.iter_rows(named=True)):
        _planes = flagged_planes[_row["fruit_id"]]
        for _v, (_plane, _transpose, _view) in enumerate(_views):
            _ax = _axes[_k // _per_row, 3 * (_k % _per_row) + _v]
            _picture = _planes[_plane].T if _transpose else _planes[_plane]
            # Fixed grey scale 0 to 1 (the scaled range), so fruit can be compared.
            _ax.imshow(_picture, origin="lower", cmap="gray", vmin=0, vmax=1)
            _ax.set_title(_view, fontsize=8)
            if _v == 0:
                _ax.text(
                    0,
                    1.18,
                    f"{_row['fruit_id']} · {_row['group']} · beyond {_row['limit_hit']}"
                    + (
                        f" · browning {_row['browning']}, cavity {_row['cavity']}"
                        if show_grades.value
                        else ""
                    ),
                    transform=_ax.transAxes,
                    fontsize=9,
                    fontweight="bold",
                )
    # Fixed margins instead of tight_layout, which fails with the fruit names above the rows.
    _height = 2.8 * _n_rows
    _fig.subplots_adjust(
        top=1 - 0.9 / _height, bottom=0.02, left=0.01, right=0.99, hspace=0.5
    )
    _fig.suptitle(
        f"Fruit beyond a 99 % limit (A = {n_model_components.value} components)",
        y=1 - 0.2 / _height,
    )
    outlier_figure = _fig
    outlier_figure
    return (outlier_figure,)


@app.cell
def _():
    save_outlier_figure = mo.ui.run_button(label="Save the panel to derived/figures/")
    save_outlier_figure
    return (save_outlier_figure,)


@app.cell
def _(cfg, outlier_figure, save_outlier_figure):
    # Saved under derived/ (outside git): CT images stay off GitHub until Hugo agrees.
    mo.stop(mo.running_in_notebook() and not save_outlier_figure.value)
    _out = cfg.path("derived_dir") / "figures" / "05_outlier_panel.png"
    _out.parent.mkdir(parents=True, exist_ok=True)
    outlier_figure.savefig(_out, dpi=130, bbox_inches="tight")
    print(f"saved {_out}")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 9. Contact sheet of every shown fruit, and the descriptor screen

    The PCA screen of sections 7 and 8 does not catch every faulty volume (23 October 2026: dark
    volumes and volumes with a stray object were not flagged). So every shown fruit is looked at
    once (`docs/outlier_rule.md`, section 2): a long section and a cross-section through the
    centre of the body, from `planes.npy` of the encoding run (CT before encoding, min-max scaled,
    fixed grey scale 0 to 1). Only the fruit identifier is shown, never a grade. Order: by fruit
    identifier, 48 fruit per page.

    Third route to a candidate, added on 23 October after the descriptors were first computed:
    a fruit whose median tissue intensity is far below the others (below 0.7, against a median
    of about 0.87; failed intensity scaling) or with more than 100 voxels of a stray object
    above the threshold (`pearlatent.descriptors`). These fruit are listed in the table below
    the sheet and marked with a red frame on it.
    """)
    return


@app.cell
def _(descriptors, run_dir):
    # Planes through the body centre, in the order of `recon_row` of the manifest:
    # (fruit, original or reconstruction, plane, 128, 128).
    planes = np.load(run_dir / "planes.npy", mmap_mode="r")
    print(
        f"descriptors: {descriptors.height} fruit; planes: {planes.shape[0]} decoded fruit"
    )
    return (planes,)


@app.cell
def _(descriptors, fruit, shown_df):
    # Thresholds of the third route (see the text above), fixed on 23 October 2026.
    DARK_TISSUE = 0.7
    STRAY_VOXELS = 100

    sheet_fruit = (
        shown_df.select("fruit_id", "group")
        .join(fruit.select("fruit_id", "recon_row"), on="fruit_id", validate="1:1")
        .join(descriptors, on="fruit_id", validate="1:1")
        .with_columns(
            descriptor_flag=pl.when(pl.col("tissue_intensity_median") < DARK_TISSUE)
            .then(pl.lit("dark tissue"))
            .when(pl.col("stray_voxels") > STRAY_VOXELS)
            .then(pl.lit("stray object"))
            .otherwise(pl.lit(""))
        )
        .sort("fruit_id")
    )
    _missing = sheet_fruit.filter(pl.col("recon_row").is_null()).height
    print(
        f"contact sheet: {sheet_fruit.height} fruit;"
        f" without decoded planes (not shown): {_missing}"
    )
    sheet_fruit = sheet_fruit.filter(pl.col("recon_row").is_not_null())

    FRUIT_PER_PAGE = 48
    n_pages = -(-sheet_fruit.height // FRUIT_PER_PAGE)  # rounded up
    page_choice = mo.ui.slider(
        start=1, stop=max(n_pages, 1), value=1, label="Page", show_value=True
    )
    page_choice
    return FRUIT_PER_PAGE, n_pages, page_choice, sheet_fruit


@app.cell
def _(FRUIT_PER_PAGE, planes, sheet_fruit):
    def contact_page(page: int):
        """One page of the contact sheet: 6 rows of 8 fruit, two slices per fruit.

        In: the page number, counting from 1. Out: a Matplotlib figure. Long section: plane 0
        (axis 0 fixed), transposed so the stem-calyx axis is vertical, calyx at the bottom.
        Cross-section: plane 2 (the third axis fixed).
        """
        _rows = sheet_fruit.slice((page - 1) * FRUIT_PER_PAGE, FRUIT_PER_PAGE)
        _per_row = 8
        _fig, _axes = plt.subplots(6, 2 * _per_row, figsize=(16, 6 * 1.25))
        for _ax in _axes.flat:
            _ax.axis("off")
        for _k, _row in enumerate(_rows.iter_rows(named=True)):
            _original = np.asarray(planes[_row["recon_row"], 0])
            _r, _c = divmod(_k, _per_row)
            for _j, _picture in enumerate([_original[0].T, _original[2]]):
                _ax = _axes[_r, 2 * _c + _j]
                _ax.imshow(_picture, origin="lower", cmap="gray", vmin=0, vmax=1)
                if _row["descriptor_flag"]:
                    # A red frame marks the descriptor screen; axis("off") hides spines,
                    # so draw a rectangle instead.
                    _ax.add_patch(
                        plt.Rectangle(
                            (0, 0), 127, 127, fill=False, edgecolor="red", linewidth=2
                        )
                    )
            _axes[_r, 2 * _c].set_title(_row["fruit_id"], fontsize=8, loc="left")
        _fig.subplots_adjust(
            left=0.005, right=0.995, top=0.96, bottom=0.005, wspace=0.03, hspace=0.3
        )
        return _fig

    return (contact_page,)


@app.cell
def _(contact_page, page_choice):
    contact_page(page_choice.value)
    return


@app.cell
def _(sheet_fruit):
    # The fruit of the third route, with the numbers that put them there.
    sheet_fruit.filter(pl.col("descriptor_flag") != "").select(
        "fruit_id",
        "group",
        "descriptor_flag",
        "tissue_intensity_median",
        "stray_voxels",
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Flat-face screen (`docs/outlier_rule.md`, section 2, route 3)

    The largest flat patch of the fruit surface as a share of the whole surface, for every
    fruit of the contact sheet. No labels are shown. Choose the cut-off on this histogram
    before judging the candidates; the fruit at or above it are listed below and judged by eye.
    The chosen value is written into `docs/outlier_rule.md`.
    """)
    return


@app.cell
def _():
    flat_face_cutoff = mo.ui.slider(
        start=0.08,
        stop=0.20,
        step=0.005,
        value=0.11,
        label="Flat-face cut-off",
        show_value=True,
    )
    flat_face_cutoff
    return (flat_face_cutoff,)


@app.cell
def _(flat_face_cutoff, sheet_fruit):
    _histogram = (
        alt.Chart(sheet_fruit)
        .mark_bar()
        .encode(
            x=alt.X(
                "flat_face_share:Q",
                bin=alt.Bin(step=0.005),
                title="largest flat face / surface",
            ),
            y=alt.Y("count():Q", title="fruit"),
        )
    )
    _cutoff_line = (
        alt.Chart().mark_rule(color="red").encode(x=alt.datum(flat_face_cutoff.value))
    )
    _above = sheet_fruit.filter(
        pl.col("flat_face_share") >= flat_face_cutoff.value
    ).sort("flat_face_share", descending=True)
    mo.vstack(
        [
            alt.layer(_histogram, _cutoff_line).properties(width=600, height=220),
            mo.md(f"**{_above.height} fruit at or above {flat_face_cutoff.value}**"),
            _above.select("fruit_id", "group", "flat_face_share", "flat_face_voxels"),
        ]
    )
    return


@app.cell
def _():
    save_contact_sheet = mo.ui.run_button(
        label="Save every page of the contact sheet to derived/figures/"
    )
    save_contact_sheet
    return (save_contact_sheet,)


@app.cell
def _(cfg, contact_page, n_pages, save_contact_sheet):
    # Saved under derived/ (outside git), one PNG per page.
    mo.stop(mo.running_in_notebook() and not save_contact_sheet.value)
    _folder = cfg.path("derived_dir") / "figures" / "05_contact_sheet"
    _folder.mkdir(parents=True, exist_ok=True)
    for _page in range(1, n_pages + 1):
        _fig = contact_page(_page)
        _fig.savefig(_folder / f"page_{_page:02d}.png", dpi=110)
        plt.close(_fig)
    print(f"saved {n_pages} pages to {_folder}")
    return


if __name__ == "__main__":
    app.run()
