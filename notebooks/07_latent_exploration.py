import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title="07 Latent exploration")

with app.setup:
    import warnings

    import altair as alt
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
    from sklearn.decomposition import PCA

    from pearlatent.config import load_config
    from pearlatent.exclusions import read_volume_decisions
    from pearlatent.labels import load_labels
    from pearlatent.step0 import (
        MAX_RADIUS,
        N_LATENT_HEIGHT_BINS,
        N_LATENT_RADIUS_BINS,
        RADIUS_METHODS,
        fruit_bin_means,
        group_bin_summary,
        latent_map_sums,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 07 Latent exploration: what organises the latent space?

    **Question.** Which known properties of the fruit (orchard box, season, grade, cause, stalk,
    size, intensity) go with the main structure of the latent, and how strongly?

    **Plan.** `docs/exploration_plan.md` (23 October 2026). t-SNE, UMAP, the MDS map of group
    centres and PERMANOVA with PERMDISP were removed on 26 October 2026 (Ruben): they repeated
    the stalk finding of section 1 and are not used in the thesis.

    **Inputs.** Encoding run `20261013T171306Z_hugo_vae` (`latents.npy`, `manifest.parquet`,
    `reconstruction.parquet`), both label files, `derived/descriptors/20261023T155219Z_descriptors/`,
    `derived/volume_decisions.csv` when it exists.

    **Fruit.** The 346 open 2526 fruit; Hugo's 66 2425 validation fruit for the season
    comparison. Not used: his 528 training fruit (charter rule 2) and all sealed fruit.

    **Outputs.** Figures and tables only.

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
def _(cfg, descriptor_dir, run_dir):
    _manifest = pl.read_parquet(run_dir / "manifest.parquet").with_row_index("position")
    _labels = load_labels(cfg).select("fruit_id", "browning", "cavity")
    _descriptors = pl.read_parquet(descriptor_dir / "descriptors.parquet")
    _reconstruction = pl.read_parquet(run_dir / "reconstruction.parquet").select(
        "fruit_id", "mae_core"
    )
    _all = (
        _manifest.drop("storage")
        .join(
            load_labels(cfg).select("fruit_id", "storage"),
            on="fruit_id",
            validate="1:1",
        )
        .join(_labels, on="fruit_id", validate="1:1")
        .join(_descriptors, on="fruit_id", validate="1:1")
        .join(_reconstruction, on="fruit_id", how="left", validate="1:1")
        .with_columns(defective=pl.max_horizontal("browning", "cavity"))
        .with_columns(
            cause=pl.when(pl.col("defective") == 0)
            .then(pl.lit("none"))
            .when(pl.col("cavity") == 0)
            .then(pl.lit("browning only"))
            .when(pl.col("browning") == 0)
            .then(pl.lit("cavity only"))
            .otherwise(pl.lit("both"))
        )
    )
    # 2526 open fruit and Hugo's 2425 validation fruit (section "Fruit" above).
    _kept = _all.filter(
        (pl.col("season") == "2526") | (pl.col("hugo_split") == "validation")
    )
    print(f"open fruit: {_all.height}; 2526 plus 2425 validation: {_kept.height}")

    _decisions_file = cfg.path("derived_dir") / "volume_decisions.csv"
    if _decisions_file.exists():
        _faulty = read_volume_decisions(_decisions_file).filter(
            pl.col("decision") != "keep"
        )
        fruit = _kept.join(_faulty.select("fruit_id"), on="fruit_id", how="anti")
        print(f"faulty volumes dropped: {_kept.height - fruit.height}")
    else:
        fruit = _kept
        print("no decisions file yet: no faulty volume dropped")
    print(f"fruit in this notebook: {fruit.height}")
    fruit.group_by("season").len().sort("season")
    return (fruit,)


@app.cell(hide_code=True)
def _(fruit, run_dir):
    # Rows of latents.npy in the order of `fruit`, flattened to 32,768 numbers per fruit.
    _latents = np.load(run_dir / "latents.npy", mmap_mode="r")
    flat = np.asarray(_latents[fruit["position"].to_numpy()]).reshape(fruit.height, -1)
    is_2526 = (fruit["season"] == "2526").to_numpy()

    # PCA as in notebook 05: mean-centred, not scaled, exact solver, fitted on the 2526 fruit;
    # every fruit projected.
    _pca = PCA(n_components=50, svd_solver="full").fit(flat[is_2526])
    pc_scores = _pca.transform(flat)
    print(
        f"latents {flat.shape}; PCA fitted on {is_2526.sum()} fruit; scores {pc_scores.shape}"
    )
    return is_2526, pc_scores


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1. What do the first principal components follow?

    Spearman rank correlation of PC1 to PC10 (2526 fruit) with the fruit descriptors and the
    grades. A component that correlates strongly with stalk length or body size describes the
    shape of the fruit in the grid, not its inside.
    """)
    return


@app.cell(hide_code=True)
def _(fruit, is_2526, pc_scores):
    _properties = [
        "stalk_length",
        "body_voxels",
        "body_length",
        "diameter_to_length",
        "tissue_intensity_median",
        "cavity_fraction",
        "flat_face_share",
        "mae_core",
        "browning",
        "cavity",
        "defective",
    ]
    _in_2526 = fruit.filter(pl.Series(is_2526)).with_columns(
        pl.Series(f"PC{i + 1}", pc_scores[is_2526, i]) for i in range(10)
    )
    pc_correlations = pl.DataFrame(
        {
            "property": _properties,
            **{
                f"PC{i + 1}": [
                    round(
                        _in_2526.select(
                            pl.corr(f"PC{i + 1}", p, method="spearman")
                        ).item(),
                        2,
                    )
                    for p in _properties
                ]
                for i in range(10)
            },
        }
    )
    pc_correlations
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2. Score plot PC1 against PC2, four colourings side by side

    The same PCA as in section 1 (fitted on the 2526 fruit after the exclusions). Panels 1 to 3:
    2526 fruit only, coloured by stalk length, body voxels (both in grid voxels, so relative to
    the 128-voxel length of each fruit) and defective grade. Panel 4: Hugo's 2425 validation
    fruit projected onto the same components, coloured by season. For the meeting with Hugo
    (`docs/meeting_2026-10-29.md`, Part 2 item 2): if the colour runs along PC1 in panels 1
    and 2 but not in panel 3, the first component describes the fruit shape in the grid,
    not the defect.
    """)
    return


@app.cell(hide_code=True)
def _(fruit, is_2526, pc_scores):
    _x, _y = pc_scores[:, 0], pc_scores[:, 1]
    _fig, _axes = plt.subplots(1, 4, figsize=(20, 4.8), constrained_layout=True)
    _continuous = [
        ("stalk_length", "stalk length (voxels)"),
        ("body_voxels", "body voxels"),
    ]
    for _ax, (_column, _label) in zip(_axes[:2], _continuous):
        _points = _ax.scatter(
            _x[is_2526],
            _y[is_2526],
            c=fruit.filter(pl.Series(is_2526))[_column].to_numpy(),
            cmap="viridis",
            s=12,
        )
        _fig.colorbar(_points, ax=_ax, label=_label)
        _ax.set_title(f"2526, coloured by {_label}")
    # Defective grade: one colour per grade, light to dark.
    _grade = fruit.filter(pl.Series(is_2526))["defective"].to_numpy()
    for _g, _colour in zip(range(4), ["#c6dbef", "#6baed6", "#2171b5", "#08306b"]):
        _axes[2].scatter(
            _x[is_2526][_grade == _g],
            _y[is_2526][_grade == _g],
            color=_colour,
            s=12,
            label=f"grade {_g}",
        )
    _axes[2].legend(fontsize=8)
    _axes[2].set_title("2526, coloured by defective grade")
    for _season, _colour in [("2526", "lightgrey"), ("2425", "darkorange")]:
        _in = (fruit["season"] == _season).to_numpy()
        _axes[3].scatter(_x[_in], _y[_in], color=_colour, s=12, label=_season)
    _axes[3].legend(fontsize=8)
    _axes[3].set_title("season (2425: Hugo's validation fruit, projected)")
    for _ax in _axes:
        _ax.set_xlabel("PC1")
        _ax.set_ylabel("PC2")
    _fig.suptitle(
        f"PCA of the latent, fitted on {int(is_2526.sum())} 2526 fruit after the exclusions"
    )
    pc_panel_figure = _fig
    pc_panel_figure
    return (pc_panel_figure,)


@app.cell(hide_code=True)
def _():
    save_pc_panel = mo.ui.run_button(label="Save the score plots to derived/figures/")
    save_pc_panel
    return (save_pc_panel,)


@app.cell(hide_code=True)
def _(cfg, pc_panel_figure, save_pc_panel):
    # Saved under derived/ (outside git).
    mo.stop(mo.running_in_notebook() and not save_pc_panel.value)
    _out = cfg.path("derived_dir") / "figures" / "07_pc1_pc2_panels.png"
    pc_panel_figure.savefig(_out, dpi=120, bbox_inches="tight")
    print(f"saved {_out}")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 3. Height-radius spectra of the latent (exploration only)

    **Question.** Folded onto height × radius, does the latent differ between grades? Only to
    look: nothing is fitted or tested here, and bins picked by eye are not a variable
    selection (a later selection starts from the full range, inside the box folds).

    **Binning, per fruit, without labels** (`pearlatent.step0.latent_map_sums`, the same
    function as the latent maps of notebook 06 section 5). Every body cell of the 32³ latent
    `mu` (body mask from `masks.npy`: fruit cells, largest connected piece, stalk slices
    removed, holes filled) gets a height (0 at the calyx end of the body, 1 at the stalk end)
    and a radius. A bin of a fruit holds the mean latent value of its cells. 20 height × 8
    radius bins, about one cell each, no smoothing. No value is interpolated: every cell only
    gets a coordinate. Two radius methods:

    - *one radius per fruit*: distance to one straight stem-calyx axis, divided by the
      distance of the outermost body cell of the fruit. Keeps the shape; outer bins near the
      stalk stay empty.
    - *radius per slice*: distance to the centre of the same slice, divided by the distance of
      the outermost body cell of that slice. Removes the shape; radius 1 is the skin at every
      height.

    Height and radius both run from 0 to 1, as in every height-radius map of notebook 06.

    **Spectrum.** The 160 bins of a fruit in one row, height by height:
    bin = 8 × height bin + radius bin. Bins 0 to 7 are the calyx end from the axis to the
    skin, 152 to 159 the stalk end. Grey vertical lines in the spectrum plot separate the
    height bins; the axis label at each line is the lower height of the bin after it.

    **Selecting, both ways.** Drag a box on the map: the spectra show only the bins whose
    centre lies in the box. Drag along the spectra: the map fades the bins outside that
    stretch. Double-click a view to clear its selection. The spectrum line joins neighbouring
    shown bins, also across a gap.

    **Corrections**, per bin, from the open 2526 grade 0 fruit (in a model they would be
    computed inside the training fold):

    - none;
    - minus the grade 0 mean;
    - minus the grade 0 mean, divided by the grade 0 standard deviation (the difference in
      units of the spread between healthy fruit);
    - minus the grade 0 mean of the fruit's own box (removes differences between boxes,
      that is orchards). Uses the labels of every box: for looking only. In cross-validation
      a held-out box cannot use its own grade 0 fruit, because picking them needs their
      labels; there only the grade 0 mean of the training boxes, or the mean of all fruit of
      a box (no labels), can be subtracted.

    A grade 0 reference exists only in bins where at least half of the grade 0 fruit (of the
    box, for the box correction) have cells; elsewhere the corrected value is empty.

    **Groups** as in notebook 06: grade 0 (defective 0), browning only 1, 2 and 3, cavity 2
    or 3 with any browning. 2526 fruit only. Group value of a bin: the rule of every map in
    notebooks 06 and 07 (`pearlatent.step0.group_bin_summary`): the mean of each fruit over
    its cells in the bin, then the mean over the fruit, so every fruit counts once; drawn
    only when at least half of the fruit of the group have cells in the bin.
    """)
    return


@app.cell(hide_code=True)
def _(fruit, run_dir):
    # 2526 fruit only: the groups below are 2526 groups. `position` is the row of the fruit in
    # latents.npy and masks.npy.
    hr_fruit = fruit.filter(pl.col("season") == "2526")
    _latents = np.load(run_dir / "latents.npy", mmap_mode="r")
    _cells = np.load(run_dir / "masks.npy", mmap_mode="r")
    # Per radius method: (fruit, height, radius) bin means, NaN where the fruit has no cell.
    hr_means = {}
    for _method in RADIUS_METHODS:
        _counts, _sums = [], []
        for _position in hr_fruit["position"].to_list():
            _body = np.asarray(_cells[_position])
            _c, _s = latent_map_sums(np.asarray(_latents[_position]), _body, _method)
            _counts.append(_c)
            _sums.append(_s)
        _counts = np.stack(_counts)
        hr_means[_method] = fruit_bin_means(_counts, np.stack(_sums))
        _empty = (_counts == 0).sum(axis=(1, 2))
        print(
            f"radius per {_method}: {hr_fruit.height} fruit;"
            f" empty bins per fruit median {np.median(_empty):.0f}, max {_empty.max()}"
            f" of {N_LATENT_HEIGHT_BINS * N_LATENT_RADIUS_BINS}"
        )
    return hr_fruit, hr_means


@app.cell(hide_code=True)
def _(hr_fruit, hr_means):
    # Groups as in notebook 06 section 5. Cavity only has 3 fruit in the open 2526 boxes, too
    # few; the last group is cavity 2 or 3 with any browning.
    HR_GROUPS = [
        ("grade 0", pl.col("defective") == 0),
        ("browning 1 only", (pl.col("browning") == 1) & (pl.col("cavity") == 0)),
        ("browning 2 only", (pl.col("browning") == 2) & (pl.col("cavity") == 0)),
        ("browning 3 only", (pl.col("browning") == 3) & (pl.col("cavity") == 0)),
        ("cavity 2-3, any browning", pl.col("cavity") >= 2),
    ]
    hr_group_fruit = {
        _name: hr_fruit.select(_condition).to_series().to_numpy()
        for _name, _condition in HR_GROUPS
    }
    print({_name: int(_in.sum()) for _name, _in in hr_group_fruit.items()})

    # Correction name -> short name for the saved files.
    CORRECTIONS = {
        "none": "none",
        "minus grade 0": "minus_grade0",
        "minus grade 0, divided by grade 0 SD": "grade0_sd",
        "minus grade 0 of the same box": "minus_box_grade0",
    }
    # Group summaries: the shared rule of notebooks 06 and 07 (every fruit counts once; a bin
    # only where at least half of the fruit have cells), `pearlatent.step0.group_bin_summary`.
    group_summary = group_bin_summary
    _healthy = hr_group_fruit["grade 0"]
    _boxes = hr_fruit["box"].to_numpy()

    def corrected_means(method: str, correction: str) -> np.ndarray:
        """Per-fruit bin means (fruit, height, radius) after the correction.

        The grade 0 reference of a bin follows the same rule as every group: it exists only
        where at least half of the grade 0 fruit have cells in the bin, otherwise the
        corrected value is NaN. "Same box": each fruit minus the grade 0 mean of its own box
        (8 to 26 grade 0 fruit per box). It uses the labels of every box, so it is for
        looking only; in cross-validation a held-out box cannot use its own grade 0 fruit.
        """
        _values = hr_means[method]
        if correction == "none":
            return _values
        if correction == "minus grade 0 of the same box":
            _reference = np.full_like(_values, np.nan)
            for _box in np.unique(_boxes):
                _in_box = _boxes == _box
                _reference[_in_box] = group_summary(_values, _healthy & _in_box)["mean"]
            return _values - _reference
        _difference = _values - group_summary(_values, _healthy)["mean"]
        if correction == "minus grade 0":
            return _difference
        # Standard deviation (ddof 1) over the grade 0 fruit with cells in the bin, only
        # where the grade 0 mean exists.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            _spread = np.nanstd(_values[_healthy], axis=0, ddof=1)
        return _difference / _spread

    return CORRECTIONS, corrected_means, group_summary, hr_group_fruit


@app.cell(hide_code=True)
def _(group_summary, hr_group_fruit, hr_means):
    # Comparison of the two radius methods: per method one row. Column 1: share of the 2526
    # fruit with cells in each bin (empty bins). Column 2: grade 0 mean. Then each group minus
    # grade 0, on one symmetric scale per row (99th percentile of the absolute differences).
    _names = list(hr_group_fruit)
    _all = np.ones(next(iter(hr_means.values())).shape[0], dtype=bool)
    _fig, _axes = plt.subplots(
        len(RADIUS_METHODS),
        len(_names) + 1,
        figsize=(2.6 * (len(_names) + 1) + 1, 3.0 * len(RADIUS_METHODS)),
        squeeze=False,
        constrained_layout=True,
    )
    _extent = [0, MAX_RADIUS, 0, 1]
    for _row, _method in enumerate(RADIUS_METHODS):
        _values = hr_means[_method]
        _share = (~np.isnan(_values[_all])).mean(axis=0)
        _healthy = group_summary(_values, hr_group_fruit["grade 0"])["mean"]
        _differences = [
            group_summary(_values, hr_group_fruit[_n])["mean"] - _healthy
            for _n in _names[1:]
        ]
        _d = float(np.nanpercentile(np.abs(np.stack(_differences)), 99))
        _panels = [
            ("share of fruit\nwith cells", _share, "gray", (0, 1)),
            (
                "grade 0 mean",
                _healthy,
                "viridis",
                tuple(np.nanpercentile(_healthy, [1, 99])),
            ),
        ] + [
            (f"{_n}\nminus grade 0", _diff, "RdBu_r", (-_d, _d))
            for _n, _diff in zip(_names[1:], _differences, strict=True)
        ]
        for _col, (_title, _array, _cmap, _limits) in enumerate(_panels):
            _ax = _axes[_row, _col]
            _image = _ax.imshow(
                _array,
                origin="lower",
                extent=_extent,
                aspect="auto",
                cmap=_cmap,
                vmin=_limits[0],
                vmax=_limits[1],
            )
            _ax.set_title(_title, fontsize=9)
            if _col == 0:
                _ax.set_ylabel(f"radius per {_method}\n\nheight (calyx 0, stalk 1)")
            if _row == len(RADIUS_METHODS) - 1:
                _ax.set_xlabel("radius (axis 0, skin 1)")
            if _col < 2 or _col == len(_panels) - 1:
                _fig.colorbar(_image, ax=_ax, shrink=0.8)
    _fig.suptitle(
        "Latent height-radius maps, 2526 fruit: the two radius methods (red = higher value)"
    )
    hr_comparison_figure = _fig
    hr_comparison_figure
    return (hr_comparison_figure,)


@app.cell(hide_code=True)
def _(CORRECTIONS, hr_group_fruit):
    hr_method = mo.ui.dropdown(
        options={"one radius per fruit": "fruit", "radius per slice": "slice"},
        value="one radius per fruit",
        label="Radius",
    )
    hr_correction = mo.ui.dropdown(
        options=list(CORRECTIONS), value="minus grade 0", label="Correction"
    )
    hr_map_group = mo.ui.dropdown(
        options=list(hr_group_fruit), value="browning 3 only", label="Map of group"
    )
    hr_show_fruit = mo.ui.checkbox(label="Show the single fruit of that group")
    mo.hstack([hr_method, hr_correction, hr_map_group, hr_show_fruit], justify="start")
    return hr_correction, hr_map_group, hr_method, hr_show_fruit


@app.cell(hide_code=True)
def _(
    corrected_means,
    group_summary,
    hr_correction,
    hr_group_fruit,
    hr_map_group,
    hr_method,
    hr_show_fruit,
):
    # One chart with the map (left) and the spectra (right), linked both ways in the browser:
    # - drag a box on the map: the spectra show only the bins whose centre lies in the box;
    # - drag along the spectra: the map fades the bins outside that stretch of bin numbers.
    # Both views are one Altair chart because two marimo charts that read each other's
    # selection would make two cells depend on each other, which marimo does not allow.
    # Double-click a view to clear its selection.
    _all_bins = N_LATENT_HEIGHT_BINS * N_LATENT_RADIUS_BINS
    _values = corrected_means(hr_method.value, hr_correction.value)
    _flat = _values.reshape(-1, _all_bins)

    # Bin number and bin centre of every (height, radius) bin; the spectra rows carry the
    # centre too, so the map box can filter them.
    _h, _r = np.meshgrid(
        np.arange(N_LATENT_HEIGHT_BINS), np.arange(N_LATENT_RADIUS_BINS), indexing="ij"
    )
    _coordinates = pl.DataFrame(
        {
            "bin": (_h * N_LATENT_RADIUS_BINS + _r).ravel(),
            "height": (_h.ravel() + 0.5) / N_LATENT_HEIGHT_BINS,
            "radius": (_r.ravel() + 0.5) * MAX_RADIUS / N_LATENT_RADIUS_BINS,
        }
    )
    map_box = alt.selection_interval(name="map_box", encodings=["x", "y"])
    spectrum_range = alt.selection_interval(name="spectrum_range", encodings=["x"])

    # Map: group mean of one group after the correction, each bin drawn at its centre.
    _mean = group_summary(_values, hr_group_fruit[hr_map_group.value])["mean"]
    _map_table = _coordinates.with_columns(value=pl.Series(_mean.ravel())).filter(
        pl.col("value").is_not_nan()
    )
    _limit = float(np.nanpercentile(np.abs(_mean), 99))
    _map_colour = (
        alt.Scale(scheme="viridis")
        if hr_correction.value == "none"
        else alt.Scale(scheme="redblue", reverse=True, domain=[-_limit, _limit])
    )
    _width, _height = 300, 400
    _map = (
        alt.Chart(_map_table)
        .mark_rect(
            width=_width / N_LATENT_RADIUS_BINS, height=_height / N_LATENT_HEIGHT_BINS
        )
        .encode(
            x=alt.X(
                "radius:Q",
                scale=alt.Scale(domain=[0, MAX_RADIUS]),
                title="radius (axis 0, skin 1)",
            ),
            y=alt.Y(
                "height:Q",
                scale=alt.Scale(domain=[0, 1]),
                title="height (calyx 0, stalk 1)",
            ),
            color=alt.Color("value:Q", scale=_map_colour, title=hr_correction.value),
            opacity=alt.condition(spectrum_range, alt.value(1.0), alt.value(0.15)),
            tooltip=["bin", "height", "radius", "value"],
        )
        .add_params(map_box)
        .properties(
            width=_width,
            height=_height,
            title=f"{hr_map_group.value}, radius per {hr_method.value}",
        )
    )

    # Spectra: mean per group and the 25 % to 75 % band, every bin at its own number.
    _rows = []
    for _name, _selected in hr_group_fruit.items():
        _summary = group_summary(_flat, _selected)
        _rows.append(
            pl.DataFrame(
                {
                    "group": _name,
                    "bin": np.arange(_all_bins),
                    "mean": _summary["mean"],
                    "q25": _summary["q25"],
                    "q75": _summary["q75"],
                }
            )
        )
    _spectra = (
        pl.concat(_rows)
        .join(_coordinates, on="bin", validate="m:1")
        .filter(pl.col("mean").is_not_nan())
    )
    # x axis: a tick at the first bin of every height bin, labelled with that bin's lower
    # height; within a height bin the radius runs from the axis to the skin.
    _x = alt.X(
        "bin:Q",
        scale=alt.Scale(domain=[-0.5, _all_bins - 0.5], nice=False),
        axis=alt.Axis(
            values=list(range(0, _all_bins, N_LATENT_RADIUS_BINS)),
            labelExpr=(
                f"format(datum.value / {N_LATENT_RADIUS_BINS} / {N_LATENT_HEIGHT_BINS},"
                " '.2f')"
            ),
            labelAngle=-90,
        ),
        title=(
            "bin; grey lines separate the height bins, label = lower height of the bin;"
            " within a height bin radius runs from axis to skin"
        ),
    )
    _group_colour = alt.Color("group:N", sort=list(hr_group_fruit), title="group")
    _base = alt.Chart(_spectra).transform_filter(map_box).encode(x=_x)
    # Height delimiters: between the last radius bin of one height bin and the first of the
    # next.
    _delimiters = (
        alt.Chart(
            pl.DataFrame(
                {"bin": np.arange(1, N_LATENT_HEIGHT_BINS) * N_LATENT_RADIUS_BINS - 0.5}
            )
        )
        .mark_rule(color="lightgrey", strokeWidth=0.8)
        .encode(x=_x)
    )
    _layers = [
        _delimiters,
        _base.mark_area(opacity=0.15).encode(
            y="q25:Q", y2="q75:Q", color=_group_colour
        ),
        _base.mark_line(point=True)
        .encode(
            y=alt.Y("mean:Q", title=hr_correction.value),
            color=_group_colour,
            tooltip=["group", "bin", "height", "radius", "mean", "q25", "q75"],
        )
        .add_params(spectrum_range),
    ]
    if hr_show_fruit.value:
        _in = hr_group_fruit[hr_map_group.value]
        _single = (
            pl.DataFrame(
                {
                    "fruit": np.repeat(np.flatnonzero(_in), _all_bins),
                    "bin": np.tile(np.arange(_all_bins), int(_in.sum())),
                    "value": _flat[_in].ravel(),
                }
            )
            .join(_coordinates, on="bin", validate="m:1")
            .filter(pl.col("value").is_not_nan())
        )
        _layers.insert(
            1,
            alt.Chart(_single)
            .transform_filter(map_box)
            .mark_line(color="grey", opacity=0.15, strokeWidth=0.7)
            .encode(x=_x, y="value:Q", detail="fruit:N"),
        )
    _spectrum = alt.layer(*_layers).properties(
        width=620,
        height=_height,
        title="Mean spectrum per group, band 25 % to 75 % of the fruit",
    )
    # Map colours and group colours are separate scales. marimo's own selection is off; the
    # two selections above run in the browser, and marimo passes their ranges back to Python
    # in `hr_linked.selections` (used by the PCA below).
    hr_linked = mo.ui.altair_chart(
        alt.hconcat(_map, _spectrum).resolve_scale(color="independent"),
        chart_selection=False,
        legend_selection=False,
    )
    hr_linked
    return (hr_linked,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### PCA on the selected bins

    The bins selected by both drags: inside the box on the map, and inside the stretch
    dragged along the spectra (bin numbers); with both, the bins that satisfy both; with
    neither, all bins. The radius method and correction are the ones chosen above. One row per 2526 fruit, one column per
    selected bin; PCA centred, not scaled (as in section 1). A PCA cannot take empty values,
    so bins that some fruit do not reach are left out (the number is printed); the radius per
    slice leaves far fewer out.

    How to read it:

    - **Circular on purpose.** Bins chosen because the grades differ there will separate the
      grades in this plot almost by construction. A picture for insight, not evidence.
    - **"Minus grade 0" gives the same plot as "none".** PCA centres every column first, and
      subtracting the same grade 0 mean from every fruit in a bin is only a shift. Only the
      box correction (a different shift per box) and the standard deviation option (a
      rescaling per bin) change the result.
    - **The useful question:** do stalk length and body size (panels 1 and 2) still run along
      PC1 in the selected bins, as they do for the full latent in section 2? If yes, the
      selection has not escaped the size and stalk gradient (question 38).
    """)
    return


@app.cell(hide_code=True)
def _(corrected_means, hr_correction, hr_fruit, hr_linked, hr_method):
    def selected_bins(selections: dict) -> np.ndarray:
        """Bin numbers selected by both drags; a drag that is not used selects every bin.

        - `selections["map_box"]`: {"radius": [low, high], "height": [low, high]}, the box on
          the map; a bin is in it when its centre lies inside.
        - `selections["spectrum_range"]`: {"bin": [low, high]}, the stretch along the
          spectra; a bin is in it when its bin number lies inside.
        Both are vega-lite interval selections, sent back by marimo. With both drags, the
        bins that satisfy both are used.
        """
        _all = np.arange(N_LATENT_HEIGHT_BINS * N_LATENT_RADIUS_BINS)
        _keep = np.ones(len(_all), dtype=bool)
        _box = (selections or {}).get("map_box") or {}
        if "radius" in _box and "height" in _box:
            _r_low, _r_high = sorted(_box["radius"])
            _h_low, _h_high = sorted(_box["height"])
            _height = (_all // N_LATENT_RADIUS_BINS + 0.5) / N_LATENT_HEIGHT_BINS
            _radius = (
                (_all % N_LATENT_RADIUS_BINS + 0.5) * MAX_RADIUS / N_LATENT_RADIUS_BINS
            )
            _keep &= (
                (_radius >= _r_low)
                & (_radius <= _r_high)
                & (_height >= _h_low)
                & (_height <= _h_high)
            )
        _range = (selections or {}).get("spectrum_range") or {}
        if "bin" in _range:
            _b_low, _b_high = sorted(_range["bin"])
            _keep &= (_all >= _b_low) & (_all <= _b_high)
        return _all[_keep]

    _bins = selected_bins(hr_linked.selections)
    _values = corrected_means(hr_method.value, hr_correction.value).reshape(
        hr_fruit.height, -1
    )[:, _bins]
    # Keep only bins where every fruit has a value.
    _complete = ~np.isnan(_values).any(axis=0)
    _table = _values[:, _complete]
    print(
        f"bins selected (map box and spectrum stretch): {len(_bins)}; left out (not reached by every fruit):"
        f" {int((~_complete).sum())}; used: {_table.shape[1]};"
        f" fruit: {_table.shape[0]}"
    )
    mo.stop(
        _table.shape[1] < 2,
        mo.md("_Fewer than 2 usable bins selected: select a larger region._"),
    )

    # Up to 20 components for the scree plot (fewer when the box holds fewer bins).
    _pca = PCA(n_components=min(20, *_table.shape), svd_solver="full").fit(_table)
    _scores = _pca.transform(_table)
    _explained = _pca.explained_variance_ratio_

    # Spearman correlation of the first components with shape and grade, as in section 1.
    _properties = ["stalk_length", "body_voxels", "defective", "browning", "cavity"]
    _components = min(3, _scores.shape[1])
    _correlations = pl.DataFrame(
        {
            "property": _properties,
            **{
                f"PC{_k + 1}": [
                    round(
                        pl.DataFrame(
                            {"pc": _scores[:, _k], "p": hr_fruit[_p].to_numpy()}
                        )
                        .select(pl.corr("pc", "p", method="spearman"))
                        .item(),
                        2,
                    )
                    for _p in _properties
                ]
                for _k in range(_components)
            },
        }
    )

    _x, _y = _scores[:, 0], _scores[:, 1]
    _fig, _axes = plt.subplots(1, 4, figsize=(20, 4.8), constrained_layout=True)
    for _ax, (_column, _label) in zip(
        _axes[:2],
        [("stalk_length", "stalk length (voxels)"), ("body_voxels", "body voxels")],
    ):
        _points = _ax.scatter(
            _x, _y, c=hr_fruit[_column].to_numpy(), cmap="viridis", s=12
        )
        _fig.colorbar(_points, ax=_ax, label=_label)
        _ax.set_title(f"coloured by {_label}")
    _grade = hr_fruit["defective"].to_numpy()
    for _g, _colour in zip(range(4), ["#c6dbef", "#6baed6", "#2171b5", "#08306b"]):
        _axes[2].scatter(
            _x[_grade == _g], _y[_grade == _g], color=_colour, s=12, label=f"grade {_g}"
        )
    _axes[2].legend(fontsize=8)
    _axes[2].set_title("coloured by defective grade")
    _cause = hr_fruit["cause"].to_numpy()
    for _c, _colour in zip(
        ["none", "browning only", "cavity only", "both"],
        ["lightgrey", "#d95f02", "#1b9e77", "#7570b3"],
    ):
        _axes[3].scatter(
            _x[_cause == _c], _y[_cause == _c], color=_colour, s=12, label=_c
        )
    _axes[3].legend(fontsize=8)
    _axes[3].set_title("coloured by cause")
    for _ax in _axes:
        _ax.set_xlabel(f"PC1 ({_explained[0]:.0%})")
        _ax.set_ylabel(f"PC2 ({_explained[1]:.0%})")
    _fig.suptitle(
        f"PCA of {_table.shape[1]} selected latent bins, radius per {hr_method.value},"
        f" correction: {hr_correction.value}; {_table.shape[0]} 2526 fruit"
    )
    # Scree plot: share of the variance of the selected bins per component (bars) and the
    # cumulative share (line), for the first components.
    _k = np.arange(1, len(_explained) + 1)
    _scree, _ax_scree = plt.subplots(figsize=(8, 3.6), constrained_layout=True)
    _ax_scree.bar(_k, _explained, color="#6baed6", label="per component")
    _ax_scree.plot(
        _k, np.cumsum(_explained), color="#08306b", marker="o", label="cumulative"
    )
    _ax_scree.set_xticks(_k)
    _ax_scree.set_ylim(0, 1)
    _ax_scree.set_xlabel("principal component")
    _ax_scree.set_ylabel("share of variance")
    _ax_scree.legend(fontsize=8)
    _ax_scree.set_title(
        f"Scree plot, {_table.shape[1]} selected bins: first {len(_k)} components explain"
        f" {np.cumsum(_explained)[-1]:.0%}"
    )
    mo.vstack(
        [
            _fig,
            _scree,
            mo.md("**Spearman correlation with the first components**"),
            _correlations,
        ]
    )
    return


@app.cell(hide_code=True)
def _(corrected_means, group_summary, hr_group_fruit):
    def hr_surface_figure(method: str, correction: str):
        """3D surfaces of the group means over (radius, height), one panel per group, on one
        shared vertical scale. With a grade 0 correction the grade 0 panel is left out (zero
        by definition). Empty bins (NaN) leave holes."""
        _values = corrected_means(method, correction)
        _names = [
            _n for _n in hr_group_fruit if correction == "none" or _n != "grade 0"
        ]
        _means = [group_summary(_values, hr_group_fruit[_n])["mean"] for _n in _names]
        _low = float(np.nanmin(np.stack(_means)))
        _high = float(np.nanmax(np.stack(_means)))
        _r = (np.arange(N_LATENT_RADIUS_BINS) + 0.5) * MAX_RADIUS / N_LATENT_RADIUS_BINS
        _h = (np.arange(N_LATENT_HEIGHT_BINS) + 0.5) / N_LATENT_HEIGHT_BINS
        _grid_r, _grid_h = np.meshgrid(_r, _h)
        _fig = plt.figure(figsize=(3.6 * len(_names), 3.8), constrained_layout=True)
        for _k, (_name, _mean) in enumerate(zip(_names, _means, strict=True)):
            _ax = _fig.add_subplot(1, len(_names), _k + 1, projection="3d")
            _ax.plot_surface(
                _grid_r,
                _grid_h,
                _mean,
                cmap="viridis" if correction == "none" else "RdBu_r",
                vmin=_low if correction == "none" else -max(abs(_low), abs(_high)),
                vmax=_high if correction == "none" else max(abs(_low), abs(_high)),
                edgecolor="none",
            )
            _ax.set_zlim(_low, _high)
            _ax.set_xlabel("radius (axis 0, skin 1)")
            _ax.set_ylabel("height (calyx 0, stalk 1)")
            _ax.set_title(
                f"{_name} (n = {int(hr_group_fruit[_name].sum())})", fontsize=9
            )
        _fig.suptitle(
            f"Latent group means, radius per {method}, correction: {correction}"
        )
        return _fig

    return (hr_surface_figure,)


@app.cell(hide_code=True)
def _(hr_correction, hr_method, hr_surface_figure):
    hr_surface_figure(hr_method.value, hr_correction.value)
    return


@app.cell(hide_code=True)
def _():
    save_hr_figures = mo.ui.run_button(
        label="Save the height-radius figures to derived/figures/"
    )
    save_hr_figures
    return (save_hr_figures,)


@app.cell(hide_code=True)
def _(
    CORRECTIONS,
    cfg,
    hr_comparison_figure,
    hr_surface_figure,
    save_hr_figures,
):
    # Saved under derived/ (outside git): the method comparison and the 3D surfaces for both
    # radius methods with every grade 0 correction.
    mo.stop(mo.running_in_notebook() and not save_hr_figures.value)
    _folder = cfg.path("derived_dir") / "figures"
    _folder.mkdir(parents=True, exist_ok=True)
    hr_comparison_figure.savefig(
        _folder / "07_hr_radius_methods.png", dpi=130, bbox_inches="tight"
    )
    _saved = ["07_hr_radius_methods.png"]
    for _method in RADIUS_METHODS:
        for _correction, _short in list(CORRECTIONS.items())[1:]:
            _name = f"07_hr_surfaces_{_method}_{_short}.png"
            _figure = hr_surface_figure(_method, _correction)
            _figure.savefig(_folder / _name, dpi=130, bbox_inches="tight")
            plt.close(_figure)
            _saved.append(_name)
    print(f"saved {', '.join(_saved)} to {_folder}")
    return


if __name__ == "__main__":
    app.run()
