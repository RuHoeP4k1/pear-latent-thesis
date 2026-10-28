import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title="07 Latent exploration")

with app.setup(hide_code=True):
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
    `reconstruction.parquet`, `masks.npy`), both label files, `derived/descriptors/20261023T155219Z_descriptors/`,
    `derived/volume_decisions.csv` when it exists. For the comparison rows (charter, Methods,
    point 3): random encoder run `20261020T120336Z_random_init` (`latents.npy`) and CT run
    `20261005T142026Z_raw_ct_32` (`grids.npy`).

    **Representation.** Chosen with the dropdown below; every section uses it. Where the text
    says "latent", read "the chosen representation". The fruit, labels and body masks are the
    same for all three (the CT run's body masks are identical to those of Hugo's run for all
    940 open fruit, checked 28 October 2026).

    **Fruit.** The 346 open 2526 fruit; Hugo's 66 2425 validation fruit for the season
    comparison. Not used: his 528 training fruit (charter rule 2) and all sealed fruit.

    **Outputs.** Figures and tables only.

    **Status.** draft
    """)
    return


@app.cell(hide_code=True)
def _():
    # The three 32³ representations compared in the charter (Methods, point 3); the 128³
    # input and the ResNet-18 are not 32³ grids and are not shown here.
    REPRESENTATIONS = {
        "hugo_vae": "Hugo's latent (mu)",
        "random_init": "random encoder (same architecture, seed 0)",
        "raw_ct_32": "CT averaged to 32³ (mean of 4 × 4 × 4 voxel blocks)",
    }
    representation_choice = mo.ui.dropdown(
        options={_label: _key for _key, _label in REPRESENTATIONS.items()},
        value=REPRESENTATIONS["hugo_vae"],
        label="Representation",
    )
    mo.vstack(
        [
            representation_choice,
            mo.md(
                "_Values are in different units (CT intensity 0 to 1, the latents in their"
                " own units). To compare representations, use the correction 'minus grade 0,"
                " divided by grade 0 SD' in sections 3 and 4._"
            ),
        ]
    )
    return REPRESENTATIONS, representation_choice


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
def _(cfg, run_dir):
    def grid_reader(representation: str):
        """A function that reads the 32³ grids of one representation.

        In: the representation key ("hugo_vae", "random_init" or "raw_ct_32"). Out: a
        function `read(positions)` that takes rows of Hugo's `manifest.parquet` (the
        `position` column of `fruit`) and returns the grids of those fruit, in the same
        order, as (fruit, 32, 32, 32) float32. For the CT the position is first translated
        to the row of the same fruit in the CT run. Defined apart from the dropdown so that
        the table at the end of section 1 can read all three representations.
        """
        _derived = cfg.path("derived_dir")
        _manifest = pl.read_parquet(run_dir / "manifest.parquet").select("fruit_id")
        if representation == "hugo_vae":
            _grids = np.load(run_dir / "latents.npy", mmap_mode="r")
            _rows = np.arange(_manifest.height)
        elif representation == "random_init":
            _dir = _derived / "latents" / "random_init" / "20261020T120336Z_random_init"
            # Same fruit in the same order as Hugo's run: checked here, not assumed.
            _other = pl.read_parquet(_dir / "manifest.parquet").select("fruit_id")
            if not _other["fruit_id"].equals(_manifest["fruit_id"]):
                raise ValueError("random encoder manifest differs from Hugo's run")
            _grids = np.load(_dir / "latents.npy", mmap_mode="r")
            _rows = np.arange(_manifest.height)
        else:
            _dir = _derived / "raw_ct_32" / "20261005T142026Z_raw_ct_32"
            # This run holds all 1,093 fruit, the sealed test fruit included (it was made
            # on 5 October, before the test boxes were sealed). Only the rows of the open
            # fruit of Hugo's run are looked up; the sealed rows are never read.
            _ct = pl.read_parquet(_dir / "manifest.parquet").with_row_index("ct_row")
            _joined = (
                _manifest.with_row_index("position")
                .join(_ct.select("fruit_id", "ct_row"), on="fruit_id", validate="1:1")
                .sort("position")
            )
            if _joined.height != _manifest.height:
                raise ValueError("some open fruit are missing from the CT run")
            print(
                f"CT run: {_ct.height} fruit; open fruit looked up: {_joined.height};"
                f" never read: {_ct.height - _joined.height}"
            )
            _grids = np.load(_dir / "grids.npy", mmap_mode="r")
            _rows = _joined["ct_row"].to_numpy()

        def read(positions: np.ndarray) -> np.ndarray:
            return np.asarray(_grids[_rows[np.asarray(positions)]], dtype=np.float32)

        return read

    return (grid_reader,)


@app.cell(hide_code=True)
def _(REPRESENTATIONS, grid_reader, representation_choice):
    representation = representation_choice.value
    representation_name = REPRESENTATIONS[representation]
    # Saved figures of Hugo's latent keep their names (docs/meeting_2026-10-29.md uses them);
    # the other representations get their key appended.
    figure_suffix = "" if representation == "hugo_vae" else f"_{representation}"
    # Grids of the representation chosen in the dropdown; used by every section.
    read_grids = grid_reader(representation)
    print(f"representation: {representation_name}")
    return figure_suffix, read_grids, representation_name


@app.cell(hide_code=True)
def _(fruit, read_grids):
    # Grids of the chosen representation in the order of `fruit`, flattened to 32,768
    # numbers per fruit.
    flat = read_grids(fruit["position"].to_numpy()).reshape(fruit.height, -1)
    is_2526 = (fruit["season"] == "2526").to_numpy()

    # PCA as in notebook 05: mean-centred, not scaled, exact solver, fitted on the 2526 fruit;
    # every fruit projected.
    _pca = PCA(n_components=50, svd_solver="full").fit(flat[is_2526])
    pc_scores = _pca.transform(flat)
    print(
        f"grids {flat.shape}; PCA fitted on {is_2526.sum()} fruit; scores {pc_scores.shape}"
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
    ### The same correlations for all three representations

    Independent of the dropdown: for Hugo's latent, the random encoder and the CT averaged to
    32³, the same PCA as above (full grid, centred, not scaled, fitted on the 2526 fruit), and
    the Spearman correlation of PC1 to PC5 with stalk, size and the grades, 2526 fruit only.
    The sign of a component is arbitrary, so compare sizes, not signs. The last two columns
    give the largest absolute correlation over PC1 to PC5 and the component where it is.

    How to read it: if the CT averaged to 32³ also puts stalk length on PC1, the stalk
    dominance is in the data (the grid includes the stalk), not something Hugo's encoder
    added. If the random encoder looks like Hugo's latent, the pattern comes from the
    architecture and the input, not from the training.
    """)
    return


@app.cell(hide_code=True)
def _(REPRESENTATIONS, fruit, grid_reader, is_2526):
    # Five components keep the table readable; the descriptors as in section 1, shortened
    # to the ones that matter for the stalk question and the grades.
    _n_components = 5
    _properties = [
        "stalk_length",
        "body_voxels",
        "body_length",
        "diameter_to_length",
        "defective",
        "browning",
        "cavity",
    ]
    _fruit_2526 = fruit.filter(pl.Series(is_2526))
    _positions = fruit["position"].to_numpy()
    _rows = []
    for _key, _name in REPRESENTATIONS.items():
        _flat = grid_reader(_key)(_positions).reshape(fruit.height, -1)
        # Same PCA as section 1: centred, not scaled, exact solver, fitted on 2526 fruit.
        _pca = PCA(n_components=_n_components, svd_solver="full").fit(_flat[is_2526])
        _scores = _pca.transform(_flat[is_2526])
        _with_scores = _fruit_2526.with_columns(
            pl.Series(f"PC{_k + 1}", _scores[:, _k]) for _k in range(_n_components)
        )
        for _property in _properties:
            _r = [
                _with_scores.select(
                    pl.corr(f"PC{_k + 1}", _property, method="spearman")
                ).item()
                for _k in range(_n_components)
            ]
            _strongest = int(np.argmax(np.abs(_r)))
            _rows.append(
                {
                    "property": _property,
                    "representation": _name.split(" (")[0],
                    **{f"PC{_k + 1}": round(_r[_k], 2) for _k in range(_n_components)},
                    "largest |r|": round(abs(_r[_strongest]), 2),
                    "at": f"PC{_strongest + 1}",
                }
            )
        print(
            f"{_name}: first {_n_components} components explain"
            f" {_pca.explained_variance_ratio_.sum():.0%} of the variance"
            f" (PC1 {_pca.explained_variance_ratio_[0]:.0%})"
        )
    # Rows grouped by property, so the three representations sit under each other.
    pc_correlations_by_representation = pl.DataFrame(_rows)
    pc_correlations_by_representation
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
def _(fruit, is_2526, pc_scores, representation_name):
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
        f"PCA of {representation_name}, fitted on {int(is_2526.sum())} 2526 fruit after"
        " the exclusions"
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
def _(cfg, figure_suffix, pc_panel_figure, save_pc_panel):
    # Saved under derived/ (outside git).
    mo.stop(mo.running_in_notebook() and not save_pc_panel.value)
    _out = cfg.path("derived_dir") / "figures" / f"07_pc1_pc2_panels{figure_suffix}.png"
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
def _(fruit):
    # 2526 fruit only: the groups below are 2526 groups. `position` is the row of the fruit in
    # Hugo's manifest and masks.npy. Own cell, so it does not depend on the representation.
    hr_fruit = fruit.filter(pl.col("season") == "2526")
    return (hr_fruit,)


@app.cell(hide_code=True)
def _(hr_fruit, read_grids, run_dir):
    _positions = hr_fruit["position"].to_numpy()
    _grids = read_grids(_positions)
    _cells = np.load(run_dir / "masks.npy", mmap_mode="r")
    # Per radius method: (fruit, height, radius) bin means, NaN where the fruit has no cell.
    hr_means = {}
    for _method in RADIUS_METHODS:
        _counts, _sums = [], []
        for _k, _position in enumerate(_positions):
            _body = np.asarray(_cells[_position])
            _c, _s = latent_map_sums(_grids[_k], _body, _method)
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
    return (hr_means,)


@app.cell(hide_code=True)
def _():
    # Fixed choices of sections 3 and 4, in a cell that reads no data: the controls built
    # from them are then not recreated (and reset to their defaults) when the
    # representation at the top changes.
    # Groups as in notebook 06 section 5. Cavity only has 3 fruit in the open 2526 boxes, too
    # few; the last group is cavity 2 or 3 with any browning.
    HR_GROUPS = [
        ("grade 0", pl.col("defective") == 0),
        ("browning 1 only", (pl.col("browning") == 1) & (pl.col("cavity") == 0)),
        ("browning 2 only", (pl.col("browning") == 2) & (pl.col("cavity") == 0)),
        ("browning 3 only", (pl.col("browning") == 3) & (pl.col("cavity") == 0)),
        ("cavity 2-3, any browning", pl.col("cavity") >= 2),
    ]
    # Correction name -> short name for the saved files.
    CORRECTIONS = {
        "none": "none",
        "minus grade 0": "minus_grade0",
        "minus grade 0, divided by grade 0 SD": "grade0_sd",
        "minus grade 0 of the same box": "minus_box_grade0",
    }
    return CORRECTIONS, HR_GROUPS


@app.cell(hide_code=True)
def _(HR_GROUPS, hr_fruit):
    hr_group_fruit = {
        _name: hr_fruit.select(_condition).to_series().to_numpy()
        for _name, _condition in HR_GROUPS
    }
    print({_name: int(_in.sum()) for _name, _in in hr_group_fruit.items()})
    return (hr_group_fruit,)


@app.cell(hide_code=True)
def _(hr_fruit, hr_group_fruit, hr_means):
    # Group summaries: the shared rule of notebooks 06 and 07 (every fruit counts once; a bin
    # only where at least half of the fruit have cells), `pearlatent.step0.group_bin_summary`.
    group_summary = group_bin_summary
    _healthy = hr_group_fruit["grade 0"]
    _boxes = hr_fruit["box"].to_numpy()

    def grade0_correction(values: np.ndarray, correction: str) -> np.ndarray:
        """Per-fruit values (2526 fruit in the order of `hr_fruit`, then any bin or cell axes)
        after the correction, computed per bin or cell.

        The grade 0 reference of a bin follows the same rule as every group: it exists only
        where at least half of the grade 0 fruit have cells in the bin, otherwise the
        corrected value is NaN. "Same box": each fruit minus the grade 0 mean of its own box
        (8 to 26 grade 0 fruit per box). It uses the labels of every box, so it is for
        looking only; in cross-validation a held-out box cannot use its own grade 0 fruit.
        Used by section 3 (height-radius bins) and section 4 (cells of one latent plane).
        """
        _values = values
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

    def corrected_means(method: str, correction: str) -> np.ndarray:
        """Per-fruit height-radius bin means (fruit, height, radius) after the correction."""
        return grade0_correction(hr_means[method], correction)

    return corrected_means, grade0_correction, group_summary


@app.cell(hide_code=True)
def _(group_summary, hr_group_fruit, hr_means, representation_name):
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
        f"Height-radius maps of {representation_name}, 2526 fruit: the two radius methods"
        " (red = higher value)"
    )
    hr_comparison_figure = _fig
    hr_comparison_figure
    return (hr_comparison_figure,)


@app.cell(hide_code=True)
def _(CORRECTIONS, HR_GROUPS):
    hr_method = mo.ui.dropdown(
        options={"one radius per fruit": "fruit", "radius per slice": "slice"},
        value="one radius per fruit",
        label="Radius",
    )
    hr_correction = mo.ui.dropdown(
        options=list(CORRECTIONS), value="minus grade 0", label="Correction"
    )
    hr_map_group = mo.ui.dropdown(
        options=[_name for _name, _ in HR_GROUPS],
        value="browning 3 only",
        label="Map of group",
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
def _(hr_fruit):
    def pca_on_selection(values: np.ndarray, unit: str, description: str):
        """PCA of the selected bins or cells, with score plots, scree plot and correlations.

        In: values (fruit, selected columns), one row per 2526 fruit in the order of
        `hr_fruit`, NaN where a fruit has no value; `unit` ("bins" or "cells") and a
        `description` of the choices (radius method or plane, correction) for the titles.
        Out: the figures and the correlation table, stacked for display.
        Used by sections 3 and 4. A PCA cannot take empty values, so columns that some fruit
        do not reach are left out (the number is printed). PCA centred, not scaled, as in
        section 1.
        """
        _complete = ~np.isnan(values).any(axis=0)
        _table = values[:, _complete]
        print(
            f"{unit} selected: {values.shape[1]}; left out (not reached by every fruit):"
            f" {int((~_complete).sum())}; used: {_table.shape[1]};"
            f" fruit: {_table.shape[0]}"
        )
        if _table.shape[1] < 2:
            return mo.md(
                f"_Fewer than 2 usable {unit} selected: select a larger region._"
            )

        # Up to 20 components for the scree plot (fewer when the selection is smaller).
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
            [
                ("stalk_length", "stalk length (voxels)"),
                ("body_voxels", "body voxels"),
            ],
            strict=True,
        ):
            _points = _ax.scatter(
                _x, _y, c=hr_fruit[_column].to_numpy(), cmap="viridis", s=12
            )
            _fig.colorbar(_points, ax=_ax, label=_label)
            _ax.set_title(f"coloured by {_label}")
        _grade = hr_fruit["defective"].to_numpy()
        for _g, _colour in zip(
            range(4), ["#c6dbef", "#6baed6", "#2171b5", "#08306b"], strict=True
        ):
            _axes[2].scatter(
                _x[_grade == _g],
                _y[_grade == _g],
                color=_colour,
                s=12,
                label=f"grade {_g}",
            )
        _axes[2].legend(fontsize=8)
        _axes[2].set_title("coloured by defective grade")
        _cause = hr_fruit["cause"].to_numpy()
        for _c, _colour in zip(
            ["none", "browning only", "cavity only", "both"],
            ["lightgrey", "#d95f02", "#1b9e77", "#7570b3"],
            strict=True,
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
            f"PCA of {_table.shape[1]} selected latent {unit}, {description};"
            f" {_table.shape[0]} 2526 fruit"
        )
        # Scree plot: share of the variance of the selection per component (bars) and the
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
            f"Scree plot, {_table.shape[1]} selected {unit}: first {len(_k)} components"
            f" explain {np.cumsum(_explained)[-1]:.0%}"
        )
        return mo.vstack(
            [
                _fig,
                _scree,
                mo.md("**Spearman correlation with the first components**"),
                _correlations,
            ]
        )

    return (pca_on_selection,)


@app.cell(hide_code=True)
def _(
    corrected_means,
    hr_correction,
    hr_fruit,
    hr_linked,
    hr_method,
    pca_on_selection,
    representation_name,
):
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
    pca_on_selection(
        _values,
        "bins",
        f"{representation_name}, radius per {hr_method.value},"
        f" correction: {hr_correction.value}",
    )
    return


@app.cell(hide_code=True)
def _(corrected_means, group_summary, hr_group_fruit, representation_name):
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
            f"Group means of {representation_name}, radius per {method},"
            f" correction: {correction}"
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
    figure_suffix,
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
        _folder / f"07_hr_radius_methods{figure_suffix}.png",
        dpi=130,
        bbox_inches="tight",
    )
    _saved = [f"07_hr_radius_methods{figure_suffix}.png"]
    for _method in RADIUS_METHODS:
        for _correction, _short in list(CORRECTIONS.items())[1:]:
            _name = f"07_hr_surfaces_{_method}_{_short}{figure_suffix}.png"
            _figure = hr_surface_figure(_method, _correction)
            _figure.savefig(_folder / _name, dpi=130, bbox_inches="tight")
            plt.close(_figure)
            _saved.append(_name)
    print(f"saved {', '.join(_saved)} to {_folder}")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 4. Long sections of the real latent (exploration only)

    **Question.** Section 3 folds the latent onto height × radius. Is a difference between
    grades already visible in the latent itself, cell by cell, without that folding? Only to
    look: nothing is fitted or tested here.

    **What is shown.** One plane of the 32³ latent `mu`, 32 × 32 cells, chosen with the slider:
    a plane at a fixed x (showing y and height) or at a fixed y (showing x and height). The
    stem-calyx axis is the third grid axis, so every such plane is a long section. Nothing is
    averaged or rescaled: every point is one latent cell at its own grid position. The slider
    starts on the plane through the median axis position of the fruit (printed below).

    **Differences with section 3**, each one on purpose:

    - Height is the raw grid slice (0 to 31, calyx end low), not the height over the body.
      Fruit differ in length (the 2526 body spans 21 to 32 slices, median 28) and in how much
      stalk they have, so the same slice lies at a different relative height in different
      fruit.
    - No average over the angle. Rotation about the axis is not aligned between fruit, so one
      plane shows a different side of each fruit. A lesion on one side of a fruit appears in
      this plane only when that side happens to lie in it.
    - A cell counts for a fruit only where it is a body cell of that fruit (`masks.npy`);
      outside the body the value is empty. A group value is drawn only where at least half of
      the fruit of the group have a body cell, the rule of every map in notebooks 06 and 07.

    **Spectrum.** The 1,024 cells of the plane in one row, slice by slice:
    cell = 32 × slice + width position. Grey vertical lines separate the slices; the axis
    labels give the slice number every 4 slices.

    **Selecting, corrections and groups** as in section 3: drag a box on the map or a stretch
    along the spectra, double-click to clear; the same four grade 0 corrections, computed per
    cell; the same five 2526 groups.
    """)
    return


@app.cell(hide_code=True)
def _(hr_fruit, run_dir):
    # Where the stem-calyx axis lies in the grid, from the body masks only (the same for
    # every representation, so the slider below is not reset when the representation
    # changes): per fruit the mean x and mean y of its body cells. The slider starts on the
    # plane through the median of these positions.
    _cells = np.load(run_dir / "masks.npy", mmap_mode="r")
    _body = np.asarray(_cells[hr_fruit["position"].to_numpy()], dtype=bool)
    grid_size = _body.shape[1]
    _body_cells = _body.sum(axis=(1, 2, 3))
    _index = np.arange(grid_size, dtype=float)
    _centre_x = (_body * _index[:, None, None]).sum(axis=(1, 2, 3)) / _body_cells
    _centre_y = (_body * _index[None, :, None]).sum(axis=(1, 2, 3)) / _body_cells
    axis_plane = {
        "x": int(round(float(np.median(_centre_x)))),
        "y": int(round(float(np.median(_centre_y)))),
    }
    print(
        f"body cells per fruit median {np.median(_body_cells):.0f}"
        f" (min {_body_cells.min()}, max {_body_cells.max()})"
    )
    print(
        f"axis position per fruit: x median {np.median(_centre_x):.1f}"
        f" (range {_centre_x.min():.1f} to {_centre_x.max():.1f}),"
        f" y median {np.median(_centre_y):.1f}"
        f" (range {_centre_y.min():.1f} to {_centre_y.max():.1f})"
    )
    return axis_plane, grid_size


@app.cell(hide_code=True)
def _(hr_fruit, read_grids, run_dir):
    # Grids of the 2526 fruit of section 3 (same order as `hr_fruit`), empty (NaN) outside
    # the body cells of each fruit. About 315 fruit × 32,768 cells as float32: about 41 MB.
    _cells = np.load(run_dir / "masks.npy", mmap_mode="r")
    _positions = hr_fruit["position"].to_numpy()
    _body = np.asarray(_cells[_positions], dtype=bool)
    grid_values = np.where(_body, read_grids(_positions), np.nan)
    print(f"grid cells: {grid_values.shape}")
    return (grid_values,)


@app.cell(hide_code=True)
def _(CORRECTIONS, HR_GROUPS):
    plane_direction = mo.ui.dropdown(
        options={
            "fixed x (shows y and height)": "x",
            "fixed y (shows x and height)": "y",
        },
        value="fixed x (shows y and height)",
        label="Plane",
    )
    plane_correction = mo.ui.dropdown(
        options=list(CORRECTIONS), value="minus grade 0", label="Correction"
    )
    plane_group = mo.ui.dropdown(
        options=[_name for _name, _ in HR_GROUPS],
        value="browning 3 only",
        label="Map of group",
    )
    plane_show_fruit = mo.ui.checkbox(label="Show the single fruit of that group")
    mo.hstack(
        [plane_direction, plane_correction, plane_group, plane_show_fruit],
        justify="start",
    )
    return plane_correction, plane_direction, plane_group, plane_show_fruit


@app.cell(hide_code=True)
def _(axis_plane, grid_size, plane_direction):
    # Separate cell: the slider restarts on the axis plane whenever the direction changes.
    plane_index = mo.ui.slider(
        start=0,
        stop=grid_size - 1,
        value=axis_plane[plane_direction.value],
        label=f"Plane position ({plane_direction.value}); axis at about {axis_plane[plane_direction.value]}",
        show_value=True,
    )
    plane_index
    return (plane_index,)


@app.cell(hide_code=True)
def _(
    grade0_correction,
    grid_values,
    group_summary,
    hr_group_fruit,
    plane_correction,
    plane_direction,
    plane_group,
    plane_index,
    plane_show_fruit,
):
    # One plane of every fruit, as (fruit, slice, width): the slice (third grid axis, the
    # stem-calyx axis) first so it becomes the vertical axis of the map.
    if plane_direction.value == "x":
        _plane = grid_values[:, plane_index.value, :, :]  # (fruit, y, slice)
        _width_name = "y"
    else:
        _plane = grid_values[:, :, plane_index.value, :]  # (fruit, x, slice)
        _width_name = "x"
    _plane = np.swapaxes(_plane, 1, 2)
    _n_slices, _n_width = _plane.shape[1], _plane.shape[2]
    _all_cells = _n_slices * _n_width
    # The correction is per cell, so correcting the plane alone gives the same values as
    # correcting the whole grid and then taking the plane.
    _values = grade0_correction(_plane, plane_correction.value)
    # (fruit, 1,024 cells), cell = slice × width + width position; also used by the PCA below.
    plane_flat = _values.reshape(-1, _all_cells)

    _s, _w = np.meshgrid(np.arange(_n_slices), np.arange(_n_width), indexing="ij")
    _coordinates = pl.DataFrame(
        {
            "cell": (_s * _n_width + _w).ravel(),
            "slice": _s.ravel(),
            "width": _w.ravel(),
        }
    )
    plane_box = alt.selection_interval(name="plane_box", encodings=["x", "y"])
    plane_range = alt.selection_interval(name="plane_range", encodings=["x"])

    # Map: group mean of one group after the correction, one rectangle per cell.
    _mean = group_summary(_values, hr_group_fruit[plane_group.value])["mean"]
    _map_table = _coordinates.with_columns(value=pl.Series(_mean.ravel())).filter(
        pl.col("value").is_not_nan()
    )
    _limit = float(np.nanpercentile(np.abs(_mean), 99))
    _map_colour = (
        alt.Scale(scheme="viridis")
        if plane_correction.value == "none"
        else alt.Scale(scheme="redblue", reverse=True, domain=[-_limit, _limit])
    )
    _size = 400
    _map = (
        alt.Chart(_map_table)
        .mark_rect(width=_size / _n_width, height=_size / _n_slices)
        .encode(
            x=alt.X(
                "width:Q",
                scale=alt.Scale(domain=[-0.5, _n_width - 0.5], nice=False),
                title=f"grid {_width_name}",
            ),
            y=alt.Y(
                "slice:Q",
                scale=alt.Scale(domain=[-0.5, _n_slices - 0.5], nice=False),
                title="grid slice (calyx end low)",
            ),
            color=alt.Color("value:Q", scale=_map_colour, title=plane_correction.value),
            opacity=alt.condition(plane_range, alt.value(1.0), alt.value(0.15)),
            tooltip=["cell", "slice", "width", "value"],
        )
        .add_params(plane_box)
        .properties(
            width=_size,
            height=_size,
            title=(
                f"{plane_group.value}, plane {plane_direction.value} = {plane_index.value}"
            ),
        )
    )

    # Spectra: mean per group and the 25 % to 75 % band, every cell at its own number.
    _rows = []
    for _name, _selected in hr_group_fruit.items():
        _summary = group_summary(plane_flat, _selected)
        _rows.append(
            pl.DataFrame(
                {
                    "group": _name,
                    "cell": np.arange(_all_cells),
                    "mean": _summary["mean"],
                    "q25": _summary["q25"],
                    "q75": _summary["q75"],
                }
            )
        )
    _spectra = (
        pl.concat(_rows)
        .join(_coordinates, on="cell", validate="m:1")
        .filter(pl.col("mean").is_not_nan())
    )
    # x axis: a label every 4 slices, at the first cell of that slice.
    _x = alt.X(
        "cell:Q",
        scale=alt.Scale(domain=[-0.5, _all_cells - 0.5], nice=False),
        axis=alt.Axis(
            values=list(range(0, _all_cells, 4 * _n_width)),
            labelExpr=f"datum.value / {_n_width}",
            labelAngle=0,
        ),
        title="cell; grey lines separate the slices, label = slice number",
    )
    _group_colour = alt.Color("group:N", sort=list(hr_group_fruit), title="group")
    _base = alt.Chart(_spectra).transform_filter(plane_box).encode(x=_x)
    _delimiters = (
        alt.Chart(pl.DataFrame({"cell": np.arange(1, _n_slices) * _n_width - 0.5}))
        .mark_rule(color="lightgrey", strokeWidth=0.8)
        .encode(x=_x)
    )
    _layers = [
        _delimiters,
        _base.mark_area(opacity=0.15).encode(
            y="q25:Q", y2="q75:Q", color=_group_colour
        ),
        _base.mark_line(point=alt.OverlayMarkDef(size=6))
        .encode(
            y=alt.Y("mean:Q", title=plane_correction.value),
            color=_group_colour,
            tooltip=["group", "cell", "slice", "width", "mean", "q25", "q75"],
        )
        .add_params(plane_range),
    ]
    if plane_show_fruit.value:
        _in = hr_group_fruit[plane_group.value]
        _single = (
            pl.DataFrame(
                {
                    "fruit": np.repeat(np.flatnonzero(_in), _all_cells),
                    "cell": np.tile(np.arange(_all_cells), int(_in.sum())),
                    "value": plane_flat[_in].ravel(),
                }
            )
            .join(_coordinates, on="cell", validate="m:1")
            .filter(pl.col("value").is_not_nan())
        )
        _layers.insert(
            1,
            alt.Chart(_single)
            .transform_filter(plane_box)
            .mark_line(color="grey", opacity=0.15, strokeWidth=0.7)
            .encode(x=_x, y="value:Q", detail="fruit:N"),
        )
    _spectrum = alt.layer(*_layers).properties(
        width=620,
        height=_size,
        title="Mean spectrum per group, band 25 % to 75 % of the fruit",
    )
    plane_linked = mo.ui.altair_chart(
        alt.hconcat(_map, _spectrum).resolve_scale(color="independent"),
        chart_selection=False,
        legend_selection=False,
    )
    plane_linked
    return plane_flat, plane_linked


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### PCA on the selected cells

    As in section 3, with cells of the plane instead of bins: the cells inside the box on the
    map and inside the stretch dragged along the spectra (cell numbers); with neither, the
    whole plane. One row per 2526 fruit, one column per selected cell, after the chosen
    correction. A cell that lies outside the body of even one fruit is empty for that fruit
    and is left out, so selections near the skin, the stalk or the calyx end lose many cells,
    and a plane far from the axis may have none left.

    Read it as in section 3: circular when the region is picked because the grades differ
    there; "minus grade 0" gives the same plot as "none"; the useful question is whether stalk
    length and body size still run along PC1.
    """)
    return


@app.cell(hide_code=True)
def _(
    grid_size,
    pca_on_selection,
    plane_correction,
    plane_direction,
    plane_flat,
    plane_index,
    plane_linked,
    representation_name,
):
    def selected_cells(selections: dict, n_width: int, n_cells: int) -> np.ndarray:
        """Cell numbers of the plane selected by both drags; a drag that is not used selects
        every cell.

        - `selections["plane_box"]`: {"width": [low, high], "slice": [low, high]}, the box on
          the map; a cell is in it when its centre (its integer position) lies inside.
        - `selections["plane_range"]`: {"cell": [low, high]}, the stretch along the spectra.
        Cell number = slice × n_width + width position, as in the chart.
        """
        _all = np.arange(n_cells)
        _keep = np.ones(n_cells, dtype=bool)
        _box = (selections or {}).get("plane_box") or {}
        if "width" in _box and "slice" in _box:
            _w_low, _w_high = sorted(_box["width"])
            _s_low, _s_high = sorted(_box["slice"])
            _width = _all % n_width
            _slice = _all // n_width
            _keep &= (
                (_width >= _w_low)
                & (_width <= _w_high)
                & (_slice >= _s_low)
                & (_slice <= _s_high)
            )
        _range = (selections or {}).get("plane_range") or {}
        if "cell" in _range:
            _c_low, _c_high = sorted(_range["cell"])
            _keep &= (_all >= _c_low) & (_all <= _c_high)
        return _all[_keep]

    # The width axis of a plane is one grid axis (x or y), 32 cells long.
    _cells = selected_cells(plane_linked.selections, grid_size, plane_flat.shape[1])
    pca_on_selection(
        plane_flat[:, _cells],
        "cells",
        f"{representation_name}, plane {plane_direction.value} = {plane_index.value},"
        f" correction: {plane_correction.value}",
    )
    return


if __name__ == "__main__":
    app.run()
