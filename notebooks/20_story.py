import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title="20 Story for Hugo")

with app.setup(hide_code=True):
    import json

    import marimo as mo
    import polars as pl

    from pearlatent.config import load_config
    from pearlatent.runs import latest_run_dir


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 20 Story: what we did, what it taught us, what comes next

    **Question.** What have the models on Hugo's latent, the CT at 32³ and Med3D shown so far,
    with which caveats, and what is the plan? For the meeting with Hugo on 24 November 2026.

    **Form.** Sections 0 to 9. Each: what we did, the figure, one caveat line, what it taught
    us, next step. Nothing is computed here: figures come from `derived/figures/` (made by the
    notebook named under each figure), numbers from `docs/methods_plsda.md` and
    `docs/methods_step0_and_maps.md`. Wording partly from Ruben's answers in the walkthrough of
    20 November.

    **All numbers:** open 2526 boxes only (12 boxes, 315 fruit after the faulty volumes); the
    3 sealed test boxes F, J, M are still closed. Design A (20 repeats of 4-fold box
    cross-validation, latent variables chosen by Ruben at the RMSECV elbow); AUC with a 95 %
    interval from 2,000 box-bootstrap draws.

    **Inputs.** Saved figures in `derived/figures/`. **Outputs.** None.

    **Status.** draft, 21 November 2026, for Ruben to read and correct.
    """)
    return


@app.cell(hide_code=True)
def _():
    cfg = load_config()
    _figures = cfg.path("derived_dir") / "figures"

    def show_figure(name: str, caption: str):
        """A saved figure with its caption, or a note when the file is missing.

        In: file name in `derived/figures/`, caption. Out: a marimo element.
        """
        path = _figures / name
        if not path.exists():
            return mo.md(
                f"_Figure `{name}` not found; run the notebook that makes it._"
            )
        return mo.vstack([mo.image(src=path, width="100%"), mo.md(f"_{caption}_")])

    return cfg, show_figure


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 0. The data and what is held back

    **What we did.**

    - 2526 season, the primary data: 450 labelled fruit, 433 scanned, 15 boxes; a box letter is
      an orchard (Hugo). 3 boxes (F, J, M; 87 fruit) drawn at random on 13 October and sealed.
      12 open boxes, 346 fruit; 31 faulty volumes (part of the fruit missing, failed
      preprocessing; `docs/outlier_rule.md`) leave **315 fruit, 129 non-consumable**.
    - 2425 season, secondary: 660 fruit. Hugo's 528 encoder training fruit are excluded; his 66
      validation fruit serve only the reverse check; his 66 test fruit are sealed.
    - Every model: split by box. 12 boxes in 4 groups of 3; fit on 9, score the 3 unseen;
      repeated 20 times with other groupings; a fruit's score is the mean of its 20
      out-of-fold scores. The number of latent variables is chosen on the same curve; one final
      model on all 12 boxes is the one for the sealed test and the counterfactuals.

    **Caveat.** 12 boxes is few, so the intervals are wide.

    **What it taught us** (Ruben). A split by fruit would put fruit of one box on both sides;
    fruit of one box are correlated, so the model could learn the orchard or scan artefacts
    and the estimate would be too optimistic. The encoder training fruit are excluded because
    the question is whether the latent captures browning in any pear, not in the fruit it was
    trained to reconstruct.

    **Next.** The sealed boxes are opened once, when the rows stop changing.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1. What the encoder does to a pear

    **What we did.** Hugo's frozen VAE encodes each 128³ volume (scaled 0 to 1 per fruit,
    aligned on the stem-calyx axis, stretched to 128 voxels in length) to `mu`, 32³ cells. One
    cell every 4 voxels (about 2.5 mm), but each cell sees 31 voxels (about 19 mm).
    InstanceNorm normalises over the whole fruit; KL weight 1e-6, so a compression latent.
    Rotation about the axis is free, so the same cell is different tissue in different fruit;
    hence the height-radius rows (20 heights × 8 rings, mean and SD per ring).
    """)
    return


@app.cell(hide_code=True)
def _(show_figure):
    show_figure(
        "07_pc1_pc2_panels.png",
        "PCA of mu, 315 open 2526 fruit (notebook 07). The first two components follow stalk "
        "length and body voxels, not the grade.",
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **Caveat.** "Size" here is size after stretching to 128 voxels: with the stalk counted in
    the length, stalk length and body size correlate at −0.85 on the 145 healthy fruit. They
    are one measurement, not two.

    **What it taught us** (Ruben). The largest variation in the latent is the pear itself
    (shape, stalk), not the browning. A small lesion is diluted inside a cell that sees 19 mm.

    **Next.** Keep size and stalk in every check (section 7).
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2. Step 0: is browning kept in the reconstruction?

    **What we did.** Decode `mu` (`vae.decode(mu)`, not `vae(x)`, which adds random noise; all
    models use `mu`) and compare with the CT. On Hugo's 66 validation fruit: SSIM 0.9895 and
    MAE 0.0028 against his 0.9915 and 0.0025. Contrast retention (how much of the core
    darkening against grade 0 is still there): about 0.9 to 1 for the mean darkening; fine
    texture is smoothed. Hugo's training fruit are not reconstructed better than unseen fruit.
    """)
    return


@app.cell(hide_code=True)
def _(show_figure):
    show_figure(
        "06_examples.png",
        "CT, reconstruction and residual for grade 0 and browning 1, 2 and 3 (notebook 06). "
        "The broad core darkening is kept; the fine spots of browning 3 are smoothed.",
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **Caveat.** Retention uses only the two central long sections, not the whole 3D core.

    **What it taught us** (Ruben). Browning is a small part of the image, so a whole-volume
    SSIM of 0.99 says nothing about it; the change has to be measured in the defect region
    against grade 0. The signal of grade 1 is mostly blurred away in the encoding.

    **Next.** Set aside on 9 November; it remains charter question 1 for the thesis.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 3. Consumable against non-consumable (`binary_23`)

    **What we did.** PLS-DA on 8 rows: latent and CT at 32³, each as full grid, height-radius
    mean, height-radius mean + SD; tissue features (128³); Med3D. One SVM check row.

    | Row | AUC (95 % box interval) |
    |---|---|
    | SVM, latent mean + SD (check row) | 0.967 (0.948 to 0.980) |
    | Med3D | 0.958 (0.934 to 0.973) |
    | **Latent height-radius mean + SD** | **0.957 (0.939 to 0.971)** |
    | Tissue features | 0.938 (0.886 to 0.966) |
    | CT height-radius mean + SD | 0.931 (0.903 to 0.947) |
    | Latent height-radius | 0.927 (0.886 to 0.955) |
    | Latent grid | 0.887 (0.833 to 0.928) |
    | CT height-radius | 0.884 (0.838 to 0.915) |
    | CT grid | 0.839 (0.781 to 0.888) |
    """)
    return


@app.cell(hide_code=True)
def _(show_figure):
    show_figure("18_auc_rows.png", "AUC per row with box intervals (notebook 18).")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **Caveat.** The mean + SD rows were added after the other rows were seen; the sealed boxes
    must confirm them.

    **What it taught us.**

    - The latent is above the CT at 32³ for every pooling (paired: grid +0.048, 0.019 to
      0.082; height-radius +0.043, 0.008 to 0.082). Same number of cells, but each latent cell
      is a learned summary of 31³ voxels at full resolution; the CT row averages blocks of 4³
      and loses what varies inside a block.
    - Height-radius beats the full grid for both: rotation invariance, and 160 instead of
      32,768 features for 315 fruit, so fewer directions that fit the labels by chance.
    - Latent mean + SD against Med3D: +0.001 (−0.029 to +0.023). We find no difference; the data
      fit the latent being up to 0.029 worse or 0.023 better. An encoder trained only to
      reconstruct 528 pears reaches the level of a large medical network.
    - SVM against PLS-DA on the same features: +0.011 (−0.004 to +0.025); no clear gain from a
      non-linear boundary.

    **Next.** Rerun the rows with the chosen preprocessing (section 9).
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 4. By cause: is browning carried by cavity?

    **What we did.** Each defect group of the `binary_23` models against the 145 grade 0 fruit;
    and a separate model trained on cavity-free fruit only (browning ≥ 2, 233 fruit).
    """)
    return


@app.cell(hide_code=True)
def _(show_figure):
    show_figure(
        "18_auc_by_cause.png",
        "AUC of each defect group against grade 0, per row (notebook 18).",
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    | Row | browning only 3 (n 30) | cavity 2 or 3 (n 61) | browning only 2 (n 20) | browning only 1 (n 38) |
    |---|---|---|---|---|
    | Latent mean + SD | 0.989 | 0.954 | 0.941 | 0.671 |
    | Med3D | 0.988 | 0.978 | 0.914 | 0.703 |
    | Latent grid | 0.916 | 0.917 | 0.806 | 0.643 |
    | CT grid | 0.848 | 0.882 | 0.751 | 0.594 |

    **Caveat.** Browning only 2 has 20 fruit from few boxes: wide intervals, rows cannot be
    told apart in that column.

    **What it taught us.** The charter predicted, before looking, that non-consumable fruit
    would be found mostly through cavity. It did not hold: browning without any cavity is found
    about as well. Browning 3 is a large diffuse darkening that shifts many cells and rings at
    once. The cavity-free model gives the same order and level (latent mean + SD 0.942, Med3D
    0.945), so training on all fruit does not hurt browning detection (Ruben: the check was
    whether the result was carried by cavity alone).

    **Next.** Report `binary_23` split by cause throughout (charter rule 5).
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 5. Where it stops: grade 1

    **What we did.** Neighbouring grades on every row: 0 against 1 is 0.59 to 0.72, 1 against 2
    is 0.76 to 0.88, 2 against 3 is 0.57 to 0.65. Training on the grade or on the 0/1 cut does
    no better; the ordinal and per-grade models assign no fruit to grade 1. Then one number
    per fruit, no model: the SD of the interior cells.
    """)
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(show_figure):
    show_figure(
        "15_interior_spread.png",
        "SD of the interior cells per fruit, 233 cavity-free fruit (notebook 15). CT at 32³ "
        "separates grade 0 from 1 (0.82); the latent does not (0.50).",
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **Caveat.** The outer layer of cells was removed after looking (all body cells: 0.53).

    **What it taught us.**

    - Ruben: the spread catches the onset (0 against 1), the level of the cells catches
      defective against healthy. Browning likely starts patchy (SD goes up) and becomes a
      whole-core darkening (the mean shifts).
    - Grade 1 is mostly, not fully, lost in the encoder (PLS-DA and SVM 0.67 and 0.68, lower
      bounds above 0.5). A second loss sits in the features: the CT PLS-DA rows stay at 0.6 to
      0.7 while one SD gives 0.82.
    - Ruben: 0 against 1 and 2 against 3 may partly be label noise; a fruit was graded on one
      or two cuts, so the grader saw part of the lesion. We have no cut-surface photographs.

    **Next.** Spread features for grade 1 (section 9). For Hugo: question 53.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 6. Where in the fruit: interval PLS (iPLS)

    **What we did.** iPLS on the 160 height-radius bins of the latent (notebook 13): one model
    per interval, then a forward search, scored by RMSECV (no model 0.492; all 160 bins 0.339).

    - By height (0 = calyx end): slices 6 to 9 alone are best, about 30 to 50 % up from the
      calyx (RMSECV about 0.36); bottom and top slices are near no model.
    - By depth (0 = axis, 7 = skin): rings 2 to 4 are best; the two outer rings (0.49) are no
      better than no model. The CT shows the same pattern.
    - Selecting intervals barely helps: 0.339 to 0.330.

    **Figure.** Notebook 13 (not saved as a file).

    **Caveat.** The axis ring has the fewest cells and the most filled-in values, so its weak
    score does not prove there is nothing at the axis.

    **What it taught us.** A location map, not a better model: the information sits at mid
    height and mid depth, around the core.

    **Next.** Used as the reason for a core-only model, which was then tried (section 7).
    """)
    return


@app.cell(hide_code=True)
def _(cfg):
    # The selection Ruben saved in notebook 13 section 5 (heights, then rings, mean + SD).
    _run = latest_run_dir(cfg.path("results_dir"), "ipls_selection_hugo_vae_hr_sd")
    mo.stop(
        _run is None,
        mo.md("_No selection saved yet (notebook 13, section 5, save button)._"),
    )
    _record = json.loads((_run / "run.json").read_text())["params"]
    _scores = json.loads((_run / "scores.json").read_text())
    _paired = _scores["paired"]
    _measures = "\n".join(
        f"| {_name} | {_s['cutoff']:.3f} | {_s['recall']:.3f} | {_s['precision']:.3f} |"
        for _name, _s in _scores.items()
        if _name != "paired"
    )
    _groups = (
        pl.read_parquet(_run / "auc_by_group.parquet")
        .with_columns(pl.col("auc", "lower_95", "upper_95").round(3))
        .select("group", "model", "n_group", "auc", "lower_95", "upper_95")
    )
    _check_c = pl.read_parquet(
        _run / "check_c_healthy_correlations.parquet"
    ).with_columns(pl.col("spearman", "lower_95", "upper_95").round(2))
    mo.vstack(
        [
            mo.md(
                f"""
    ### 6b. A model on the selected region (latent mean + SD)

    **What we did** (notebook 13 sections 3 to 5, run `{_run.name}`). Heights chosen along
    the forward path, then rings inside them, then PLS-DA on the kept bins:
    heights **{_record["heights"]}** (0 = calyx end), rings **{_record["rings"]}** (0 = axis),
    {_record["n_lv_chosen"]} LVs. Against all 320 features ({_record["compared_n_lv"]} LVs):
    AUC {_paired["auc_a"]:.3f} against {_paired["auc_b"]:.3f}, paired difference
    **{_paired["difference"]:+.3f}** ({_paired["lower_95"]:+.3f} to {_paired["upper_95"]:+.3f}).

    | model, cut-off | cut-off | recall | precision |
    |---|---|---|---|
    {_measures}
    """
            ),
            _groups,
            mo.md("Check C on the 145 healthy fruit (score minus box median):"),
            _check_c,
        ]
    )
    return


@app.cell(hide_code=True)
def _(show_figure):
    show_figure(
        "13_selection_hugo_vae_hr_sd.png",
        "Weights of the final fit on the kept bins: on the height-radius map (grey = not "
        "kept) and on the body cells of one fruit, for the bin means and the bin SDs "
        "(red raises the score towards non-consumable; notebook 13 section 5).",
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **Caveat.** Heights, rings and LVs are all chosen on the cross-validation that scores
    them, so this AUC is optimistic; only the sealed boxes can test a selection. Neighbouring
    height bins are strongly correlated, so the forward search skips some (3, 6, 8, 9): the
    gaps do not mean those heights carry nothing.

    **What it taught us** (run `20261122T162303Z_ipls_selection_hugo_vae_hr_sd`).

    - The lower 60 % of the body carries the model. The kept heights (bins 1 to 11, 5 to 60 %
      of the body height from the calyx end) lie in the range Ruben chose from biology for the
      core-only rows (5 to 60 %); no ring is dropped. 112 of the 320 features give AUC 0.966
      against 0.957 for all (paired +0.010, −0.001 to +0.024): no loss, no clear gain.
    - By cause the same picture: browning only 2 0.961 against 0.941, cavity 0.967 against
      0.954, grade 1 unchanged (0.67). Selecting a region does not rescue grade 1.
    - Per grade at the Youden cut-off (0.373): called non-consumable grade 0 4 %, grade 1
      22 %, grade 2 90 %, grade 3 94 % (all features: 6 %, 22 %, 87 %, 95 %).
    - Check C unchanged (stalk −0.29, size +0.33): the link is not in the top of the body or
      the neck, as the core-only rows already showed (section 7).
    - The weights: a higher **SD** in the middle rings (2 to 5) at heights 2 to 7 raises the
      score: a ring that is more uneven in the latent, read as patchy tissue around the core
      (a reading, not tested). A higher **mean** of rings 1 to 4 at
      heights 4 to 7 lowers it. The model reads browning as spread around the core more than as
      a shift in level, the same message as the interior SD for grade 1 (section 5).

    **Next.** Test the region on the sealed boxes together with the other rows; the SD weights
    support spread features for grade 1 (plan step 5).
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 7. Checks, and the two attempts to remove the stalk and size link

    **What we did.**

    - Box (check A): the within-box AUC is close to the overall AUC on every row, so the score
      is not the orchard.
    - Size and stalk (check C, on the 145 healthy fruit): the latent height-radius score follows
      stalk length (−0.31) and body size (+0.34); the CT rows, tissue features and Med3D do not.
    - All 346 volumes, faulty ones kept: same order of rows. Reverse check on 61 of Hugo's
      validation fruit: AUC 0.97 to 1.00 (tissue features 0.71), but there every
      non-consumable fruit has a cavity.
    - **Core-only rows** (methods section 16): only cells at 5 to 60 % of the body height. Latent
      AUC 0.869 against 0.887; stalk link unchanged (−0.19), size link worse (+0.27). Dropped.
    - **EPO** (methods section 17): remove from every fruit the main axis of variation among the
      healthy training fruit (PC 1, which follows body size at r = 0.97 and stalk at −0.83).
      Latent rows keep their AUC (height-radius mean + SD 0.957 → 0.952) and lose the link
      (stalk −0.29 → +0.07, size +0.30 → −0.04). Not suited to the CT height-radius rows (their
      PC 1 is not size). EPO must be fitted in the model's own features: EPO on the grid and
      then binning removes nothing.

    **Caveat.** The core-only and the height-radius EPO numbers are scratch runs; the EPO runs
    on the latent grid are saved (22 November, `EPO_RUN_DIRS`). Stalk alone
    separates every defect group from grade 0 at AUC 0.58 to 0.65; larger fruit brown more
    (Ho et al. 2008), so part of the link may be real biology.

    **What it taught us.** The link does not sit in the stalk and neck cells; it most likely
    enters through the stretching to 128 voxels. EPO removes it from the scores at no cost.

    **Next.** For Hugo: artefact or biology (question 59)? For Wouter Saeys: EPO with healthy
    fruit as the nuisance set (question 60).
    """)
    return


@app.cell(hide_code=True)
def _(show_figure):
    show_figure(
        "22_pca_before_after_epo.png",
        "PCA of the latent full grid, 315 open 2526 fruit, before and after removing the "
        "healthy PC 1 (notebook 22 section 5; direction fitted once on the 145 healthy fruit).",
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **What the PCA shows** (notebook 22 section 5, Spearman r with each PC):

    | | variance | body size | stalk | grade |
    |---|---|---|---|---|
    | Before EPO, PC 1 | 22 % | +0.97 | −0.78 | +0.18 |
    | Before EPO, PC 2 | 8 % | +0.10 | +0.07 | −0.05 |
    | After EPO, PC 1 | 11 % | +0.12 | −0.25 | +0.09 |
    | After EPO, PC 2 | 6 % | +0.16 | +0.06 | +0.13 |

    - Before: PC 1 is size, and PC 1 against PC 2 is a horseshoe. That shape is what PCA makes
      of one strong gradient: PC 2 bends with PC 1 and carries no new information.
    - After: the horseshoe is gone and PC 1 no longer follows size overall (0.12). But the
      largest fruit (about 250,000 body voxels and more) still sit together at the right. One
      straight direction removes the main size trend, not its curved part at the large end.
    - The grade does not show in the first two components, before or after.
    """)
    return


@app.cell(hide_code=True)
def _(show_figure):
    show_figure(
        "22_check_c_before_after_epo.png",
        "Check C as a figure: the score of the 145 healthy fruit (minus its box median) "
        "against body size and stalk length, latent height-radius mean + SD, 3 LVs, without "
        "and with EPO (notebook 22 section 6).",
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **What the check C figure shows.** Without EPO, the healthy fruit's score rises with size
    (+0.30) and falls with stalk length (−0.29). With EPO the trend is gone (−0.04, +0.07) at
    nearly the same AUC (0.957 → 0.952). But the smallest fruit (below about 100,000 body
    voxels) now score high, up to +0.4: one straight direction overshoots at the small end. The
    rank correlation near 0 hides this. Same message as the PCA: the size effect is curved, and
    one EPO component removes only its straight part. Candidates: GLSW (a soft weighting of
    the nuisance directions), or the stretch factor per fruit as the external parameter
    (question 18). Stalk length moves in steps of 4 voxels because the body-stalk boundary is
    set on the 32³ grid (question 41).
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 8. Counterfactual images

    **What we did** (methods sections 14 and 18). Direction: the regression vector b of the
    cavity-free model on the full latent grid, fitted on all 12 boxes; with EPO, b cannot move a
    fruit along the removed size axis. Scale t: 0 = mean score of grade 0, 1 = grade 3. A fruit's
    `mu` is moved along b to a target t, decoded, encoded again and scored. Three steps per fruit:
    halfway (t 0.5), to the other grade mean, and mirrored across the decision boundary. The
    score of the moved `mu` is the target by construction; re-encoding asks whether the change
    survives as an image.
    """)
    return


@app.cell(hide_code=True)
def _(show_figure):
    show_figure(
        "17_asked_against_got.png",
        "EPO direction, 4 fruit. Black bar: the fruit's own t; open circle: the t asked for; "
        "blue dot: the t of the decoded image after encoding it again (notebook 17).",
    )
    return


@app.cell(hide_code=True)
def _(show_figure):
    show_figure(
        "17_all_runs_images.png",
        "Long section through the body centre: CT, plain reconstruction, and each step minus "
        "the plain reconstruction (blue brighter, red darker; notebook 17).",
    )
    return


@app.cell(hide_code=True)
def _(show_figure):
    show_figure(
        "17_mirrored_steps.png",
        "Mirrored step per fruit (notebook 17, section 6). Top: original CT. Middle: the "
        "decoded mirrored fruit, with its t before, asked and got. Bottom: the original CT with "
        "the change against the plain reconstruction painted on top where it is at least 0.02 "
        "(blue brighter, red darker).",
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **Caveat.** 4 fruit, examples only. The decode-encode round trip alone moves t by −0.12 to
    −0.23, so each fruit is read against its own plain reconstruction.

    **What it taught us.**

    - Only part of each step comes back: 43 % (K07) to 73 % (A03) of the asked score change,
      the same share for every step of one fruit. Whether the decoder ignores part of the
      change or the encoder does not return to it is not known (question 61).
    - The outline covers the largest area (a change of shape), then a stalk patch at the neck
      (C23 towards healthy, K07 towards defective). Painted on the CT (last figure), the core
      also changes clearly: C23 brighter inside towards healthy, K07 and A03 darker around
      the seed cavity towards defective; N17 hardly at all.
    - With and without EPO the same within 0.02 in t, and the same images: removing the main
      size axis does not take the stalk out of the direction. Mirrored: C23, N17 and A03 cross
      the boundary, K07 does not.
    - Mirrored across the Youden cut-off of the same model (0.298, t 0.31; notebook 22 run
      `20261122T152905Z_epo_cavity_free`): the two grade 3 fruit cross it (C23 t −0.02, N17
      −0.08); the two grade 0 fruit do not (K07 0.04, A03 0.24), because the Youden mirror asks
      them for a shorter step.
    - Check C (the score follows the stalk on healthy fruit) and the stalk in the counterfactual
      are one finding seen in two ways.

    **Next.** Section 9.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 9. The plan

    Updated 21 November 2026 from the plan of 20 November.

    | # | Step | Status, why |
    |---|---|---|
    | 1 | Core region from biology and the rubric | Done, dropped: link not removed (section 7). |
    | 2 | EPO on the healthy fruit | Done for the latent rows and the counterfactual direction; removes the score link, not the stalk in the images. |
    | 3 | Better stalk measure (voxel-level body-stalk boundary, question 41); projection of b on it | Open. The stalk patch in the counterfactuals needs a direction EPO did not remove. |
    | 4 | Rerun all rows with the chosen preprocessing (EPO on the latent rows?) | To decide with Hugo; today's rows become the preliminary analysis. |
    | 5 | Spread features for grade 1 | The SVM check gave no gain from non-linearity alone. |
    | 6 | N-way PLS | Charter row. |
    | 7 | Counterfactuals again after step 3 | Should then change tissue, not shape. |
    | 8 | ResNet-18 on the lab workstation | Charter baseline; needs the workstation (question 9). |
    | 9 | Sealed test (F, J, M) | Once the rows stop changing. |

    **Questions for Hugo** (full list with what each answer decides:
    `docs/meeting_2026-11-24.md`, section 4):

    - Q1. Height-radius rows as model rows beside the full grid? On 29 October he said
      exploration only; they now carry the best latent result.
    - Q2 (59). A short stalk (a relatively large body) goes with defects: artefact of the
      stretching or real risk factor? Decides EPO on the final latent rows.
    - Q4 (9). Lab workstation access for the ResNet-18 baseline.
    - Q5 (18). Per fruit, the scale factor of the rescaling to 128 voxels, and was it the
      same along all axes? Used as a measurement, not to reprocess: check C with real size
      against the stretch factor (decides Q2 from the data), and EPO on the stretch factor
      for every row.

    Answered by Ruben (23 November): grading by Hugo's decision table, not every region cut,
    so label noise at 0/1 and 2/3 exists (53); Ruben asks Wouter Saeys directly about the EPO
    variant (60). Answered on 29 October, not asked again: 33 (no reprocessing), 17 (no extra
    labels), 16 (ResNet settings are Ruben's choice).
    """)
    return


if __name__ == "__main__":
    app.run()
