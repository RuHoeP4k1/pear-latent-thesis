# HANDOFF.md

The living state of the code work. Claude reads this at the start of every session (a hook prints
the "Current state" section) and updates it with `/handoff` at the end. Ruben may edit it freely.

## Current state

- **Date:** 2026-10-28
- **Phase:** preparation of the meeting with Hugo (Thursday 29 Oct, 11:00). Notebook 07 now
  compares the three 32³ representations (Hugo's latent, random encoder, CT averaged to 32³)
  and shows long sections of the real grid. No model fitted on labels yet. Material for Hugo:
  `docs/meeting_2026-10-29.md` (not updated this session) and notebook 07, shown live.
- **Decided this session (Ruben, 28 Oct):**
  - The height-radius map is an exploratory method: the number of latent cells behind a bin
    differs between bins and between fruit (estimate: about 2 cells in the axis bin of a
    small fruit, about 140 in the outer ring of a large one). See question 39.
  - The real grid is shown as a long section (one plane at fixed x or y, slider), not as a
    mean over one axis.
  - The comparison with the CT uses the existing 4 × 4 × 4 block mean of notebook 03 (charter
    comparison row); no new downsampling.
  - Hugo is shown notebook 07 live, not a fixed comparison figure.
- **Done this session (28 Oct, Mac), commit bba5df6:**
  - Notebook 07: dropdown at the top for the representation, read through `grid_reader`
    (the CT run holds all 1,093 fruit; it is joined on the open manifest, so the 153 sealed
    rows are never read). Saved figures of the other two representations get
    `_random_init` / `_raw_ct_32` in their names.
  - Section 1: table of Spearman correlations of PC1 to PC5 with stalk, size and grades for
    all three representations at once.
  - Section 4 (new): long sections of the grid, map and 1,024-cell spectrum linked both
    ways, the four grade 0 corrections, PCA on the selected cells. Section 3 and 4 share
    `pca_on_selection` and `grade0_correction`.
  - Controls no longer reset when the representation changes (groups, corrections, the 2526
    fruit and the axis position moved to cells that read no grid data).
  - Checked: the CT run's body masks equal those of Hugo's run for all 940 open fruit; the
    random encoder run is complete (940 × 32³, no NaN, no all-zero fruit; sealed part in
    `derived/sealed/latents/random_init/`).
- **Found (exploratory, 315 open 2526 fruit, seen by Ruben):**
  - PC1 follows body size and stalk length equally in all three representations (body voxels
    |r| 0.97 / 0.97 / 1.00; stalk 0.78 / 0.74 / 0.77 for Hugo / random / CT): the stalk and
    size structure is in the volumes, not added by Hugo's encoder.
  - Only Hugo's latent has a component (PC4) that follows the grades more than size
    (|r| 0.25 defective, 0.23 browning, 0.27 cavity); random encoder and CT reach about 0.16,
    on PC1 (size). One component, no test.
  - Only 78 of the 1,024 cells of the central long section lie in the body of every fruit;
    a PCA on the plane drops the rest.
- **Next step:** the meeting with Hugo on 29 Oct (Ruben), with notebook 07 open on Hugo's
  latent. Discuss the stalk route (questions 33, 18, 38) with the section 1 comparison table,
  and the Med3D choices (question 40). After the meeting: record the 28 October method rules
  in `docs/methods_step0_and_maps.md`, then the first PLS-DA on `binary_23` with Hugo's full
  flattened latent (`pearlatent.plsda`, box folds in `derived/outer_folds_20/`, test boxes
  F, J, M sealed).
- **Blocked on:** modelling on labels waits for the meeting (stalk reprocessing, original
  fruit lengths, size and stalk in the maps). Med3D waits for the choices in question 40.
- **Working note:** with `marimo edit --watch` open, the editor reloads and saves the file;
  several quick file edits in a row were overwritten on 28 Oct. Make notebook edits in one
  write, or through the live session (marimo-pair, needs the server token).
- **Reference, still valid (from earlier sessions):**
  - Encoding run `20261013T171306Z_hugo_vae` (code 0c9052f; `dirty: true` only because of an
    untracked handoff file): open part `derived/latents/hugo_vae/20261013T171306Z_hugo_vae/`,
    940 fruit (346 2526, 594 2425), 478 decoded (346 2526, Hugo's 66 validation, 66 of his
    training fruit, seed 369); sealed part `derived/sealed/latents/...`, 153 fruit, encoded
    only. Random encoder run `20261020T120336Z_random_init` (seed 0, same fruit and order).
  - Descriptors run `derived/descriptors/20261023T155219Z_descriptors/` (with
    `flat_face_share`, `stalk_present`).
  - Test boxes drawn 13 Oct: **F, J, M** (87 fruit). Fold partitions of the 12 other boxes,
    repeats: 1 ABI CGN DEO HKL; 2 ACE BGL DIN HKO; 3 AEH BGL CKO DIN; 4 AHN BDL CEI GKO;
    5 AGK BCI DNO EHL. They stay as drawn after the exclusions (Ruben, 23 Oct); the balance
    after exclusion is reported.
  - Decoding: about 6 minutes per fruit on the Mac CPU (17 GB of memory), 56 s on the laptop.
  - `pearlatent.plsda` and `pearlatent.feature_sets` (commit 4265761) are built and tested,
    not run on real labels. One fit of 15 components on 260 × 32,768 takes about 0.1 s.
  - Size and stalk (Ruben, 23 Oct): preferred route is reprocessing 2526 without the stalk
    before the rescaling (question 33); otherwise a sensitivity row with stalk length and body
    size regressed out inside each training fold and/or a check within stalk classes. Why it
    matters: Ho et al. 2008 (PLoS Computational Biology 4, e1000023): fruit size changes
    internal O₂ and CO₂, and low core O₂ leads to browning.
  - Mac (main machine from 20 Oct): `/Users/rubenhoeven/code/pear-latent-thesis`, Python
    3.12.15, torch 2.14.1 (`device = "cpu"` in `config/local.toml`), marimo 0.25.1; data in
    `~/data/pear-thesis`; weights in `data/weights/hugo_vae/`; Hugo's repository at
    `../Synthetic_CT_pear`, commit 183112f. `uv run pytest`: 98 passed on 27 Oct.
    The workstation is only for the ResNet-18 baseline and is not set up. School PC not set
    up: `uv sync`, copy `config/local.example.toml` to `config/local.toml`, run
    `notebooks/00_environment_check.py`; for live pairing also
    `npx skills add marimo-team/marimo-pair` and `jq`, `curl` on the PATH.

## Setting up the Mac (written 17 Oct, on the Windows laptop)

Nothing below travels through git; copy it by drive or USB stick. Sizes on 17 Oct.

1. **Transfer (17 Oct):** Ruben carries the data via KU Leuven OneDrive. Unzip it outside the
   repository. **Copy from the laptop:** `C:/Users/hoeve/thesis-data/` (4.3 GB, volumes and both label
   files); the repository's `data/weights/` (372 MB, Hugo's `best.pth` and hyperparameters);
   `derived/` (900 MB, includes the finished encoding run, `exclusions.csv`, `sealed_fruit.csv`,
   `test_boxes/`, `hugo_vae_split_2425/`). Do not re-run the test-box draw: `derived/test_boxes/`
   is the one draw. Do not copy `.venv/` or `__pycache__/`.
2. **Repository:** `git clone` it, then put `data/weights/` and `derived/` in place.
3. **Hugo's repository:** `git clone https://github.com/Hugo-Li-Junyan/Synthetic_CT_pear` next to
   this one and `git checkout 183112f` (12 MB; copying the folder also works).
4. **Config:** Claude writes it on the Mac: copy `config/local.example.toml` to
   `config/local.toml` and set the Mac paths
   (`ct_dir_2526`, `ct_dir_2425`, `labels_2425`, `labels_2526`; `hugo_repo = "../Synthetic_CT_pear"`
   if cloned next to it). Set `device = "cpu"` for anything that re-encodes or decodes, so new
   runs stay comparable with the laptop runs (CPU, float32); `"auto"` would pick Apple MPS.
   Ask Ruben before using MPS for a run whose numbers enter the thesis.
5. **Environment:** install `uv`, then `uv sync` (torch comes from the CPU index on macOS, see
   `pyproject.toml`); `uv run pytest`; `uv run python notebooks/00_environment_check.py`.
6. **Check the copy:** load `derived/latents/hugo_vae/20261013T171306Z_hugo_vae/latents.npy`
   (940 × 32 × 32 × 32) and the sealed one (153 fruit).
7. **Claude Code on the Mac:** Ruben brings `claude-config.zip` (made 17 Oct; folders `rules`,
   `agents`, `skills`, `commands` go to `~/.claude/`, `project-memory/*` goes to the project's
   `memory/` folder; no login token, no `settings.json`). Background: the memory folder is keyed by the project path, so copy
   `C:/Users/hoeve/.claude/projects/C--Users-hoeve-code-pear-latent-thesis/memory/` into the new
   project folder under `~/.claude/projects/` (two notes: record only seen results; marimo
   pairing). The user-level `~/.claude/` (rules, `agents/challenger.md`, skills `marimo`,
   `synced`) also needs copying. The marimo pairing note is Windows-specific (jq path); on the
   Mac `brew install jq` and rerun `npx skills add marimo-team/marimo-pair`.
8. Update the "Laptop" line above to describe the Mac once it works.

## Verified from Hugo's repository (Tier 1)

Source: github.com/Hugo-Li-Junyan/Synthetic_CT_pear, commit 183112f (2026-08-31). Read on 2026-10-03.

1. Volumes are NIfTI (`.nii` / `.nii.gz`), single channel, 128 x 128 x 128, loaded with nibabel and
   min-max scaled to [0, 1] per volume (`utils/volumes.py`).
2. File names look like `A30.nii`: box letter A to O, then a number. Regular expression in
   `train_clf_3d.py`: `^([A-O])\d+`. Applies to the data that code was written for; whether the
   2025 and 2026 files follow the same pattern is not confirmed.
3. Encoder: `component/vae.py`, built with `featuremap_size=32`, `base_channel=256`, residual
   blocks, 3D latent of shape (1, 32, 32, 32). `utils/load_models.py` loads `checkpoint.pth` plus
   `vae_hyperparameter.json` from one run folder. Hugo's own latent plots use the mean `mu`, not a
   sampled `z` (`scripts/vae_latent_2d_visualize.py`). We use `mu`.
4. Receptive field of one latent cell: 31 voxels, about 19.4 mm; cell spacing 4 voxels, about
   2.5 mm (computed from the layer list). Caveat: InstanceNorm normalises each channel over the whole
   volume, so every latent cell also depends on global intensity statistics.
5. VAE training augmentation: random flips on two axes, isotropic scaling 0.9 to 1.1, rotation of
   -30 to 30 degrees around one axis (`train_vae.py`).
6. No trained weights in the repository (`.gitignore` excludes `*.pth`).
7. The repository has no licence file. Do not copy its code into this repository; reference it at a
   pinned commit instead (git submodule or a separate clone).
8. Preprint, section 2.2: the scans were resampled to 0.6267 mm isotropic, then each fruit was
   rotated to the vertical axis, rescaled to 128 voxels in length and zero-padded to 128³
   (MATLAB). Section 2.5: random 80/10/10 split; the validation set was used for hyperparameter
   tuning and model selection.

## Weights, received 13 October

Read on the laptop on 13 October 2026 (handoff B1), on CPU, with `torch.load(weights_only=True)`.

- Received: one file, `best.pth` (389,867,547 bytes, SHA-256 `b416375b…2c64d37`), copied from
  Ruben's Downloads folder to `data/weights/hugo_vae/best.pth` (ignored by git). Not received:
  `checkpoint.pth`, `vae_hyperparameter.json`, the rest of the run folder.
- Keys: `epoch` = 430, `random_state` = 369 (matches Hugo's statement), `vae_state_dict`,
  `gan_state_dict`, `vae_optimizer_state_dict`, `gan_optimizer_state_dict`. There is no `use_gan`
  key; the discriminator weights (`gan_state_dict`, 5,891,004 numbers) show that the run trained
  with the adversarial loss.
- `vae_state_dict`: 12 `encoder.` and 16 `decoder.` tensors, 26,597,379 numbers. Encoder and
  decoder are both present.
- Without the JSON, `vae_base_channel` and `vae_featuremap_size` cannot be read. Checked instead:
  Hugo's `VAE` (clone at 183112f) built with base channel 256 and feature-map size 32 loads the
  state dict with `strict=True` (all names and shapes match); a random 128³ input gives `mu` and
  `logvar` of 1 × 1 × 32 × 32 × 32 and `decode(mu)` of 1 × 1 × 128³.
- `best.pth` is the checkpoint at the lowest validation reconstruction loss (Hugo's
  `train_vae.py`), epoch 430.
- `vae_hyperparameter.json` received later the same day (Ruben's Downloads folder; to be copied
  to `data/weights/hugo_vae/`): model "VAE-GAN", `vae_featuremap_size` 32, `vae_base_channel`
  256, `vae_use_residual` true, latent 1 × 32 × 32 × 32, loss MAE, beta 1e-06, 500 epochs, batch
  size 4, Adam learning rate 1e-4 for VAE and discriminator, gamma 0.01, discriminator patch size
  16 and base channel 16. Matches the strict load and CLAUDE.md section 3.
- `load_encoder` and notebook 04 now default to `best.pth` (commit bbaa33c).
- Read in `train_vae.py` (13 October, for open question 21): the validation and test subsets are
  `random_split` subsets of the one dataset that carries the training augmentation (random
  flips, scaling 0.9 to 1.1, rotation up to 30 degrees), so Hugo's validation and test metrics
  were computed on randomly augmented volumes, with `vae(x)` (sampled z). The final test metrics
  are computed with the model at the end of training, not reloaded from `best.pth`. Our numbers
  use unaugmented volumes, `decode(mu)` and `best.pth`, so they need not match Table D.1 or
  section 3.1.1 exactly. Hugo's PSNR is averaged per batch of 4, ours per fruit.

## Points to raise with Hugo (found in his code)

1. Discussed 13 October 2026. Hugo's VAE train/validation/test split is random by fruit
   (`utils/splits.py`). This does not affect us if the 2025 and 2026 fruit are truly unseen, but
   it should be stated in the thesis.
2. Hugo's 3D classifier (`train_clf_3d.py`, `split_train_val_test_by_batch`) puts fruit from every
   box into train, validation and test. Our rule keeps whole boxes together. Classification numbers
   from his script and from ours are therefore not directly comparable.
3. `README.md` describes `test_clf_3d.py`, but that file was deleted in commit e9982b8 (2026-06-23).

## 2526 volumes as downloaded (listed 3 October 2026, names and sizes only)

- 15 archives `set-r0948344-2526_<box>.tar`, one per box A to O, in Ruben's Downloads folder.
- Inside each: one folder, files `<box><NN>.nii` (for example `A01.nii`), 4,194,656 bytes each:
  a 352-byte NIfTI-1 header plus 128³ voxels of 2 bytes (16-bit integers). Already 128³.
- 433 volumes, not 450. Missing: C27, E24, F13, F22, F28, G08, G15, G16, G19, G23, G25, G28,
  H08, H24, I16, K25, N24 (17; 7 of them in box G). Check against the label file whether these
  fruit have labels, and ask Hugo why they have no scan.
- 2425 (listed the same day): one archive `set-r0948344-2425.tar`, one top folder, 660 volumes,
  complete. A01–J60 (60 per letter) plus G_opt01–G_opt30 and I_opt01–I_opt30. Same file size.
  The `_opt` names do not match the box pattern; `box_from_filename` will refuse them on purpose.

## Plan items to settle with Ruben

1. Test boxes: decided 5 Oct to hold out 2526 test boxes; the rule (restricted randomisation,
   proposal in `docs/meeting_2026-10-15.md` section 2) is discussed with Hugo first. Also still to
   do: the reversed evaluation design (charter rule 2) in the Approved ideas tab.
2. "2025 and 2026 fruit" or one harvest (2025) stored into 2026? Wording in charter and plan.
3. Settled in the charter of 10 October: "Constraints" now states that most 2526 volumes include
   the stalk and 2425 volumes do not, marked *(to settle with Hugo)*.
4. Block 0 checkpoint of 8 October ("encode and decode one 2526 pear") cannot be met: the weights
   arrive on 15 October. The gate test cannot use browning annotations for 2526: cut-open
   photographs exist only for the 2024–2025 season (in the pear folder).

## Open questions (must be answered from Tier 1 or by Hugo)

Answered and removed on 5 Oct (Hugo, October 2026): the 17 unscanned 2526 fruit are excluded
(poor scan setup); a 2526 box letter is an orchard ID. Answered and removed on 21 Oct (Hugo's mail): 19 (his VAE split
confirmed). Answered and removed on 23 Oct (Ruben): 23 (outlier rule,
`docs/outlier_rule.md`), 26 (van der Voet test per box, `docs/cv_explained.md` section 4),
32 (PLS-DA centres only, no scaling; CLAUDE.md section 7). Answered and removed on 26 Oct
(Ruben): 22 (2526_N29 and L29 are deformed volumes, excluded; K27 a long stalk, kept; O19 a
real odd shape, kept), 25 (season split on PC1 is mostly the stalk; PC panels in notebook 07),
28 (no flat-face cut-off; visual inspection decides). Answered and removed on 13 Oct (meeting):
1 (weights received; seed 369 per Hugo; checkpoint read in B1), 3 (rot not really a problem in
2526), 6 (stalks not a problem), 7 (we reproduce the split ourselves, B2), 8 (test-box rule,
charter rule 3), 10 (item 2 dropped with the self-supervised extension). Numbers are kept so
earlier references stay valid.

2. Whether the data may be uploaded to cloud services. Ruben (3 Oct): Claude may read notebook
   outputs. molab and other uploads: no until Hugo confirms in writing.
4. Are 2024 and 2526 orchards with the same letter different orchards?
5. Ruben's position on the grade 2 non-consumable threshold (Hugo asked).
9. Lab workstation (only for the ResNet-18 baseline now): GPU model, NVIDIA driver version,
   operating system, `uv`, tmux.
11. Cavity-label issues: grades that disagree with visible voids (to back with the cavity
    cross-check, if built).
12. The month0 to month4 repeat scans: which fruit, and are they available?
13. Overlap with Hugo's own classification and synthetic-data work.
16. ResNet-18: "full" means 16 base channels (his default) or 64? Epochs, batch size and memory
    on the lab GPU. Can we see his own ResNet results, and with which split?
17. Which extra labels he can provide, and in what form.
18. Fruit dimensions: the length of each fruit before rescaling to 128 voxels (from his MATLAB
    preprocessing), and whether that rescaling was the same along all three axes. If both, every
    dimension in millimetres follows from our volumes. Otherwise the original scans.
21. The preprint reports MAE 0.0022 on the test set (section 3.1.1) and MAE 0.0025 on the
    validation set (Table D.1), with the same SSIM and PSNR. Which numbers belong to which set?

For Ruben, added 21 Oct:
24. Fit PCA and PLS on all 32,768 cells or only on cells inside the fruit (body masks in
    `masks.npy`, for example cells inside the body of at least half the fruit)? Would remove
    the background and the edge band.
27. Bootstrap on the test set: the 3 test boxes are too few for a bootstrap over boxes, so the
    test interval resamples fruit (`docs/cv_explained.md` section 8). Hugo's mail speaks of
    paired bootstrap intervals in general; state this difference at the meeting.

For Hugo (meeting 29 Oct), added 23 Oct:
30. Volumes with a straight, often oblique cut through the fruit (2526 boxes I and L most;
    2425_I44, I45, C59): field of view of the scanner, made oblique by the alignment rotation?
    Known to him? Dark volumes (2425_A36; 2526_C03, I03, L03, all three number 03 of their box):
    a scanning-position effect or a bright object setting the min-max scaling?
31. Stray objects above the intensity threshold beside the fruit (2526_K13: 607 voxels): holder
    or label in the scan?
33. Reprocessing the 2526 volumes without the stalk before the rescaling to 128 voxels (his
    MATLAB pipeline, from the original scans), so the body fills the grid as in his 2425
    training data; then re-encode (about 1.5 hours) and re-decode what Step 0 needs. Possible?
    And the original length of every fruit (open question 18), because size is a known cause of
    browning (Ho et al. 2008) and is not in the rescaled volumes.

For Hugo (meeting 29 Oct), added 26 Oct:
34. The cut volumes are uneven over the grades (2526 grade 1 18 %, grade 3 14 %, grade 0 5 %,
    grade 2 4 % excluded; p = 0.006 within box). Larger fruit sticking out of the field of
    view? Needs question 18.
35. Contrast retention uses only the two central long sections per fruit. Does he think the
    full 3D core is needed (all 2526 fruit decoded again, about 5.5 hours)?

For Hugo (meeting 29 Oct), added 26 Oct (second session):
37. May Ruben work forward from his 3D ResNet design as the reference classifier, with his own
    training objective (binary_23, one output), box splits, augmentation and no pseudo-labels?
    Full list in `docs/meeting_2026-10-29.md`, Part 2 item 4.

For Hugo (meeting 29 Oct), added 27 Oct:
38. The height-radius map groups differ in stalk and body size, not only in grade (27 Oct,
    315 open 2526 fruit: median stalk 20, 19, 13.5, 12 voxels and median body 142,600,
    150,900, 174,300, 182,500 voxels for grade 0 and browning only 1, 2, 3; browning 3 mostly
    in boxes O and A). Ruben discusses with Hugo how to separate browning from size and stalk
    in the maps (linked to questions 18 and 33).

For Ruben, added 27 Oct:
39. After the meeting: fit the models on the height-radius bins, on the full flattened
    latent, or both? A height-radius row reopens the decision of 10 October that dropped the
    cylindrical bins (and item 20, removing the old cylindrical code in `pooling.py`).
    28 Oct: Ruben agrees the height-radius map is exploratory (unequal number of cells per
    bin); a model row on the bins would also need a rule for empty bins. Leaning to the full
    grid for the models; to confirm.

For Ruben and Hugo, added 28 Oct:
40. Med3D (charter comparison row, frozen; no code yet): which network (ResNet-10 is the
    realistic one on the CPU), which input normalisation (Med3D was trained on z-scored CT,
    our volumes are min-max scaled), which layer and pooling (the last layer gives about
    16³ × 512 for a 128³ input, not 32³), and the weights: source, version, checksum,
    licence. Adding the package needs `uv add`.
41. Stalk length and body size are measured on the 128³ volumes, but the body-stalk
    boundary is decided on the 32³ CT grid (slices with fewer than 4 cells), so it moves in
    steps of 4 voxels and the neck tip counts as stalk. Define the boundary at voxel level
    (for example fewer than 64 fruit voxels per slice) and compare on all fruit? A change
    to a method rule; after the meeting.

For Ruben, added 26 Oct:
36. Put the grading rubric table (Tier 1) in the repository or in `data/` (outside git), so
    the grade definitions (for example browning 1 = light discoloration of the kernel
    membranes) can be cited from it.

For Ruben (not Hugo), added 10 Oct:

14. Reference coordinates of the healthy fruit for the deviation map. The earlier reference used
    height and radius; the same cell (x, y, z) is affected by the free rotation around the
    stem–calyx axis.
15. Whether Solo accepts a four-way array for N-way partial least squares. The documentation found
    does not say. Test with a small synthetic array (fruit × 8 × 8 × 8) before relying on it.

For Ruben, added 13 Oct:

20. Remove `cylindrical_coordinates`, `cylindrical_profile` and the profile part of `pool_grid`
    from `src/pearlatent/pooling.py`, with their tests? Notebook 03 no longer uses them (1e5d1f1).

## Decided

- 26 Oct (Ruben): Step 0 reconstructs with `vae.decode(mu)`, not `vae(x)`, because the latent
  analysed downstream is `mu`, so Step 0 must test whether `mu` keeps the browning; `vae(x)`
  adds noise to z and gives a different output every run. No comparison with sampled z is run.
  Thesis wording: our overall scores (`decode(mu)`, unaugmented volumes, `best.pth`) are not
  directly comparable with Hugo's (`vae(x)`, augmented volumes, model at the end of training);
  on his 66 validation fruit, ours are mean MAE 0.0028, SSIM 0.9895, PSNR 35.51 against his
  0.0025 (Table D.1), 0.9915, 36.92. Contrast retention per grade is our own measure; Hugo has
  only whole-volume scores and the qualitative `scripts/difference_map.py`.
- 6 Oct (plan review): Med3D is the one external model (no 3DINO, no DINOv3); attention pooling and
  the 512-channel layer are dropped; the two-feature support vector machine of Van De Looverbosch
  2020 is literature context only; Ideas 4, 5 and 6 are not proposed; order: core study, then own
  self-supervised extension, then counterfactuals in January–February.
- 10 Oct: the cylindrical bins (height × radius) are dropped everywhere; N-way partial least
  squares is one extra model row after the core study, on fruit × x × y × z, compared with unfolded
  partial least squares on the same box splits. Implementation (own numpy, Bro 1996) waits until
  Ruben asks; see `docs/NOTE_n-way-PLS_2026-10-10.md`.
- Code not yet changed for the dropped bins: `src/pearlatent/pooling.py` (`cylindrical_coordinates`,
  `cylindrical_profile`, `pool_grid`), `tests/test_pooling.py`, `notebooks/03_raw_ct_pooled.py`
  (profile figure and text). Ruben to decide whether to remove them.
- 13 Oct (meeting with Hugo): counterfactuals welcomed; no 3D self-supervised training (GPU), so
  the own self-supervised extension is dropped; ResNet-18 from scratch as baseline (Hugo has
  probably trained one himself, but wants Ruben to train his own); target order: `defective`
  first (consumable versus not, per grade), then browning versus cavity; downsampled CT kept as
  comparison; stalks not a problem; test-box rule agreed (restricted randomisation, charter rule
  3); Hugo's VAE split reproducible from his split function and seed; rot not really a problem in
  2526; grade approach agreed; Hugo reads Keicher et al. 2023; Hugo may provide extra labels; fruit
  dimensions can be computed from the original CT scans. Hugo confirmed his VAE seed: 369.
- 13 Oct (Ruben): the random encoder stays as a comparison row. Order of work: test boxes first
  (rule approved, then draw, then summary to Hugo for a green light), then the reconstruction
  errors per label group with visual comparisons and the stalk, then the latent analysis. Code
  must be readable and every step explained (CLAUDE.md section 11).
- 13 Oct (Ruben): in 2425, Hugo's 528 encoder-training fruit are excluded from all analysis (the
  encoder has seen them), except one Step 0 comparison of reconstruction error, never pooled with
  the other fruit. His 66 validation fruit are on the training side. His 66 test fruit are sealed
  and join the final test as one batch, reported per storage group and separately from the 2526
  test boxes. Hugo's split by fruit means every 2024 orchard is also in his training set; this
  cannot be undone and is stated in the thesis.
- 13 Oct (Ruben): test boxes by a random draw among the sets within ±30 % (charter rule 3), seed
  369. A ranking variant ("keep the best 20, draw among them") was considered the same day and
  not chosen. Balance counts: non-consumable, browning ≥ 2 without cavity, browning 3, any cavity
  ("browning 3 with cavity 3" dropped). Fallback ±40 %, then ±50 % if fewer than 10 sets qualify.
- 13 Oct (Ruben): cross-validation design settled as in `docs/cv_explained.md`: outer 4 folds of 3
  boxes over the 12 non-test boxes, 5 repeats from eligible partitions (seed 369), inner split
  inside the training boxes only to choose settings (its score never reported as performance),
  bootstrap over boxes for intervals, paired for comparisons, sealed test once at the end.
- 21 Oct (Ruben): design A replaces the nested design of 13 Oct. 20 repeats of the 4-fold box
  cross-validation on the 12 non-test boxes; PRESS averaged over the repeats gives one RMSECV
  curve; one number of PLS components chosen from it (rule to settle when the curves are seen:
  a clear minimum, or the Haaland and Thomas 1988 F-test at α = 0.25, whose limitation is that it
  treats fruit of one box as independent); one fit on all 12 boxes; test once. No inner split.
  The cross-validation score at the chosen number is slightly optimistic; stated in the thesis.
  AUC is the main classification score, with recall per grade and precision at a cut-off of 0.5
  fixed in advance. Ordinal model on PLS scores. Penalised logistic regression dropped (does
  about what PLS-DA does in p ≫ n data). SVM not added. `docs/cv_explained.md` section 4.
- 13 Oct (Ruben): a model fitted on Hugo's 2425 validation fruit is not evaluated on his 2425 test
  fruit (same ten orchards). Main models are fitted on 2526.

- 23 Oct (Ruben): no cropping of the latent grid. Faulty-volume rule approved in principle
  (`docs/outlier_rule.md`). Number of PLS components by the van der Voet randomisation test with
  signs flipped per box, p > 0.10 (`docs/cv_explained.md` section 4), chosen before any curve.
  PERMANOVA assumptions written before running (`docs/exploration_plan.md`).

## Unverified assumptions in code

| Where | Assumption | Verify against |
|---|---|---|
| `src/pearlatent/labels.py` `fruit_keys` | G_opt and I_opt fruit come from orchards G and I, so they share box `2425_G` / `2425_I` (decided with Ruben 3 Oct). Supported by the preprint, section 2.2, as summarised in Ruben's other Claude session on 5 Oct (the two orchards with optimal storage have fruit in three boxes); not yet read in the preprint itself | preprint section 2.2, Hugo |
| `src/pearlatent/labels.py` `_STORAGE` | 2526 storage condition unknown; recorded as `after_storage` | Hugo |
| `src/pearlatent/encoding.py` `load_encoder` | default checkpoint `checkpoint.pth` is the encoder of the preprint (not `best.pth`). Hugo sent only `best.pth` (epoch 430, seed 369) and `vae_hyperparameter.json`; the loader now defaults to `best.pth` | Hugo: is `best.pth` the model of the preprint? |
| `pyproject.toml` | workstation driver supports CUDA 12.8 wheels (only matters for the ResNet-18 baseline) | `nvidia-smi` on the workstation |

Note on names: in 2425 the `box` column holds the orchard letter (60 to 90 fruit). Per the
preprint a physical box is 30 fruit of one orchard in one storage group. Grouping by orchard is
stricter than grouping by box, so the splits stay valid; only the name is loose.

## Session log

Newest first. One entry per session, three to six lines.

### 2026-10-28 — three representations side by side, long sections of the grid
- Decided (Ruben): the height-radius map is exploratory; the real grid is shown as long
  sections; the CT comparison is the existing 4³ block mean; notebook 07 is shown live.
- Notebook 07: representation dropdown (Hugo, random encoder, CT at 32³), section 1 table for
  all three, section 4 long sections with linked spectra and PCA; controls keep their values.
- Found: PC1 is size and stalk in all three representations; only Hugo's latent has a
  component (PC4) that follows the grades (|r| about 0.25).
- New questions 40 (Med3D choices) and 41 (stalk boundary at voxel level).

### 2026-10-27 — one height-radius rule, latent spectra, methods record
- Decided (Ruben): radius divided by the outermost voxel or cell; every fruit counts once in
  a group mean; two latent radius methods; training-box grade 0 mean as the correction in
  the models; deep learning waits; methods kept in `docs/methods_step0_and_maps.md`.
- Notebook 07 section 3: spectra with linked drag selection, corrections, 3D surfaces, PCA
  and scree plot on the selected bins; notebook 06 shows both latent views.
- Found: map groups differ in stalk, body size and box (question 38).
- Meeting document: update of 27 October and walkthrough of notebooks 06 and 07.

### 2026-10-26 — faulty volumes decided, Step 0 explained, meeting material ready
- Decided (Ruben): visual inspection decides exclusions, neck cuts out, 35 volumes excluded;
  PERMANOVA, t-SNE, UMAP and MDS removed; Step 0 decodes `mu`; browning 1 = kernel membranes.
- Exclusions uneven over grades (p = 0.006 within box); the PCA screen alone misses most cuts.
- Notebooks 05, 06, 07 rerun on 377 fruit; Step 0 numbers barely changed; residual shows a
  few percent of the darkening repaired; latent maps show browning 2, 3, not 1.
- New: excluded-fruit figure, PC1-PC2 panels, latent height-radius view; meeting document
  complete. Next session: ResNet-18 baseline design.

### 2026-10-23 — outlier rule, component rule, Step 0, exploration notebook
- Decided (Ruben): no cropping; faulty-volume rule (missing part or failed preprocessing, never
  the defect); van der Voet test per box for the number of PLS components; PERMANOVA plan first.
- Notebook 05 outlier panel and contact sheet; `pearlatent.descriptors` on all open fruit.
- Notebook 06 (Step 0): mean core darkening kept in the reconstruction (retention about 0.9 to
  1), fine texture smoothed; Hugo's training fruit not reconstructed better than unseen fruit.
- Notebook 07: PC1 is body size and stalk; box R² 0.10, grade R² 0.02 (mostly spread).
- Faulty volumes cluster in 2526 boxes I and L; per-fruit decisions pending (Ruben).

### 2026-10-21 — Hugo confirms the split; design A; notebook 05
- Decided (Ruben): design A (repeated box cross-validation, one choice of components from the
  averaged curve, one fit, test once); 20 repeats; penalised logistic regression dropped; AUC
  main score; next work: exploration, outlier rule, Step 0.
- Hugo confirmed his VAE split; reply with the design drafted outside git; meeting 29 Oct 11:00.
- Notebook 05: PCA, filters, parallel coordinates, single-fruit view, T² and Q screen. Fixed the
  `row` / position mix-up. Found the encoder's edge band (zero padding).
- Ruben: about 13 PCs at the elbow, season visible on PC1/PC2; outliers 2526_N29, 2526_K27,
  2526_L29 (2 broken, 1 odd shape) and 2425_I44, 2425_C24.

### 2026-10-17 — encoding checked, move to a Mac prepared
- Checked the full encoding run `20261013T171306Z_hugo_vae`: 940 open + 153 sealed fruit,
  478 decoded, all finite. Random-encoder run started 15 Oct but empty.
- Wrote the Mac setup checklist. No code changed.

### 2026-10-05 — Hugo's answers, data checks, pooling and encoding code
- Decided (Ruben): hold out 2526 test boxes, rule to agree with Hugo first, folds wait; within-box
  comparison removed from charter rule 1; rotational symmetry test dropped; encode on the laptop.
- Plan files updated after Hugo's answers; added volume quality check, exclusion table, pooling
  and encoding code with tests; notebooks 01 (section 5), 02, 03, 04.
- Found: 2425 defect = storage group; rot explains 11 of 175 fruit with browning 3 and cavity 3;
  all volumes pass; most 2526 volumes include the stalk within the 128-voxel length.
- Corrected the same day: "the 2526 volumes include the stalk" to "most" (8 of 12 random).
- Meeting notes in `docs/meeting_2026-10-15.md`.

### 2026-10-03 — label inventory
- Decided: G_opt and I_opt fruit join boxes `2425_G` / `2425_I`; storage group in a separate
  `storage` column. Box letters are unrelated across seasons (Ruben).
- Added label loading, volume listing, 16 tests and notebook 01; all checks pass on real data.
- Verified on the label files: `defective`, `binary_123`, `binary_23` follow the assumed rules
  exactly; 2425 `non-consumable = defective ≥ 2` (rot plays no role). 17 fruit without a volume
  confirmed. (An earlier note in this session that the binary columns disagreed was wrong.)
- Ruben allowed Claude to read notebook outputs; jq installed for the `marimo-pair` skill.
  Mail with the open questions sent to Hugo.

### 2026-10-03 — Hugo's repository read
- Verified volume format, file-name pattern, encoder construction and receptive field from code.
- Found that Hugo's classifier splits within boxes; recorded as a point for Hugo.
- Remaining blocker: encoder weights.

### 2026-10-02 — setup
- Created the repository skeleton and agent rulebook.
- Decided: uv project (not per-notebook sandboxes), Polars, PyTorch, marimo with project config,
  data outside git, molab not used for data.
