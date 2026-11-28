# HANDOFF.md

The living state of the code work. Claude reads this at the start of every session (a hook prints
the "Current state" section) and updates it with `/handoff` at the end. Ruben may edit it freely.

## Current state

- **Date:** 2026-11-27 (Mac), second session of the day
- **Phase:** ResNet-18 baseline: code written and smoke-run on the Mac; the full run waits for
  the lab workstation. Next meeting with Hugo: about **8 December**. Ruben promised: the
  preprocessing settled (done, first session of 27 November) and the ResNet-18 close to running
  (code ready; full run blocked on the workstation). Material for Hugo: `notebooks/23_stretch_and_size.py`
  (7 figures, `derived/figures/23_*.png`); the ResNet has no results to add to it.
- **Done this session (commits 91ccd0d to 60cb023, pushed):**
  - `src/pearlatent/resnet_baseline.py`: Hugo's `ResNet3D` (`train_clf_3d.py` at 183112f,
    blocks 2-2-2-2) imported from his clone, not copied; `tqdm` added with `uv add` because that
    file imports it. Open 2526 fruit without faulty volumes (315); repeat 1 of
    `outer_folds.csv` (4 folds of 3 boxes); per fold 2 of the 9 training boxes drawn with seed
    369 for early stopping (about 185 train, 50 validation, 77 test fruit); Hugo's defaults
    (AdamW, learning rate 1e-4, cosine, at most 100 epochs, patience 20, best model only after
    epoch 10, batch 16); no augmentation, no pseudo-labels. Output per run in
    `results/<stamp>_resnet18_binary23/`: `oof_predictions.parquet`, `log_fold<k>.csv`,
    `best_fold<k>.pth`, `fold_plan.csv`, `run.json`.
  - Ordinal step of notebook 12 on the run 1 out-of-fold score (Ruben): `ordinal_step`,
    `ordinal_for_run`, on the probability and on its log-odds; writes
    `ordinal_predictions.parquet` and `ordinal_summary.csv` (all fruit and by cause). Rerun on a
    copied run: `--ordinal-for results/<run>`. `grade_summary` moved into `pearlatent.ordinal`.
    The network itself trains on `binary_23` only.
  - Run 2 built (`--target grades`, Ruben: cumulative head): Hugo's `fc` replaced by one score
    and 3 ordered thresholds (P(>= 1) >= P(>= 2) >= P(>= 3) by construction), loss the mean
    binary cross-entropy over the thresholds; same folds, validation boxes and settings as run
    1; writes grade probabilities per fruit, `grade_summary.csv` (all fruit and by cause), and
    `probability` = P(>= 2) for the binary_23 AUC. Smoke run (4 folds, 1 epoch, 6 per box):
    `results/20261127T163132Z_resnet18_grades_smoke`, ran end to end.
  - Methods record `docs/methods_resnet.md` (28 Nov): design, network, training, both runs, with
    worked examples; no results.
  - Charter updated (Ruben): width 16; 2526 fruit only, no 2425; two runs: run 1 `binary_23`,
    run 2 a network trained on the `defective` grades with an ordinal output.
  - Tests: 12 new in `tests/test_resnet_baseline.py`; 194 pass. Smoke runs (CPU, 1 epoch):
    `results/20261127T150858Z_resnet18_binary23_smoke` (fold 1, 3 fruit per box, 18 s) and
    `results/20261127T152800Z_resnet18_binary23_smoke` (4 folds, 6 per box, 2 min, ordinal step
    ran). Their numbers mean nothing (1 epoch). Rough estimate: full run on the Mac CPU about
    10 hours.
- **Next step:** when Hugo gives workstation access: on the workstation `git pull`, `uv sync`,
  `nvidia-smi` (GPU free? driver supports CUDA 12.8 wheels?), fill `config/local.toml`, then in
  tmux a short check `uv run python -m pearlatent.resnet_baseline --folds 1 --epochs 2 --amp`,
  then the full run `uv run python -m pearlatent.resnet_baseline --amp 2>&1 | tee resnet_run1.log`.
  Then run 2 the same way with `--target grades`. Copy the run folders to the Mac (weights
  stay out of git).
- **Blocked on:** the workstation (Hugo arranges it) for the full ResNet-18 run.
- **Done in the first session of 27 November (commits 4881db1 to c532094):** Hugo's 2526
  originals unpacked (`ct_orig_dir_2526`), uniform rescaling verified; `pearlatent.real_size`
  (stretch factor and real size), `pearlatent.stretch_check`; EPO on the measured stretch factor
  for the latent rows (final latent height-radius mean + SD, 2 LVs: AUC 0.958, 0.946 to 0.968;
  Med3D 0.958); methods sections 20 to 23; notebook 23; outcome of the 24 November meeting in
  `docs/meeting_2026-11-24.md` section 8.
- **Still to do after the ResNet:** N-way PLS (charter row); counterfactuals again with the
  stretch-EPO direction (question 55); sealed test F, J, M once, all final rows.
- **Still to fix:** methods section 18 says "the core change is the smallest" (the story is
  corrected); an empty cell at line 284 of `notebooks/20_story.py`; the variant-B results of
  section 17 were read with the old 128³ size measure.
- **Reference, still valid:**
  - Encoding run `20261013T171306Z_hugo_vae`: open part 940 fruit (346 2526, 594 2425);
    sealed part `derived/sealed/latents/...`, 153 fruit. CT at 32³ run `20261005T142026Z`.
    Tissue features `20261102T151947Z`, Med3D `20261102T185107Z`. Runs in use:
    `pearlatent.runs` (`PLSDA_BINARY23_RUN_DIRS`, `EPO_STRETCH_RUN_DIRS`, `REAL_SIZE_RUN_DIR`,
    `STRETCH_CHECK_RUN_DIR`).
  - Test boxes **F, J, M** (87 fruit), never opened. Fold partitions in
    `derived/outer_folds_20/outer_folds.csv`.
  - Faulty volumes: `derived/volume_decisions.csv` through `drop_faulty_volumes`; 346 → 315
    open 2526 fruit.
  - Mac: `/Users/rubenhoeven/code/pear-latent-thesis`, Python 3.12.15, torch 2.14.1 (CPU),
    marimo 0.25.1; Hugo's repository `../Synthetic_CT_pear` at 183112f; MedicalNet
    `../MedicalNet` at 20f76aa.
  - Working note: with `marimo edit --watch` open, quick edits in a row can be overwritten;
    after code changes run `graphify update .`.

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

Answered and removed on 28 Nov: 70 (sealed test: one ResNet trained on the 12 open boxes with 2 validation boxes, Ruben). Answered and removed on 27 Nov: 68 (run 2 uses a cumulative head, Ruben; built), 66 (validation group: 2 whole boxes per fold, fixed for the whole training of that fold, drawn with seed 369; clean nested version kept, Ruben), 5 (threshold is a market decision, Hugo and Ruben, 24 Nov), 64 (outcome written in `docs/meeting_2026-11-24.md` section 8), 18 (originals received, uniform rescaling verified, `pearlatent.real_size`), 52 (cause: the stretch to 128 voxels plus real size, methods sections 20 to 22), 59 (stretch part is an artefact, real size is a real risk factor; Ruben), 62 (replaced by EPO on the measured stretch factor, methods section 23). Answered and removed on 23 Nov (from the outcome of 29 Oct, `docs/meeting_2026-10-29.md`, never removed then): 2, 4, 12, 13, 16, 17, 30, 31, 33, 34, 35, 37, 38. Answered and removed on 21 Nov (Ruben): 57 (core region: height 0.05 to 0.60 over the body, all radii; route then dropped). Answered and removed on 7 Nov (Ruben): 39 (height-radius rows added beside the full grid, `docs/methods_plsda.md` section 10), 49 (notebook 10 built and saved), 50 (runs of 2 November deleted). Answered and removed on 2 Nov (Ruben): 40 (Med3D choices, see Current state and `pearlatent.med3d_features`). Removed on 3 Nov: 44 (`docs/med3d_open_points.md` updated), 48 (nnU-Net dropped, Ruben). Answered and removed on 5 Oct (Hugo, October 2026): the 17 unscanned 2526 fruit are excluded
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

9. Lab workstation (only for the ResNet-18 baseline now): GPU model, NVIDIA driver version,
   operating system, `uv`, tmux.
11. Cavity-label issues: grades that disagree with visible voids (to back with the cavity
    cross-check, if built).
21. The preprint reports MAE 0.0022 on the test set (section 3.1.1) and MAE 0.0025 on the
    validation set (Table D.1), with the same SSIM and PSNR. Which numbers belong to which set?

For Ruben, added 21 Oct:
24. Fit PCA and PLS on all 32,768 cells or only on cells inside the fruit (body masks in
    `masks.npy`, for example cells inside the body of at least half the fruit)? Would remove
    the background and the edge band.
27. Bootstrap on the test set: the 3 test boxes are too few for a bootstrap over boxes, so the
    test interval resamples fruit (`docs/cv_explained.md` section 8). Hugo's mail speaks of
    paired bootstrap intervals in general; state this difference at the meeting.

For Ruben, added 2 Nov:
45. Tissue features: the partial-volume rim under the skin is always classed as low-density
    tissue (darker than its local mean). Accepted for now; a sensitivity row that erodes the
    tissue mask by one voxel would show whether it matters.
46. Med3D: the same rim (voxels just above 0.1) enters as values around −5. Same question as 45.
47. Optional extra row: the density-distribution-model residual of Van De Looverbosch et al.
    (2022, anomaly detection, Youden threshold). Wanted?

For Ruben, added 9 Nov (audit session):
55. The counterfactual direction (cavity-free, latent grid, 5 LVs) adds a stalk when a fruit is
    moved towards healthy (2526_C23). Remove the stalk part? Options (`docs/methods_plsda.md`
    section 14): fit on body cells only; project b orthogonal to the latent direction that
    predicts stalk length on healthy fruit; or external parameter orthogonalisation (EPO) of the
    latent before PLS. The stalk measure has the limits of check D (2 November).
    Ruben, 9 Nov: options 2 (projection) and 3 (EPO) preferred, under the counterfactuals;
    after the meeting of 17 November, together with a better stalk length (question 41:
    body-stalk boundary at voxel level; question 18: fruit length before rescaling).
    21 Nov: EPO (healthy PC 1) tried; the stalk patch still appears (C23, K07) and the outline
    change dominates (`docs/methods_plsda.md` section 18). Projection on a stalk direction not
    tried. Open: more EPO components, or accept and report.
    27 Nov: the stalk link is the stretch artefact; EPO on the measured stretch factor removes
    it from the score (methods section 23). Counterfactuals to be redone with that direction.
56. 2526_I09 and 2526_N17 end in a flat, slanted neck in the CT, like the neck cuts excluded on
    26 October, but were kept then. Were some small neck cuts missed? Changes no result now.

For Ruben, added 20 Nov:
58. The SVM score shows no stalk or size link on healthy fruit. Check that this is not because
    the RBF score is nearly constant among healthy fruit (spread of their scores).
For Wouter Saeys, added 21 Nov:
60. EPO with the healthy fruit as the nuisance matrix D (no paired measurements under a varying
    external parameter): an accepted variant? How many components (we use 1; more began to
    cost AUC)? Could D be built by simulating the rescaling to 128 voxels (depends on
    question 18)? 27 Nov: now replaced by EPO on a measured external parameter (stretch factor,
    regression at equal real volume, healthy fruit, methods section 23); ask Wouter Saeys
    whether this form is accepted.

For Ruben, added 21 Nov:
61. Only 43 to 73 % of a counterfactual step comes back after decode and encode. Does the
    decoder ignore part of the change of `mu`, or does the encoder not return to the moved
    `mu`? Possible check: compare the step with the change of the re-encoded `mu` cell by
    cell, inside and outside the body.

For Ruben, added 9 Nov:
54. Scale the height-radius rows per bin? The reason for centring only (background cells) does
    not apply to them, because their bins hold body cells only. Scratch, dividing by the grade 0
    SD of the training boxes: CT height-radius +0.026 (0.006 to 0.052), latent +0.007 (−0.005 to
    0.023), LVs kept at 3 (`docs/methods_plsda.md` section 12). A decision about the model rows.

For Hugo (meeting 24 Nov), added 7 Nov, updated 20 Nov:
53. Grade 0 against grade 1 and grade 2 against grade 3 hardly separate on every
    representation (pairwise AUC 0.6 to 0.7), grade 1 against 2 does (0.8 to 0.9). Is grade 1
    browning too small for the CT resolution, or is the 0/1 boundary of the visual grading
    uncertain? Can the cut-open photographs show it? 20 Nov (Ruben): there are no cut surfaces or
   photographs for 2526; grading was on one or two cuts, so part of a lesion is not seen. Ask
   Hugo how the 2526 fruit were graded (which cuts, how many).
   23 Nov (Ruben, answered): Hugo's grading decision table; not every region cut, lesions can
   be missed; label noise at 0/1 and 2/3 exists, not quantifiable. Thesis: grade 1 failure is
   partly label noise, partly the encoder.

For Ruben and Hugo, added 28 Oct:
41. Stalk length and body size are measured on the 128³ volumes, but the body-stalk
    boundary is decided on the 32³ CT grid (slices with fewer than 4 cells), so it moves in
    steps of 4 voxels and the neck tip counts as stalk. Define the boundary at voxel level
    (for example fewer than 64 fruit voxels per slice) and compare on all fruit? A change
    to a method rule; after the meeting.
42. Two exclusion mechanisms exist: `derived/exclusions.csv` with
    `pearlatent.exclusions.apply_exclusions` (documented as the one every notebook uses) and
    `derived/volume_decisions.csv`, read by `pearlatent.fruit_table.drop_faulty_volumes` in
    notebooks 06 and 07. `exclusions.csv` had fallen behind (35 against 41 faulty volumes) and
    was rebuilt on 9 Nov by notebook 02. The model notebooks use `drop_faulty_volumes`;
    notebook 06 keeps Hugo's training sample on purpose, which `apply_exclusions` in mode
    "primary" would drop. `test_box_draw` now uses `drop_unscanned`.
43. Remaining refactors (28 Oct): one shared map-and-spectrum chart for sections 3 and 4 of
    notebook 07; `pca_on_selection` into `src/pearlatent/` with a test; 8 ruff warnings in
    the notebooks; "Status: draft" in notebooks 00 to 04.

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
    7 Nov: the height-radius rows use `pearlatent.step0.latent_map_sums`, not this code.

For Hugo, added 27 Nov:
63. Voxel size of the 2526 originals: the headers say 1 mm; the preprint gives 0.6267 mm (for
    2425). Ruben says the same procedure was followed; a written confirmation from Hugo would
    close the `# UNVERIFIED` in `pearlatent.real_size`. Only the mm values depend on it.

For Ruben, added 27 Nov:
65. On healthy fruit the latent and CT height-radius scores rise with real size (+0.30 to
    +0.39 after EPO). Either the model uses size, or larger grade 0 fruit hide browning that the
    grading missed; the labels cannot tell. State both in the thesis (methods section 23).

For Ruben, added 27 Nov (ResNet-18):
67. Ordinal step on the run 1 output: on the probability or on its log-odds? Both are computed
    (`ordinal_summary.csv`); choose after the full run, before reading the grade results as final.
69. The ResNet sees the 128³ volumes with the stretch built in; no EPO applies to it. Run the
    partial Spearman of `pearlatent.stretch_check` on its healthy-fruit out-of-fold scores after
    the full run?

## Decided

- 28 Nov (Ruben): for the sealed test F, J, M the ResNet is one network trained on the 12 open
  boxes, 2 of them as validation boxes, not the mean of the 4 fold networks
  (`docs/methods_resnet.md` section 12). Code for that final fit still to write.
- 27 Nov, second session (Ruben): ResNet-18 width 16 (Hugo's default); open 2526 fruit only, no
  2425 fruit in its training; run 1 trained on `binary_23` (same task as the PLS-DA rows), with
  the ordinal step of notebook 12 on its out-of-fold score; run 2 a separate network trained on
  the `defective` grades with an ordinal output. Charter, Methods, updated. Early stopping on 2
  whole validation boxes per fold (the same 2 for every epoch of that fold), not on the held-out
  fold: the stopping choice never sees the held-out boxes (nested), at the cost of about 50
  training fruit per fold.
- 27 Nov (Ruben): the stretch to 128 voxels is an artefact and is removed; real fruit size is
  biology (larger fruit brown more) and stays. EPO on the measured stretch factor (one
  direction, at equal real volume, healthy training fruit, per fold) for the latent rows only;
  the other rows' scores do not follow the stretch. Variant B (healthy PC 1) not used further.
  LVs with EPO: grid 3, height-radius 2, height-radius mean + SD 2 (by eye, before the test
  boxes). The 2425 originals are not needed (primary design is 2526). AUC within thirds of real
  size (cut-offs fixed before looking) reported beside the AUC.
- 2 Nov (Ruben): LVs chosen by eye at the elbow of the mean RMSECV curve; Youden cut-off
  reported beside 0.5; tissue-feature row (Van De Looverbosch et al. 2020); Med3D ResNet-18
  with z-score on the fruit tissue and 0 outside the fruit; iPLS selection stops at the layers,
  height bins description only; SVM not added; random encoder row dropped (Hugo, 29 Oct).
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
| `src/pearlatent/tissue_features.py` `tissue_masks` | the paper's MATLAB adaptive threshold (no sensitivity given) means "below the plain local mean of the tissue in a 31³ window" | Van De Looverbosch et al. 2020 authors, or MATLAB `adaptthresh` defaults |
| `src/pearlatent/real_size.py` `VOXEL_SIZE_MM` | the 2526 originals have 0.6267 mm voxels (headers say 1 mm; preprint procedure, Ruben 27 Nov) | Hugo (question 63) |
| `pyproject.toml` | workstation driver supports CUDA 12.8 wheels (only matters for the ResNet-18 baseline) | `nvidia-smi` on the workstation |

Note on names: in 2425 the `box` column holds the orchard letter (60 to 90 fruit). Per the
preprint a physical box is 30 fruit of one orchard in one storage group. Grouping by orchard is
stricter than grouping by box, so the splits stay valid; only the name is loose.

## Session log

Newest first. One entry per session, three to six lines.

### 2026-11-27 (second session) — ResNet-18 baseline written and smoke-run
- Decided (Ruben): width 16; 2526 fruit only (no 2425); run 1 on `binary_23`, run 2 a network
  trained on the grades with an ordinal output; the ordinal step of notebook 12 also on run 1.
- `pearlatent.resnet_baseline`: Hugo's ResNet3D imported from his clone, repeat 1 of the outer
  folds, 2 validation boxes per fold, Hugo's training defaults, out-of-fold probabilities.
- CPU smoke runs on the Mac ran end to end (numbers meaningless); 190 tests pass; pushed.
- Full run waits for the workstation.

### 2026-11-27 — originals, stretch against real size, EPO on the stretch factor
- Hugo (24 Nov): height-radius rows agreed; size real, correct the preprocessing; threshold is a market decision.
- Decided (Ruben): stretch removed by EPO on the measured stretch factor, latent rows only;
  real size kept; LVs 3 / 2 / 2; 2425 originals not needed.
- Hugo's 2526 originals unpacked; uniform rescaling verified; stretch factor and real sizes saved.
- Found: old check C "size" was the stretch; real size predicts defects (AUC 0.79); only latent
  scores follow the stretch; models stay at AUC 0.90 to 0.98 within size thirds.
- Final latent rows with EPO saved (height-radius mean + SD 0.958); notebook 23 with 7 figures
  for Hugo; methods sections 20 to 23 with the reasoning and its limits.

### 2026-11-23 — meeting brief, open questions pruned, EPO figures
- Decided (Ruben): grading label noise accepted as is (53); Ruben contacts Wouter Saeys
  directly (60); ask Hugo for the scale factor per fruit, used as a measurement, not to
  reprocess (18). Figure for check C before and after EPO added; iPLS curve figure not wanted.
- Built: `docs/meeting_2026-11-24.md`; notebook 22 sections 5 (PCA) and 6 (check C figure).
- Found: 13 answered questions had stayed open since 29 October; EPO removes only the
  straight part of the size effect (smallest fruit overshoot, largest still grouped).
### 2026-11-22 — story notebook, Youden mirrors, iPLS selection
- Decided (Ruben): `binary_23` EPO model at 3 LVs; mirror also across the Youden cut-off;
  iPLS selection heights 1, 2, 4, 5, 7, 10, 11, all rings, 3 LVs; ready to present Tuesday.
- Built: notebook 20 (story), `--flip-at`, Youden and predictions in notebook 22, notebook 13
  sections 3 to 5, `step0.cell_bins`, mirrored-step figure; methods section 19.
- Found: selection keeps the AUC (0.966) and the stalk link; Youden mirror moves only the grade
  3 fruit across; painted changes show core darkening in A03 and K07.
### 2026-11-20/21 — core-only dropped, EPO on healthy fruit, EPO counterfactuals
- Decided (Ruben): core-only rows dropped (no gain, size link worse); EPO variant B kept for
  latent rows and counterfactuals; `vae.decode(mu)`, no sampled reconstruction.
- Built: `pearlatent.core_region`, `pearlatent.epo`, fold hooks in `plsda`, notebooks 21 and
  22, `--flip` in `counterfactual_run`, notebook 17 sections 4 and 5; methods sections 16 to 18.
- Found: EPO removes the check C link on latent rows at equal AUC; counterfactuals keep 43 to
  73 % of each step, change mostly outline and stalk, the same with and without EPO.
### 2026-11-20 — SVM check row, walkthrough, plan for the core-only model
- Decided (Ruben): core-only model as a prior choice and quick alternative to a better stalk
  measure; no re-encode; story notebook 20 for Ruben and Hugo; plan in Current state.
- Built: `pearlatent.svm`, notebook 19, run `20261120T090152Z`; methods section 15. Found: SVM
  0.967 against PLS-DA 0.957 (interval includes 0), grade 1 not rescued, no check C link.
- Scratch: stalk and body size correlate −0.85 on healthy fruit (one effect).
- Walkthrough of sections 0 to 8 with questions. Meeting moved to 24 November.

### 2026-11-09 — audit, cavity-free browning, counterfactual smoke test
- Decided (Ruben): `binary_123` and severity plans dropped; grade 0 corrections not in models;
  counterfactual steps along the model direction on a t scale, then re-encode; I09 rejected.
- Audit: results reproduce; fixed docs, `drop_unscanned`, stale `exclusions.csv`, run records.
- Built: `pearlatent.targets`, notebook 14 (cavity-free runs), `pearlatent.counterfactual`,
  `counterfactual_run`, notebook 17. Found: browning without cavity detected (latent mean + SD
  0.942); the counterfactual direction adds a stalk (question 55).
- Next: inspect the three running counterfactuals, then question 55.

### 2026-11-09 — mean and SD rows, interior spread, encoder SD, meeting figures
- Audit before the meeting: the six saved rows reproduce exactly; signals not followed up listed.
- Built: SD per height-radius bin (two rows, 3 LVs), `pearlatent.interior`, `pearlatent.logvar_run`,
  notebooks 15 (interior spread), 16 (all volumes, reverse check), 18 (two AUC figures).
- Found: latent mean + SD level with Med3D; grade 1 visible in the CT interior SD, not in the
  latent; the encoder SD adds nothing to `mu`.
- Deleted (Ruben): the 8 notebook 14 runs `20261109T171155Z` to `20261109T171509Z` saved with
  `dirty: true`; each had predictions identical to its clean re-save (`20261109T1717..Z`).
- Next: ResNet-18 design note and the workstation mail to Hugo.

### 2026-11-07 — comparison saved, per-grade PLS-DA fails for grades 1 and 2, height-radius rows
- Decided (Ruben): check B dropped (ICC beside check A); one-versus-rest PLS-DA per grade kept as
  a negative result; height-radius bins reopened as two rows (3 LVs each); counterfactuals on
  the full latent with the general model.
- Built: notebooks 10 (paired comparison), 11 (per grade), 12 (ordinal), 13 (iPLS on
  height-radius); modules `ordinal`, `height_radius`; all saved as runs.
- Found: grades 0/1 and 2/3 hardly separate on any row; latent height-radius AUC 0.927, +0.040
  over the grid; its score follows size and stalk on healthy fruit (question 52).
- Next: b maps, then counterfactuals.

### 2026-11-03 — all four PLS-DA rows saved, nuisance checks, methods record
- Decided (Ruben): Med3D at 7 LVs; the 2 Nov runs of the latent and CT resaved at 4 and 5 LVs
  so every run has the Youden cut-off.
- Fixed: `run.json` of notebook 08 always recorded `scale: false`; runs saved from clean
  commits by running the notebook headless with the choices filled in.
- Found: AUC Med3D 0.958, tissue features 0.938, latent 0.887, CT 0.839; within-box AUC close
  to the overall AUC in every row; browning-only grade 2 the weakest group everywhere.
- Written: `docs/methods_plsda.md`. Nuisance numbers not yet saved as a run (question 49).

### 2026-11-02 — first PLS-DA models, tissue features, Youden cut-off, Med3D
- Decided (Ruben): LVs at the elbow (latent 4, CT 5, tissue features 7, Med3D 7); Youden
  cut-off beside 0.5; iPLS selection stops at the layers; SVM out; Med3D input z-scored on the
  fruit, 0 outside (nnU-Net), cavities kept.
- Built: notebook 08 (PLS-DA), VIP, nuisance checks, notebook 09 (iPLS), tissue features,
  Med3D features. Meeting of 29 Oct logged; next meeting 17 November.
- Found: latent AUC 0.887, CT 0.839, tissue features 0.938; InstanceNorm puts whole-fruit
  intensity into every latent cell. Med3D scores not computed yet.
- Two Med3D input bugs found from the input figure (background −6.5; cavities set to 0),
  fixed before any score was seen.

### 2026-10-28 — three representations side by side, long sections of the grid
- Decided (Ruben): the height-radius map is exploratory; the real grid is shown as long
  sections; the CT comparison is the existing 4³ block mean; notebook 07 is shown live.
- Notebook 07: representation dropdown (Hugo, random encoder, CT at 32³), section 1 table for
  all three, section 4 long sections with linked spectra and PCA; controls keep their values.
- Found: PC1 is size and stalk in all three representations; only Hugo's latent has a
  component (PC4) that follows the grades (|r| about 0.25).
- New questions 40 (Med3D choices) and 41 (stalk boundary at voxel level).
- Refactor: one fruit table (`pearlatent.fruit_table`) and the runs in `pearlatent.runs`
  for notebooks 05 to 07, output checked identical; questions for Hugo listed at the bottom
  of the meeting document. New questions 42 (exclusion mechanism) and 43 (refactors left).

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
