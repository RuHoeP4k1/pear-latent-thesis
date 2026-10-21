# HANDOFF.md

The living state of the code work. Claude reads this at the start of every session (a hook prints
the "Current state" section) and updates it with `/handoff` at the end. Ruben may edit it freely.

## Current state

- **Date:** 2026-10-21
- **Phase:** data exploration on the saved latents (notebook 05). Evaluation design fixed
  (design A, 20 repeats; see "Decided"). No model fitted on labels yet. Hugo confirmed his VAE
  split (21 Oct) and gave the green light for plots of 2526 labels (B4, Ruben 21 Oct).
- **Done this session (21 Oct, Mac):**
  - Hugo's mail of 21 Oct: split file "exactly matches my implementation" (same indices with
    PyTorch 2.7.1, 2.7.1+cuda and 2.14.1; same label file). B2 and open question 19 closed;
    `UNVERIFIED` removed from CLAUDE.md section 3 and `hugo_split.py`. He described the CV as one
    tuning fold per repeat and mean ± standard deviation of test accuracy across repeats.
    Reply in `derived/mail_hugo_2026-10-21/mail_to_hugo.md` (outside git): design A in four
    steps, test used once, test boxes kept as drawn, meeting Thursday 29 Oct 11:00 (moved from
    10:00).
  - Repeats raised from 5 to 20: rule amendment `docs/test_box_rule.md` section 6.4 item 5,
    committed before the draw; `python -m pearlatent.extra_repeats` → `derived/outer_folds_20/`
    (repeats 1 to 5 reproduced and kept; 15 more from the other 181 eligible partitions, seed
    369). K and O share a fold in 13 of 20 repeats, D and N in 11; 55 of 66 box pairs share a
    fold at least once. Use `derived/outer_folds_20/outer_folds.csv` from now on.
  - Design A and dropping penalised logistic regression written into `docs/cv_explained.md`
    (sections 4, 5, 8), `CHARTER.md` (Methods), `CLAUDE.md`, `docs/test_box_rule.md`.
  - `notebooks/05_latent_sanity.py` (draft, runs as a script): per-fruit numbers per box; mean
    and standard deviation per cell; PCA (50 components, exact solver, fitted on 2526 or all open
    fruit, all fruit projected); score plot with filters on group, defective grade and cause,
    colour and symbol choice; Plotly parallel coordinates with axis brushing (adds `plotly`);
    view of one fruit (CT, latent, latent minus mean); Hotelling T² and Q residual screen with
    95 % and 99 % limits.
  - Bug fixed in notebook 05: the manifest column `row` is the row among all 1,093 volumes, not
    the place in `latents.npy`. Notebooks must use the place in the file (notebook 05 adds
    `position` with `with_row_index`). No other code used `row` that way (checked).
- **Hugo's second answer (21 Oct):** agrees with design A ("makes sense to me and is even more
  rigorous ... a solid plan"): cross-validation and the RMSECV curve choose the number of latent
  variables; the test set is fixed and used only to report the final model, fitted on all 12
  boxes; paired bootstrap confidence intervals reported. Meeting moved to Thursday 29 Oct, 11:00.
  The reply Ruben sent (21 Oct, 14:06) says the number of latent variables is "chosen at the
  minimum of the averaged curve"; `docs/cv_explained.md` section 4 still leaves the rule open
  (minimum, elbow, or Haaland F-test). To settle (open question 26).
- **Results seen (21 Oct):**
  - PCA on the 346 open 2526 fruit: cumulative explained variance PC1 22 %, 2 PCs 31 %, 10 PCs
    50 %, 50 PCs 74 %, 100 PCs 84 %, 200 PCs 94 %; no elbow after PC2 in the cumulative curve.
    Ruben: elbow of the scree plot at about 13 PCs; many PCs needed, as expected for a latent
    with KL weight 1e-6.
  - Ruben: PC1 against PC2 shows a season difference; not all label colourings checked yet. The
    parallel coordinates showed little.
  - Edge band in the mean latent: background cells within one to three cells of the grid edge
    differ from the inner background by about 0.5 to 1 unit (layer 0: −0.31, layer 1: +0.18,
    inner background: −0.43; fruit cells about 10). Cause: zero padding (`padding=1`) in every
    convolution of Hugo's encoder (`component/vae.py`, 183112f). The fruit touches the grid at
    z = 0 and z = 31, so calyx and stalk cells mix fruit with this edge effect.
  - Outlier screen, A = 13, PCA fitted on 2526, 99 % limits (computed by Claude, same code as the
    notebook): 2526 beyond T² 5 and beyond Q 7 of 346; 2425 validation harvest Q 5 of 30,
    suboptimal storage Q 14 of 31 and T² 1, optimal storage 0 of 5. Ruben inspected the
    outliers: in 2526 `2526_N29`, `2526_K27`, `2526_L29` (2 with something wrong in the volume,
    1 a very odd, "banana-like" shape; which is which not yet recorded); in the 2425 validation
    fruit `2425_I44` and `2425_C24` (Ruben, 21 Oct).
- **Encoding done (13–15 Oct, checked 17 Oct):** full run with Hugo's weights is
  `20261013T171306Z_hugo_vae` (two sessions: 13 Oct 19:13 to 14 Oct 03:19, stopped by the time
  limit; resumed 15 Oct, finished). Open part `derived/latents/hugo_vae/20261013T171306Z_hugo_vae/`:
  940 fruit (346 of 2526, 594 of 2425), latents 940 × 32³ all finite, 478 rows in
  `reconstruction.parquet` (346 + 66 + 66 decoded, as planned). Sealed part
  `derived/sealed/latents/hugo_vae/20261013T171306Z_hugo_vae/`: 153 fruit, encoded only.
  `run.json` says `dirty: true` because the untracked `HANDOFF_claude-code_2026-10-13.md` was in
  the folder; the code was commit 0c9052f.
  **Random encoder done (20 Oct, Mac, CPU):** run `20261020T120336Z_random_init`, seed 0, code
  54101c4 (clean), 14:03 to 14:31 (about 1.5 s per fruit on the Mac CPU). Open part 940 fruit
  (346 of 2526, 594 of 2425), sealed part 153 (87 + 66), all finite, nothing decoded (no trained
  decoder); same fruit in the same order as the Hugo run, both parts. The empty attempt of
  15 Oct was deleted; the 2-fruit test `20261005T142117Z_random_init` is kept.
- **Results of the draw (13 Oct):** season totals over 433 scanned fruit: non-consumable 179,
  browning ≥ 2 without cavity 73, browning 3 100, any cavity 115. Eligible sets: 120 at ±30 %
  (222 at ±40 %, 296 at ±50 %); every box occurs in 18 to 31 eligible sets. Drawn test boxes
  **F, J, M** (87 fruit; counts 35 / 18 / 19 / 22 against expected 36.0 / 14.7 / 20.1 / 23.1).
  Fold partitions of the 12 other boxes: 186 eligible of 15,400 at ±30 %. Repeats: 1 ABI CGN DEO
  HKL; 2 ACE BGL DIN HKO; 3 AEH BGL CKO DIN; 4 AHN BDL CEI GKO; 5 AGK BCI DNO EHL. Repeats 2 and 3
  share two folds (BGL, DIN); BL, DN and KO share a fold in 3 of 5 repeats.
- **B5 (13 Oct):** decoding takes about 56 s per fruit on the laptop CPU (encoding about 4 s;
  the decoder runs 3D convolutions with 128 channels at 128³), so all 1,093 would take about
  18 hours. Ruben (13 Oct): decode only what is needed now, over two nights, with batch saves.
  Decoded: the 346 open 2526 fruit, Hugo's 66 validation fruit, a sample of 66 of his 528
  training fruit (`hugo_training_sample`, seed 369); encoded only: the other 462 training fruit
  and the 153 sealed fruit (decoded at the final test). Work saved in chunks of 20 fruit
  (`src/pearlatent/encode_run.py`). Smoke test on 25 fruit with Hugo's weights, stopped and
  resumed: complete, all numbers finite, 464 s for 8 decoded + 12 encoded fruit. Test runs left
  in `derived/latents/hugo_vae/` (20261013T140149Z: 4 fruit, old format; 20261013T141619Z: 25
  fruit); the full run will be newer.
- **Next step:** record which of the 3 2526 outliers is the odd shape and what is wrong with the
  other two (open question 22), then write the outlier rule
  in `docs/` before removing anything: number of PCA components (about 13), which limit (99 %),
  T² or Q or both, what counts as "something wrong" (checked by eye in the CT view of notebook
  05), and whether an odd but real shape stays in. Then the reconstruction tests (Step 0,
  charter question 1) as notebook 06 on `reconstruction.parquet` of run
  `20261013T171306Z_hugo_vae`.
- **Blocked on:** nothing.
- **Mac (main machine from 20 Oct):** `/Users/rubenhoeven/code/pear-latent-thesis`, Python
  3.12.15, torch 2.14.1 (MPS available, not used: `device = "cpu"` in `config/local.toml`),
  marimo 0.25.1; data in `~/data/pear-thesis` (433 volumes 2526, 660 volumes 2425, both label
  files). Weights in `data/weights/hugo_vae/` as on the laptop. Hugo's
  repository cloned at `../Synthetic_CT_pear`, commit 183112f. Checked 20 Oct: `uv run pytest`
  62 passed (Hugo split tests included), notebook 00 runs, both latent arrays of run
  `20261013T171306Z_hugo_vae` load (940 and 153 fruit, finite, 478 reconstruction rows).
  Encoding ran on the Windows laptop (CPU, about 5 seconds per fruit, decided 5 Oct); the
  workstation is only for the ResNet-18 baseline and is not set up.
  School PC not set up: `uv sync`, copy `config/local.example.toml` to `config/local.toml`, run
  `notebooks/00_environment_check.py`. For pairing on a live notebook also run
  `npx skills add marimo-team/marimo-pair` (not in git) and put `jq` and `curl` on the PATH.

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
confirmed). Answered and removed on 13 Oct (meeting):
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
22. Of the 2526 outliers `2526_N29`, `2526_K27`, `2526_L29` (notebook 05, A = 13): which is
    the odd shape, and what is wrong with the other two volumes? Why `2425_I44` and `2425_C24`
    stand out.
23. Outlier rule (see next step): components, limit, T² and/or Q, and what to do with a real
    but odd shape.
24. Fit PCA and PLS on all 32,768 cells or only on cells inside the fruit (body masks in
    `masks.npy`, for example cells inside the body of at least half the fruit)? Would remove
    the background and the edge band.
25. The season difference on PC1 against PC2: check whether it follows storage or browning
    (colour by browning within the 2425 validation groups).

26. Rule for the number of latent variables: the plain minimum of the averaged RMSECV curve (as
    written to Hugo on 21 Oct) or a rule that prefers fewer (Haaland F-test, α = 0.25)? Decide
    before the curves are seen and write it in `docs/cv_explained.md` section 4.
27. Bootstrap on the test set: the 3 test boxes are too few for a bootstrap over boxes, so the
    test interval resamples fruit (`docs/cv_explained.md` section 8). Hugo's mail speaks of
    paired bootstrap intervals in general; state this difference at the meeting.

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
