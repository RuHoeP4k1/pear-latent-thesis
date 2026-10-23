# CLAUDE.md — rulebook for this repository

Master's thesis of Ruben Hoeven (KU Leuven, Bioscience Engineering), supervisor Hugo (Junyan) Li,
MeBioS. The research question, the fixed decisions and the evaluation rules are in `CHARTER.md`.
This file covers how the code works and what not to get wrong.

Read `HANDOFF.md` at the start of every session. It holds the current state, the next step and the
open questions.

## 1. What this project does

Encodes pear X-ray computed tomography (CT) volumes with Hugo's frozen, pretrained 3D variational
autoencoder (VAE), then analyses the latent space statistically: is internal browning detectable from
it, and at which severity does detection stop working?

**No encoder is trained and no self-supervised model is trained** (dropped 13 October 2026: the
GPU is too small). One network is trained from scratch: the ResNet-18 baseline, architecture
imported from Hugo's pinned clone, on the lab workstation with our box folds. Everything else is a
small model on frozen features: partial least squares discriminant analysis (PLS-DA), ordinal
models on PLS scores (penalised logistic regression dropped 21 October 2026), N-way partial least squares (one extra row after the core
study, see `CHARTER.md`, Methods). If a task seems to need training a network from scratch other
than the ResNet-18 baseline, the plan has drifted: stop and check `CHARTER.md`.

Design: primary fitting is grouped cross-validation within the 433 scanned 2526 fruit, whole boxes
held out. The 660 2425 fruit are secondary (comparison with Hugo, reverse check, healthy
reference). 433 + 660 = 1,093 volumes.

## 2. Non-negotiable rules

1. **Data never enters git.** No CT volumes, label CSV files, weights, latents or result tables in a
   commit. Paths come from `config/local.toml` via `pearlatent.config.load_config()`. Never write an
   absolute path to data into code.
2. **Raw data is read-only.** Nothing writes into `data/`. Products go to `derived/` (rebuildable
   intermediates) or `results/<run_id>/`, each with a `run.json` from `pearlatent.runs`.
3. **Split by box, never by fruit.** Every model fitted on labels uses `pearlatent.splits`
   (`group_kfold` or an explicit box hold-out) and calls `assert_no_group_leakage` before fitting.
   This includes choosing settings such as the number of PLS components. The frozen encoder is not
   affected.
4. **Every fruit identifier carries its season.** File names A01 to J30 occur in both label files and
   are different fruit. Build `fruit_id = "<season>_<filename stem>"` (for example `2526_A01`) before
   joining or concatenating anything, and the box as `"<season>_<letter>"`.
5. **Target order (charter rule 5, agreed with Hugo, 13 October 2026).** First `defective` (the
   maximum of browning and cavity): consumable versus non-consumable (`binary_23`, defective ≥ 2),
   with recall per defective grade; then browning and cavity separately, and whether the two can be
   separated. Results on `defective` are always split by cause (browning only, cavity only, both):
   cavity (gas voids, high contrast) is the easier signal and can carry the score on its own, while
   browning is a small density shift.
6. **Claims about our data trace to Tier 1** (Hugo's preprint SSRN 6433761 and its supplement,
   github.com/Hugo-Li-Junyan/Synthetic_CT_pear, the two label CSV files, the grading rubric).
   If code depends on an unverified fact, mark it `# UNVERIFIED:` and list it in `HANDOFF.md`.
   Numbers in the thesis come from a run, not from memory.
7. **No cloud for data.** molab notebooks are public through their link by default. Until Hugo
   confirms in writing that the data may leave university systems, molab is used only for toy or
   public data.

## 3. The numbers that constrain everything

Verified against Hugo's code (commit 183112f) or the preprint. Do not re-derive them in notebooks;
cite this table or the source.

| Quantity | Value | Source |
|---|---|---|
| Voxel size | 0.6267 mm is the scanner voxel size before rescaling, not the size of a 128³ voxel; after rescaling to 128 voxels in length the size differs per fruit | preprint |
| Volume | 128 × 128 × 128, NIfTI, min-max scaled to [0, 1] per volume at load | `utils/volumes.py` |
| Latent | 1 × 32 × 32 × 32, one scalar per cell, use the mean `mu` | `component/vae.py` |
| Layer before the latent heads | 512 channels × 32³ | `component/vae.py` (base_channel 256) |
| Latent cell spacing | 4 voxels, about 2.5 mm (approximate, differs per fruit) | layer strides |
| Receptive field of one cell | 31 voxels, about 19 mm (approximate, differs per fruit) | computed from layer list |
| KL weight (beta) | 1e-6: a compression latent, not a semantic one | `train_vae.py` default |
| Reconstruction | Section 3.1.1: MAE 0.0022, SSIM 0.9915, PSNR 36.9226 "on the test set". Section 2.5.1 says the metrics were computed on the validation set; Table D.1 ("Evaluation metrics on validation dataset"), best model: MAE 0.0025, SSIM 0.9915, PSNR 36.9226. Which set: ask Hugo | preprint |
| Encoder training set | about 528 of the 660 fruit of 2024; random 80/10/10 split by fruit, all orchards pooled (preprint sections 2.2 and 2.5, which give no seed and no fruit list); in code `utils/splits.split_train_val_test` (validation 0.1, test 0.1) over the rows of the label CSV in file order; the seed is stored in `checkpoint.pth` under `random_state`; seed 369 (Hugo, 13 October 2026; also in `slurm/train_vae.slurm`). Seed 42 is only the default in `train_vae.py`. Fruit list reproduced (handoff B2) and confirmed by Hugo on 21 October 2026 (same indices with PyTorch 2.7.1, 2.7.1+cuda and 2.14.1, same label file) | `train_vae.py`, `utils/splits.py` |

Consequences: a 1 cm³ lesion covers roughly 60 of 32,768 cells, so mean pooling moves by about
0.2 %; any localisation map is blurred to about the receptive field, so per-cell detail finer than
that is an artefact of the spacing. InstanceNorm (no trainable parameters) normalises every channel
over the whole fruit, so each cell also depends on global intensity statistics.

The 512-channel layer is 64 MB per fruit as float32 (about 70 GB for all fruit). Never save it in
full; pool it during encoding.

## 4. Data

- Volumes: `.nii`, loaded with nibabel, one folder per season, `cfg.path("ct_dir_2526")` and `cfg.path("ct_dir_2425")`.
- `labels_conference_pear_2425.csv`: 660 fruit, 2024 harvest. Columns `filename, rot, browning,
  cavity, defective, non-consumable`. Groups: A31–J60 scanned at harvest, A01–J30 after suboptimal
  storage, `G_opt` / `I_opt` after optimal storage (these do not match the box pattern; handle them
  explicitly).
- `labels_conference_pear_2526.csv`: 450 fruit, 15 boxes of 30 (A to O), all scanned after storage.
  Columns `filename, browning, cavity, defective, binary_123, binary_23`.
  Box letter = orchard ID (Hugo, October 2026). The 17 unscanned fruit are excluded. No rot
  column. Hugo (13 October): rot is not really a problem in 2526 (storage conditions less
  extreme); the 24 fruit with browning 3 and cavity 3 stay in, with one sensitivity row without
  them.
- Stalks: Hugo (13 October): the stalks are not a problem; no stalk removal. Their effect on the
  reconstruction is checked in Step 0. `body_mask` is still used to measure height over the fruit
  body.
- Only `browning` and `cavity` are common to both files. Harmonise on those.
- Box letters are unrelated across seasons (Ruben, 3 October 2026): box A in 2425 and box A in
  2526 have nothing in common. Whatever a letter stands for (orchard, batch, storage regime), it
  is kept in the `box` column (`"<season>_<letter>"`) and the 2425 storage group in `storage`,
  so it can be related to other information later.
  To confirm with Hugo now that both seasons use orchard IDs.
- Verified on the label files (notebook 01, 3 October 2026): `defective = max(browning, cavity)`,
  `binary_123 = defective ≥ 1`, `binary_23 = defective ≥ 2` (2526) and `non-consumable =
  defective ≥ 2` (2425) hold for every fruit; `rot` does not enter `non-consumable`.

## 5. Pipeline shape

1. **Encode once** (laptop, CPU, about 1.5 hours for all fruit). Forward pass through the frozen encoder; write
   `derived/latents/<encoder>/<run_id>/latents.npy`, `manifest.parquet` (one row per fruit:
   `fruit_id`, season, box, file, row index) and `run.json`. About 145 MB for all fruit.
   Decode `mu` with `vae.decode(mu)` for Step 0. Do not use `vae(x)`: its forward pass samples z.
   The random encoder stays a planned run (Ruben, 13 October 2026).
2. **Pool** (any machine). Array arithmetic on the saved latents. Options: the full 32³ grid, and
   the deviation-from-healthy map with persistence summaries (high percentile as comparison). No
   cylindrical bins (dropped 10 October 2026), no attention pooling, no 512-channel layer.
3. **Model** (any machine). Statistics on that table. From here it is an ordinary dataset.

Step 1 is cached to disk and never recomputed by a notebook. marimo reruns cells reactively; without
the cache, touching a downstream cell would rerun a forward pass over the whole dataset.

## 6. Coordinates

Preprocessing aligns the stem–calyx axis and rescales each fruit to 128 voxels in length, so height
is comparable across fruit and rotation about that axis is not: the same cell (x, y, z) can lie at
a different angle in different fruit. Cylindrical bins (height × radius, averaged over angle) were
dropped on 10 October 2026; models use the full grid. For N-way partial least squares the charter
adds two checks for the free rotation (core crop, random rotation of the test fruit). The reference
coordinates of the healthy fruit for the deviation map are still to settle (`CHARTER.md`, Methods).

Verified on all volumes (notebooks 02 and 03, 5 October 2026): the stem–calyx axis is the third
array axis, with the wide calyx end at the low end, in every volume. Most 2526 volumes include the
stalk (8 of 12 in a random sample) and the stalk counts in the 128-voxel length, so there the fruit body spans 21 to 32 of the
32 latent slices (median 28; 2425: median 31). Measure height over the body:
`pearlatent.pooling.body_mask` removes the stalk and fills internal cavities. Hugo (13 October):
the stalks are not a problem; no stalk removal. Their effect on the reconstruction is checked in
Step 0. `body_mask` is still used to measure height over the fruit body.

## 7. Statistics

- p ≫ n throughout (32,768 cells, a few hundred fruit). Centre inside the training fold; do not
  scale cells to unit variance (Ruben, 23 October 2026: background cells hardly vary, scaling
  would raise their noise to the weight of fruit cells). `PLSRegression(scale=False)`.
- Prefer PLS over principal component regression: components chosen by variance describe pear size
  and shape, not browning.
- Number of PLS components (design A, Ruben, 21 October 2026): 20 repeats of the 4-fold box
  cross-validation on the 12 non-test boxes, PRESS averaged over the repeats, one choice from that
  curve, one fit on all 12 boxes. No inner split. The choice: van der Voet randomisation test
  with signs flipped per box, smallest h with p > 0.10 (Ruben, 23 October 2026). See
  `docs/cv_explained.md`, section 4.
- Faulty volumes (missing part of the fruit, failed preprocessing) are excluded by
  `docs/outlier_rule.md`; never because a fruit is defective.
- Severity is ordinal: cumulative-link models (`statsmodels` `OrderedModel`); report quadratic
  weighted kappa or Spearman correlation, not plain accuracy.
- Testing many cells needs false-discovery-rate control; prefer cluster-based permutation tests,
  because neighbouring cells are correlated.
- Confidence intervals by a bootstrap over whole boxes; method comparisons paired on the same out-of-fold predictions, not the spread across folds.
- Recall and precision always beside accuracy.
- Nuisance check on every representation: does it predict box or storage group better than the
  defect?

## 8. Stack

| Purpose | Tool | Not |
|---|---|---|
| Environment | `uv` (`uv sync`, `uv run`, `uv add`) | pip, conda, `uv pip install` into the project |
| Notebooks | marimo `.py` files in `notebooks/` | Jupyter `.ipynb` |
| Tables | Polars (lazy `scan_*` for files) | pandas, except at a library boundary (`.to_pandas()` on the last line before the call) |
| Statistics | scikit-learn (PCA, PLS-DA), statsmodels (`OrderedModel`) | ad-hoc reimplementations |
| Deep learning | PyTorch | TensorFlow, Lightning |
| Volumes | nibabel | |
| Plots | Altair inside notebooks; Matplotlib for thesis figures in `reports/figures/` | figures pasted by hand |

Add packages only with `uv add <pkg>` and commit `pyproject.toml` and `uv.lock` together.

Hugo's repository has no licence file. Do not copy its code into this repository; reference it at a
pinned commit.

## 9. marimo conventions

Load the `thesis-notebook` skill before writing or editing any notebook. In short:

1. One notebook answers one question. Name `NN_short_name.py`. Start from `notebooks/_template.py`;
   keep its header (question, inputs, outputs, status).
2. Imports only in the `with app.setup:` block.
3. A variable is defined in exactly one cell and never mutated in another cell. Temporaries start
   with `_`.
4. Logic reused by two notebooks, or worth testing, moves to `src/pearlatent/` with a test in
   `tests/`. Notebooks hold narrative and figures.
5. UI element created in one cell, its `.value` read in another.
6. Anything slower than a few seconds sits behind `mo.ui.run_button` + `mo.stop`, and its result is
   cached: `mo.persistent_cache` for small results, files in `derived/` for large arrays.
7. GPU tensors live inside functions; use `torch.inference_mode()` for encoding.
8. Every notebook also runs as a script: `uv run python notebooks/NN_name.py`.
9. After every edit, `marimo check` runs automatically (hook). Fix every error before moving on.
10. Random seeds fixed and recorded in `run.json`.

## 10. Machines

- **Laptop (Windows)**: editing, encoding (CPU, about 5 seconds per fruit; decided by Ruben,
  5 October 2026), statistics on saved latents, plots. CPU PyTorch.
- **School PC (Windows, no admin)**: editing, statistics on saved latents, plots.
- **Lab workstation (Linux, GPU, via SSH)**: ResNet-18 baseline. Load the `workstation-run` skill.
- Code syncs only through git (`git pull` at start, `git push` at end). Data is synced separately.

Details: `docs/workflow.md`.

## 11. How to work with Ruben

1. Decisions in `CHARTER.md` were made with evidence. Do not revisit them casually; if new evidence
   contradicts one, say so explicitly and state what changed. Ask before any change to the analysis
   plan.
2. Prefer few well-built paths over many options. Say plainly when an idea is weak.
3. Flag contradictions between documents instead of working around them.
4. Plain literal language in comments, markdown cells and commit messages: no metaphors,
   abbreviations written out on first use.
5. Statistical framing first (PLS-DA, ordinal models, PCA), not
   machine-learning-first framing.
6. Commit small, messages in the imperative. At the end of a session, run `/handoff`.
7. **No black box** (Ruben, 13 October 2026). Ruben must be able to follow and backtrack every
   step. Before writing code for a step, say in a few lines what it will do and why; after running
   it, say what it did and what came out. Write plain, readable code: short functions with a
   docstring that says what goes in, what comes out and why; a comment on every choice (a
   threshold, a seed, a region, a tolerance), not on syntax; names that say what a thing is; no
   long chained one-liners. No hidden steps: every filter, exclusion, rescaling or join prints how
   many fruit went in and came out. When a library function's behaviour matters for the result
   (for example `torch.utils.data.random_split`), explain it in a comment. Ask Ruben before any run
   longer than a few minutes and before any change to the plan.

## 12. Commands

```bash
uv sync                                         # install exactly what uv.lock says
uv run marimo edit notebooks/                   # open the notebook browser
uv run marimo edit --watch notebooks/NN_x.py    # editor reloads when Claude edits the file
uv run marimo check notebooks/                  # lint all notebooks
uv run pytest                                   # tests for src/pearlatent
uv run ruff check src tests && uv run ruff format src tests
```
