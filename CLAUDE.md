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

**No encoder is trained.** The only things fitted are small models on frozen features: partial least
squares discriminant analysis (PLS-DA), penalised logistic regression, ordinal models, optionally a
small attention-pooling layer or prototype head, and one fine-tuning arm of Hugo's encoder. If a task
seems to need training a network from scratch, the plan has drifted: stop and check `CHARTER.md`.

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
5. **Browning is the primary target; `defective` is never the primary target.** `defective` is the
   maximum of browning and cavity and merges an easy task (cavities are gas voids, high contrast)
   with a hard one (browning is a small density shift). Cavity is reported separately as the control.
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
| Voxel size | 0.6267 mm isotropic | preprint |
| Volume | 128 × 128 × 128, NIfTI, min-max scaled to [0, 1] per volume at load | `utils/volumes.py` |
| Latent | 1 × 32 × 32 × 32, one scalar per cell, use the mean `mu` | `component/vae.py` |
| Layer before the latent heads | 512 channels × 32³ | `component/vae.py` (base_channel 256) |
| Latent cell spacing | 4 voxels, about 2.5 mm | layer strides |
| Receptive field of one cell | 31 voxels, about 19.4 mm | computed from layer list |
| KL weight (beta) | 1e-6: a compression latent, not a semantic one | `train_vae.py` default |
| Reconstruction | MAE 0.0022, SSIM 0.9915, PSNR 36.92 | preprint |
| Encoder training set | about 528 of the 660 fruit of 2024 (random 80/10/10 split, seed 42) | `train_vae.py`, `utils/splits.py` |

Consequences: a 1 cm³ lesion covers roughly 60 of 32,768 cells, so mean pooling moves by about
0.2 %; any localisation map is blurred to about the receptive field, so per-cell detail finer than
that is an artefact of the spacing. InstanceNorm (no trainable parameters) normalises every channel
over the whole fruit, so each cell also depends on global intensity statistics.

The 512-channel layer is 64 MB per fruit as float32 (about 70 GB for all fruit). Never save it in
full; pool it during encoding.

## 4. Data

- Volumes: `.nii`, loaded with nibabel, path from `cfg.path("ct_dir")`.
- `labels_conference_pear_2425.csv`: 660 fruit, 2024 harvest. Columns `filename, rot, browning,
  cavity, defective, non-consumable`. Groups: A31–J60 scanned at harvest, A01–J30 after suboptimal
  storage, `G_opt` / `I_opt` after optimal storage (these do not match the box pattern; handle them
  explicitly).
- `labels_conference_pear_2526.csv`: 450 fruit, 15 boxes of 30 (A to O), all scanned after storage.
  Columns `filename, browning, cavity, defective, binary_123, binary_23`.
- Only `browning` and `cavity` are common to both files. Harmonise on those.

## 5. Pipeline shape

1. **Encode once** (GPU, lab workstation). Forward pass through the frozen encoder; write
   `derived/latents/<encoder>/<run_id>/latents.npy`, `manifest.parquet` (one row per fruit:
   `fruit_id`, season, box, file, row index) and `run.json`. About 145 MB for all fruit.
2. **Pool** (any machine). Array arithmetic on the saved latents: mean, 99th percentile, cylindrical
   binning (height × radius, averaged over angle). Produces a feature table.
3. **Model** (any machine). Statistics on that table. From here it is an ordinary dataset.

Step 1 is cached to disk and never recomputed by a notebook. marimo reruns cells reactively; without
the cache, touching a downstream cell would rerun a forward pass over the whole dataset.

## 6. Coordinates

Preprocessing aligns the stem–calyx axis and rescales each fruit to 128 voxels in length, so height
is comparable across fruit and rotation about that axis is not. For statistics across fruit, convert
to cylindrical coordinates (height, radius) and average over angle, but test rotational symmetry
first. Keep per-fruit analyses at full resolution, where angle still carries information.

## 7. Statistics

- p ≫ n throughout (32,768 cells, a few hundred fruit). Standardise inside the training fold.
- Prefer PLS over principal component regression: components chosen by variance describe pear size
  and shape, not browning.
- Prefer ridge or elastic net over lasso when the coefficient map is to be read: lasso picks
  arbitrarily among correlated neighbouring cells, and the map will not survive bootstrapping.
- Severity is ordinal: cumulative-link models (`statsmodels` `OrderedModel`); report quadratic
  weighted kappa or Spearman correlation, not plain accuracy.
- Testing many cells needs false-discovery-rate control; prefer cluster-based permutation tests,
  because neighbouring cells are correlated.
- Confidence intervals by resampling fruit (respecting boxes), not the spread across folds.
- Recall and precision always beside accuracy.
- Nuisance check on every representation: does it predict box or storage group better than the
  defect?

## 8. Stack

| Purpose | Tool | Not |
|---|---|---|
| Environment | `uv` (`uv sync`, `uv run`, `uv add`) | pip, conda, `uv pip install` into the project |
| Notebooks | marimo `.py` files in `notebooks/` | Jupyter `.ipynb` |
| Tables | Polars (lazy `scan_*` for files) | pandas, except at a library boundary (`.to_pandas()` on the last line before the call) |
| Statistics | scikit-learn (PCA, PLS-DA, penalised regression), statsmodels (`OrderedModel`) | ad-hoc reimplementations |
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

- **Laptop (Windows)** and **school PC (Windows, no admin)**: editing, statistics on saved latents,
  plots. CPU PyTorch.
- **Lab workstation (Linux, GPU, via SSH)**: encoding, fine-tuning. Load the `workstation-run` skill.
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
5. Statistical framing first (PLS-DA, penalised regression, ordinal models, PCA), not
   machine-learning-first framing.
6. Commit small, messages in the imperative. At the end of a session, run `/handoff`.

## 12. Commands

```bash
uv sync                                         # install exactly what uv.lock says
uv run marimo edit notebooks/                   # open the notebook browser
uv run marimo edit --watch notebooks/NN_x.py    # editor reloads when Claude edits the file
uv run marimo check notebooks/                  # lint all notebooks
uv run pytest                                   # tests for src/pearlatent
uv run ruff check src tests && uv run ruff format src tests
```
