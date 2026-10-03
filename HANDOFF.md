# HANDOFF.md

The living state of the code work. Claude reads this at the start of every session (a hook prints
the "Current state" section) and updates it with `/handoff` at the end. Ruben may edit it freely.

## Current state

- **Date:** 2026-10-02
- **Phase:** setup — repository skeleton created, not yet run against real data.
- **Last done:** created repository structure, rulebook (CLAUDE.md), hooks, skills, environment
  check notebook, box-level split and label utilities with tests.
- **Verified (2026-10-02, Linux test copy, marimo 0.25.1, torch 2.14.1):** `marimo check`
  clean, 7 tests pass, ruff clean, both notebooks run as scripts, all three hooks behave.
  `uv.lock` is not yet committed: download.pytorch.org was not reachable from the build
  machine. The first `uv sync` on the laptop creates it; commit it straight away.
- **3 Oct:** pushed to github.com/RuHoeP4k1/pear-latent-thesis (private). CLAUDE.md merged with the
  earlier draft from the chat project; CHARTER.md added with review flags.
- **Next step:** on each machine: `uv sync`, copy `config/local.example.toml` to
  `config/local.toml`, run `notebooks/00_environment_check.py`. Then write
  `01_label_inventory.py`: load both label CSV files, confirm column names, run
  `validate_ranges` and `check_derived_labels`, count fruit per box and per harvest year.
- **Blocked on:** see "Open questions".

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

## Points to raise with Hugo (found in his code)

1. Hugo's VAE train/validation/test split is random by fruit (`utils/splits.py`). This does not
   affect us if the 2025 and 2026 fruit are truly unseen, but it should be stated in the thesis.
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
- The 2425 volumes are not downloaded yet.

## Plan items to settle with Ruben (before the mail to Hugo)

1. Done 3 Oct: evaluation design reversed in CHARTER.md (rule 2). Still to do: same change in the
   Approved ideas tab (ground rule 2) during the plan review. Open: whether to set aside two or
   three 2526 boxes as a final test before any exploration with labels.
2. "2025 and 2026 fruit" or one harvest (2025) stored into 2026? Wording in charter and plan.
3. Block 0 checkpoint of 8 October ("Hugo's code runs, encode and decode one 2526 pear") depends
   on the weights; the gate test of 15 October depends on browning region annotations, which may
   not exist.

## Open questions (must be answered from Tier 1 or by Hugo)

1. The trained encoder weights: the run folder (`checkpoint.pth` + `vae_hyperparameter.json`) of
   the VAE trained on the 2024 harvest (about 528 of the 660 fruit). Which run, and was it `checkpoint.pth`
   (last epoch, what `load_vae` loads) or `best.pth`?
2. Why 17 of the 450 fruit in 2526 have no volume (list above). Are box letters reused across
   seasons (box A in 2526 a different group from box A in 2425)?
3. What a box letter means in the 2526 data (orchard, storage condition, scan session?). In 2024,
   A to J match the ten orchards of the preprint.
4. Whether the data may be uploaded to cloud services (molab, Claude). Until answered: no.
5. Lab workstation: GPU model, NVIDIA driver version (`nvidia-smi`), operating system, `uv`, tmux.

## Unverified assumptions in code

| Where | Assumption | Verify against |
|---|---|---|
| `src/pearlatent/labels.py` | binary_123 = defective ≥ 1, binary_23 = defective ≥ 2 | label CSV via `check_derived_labels` |
| `pyproject.toml` | workstation driver supports CUDA 12.8 wheels | `nvidia-smi` on the workstation |
| `uv.lock` | missing; first `uv sync` on the laptop resolves it | commit `uv.lock` after the first sync |

## Session log

Newest first. One entry per session, three to six lines.

### 2026-10-03 — Hugo's repository read
- Verified volume format, file-name pattern, encoder construction and receptive field from code.
- Found that Hugo's classifier splits within boxes; recorded as a point for Hugo.
- Remaining blocker: encoder weights.

### 2026-10-02 — setup
- Created the repository skeleton and agent rulebook.
- Decided: uv project (not per-notebook sandboxes), Polars, PyTorch, marimo with project config,
  data outside git, molab not used for data.
