# HANDOFF.md

The living state of the code work. Claude reads this at the start of every session (a hook prints
the "Current state" section) and updates it with `/handoff` at the end. Ruben may edit it freely.

## Current state

- **Date:** 2026-10-05
- **Phase:** data checks done; pooling and encoding code ready and tested with a randomly
  initialised encoder. No real latents yet: Hugo's weights are copied at the meeting on
  15 October. No model fitted on labels yet, on purpose (split rule not agreed).
- **Laptop:** `C:\Users\hoeve\code\pear-latent-thesis`, Python 3.12.12, torch 2.14.1+cpu,
  marimo 0.25.1; data in `C:/Users/hoeve/thesis-data` (433 volumes 2526, 660 volumes 2425,
  both label files). Hugo's repository cloned at `C:/Users/hoeve/code/Synthetic_CT_pear`, commit
  183112f (`hugo_repo` in `config/local.toml`). Encoding runs on this laptop (CPU, about
  5 seconds per fruit, decided 5 Oct); the workstation is only for fine-tuning and is not set up.
  School PC not set up: `uv sync`, copy `config/local.example.toml` to `config/local.toml`, run
  `notebooks/00_environment_check.py`. For pairing on a live notebook also run
  `npx skills add marimo-team/marimo-pair` (not in git) and put `jq` and `curl` on the PATH.
- **Done this session (5 Oct):**
  - Plan files updated after Hugo's answers (box letter = orchard ID; 17 unscanned fruit
    excluded; rotten fruit carry the highest grades). Charter rule 1: within-box comparison added
    and removed the same day. Rotational symmetry test dropped (Ruben).
  - Code with tests (38 in total, all pass): `volume_qc.py`, `exclusions.py`, `pooling.py`
    (fruit body mask without stalk and with cavities filled, block mean to 32³, mean, 99th
    percentile, cylindrical profile 8 heights × 4 radii), `encoding.py` (Hugo's encoder from the
    separate clone; `mu` plus the 512-channel layer pooled inside the body mask), `latest_run_dir`.
  - Notebooks: 01 section 5 (rot), `02_volume_qc.py`, `03_raw_ct_pooled.py`,
    `04_encode_latents.py` (all status draft, all run as scripts). Products in `derived/`:
    `volume_qc/`, `exclusions.csv`, `raw_ct_32/<run_id>/`, `stalk_check.png`, and a 2-fruit test
    run in `latents/random_init/`.
  - `docs/meeting_2026-10-15.md`: data summary, proposed test-box rule and fold design, questions
    for Hugo, uses of the 2024 data, the title.
- **Results (5 Oct, read from script runs and images on the laptop):**
  - 2425: defect = storage group (all 300 suboptimal-storage fruit have browning; none of the
    360 others have browning or cavity). Rot = 1 in 16 fruit, 11 of them browning 3 and cavity 3;
    of the 175 fruit with browning 3 and cavity 3, 11 are rotten.
  - 2526: 24 fruit with browning 3 and cavity 3 (flagged in `exclusions.csv`, 23 scanned). Grades
    depend strongly on the box (K: 27 of 30 without browning; O: 15 with browning 3).
  - All 1093 volumes 128³ uint16, none flagged, no duplicates; header voxel size 1.0 is
    meaningless; long axis = third array axis, calyx end at the low end, in every volume.
  - Most 2526 volumes include the stalk (8 of 12 in a random sample of projections, seed 1;
    2425: at most a short stub) and it counts in the 128-voxel length: fruit body spans median 28
    of 32 latent slices in 2526 (21–32, box K median 25), 31 in 2425. Fruit size in voxels
    differs strongly per 2526 box; candidate shortcut for the nuisance check.
  - Earlier (3 Oct, notebook 01): label rules for `defective`, `binary_123`, `binary_23`,
    `non-consumable` hold for every fruit; 17 labelled 2526 fruit have no volume.
- **Next step:** Ruben reviews `docs/meeting_2026-10-15.md` in a separate Claude session and meets
  Hugo on 15 October. Then, on the laptop: put Hugo's run folder at `encoder_dir` in
  `config/local.toml`; `uv run python notebooks/04_encode_latents.py --encoder hugo_vae --limit 4`,
  check shapes and that `mu` is finite; run without `--limit` (about 1.5 hours); run
  `--encoder random_init` for the lower bound. After the split rule is agreed with Hugo: write it
  into CHARTER.md, then implement the test-box draw and the folds in `src/pearlatent/splits.py`.
- **Blocked on:** Hugo's weights (15 October) for encoding; the split rule (meeting) for any
  model fitted on labels.

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
(poor scan setup); a 2526 box letter is an orchard ID. The full list for the meeting is in
`docs/meeting_2026-10-15.md` section 3; not all questions below are in that document.

1. Weights: Hugo has the encoder ready and copies it on 15 October. Still ask for the decoder,
   `vae_hyperparameter.json`, which checkpoint (`checkpoint.pth` or `best.pth`) and which seed.
2. Whether the data may be uploaded to cloud services. Ruben (3 Oct): Claude may read notebook
   outputs. molab and other uploads: no until Hugo confirms in writing.
3. Which 2526 fruit were rotten (the 2526 file has no rot column).
4. Are 2024 and 2526 orchards with the same letter different orchards?
5. Ruben's position on the grade 2 non-consumable threshold (Hugo asked).
6. Preprocessing: why most 2526 volumes include the stalk and the 2425 volumes do not, and
   whether the 2526 volumes can be preprocessed the same way.
7. The list of 2024 fruit in Hugo's VAE test split (seed 42): the 2024 fruit the encoder has not
   seen.
8. The test-box rule and fold design (`docs/meeting_2026-10-15.md` section 2).
9. Lab workstation (only for fine-tuning now): GPU model, NVIDIA driver version, operating
   system, `uv`, tmux.
10. Item 2 of Hugo's slide: a head on his encoder, or a new model? (Decides whether the own
    self-supervised extension is allowed.)
11. Cavity-label issues: grades that disagree with visible voids (to back with the cavity
    cross-check, if built).
12. The month0 to month4 repeat scans: which fruit, and are they available?
13. Overlap with Hugo's own classification and synthetic-data work.

For Ruben (not Hugo), added 10 Oct:

14. Reference coordinates of the healthy fruit for the deviation map. The earlier reference used
    height and radius; the same cell (x, y, z) is affected by the free rotation around the
    stem–calyx axis.
15. Whether Solo accepts a four-way array for N-way partial least squares. The documentation found
    does not say. Test with a small synthetic array (fruit × 8 × 8 × 8) before relying on it.

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

## Unverified assumptions in code

| Where | Assumption | Verify against |
|---|---|---|
| `src/pearlatent/labels.py` `fruit_keys` | G_opt and I_opt fruit come from orchards G and I, so they share box `2425_G` / `2425_I` (decided with Ruben 3 Oct). Supported by the preprint, section 2.2, as summarised in Ruben's other Claude session on 5 Oct (the two orchards with optimal storage have fruit in three boxes); not yet read in the preprint itself | preprint section 2.2, Hugo |
| `src/pearlatent/labels.py` `_STORAGE` | 2526 storage condition unknown; recorded as `after_storage` | Hugo |
| `src/pearlatent/encoding.py` `load_encoder` | default checkpoint `checkpoint.pth` is the encoder of the preprint (not `best.pth`) | Hugo (open question 1) |
| `pyproject.toml` | workstation driver supports CUDA 12.8 wheels (only matters for fine-tuning) | `nvidia-smi` on the workstation |

Note on names: in 2425 the `box` column holds the orchard letter (60 to 90 fruit). Per the
preprint a physical box is 30 fruit of one orchard in one storage group. Grouping by orchard is
stricter than grouping by box, so the splits stay valid; only the name is loose.

## Session log

Newest first. One entry per session, three to six lines.

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
