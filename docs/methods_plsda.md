# Methods record: PLS-DA on binary_23 and its comparison rows (notebook 08)

Written 3 November 2026 for the rules fixed on 2 November, so the methods are not lost before
the thesis is written. The code is the reference: `src/pearlatent/plsda.py`,
`src/pearlatent/tissue_features.py`, `src/pearlatent/med3d_features.py`,
`src/pearlatent/nuisance.py`, notebook `08_plsda_binary23.py`. Results are not recorded here;
they are in the runs of section 7.

## 1. Fruit and design

- 315 open 2526 fruit: the 346 of the 12 open boxes minus the faulty volumes
  (`derived/volume_decisions.csv`, `docs/outlier_rule.md` sections 9 and 10). 129 are
  non-consumable (`binary_23` = defective ≥ 2). Test boxes F, J, M (87 fruit) are not loaded.
- Design A (`docs/cv_explained.md` sections 3 to 7): 20 repeats of a 4-fold box
  cross-validation on the 12 boxes (`derived/outer_folds_20/outer_folds.csv`), 1 to 15 latent
  variables (LVs; at most 10 for the 10 tissue features). The score of a fruit is the average of
  its 20 out-of-fold predictions.

## 2. Scaling (fixed 2 November 2026)

- Cell grids (Hugo's latent, CT at 32³): centred only, `PLSRegression(scale=False)`
  (CLAUDE.md section 7: background cells hardly vary; scaling would give their noise the
  weight of fruit cells).
- Tissue features and Med3D channels: centred and scaled to unit variance, `scale=True`. They
  are different variables in different units, not neighbouring cells of one image. Example
  (940 open fruit, run `20261102T151947Z`): `air_fraction` has SD 0.034 (median 0.0014,
  maximum 0.155), `fruit_sd` and `fruit_mean` (median 0.83) have their own SDs; unscaled,
  each feature would weigh by its SD, i.e. by its unit, not by how much it tells about the
  defect.
- `PLSRegression` computes mean and SD on the training fruit of each fold only.
- 3 November 2026: until commit 57277c3 the save cell wrote `"scale": false` into `run.json`
  for every row. The computation was right; only the record was wrong. No run with a wrong
  record was kept.

## 3. Number of LVs and the two cut-offs (2 November 2026)

- LVs: by eye at the elbow of the mean RMSECV curve (PRESS averaged over the 20 repeats),
  chosen by Ruben before any test box is opened. Replaces the van der Voet rule of 23 October.
  Chosen: Hugo's latent 4, CT at 32³ 5, tissue features 7, Med3D 7 (Med3D fixed at 7 on
  3 November; its curve still falls slightly to 8 and 10 LVs).
- Cut-off 0.5: fixed in advance.
- Youden cut-off (`pearlatent.plsda.youden_cutoff`): the cut-off with the largest
  recall − false-positive rate on the averaged out-of-fold predictions. Small example: 4
  non-consumable fruit with outputs 0.9, 0.45, 0.40, 0.2 and 4 consumable fruit with 0.6, 0.3,
  0.1, 0.0. At 0.5: recall 1/4, false-positive rate 1/4, J = 0. At 0.40: recall 3/4,
  false-positive rate 1/4, J = 0.5, the largest, so the Youden cut-off is 0.40. It is saved in
  `run.json` and applied unchanged to the test boxes. Being chosen on the same predictions, it
  is slightly optimistic in the cross-validation.

## 4. Tissue features (comparison row, 2 November 2026)

The 10 features of Van De Looverbosch et al. (2020, Food Control 113, 107170, section 2.3.1),
on the min-max scaled 128³ volume:

1. tissue = voxels above a global Otsu threshold, largest connected part (our addition);
2. fruit = tissue with enclosed holes filled; internal air = fruit − tissue;
3. low-density tissue = tissue voxels below the mean of the tissue voxels in a 31 × 31 × 31
   window around them (air in the window not counted); high density = the rest;
4. features: air volume / fruit volume; mean and SD of the intensity in fruit, tissue, low and
   high density (8); Kolmogorov-Smirnov statistic between low and high density.

Example for step 3 (made-up numbers): a voxel of 0.80 whose window holds 20,000 tissue voxels
with a summed intensity of 16,600 has local mean 0.83, so it is low density; the same voxel in
a window with local mean 0.78 is high density.

Differences from the paper: largest part kept; stalk kept (Hugo, 13 October); window 31
voxels is about 20 mm here (fruit length / 128 ≈ 0.6 to 0.7 mm) against about 16 mm in the
paper; intensities are the scaled volume, not raw values. The local-mean rule is our reading of
the paper (`# UNVERIFIED` in the code). Open: the partial-volume rim under the skin is always
low density (question 45).

## 5. Med3D ResNet-18 (comparison row, 2 November 2026)

Frozen MedicalNet ResNet-18 (`resnet_18_23dataset.pth`, SHA-256 61224f93…, code 20f76aa),
pretrained on 23 medical CT and MRI datasets.

1. Fruit mask: voxels above 0.1, largest connected part, enclosed holes filled.
2. z-score with the mean and SD of the voxels above 0.1 inside the fruit (MedicalNet's
   training normalisation; "above 0" there, but our background is not exactly 0).
3. Everything outside the fruit set to 0 (nnU-Net rule, Isensee et al. 2021). Enclosed
   cavities keep their own (low) z-score.
4. Backbone up to `layer4` (512 channels × 16³), averaged over the 16³ grid: 512 features.

Example (made-up but typical numbers): a fruit whose voxels above 0.1 have mean 0.83 and SD
0.12. A tissue voxel of 0.95 becomes (0.95 − 0.83) / 0.12 = 1.0; an enclosed cavity voxel of
0.12 becomes −5.9; a background voxel (0.02) becomes 0, not (0.02 − 0.83) / 0.12 = −6.75. The first run
(`20261102T180339Z`) left the background at about −6.5 and is not used; a later version that
set cavities to 0 as well made a cavity look like average tissue and was replaced (commit
1911753). Because of step 2 the features do not carry the fruit's overall brightness, unlike
Hugo's latent through InstanceNorm. Open: the skin rim enters at about −5 (question 46).

## 6. Nuisance checks (`docs/nuisance_check.md` section 3; computed 3 November 2026)

On the averaged out-of-fold score of each row, for all four rows:

- A. Within-box AUC: only pairs (one non-consumable, one consumable) from the same box. Example:
  box with non-consumable scores 0.8, 0.4 and consumable 0.5, 0.1: of the 4 pairs, 3 have the
  non-consumable fruit higher (0.8 > 0.5, 0.8 > 0.1, 0.4 > 0.1), so 0.75. Interval: 2,000
  draws of 12 boxes with replacement, seed 369; a box drawn twice counts as two boxes.
- B. Logistic regression of `binary_23` on the box mean of the score and the fruit's deviation
  from it, both divided by the SD of the score, standard errors clustered by box (not a
  random-intercept model: statsmodels has none for a binary outcome; same within-box effect by
  the Mundlak argument). Plus the intraclass correlation of the score given the grade (linear
  mixed model, random intercept per box).
- C. Grade 0 fruit only (145): Spearman correlation of the score minus its box median with
  stalk length and with body size (`derived/descriptors/20261023T155219Z_descriptors`);
  interval from 1,000 box draws.
- D (descriptors as covariates): not done (Ruben, 2 November: the stalk and size measures on
  the rescaled grid are not accurate enough).

## 7. Runs

| Row | Feature run | PLS-DA run | LVs |
|---|---|---|---|
| Hugo's latent | `derived/latents/hugo_vae/20261013T171306Z_hugo_vae` | `results/20261102T110957Z_plsda_binary23_hugo_vae` | 4 |
| CT at 32³ | `derived/raw_ct_32/20261005T142026Z_raw_ct_32` | `results/20261102T111004Z_plsda_binary23_ct_32` | 5 |
| Tissue features | `derived/tissue_features/20261102T151947Z_tissue_features` | `results/20261103T100917Z_plsda_binary23_tissue_features` | 7 |
| Med3D | `derived/med3d/20261102T185107Z_med3d_resnet18` | `results/20261103T100914Z_plsda_binary23_med3d` | 7 |

The two runs of 2 November were saved before the Youden cut-off existed: their `run.json` has
no `cutoff_youden` (to resave from notebook 08 at the same LVs). The two runs of 3 November
were saved by running notebook 08 headless with the choices filled in
(`app.run(defs=...)`: representation, 7 LVs, save), from the clean commit 57277c3.
