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

## 6. Nuisance checks (`docs/nuisance_check.md` section 3; computed 3 November 2026, saved by notebook 10)

On the averaged out-of-fold score of each row, for all six rows (the four below in section 7 and
the two height-radius rows of section 10):

- A. Within-box AUC: only pairs (one non-consumable, one consumable) from the same box. Example:
  box with non-consumable scores 0.8, 0.4 and consumable 0.5, 0.1: of the 4 pairs, 3 have the
  non-consumable fruit higher (0.8 > 0.5, 0.8 > 0.1, 0.4 > 0.1), so 0.75. Interval: 2,000
  draws of 12 boxes with replacement, seed 369; a box drawn twice counts as two boxes.
- B. Intraclass correlation of the score given the grade (linear mixed model, score ~ grade as
  categories + random intercept per box): the share of the score's variation that is a shift per
  box among fruit of the same grade. Reported beside check A.
  Dropped on 7 November 2026 (Ruben): the logistic regression of `binary_23` on the box mean
  of the score and the fruit's deviation from it (computed on 3 November). Its within-box
  coefficient repeats check A; its box-mean coefficient cannot tell a box effect in the
  features from real differences in defect rate between orchards (example: a box with 20
  non-consumable fruit out of 30 has a high mean score whether or not the features carry the
  orchard), and it rests on 12 boxes. `pearlatent.nuisance.within_between_logit` stays in the
  code but is not used.
- C. Grade 0 fruit only (145): Spearman correlation of the score minus its box median with
  stalk length and with body size (`derived/descriptors/20261023T155219Z_descriptors`);
  interval from 1,000 box draws.
- D (descriptors as covariates): not done (Ruben, 2 November: the stalk and size measures on
  the rescaled grid are not accurate enough).

## 7. Runs

| Row | Feature run | PLS-DA run | LVs |
|---|---|---|---|
| Hugo's latent | `derived/latents/hugo_vae/20261013T171306Z_hugo_vae` | `results/20261103T101426Z_plsda_binary23_hugo_vae` | 4 |
| CT at 32³ | `derived/raw_ct_32/20261005T142026Z_raw_ct_32` | `results/20261103T101429Z_plsda_binary23_ct_32` | 5 |
| Tissue features | `derived/tissue_features/20261102T151947Z_tissue_features` | `results/20261103T100917Z_plsda_binary23_tissue_features` | 7 |
| Med3D | `derived/med3d/20261102T185107Z_med3d_resnet18` | `results/20261103T100914Z_plsda_binary23_med3d` | 7 |

All four runs were saved on 3 November by running notebook 08 headless with the choices
filled in (`app.run(defs=...)`: representation, number of LVs, save), from clean commits
(57277c3 for tissue features and Med3D, a4b318f for the other two). The first runs of Hugo's
latent and CT at 32³ (`20261102T110957Z`, `20261102T111004Z`) were saved before the Youden
cut-off existed and have no `cutoff_youden` in `run.json`; the resaved runs have identical
predictions (largest difference 0.0) and replace them. The two first runs were deleted on
7 November 2026 (Ruben, question 50).

## 8. PLS-DA with one model per grade (decided 7 November 2026, Ruben)

Notebook 11. Following the chemometrics course (chapter 4, "Discrimination via regression"):
one 0/1 dummy per grade (0, 1, 2, 3), one PLS model per dummy (one-versus-rest), each with its
own number of LVs chosen by Ruben at the elbow of its mean RMSECV curve, under the same design A
as section 1. A fruit is assigned to the class whose model gives the highest predicted
membership (averaged over the 20 repeats). Reported: error rate per class and in total, the
4 × 4 table of true against assigned class, AUC of each class model against the rest.

Example: a fruit with predicted memberships 0.10 (grade 0), 0.35 (grade 1), 0.55 (grade 2),
0.20 (grade 3) is assigned to grade 2. The four values come from four separate models and need
not sum to 1.

Targets: the browning grade on the cavity-free fruit (cavity grade 0; 233 of the 315 fruit), or
the defective grade on all 315 fruit. A single PLS regression on the grade as a number
(notebook 11 of the first hours of 7 November) was deleted the same day (Ruben).

**Result and why it fails (7 November 2026, all 315 fruit, defective grade; seen in a scratch
run of the same computation as notebook 11).** For every class model, RMSECV compared with the
RMSE of no model (every fruit predicted at the class share, that is the standard deviation of
the dummy):

| Hugo's latent | no model | best RMSECV | AUC at 3 LVs |
|---|---|---|---|
| grade 0 against the rest | 0.498 | 0.416 (3 LVs) | 0.84 |
| grade 1 against the rest | 0.336 | 0.348 (1 LV) | 0.55 |
| grade 2 against the rest | 0.371 | 0.381 (1 LV) | 0.67 |
| grade 3 against the rest | 0.430 | 0.386 (3 LVs) | 0.82 |

Lowest ratio RMSECV / no model over all LVs, from the saved runs of 7 November (corrected on
9 November 2026; this text first said that grades 1 and 2 never beat no model in any row):

| | grade 1 | grade 2 |
|---|---|---|
| Hugo's latent | 1.035 | 1.027 |
| CT at 32³ | 1.029 | 1.024 |
| tissue features | 0.999 (1 LV only) | 0.938 (7 LVs; 0.938 to 0.954 at every LV) |
| Med3D | 1.001 | 0.962 (2 LVs) |

So grade 1 is not recognised by any row. Grade 2 is not recognised on the latent and CT, and
weakly on the two rows that carry whole-fruit features (AUC of their own model 0.73 and 0.75).
The browning target (cavity-free fruit) was only seen in the scratch run of 7 November. The
spread of RMSECV over the 20 repeats is 0.003 to 0.008, so the values above 1 are no instability
between repeats but absence of signal. Highest-membership assignment (latent, LVs at the minimum):
error rate 0.394 against 0.540 for always grade 0; 2 fruit assigned to grade 1 and 10 to
grade 2, out of 315.

Reason: on the binary PLS-DA prediction (section 3) the grades lie in order (latent, median
per grade 0.15, 0.28, 0.61, 0.75; same order for the other three rows). A linear model cannot
give a middle grade a high value while both neighbours get a low one. One-versus-rest PLS-DA
fits classes that are separate groups (varieties, origins), not steps on one scale. Kept as a
reported negative result (Ruben, 7 November 2026). Notebook 11, section 1, shows the four
curves in one plot as RMSECV divided by the no-model RMSE (dashed line at 1). LVs chosen by
Ruben on 7 November (grade 0/1/2/3): latent 3/1/1/3, CT 5/1/1/5, tissue features 10/1/1/7, Med3D
8/3/2/7.

## 9. Ordinal model on the binary PLS-DA prediction (7 November 2026, Ruben)

Notebook 12, module `pearlatent.ordinal`. Input: the saved out-of-fold prediction of the four
`binary_23` runs (section 7), no new PLS model and no new choice of LVs. Model: cumulative
logit (`statsmodels` `OrderedModel`, `distr="logit"`, no constant),
P(grade ≤ j) = logistic(c_j − β · prediction), j = 0, 1, 2. Fitted per repeat and fold of the
same 20 × 4-fold box design on the training boxes, applied to the held-out boxes, probabilities
averaged over the repeats; assignment to the most probable grade. Reported: error rate,
quadratic weighted kappa, Spearman correlation of the expected grade with the true grade,
share within one grade, recall per grade, split by cause.

Limit: the saved prediction of a training fruit is an average of fold models, some fitted on
the held-out boxes. The ordinal step has four parameters, so the effect is small, but the
scheme is not fully nested.

Example (latent, fit on all 315 fruit): β = 4.91, boundaries on the prediction axis
c_j / β = 0.325 (grade 0 | 1), 0.500 (1 | 2), 0.749 (2 | 3). A fruit with prediction 0.40
lies between the first two boundaries; because grade 0 holds 145 fruit and grade 1 only 41
and their predictions overlap, its most probable grade is still 0.

First result (7 November 2026; saved as `results/20261107T170442Z_ordinal_defective`, six rows):

| | error rate | recall grade 0 | 1 | 2 | 3 |
|---|---|---|---|---|---|
| Hugo's latent | 0.406 | 0.924 | 0.000 | 0.019 | 0.675 |
| CT at 32³ | 0.416 | 0.931 | 0.000 | 0.000 | 0.636 |
| tissue features | 0.349 | 0.986 | 0.000 | 0.212 | 0.662 |
| Med3D | 0.349 | 0.966 | 0.000 | 0.269 | 0.662 |

Always grade 0: error rate 0.540. No representation assigns a grade 1 fruit to grade 1 (Med3D
assigns 3 fruit of other grades to grade 1, the other rows none); most
grade 1 fruit are assigned grade 0, and grade 2 fruit mostly to grade 3.

Check on the limit "the score was trained on grade ≥ 2 or not" (7 November 2026, scratch run,
same design A). AUC between neighbouring grades (0 | 1, 1 | 2, 2 | 3) of three out-of-fold
scores, each averaged over the 20 repeats:

| | `binary_23` (saved run) | PLS on the grade 0 to 3, 3 to 7 LVs | PLS on `binary_123`, 3 to 7 LVs |
|---|---|---|---|
| Hugo's latent | 0.65, 0.81, 0.62 | 0.61 to 0.65, 0.78 to 0.79, 0.60 to 0.63 | 0.63 to 0.65, 0.72 to 0.78, 0.56 to 0.62 |
| Med3D | 0.72, 0.86, 0.64 | 0.67 to 0.70, 0.86 to 0.89, 0.59 to 0.65 | 0.66 to 0.70, 0.87 to 0.88, 0.57 to 0.62 |

A score trained on the full grade, or on the 0 | 1 cut itself, does not separate grade 0 from
grade 1 better than the `binary_23` score. The weak 0 | 1 and 2 | 3 separation is therefore not
caused by the target the score was trained on.

## 10. Height-radius rows (reopened 7 November 2026, Ruben)

The decision of 10 October 2026 dropped the cylindrical bins everywhere. Ruben reopened them as
two model rows on 7 November 2026, because the height-radius maps per grade (notebooks 06 and
07) showed the grades in a clear order. Noted: the rows were added after seeing those maps
(open boxes only; the sealed boxes F, J, M stay closed), and a clear order of group means does
not by itself mean that single fruit separate; the box cross-validation decides.

Rows: `hugo_vae_hr` (Hugo's latent `mu`) and `ct_32_hr` (CT at 32³, same bins on the same body
cells), module `pearlatent.height_radius`, loaded through `load_features`. Per fruit the mean of
all 32³ body cells in each of 20 height × 8 radius bins (`latent_map_sums`, radius per slice:
a radius bin is a relative depth below the skin), 160 features, centred only like the grid
rows. Reason beyond the maps: rotation about the stem-calyx axis is not aligned between fruit
(CLAUDE.md section 6), and the mean over a ring does not depend on it.

Empty bins (a slice has few cells and may not reach every radius bin; no fruit of the 315 has a
whole height bin empty, corrected 9 November 2026):
each takes the mean of the nearest filled bins of the same fruit, in bin steps; no labels, no
other fruit. On the 315 fruit: median 9, at most 22 of 160 bins filled per fruit; mean number of
filled bins per radius bin, axis to skin, 3.37, 1.72, 1.43, 1.00, 0.87, 0.66, 0.58, 0.00 (of 20),
so the axis bin is the most filled. Example: a
fruit with bin (height 0, radius 1) empty and neighbours 1.0, 3.0 (same height) and 6.0 (next
height) gets (1 + 3 + 6) / 3 = 3.33.

First look (7 November 2026, scratch run of the notebook 08 computation, `binary_23`, 315
fruit, design A; LVs not yet chosen by Ruben). Mean RMSECV at 1 to 5 LVs: latent 0.398, 0.357,
0.339, 0.354, 0.366; CT 0.431, 0.406, 0.381, 0.381, 0.395. AUC at 3 LVs: latent 0.927, CT
0.884 (grid rows, saved runs: latent 0.887 at 4 LVs, CT 0.839 at 5 LVs). AUC between
neighbouring grades at 3 LVs (0 | 1, 1 | 2, 2 | 3): latent 0.64, 0.88, 0.65; CT 0.66, 0.78,
0.60.

Saved runs (Ruben, 7 November 2026, 3 LVs each, clean commit b8af65a):
`results/20261107T165738Z_plsda_binary23_hugo_vae_hr` (AUC 0.927) and
`results/20261107T165755Z_plsda_binary23_ct_32_hr` (AUC 0.884); added to
`PLSDA_BINARY23_RUN_DIRS`, so notebooks 10 and 12 include them.

Saved on 7 November 2026 as `results/20261107T170510Z_compare_plsda_binary23` and
`results/20261107T170442Z_ordinal_defective`:
- Paired AUC difference, height-radius minus full grid: latent +0.040 (0.005 to 0.077), CT
  +0.045 (−0.007 to 0.102). Latent height-radius minus CT height-radius +0.043 (0.008 to
  0.082); Med3D minus latent height-radius +0.031 (−0.008 to 0.073); tissue features minus
  latent height-radius +0.011 (−0.018 to 0.042).
- Check A: within-box AUC latent height-radius 0.904 (0.841 to 0.949), CT height-radius 0.867;
  ICC given grade 0.101 and 0.094 (grid rows 0.097 and 0.062).
- Check C, latent height-radius, 145 healthy fruit: Spearman of the score with stalk length
  −0.306 (−0.491 to −0.203) and with body size (body voxels) +0.342 (0.189 to 0.523); both
  intervals exclude 0. The CT height-radius row does not show this (−0.021 and +0.037). On
  healthy fruit, the latent height-radius score therefore rises with fruit size and falls with
  stalk length: part of it describes the fruit, not the defect. Not yet explained.
- Ordinal model (notebook 12): latent height-radius error rate 0.365, weighted kappa 0.697,
  recall per grade 0.945, 0.000, 0.250, 0.649; CT height-radius 0.406, 0.590.

## 11. Interval PLS on the height-radius rows (7 November 2026, Ruben)

Notebook 13. Forward iPLS (Nørgaard et al. 2000; the SOLO description checked on 7 November:
same forward mode, non-overlapping intervals, each model at the lowest point of its own RMSECV
curve, `pearlatent.ipls.forward_ipls` as notebook 09) on the 160 height-radius features, with
two ways to cut them: 20 height intervals (8 radius bins each; bin 0 calyx end, 19 stalk end)
and 8 depth intervals (20 height bins each; bin 0 axis, 7 skin). One difference from SOLO: the
path runs to the end and its lowest point is taken, where SOLO with `numintervals = inf` stops
at the first step without improvement.

First result (`binary_23`, 315 fruit, design A; seen in a headless run, figures only):
- Latent, heights alone: lowest RMSECV in bins 7, 8, 6, 9 (0.357, 0.360, 0.367, 0.379; all
  160 features at 3 LVs 0.339, no model 0.492: 129 of 315 non-consumable); bins 0 to 2 and 14 to 19 alone are near
  no model (0.47 to 0.50). Forward path: 0.357 with bin 7, 0.338 after three bins, lowest
  0.330 with ten bins.
- Latent, depths alone: bins 3, 2, 4 best (0.362, 0.363, 0.371); axis (0) 0.438, the two
  outer bins (6, 7) 0.489 and 0.490. Lowest path point 0.337.
- CT, heights alone: bins 7, 8, 6 best (0.420, 0.421, 0.429); depths: bin 3 best (0.384).
  Lowest path points 0.368 (heights) and 0.372 (depths), all features 0.381.

Reading: the information sits in the middle of the body height (about 30 to 50 % of the way
from the calyx end) and at middle depth, for the
latent and the CT alike. Selecting intervals lowers RMSECV only slightly (latent 0.339 to
0.330), within the optimism of choosing on the same cross-validation, so no interval
selection is proposed as a model; the result is a location map.

Limit on the depth reading (9 November 2026): the axis bin is the weakest alone, but it also holds
the fewest cells (a ring near the axis is small) and the most filled values (about 3.4 of its 20
bins per fruit), so a poor score there does not show that the axis carries no information. For the
latent, one cell also sees about 8 cells around it (receptive field 31 voxels), much wider than
one bin.

## 12. Fruit-wide level and grade 0 corrections (scratch, 9 November 2026)

Question 52 asked why the latent height-radius score follows size and stalk on healthy fruit.
Scratch run: not saved as a run, 315 fruit, design A, 3 LVs, height-radius rows. "Stalk" and "size" are
check C on the 145 grade 0 fruit. The last column is the paired difference with the base row.

| Latent height-radius | AUC | within-box | stalk | size | minus base |
|---|---|---|---|---|---|
| base | 0.927 | 0.904 | −0.306 | +0.342 | |
| minus the fruit's own mean | 0.923 | 0.900 | −0.230 | +0.279 | −0.004 (−0.017 to 0.007) |
| minus grade 0 mean of the training boxes | 0.927 | 0.904 | −0.306 | +0.342 | 0.000 (exactly) |
| same, divided by the grade 0 SD | 0.934 | 0.920 | −0.251 | +0.348 | +0.007 (−0.005 to 0.023) |

| CT height-radius | AUC | within-box | stalk | size | minus base |
|---|---|---|---|---|---|
| base | 0.884 | 0.867 | −0.021 | +0.037 | |
| minus the fruit's own mean | 0.884 | 0.862 | −0.065 | +0.052 | 0.000 (−0.028 to 0.028) |
| minus grade 0 mean of the training boxes | 0.884 | 0.867 | −0.021 | +0.037 | 0.000 (exactly) |
| same, divided by the grade 0 SD | 0.909 | 0.892 | +0.098 | −0.008 | +0.026 (0.006 to 0.052) |

Reading:

- Own mean: the AUC does not depend on the fruit-wide level, but the size and stalk link only
  shrinks and its interval still excludes 0 (stalk −0.230, −0.381 to −0.139; size +0.279, 0.155
  to 0.446). The level is not the main route of that link; question 52 stays open. After this
  correction the RMSECV curve has its lowest point at 2 LVs (0.336), not 3 (0.346).
- Minus the grade 0 mean of the training boxes (decision of 27 October 2026,
  `docs/methods_step0_and_maps.md` section 9): no effect on PLS at all. It subtracts one constant
  from a bin for all fruit, and PLS centres each bin anyway. Numeric example: bin values 5, 7, 9
  centre to −2, 0, 2; minus a grade 0 mean of 6 they become −1, 1, 3, which centre to −2, 0, 2.
- Divided by the grade 0 SD: this is scaling per bin, not centring, so it does change the model.
  It is close to `scale=True`, using the SD of the grade 0 training fruit. The reason for
  centring only (background cells would gain weight, section 2) does not apply to height-radius
  bins, which hold body cells only. These are extra tries on the same folds, with the LVs kept at
  3, so the gains are optimistic. Not a decision: question 54 in `HANDOFF.md`.

## 13. Mean and SD of each height-radius bin, the fruit interior, the encoder SD (9 November 2026, Ruben)

**Mean and SD rows** (`hugo_vae_hr_sd`, `ct_32_hr_sd`; `pearlatent.height_radius`,
`with_sd=True`). The same 20 height × 8 radius bins as section 10. Per bin: the mean of the
body cells, and their SD (divided by n), so how uneven that ring is. 320 numbers per fruit,
centred only. A bin with one cell gets SD 0; empty bins are filled as in section 10.

Example: a ring of 4 cells with values 2, 2, 2, 2 has mean 2 and SD 0; a ring with 1, 3, 1, 3
has the same mean 2 and SD 1. The mean row cannot tell the two apart.

Found by looking (a scratch run of 9 November, after all the earlier rows had been seen), so
it is an idea the sealed boxes must test, not a pre-registered row. LVs chosen by Ruben at the
elbow (9 November 2026): latent 3 (mean RMSECV 0.377, 0.335, 0.311, 0.319 at 1 to 4 LVs), CT 3
(0.406, 0.363, 0.353, 0.364). Runs `results/20261109T173414Z_plsda_binary23_hugo_vae_hr_sd`
(AUC 0.957, box bootstrap 0.939 to 0.971) and `20261109T173415Z_plsda_binary23_ct_32_hr_sd`
(0.931, 0.903 to 0.947); comparison `results/20261109T173617Z_compare_plsda_binary23` (28
pairs). Scratch checks, not saved as runs: fewer SD features do not keep the gain (one SD per
height bin 0.929, per depth bin 0.933, one per fruit 0.928, against 0.927 for the means
alone); a z-score of each fruit's body cells first adds +0.002 (−0.014 to +0.014); adding the
encoder SD per bin gives −0.008 (−0.028 to +0.008).

**Interior spread** (notebook 15, `pearlatent.interior`; run
`results/20261109T171633Z_interior_spread`). One number per fruit, no model: the SD of the grid
over the body cells without their outer layer (6-neighbour erosion, one cell, about 2.5 mm). The
outer layer straddles skin and air and varies in every fruit. Chosen by looking: CT SD over all
body cells, grade 0 against 1 on cavity-free fruit, AUC 0.53; without 1, 2 or 3 layers 0.82.

Example: a fruit whose interior cells are 0.84 everywhere except a patch of 0.78 has a larger
interior SD than a fruit at 0.84 everywhere, even if its mean hardly moves.

**Encoder SD** (`pearlatent.logvar_run`, run `derived/logvar/20261109T171513Z_logvar`, 346 open
2526 fruit and Hugo's 66 validation fruit). The encoder gives a second 32³ grid beside `mu`,
`logvar`; its SD exp(logvar / 2) is about 0.001 inside the fruit against a spread of `mu` of
about 5 (KL weight 1e-6), so it cannot scale `mu`. Used only as one number per fruit: the median
over the interior cells. `mu` recomputed in the same run matches the saved `mu` to 1.4e-4. The
run records `dirty: true`: notebook 15 was untracked when it finished; `logvar_run.py` itself
is unchanged since commit 868af71.

**All volumes and reverse check** (notebook 16, run
`results/20261109T173656Z_all_volumes_reverse_check`). Every saved `binary_23` row refitted
under design A on all 346 open 2526 volumes (faulty volumes kept), its saved LVs and scaling;
and each row's general model (fit on the 315 fruit) applied to Hugo's 61 validation fruit left
after the faulty volumes. In those 61, all 29 non-consumable fruit have a cavity and the defect
follows the storage group, so the reverse check tests the change of season, not browning.

## 14. Cavity-free browning and counterfactual steps (9 November 2026, Ruben)

**Targets beside `binary_23`** (notebook 14, `pearlatent.targets`; charter rule 6). Kept:
browning grade ≥ 2 on the cavity-free fruit (233 of 315, 50 positive), design A, LVs chosen by
Ruben at the elbow. Dropped (Ruben): `binary_123` (section 9: training on it separates grade 0
from 1 no better) and the severity score (a model fitted on grade 0 and grade 3 fruit only,
predicting all held-out fruit; `cross_validated_predictions(..., fit_on=...)`), after one run
showed grade 0 | 1 at 0.61 against 0.64 for `binary_23` (latent height-radius).

Cavity-free runs `results/20261109T1717*` (clean, commit d451c6b):

| Row | LVs | AUC (95 % box interval) | 0 \| 1 | 2 \| 3 |
|---|---|---|---|---|
| Med3D | 3 | 0.945 (0.915 to 0.969) | 0.71 | 0.62 |
| Latent height-radius mean + SD | 2 | 0.942 (0.896 to 0.974) | 0.67 | 0.75 |
| Tissue features | 7 | 0.927 (0.867 to 0.960) | 0.64 | 0.68 |
| Latent height-radius | 2 | 0.916 (0.867 to 0.957) | 0.65 | 0.73 |
| CT height-radius mean + SD | 3 | 0.890 (0.837 to 0.934) | 0.57 | 0.70 |
| Latent grid | 5 | 0.848 (0.805 to 0.887) | 0.57 | 0.70 |
| CT height-radius | 3 | 0.840 (0.772 to 0.900) | 0.59 | 0.71 |
| CT grid | 4 | 0.730 (0.658 to 0.781) | 0.59 | 0.65 |

**Counterfactual steps** (notebook 17, `pearlatent.counterfactual`, `counterfactual_run`).

1. Direction: the regression vector b of one PLS-DA model on the full 32³ latent grid, fitted
   on all open fruit of its target (cavity-free: 233 fruit, 5 LVs; general `binary_23`: 315
   fruit, 4 LVs). The two b point largely the same way (cosine 0.74).
2. Scale t (Ruben): t = 0 at the mean fitted score of grade 0 fruit, t = 1 at grade 3, as
   Keicher et al. (2023). Cavity-free: 0.042 and 0.878, so t = 0.5 is the score 0.46.
3. Step to t: `mu` moves by (target − current score) × b / (b · b), the smallest change that
   gives the target score. Example: b = (2, 0), score 0.2, target 1.0: change (0.4, 0).
   Why not the line between the grade 0 and grade 3 centroids itself (Ruben's first idea): on
   the 145 healthy cavity-free fruit, the position along that line follows body size
   (Spearman 0.91) and stalk length (−0.75); along b it does not (0.07, −0.04). The step along
   b is the projection of the centroid line onto b (the part the model uses).
4. Size of the steps: every cell changes by less than 2 of its own SDs over the 315 fruit, so
   the moved latent stays within the range of real fruit.
5. Decode (`vae.decode`, 295 s and 18 GB on the Mac CPU), min-max scale, encode again, predict
   with the same model. The plain reconstruction is re-encoded as the control.
6. Draw: 2 grade 3 (browning 3, cavity 0) and 2 grade 0 fruit from 4 boxes, seed 369; fruit
   Ruben rejects on their CT are skipped (2526_I09). Fruit: C23, N17, K07, A03.

First result, 2526_C23 (t 1.38) towards healthy:

| Step | target score | re-encoded score (t) |
|---|---|---|
| plain reconstruction | 1.196 | 1.089 (1.25) |
| t = 0.5 | 0.460 | 0.709 (0.80) |
| t = 0 | 0.042 | 0.431 (0.47) |

The decoded image keeps about half of each shift. The core brightens by only 0.008 (the rest of
the body by 0.005); the dark spots, already smoothed in the plain reconstruction, stay. At t = 0
a stalk appears above the neck (6,790 voxels outside the body change by more than 0.1). The
direction carries stalk and shape besides browning: grade 0 fruit have longer stalks in the grid
(question 38), and the scores follow the stalk on healthy fruit (question 52).

Options to remove the stalk part (question 55, not decided):
- b on body cells only (set b to 0 outside the cells inside the body of most fruit). Removes the
  stalk cells, but InstanceNorm passes the stalk into every cell, so not all of it.
- Project b orthogonal to the stalk direction s (the latent direction that predicts stalk
  length on the healthy training fruit): b' = b − (b · s / s · s) s. Example: b = (1, 1),
  s = (1, 0): b' = (0, 1).
- External parameter orthogonalisation (EPO, Roger et al. 2003): remove the subspace of the
  latent that varies with stalk length from every fruit before PLS, inside the folds. The
  standard chemometric tool for a known nuisance; costs a new model row.
All three depend on the stalk measure, which Ruben judged not accurate enough as a covariate on
2 November (check D).

The other three fruit (9 November 2026; `derived/counterfactuals/20261109T190441Z_2526_K07_*`,
`20261109T194041Z_2526_A03_*`, `20261109T195712Z_2526_N17_*`; figures
`derived/figures/17_<run>.png`). Re-encoded score, t in brackets:

| Fruit (start t) | plain | t = 0.5 | last step | kept of the shift |
|---|---|---|---|---|
| K07, grade 0 (0.01), to t = 1 | −0.146 (−0.23) | 0.040 (0.00) | 0.223 (0.22) | 44 % |
| A03, grade 0 (0.11), to t = 1 | 0.003 (−0.05) | 0.234 (0.23) | 0.546 (0.60) | 73 % |
| N17, grade 3 (0.87), to t = 0 | 0.643 (0.72) | 0.420 (0.45) | 0.127 (0.10) | 71 % |
| C23, grade 3 (1.38), to t = 0 | 1.089 (1.25) | 0.709 (0.80) | 0.431 (0.47) | 57 % |

"Kept of the shift": change of the re-encoded score from the plain reconstruction to the last
step, divided by the change of the target score from the start score (K07: 0.369 / 0.830). The
decode-encode round trip alone moves the score by −0.19 to −0.13. No fruit reaches the other
grade mean in the re-encoded score.

What changes in the images (long section through the body centre, step minus plain):
mostly the outline (edge voxels brighter on one side, darker on the other: a shift of shape),
and a slight darkening around the core of the two grade 0 fruit towards t = 1. At t = 1 a small
bright spot appears above the neck of K07, the same place as the stalk of C23 at t = 0; N17 and
A03 show none. The direction changes shape and the neck region more than the tissue.
Ruben, 9 November 2026: the projection and EPO are preferred, as part of the counterfactual work,
in a session after the meeting of 17 November. First a better stalk length is needed (questions
41 and 18 in `HANDOFF.md`).

## 15. SVM-DA check row (20 November 2026, Ruben)

**Why.** The CT interior SD alone separates grade 0 from grade 1 (AUC 0.82, section 13), every
linear PLS-DA row does not (0.59 to 0.72). A weighted sum of cells cannot form a spread, so one
non-linear model was added as a check. This reopens "SVM not added" (2 November).

**What** (`pearlatent.svm`, notebook 19, run
`results/20261120T090152Z_svm_binary23_hugo_vae_hr_sd`). Support vector machine with a radial
basis function kernel on `hugo_vae_hr_sd` (320 features), `binary_23`, the 315 fruit and the
20 × 4 box folds of the PLS-DA rows. Every feature standardised inside the training fold (the
kernel uses distances; PLS-DA on the same features centres only). Grid fixed before any result:
C 0.1, 1, 10, 100; gamma = factor / 320 with factor 0.1, 1, 10. Rule (Ruben): the setting with
the highest AUC of the decision value averaged over the 20 repeats: C 1, factor 0.1. The chosen
AUC is the best of 12 tries on the same boxes, so slightly optimistic.

Example: with gamma factor 0.1, two fruit at a standardised distance of 10 (‖a − b‖² = 100)
have kernel value exp(−0.1 / 320 × 100) = 0.97, so the boundary is nearly linear; at factor 10
it is exp(−3.1) = 0.04, so each training fruit only influences its close neighbours.

**Found.** AUC 0.967 (0.948 to 0.980) against 0.957 for PLS-DA on the same features; paired
difference +0.011 (−0.004 to +0.025), SVM higher in 93 % of the box draws. The grid is flat for
factor ≤ 1 (0.948 to 0.967) and fails at factor 10 (0.51 to 0.56). Grade 0 against browning
only 1: 0.68 (PLS-DA 0.67); cavity 2 or 3: 0.98 (0.95). Check C on the 145 healthy fruit: stalk
−0.06 (−0.31 to +0.20), size +0.09 (−0.20 to +0.35), against −0.29 and +0.30 for PLS-DA.
Scratch (not saved): PLS-DA standardised, 3 LVs, stalk −0.26, size +0.28; linear SVM stalk
−0.14 to −0.16, size +0.20 to +0.23. So standardising alone does not remove the link.

**Reading.** No clear gain from the non-linear boundary, and grade 1 is not rescued on these
features. Check C to confirm: a correlation near 0 can also mean that the SVM score carries
little information among healthy fruit (an RBF score flattens far from the boundary); the
spread of the healthy-fruit scores is not yet checked.

## 16. Core-only rows (20 November 2026, Ruben; dropped)

**Why.** The stalk and the neck could carry the stalk and size link of the latent rows (check C,
section 6). If only the core region enters the model, that link should drop.

**Region** (`pearlatent.core_region`, notebook 21). Height over the fruit body (from
`body_mask`, 0 = calyx end, 1 = top of the body) from 0.05 to 0.60, all radii. Cells outside a
fruit's own region get the training-fold mean of the fruit that have the cell inside their
region, so after centring they are 0 and do not enter the model; the fill is fitted inside
every fold. The full 32³ grid is kept, so a counterfactual can still be decoded.
Dropped options:
- a radius limit of 0.5: the browning of grade 3 fruit L14 and A15 lies at radius 0.5 to 1;
- one fixed set of cells shared by 95 % of the fruit: it kept a median 57 % of each fruit's own
  region, 48 % for browning 3 against 61 % for grade 0, r = −0.94 with body size. A size effect
  of its own.

**Found** (scratch, latent grid, 4 LVs, against the full grid):

| Row | AUC | Check C stalk | Check C size |
|---|---|---|---|
| Latent full grid | 0.887 | −0.20 | +0.12 |
| Latent core only | 0.869 | −0.19 | +0.27 |
| CT 32³ full grid | 0.839 | | |
| CT 32³ core only | 0.821 | | |

**Reading.** The stalk link does not sit in the neck and stalk cells; the likely route is the
rescaling of every fruit to 128 voxels in length (a long stalk makes the body smaller in the
volume). The size link got worse. Ruben: dropped, not continued.

## 17. EPO on the healthy fruit (20 November 2026, Ruben)

**What** (`pearlatent.epo`, `cross_validated_predictions(..., epo_healthy=...)`, notebook 22).
External parameter orthogonalisation (EPO, Roger et al. 2003; course Chemometrics 2020,
chapter 2): X* = X (I − P Pᵀ), where P holds the first principal components of a matrix D that
contains only nuisance variation. Our D: the healthy (grade 0) training fruit, centred on their
own mean (variant B). One component (`N_DIRECTIONS = 1`). Fitted inside every fold, on the
training boxes only. A variant of the textbook method, which uses the same objects measured
while the nuisance varies (question for Wouter Saeys).

Example: P = (0.6, 0.8)ᵀ, a fruit x = (1, 2): x · P = 2.2, x* = (1, 2) − 2.2 (0.6, 0.8) =
(−0.32, 0.24), and x* · P = 0.

Why it suits a counterfactual: a PLS model fitted on X* has its regression vector b in the row
space of X*, so b · P = 0. A step along b then cannot move a fruit along the removed axis, and
the model gives the same prediction on X and on X*.

**The removed axis** (latent grid, all 145 healthy fruit, scratch): healthy PC 1 carries 22 % of
the healthy variation and follows body size (r = 0.97) and stalk length (r = −0.83); PCs 2 to 4
do not (|r| ≤ 0.11). Its pattern is the band at the body outline with the opposite sign inside.
It points the same way as variant A, the latent's covariance with body size (cosine 0.99), so
the PCA finds the size axis without the measured size, which Ruben judged not accurate enough as
a covariate (2 November).

**Found** (scratch, design A folds, check C on the 145 healthy fruit):

| Row (LVs) | AUC none → EPO | Stalk none → EPO | Size none → EPO |
|---|---|---|---|
| Latent full grid (4) | 0.887 → 0.888 | −0.20 → −0.10 | +0.12 → +0.07 |
| Latent height-radius mean + SD | 0.957 → 0.952 | −0.29 → +0.07 | +0.30 → −0.04 |
| Latent height-radius | 0.927 → 0.908 | −0.31 → +0.02 | +0.34 → −0.01 |
| CT height-radius | 0.884 → 0.880 | −0.02 → −0.17 | +0.04 → +0.13 |
| CT height-radius mean + SD | 0.931 → 0.920 | +0.00 → −0.19 | +0.02 → +0.13 |
| CT 32³ full grid | 0.839 → 0.837 | +0.01 → −0.04 | +0.02 → +0.04 |

Variant A on the latent full grid: AUC 0.887, stalk −0.14, size +0.12. Latent height-radius
paired difference −0.019 (−0.045 to +0.001).

**Reading.** On the latent rows EPO removes the stalk and size link at no clear cost in AUC. On
the CT height-radius rows the healthy PC 1 is not the size axis (r 0.41 to 0.43 with body
size), so EPO does not suit them and makes the link worse.

**EPO on the grid, then binning to height-radius** (scratch): no change (height-radius AUC
0.927, stalk −0.28, size +0.27; mean + SD 0.956, −0.31, +0.29). EPO has to be fitted in the
features the model uses. Ruben: this is a result, not a failure; the goal of EPO here is a
cleaner direction for the counterfactuals.

**Figures (23 November 2026, notebook 22 sections 5 and 6).**
- PCA of the latent grid, 315 fruit, the healthy PC 1 fitted once on all 145 healthy fruit
  (description only, no model): before, PC 1 holds 22 % (size r +0.97, stalk −0.78, grade
  +0.18) and PC 1 against PC 2 is a horseshoe, the shape PCA gives one strong gradient; after,
  PC 1 holds 11 % (size +0.12, stalk −0.25, grade +0.09), the horseshoe is gone, but the
  largest fruit (about 250,000 body voxels and more) still group together
  (`derived/figures/22_pca_before_after_epo.png`).
- Check C for the latent height-radius mean + SD row, 3 LVs, recomputed in the notebook: AUC
  0.957 → 0.952, size +0.30 → −0.04, stalk −0.29 → +0.07 (the scratch numbers above). In the
  figure, with EPO the smallest healthy fruit (below about 100,000 body voxels) score up to
  +0.4 (`derived/figures/22_check_c_before_after_epo.png`).
- Reading: the size effect is curved; one EPO component removes its straight part only.
  Example: if the score of healthy fruit followed (size − mean)², a straight direction along
  size would remove nothing of it, and a rank correlation could be near 0 while the smallest
  and the largest fruit both score high. Options in `HANDOFF.md` question 62.

## 18. Counterfactual steps along the EPO direction, and the mirrored step (20 November 2026, Ruben)

**What.** Same procedure as section 14 (same 4 fruit, same decode), with the direction
`cavity_free_epo`: the cavity-free browning ≥ 2 PLS-DA on the latent grid with EPO (section 17),
4 LVs, 233 fruit. Scale t: grade 0 mean score 0.042, grade 3 mean 0.872; the decision boundary
(score 0.5) lies at t = 0.55. Check: the step's part along the removed axis is at most 2.3e-12
for every step (0 by construction).

New step, the **mirrored step** (`counterfactual_run --flip`, Ruben): target score =
2 × 0.5 − start score, the mirror image of the fruit's score in the decision boundary.
Example: N17, start score 0.777 → target 0.223.

**How to read "kept".** The decoded image is encoded again and scored. Two numbers are compared:
- asked change = target score − the fruit's own score;
- got change = re-encoded score of the step − re-encoded score of the plain reconstruction
  (`vae.decode(mu)` without a step). Subtracting the plain reconstruction removes what the
  decode-encode round trip does on its own.
- kept = got change / asked change.

Example, C23 to the grade 0 mean: own score 1.163, target 0.042, asked change −1.121. Plain
reconstruction re-encoded 1.060, step re-encoded 0.429, got change −0.631. Kept 0.631 / 1.121 =
0.56: the re-encoded image shows 56 % of the score change that was put into `mu`.

**Found** (runs `derived/counterfactuals/20261120T16*_epo`, `20261120T17*_epo` and, mirrored,
`20261120T18*`/`19*_epo`; notebook 17 sections 4 and 5; figures
`derived/figures/17_asked_against_got.png`, `17_all_runs_images.png`). t asked → t got:

| Fruit (own t; plain re-encoded t) | halfway | to the other grade mean | mirrored | kept |
|---|---|---|---|---|
| C23, grade 3 (1.35; 1.23) | 0.50 → 0.80 | 0.00 → 0.47 | −0.25 → 0.31 | 0.51 to 0.57 |
| N17, grade 3 (0.89; 0.74) | 0.50 → 0.46 | 0.00 → 0.11 | 0.22 → 0.26 | 0.71 to 0.72 |
| K07, grade 0 (0.00; −0.23) | 0.50 → −0.01 | 1.00 → 0.20 | 1.10 → 0.25 | 0.43 to 0.44 |
| A03, grade 0 (0.10; −0.05) | 0.50 → 0.23 | 1.00 → 0.61 | 1.00 → 0.61 | 0.71 to 0.73 |

1. The old direction (section 14, no EPO, 5 LVs) gives the same re-encoded t within 0.02 for
   every fruit and step, and the same images.
2. Kept is constant within a fruit (the spread over its steps is at most 0.06) but differs
   between fruit (0.43 to 0.73). The round trip alone moves t by −0.12 to −0.23.
3. Mirrored, re-encoded score against the boundary 0.5: C23 0.30, N17 0.26 and A03 0.55 cross;
   K07 0.25 does not.
4. Along the removed size axis the re-encoded fruit moves by −90 to −124 (projection of
   re-encoded `mu` minus own `mu` on the unit axis), but within one fruit by at most 7 between
   steps of very different length (step length 34 to 135): the move comes from the round
   trip, not from the step.

**Images** (long section, step minus plain reconstruction). Largest change: the outline
(brighter on one side, darker on the other: a change of shape). Then a bright patch at the
stalk end: C23 towards healthy and K07 towards defective, at the far and mirrored steps.
Smallest: the tissue: darker around the core of A03 towards defective, slightly brighter
inside C23 towards healthy.

**Reading.** EPO removed one size axis from the model, but not what the decoded direction
shows: outline and stalk still dominate, the core change is the smallest part. Not known: why
only part of the step comes back. Either the decoder does not turn every change of `mu` into
an image change, or the encoder does not return to the moved `mu`; these runs cannot tell the
two apart. 4 fruit; examples only.

**Sampled reconstruction (Ruben's question, 21 November 2026).** Not used: `vae(x)` adds
noise with SD exp(logvar / 2), about 0.001 per cell (section 13), against a step of root mean
square 0.2 to 0.75 per cell (step length 34 to 135 over 32,768 cells). The noise would not
change the images or scores visibly and would make each decode differ (section 2 of
`methods_step0_and_maps.md`: `vae.decode(mu)`, never `vae(x)`).

## 19. A model on an iPLS selection: heights, then rings (22 November 2026, Ruben)

**What** (notebook 13 sections 3 to 5; `pearlatent.step0.cell_bins`). On the latent
height-radius mean + SD row (`hugo_vae_hr_sd`, 320 features; an interval holds the means and
the SDs of its bins):
1. forward iPLS over the 20 height intervals (section 11); Ruben chooses how many to keep;
2. inside the kept heights, one forward pass over the 8 rings; Ruben chooses how many;
3. PLS-DA on the kept bins, design A folds, centring only, LVs by Ruben at the elbow; AUC,
   paired difference against the saved all-feature row, recall and precision at 0.5 and at
   the Youden cut-off, AUC per defect group, check C;
4. one fit on all 315 fruit; its weights drawn on the 20 × 8 map and on the 32³ body cells of
   one fruit, each cell taking the weight of its bin (`cell_bins`, the rule of the features).

Example of a feature index: height bin 4, ring 3 → mean at 8 × 4 + 3 = 35, SD at 160 + 35 = 195.

**Choice and result** (run `results/20261122T162303Z_ipls_selection_hugo_vae_hr_sd`). Heights
1, 2, 4, 5, 7, 10, 11 (the lowest point of the height path at 7 intervals, RMSECV 0.303 against
0.311 for all 20; from 5 intervals on the path is flat, 0.305 to 0.303), all 8 rings (the ring
path falls from 0.317 for ring 3 alone to 0.303 with all 8), 3 LVs; 112 features.

| | Selection | All 320 features (3 LVs) |
|---|---|---|
| AUC | 0.966 | 0.957 |
| Paired difference | +0.010 (−0.001 to +0.024) | |
| Recall / precision at 0.5 | 0.775 / 0.962 | 0.736 / 0.941 |
| Youden cut-off; recall / precision | 0.373; 0.922 / 0.888 | 0.355; 0.915 / 0.868 |
| Browning only 1 / 2 / 3 against grade 0 | 0.670 / 0.961 / 0.990 | 0.671 / 0.941 / 0.989 |
| Cavity 2 or 3 against grade 0 | 0.967 | 0.954 |
| Check C stalk / size | −0.29 / +0.33 | −0.29 / +0.30 |

Weights of the final fit: positive for the SD of rings 2 to 5 at heights 2 to 7, negative for
the mean of rings 1 to 4 at heights 4 to 7.

**Reading.** Heights, rings and LVs are chosen on the same cross-validation, so the
selection's AUC is optimistic. The kept region (5 to 60 % of the body height) carries all of
the model; the stalk and size link stays; grade 1 is not rescued.

## 20. Real size and stretch factor from Hugo's originals (27 November 2026, Ruben)

**What** (`pearlatent.real_size`, run `derived/real_size/20261127T125309Z_real_size`). On 27
November Hugo gave the 2526 volumes before the rescaling to 128 voxels (15 tar files, one per
box, 434 files: our 433 plus 2526_F28, which has no 128³ volume and stays excluded). They are
cropped, aligned (stem–calyx axis along the third array axis), segmented uint16 volumes; the
header holds no voxel size. Ruben (27 November): the preprint procedure (section 2.1) was
followed for 2526, so the voxels are 0.6267 mm cubes.

**Checks** (all 433; the run repeats the per-axis check on the 346 open fruit):
- The original array is the tight crop of the fruit on every axis.
- The rescaling is uniform: factor of each short axis over factor of the long axis 0.993 ±
  0.005 (range 0.980 to 1.012; one voxel of rounding on about 55 voxels is 0.018).
- Shrinking the original by s and centring it reproduces Hugo's volume: voxel correlation
  0.985 to 0.992 (2526_A01, C23, K07, O19; scratch figure, open boxes only).

**Rule.** Stretch factor s = 128 / original length in voxels (stalk included). One 128³ voxel
is 0.6267 / s mm. Example: 2526_A01, length 243 voxels (152 mm), s = 0.527, 128³ voxel 1.19 mm.

**Found** (open 2526 fruit after the faulty volumes, 315; healthy = defective 0, 145):

| | Healthy (145) | All (315) |
|---|---|---|
| Stretch factor s | 0.476 to 0.848 (median 0.601) | 0.457 to 0.848 |
| Total length, stalk included | 95 to 169 mm | 95 to 175 mm |
| Real body volume | 85 to 252 cm³ (median 158) | 85 to 380 cm³ (median 181) |
| Stalk length | 0 to 55 mm | 0 to 55 mm |
| log real body volume vs log s (Pearson) | −0.52 | −0.54 |
| 128³ body voxels vs log real body volume | +0.13 | +0.20 |
| 128³ body voxels vs log s | +0.76 | +0.71 |
| Stalk length (mm) vs log s | −0.78 | −0.65 |

**Reading.** The "size" of check C so far (body voxels in the 128³ grid) is mostly the stretch
factor (r +0.76 on healthy fruit) and hardly real size (+0.13). Real body volume and the
stretch factor share about a quarter of their variance (r −0.52), so their effects on a score
can be separated by a regression on both (step 4 of the plan of 27 November).

## 21. Which rows follow the stretch, which follow real size (27 November 2026, Ruben)

**What** (`pearlatent.stretch_check`, `nuisance.healthy_stretch_correlations`, run
`results/20261127T131002Z_stretch_check`). No model fitted: the saved out-of-fold scores of the
eight `binary_23` PLS-DA rows. On the 145 healthy fruit, score minus box median (as check C),
Spearman partial correlation with the stretch factor given real body volume, and with real body
volume given the stretch factor (ranks; the straight-line part of the other measure removed
from both). 95 % box-bootstrap intervals. Why partial: the two measures correlate −0.52
(section 20), so a plain correlation with one also picks up the other.

**Found:**

| Row | Stretch, given volume | Real volume, given stretch |
|---|---|---|
| Latent full grid | +0.08 (−0.20 to +0.33) | +0.21 (+0.05 to +0.33) |
| CT 32³ full grid | −0.00 (−0.15 to +0.26) | +0.07 (−0.03 to +0.19) |
| Tissue features | −0.07 (−0.23 to +0.16) | +0.05 (−0.13 to +0.24) |
| Med3D | −0.12 (−0.30 to +0.12) | +0.00 (−0.10 to +0.16) |
| Latent height-radius | +0.23 (+0.07 to +0.45) | +0.40 (+0.26 to +0.57) |
| CT height-radius | −0.04 (−0.18 to +0.14) | +0.25 (+0.15 to +0.37) |
| Latent height-radius mean + SD | +0.18 (+0.00 to +0.42) | +0.39 (+0.23 to +0.56) |
| CT height-radius mean + SD | −0.05 (−0.17 to +0.09) | +0.19 (+0.10 to +0.30) |

Without a model, all 315 open fruit: real body volume alone separates `binary_23` with AUC
0.787 (0.722 to 0.834), within-box AUC 0.752; the stretch factor alone 0.401 (0.311 to 0.510).
Median real body volume per defective grade 0 / 1 / 2 / 3: 158 / 181 / 203 / 213 cm³ (145, 41,
52, 77 fruit). Per box, median volume and share of non-consumable fruit: Spearman +0.84 over
the 12 boxes.

**Reading.** Larger fruit are more often non-consumable in 2526, within boxes as well as
between them (Ho et al. 2008 report the same direction). On healthy fruit the latent
height-radius scores follow real size more than the stretch (+0.39 against +0.18); the CT
height-radius rows follow real size only; the full-grid CT row, tissue features and Med3D
follow neither. The stretch reaches only the latent rows, and less than real size does. The
earlier check C "size" (128³ body voxels) was mostly the stretch factor (section 20), so check C
had mixed the two. How a score sees real size when the 128³ body hardly shows it (r +0.13) is
open. Decision for the preprocessing still open (Ruben).

## 22. AUC within thirds of real size; the stalk (27 November 2026, Ruben)

**What.** Run `results/20261127T133436Z_stretch_check` (supersedes 20261127T131002Z, same
numbers for sections 21 plus this table). All 315 open fruit split into thirds of real body
volume, cut-offs 160 and 203 cm³ (fixed before looking, `nuisance.size_thirds`); AUC of each
row's out-of-fold score inside each third, size alone as control. 105 fruit per third, 17 / 39 /
73 non-consumable.

| Row | Third 1 (≤ 160 cm³) | Third 2 (160 to 203) | Third 3 (> 203) |
|---|---|---|---|
| Size alone (control) | 0.58 (0.41 to 0.78) | 0.66 (0.55 to 0.76) | 0.71 (0.58 to 0.81) |
| Latent height-radius mean + SD | 0.98 (0.94 to 1.00) | 0.91 (0.85 to 0.95) | 0.96 (0.93 to 0.99) |
| Med3D | 0.96 (0.88 to 1.00) | 0.93 (0.87 to 0.97) | 0.96 (0.93 to 0.98) |
| Latent full grid | 0.87 | 0.84 | 0.89 |
| CT 32³ full grid | 0.79 | 0.77 | 0.86 |
| CT height-radius mean + SD | 0.91 | 0.90 | 0.92 |

The control does not fall to 0.5 (the top third spans 203 to 380 cm³), so some size is left
inside each third; every row stays far above it.

**Stalk** (scratch, saved scores and the real-size run; healthy 145 unless stated). Stalk length
in mm against stretch factor −0.77, against real body volume −0.00: the stalk is the part of the
stretch that is not fruit size. Latent height-radius mean + SD score: with stalk length −0.24,
given real volume −0.26; with the stretch factor given real volume +0.18. Without a model, all
315: shorter stalk goes with non-consumable, AUC 0.604, within box 0.605; given real volume the
partial Spearman with `binary_23` is −0.15 (real volume given stalk +0.48).

**Reading.** The model scores stay high among fruit of similar size: the latent score is not a
size measure. The stretch the latent score follows is mostly the stalk, which is independent of
fruit size. A weak link of real stalk length with defects exists on its own (AUC 0.60).

## 23. EPO on the measured stretch factor, latent rows (27 November 2026, Ruben)

**Rule** (Ruben, 27 November, after sections 21 and 22). The stretch to 128 voxels is an
artefact of the preprocessing and is removed; real fruit size is biology and stays. Only the
latent rows get it: the other rows' scores do not follow the stretch (section 21).

**What** (`pearlatent.epo.fit_stretch_direction`, `cross_validated_predictions(...,
epo_stretch=...)`, `pearlatent.epo_stretch_run`). In every fold, on the healthy fruit of the
training boxes: each feature regressed on the standardised stretch factor and the standardised
real body volume together (with an intercept). The stretch coefficients, one per feature, form
one vector: the change of the feature per SD of stretch at equal real size. Scaled to length 1,
it is removed from the training and held-out fruit of that fold, X* = X − (X p) pᵀ. Healthy
fruit only, because real size goes with the defect. Same fruit (315), folds, centring and LVs
as the rows without EPO. Example: if features 1 to 10 rise with s and features 6 to 20 with
real volume, p points along features 1 to 10; the volume pattern loses only its part along p
(features 6 to 10), about 9 % of its length in the test example (`tests/test_epo.py`).
Difference with section 17 (variant B, healthy PC 1): there the direction is found by a PCA,
here it is defined by a measured value. In the features they nearly coincide (next paragraph).

**What it removes** (scratch, latent height-radius mean + SD, one fit on all 145 healthy fruit,
27 November). One direction p out of 320 features, so one number per fruit, t = x · p, is set
to the same value for every fruit. t follows the stretch (+0.55), the stalk (−0.72) and also
real volume (+0.34) on the healthy fruit. That one direction holds 31.9 % of the variation of
the healthy fruit; it is almost healthy PC 1 (32.7 %, cosine 0.98). 94 % of p lies on the 160
mean features, spread about evenly over the 20 heights (4 to 8 % each): mostly the overall
level of the latent of a fruit. Example, feature 0 (height bin 0, ring 0, mean): −2.67 per SD
of stretch and −2.44 per SD of real volume. So the earlier check C "size" (128³ body voxels,
mostly stretch, section 20) was already what variant B removed; the two variants are expected
to behave alike, and variant B's results (section 17) should be read again with the measured
sizes before any claim that one is better.

**Found** (runs in `EPO_STRETCH_RUN_DIRS`; healthy 145, partial Spearman, box-bootstrap
intervals; before → after):

| Row (LVs) | AUC | Paired | Stretch given volume | Real volume given stretch | Stalk given volume |
|---|---|---|---|---|---|
| Latent grid (4) | 0.887 → 0.888 | +0.001 (−0.019 to +0.026) | +0.08 → +0.05 | +0.21 → +0.26 | −0.17 → −0.12 |
| Latent height-radius (3) | 0.927 → 0.921 | −0.006 (−0.021 to +0.007) | +0.23 → +0.08 | +0.40 → +0.38 | −0.28 → −0.10 |
| Latent height-radius mean + SD (3) | 0.957 → 0.957 | +0.001 (−0.012 to +0.010) | +0.18 → +0.01 (−0.15 to +0.18) | +0.39 → +0.34 (+0.18 to +0.52) | −0.26 → −0.02 (−0.19 to +0.12) |

RMSECV with EPO is lowest at 2 LVs for both height-radius rows (mean + SD: 0.349, 0.313, 0.318,
0.332 for 1 to 4 LVs), at 3 for the grid. The LVs were kept from the rows without EPO; the
choice for the final rows is Ruben's (by eye, before the test boxes).

**Reading.** The stretch and stalk link of the latent scores is gone (intervals include 0),
the real-size link stays, the AUC does not change. Proposed: this replaces variant B (section
17) for the final latent rows and the counterfactual direction; to be confirmed by Ruben.

**LVs of the final latent rows with EPO** (Ruben, 27 November 2026, by eye at the elbow of the
mean RMSECV curves with EPO, notebook 23 figure 7, before any test box is opened): grid 3,
height-radius 2, height-radius mean + SD 2 (`epo_stretch_run.FINAL_N_LV`). Runs in
`EPO_STRETCH_RUN_DIRS` (the runs at the old LVs, 134101Z to 134132Z, are kept for the record).
Healthy 145, partial Spearman, before (rows without EPO, their own LVs) → after:

| Row (LVs) | AUC | Paired | Stretch given volume | Real volume given stretch | Stalk given volume |
|---|---|---|---|---|---|
| Latent grid (3) | 0.887 → 0.889 | +0.002 (−0.005 to +0.009) | +0.08 → +0.03 (−0.24 to +0.28) | +0.21 → +0.17 (+0.00 to +0.29) | −0.17 → −0.11 (−0.37 to +0.12) |
| Latent height-radius (2) | 0.927 → 0.924 | −0.002 (−0.015 to +0.008) | +0.23 → −0.01 (−0.17 to +0.21) | +0.40 → +0.30 (+0.17 to +0.45) | −0.28 → −0.09 (−0.29 to +0.08) |
| Latent height-radius mean + SD (2) | 0.957 → 0.958 | +0.001 (−0.007 to +0.011) | +0.18 → −0.02 (−0.21 to +0.21) | +0.39 → +0.34 (+0.21 to +0.50) | −0.26 → −0.10 (−0.33 to +0.09) |
