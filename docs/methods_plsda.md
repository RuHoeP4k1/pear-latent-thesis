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
