# Nuisance check — does a model detect the defect, or the orchard, stalk or size?

Written 26 October 2026. Status: design only, nothing built or run. To be built together with
the first PLS-DA run (binary_23, Hugo's latent), then applied unchanged to every model row.

## 1. Why

A model can score well without looking at the defect, if something easier to see goes together
with the defect in our fruit. Known candidates in 2526:

- **Orchard (box).** Boxes differ in how many defective fruit they contain. Notebook 02: box K
  has the smallest fruit bodies and the fewest defects.
- **Stalk.** Most 2526 volumes include the stalk, and it counts in the 128-voxel length, so the
  stalk changes how much of the grid the body fills. Notebook 07: PC1 follows stalk length
  (Spearman −0.78) and body size (0.97).
- **Size in the grid.** Size is a known cause of browning (Ho et al. 2008), so a size effect is
  not necessarily a shortcut, but it has to be shown separately from the defect.

Rules this note puts into practice: CLAUDE.md section 7 ("does it predict box or storage group
better than the defect?") and charter evaluation rule 8 (mixed model with a random intercept
per box, score split into box mean and within-box deviation, intraclass correlation, fruit
dimensions as covariates).

## 2. Input

Only the out-of-fold score of each fruit, so the same check works for every model row (PLS-DA
on Hugo's latent, random encoder, CT at 32³, CT at 128³, ResNet-18), whatever is inside the
model. Per fruit: `fruit_id`, `box`, out-of-fold score (the logit of the probability for the
ResNet, the PLS-DA prediction for PLS-DA), `defective`, cause (browning only, cavity only,
both), and the descriptors of `pearlatent.descriptors` (`body_voxels`, `stalk_length`,
`stalk_present`).

## 3. The checks

**A. Within box: orchard.** Compare non-consumable and consumable fruit of the same box only.
Measure: the within-box area under the ROC curve, the share of pairs (one non-consumable, one
consumable fruit, same box) in which the non-consumable fruit gets the higher score. A box with
only one of the two classes adds no pairs. Report it beside the ordinary area under the curve
over all fruit, with an interval from the bootstrap over boxes (charter rule 4).
Reading: if the overall value is high but the within-box value is near 0.5, the model ranks
orchards, not fruit.

**B. Mixed model (charter rule 8).** Outcome `binary_23`; the score split into the box mean of
the score and each fruit's deviation from that mean; random intercept per box (statsmodels).
Reading: if the defect follows the within-box deviation, the score works inside an orchard; if
it follows only the box mean, it describes the orchard. Also report the intraclass correlation
of the score: the share of its variation that lies between boxes once the grade is accounted
for. Reading of rule 8 (Ruben, 26 October 2026): "score split into box mean and within-box
deviation" means the two parts are the predictors of the defect, not the outcome.

**C. Healthy fruit only: stalk and size.** Only fruit with `defective` = 0. All of them should
get a low score. Spearman correlation of the score with `stalk_length` and with
`body_voxels`, and the score compared between `stalk_present` true and false; computed within
box (the score minus its box median), so that box K's small fruit do not produce a correlation
on their own. Interval from the bootstrap over boxes.
Reading: if healthy fruit with a long stalk, or small healthy fruit, score systematically
higher, the model uses stalk or size.

**D. Descriptors as covariates.** Model B again with `body_voxels` and `stalk_length` added.
Reading: if the within-box effect of the score stays about the same, it is not explained by
size or stalk.

All four are reported per row of the comparison table, split by cause where the counts allow.

## 4. What is not done, and why

- **Predicting box from the features** (the literal wording in CLAUDE.md section 7). For the
  latent rows this is possible because the features are fixed: a classifier for box, with
  fruit split within each box (box cannot be held out when box is the target). For the
  ResNet-18 the features change with every fold, so for that row only checks A to D apply. To
  decide whether the feature version is run for the latent rows at all, or whether A to D are
  enough everywhere.
- **2425.** Primary fitting is in 2526. For the reverse check on the 2425 validation fruit,
  storage group takes the place of box in check A.

## 5. Open points

- Reading thresholds (what counts as "near 0.5", "about the same") are not fixed yet. The
  charter's evaluation rules are fixed before any result is seen, so these have to be set
  before the first PLS-DA run.
- Check C depends on question 33 (2526 reprocessed without the stalk, or kept). If the volumes
  are reprocessed, the stalk part of C drops out and only size remains.
- Code: one function in `src/pearlatent/` with a test in `tests/`, used by every model
  notebook. Output per model row in `results/<run_id>/`.
