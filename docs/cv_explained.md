# How the splits, the cross-validation and the bootstrap work

Reference for Ruben, 13 October 2026. Numbers are for the 2025–26 data (2526): 15 boxes
(= orchards), 433 scanned fruit, about 29 per box. Items marked *(proposal)* are not decided yet.

## 1. Why we split at all

Every choice made while developing (which features, which model, how many components, which
threshold) is tuned on the fruit we look at. A score measured on those same fruit is too
optimistic. So some data must be used only to measure, never to choose.

Why by box and never by fruit: fruit of one orchard resemble each other (shape, size, stalk,
storage history). If fruit of one box are on both sides of a split, a model can score well by
recognising the orchard instead of the browning. Keeping whole boxes together forces the model to
work on orchards it has never seen.

## 2. Step 1 — the sealed final test

1. List all 455 ways to choose 3 of the 15 boxes.
2. A set is eligible when its counts of non-consumable fruit, browning ≥ 2 without cavity,
   browning 3, and any cavity each lie within ±30 % of what is expected for those 3 boxes
   (expected = season total × fruit in the set / 433).
3. Draw one eligible set at random, seed 369. About 87 fruit.
4. These 3 boxes, and Hugo's 66 VAE test fruit of 2024–25, are put away until the very end.

Unusual orchards (box K almost all healthy, box O many grade 3) may never be in an eligible set,
so they can never be test boxes. They are still used in the cross-validation below.

## 3. Step 2 — outer cross-validation on the other 12 boxes

The 12 boxes are divided into 4 folds of 3 boxes. Then 4 rounds:

| Round | Fit the model on | Predict |
|---|---|---|
| 1 | folds 2, 3, 4 (9 boxes, about 260 fruit) | fold 1 (3 boxes, about 87 fruit) |
| 2 | folds 1, 3, 4 | fold 2 |
| 3 | folds 1, 2, 4 | fold 3 |
| 4 | folds 1, 2, 3 | fold 4 |

After the 4 rounds, every fruit of the 12 boxes has exactly one prediction, made by a model that
never saw its box. These are the out-of-fold predictions. The score (area under the curve, recall
per grade, and so on) is computed on all of them together, about 346 fruit.

## 4. The inner split — choosing settings without looking at the held-out fold

Most models have settings: the number of partial least squares components, the ridge penalty.
If the setting were chosen by how well it predicts the held-out fold, that fold would no longer be
unseen. So, inside each round, the 9 training boxes are split again:

1. For each candidate setting: fit on 8 of the 9 boxes, predict the 9th; repeat so each of the 9
   boxes is predicted once (leave one box out) *(proposal: leave one box out, because it needs no
   random grouping and uses every box)*.
2. Pick the setting with the best score over those inner predictions.
3. Refit on all 9 boxes with that setting; predict the held-out fold once.

This is called nested cross-validation. In chemometrics terms: in each round the 9 training
boxes are the calibration set and the 3 held-out boxes the validation set (their error is the
prediction error, RMSEP); the inner split is the internal cross-validation used to pick the number
of latent variables from the RMSECV curve. The RMSECV at the chosen setting is not reported as
performance, because it is the best of several tries. A by-product: if the inner split picks very
different settings in different rounds (3 components in one, 12 in another), the setting is
unstable, and that is worth reporting. The chosen setting may differ between rounds. That is
expected: what is evaluated is the whole procedure (including how it chooses its setting), not one
fitted model.

## 5. Partitions and repeats — where the fold numbers come from

A partition is one way to divide the 12 boxes into 4 folds of 3. How many exist:

- Put the 12 boxes in a row in any order: 12! orders. Cut the row into four groups of 3.
- The order inside a group does not matter: divide by 3! for each of the 4 groups.
- The order of the 4 groups does not matter: divide by 4!.
- 12! / ((3!)⁴ × 4!) = 479,001,600 / (1,296 × 24) = 15,400 partitions.

Not every partition is useful. If one fold gets box K and two other healthy boxes, that fold has
almost no defects and its round says little. So a partition is eligible when each of its 4 folds
passes the same ±30 % check as the test set (if too few pass: ±40 %, then ±50 %).

The score depends somewhat on which boxes happen to share a fold. To see how much, the whole outer
cross-validation is repeated with 20 different eligible partitions, drawn at random with seed 369
(5 drawn on 13 October, 15 added on 21 October; `derived/outer_folds_20/`, rule section 6.4
item 5). This is repeated double cross-validation (Filzmoser, Liebmann and Varmuza 2009,
*Journal of Chemometrics* 23, 160–171): an inner loop chooses the settings, an outer loop
estimates the prediction error, and the whole is repeated over different splits, because a
single split gives one draw of an estimate that depends on the split. Krstajic et al. (2014,
*Journal of Cheminformatics* 6:10) list reporting a single nested cross-validation as a pitfall
for the same reason.

- In every repeat, each box is predicted exactly once. No box counts more than another.
- Each fruit then has 20 out-of-fold predictions, one per repeat. Report the score of each repeat
  (to show the spread) and the score of the per-fruit average of the 20.
- What can recur across repeats is which boxes share a fold (balanced partitions may keep pairing
  the same boxes). The report shows how often each pair of boxes shares a fold. In the 20 drawn
  repeats, K and O share a fold 13 times and 11 of the 66 pairs never do.
- The spread across repeats is not a confidence interval: every repeat uses the same 12 boxes.
  The interval comes from the bootstrap over boxes (section 6).

Cost per model: 20 repeats × 4 rounds × (9 inner fits × number of candidate settings). Small for
partial least squares or ridge regression. The ResNet-18 gets one partition only, with one
validation box group taken from its training boxes for early stopping instead of the full inner
split.

## 6. The bootstrap over boxes — how certain is the score?

The score from 346 out-of-fold predictions would have come out differently with other orchards.
The bootstrap estimates by how much:

1. Draw 12 boxes from the 12, with replacement: some boxes are drawn twice, some not at all.
2. Take all fruit of the drawn boxes, with their existing out-of-fold predictions (a box drawn
   twice counts twice).
3. Compute the score.
4. Repeat 2,000 times. The middle 95 % of the 2,000 scores is the 95 % confidence interval.

No model is refitted in the bootstrap; only the predictions already made are resampled, so it is
fast.

Why boxes and not fruit: the independent units are the orchards, not the fruit. Resampling single
fruit would treat 346 fruit as independent and give an interval that is too narrow. With 12 boxes
the interval will be wide. That is the honest amount of certainty these data allow.

## 7. Comparing two methods — the paired bootstrap

Example: Hugo's latent against CT averaged to 32³. Both are evaluated on the same fruit with the
same folds. In each of the 2,000 bootstrap draws, compute both scores on the same drawn boxes and
take the difference. The middle 95 % of the 2,000 differences is the interval of the difference.
If it does not contain 0, the difference is larger than the variation between orchards.

Why paired: both methods find the same boxes easy or hard. Taking the difference inside each draw
removes that shared part, so the comparison is sharper than comparing two separate intervals.

## 8. At the very end — the sealed test, once

1. Freeze everything: features, model, how settings are chosen, exclusions, threshold.
2. Choose the settings by cross-validation over all 12 boxes, fit on all 12 boxes.
3. Predict the 3 sealed boxes and Hugo's 66 test fruit, once. Report the result, whatever it is,
   the two parts separately.
4. With only 3 boxes, a bootstrap over boxes is not possible. The interval resamples fruit and the
   report says so. The sealed test confirms or contradicts the cross-validation estimate; it does
   not replace it.

## 9. Where the 2024–25 fruit fit

- Hugo's 528 encoder-training fruit: not used, except one Step 0 comparison of reconstruction
  error (separate group, never pooled).
- Hugo's 66 validation fruit: on the training side. They may be used to fit a model, but a model
  fitted on them cannot be evaluated on Hugo's 66 test fruit: both groups come from the same ten
  orchards, which would be a split by fruit inside orchards.
- Hugo's 66 test fruit: sealed, evaluated once at the end with the model fitted on 2526.
