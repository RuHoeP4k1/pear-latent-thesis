# Methods record: ResNet-18 baseline trained from scratch (runs 1 and 2)

Written 28 November 2026 for the rules fixed on 27 November, so the methods are not lost before
the thesis is written. The code is the reference: `src/pearlatent/resnet_baseline.py`, tests in
`tests/test_resnet_baseline.py`. Charter, Methods, holds the decisions in short.

**Status (28 November 2026): no results.** The code has run only as smoke runs on the Mac CPU
(1 epoch, a few fruit per box; section 11). Their numbers mean nothing and are not recorded
here. The full runs wait for the lab workstation.

## 1. Why this row

The one network trained from scratch in the thesis (Hugo, 13 October 2026; CLAUDE.md section 1).
The PLS-DA rows use fixed features (Hugo's latent, CT at 32³, tissue features, Med3D). The
ResNet-18 shows what a network reaches when it learns its own features from the 128³ CT of the
same fruit, with the same box folds. Charter prediction *(proposal)*: trained on about 185
fruit per fold, it does not beat the latent on grade 3; no prediction for grade 2.

Two runs (Ruben, 27 November 2026):

- **Run 1** (`--target binary`): trained on `binary_23` (defective ≥ 2), the same task as the
  PLS-DA rows, for the paired comparison. Grades afterwards from the ordinal step of
  notebook 12 on its output (section 8).
- **Run 2** (`--target grades`): trained on the `defective` grade 0 to 3 with a cumulative
  head (section 9). Question: can the network grade at all?

The two runs differ only in the label and the last layer; fruit, folds, validation boxes,
network, settings and seed are the same.

## 2. Fruit (fixed 27 November 2026)

- 315 open 2526 fruit: the 346 of the 12 open boxes minus the faulty volumes
  (`derived/volume_decisions.csv`, `docs/outlier_rule.md`), as for the PLS-DA rows. Grades:
  145 grade 0, 41 grade 1, 52 grade 2, 77 grade 3; 129 non-consumable (`binary_23` = 1).
- 2526 only, no 2425 fruit in the training (Ruben, 27 November 2026): the row stays paired with
  the PLS-DA rows, which were fitted on the same 315 fruit. Adding 2425 would make the ResNet
  the only row trained on two seasons, so a difference could come from the extra data.
- Test boxes F, J, M (87 fruit) are not loaded: they are not in the encoding manifest the fruit
  table starts from.

## 3. Folds (fixed 27 November 2026)

**Outer folds.** One partition only: repeat 1 of `derived/outer_folds_20/outer_folds.csv`,
4 folds of 3 boxes. PLS-DA gets 20 repeats (80 fits per number of LVs, under a second each);
one ResNet fit is a full network training, so 80 are too many (`docs/cv_explained.md`
section 5). Each fold trains a new network from random weights; nothing carries over between
folds. After the 4 folds every fruit has one out-of-fold prediction, from the network that
never saw its box.

**Validation boxes** (Ruben, 27 November 2026; question 66). Inside each fold, 2 of the 9
training boxes are set aside for early stopping, drawn once with seed 369 over the sorted box
names (`fold_plan`). The same 2 boxes serve every epoch of that fold. Rule checked in code
(`check_fold_plan`): every box is tested exactly once, and within a fold a box has one role.

| Fold | Test boxes | Validation boxes | Train / validation / test fruit |
|---|---|---|---|
| 1 | A B I | L N | 186 / 52 / 77 |
| 2 | C G N | D E | 188 / 50 / 77 |
| 3 | D E O | C H | 178 / 57 / 80 |
| 4 | H K L | D E | 184 / 50 / 81 |

D and E are the validation pair in folds 2 and 4 (the draw is random per fold); in fold 3,
where they are test boxes, they take no part in training or stopping.

**Why validation boxes and not the cross-validation curve** (Ruben chose the nested version).
The number of epochs plays the role the number of LVs plays for PLS-DA. Two ways were weighed:

- Validation boxes (chosen): the stopping choice never sees the held-out boxes, so the
  out-of-fold score is clean (nested). Cost: about 185 training fruit per fold instead of
  about 236.
- The design A way: train each fold on all 9 boxes for a fixed number of epochs and choose the
  epoch from the mean held-out loss over the 4 folds. Same scheme as the PLS-DA rows and more
  training fruit, but the choice sees the held-out boxes (slightly optimistic, as the LV
  choice is).

Both keep every box whole; the box rule does not decide between them.

**Training fruit per grade per fold** (training role only):

| Fold | Grade 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| 1 | 83 | 20 | 27 | 56 |
| 2 | 80 | 27 | 33 | 48 |
| 3 | 94 | 25 | 28 | 31 |
| 4 | 75 | 22 | 34 | 53 |

20 to 27 grade 1 fruit per fold: the grade ≥ 1 threshold of run 2 rests on few fruit.

**Difference with Hugo's split** (`train_clf_3d.py`, `split_train_val_test_by_batch`). Hugo
splits within each box: per box about 10 % test, and the rest 2 to 1 into training and
validation (for 30 fruit: 3 test, 9 validation, 18 training). Every box, so every orchard,
is then in training, validation and test. His network therefore trains on more fruit (about
260 of the 433 scanned fruit), but his test fruit come from orchards seen in training, and his test set is
about 45 fruit from one split. Ours: no test fruit from a training orchard, and all 315 fruit
predicted once. Expect our score to be lower than a score on his split; it is the one
comparable with the PLS-DA rows.

## 4. Input

Each 128³ volume min-max scaled to [0, 1] per volume (`pooling.scale_minmax`, the same as
Hugo's `utils/volumes.py` `normalize_minmax`), one channel. Read from disk at every use
(8 MB per volume as float32). No augmentation: Hugo's `--augment` (random flips, affine,
gamma; needs torchio) is off by default in his script. No pseudo-labels: his
`--unlabeled_dir` is optional and this design has no unlabeled fruit. No correction for the
stretch to 128 voxels (section 12).

## 5. Network

Hugo's `ResNet3D` from `train_clf_3d.py` at commit 183112f, imported from his clone
(`import_hugo_resnet`, which refuses another commit), not copied (no licence file). Basic
residual blocks 2-2-2-2: the layout of ResNet-18. For a 128³ input (shapes checked on
28 November 2026):

| Part | Output (channels × grid) |
|---|---|
| Stem: 3³ convolution stride 1, batch norm, ReLU, 3³ max pooling stride 2 | 16 × 64³ |
| layer1 (2 blocks, stride 1) | 16 × 64³ |
| layer2 (2 blocks, stride 2) | 32 × 32³ |
| layer3 (2 blocks, stride 2) | 64 × 16³ |
| layer4 (2 blocks, stride 2) | 128 × 8³ |
| Global average pooling, dropout 0.2 | 128 numbers per fruit |
| Last layer `fc` | run 1: 2 outputs; run 2: cumulative head (section 9) |

**Width** (Ruben, 27 November 2026): 16 base channels, Hugo's default; the channels double at
each stage (16, 32, 64, 128). 2,073,618 parameters (run 1); 2,073,491 (run 2, whose head has
128 + 3 parameters instead of 128 × 2 + 2). A standard ResNet-18 width (64, 128, 256, 512)
has 33,140,802: 16 times as many, for about 185 training fruit per fold, and about 4 times
the activation memory per volume.

Batch normalisation (not Hugo's VAE's instance normalisation): in evaluation mode it uses the
statistics stored during training, so a fruit's prediction does not depend on the other fruit
in its batch.

## 6. Training (Hugo's defaults unless stated)

| Setting | Value | Source |
|---|---|---|
| Optimiser | AdamW, learning rate 1e-4, weight decay 1e-4 | Hugo's defaults |
| Learning rate schedule | cosine from 1e-4 towards 0 over the maximum number of epochs | Hugo |
| Batch size | 16 | Hugo |
| Maximum epochs | 100 | Hugo |
| Early stopping | stop after 20 epochs without a lower validation loss | Hugo |
| First epoch that can be "best" | 11 | Hugo (see below) |
| Loss | run 1: cross-entropy on 2 classes (Hugo); run 2: section 9 | |
| Seed | 369, reset at the start of every fold | this project |
| Mixed precision | `--amp` on the GPU only | Hugo's option |

**One epoch** = every training fruit through the network once. The training fruit are
shuffled and cut into batches of 16; after each batch the weights are updated once. Example,
fold 1: 186 training fruit = 11 batches of 16 and one of 10, so 12 weight updates per epoch.
After the epoch the 52 validation fruit go forward through the network only (no update) and
their mean loss is written to `log_fold1.csv`, with the training loss, the validation AUC on
`binary_23` and the learning rate.

**Best model.** The weights of the epoch with the lowest validation loss are kept. Hugo starts
watching the validation loss only after epoch 10 (his `pseudo_start_epoch`, which applies also
without pseudo-labels); kept, so the best model is never from the first 10 epochs. Example: if
the validation loss is lowest at epoch 34 and no epoch from 35 to 54 is lower, training stops
after epoch 54 and the weights of epoch 34 predict the 3 test boxes. The model is not refitted
on all 9 boxes afterwards: the number of epochs found with 7 training boxes would not transfer
to 9.

**Seed.** `seed_everything(369)` at the start of every fold, so a fold's result does not depend
on which folds ran before it. On the GPU some operations are not bitwise repeatable; a rerun
can differ slightly.

## 7. Output of a run

In `results/<stamp>_resnet18_binary23/` (run 1) or `results/<stamp>_resnet18_grades/` (run 2),
outside git: `run.json` (all settings, git state, versions), `fold_plan.csv`,
`log_fold<k>.csv` per fold, `best_fold<k>.pth` (weights, never in git), and
`oof_predictions.parquet`: one row per fruit with box, grades, cause, outer fold, best epoch
and `probability` = the out-of-fold probability of `binary_23` = 1 (run 2: P(grade ≥ 2)).
Run 1 adds the files of section 8, run 2 those of section 9.

## 8. Run 1: ordinal step on the output (Ruben, 27 November 2026)

The network itself learns `binary_23` only. Grades come afterwards, exactly as for the PLS-DA
rows (notebook 12, `docs/methods_plsda.md` section 9): a cumulative logit model
(`pearlatent.ordinal`, `OrderedModel`, 4 parameters) on the one out-of-fold number per fruit.
For each outer fold it is fitted on the out-of-fold scores of the other 3 folds and predicts
this fold (`ordinal_step`). One partition, so no averaging over repeats.

Two scales of the score, to compare before one is chosen (question 67): the probability as
saved, and its log-odds log(p / (1 − p)). Reason: the cumulative logit model is linear in its
score; the PLS-DA prediction is roughly linear in the grade, but a network probability is
squeezed towards 0 and 1. Example: p = 0.5, 0.9, 0.99, 0.999 give log-odds 0, 2.20, 4.60,
6.91. On the probability scale the last three lie within 0.1 of each other; on the log-odds
scale they are about 2.3 apart each. Probabilities are clipped to [1e-6, 1 − 1e-6] before
the log-odds.

Files: `ordinal_predictions.parquet` (per fruit and scale: grade probabilities, assigned grade,
expected grade) and `ordinal_summary.csv` (section 10). Rerun on a copied run folder:
`--ordinal-for results/<run>`. Same limit as notebook 12: the score of a training fruit comes
from a network fitted partly on the held-out boxes; with 4 parameters the effect is small.

## 9. Run 2: the network trained on the grades (Ruben, 27 November 2026)

**Last layer.** Hugo's `fc` is replaced by a cumulative head (`cumulative_head`); the rest of
his network is unchanged (its forward pass ends in `self.fc(x)`). The head computes one score
per fruit from the 128 pooled numbers (a linear map without bias) and has 3 thresholds
t1 < t2 < t3. Its 3 outputs are log-odds:

    log-odds of (grade ≥ k) = score − t_k,   k = 1, 2, 3

so P(grade ≥ k) = 1 / (1 + exp(−(score − t_k))). It is the network form of the cumulative
logit model of notebook 12, where the score came from PLS-DA; here the network learns its own
score. Chosen over a 4-class output (Hugo's `output_size=4`), which treats the grades as
unordered: calling a grade 3 fruit grade 0 would count the same as calling it grade 2.

**Order of the thresholds.** Stored as t1 and two gaps: t2 = t1 + softplus(g1),
t3 = t2 + softplus(g2), softplus(g) = log(1 + exp(g)) > 0. So t1 < t2 < t3 always, and
P(≥ 1) ≥ P(≥ 2) ≥ P(≥ 3) for every fruit: no grade probability can be negative. Start:
t1 = 0 and g1 = g2 = 0, so thresholds 0, 0.693, 1.386; training moves them.

**Grade probabilities** are differences: P(0) = 1 − P(≥ 1), P(1) = P(≥ 1) − P(≥ 2),
P(2) = P(≥ 2) − P(≥ 3), P(3) = P(≥ 3).

Example (start thresholds, score 1.0): log-odds 1.0 − (0, 0.693, 1.386) = 1.000, 0.307,
−0.386; P(≥ 1, 2, 3) = 0.731, 0.576, 0.405; grade probabilities 0.269, 0.155, 0.172, 0.405
(sum 1). Assigned grade: the most probable, here 3. Expected grade: 0 × 0.269 + 1 × 0.155 +
2 × 0.172 + 3 × 0.405 = 1.71.

**Loss.** Each fruit gives 3 yes/no targets, one per threshold: grade ≥ 1, ≥ 2, ≥ 3. A grade 2
fruit has targets (1, 1, 0). The loss is the binary cross-entropy of each output against its
target, averaged over the 3 thresholds and the fruit (`cumulative_loss`). Example, the grade 2
fruit above: −log 0.731 = 0.313, −log 0.576 = 0.551, −log(1 − 0.405) = 0.519; mean 0.461. The
loss is lowest when P(≥ 1) and P(≥ 2) are near 1 and P(≥ 3) near 0. Early stopping uses this
loss on the validation boxes.

**Files.** `oof_predictions.parquet` holds per fruit `p_at_least_1` to `p_at_least_3`,
`p_grade_0` to `p_grade_3`, `assigned`, `expected_grade`, and `probability` = P(≥ 2), so its
`binary_23` AUC is read the same way as run 1's. `grade_summary.csv`: section 10.

## 10. What is reported

Written by the code now: the AUC per fold and pooled, and the grade metrics below. Still to
compute from the saved predictions, in a results notebook after the full runs: recall and
precision, the paired comparison and the bootstrap intervals.

- `binary_23`: ROC AUC of `probability` (run 1: P(binary_23 = 1); run 2: P(≥ 2)), with recall
  and precision beside it (CLAUDE.md section 7). Per fold and pooled over the 315 fruit.
- Grades, the metrics of notebook 12 (`pearlatent.ordinal.grade_summary`, the same function for
  run 1's ordinal step and run 2): error rate, error of always choosing the largest class,
  quadratic weighted kappa, Spearman correlation of the expected grade with the true grade,
  share within one grade, recall per grade. All fruit, and per cause with the grade 0 fruit
  beside each cause (charter rule 5).
- Paired comparison with the PLS-DA rows on the same fruit; interval by the bootstrap over
  boxes (`docs/cv_explained.md` section 6). Difference to state: a PLS-DA score is the mean of
  20 out-of-fold predictions; a ResNet score is one.

## 11. Runs

Smoke runs on the Mac CPU (1 epoch; numbers meaningless, only that the code runs end to end):

| Run folder (`results/`) | What |
|---|---|
| `20261127T150858Z_resnet18_binary23_smoke` | run 1, fold 1, 3 fruit per box, 18 s |
| `20261127T152800Z_resnet18_binary23_smoke` | run 1, 4 folds, 6 per box, ordinal step, 2 min |
| `20261127T163132Z_resnet18_grades_smoke` | run 2, 4 folds, 6 per box, 2 min |
| `20261127T163338Z_resnet18_binary23_smoke` | run 1 after the run 2 changes, fold 1 |

Full runs: none yet (workstation). Rough estimate from the smoke runs: about 10 hours for one
full run on the Mac CPU.

## 12. Limits and open points

- One partition: the score depends on which boxes share a fold; the spread across repeats that
  PLS-DA shows is not available.
- Stretch: the network sees the 128³ volumes with the stretch to 128 voxels built in; the
  stretch EPO of the latent rows (`docs/methods_plsda.md` section 23) has no counterpart here.
  Check afterwards with `pearlatent.stretch_check` on its healthy-fruit scores (question 69).
- Grade 1: 20 to 27 training fruit per fold (section 3).
- Sealed test boxes F, J, M: how the ResNet predicts them is not decided (the 4 fold networks,
  or one network trained on the 12 open boxes with validation boxes). To settle before the
  test (question 70).
- Ordinal step scale (probability or log-odds): question 67.

## Decision log

| Date | Decision | By |
|---|---|---|
| 13 Oct 2026 | ResNet-18 from scratch as the one trained network; architecture of Hugo's `train_clf_3d.py` | Hugo |
| 29 Oct 2026 | Width, epochs, batch size for Ruben to decide | Hugo |
| 27 Nov 2026 | Width 16 (Hugo's default) | Ruben |
| 27 Nov 2026 | 2526 fruit only, no 2425 | Ruben |
| 27 Nov 2026 | Run 1 on `binary_23`; ordinal step of notebook 12 on its output | Ruben |
| 27 Nov 2026 | Run 2 on the `defective` grades, cumulative head | Ruben |
| 27 Nov 2026 | Early stopping on 2 whole validation boxes per fold (nested), not on the cross-validation curve | Ruben |
