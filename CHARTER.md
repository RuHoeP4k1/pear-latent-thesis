# Thesis charter — latent space for pear internal quality

Ruben Hoeven · KU Leuven, Bioscience Engineering · supervisor Hugo (Junyan) Li, MeBioS
One page. If something here needs more than a line, it belongs in the proposal document, not here.

> **Version of 13 October 2026**, after the meeting with Hugo on 13 October. Changed from the
> version of 10 October: target order (consumable first), ResNet-18 baseline added, own
> self-supervised extension dropped, test-box rule (random draw among sets within ±30 %, seed
> 369), Hugo's 2425 training fruit excluded and his test fruit added to the final test, stalks and rot settled, fruit dimensions
> added. Items marked *(to settle with Hugo)* or *(to settle)* are still open.

## Title

Latent space is all you need? Detecting internal browning in pear CT across severity grades

## The question

Does the latent space of a pretrained model retain sufficient information to detect internal
disorders in pear CT, and at which browning grade does that stop?

1. Is browning kept? Reconstruction residual in the core, by grade, with visual comparisons
   (Step 0). Nothing fitted. The decoder is in hand (13 October); Step 0 is the next step.
2. Is it usable? First consumable versus non-consumable; then per grade; then browning and cavity
   separately, including whether the two can be told apart.
3. Where does it stop, and why? Per-grade detection at four fixed-feature levels — 128³ input,
   CT averaged to 32³, Hugo's latent, random encoder — beside a ResNet-18 trained from scratch on
   the 128³ CT. 128³ versus 32³ places a loss in the resolution; the latent versus the random
   encoder separates the training from the architecture; the ResNet-18 shows what supervised
   training on the same fruit reaches.

"The information is in the latent" is a hypothesis, not a fact: the 0.9915 reconstruction score is
averaged over the whole volume, and a grade 2 lesion is about 0.2 % of it.

## What is fixed

**Object of study.** μ, the 32 × 32 × 32 latent of Hugo Li's adversarial 3D variational
autoencoder (SSRN 6433761), used frozen. One value per cell.

**Data.** 2526: 433 scanned fruit of 450 labelled, 15 boxes = orchards, never seen by the encoder:
primary. 2425: 660 fruit of the 2024 harvest from ten orchards (preprint section 2.1), split by
Hugo at random by fruit into 528 encoder-training, 66 validation and 66 test fruit (seed 369;
reproduced from his code). The 528 training fruit are excluded from all analysis except one Step 0
comparison; browning equals
storage group in 2425, so it is secondary only. Browning and cavity 0–3; rot 0–1 in 2425 only.

**Constraints.** Volumes 128³; every fruit aligned on the stem–calyx axis and rescaled to 128 voxels
in length, so millimetre figures are approximate and differ per fruit, and absolute fruit size is
not in the volumes. Latent cell about 2.5 mm, receptive field about 19 mm. Most 2526 volumes
include the stalk, 2425 volumes do not; Hugo (13 October): not a problem, volumes used as they
are. The effect of the stalk on the reconstruction is checked in Step 0.

**Prediction, stated before looking.** Browning grade 3 detectable, grade 2 weak, grade 1 not;
mild cavity easier than mild browning; non-consumable detected well, mostly through cavity.

## Comparisons (one external model, no architecture study)

128³ input features · CT averaged to 32³ · Hugo's latent · random frozen encoder · Med3D (frozen) ·
ResNet-18 from scratch on 128³ CT: the 3D ResNet of Hugo's `train_clf_3d.py` (basic blocks 2-2-2-2), trained on our box
folds on the lab GPU; width *(to settle with Hugo: his default of 16 base channels, or 64)*.
Dropped on 13 October: the own self-supervised extension (GPU too small for 3D self-supervised
training). The random frozen encoder stays as a cheap extra row (code exists).

## Evaluation rules, fixed before any result is seen

1. Split by box, never by fruit, for every model fitted on labels, including setting choices.
2. Primary fitting within 2526. In 2425 (Ruben, 13 October): Hugo's 528 encoder-training fruit
   are excluded from all analysis, because the encoder has seen them, with one exception: in
   Step 0 their reconstruction error is compared with that of unseen fruit, as a separate group,
   never pooled. The 66 validation fruit are on the training side (they steered Hugo's model
   selection, not his training): usable for exploration, and for fitting wherever a model is
   fitted on 2425; a model fitted on them is not evaluated on Hugo's test fruit, which come from
   the same ten orchards. The 66 test fruit are sealed in the final test. Reverse check (fit 2526,
   evaluate 2425) on the validation fruit during development, on the test fruit once at the end. Healthy reference: grade 0 fruit of
   the 2526 training boxes, plus healthy 2425 validation fruit.
3. Final test, two sealed parts reported separately: (a) 3 of the 15 2526 boxes; (b) Hugo's 66
   VAE test fruit of 2425, kept together as one batch and reported per storage group (orchards
   cannot be separated: Hugo split by fruit, so every orchard is also in his training set).
   Choosing (a) (principle agreed with Hugo, 13 October): list all 455 sets of 3 boxes; a set is
   eligible when each balance count lies within ±30 % of its expected count for that set
   (expected = season total × scanned fruit in the set / 433); draw one eligible set at random
   with seed 369. Balance counts: non-consumable, browning ≥ 2 without cavity, browning 3, any
   cavity. If fewer than 10 sets are eligible, widen
   to ±40 %, then ±50 %. The rule is committed to git before the draw; the number of eligible
   sets is reported. The restriction keeps unusual orchards (such as box K or O) out of the test
   set; they are covered by the cross-validation. Outer folds: the 15,400 ways to split the other
   12 boxes into 4 folds of 3; a partition is eligible when every fold passes the same tolerance (same fallback);
   partitions for repeated cross-validation are drawn at random from the eligible ones. Every box
   is in exactly one test fold per partition. Sealed until the pipeline is frozen; exploration and
   plots exclude both test parts.
4. Intervals by bootstrap over whole boxes, paired on the same out-of-fold predictions.
5. Target order (agreed with Hugo, 13 October): first `defective` (the maximum of browning and
   cavity): consumable versus non-consumable (binary_23, defective ≥ 2), with recall per defective
   grade; then browning and cavity separately, and whether the two can be separated. Results on
   `defective` are always split by cause (browning only, cavity only, both), because cavity is the
   easier signal and can carry the score on its own.
6. Report per grade: recall per grade, a severity score fitted on grade 0 vs 3 only, binary_23 with
   binary_123 beside it. Two curves: any defect, and cavity-free browning.
7. Recall and precision beside accuracy.
8. Orchard check: mixed model with a random intercept per box and the score split into box mean
   and within-box deviation; report the intraclass correlation. Fruit dimensions enter this check
   as covariates.
9. Normalisation from training folds only, without reference to class.
10. Exclusions: the 17 unscanned 2526 fruit. Rot: Hugo (13 October) says rot is not really a
    problem in 2526 (storage conditions less extreme), so the 24 fruit with browning 3 and cavity 3
    stay in; one sensitivity row without them.

## Methods

Summaries of the grid: the full 32 × 32 × 32 array; deviation-from-healthy map with persistence
summaries (high percentile as comparison). Reference coordinates of the healthy fruit for the
deviation map *(to settle: the same cell (x, y, z) is affected by the free rotation around the
stem–calyx axis)*.

Models, each on the unfolded array: PLS-DA, ridge or elastic-net logistic, cumulative-link (mixed)
models. Plus one extra model row after the core study: N-way partial least squares (Bro 1996) on
fruit × x × y × z, first at the 32³ levels, compared with unfolded partial least squares on the same
box splits, with a core-crop check and a random-rotation check. Prediction *(proposal)*: no gain on
grade 3; the added value is the three weight profiles.

ResNet-18 baseline: same outer folds; early stopping on a validation box group taken from the
training boxes; out-of-fold probabilities saved for the paired comparison. Target binary_23 first;
cavity-free browning ≥ 2 second if GPU time allows. Prediction *(proposal)*: trained on about 320
fruit per fold, it does not beat the latent on grade 3; no prediction for grade 2.

Fruit dimensions: height, major and minor diameter and volume in millimetres. Not in the 128³
volumes, because each fruit was rescaled to 128 voxels in length. Two routes *(to settle with
Hugo)*: the length of each fruit before rescaling, from his preprocessing (if the rescaling was the
same along all three axes, every dimension follows from the 128³ volume times length / 128); or
the original scans. Extra labels from Hugo when they arrive. Used as responses and colouring in PLS
score plots and as covariates in the orchard check.

No attention pooling, no 512-channel layer.

## Extensions

Counterfactual images (Hugo positive, 13 October): the gate — are browned cores reconstructed —
is part of Step 0 and runs in October now that the decoder is here; the work itself
January–February. Direction from full-grid PLS, cavity-free core browning only; severity
calibrated on the distance to the decision boundary as in Keicher et al. 2023 / Atad et al. 2024.

## Do not

- Train a new architecture from scratch. One exception (Hugo, 13 October): the ResNet-18
  baseline, with the architecture of Hugo's `train_clf_3d.py`.
- Train a self-supervised 3D model.
- Use a heatmap or the weight profiles of a model as evidence of what it uses.
- Draw a quantitative conclusion from a 2D embedding.
- Report a per-cell map at a resolution the receptive field cannot support.
- Compare many architectures.

## Source hierarchy

1. Hugo's preprint, supplement, repository README, label CSVs, rubric, Hugo's own answers.
2. Papers with a verified identifier.
3. The proposal document and the Notes tab.
4. Parked ideas, old notes, past chats: history only.

## Dates

13 October 2026 meeting with Hugo · core study to mid November · ResNet-18 baseline and N-way
partial least squares mid November–December · all experiments by 18 December 2026 ·
January–February 2027 counterfactuals and consolidation · February mid-term · May 2027 defence.
