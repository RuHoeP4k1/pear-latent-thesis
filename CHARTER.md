# Thesis charter — latent space for pear internal quality

Ruben Hoeven · KU Leuven, Bioscience Engineering · supervisor Hugo (Junyan) Li, MeBioS
One page. If something here needs more than a line, it belongs in the proposal document, not here.

> **Version of 10 October 2026.** Updated from the version of 6 October: N-way partial least squares
> added as one model row; cylindrical bins dropped. Proposal for the meeting of 15 October; items
> marked *(to settle with Hugo)* or *(to settle)* are proposals, not decisions.

## Title

Latent space is all you need? Detecting internal browning in pear CT across severity grades

## The question

Does the latent space of pretrained models retain sufficient information to detect internal
browning in pear CT, and at which browning grade does that stop?

1. Is browning kept? Reconstruction residual in the core, by grade (Step 0). Nothing fitted.
2. Is it usable? Simple statistical models on the latent, compared with CT downsampled to the
   same size.
3. Where does it stop, and why? Per-grade detection at four levels — 128³ input, CT averaged to
   32³, Hugo's latent, random encoder — so a loss can be placed in the resolution or the encoder.

"The information is in the latent" is a hypothesis, not a fact: the 0.9915 reconstruction score is
averaged over the whole volume, and a grade 2 lesion is about 0.2 % of it.

## What is fixed

**Object of study.** μ, the 32 × 32 × 32 latent of Hugo Li's adversarial 3D variational
autoencoder (SSRN 6433761), used frozen. One value per cell.

**Data.** 2526: 433 scanned fruit of 450 labelled, 15 boxes = orchards, never seen by the encoder:
primary. 2425: 660 fruit of the 2024 harvest, encoder trained on about 528 of them; browning equals
storage group there, so secondary only. Browning and cavity 0–3; rot 0–1 in 2425 only.

**Constraints.** Volumes 128³; every fruit aligned on the stem–calyx axis and rescaled to 128 voxels
in length, so millimetre figures are approximate and differ per fruit. Latent cell about 2.5 mm,
receptive field about 19 mm. Most 2526 volumes include the stalk; 2425 volumes do not *(to settle
with Hugo)*.

**Prediction, stated before looking.** Browning grade 3 detectable, grade 2 weak, grade 1 not;
mild cavity easier than mild browning.

## Comparisons (one external model, no architecture study)

128³ input features · CT averaged to 32³ · Hugo's latent · random frozen encoder · Med3D (frozen) ·
Hugo's `train_clf_3d.py` from scratch, rerun with box splits · own self-supervised extension.

## Evaluation rules, fixed before any result is seen

1. Split by box, never by fruit, for every model fitted on labels, including setting choices.
2. Primary fitting within 2526. 2425 only for comparison with Hugo, the reverse check (fit 2526,
   test 2024, state that the encoder saw about 80 % of them) and the healthy reference.
3. Final test: 3 boxes held out, drawn with a recorded seed among the combinations whose label
   shares lie within ±30 % of the population; sealed until the pipeline is frozen *(to settle with
   Hugo; fallback: cross-validation only with this plan committed)*. Exploration and plots exclude
   the test boxes.
4. Intervals by bootstrap over whole boxes, paired on the same out-of-fold predictions.
5. Browning primary; cavity separate, never merged; `defective` never the target.
6. Report per grade: recall per grade, a severity score fitted on grade 0 vs 3 only, binary_23 with
   binary_123 beside it. Two curves: any defect, and cavity-free browning.
7. Recall and precision beside accuracy.
8. Orchard check: mixed model with a random intercept per box and the score split into box mean
   and within-box deviation; report the intraclass correlation.
9. Normalisation from training folds only, without reference to class.
10. Exclusions: the 17 unscanned 2526 fruit; rotten fruit once identified. Until then, grade 3
    reported with and without the 24 fruit with browning 3 and cavity 3.

## Methods

Summaries of the grid: the full 32 × 32 × 32 array; deviation-from-healthy map with persistence
summaries (high percentile as comparison). Cylindrical bins (height × radius) were dropped on
10 October 2026. Reference coordinates of the healthy fruit for the deviation map *(to settle: the
earlier reference used height and radius; the same cell (x, y, z) is affected by the free rotation
around the stem–calyx axis)*.

Models, each on the unfolded array: PLS-DA, ridge or elastic-net logistic, cumulative-link (mixed)
models. Plus one extra model row, run after the core study: N-way partial least squares (Bro 1996)
on the four-way array fruit × x × y × z, first at the 32³ levels. Each component has one score per
fruit and three weight vectors (x, y, z); the template is their product. Compared with unfolded
partial least squares on the same array, with the same box splits; number of components chosen by
cross-validation over whole boxes inside the training boxes. Two checks, because the alignment
fixes only the stem–calyx axis: a fit on a core crop around that axis, and a refit after random
rotation of the test fruit around it. Prediction, stated before looking *(proposal)*: no gain over
unfolded partial least squares on grade 3; the added value is the three weight profiles.

No attention pooling, no 512-channel layer.

## Extensions

Own self-supervised extension (mid November–December, decision 1 December): a head on Hugo's
frozen encoder, trained without labels, one vector per fruit *(to settle with Hugo)*.
Counterfactual images (January–February): only if Step 0 shows browned cores are reconstructed;
direction from full-grid PLS, cavity-free core browning only.

## Do not

- Train a new architecture from scratch (Hugo's existing scripts may be rerun).
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

15 October 2026 proposal · core study to mid November · all experiments by 18 December 2026 ·
January–February 2027 counterfactuals and consolidation · February mid-term · May 2027 defence.
