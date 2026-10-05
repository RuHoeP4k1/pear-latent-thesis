# Thesis charter — latent space for pear internal quality

Ruben Hoeven · KU Leuven, Bioscience Engineering · supervisor Hugo (Junyan) Li, MeBioS
One page. If something here needs more than a line, it belongs in the Notes document, not here.

> **Status, 3 October 2026.** Rules 1 and 2 below were changed on 2 and 3 October (box rule;
> reversed evaluation design, because in the 2024 fruit browning and cavity occur together in
> 296 of the 300 stored defective fruit). Still under review: "pears from 2025 and 2026" — the
> 2526 file is probably one harvest (2025) stored into 2026; confirm with Hugo.
>
> **Status, 5 October 2026.** Updated after Hugo's answers and the Approved ideas tab: the
> reconstruction score no longer counts as evidence that browning is kept; the 2526 data
> described as 450 fruit (433 scanned), box letter = orchard ID; voxel size given as approximate
> (rescaling per fruit); rule 1 adds stratified group folds and a single box-effect measurement;
> rule 3 uses a bootstrap over boxes and paired differences; rule 7 adds raw CT pooled to 32³, a
> fine-tuning arm and, conditionally, Hugo's classifier; new rule 10 on excluded fruit.

## The question

How much of the information needed to detect internal browning is already present in the
latent space of a pretrained model, and under what conditions does it stop being enough?

Not "is the latent space enough" — the decoder reconstructs pears at 0.9915 structural
similarity, but that score is an average over the whole volume; a 1 cm³ lesion is about 0.2 % of it, so reconstruction quality alone does not show that browning is kept. The real question is whether it is
present in a form a simple model can use, and at which defect severity that fails.

## What is fixed

**The object of study.** The latent space of Hugo Li's adversarial 3D variational
autoencoder (SSRN 6433761), used frozen. We train no encoder.

**The data.** 660 'Conference' pears from the 2024 harvest, which that encoder was trained
on. 450 further pears in the 2526 file (433 scanned; probably one harvest, 2025, stored into 2026 — to confirm), which it has never seen. Browning and
cavity graded 0–3, rot 0–1, by a written rubric. A box letter is an orchard ID (Hugo, October 2026).

**The constraints that bound every result.** Every fruit is rescaled to 128 voxels in length, so millimetre figures are approximate and differ per fruit (0.6267 mm is the voxel size before rescaling, per the preprint). Volumes 128³. Latent 32³,
one scalar per cell, spacing about 2.5 mm, encoder receptive field about 19 mm. Browning
grade 1 is kernel membrane discoloration (sub-millimetre), grade 2 up to 1 cm³ in the core,
grade 3 above that or in the flesh. A 1 cm³ lesion is about 12 mm across — smaller than what
a single latent value summarises.

**The prediction these imply**, stated before looking: grade 3 detectable, grade 2 weak,
grade 1 not at all; mild cavity easier than mild browning because a void has far more
contrast than browned flesh.

## Evaluation rules, fixed before any result is seen

1. Split by box, never by fruit: all fruit of one box stay on the same side of every split, for
   every model fitted on labels (rule adopted 2 October 2026; replaces "split at fruit level").
   Folds by stratified group cross-validation. The box effect is measured once: one model with a
   within-box split and a box-held-out split.
2. Exploration and plots may use all fruit of both seasons, always labelled by season. Model
   fitting: the primary analysis is cross-validation within the 2526 fruit (never seen by the
   encoder), with whole boxes held out. The 2024 fruit are secondary: comparison with Hugo's
   numbers and a reverse cross-season check (fit on 2526, evaluate on 2024), stating that the
   encoder saw about 80 % of them. (Changed 3 October 2026; was "fit on 2024, evaluate on 2025–26".)
3. Confidence intervals by a bootstrap over whole boxes, not the spread across cross-validation
   folds. Method comparisons report the interval of the paired difference on the same
   out-of-fold predictions. Features are fixed in advance or selected inside the cross-validation.
4. Browning is the primary target. Cavity is reported separately as the easy control.
   Never merged.
5. `binary_23` primary, `binary_123` always reported beside it. The drop between them is a
   result. Where possible use the 0–3 grades directly, with an ordinal model.
6. Recall and precision alongside accuracy — the two errors cost different things.
7. Reference points, none requiring a trained network: randomly initialised frozen encoder
   (lower bound), the two-feature support vector machine of Van De Looverbosch 2020
   (90–95%, the practical bar), one external encoder (3DINO), published numbers as the
   upper bound. Added 3 October 2026: raw CT average-pooled to 32³, and one fine-tuning arm of
   Hugo's encoder (measures the cost of freezing). Conditional: Hugo's `train_clf_3d.py`, only if
   he re-runs it with our box splits.
8. Nuisance check on every representation: does it predict storage batch or orchard better
   than it predicts the defect?
9. Normalisation from training data only, computed without reference to class.
10. Excluded fruit: the 17 unscanned 2526 fruit (Hugo's advice). Rotten fruit were given the
    highest browning and cavity grades by hand; exclude them once Hugo identifies them, and until
    then report grade 3 with and without the fruit that have browning 3 and cavity 3.

## Method stance

Statistical first: PLS-DA, penalised regression, ordinal (cumulative-link) models for the
graded outcome, PCA for effective dimensionality. Pooling choices (mean, maximum, attention)
are kept because maximum pooling expresses a decision rule — "the worst region decides" —
that no linear model on the flattened grid can represent.

Explanations must be testable. Counterfactual generation and prototype placement counts,
not heatmaps.

## Do not

- Train anything from scratch.
- Use a heatmap as evidence of what the model uses.
- Draw a quantitative conclusion from a 2D embedding.
- Report a per-cell map at a resolution the 19 mm receptive field cannot support.
- Use the `defective` column as the primary target — it is the maximum of browning and
  cavity, and merges an easy task with a hard one.
- Compare many architectures. The strategy matters more and the comparison does not transfer.

## Source hierarchy

1. Hugo's preprint, its supplement, the repository README, the label CSVs, the rubric.
   Authoritative on our own data.
2. Papers with a verified identifier. Authoritative on the field.
3. The Notes and Approved ideas tabs. Our derived decisions.
4. Parked ideas, old notes, past chats. History, never cited as fact.

A claim about our data traces to tier 1 or is marked as inference. A claim about the
literature carries a verified identifier. Nothing is passed on from one paper's description
of another without checking the original.

## Dates

15 October 2026 proposal · all experiments by 18 December 2026 · February 2027 mid-term ·
April 2027 writing complete · May 2027 defence.
