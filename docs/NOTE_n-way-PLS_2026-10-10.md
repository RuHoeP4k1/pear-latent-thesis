# Note — N-way partial least squares for the pear arrays (10 October 2026)

Status: a method note, not a decision record. Decisions are in `CHARTER.md`. Evidence labels: read =
read in full text by Claude in this project; abstract = abstract retrieved only; title = title and
metadata only.

## What the method does

Array X: fruit × x × y × z (433 × 32 × 32 × 32 at the latent level). Response y: browning grade.

- One component = one score vector t (one number per fruit) and one weight vector per spatial
  mode (x, y, z, 32 values each). The product of the three weight vectors is a 32 × 32 × 32
  template. The score of a fruit is the sum, over all cells, of the fruit's value times the template
  value.
- Weights are chosen so that the scores covary maximally with y. In the unfolded case the weight
  vector is proportional to Xᵀy (a free vector of 32,768 values). In the four-way case the same map
  is arranged as the array Z (Z_jkl = sum over fruit of y_i · x_ijkl); the template is its best
  rank-one approximation.
- Further components: subtract score × template from every fruit and repeat. The grade is regressed
  on the score matrix (433 × number of components).
- "Latent variable" here means the score vector t. Hugo's latent μ is the input array, not the
  partial least squares latent variables.

Claims by Bro (1996), read, pages 1–8 of 15: p. 48 parameter-count argument (each weight vector is
estimated from many more values than in the unfolded model); p. 49 whether a trilinear model is
appropriate is problem dependent, several models should be tried; p. 51 Eq. 9, singular value
decomposition of Z; p. 52 Table 2, algorithm; p. 53 extension to higher order, one-component
decomposition of Z; p. 54 for images the trilinear constraint can make interesting areas easier to
find even if the data are not truly trilinear. The summary reports comparable predictions for both
methods in the sugar example and a clear gain in interpretability; in a simulated noisy case
unfolded partial least squares was almost twice as bad. Both examples are fluorescence data, where
trilinearity has a physical basis; pear CT has none. The application section (pages 9–15) is not read.

## Where it does not fit without care

- The alignment fixes only the stem–calyx axis; rotation around it is free, so x and y positions of
  peripheral patterns differ between fruit. Checks in the charter: core crop, random rotation.
- The z axis behaves most like a spectrum (ordered, aligned by length normalisation), but there is no
  physical mixing of fixed profiles along it; the low-rank structure is a constraint, not a
  description.
- Effective sample size is far below 433 × 32 × 32: neighbouring cells are correlated, and in effect
  there are 15 boxes.
- Inspect Z itself: a compact blob in the core supports a rank-one template; several separate
  regions mean more components or a poor fit.

## Literature (identifiers from Undermind records)

- Bro (1996), Journal of Chemometrics 10:47–61, doi 10.1002/(SICI)1099-128X(199601)10:1<47::AID-CEM400>3.0.CO;2-C. Read.
- Liland, Indahl, Skogholt & Mishra (2022), canonical partial least squares for multiway data,
  doi 10.1002/cem.3432. Abstract.
- Zhao et al. (2012), higher order partial least squares, IEEE TPAMI, doi 10.1109/TPAMI.2012.254. Abstract.
- Li et al., Tucker tensor regression and neuroimaging analysis, Statistics in Biosciences,
  doi 10.1007/s12561-018-9215-6. Abstract. Year unverified: the record says 2013, the identifier
  suggests 2018.
- Folch-Fortuny et al. (2016), N-way PLS-DA on VIS/NIR hyperspectral images of citrus decay,
  doi 10.1016/J.CHEMOLAB.2016.05.005. Abstract.
- Mora Jimena, De Ketelaere & Saeys (2025), partial Tucker decomposition for hyperspectral
  classification with image-level labels, doi 10.1016/j.saa.2025.126584. Abstract.
- Lu, Plataniotis & Venetsanopoulos (2008), multilinear principal component analysis, doi
  10.1109/TNN.2007.901277. Title. Itoh, Imiya & Sakai (2016), three-way principal component analysis
  of volumetric images, doi 10.1007/978-3-319-51237-2_9. Title.
- Van De Looverbosch et al. (2020), pear CT with a support vector machine, doi
  10.1016/j.foodcont.2020.107170. Abstract.
- Novelty: four abstract-level searches found no application of N-way methods to fruit X-ray CT.
  This does not establish novelty.

## Tools (documentation read; nothing run)

- Solo (Eigenvector): lists N-way partial least squares under regression, no MATLAB needed; orders
  of X accepted not documented; Ruben has access. Whether its cross-validation can split by whole
  boxes: not verified.
- R: NPLStoolbox 1.1.0 (`triPLS1`, `ncrossreg`, `npred`) and sNPLS 1.0.27 (L1 penalised): three-way.
- Python: no package found in two searches; nirs4all-methods (C++ with bindings) documents three-way
  and no verified Python interface.
- Plan: own numpy implementation later, cross-checked on a three-way slice (see the handoff of
  10 October).

## Considered on 9 October, not adopted

- Multivariate image analysis (Wise & Geladi note): needs a channel mode; one value per voxel gives
  none. Not used.
- John, Kabuka & Ibrahim (2004), brain MRI segmentation (doi 10.1007/s10278-003-1664-9): rare small
  classes are disfavoured by the priors in its decision rule (Eq. 14); no voxel labels to validate
  against. Not used.
- Healthy-reference principal component model with residual (Q) maps: open, not decided.
