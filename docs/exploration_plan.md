# Exploration of the latent space: plan and assumptions (notebook 07)

Written 23 October 2026, before notebook 07 computed anything. Nothing here fits a model that
predicts a label; the labels are used as colours and as groupings to test. No number from this
notebook is used to choose a setting of the main models.

## 1. Fruit

- The 346 open 2526 fruit (the 87 fruit of the test boxes F, J, M stay sealed).
- For the season comparison only: Hugo's 66 2425 validation fruit. His 528 training fruit are
  left out (charter rule 2), and so are his 66 test fruit (sealed).
- Faulty volumes are dropped once Ruben's decisions exist (`docs/outlier_rule.md`).

## 2. Pictures (qualitative only)

Charter, "Do not": no quantitative conclusion from a 2D embedding. t-SNE and UMAP distances
between clusters, cluster sizes and densities have no fixed meaning; only "which fruit lie near
which" is read from them.

- Input: the scores on 50 principal components of the full grid, PCA fitted on the 346 open
  2526 fruit (as in notebook 05), all fruit projected. Using 50 scores instead of 32,768 cells
  removes noise and makes the neighbour search stable.
- t-SNE (scikit-learn): perplexity 30, PCA initialisation, seed 369.
- UMAP (`umap-learn`): 15 neighbours, minimum distance 0.1, Euclidean, seed 369.
- Colours: season, box, storage, defective grade, cause, browning, cavity, and the descriptors
  stalk length, body voxels, diameter to length, tissue intensity, core reconstruction error.
- With the scatter plots: Spearman correlations of PC1 to PC10 with the descriptors and grades
  (which fruit properties the main directions follow).

## 3. PERMANOVA (quantitative)

**Dropped from the analysis (Ruben, 26 October 2026).** Not shown to Hugo and not used in the
thesis. The code (`pearlatent.permanova`, notebook 07 section 3) and the section below are
kept as a record of what was tried until Ruben decides whether to delete them.

Permutational multivariate analysis of variance (Anderson 2001, Austral Ecology 26, 32-46):
does a grouping explain more of the distances between fruit than chance?

- **Distance:** Euclidean distance between the full latent grids (32,768 cells, not scaled),
  so no number of components has to be chosen.
- **Statistic:** pseudo-F = [SS_between / (a − 1)] / [SS_within / (N − a)], computed from the
  squared distances; effect size R² = SS_between / SS_total. Both are reported, not only p.
- **Permutations:** 9,999, seed 369; p = (number of permuted F ≥ observed F + 1) / 10,000.
- **Planned tests, 2526 fruit:**

  | Grouping | Permutations | Why restricted |
  |---|---|---|
  | box (12 orchards) | free | |
  | defective grade (0 to 3) | within box | boxes differ in grade mix; free permutation would credit box differences to the grade |
  | cause (none, browning only, cavity only, both) | within box | same |
  | stalk length in three classes (tertiles) | within box | same |

  One more test, descriptive only: season (346 2526 against 66 2425 validation fruit), free
  permutation. Season is confounded with scanner session, storage and stalk, so a difference
  says only that the seasons differ, not why.
- **Multiple tests:** the four 2526 p-values are corrected with Holm's method.

**Assumptions** (from Anderson 2001 and 2006, stated before the results):

1. **Exchangeability under the null hypothesis.** Fruit can be swapped between the groups
   being compared. Fruit of one box are not independent (same orchard), so for every grouping
   other than box the swaps are restricted to fruit of the same box.
2. **Homogeneous dispersion.** PERMANOVA reacts to a difference in the position of the group
   centres and also to a difference in their spread. So every test is paired with PERMDISP
   (Anderson 2006, Biometrics 62, 245-253): the distance of every fruit to its group centre,
   compared between groups with an F-test with the same permutations. A significant PERMANOVA
   with a non-significant PERMDISP points to a difference in position.
3. **The distance decides what counts.** The unscaled Euclidean distance is dominated by the
   cells with the largest variance between fruit: the border zone of the fruit, which carries
   shape and size (23 October 2026: 47 % of the variance in cells inside the body of at least
   half the fruit, 8 % outside every fruit). A small R² for the grade can therefore mean that the
   grade signal is small compared with shape, not that it is absent.
4. **Unbalanced groups** (for example 152 grade 0 fruit against 22 browning-only grade 2) make
   the test more sensitive to dispersion differences (Anderson and Walsh 2013), which is
   another reason for PERMDISP.

## 4. Group centres (MDS)

The mean latent of every combination of box and defective grade with at least 3 fruit, and the
Euclidean distances between these means, drawn in two dimensions with metric
multidimensional scaling (MDS; scikit-learn, seed 369). The picture shows whether centres of
the same box lie closer together than centres of the same grade. Qualitative, as in section 2;
the quantitative counterpart is section 3.

## 5. Not here

The nuisance check of CLAUDE.md section 7 (does the latent predict box or storage better than
the defect?) fits classifiers on labels and so belongs with the main models, under the same box
splits. The PERMANOVA R² of box against grade gives the first, exploratory answer.
