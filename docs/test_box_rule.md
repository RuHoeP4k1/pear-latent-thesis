# Rule for the 2526 test boxes and the outer folds

Written 13 October 2026, before any draw (charter rule 3, version of 13 October; agreed in
principle with Hugo on 13 October). The rule is committed to git before the draw is run; the commit
hash of this file goes into the `run.json` of the draw. Only labels are used, never CT features or
results. Status: **approved by Ruben on 13 October 2026**, as written below.

## 1. Words used

- **Box**: the 30 fruit of one 2526 orchard (box letter = orchard ID, Hugo). 15 boxes, A to O.
- **Test boxes**: 3 boxes put away until the whole pipeline is frozen, then predicted once.
- **Fold**: a group of 3 boxes that is held out together in one round of the cross-validation. The
  model of that round is fitted on the other 9 boxes and predicts the fold.
- **Partition**: one way to divide the 12 non-test boxes into 4 folds of 3. Within one partition
  each box is in exactly one fold, so after 4 rounds every fruit has exactly one out-of-fold
  prediction.
- **Repeat**: the whole 4-round cross-validation run again with a different partition. With 5
  repeats each fruit has 5 out-of-fold predictions; the spread over repeats shows how much the
  score depends on which boxes happen to share a fold.
- **Inner split**: inside one round, the 9 training boxes are split again (proposal: leave one box
  out) only to choose settings such as the number of partial least squares components. Its score
  is never reported as performance. The inner split is not defined by this rule; see
  `docs/cv_explained.md`, section 4.

## 2. Population

The 433 scanned 2526 fruit. The 17 unscanned fruit are not counted. The 24 fruit with browning 3
and cavity 3 are counted (they stay in, charter rule 10).

## 3. Balance counts

For a group of fruit, four counts:

| Count | Definition |
|---|---|
| non-consumable | `defective` ≥ 2 (equals `binary_23` = 1) |
| browning ≥ 2 without cavity | `browning` ≥ 2 and `cavity` = 0 |
| browning 3 | `browning` = 3, any cavity |
| any cavity | `cavity` ≥ 1 |

"Browning 3 with cavity 3" is not a count (dropped by Ruben on 13 October: it tracked suspected
rotten fruit, and rot is not really a problem in 2526).

## 4. Eligibility of a set of boxes

For a set S of boxes and one count c:

- O = the observed count of c in the scanned fruit of S.
- E = T × n / N, the expected count, where T is the total of c over the reference population, n
  the number of scanned fruit in S, and N the number of scanned fruit in the reference population.
  For the test boxes the reference population is the 433 scanned fruit (N = 433).
- S passes for c at tolerance t when (1 − t) × E ≤ O ≤ (1 + t) × E.

S is **eligible** at tolerance t when it passes for all four counts.

No rounding. E is not rounded and the bounds are not rounded; O is an integer. A count exactly on a
bound passes. The code compares exact fractions (O × N against (1 ± t) × T × n with t as a
fraction), so a value on the bound is not lost to floating-point rounding.

Example: E = 4.8, t = 0.30. The window is 0.70 × 4.8 = 3.36 to 1.30 × 4.8 = 6.24. O = 4 passes;
O = 3 and O = 7 fail. The integers that pass are 4, 5 and 6.

Consequence of no rounding: when E is small, the window may contain no integer (E = 1.5 gives 1.05
to 1.95 at ±30 %), and then no set can pass for that count. The fallback below handles that.

## 5. Test boxes

1. List all 455 sets of 3 of the 15 boxes. Within a set the boxes are written in alphabetical
   order; the sets are listed in lexicographic order (ABC, ABD, …, MNO).
2. Judge each set at t = 0.30 (reference population: the 433 scanned fruit).
3. Fallback: if fewer than 10 sets are eligible, judge all 455 again at t = 0.40; if still fewer
   than 10, at t = 0.50; if still fewer than 10, stop and ask Ruben. The tolerance used is recorded.
4. Draw: `rng = numpy.random.default_rng(369)`; `k = rng.integers(0, M)`, where M is the number
   of eligible sets; the test boxes are the k-th eligible set (0-based) in the lexicographic list.

## 6. Outer folds

1. List all 15,400 partitions of the 12 non-test boxes into 4 folds of 3 (12! / ((3!)⁴ × 4!)). In
   a partition, the boxes inside a fold are in alphabetical order and the folds are ordered by
   their first box; the partitions are listed in lexicographic order of that written form. Fold 1
   is the fold that contains the alphabetically first box, and so on.
2. A partition is eligible at tolerance t when each of its 4 folds is eligible at t (section 4).
   Reference population for the folds: **the 12 non-test boxes** (T and N are the totals over
   those boxes). *(Chosen by Ruben, 13 October 2026. Reason: the 4 folds divide exactly these 12
   boxes, so their counts add up to the 12-box total; with the whole season as reference, a fold
   could fail only because the test boxes took somewhat more or less than their share. The other
   option is the 433 scanned fruit, as for the test boxes.)*
3. Start at t = 0.30, whatever tolerance the test boxes needed. Same fallback: fewer than 10
   eligible partitions → 0.40 → 0.50 → stop and ask Ruben.
4. Draw 5 partitions (5 repeats; Ruben may change this number here before the draw):
   `rng = numpy.random.default_rng(369)` (a new generator, independent of section 5);
   `ks = rng.choice(P, size=5, replace=False)`, where P is the number of eligible partitions;
   repeat r (1 to 5) uses the eligible partition at position `ks[r − 1]` (0-based) in the
   lexicographic list.
5. *Amendment, 21 October 2026 (Ruben), after the draw of 13 October and before any model was
   fitted on labels:* the number of repeats is raised from 5 to 20, because 5 balanced partitions
   overlap strongly (repeats 2 and 3 share two folds) and repeated double cross-validation is
   normally run with many repeats (Filzmoser, Liebmann and Varmuza 2009, *Journal of
   Chemometrics* 23, 160–171). Repeats 1 to 5 stay exactly as drawn (Hugo has seen them). The
   15 new repeats are drawn from the P − 5 eligible partitions that were not drawn, kept in
   lexicographic order: `rng = numpy.random.default_rng(369)` (a new generator);
   `ks = rng.choice(P − 5, size=15, replace=False)`; repeat 5 + r uses position `ks[r − 1]`.
   Written to `derived/outer_folds_20/` (`python -m pearlatent.extra_repeats`); the draw of
   13 October in `derived/test_boxes/` is not changed.

## 7. What is written and reported

- `derived/test_box_sets.csv`: all 455 sets, with their four observed counts, expected counts and
  eligibility at each tolerance that was tried.
- `derived/test_boxes.csv`: the drawn boxes; their fruit are added to `derived/sealed_fruit.csv`.
- `derived/outer_folds.csv`: for each repeat, the fold of each of the 12 boxes.
- `run.json`: seed, tolerances tried and used, numbers of eligible sets and partitions, the commit
  of this file.
- Report to Ruben: the number of eligible sets and the tolerance; how often each box occurs among
  the eligible sets (a box that never occurs can never be a test box: the price of the
  restriction); the drawn boxes with observed beside expected counts; the number of eligible
  partitions and the drawn ones; how often each pair of boxes shares a fold across the 5 drawn
  partitions.

## 8. What the restriction costs

Unusual orchards (box K almost all healthy, box O many grade 3) may never be in an eligible set, so
they cannot be test boxes; they are covered by the cross-validation. Eligible partitions may keep
pairing the same boxes, so the 5 repeats differ less from each other than unrestricted ones
would; the pair table in section 7 shows by how much.
