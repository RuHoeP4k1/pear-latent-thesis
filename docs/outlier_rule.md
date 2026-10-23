# Rule for excluding faulty volumes

Written 23 October 2026. Status: **principle approved by Ruben on 23 October 2026**, as written
below. The decision per fruit is Ruben's and is recorded in `derived/volume_decisions.csv`
(outside git, section 6). Nothing has been removed yet.

## 1. What is excluded, and what is not

A volume is excluded when **the volume is faulty**, never because the fruit is defective. The
argument: a sorting line would reject a fruit whose measurement failed or whose shape it cannot
handle, but a model for internal browning must keep every real fruit, however damaged inside.

Excluded (judged by eye on the CT slices):

| Code | Meaning | How it looks |
|---|---|---|
| `missing_part` | Part of the fruit is not in the volume | a straight, often oblique, cut through the fruit; a half-disc cross-section |
| `preprocessing` | The preprocessing (alignment, rescaling to 128 voxels, intensity scaling) went wrong | the fruit fills a clearly smaller part of the grid than other fruit because a stray object was counted in its length; or the whole fruit is clearly darker or brighter than the others because one bright voxel outside the fruit set the maximum of the min-max scaling |

Not excluded:

- an odd but real shape (curved, banana-like, very long or very round);
- internal browning or cavities of any size (in 2425 the suboptimal-storage fruit are flagged
  by the screen of section 2 because of their large cavities; they stay in);
- a fruit that is only far from the others in a score plot, with nothing visibly wrong.

## 2. How candidates are found

Two routes, so that a fault is not missed because the screen happens not to flag it.

1. **Screen** (notebook 05, sections 7 and 8). Principal component analysis (PCA) of the latent,
   all 32,768 cells, mean-centred, not scaled, exact solver, fitted on the 346 open 2526 fruit
   (the 87 fruit of the test boxes are sealed and not in the fit). A = 13 components (the elbow
   of the scree plot, read by Ruben on 21 October 2026). Hotelling T² with the limit
   A(n − 1)/(n − A) · F(A, n − A) and the Q residual with the limit of Box (1954) as used by
   Nomikos and MacGregor (1995), both at 99 %. A fruit beyond **either** limit is a candidate.
2. **Contact sheet** (notebook 05, section 9). Two slices through the centre of the body of
   **every** open fruit (the 346 open 2526 fruit and Hugo's 66 validation fruit), shown without
   any label, in a fixed order (by fruit identifier). Ruben looks through all of them once.

Every candidate from either route is judged with the codes of section 1.

## 3. Blind to the grades

The decision uses the CT slices only. The contact sheet shows no grades. The first version of
the panel of section 8 (23 October) showed the grades in the fruit titles; from now on the
panel hides them by default. After the decisions, the grades of the excluded fruit are listed
next to the grades of all fruit of the same group, so a reader can see whether one grade was
removed more than the others.

## 4. The sealed fruit

At the final test the same procedure is applied to the sealed fruit (the 87 fruit of the test
boxes F, J, M and Hugo's 66 test fruit) **before any prediction for them is looked at**: the PCA
of section 2 (fitted on the 346 open 2526 fruit, not refitted) gives T² and Q, the contact sheet
shows all of them, and the same codes decide.

## 5. Reported

- the list of excluded fruit with code, group and grades;
- the number of candidates from each route;
- one sensitivity row in every main result table with all volumes kept (exclusion mode
  `all_volumes`, `pearlatent.exclusions`).

## 6. Where the decisions live

`derived/volume_decisions.csv` (ignored by git, like `exclusions.csv`), one row per judged fruit:
`fruit_id, season, decision, note`, with `decision` one of `missing_part`, `preprocessing`,
`keep`. `pearlatent.exclusions.build_exclusions` turns `missing_part` and `preprocessing` into
rows with action `exclude` and a reason starting with `faulty volume:`.

## 7. Candidates of 23 October (screen only, A = 13, 99 %)

30 candidates among the 412 shown fruit (346 open 2526 and 66 2425 validation). Claude's reading
of the panel on 23 October, **not a decision**:

| Reading | Fruit |
|---|---|
| straight cut (`missing_part`?) | 2526_L22, 2526_L26, 2526_L29, 2526_N29, 2425_C59, 2425_I44, 2425_I45 |
| preprocessing? | 2526_C03 (whole fruit darker: tissue intensity 0.65 against about 0.82 for other fruit; borderline) |
| long stalk (keep?) | 2526_K27: the fruit looked small in the grid with a speck in a corner; measured: no object above the threshold besides the fruit, but a stalk of 36 voxels (2526_A01: 12), so the body is short because the stalk counted in the 128-voxel length. This is the known stalk convention, not a failure |
| odd real shape (keep) | 2526_O19, 2425_B37, 2425_H41 |
| nothing visible (keep) | 2526_A25, 2526_D21, 2526_K12 |
| genuine large cavities (keep) | the 15 flagged 2425 suboptimal-storage fruit |

Hypothesis for the straight cuts, to ask Hugo: the fruit reached beyond the field of view of the
scanner, and the rotation that aligned the stem–calyx axis turned the straight edge of the field
into an oblique cut.

## 8. Contact sheet and descriptor screen of 23 October (Claude's first pass, not a decision)

Claude looked through all 9 pages (`derived/figures/05_contact_sheet/`, 412 fruit, no labels).
Small images, so every item below is "to look at", not a finding.

**Straight cut through the body** (the cross-section through the body centre has a flat edge,
or the long section is cut far below the neck): 2425_I44, 2425_I45, 2526_I13, 2526_I25,
2526_I28, 2526_L26, 2526_L29, 2526_N29; possibly 2526_I10, 2526_I07, 2526_L16, 2526_E04,
2526_A10, 2526_A13, 2526_H01.

**Neck tip cut off only** (flat top of the long section, cross-section round): 2425_C59,
2425_F59, 2526_D06, 2526_D10, 2526_D12, 2526_D21, 2526_D27, 2526_E28, 2526_G27, 2526_I12,
2526_L22, 2526_L28.

**Intensity** (descriptor screen): dark 2425_A36 (tissue median 0.54), 2526_C03 (0.67),
2526_I03 (0.68), 2526_L03 (0.69); brightest 2526_N15 (0.95, against a 2526 median of 0.85 and
99th percentile 0.93). Three of the four dark 2526 fruit are number 03 of their box.

**Stray object** (descriptor screen, more than 100 voxels): 2526_K13 (607), 2425_H53 (385),
2526_A30 (232), 2526_K02 (127); nothing visible on their central slices. 2526_O04 has a round
object on top of the neck that is connected to the fruit (counted as stalk).

**Pattern:** the cuts cluster in boxes I and L of 2526. If cuts depend on the box (scan
session), excluding them removes fruit unevenly over boxes; to report.

**Open for Ruben (amendment to section 1):** the rule says a volume with a missing part is
excluded. About 12 fruit miss only the tip of the neck, far from the core where browning
develops. Proposal: `missing_part` only when the cut reaches the body (cross-section through
the body centre not complete, or the cut is below the widest part of the neck); a cut neck tip
is kept with the note `neck tip cut`, and dropped in the sensitivity row.
