# Rule for excluding faulty volumes

Written 23 October 2026. Status: **principle approved by Ruben on 23 October 2026**, as written
below. The decision per fruit is Ruben's and is recorded in `derived/volume_decisions.csv`
(outside git, section 6). Decisions on the open fruit made on 26 October 2026 (section 9).

## 1. What is excluded, and what is not

A volume is excluded when **the volume is faulty**, never because the fruit is defective. The
argument: a sorting line would reject a fruit whose measurement failed or whose shape it cannot
handle, but a model for internal browning must keep every real fruit, however damaged inside.

Excluded (judged by eye on the CT slices):

| Code | Meaning | How it looks |
|---|---|---|
| `missing_part` | Part of the fruit is not in the volume, anywhere, including only the tip of the neck (Ruben, 23 October 2026: a cut-off neck is a faulty measurement too) | a straight, often oblique, cut through the fruit; a half-disc cross-section; a flat end of the neck where the fruit should taper towards the stalk |
| `preprocessing` | The measurement or the preprocessing (alignment, rescaling to 128 voxels, intensity scaling) went wrong | the fruit is deformed in a way a real pear is not (Ruben, 26 October 2026: 2526_L26, L29, N29); the fruit fills a clearly smaller part of the grid than other fruit because a stray object was counted in its length; or the whole fruit is clearly darker or brighter than the others because one bright voxel outside the fruit set the maximum of the min-max scaling |

Not excluded:

- an odd but real shape (curved, banana-like, very long or very round);
- internal browning or cavities of any size (in 2425 the suboptimal-storage fruit are flagged
  by the screen of section 2 because of their large cavities; they stay in);
- a fruit that is only far from the others in a score plot, with nothing visibly wrong;
- a volume that is darker or brighter than the others while the fruit itself looks normal
  (Ruben, 23 October 2026: 2526_C03, I03, L03 and N15 stay in). Intensity alone is therefore
  not a reason to exclude; `preprocessing` is used only for a failure that changes the fruit in
  the grid.

## 2. How candidates are found

Three routes, so that a fault is not missed because the screen happens not to flag it.

1. **Screen** (notebook 05, sections 7 and 8). Principal component analysis (PCA) of the latent,
   all 32,768 cells, mean-centred, not scaled, exact solver, fitted on the 346 open 2526 fruit
   (the 87 fruit of the test boxes are sealed and not in the fit). A = 13 components (the elbow
   of the scree plot, read by Ruben on 21 October 2026). Hotelling T² with the limit
   A(n − 1)/(n − A) · F(A, n − A) and the Q residual with the limit of Box (1954) as used by
   Nomikos and MacGregor (1995), both at 99 %. A fruit beyond **either** limit is a candidate.
2. **Contact sheet** (notebook 05, section 9). Two slices through the centre of the body of
   **every** open fruit (the 346 open 2526 fruit and Hugo's 66 validation fruit), shown without
   any label, in a fixed order (by fruit identifier). Ruben looks through all of them once.

3. **Flat-face screen** (added 23 October 2026). `flat_face_share` of `pearlatent.descriptors`:
   the largest flat patch of the fruit surface as a share of all surface voxels (RANSAC plane
   search, 3,000 planes, tolerance 1 voxel, seed 369). Natural skin is curved; a cut leaves a
   flat face. On 22 fruit read by eye (23 October): cuts through the body 0.13 to 0.29, normal
   fruit 0.07 to 0.09, the curved 2526_O19 0.11. No cut-off separates cuts from real shapes
   (2526_G29 and I04 at 0.110 are cuts, 2526_O19 at 0.113 and 2425_J27 at 0.114 are not), so
   the number decides nothing (Ruben, 26 October 2026): every fruit at 0.10 or above was shown
   with its flat patch in red for a second look, and the eye decided. The measure does not
   separate a cut-off neck tip (0.05 to 0.08), so neck tips are found by eye on the contact
   sheet only.

Every candidate from any route is judged by eye with the codes of section 1; the screens only
propose candidates.

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

**Decided by Ruben on 23 October 2026:** a cut-off neck tip is a faulty measurement and is
excluded as `missing_part` (section 1). Confirmed on 26 October 2026, including the very minor
neck cuts; the sensitivity row with all volumes (section 5) shows whether they matter. (An
earlier proposal to keep cut neck tips was not adopted.)

## 9. Decisions of 26 October 2026 (Ruben, open fruit)

Ruben looked through the contact sheet blind first, then compared his list with section 8 and
the flat-face panel, and looked at the fruit where the routes disagreed (10 slices per fruit
for 2526_E04, H01, I19). All decisions are in `derived/volume_decisions.csv` (52 rows).

| Decision | Fruit |
|---|---|
| `missing_part`, cut through the body (20) | 2425_C59, I44, I45; 2526_A07, A10, A13, D21 (calyx end), E07, E13, G29, G30, I01, I04, I07, I10, I13, I25, I28, L07, L16 |
| `missing_part`, neck or top cut (12) | 2425_F59; 2526_D06, D10, D12, D27, E27, E28, G27, I11, I12, L22, L28 |
| `preprocessing`, deformed volume (3) | 2526_L26, L29, N29 |
| `keep` (17) | intensity only: 2425_A36, 2526_C03, I03, L03, N15; stray object not touching the fruit: 2526_K13, A30, K02, 2425_H53; real odd shape: 2526_E04, O19, O25, 2425_J27, C24; fine: 2526_I19, H01; long stalk: 2526_K27 |

35 volumes excluded: 31 of the 346 open 2526 fruit and 4 of Hugo's 66 validation fruit. Per
2526 box: I 9 (of 29), L 6, D 5, E 4, A 3, G 3, N 1, others 0. The test boxes F, J, M are not
inspected yet (section 4), so their 0 means "not looked at".

Seen on the PCA of notebook 05 (fit on the 346 open 2526 fruit, A = 13, 99 % limits; no
grades): only 6 of the 18 body cuts then on the list and 1 of the 11 neck cuts lie beyond a
T² or Q limit, so the screen of route 1 alone would have missed most cuts. The dark volumes
stand out on PC3. Position in the PCA did not enter any decision.
