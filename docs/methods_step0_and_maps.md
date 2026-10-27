# Methods record: Step 0 and the height-radius maps (notebooks 06 and 07)

Written 27 October 2026 so the methods are not lost before the thesis is written. Every rule
here is what the code does; the code is the reference (`src/pearlatent/reconstruction.py`,
`src/pearlatent/step0.py`, `src/pearlatent/pooling.py`). Dates say when a rule was fixed or
changed. Numbers come from runs on the encoding run `20261013T171306Z_hugo_vae`.

## 1. Fruit

- Decoded for Step 0: 478 fruit = the 346 open 2526 fruit, Hugo's 66 validation fruit, and a
  sample of 66 of his 528 training fruit (seed 369; a separate group, never pooled).
- Faulty volumes excluded by visual inspection (`docs/outlier_rule.md` section 9): 35 out,
  443 left, of which 315 are 2526. Every map and spectrum uses these 315 2526 fruit.

## 2. Reconstruction and its error (notebook 06 sections 2 to 4)

- Reconstruction = `vae.decode(mu)`, never `vae(x)` (its forward pass samples z, so the output
  changes from run to run). The latent analysed is `mu`.
- Computed during the encoding run, one fruit at a time: the full 128³ reconstruction exists
  only in memory. From it all error numbers are computed, then saved
  (`reconstruction.parquet`) together with the three central planes (through the rounded mean
  position of the body voxels) of CT and reconstruction (`planes.npy`). The full
  reconstruction is discarded.
- Regions (fixed 13 October 2026, before any reconstruction was seen):
  - whole volume: all 128³ voxels (comparable with the preprint; mostly easy background);
  - body: voxels above 0.1 (scaled intensity) inside the 32³ body mask (section 4), enclosed
    holes filled, so cavities count as body;
  - core: body voxels within one third of the widest radius of the stem-calyx axis. Axis = the
    line along the third array axis through the mean x and mean y of the body voxels. Widest
    radius = sqrt(largest slice area / pi). One third because internal browning develops
    around the core and seed cavity.
- Contrast retention (fixed 23 October 2026; only the saved planes were left by then): per
  fruit the mean CT and the mean reconstruction over the core voxels of the two long sections
  (planes 0 and 1, which contain the axis). Per grade g: [mean reconstruction (g) − mean
  reconstruction (grade 0)] / [mean CT (g) − mean CT (grade 0)]. 1 = the darkening is fully
  kept, 0 = the reconstructed core looks healthy. Groups (2526): browning only, cavity only,
  any cause. Interval: 95 % percentile bootstrap over the 12 open boxes (boxes drawn with
  replacement, 2,000 draws, seed 369, the same draws for every row).
- Limitation: retention uses two long sections, not the whole 3D core (question 35).

## 3. Coordinates (preprocessing, verified in notebooks 02 and 03)

Hugo's preprocessing aligns the stem-calyx axis with the third array axis (calyx at the low
end, in every volume) and rescales every fruit to 128 voxels in length. So height is
comparable between fruit; the rotation about the axis is not, and is removed by averaging
over the angle.

## 4. Body mask (32³, `pearlatent.pooling.body_mask`)

1. A cell is fruit when at least half of its 4³ voxels are above 0.1.
2. Keep the largest connected piece.
3. Remove slices with fewer than 4 cells (the stalk).
4. Fill enclosed holes (cavities count as body).

## 5. Height and radius (fixed 27 October 2026; one rule in every map)

- **Height:** 0 at the first body slice (calyx end), 1 at the last (stalk end, stalk removed):
  (slice − first) / (last − first + 1).
- **Radius:** distance to the axis divided by the distance of the outermost body voxel (CT
  maps) or body cell (latent maps). 0 = axis, 1 = skin. No value exceeds 1, and every body
  voxel or cell is binned.
  - CT and reconstruction maps: on the two long sections; the outermost voxel is taken over
    both sections together; one straight axis per fruit (mean x, y of the 3D body voxels).
  - Latent maps, two methods (`radius_method`):
    - "fruit": one straight axis per fruit (mean x, y of the body cells), divided by the
      outermost body cell of the fruit. Keeps the shape: the narrow neck only reaches small
      radii.
    - "slice": the centre of each slice (mean x, y of its body cells), divided by the
      outermost body cell of that slice. Removes the shape: radius 1 is the skin at every
      height, a radius bin is a relative depth below the skin.
- No interpolation anywhere: every voxel or cell keeps its value and only gets a coordinate.
- Before 27 October the radius was divided by sqrt(largest slice area / pi); skin voxels of
  slices that are not round then landed up to about 1.2.
- The core of section 2 keeps its 13 October definition. On the map radius its edge lies at
  0.32 (median over the 346 open 2526 fruit; 5 % to 95 %: 0.30 to 0.34).

## 6. Bins

- CT and reconstruction maps: 32 height × 24 radius bins (voxels).
- Latent maps: 20 height × 8 radius bins (Ruben, 27 October 2026). The 2526 body spans 21 to
  32 latent slices (median 28), so 20 height bins give every fruit at least one slice per bin;
  the fruit is about 6 to 11 cells in radius (median about 7.5), so 8 radius bins are about one
  cell each. No smoothing.
- A bin is half-open, [lower, upper); radius 1 (the outermost voxel or cell) goes into the
  last bin.

## 7. Group mean per bin (fixed 27 October 2026; `pearlatent.step0.group_bin_summary`)

1. Per fruit and bin: the mean of that fruit's voxels or cells in the bin
   (`fruit_bin_means`).
2. Per group and bin: the mean of those per-fruit values over the fruit of the group. Every
   fruit counts once.
3. A bin is drawn only when at least half of the group's fruit have a value in it
   (`MIN_FRUIT_SHARE_PER_BIN = 0.5`); bins near the skin that only the widest fruit reach
   would describe those few fruit, not the group.

Example: fruit A has 10 cells in a bin with mean 5, fruit B 2 cells with mean 1. Group mean =
(5 + 1) / 2 = 3. (Before 27 October all cells were pooled: (10 × 5 + 2 × 1) / 12 = 4.33,
so large fruit weighed more.)

The spectra also show the 25 % and 75 % of the per-fruit values: the spread between fruit,
not an interval.

## 8. Groups (2526)

grade 0 (defective 0); browning only 1, 2, 3 (cavity 0); cavity 2 or 3 with any browning
(cavity only has 3 fruit in the open boxes, too few). Sizes after the exclusions: 145, 38, 20,
30, 61.

## 9. Corrections (notebook 07 section 3)

Per fruit and bin, the fruit's value minus a grade 0 reference of that bin. The reference is a
group mean (section 7), so it exists only where at least half of the reference fruit reach the
bin; elsewhere the corrected value is empty.

- minus the grade 0 mean (all open 2526 grade 0 fruit);
- minus the grade 0 mean, divided by the grade 0 standard deviation (ddof 1) of the bin;
- minus the grade 0 mean of the fruit's own box (removes differences between orchards). Uses
  the labels of every box: for looking only.

**Decision for the models (Ruben, 27 October 2026):** inside cross-validation, subtract the
grade 0 mean of the training boxes only. A held-out box cannot use its own grade 0 fruit,
because picking them out needs their labels.

## 10. Spectrum

The 160 latent bins of a fruit in one row, height by height: bin = 8 × height bin + radius
bin (bins 0 to 7: calyx end, axis to skin; 152 to 159: stalk end). Exploration only: bins
picked by eye are not a variable selection; any selection later starts from the full range,
inside the box folds. A height-radius summary as a model row would reopen the decision of 10
October that dropped the cylindrical bins.

## 10b. PCA on selected bins (notebook 07 section 3, added 27 October 2026)

The bins selected by both drags: centre inside the box on the map, and bin number inside the
stretch dragged along the spectra (with both, the bins that satisfy both; with neither, all
bins), after the chosen radius method and correction; one row per 2526 fruit. Bins that not every fruit
reaches are left out (PCA cannot take empty values). PCA centred, not scaled, up to 20
components; a scree plot shows the share of variance per component and the cumulative share. Colourings:
stalk length, body voxels, defective grade, cause; Spearman correlation of PC1 to PC3 with
stalk length, body voxels, defective, browning, cavity.

- Circular on purpose: bins chosen where grades differ separate the grades by construction.
  Insight only, not evidence.
- The grade 0 correction does not change a PCA (a constant shift per bin, removed by
  centring); the box correction and the standard deviation option do.
- Use: whether stalk length and body size still run along PC1 in the selected bins
  (question 38).

## 11. How to read a latent map (a property of Hugo's encoder)

- A latent value has no unit and no physical sign: the KL weight is 1e-6, a compression
  latent. Lower is not darker.
- One latent cell is computed from a block of about 31 voxels (about 19 mm, differs per
  fruit) around it, and InstanceNorm adds information from the whole fruit. The latent map
  says roughly where (about ±1 cm); the CT map says where.

## 12. Known confound (question 38)

The map groups differ in stalk and body size, not only in grade: median stalk 20, 19, 13.5,
12 voxels and median body 142,600, 150,900, 174,300, 182,500 voxels for grade 0 and browning
only 1, 2, 3; browning 3 mostly in boxes O (8) and A (6). A group difference in a map can
partly be shape or box. To discuss with Hugo.
