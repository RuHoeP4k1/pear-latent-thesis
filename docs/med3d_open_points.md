# Med3D as the frozen comparison row: open points (2 November 2026)

Written before any Med3D code, so Ruben can search in parallel and check what is built.
Hugo (29 October): the choices are Ruben's; features expected by 17 November.

Everything marked **[check]** is from memory of the Med3D paper or repository, not verified
against Tier 1. Please confirm or correct it from the sources:

- Chen, Ma and Zheng (2019), *Med3D: Transfer Learning for 3D Medical Image Analysis*,
  arXiv 1904.00625.
- Repository github.com/Tencent/MedicalNet (code, weights, licence).

## What Med3D is

- **[check]** A family of 3D ResNets (ResNet-10, 18, 34, 50, 101, 152, 200), pretrained on
  "3DSeg-8": eight public medical segmentation datasets of CT and MRI (brain, hippocampus,
  prostate, liver, heart, pancreas, vessels, spleen). One shared encoder with one
  segmentation head per dataset.
- **[check]** The last two stages (layer3, layer4) use dilated convolutions with stride 1, so
  the output is 1/8 of the input size in every direction. A 128³ input gives a 16³ grid; with
  512 channels (ResNet-10, 18, 34) or 2048 (ResNet-50 and larger).

## Differences between their data and ours

| Point | Med3D (pretraining) | Our volumes | Why it matters |
|---|---|---|---|
| Intensity | **[check]** Calibrated CT in Hounsfield units, or MRI; each volume normalised to zero mean and unit variance over the foreground voxels | Industrial X-ray CT, no calibration. The 16-bit range 0–65535 was set per volume before we received it (probably by the brightest voxel; Hugo does not know). Min-max scaled to [0, 1] per volume. | The pretrained filters expect a certain intensity range. Which normalisation we pass (z-score over the fruit body, or min-max as for Hugo's encoder) is a choice to fix before looking at any result. |
| Global darkening | Normalisation per volume | Same problem in our case: a fruit that is browned throughout has a lower mean | Z-scoring per fruit partly removes a whole-fruit darkening. Min-max and Hugo's InstanceNorm do something similar, so the comparison stays fair; to be stated as a limitation. |
| Spacing | **[check]** Mostly anisotropic (thick slices), in millimetres, resampled to a fixed input size | Isotropic, but every fruit rescaled to 128 voxels in length, so a voxel has a different size in every fruit (unknown, because the original lengths were not measured) | No millimetre scale is possible (no original lengths, Hugo 29 October). Filters learnt on organs see a pear of a different apparent size. |
| Axis order | **[check]** D × H × W, with D the slice axis | The stem–calyx axis is the third array axis | We must choose which of our axes goes to D. The first convolution (7 × 7 × 7) is cubic, so the effect is probably small but not zero. |
| Task | Organ segmentation: edges and boundaries | Detection of a small density shift (browning) and gas voids (cavity) | Segmentation features may again mostly code shape, as PC1 does for Hugo's latent (size and stalk). Cavity, which has high contrast, is the most likely to transfer. |
| BatchNorm | Running statistics of medical data | Our intensities lie elsewhere | Used frozen (evaluation mode, running statistics as stored). Activations may be off-scale; check the spread of the features before modelling. |
| Sample size | Thousands of volumes in pretraining | 315 open 2526 fruit after the exclusions | No fine-tuning: frozen features and PLS only, like Hugo's latent. |

## The choices that must be fixed before any result

1. **Which network.** ResNet-10 or ResNet-18 are realistic on the laptop CPU; ResNet-50 has
   2048 channels and is about 4 times slower. Proposal: ResNet-18, the same depth as the
   ResNet-18 baseline we train from scratch, so pretrained versus from scratch is at the same
   architecture depth. **[check]** whether ResNet-18 weights are released.
2. **Which layer.** A choice between three:
   - **layer1 output.** **[check]** 64 channels on a 32³ grid for a 128³ input (stride 4). The
     same grid as Hugo's latent, a receptive field of about 40 voxels (to compute exactly from
     the layer list), and 2.1 million numbers per fruit.
   - **layer4 output.** 512 channels on a 16³ grid: 2.1 million numbers per fruit, the most
     "trained" features.
   - **Global mean of layer4.** 512 numbers per fruit. Small and standard in transfer
     learning, but a 1 cm³ lesion moves a global mean very little (the same argument as for
     mean pooling of the latent, CLAUDE.md section 3).

   Proposal: layer4, mean-pooled to 8³ × 512 (262,144 numbers), and the global mean as a
   second row. Choosing among layers by looking at the results would be a selection that
   counts against the final test. So either the layer is fixed now, or it is chosen inside the
   box cross-validation of the 12 open boxes and reported as such.
3. **Which normalisation.** Z-score over the body mask (closest to their pretraining) or
   min-max (as for Hugo's encoder). Proposal: z-score over the body, as their set-up.
4. **Storage.** 2.1 million float32 numbers per fruit is 8 MB per fruit, about 9 GB for all
   1,093 fruit. Pool during encoding and never save the full layer (the same rule as for
   Hugo's 512-channel layer).
5. **Licence and source of the weights.** **[check]** The repository and its licence (MIT?),
   where the weights are hosted (cloud drive links in the README, possibly also on Hugging
   Face), and whether the file can be verified (checksum). Weights never enter git.

## Where Ruben can help search

- The licence of MedicalNet and the release of each depth (10, 18, 34, 50).
- The exact preprocessing in the paper and in the repository's data loader: normalisation,
  input size, axis order.
- Any use of Med3D or MedicalNet features outside medicine (food, plant or industrial CT).
  This tells us whether a domain gap of this size has been tried before.
- The receptive field of layer1 and layer4, to compare with the 19 mm of a latent cell.
