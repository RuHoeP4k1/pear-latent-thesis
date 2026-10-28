"""Figures for questions 30 and 31 of HANDOFF.md, for the meeting with Hugo (28 October 2026).

Question 30: volumes with a straight, often oblique cut through the fruit, and dark volumes.
Question 31: stray objects above the intensity threshold beside the fruit.

Writes to `derived/figures/` (outside git):
- `meeting_q30_cut_volumes.png`: per fruit two long sections through the centre, cross-sections
  at four heights and the outline seen along the stem-calyx axis. A cut by a flat plane shows as
  a straight edge in every cross-section, at a position that shifts with the height when the
  plane is oblique.
- `meeting_q30_dark_volumes.png`: per fruit a long section and a cross-section on one fixed
  grey scale (0 to 1 after min-max scaling), the maximum projection with the brightest raw voxel
  marked, and the histogram of the raw intensities. If the brightest voxel lies outside the
  fruit, an object other than the fruit set the top of the min-max scaling and darkened the
  fruit.
- `meeting_q31_stray_objects.png`: per fruit the maximum projection along each axis and a slice
  through the centre of the stray voxels along each axis, stray voxels outlined in red. Read by
  Ruben (28 October): only 2425_J46 has an object beside the fruit; in the other four the
  stray voxels are the stalk, bent to the side and separated from the fruit at the threshold.
- `meeting_q30_q31_numbers.csv`: raw minimum and maximum, where the brightest voxel lies, the
  tissue median and the stray voxel count per fruit shown.

Only open fruit are read: every fruit is checked against `pearlatent.fruit_table.open_fruit`
first, so no sealed volume is opened.

    uv run python scripts/meeting_q30_q31_figures.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import polars as pl

from pearlatent.config import load_config
from pearlatent.descriptors import largest_part
from pearlatent.fruit_table import open_fruit
from pearlatent.pooling import MASK_THRESHOLD, scale_minmax

# Fruit shown. Each figure ends with normal fruit of the same boxes as a reference.
CUT = [
    "2425_C59",
    "2425_I44",
    "2425_I45",
    "2526_I01",
    "2526_I04",
    "2526_I13",
    "2526_I28",
    "2526_L07",
    "2526_L16",
    "2526_L22",
]
# Dark volumes, in the order of the figure: the five 2425 volumes far below the rest of 2425
# (tissue median 0.38 to 0.54; 2425 median 0.87, 1 % below 0.70; A51, D54, A57 and C43 are not
# named in question 30 but are darker than A36), two normal 2425 fruit of the same boxes, then
# the three 2526 volumes named in question 30 (0.67 to 0.69, a normal-looking tissue peak only
# slightly lower; Ruben, 28 October: they look normal) and three normal 2526 fruit.
DARK_2425 = ["2425_A51", "2425_D54", "2425_A57", "2425_C43", "2425_A36"]
# Healthy, at harvest, no cut, no stray object, tissue median at the 2425 median (0.87).
REFERENCE_DARK_2425 = ["2425_A41", "2425_D34"]
LOW_2526 = ["2526_C03", "2526_I03", "2526_L03"]
# All open fruit with more than 100 stray voxels (descriptors run of 23 October).
STRAY = ["2526_K13", "2425_H53", "2425_J46", "2526_A30", "2526_K02"]
REFERENCE = ["2526_I02", "2526_L02"]
REFERENCE_DARK_2526 = ["2526_C02", "2526_I02", "2526_L02"]
DARK_ORDER = DARK_2425 + REFERENCE_DARK_2425 + LOW_2526 + REFERENCE_DARK_2526
# Heights of the cross-sections, as a share of the fruit's extent along the axis (calyx 0).
CROSS_SECTION_HEIGHTS = (0.2, 0.4, 0.6, 0.8)


def load(cfg, row: dict) -> dict:
    """Raw volume, scaled volume (min-max to [0, 1], as for the encoder), the fruit (largest
    connected part above the threshold) and the stray voxels (above the threshold, outside the
    fruit) of one fruit."""
    file = cfg.path(f"ct_dir_{row['season']}") / f"{row['file_stem']}.nii"
    raw = np.asarray(nib.load(file).dataobj, dtype=np.float32)
    scaled = scale_minmax(raw)
    solid = scaled > MASK_THRESHOLD
    fruit = largest_part(solid)
    return {"raw": raw, "scaled": scaled, "fruit": fruit, "stray": solid & ~fruit}


def centre(mask: np.ndarray) -> tuple[int, int, int]:
    """Mean voxel position of a mask, rounded."""
    return tuple(int(round(c.mean())) for c in np.nonzero(mask))


def show(ax, image, title, **kwargs) -> None:
    """One grey picture, transposed so the stem-calyx axis (third array axis) is vertical."""
    ax.imshow(image.T, origin="lower", cmap="gray", vmin=0, vmax=1, **kwargs)
    ax.set_title(title, fontsize=7)
    ax.axis("off")


def label_row(ax, text: str) -> None:
    ax.text(
        -0.08,
        0.5,
        text,
        transform=ax.transAxes,
        rotation=90,
        va="center",
        ha="right",
        fontsize=8,
    )


def cut_figure(cfg, table: pl.DataFrame, ids: list[str], out) -> None:
    n_cols = 3 + len(CROSS_SECTION_HEIGHTS)
    fig, axes = plt.subplots(
        len(ids), n_cols, figsize=(1.7 * n_cols, 1.8 * len(ids)), squeeze=False
    )
    for r, fid in enumerate(ids):
        row = table.row(by_predicate=pl.col("fruit_id") == fid, named=True)
        v = load(cfg, row)
        i, j, _ = centre(v["fruit"])
        occupied = np.flatnonzero(v["fruit"].any(axis=(0, 1)))
        low, high = occupied[0], occupied[-1]
        show(axes[r, 0], v["scaled"][i], "long section, axis 0")
        show(axes[r, 1], v["scaled"][:, j], "long section, axis 1")
        for c, share in enumerate(CROSS_SECTION_HEIGHTS):
            z = int(low + share * (high - low))
            # Cross-sections are not transposed: they show axis 0 against axis 1.
            axes[r, 2 + c].imshow(
                v["scaled"][:, :, z], origin="lower", cmap="gray", vmin=0, vmax=1
            )
            axes[r, 2 + c].set_title(f"cross-section at {share:.0%}", fontsize=7)
            axes[r, 2 + c].axis("off")
        axes[r, -1].imshow(v["fruit"].any(axis=2), origin="lower", cmap="gray")
        axes[r, -1].set_title("outline along the axis", fontsize=7)
        axes[r, -1].axis("off")
        kind = (
            "reference"
            if fid in REFERENCE
            else f"flat face {row['flat_face_share']:.2f}"
        )
        label_row(axes[r, 0], f"{fid}\n{kind}")
    fig.suptitle(
        "Question 30, cut volumes: is the missing part cut off by one flat plane?"
        " Long sections with the calyx at the bottom; cross-sections at four heights of the"
        " fruit (0 % calyx, 100 % stalk end); outline seen along the stem-calyx axis."
        " Last rows: normal fruit of boxes I and L.",
        fontsize=9,
        wrap=True,
    )
    fig.tight_layout(rect=(0.03, 0, 1, 0.97))
    fig.savefig(out, dpi=110)
    plt.close(fig)


def dark_figure(cfg, table: pl.DataFrame, ids: list[str], out) -> list[dict]:
    fig, axes = plt.subplots(len(ids), 4, figsize=(11, 2.0 * len(ids)), squeeze=False)
    numbers = []
    for r, fid in enumerate(ids):
        row = table.row(by_predicate=pl.col("fruit_id") == fid, named=True)
        v = load(cfg, row)
        i, _, k = centre(v["fruit"])
        brightest = np.unravel_index(np.argmax(v["raw"]), v["raw"].shape)
        in_fruit = bool(v["fruit"][brightest])
        tissue_raw = float(
            np.median(v["raw"][v["fruit"] & (v["scaled"] > MASK_THRESHOLD)])
        )
        numbers.append(
            {
                "fruit_id": fid,
                "figure": "dark",
                "raw_min": float(v["raw"].min()),
                "raw_max": float(v["raw"].max()),
                "tissue_raw_median": tissue_raw,
                "tissue_scaled_median": row["tissue_intensity_median"],
                "brightest_voxel": str(tuple(int(b) for b in brightest)),
                "brightest_in_fruit": in_fruit,
                "stray_voxels": int(v["stray"].sum()),
            }
        )
        show(axes[r, 0], v["scaled"][i], "long section (fixed scale 0 to 1)")
        axes[r, 1].imshow(
            v["scaled"][:, :, k], origin="lower", cmap="gray", vmin=0, vmax=1
        )
        axes[r, 1].set_title("cross-section, middle (fixed scale)", fontsize=7)
        axes[r, 1].axis("off")
        # Maximum projection along axis 0, with the brightest raw voxel marked.
        show(axes[r, 2], v["scaled"].max(axis=0), "maximum projection, axis 0")
        axes[r, 2].plot(brightest[1], brightest[2], "rx", markersize=8, mew=2)
        where = "inside the fruit" if in_fruit else "OUTSIDE the fruit"
        axes[r, 2].set_title(
            f"max projection; x = brightest voxel, {where}", fontsize=7
        )
        axes[r, 3].hist(v["raw"].ravel(), bins=200, color="grey", log=True)
        axes[r, 3].axvline(tissue_raw, color="tab:blue", label="tissue median")
        axes[r, 3].axvline(float(v["raw"].max()), color="tab:red", label="maximum")
        axes[r, 3].set_title("raw intensities (log count)", fontsize=7)
        axes[r, 3].tick_params(labelsize=6)
        axes[r, 3].legend(fontsize=6)
        if fid in DARK_2425:
            kind = "dark"
        elif fid in LOW_2526:
            kind = "slightly low"
        else:
            kind = "normal reference"
        label_row(
            axes[r, 0],
            f"{fid}\n{kind}, tissue {row['tissue_intensity_median']:.2f}",
        )
    fig.suptitle(
        "Question 30, dark volumes: scaled tissue median per fruit at the left (median of all"
        " open fruit: 2425 0.87, 2526 0.85). Same grey scale in every picture. Red x: the"
        " brightest raw voxel, which sets the top of the scaling. Rows: five dark 2425"
        " volumes, two normal 2425 fruit of boxes A and D, three slightly low 2526 volumes,"
        " three normal 2526 fruit of boxes C, I and L.",
        fontsize=9,
        wrap=True,
    )
    fig.tight_layout(rect=(0.03, 0, 1, 0.97))
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return numbers


def stray_figure(cfg, table: pl.DataFrame, ids: list[str], out) -> list[dict]:
    fig, axes = plt.subplots(len(ids), 6, figsize=(12, 2.2 * len(ids)), squeeze=False)
    numbers = []
    for r, fid in enumerate(ids):
        row = table.row(by_predicate=pl.col("fruit_id") == fid, named=True)
        v = load(cfg, row)
        stray = v["stray"]
        s = centre(stray)
        # Shortest distance from a stray voxel to the fruit, in voxels (128³ grid).
        from scipy import ndimage

        to_fruit = ndimage.distance_transform_edt(~v["fruit"])
        gap = float(to_fruit[stray].min())
        numbers.append(
            {
                "fruit_id": fid,
                "figure": "stray",
                "raw_min": float(v["raw"].min()),
                "raw_max": float(v["raw"].max()),
                "stray_voxels": int(stray.sum()),
                "stray_centre": str(s),
                "stray_gap_to_fruit_voxels": gap,
            }
        )
        for a in range(3):
            show(axes[r, a], v["scaled"].max(axis=a), f"max projection, axis {a}")
            axes[r, a].contour(stray.any(axis=a).T, levels=[0.5], colors="red")
        for a in range(3):
            picture = np.take(v["scaled"], s[a], axis=a)
            outline = np.take(stray, s[a], axis=a)
            show(axes[r, 3 + a], picture, f"slice through object, axis {a}")
            if outline.any():
                axes[r, 3 + a].contour(outline.T, levels=[0.5], colors="red")
        label_row(
            axes[r, 0],
            f"{fid}\n{int(stray.sum())} voxels, gap {gap:.0f}",
        )
    fig.suptitle(
        "Question 31, stray objects: voxels above the threshold that are not connected to the"
        " fruit, outlined in red. Maximum projections along each axis, then a slice through"
        " the centre of the stray voxels along each axis. Gap: shortest distance to the"
        " fruit in voxels.",
        fontsize=9,
        wrap=True,
    )
    fig.tight_layout(rect=(0.03, 0, 1, 0.96))
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return numbers


def main() -> None:
    cfg = load_config()
    table = open_fruit(cfg)
    wanted = set(CUT + DARK_ORDER + STRAY + REFERENCE)
    not_open = sorted(wanted - set(table["fruit_id"]))
    if not_open:
        raise ValueError(f"not open fruit (sealed or absent), not read: {not_open}")
    folder = cfg.path("derived_dir") / "figures"
    folder.mkdir(parents=True, exist_ok=True)

    cut_figure(cfg, table, CUT + REFERENCE, folder / "meeting_q30_cut_volumes.png")
    numbers = dark_figure(
        cfg, table, DARK_ORDER, folder / "meeting_q30_dark_volumes.png"
    )
    numbers += stray_figure(cfg, table, STRAY, folder / "meeting_q31_stray_objects.png")
    summary = pl.DataFrame(numbers, infer_schema_length=None)
    summary.write_csv(folder / "meeting_q30_q31_numbers.csv")
    with pl.Config(tbl_rows=30, tbl_cols=12, tbl_width_chars=160):
        print(summary)
    print(f"saved 3 figures and meeting_q30_q31_numbers.csv to {folder}")


if __name__ == "__main__":
    main()
