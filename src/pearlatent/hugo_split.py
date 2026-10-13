"""Reproduce Hugo's train / validation / test split of the 2425 fruit used to train his VAE.

The preprint (section 2.5) says only "random 80/10/10". The split follows from his code and seed:
`train_vae.py` builds `CsvVolumeDataset` (one sample per CSV row with a non-empty filename, in
file order) and calls `utils.splits.split_train_val_test(dataset, val_split=0.1, test_split=0.1,
random_state)`. Seed 369 (Hugo, 13 October 2026; also in his `slurm/train_vae.slurm` and stored in
`best.pth` under `random_state`).

Hugo's function is imported from the separate clone at the pinned commit, not copied (his
repository has no licence file).

# UNVERIFIED: that our `labels_conference_pear_2425.csv` has the same rows in the same order as
# the `labels.csv` Hugo trained with. A different row order gives a different list from the same
# seed. Hugo confirms the list (handoff of 13 October, B4; HANDOFF.md open question 19).
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import polars as pl

from pearlatent.encoding import PINNED_COMMIT, hugo_commit
from pearlatent.labels import read_label_file

# Settings of Hugo's run (train_vae.py defaults for the two fractions; seed from Hugo).
HUGO_VAE_SEED = 369
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1
SPLIT_NAMES = ("train", "validation", "test")


def import_hugo_split_function(repo: str | Path):
    """Return Hugo's `split_train_val_test` from his clone, after checking the pinned commit."""
    repo = Path(repo)
    commit = hugo_commit(repo)
    if not commit.startswith(PINNED_COMMIT):
        raise RuntimeError(
            f"{repo} is at commit {commit[:7]}, expected {PINNED_COMMIT}. "
            f"Run `git -C {repo} checkout {PINNED_COMMIT}`."
        )
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from utils.splits import split_train_val_test

    return split_train_val_test


def hugo_split_rows(
    n_rows: int,
    repo: str | Path,
    random_state: int = HUGO_VAE_SEED,
) -> dict[str, list[int]]:
    """Row positions (0-based, in CSV file order) of Hugo's train, validation and test parts.

    In: the number of dataset rows, Hugo's clone, the seed. Out: {"train": [...],
    "validation": [...], "test": [...]}.

    How the seed becomes a list, step by step (torch.utils.data.random_split):
    1. Hugo's function makes one generator, `torch.Generator().manual_seed(random_state)`.
    2. The sizes are `int(n * 0.1)` for validation and for test, the rest for training
       (660 rows: 66, 66, 528).
    3. The first `random_split(dataset, [528, 132], generator)` draws one random permutation of
       0 .. n-1 with `torch.randperm(n, generator)`; the first 528 entries are the training rows,
       the next 132 the validation-plus-test rows. Only `len(dataset)` is used, never the
       volumes, so a plain list of row numbers gives the same result as the real dataset.
    4. The second `random_split(val_test, [66, 66], generator)` uses the same generator, which
       has moved on after step 3, to permute the 132 positions inside the validation-plus-test
       part; the first 66 are validation, the last 66 test. These positions refer to the
       validation-plus-test part, so they are mapped back to the original row numbers.
    The result depends on n, the two fractions, the seed and the order of the rows.
    """
    split_train_val_test = import_hugo_split_function(repo)
    # The "dataset" is the list of row numbers itself, so each Subset holds row numbers.
    row_numbers = list(range(n_rows))
    train, validation, test = split_train_val_test(
        row_numbers,
        val_split=VAL_SPLIT,
        test_split=TEST_SPLIT,
        random_state=random_state,
    )
    # Indexing a Subset returns the element of the underlying list, here the row number;
    # for the nested Subsets of step 4 this resolves through both levels.
    parts = {
        "train": [train[i] for i in range(len(train))],
        "validation": [validation[i] for i in range(len(validation))],
        "test": [test[i] for i in range(len(test))],
    }
    return parts


def file_sha256(path: str | Path) -> str:
    """SHA-256 of a file, recorded so the label file the split was made from is known."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def hugo_vae_split_2425(
    labels_csv: str | Path,
    repo: str | Path,
    random_state: int = HUGO_VAE_SEED,
) -> pl.DataFrame:
    """One row per 2425 fruit: `fruit_id, filename, row, hugo_split`.

    In: our 2425 label CSV and Hugo's clone. Rows are kept in file order, as Hugo's
    `CsvVolumeDataset` reads them; `row` is the 0-based position among rows with a filename.
    """
    labels = read_label_file(labels_csv, "2425")
    n_in = labels.height
    # CsvVolumeDataset skips rows whose filename is empty; our reader would already have
    # failed on such a row (fruit_keys), so this filter is only a check.
    labels = labels.filter(pl.col("file_stem").str.strip_chars() != "")
    print(f"2425 label rows read: {n_in}; with a filename: {labels.height}")

    parts = hugo_split_rows(labels.height, repo, random_state)
    row_to_split = {row: name for name, rows in parts.items() for row in rows}
    out = labels.select(
        "fruit_id",
        (pl.col("file_stem") + ".nii").alias("filename"),
        pl.int_range(pl.len(), dtype=pl.Int32).alias("row"),
    ).with_columns(
        pl.col("row")
        .replace_strict(row_to_split, return_dtype=pl.String)
        .alias("hugo_split")
    )
    for name in SPLIT_NAMES:
        print(f"  {name}: {out.filter(pl.col('hugo_split') == name).height} fruit")
    return out


def write_hugo_vae_split(cfg) -> Path:
    """Write `derived/hugo_vae_split_2425/hugo_vae_split_2425.csv` and its `run.json`.

    A folder rather than a single file, so the run record sits next to the table it describes.
    Returns the path of the CSV file.
    """
    from pearlatent.runs import write_run_record

    labels_csv = cfg.path("labels_2425")
    repo = cfg.path("hugo_repo")
    split = hugo_vae_split_2425(labels_csv, repo)
    out_dir = cfg.path("derived_dir") / "hugo_vae_split_2425"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "hugo_vae_split_2425.csv"
    split.write_csv(out_csv)
    write_run_record(
        out_dir,
        {
            "what": "Hugo's VAE split of the 2425 fruit, reproduced from his code",
            "random_state": HUGO_VAE_SEED,
            "val_split": VAL_SPLIT,
            "test_split": TEST_SPLIT,
            "hugo_commit": hugo_commit(repo),
            "labels_csv": Path(labels_csv).name,
            "labels_csv_sha256": file_sha256(labels_csv),
            "n_fruit": split.height,
        },
    )
    print(f"written: {out_csv}")
    return out_csv


if __name__ == "__main__":
    from pearlatent.config import load_config

    write_hugo_vae_split(load_config())
