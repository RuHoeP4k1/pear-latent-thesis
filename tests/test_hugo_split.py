from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from pearlatent.hugo_split import HUGO_VAE_SEED, hugo_split_rows  # noqa: E402


def _hugo_repo() -> Path:
    """Hugo's clone from config/local.toml; skip the test on a machine without it."""
    try:
        from pearlatent.config import load_config

        repo = load_config().path("hugo_repo")
    except Exception as err:  # no config/local.toml on this machine
        pytest.skip(f"no config: {err}")
    if not (Path(repo) / "utils" / "splits.py").exists():
        pytest.skip(f"Hugo's clone not found at {repo}")
    return Path(repo)


def test_sizes_for_660_rows():
    parts = hugo_split_rows(660, _hugo_repo())
    assert {name: len(rows) for name, rows in parts.items()} == {
        "train": 528,
        "validation": 66,
        "test": 66,
    }


def test_every_row_once():
    parts = hugo_split_rows(660, _hugo_repo())
    all_rows = parts["train"] + parts["validation"] + parts["test"]
    assert sorted(all_rows) == list(range(660))


def test_deterministic_and_seed_dependent():
    repo = _hugo_repo()
    assert hugo_split_rows(660, repo) == hugo_split_rows(660, repo)
    assert hugo_split_rows(660, repo, random_state=1) != hugo_split_rows(660, repo)


def test_matches_two_randperm_draws():
    """Independent check of the docstring: one generator, two permutations in sequence."""
    generator = torch.Generator().manual_seed(HUGO_VAE_SEED)
    first = torch.randperm(660, generator=generator).tolist()
    train, val_test = first[:528], first[528:]
    second = torch.randperm(132, generator=generator).tolist()
    validation = [val_test[i] for i in second[:66]]
    test = [val_test[i] for i in second[66:]]
    assert hugo_split_rows(660, _hugo_repo()) == {
        "train": train,
        "validation": validation,
        "test": test,
    }
