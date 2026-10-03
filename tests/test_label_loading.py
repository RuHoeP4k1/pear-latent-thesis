import polars as pl
import pytest

from pearlatent.config import Config
from pearlatent.labels import fruit_keys, load_labels, read_label_file
from pearlatent.volumes import list_volumes


def test_fruit_keys_2425_storage_groups_and_boxes():
    keys = fruit_keys(
        pl.Series(["A01", "A30", "A31", "J60", "G_opt01", "I_opt30"]), "2425"
    )
    assert keys["fruit_id"].to_list() == [
        "2425_A01",
        "2425_A30",
        "2425_A31",
        "2425_J60",
        "2425_G_opt01",
        "2425_I_opt30",
    ]
    # Optimal-storage fruit join the box of their letter (decision of 3 October 2026)
    assert keys["box"].to_list() == [
        "2425_A",
        "2425_A",
        "2425_A",
        "2425_J",
        "2425_G",
        "2425_I",
    ]
    assert keys["storage"].to_list() == [
        "suboptimal_storage",
        "suboptimal_storage",
        "harvest",
        "harvest",
        "optimal_storage",
        "optimal_storage",
    ]


def test_fruit_keys_2526():
    keys = fruit_keys(pl.Series(["A01", "O30"]), "2526")
    assert keys["fruit_id"].to_list() == ["2526_A01", "2526_O30"]
    assert keys["box"].to_list() == ["2526_A", "2526_O"]
    assert keys["storage"].to_list() == ["after_storage", "after_storage"]


@pytest.mark.parametrize(
    ("stem", "season"),
    [
        ("K01", "2425"),
        ("A61", "2425"),
        ("A_opt01", "2425"),
        ("P01", "2526"),
        ("A31", "2526"),
        ("pear_A01", "2526"),
    ],
)
def test_fruit_keys_rejects_unexpected_names(stem, season):
    with pytest.raises(ValueError, match="do not match"):
        fruit_keys(pl.Series([stem]), season)


def test_fruit_keys_rejects_unknown_season():
    with pytest.raises(ValueError, match="season"):
        fruit_keys(pl.Series(["A01"]), "2324")


@pytest.fixture
def label_files(tmp_path):
    p2425 = tmp_path / "l2425.csv"
    p2425.write_text(
        "filename,rot,browning,cavity,defective,non-consumable\n"
        "A01.nii,0,2,0,2,1\n"
        "A31.nii,0,0,1,1,0\n"
        "G_opt01.nii,1,0,0,0,0\n"
    )
    p2526 = tmp_path / "l2526.csv"
    p2526.write_text(
        "filename,browning,cavity,defective,binary_123,binary_23\n"
        "A01,1,3,3,1,1\n"
        "B02,0,0,0,0,0\n"
    )
    return tmp_path, p2425, p2526


def test_read_label_file_strips_extension_and_keeps_columns(label_files):
    _, p2425, _ = label_files
    df = read_label_file(p2425, "2425")
    assert df["fruit_id"].to_list() == ["2425_A01", "2425_A31", "2425_G_opt01"]
    assert {"rot", "defective", "non-consumable", "box", "storage"} <= set(df.columns)
    assert df.schema["browning"] == pl.Int8


def test_read_label_file_rejects_missing_column(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("filename,browning\nA01,0\n")
    with pytest.raises(ValueError, match="cavity"):
        read_label_file(p, "2526")


def test_read_label_file_rejects_duplicate_fruit(tmp_path):
    p = tmp_path / "dup.csv"
    p.write_text(
        "filename,browning,cavity,defective,binary_123,binary_23\n"
        "A01,0,0,0,0,0\nA01.nii,0,0,0,0,0\n"
    )
    with pytest.raises(ValueError, match="more than once"):
        read_label_file(p, "2526")


def test_read_label_file_rejects_out_of_range(tmp_path):
    p = tmp_path / "range.csv"
    p.write_text(
        "filename,browning,cavity,defective,binary_123,binary_23\nA01,4,0,4,1,1\n"
    )
    with pytest.raises(ValueError, match="outside"):
        read_label_file(p, "2526")


def test_load_labels_combines_seasons_on_common_columns(label_files):
    root, p2425, p2526 = label_files
    cfg = Config(
        raw={"paths": {"labels_2425": str(p2425), "labels_2526": str(p2526)}}, root=root
    )
    df = load_labels(cfg)
    assert df.columns == [
        "fruit_id",
        "season",
        "box",
        "storage",
        "file_stem",
        "browning",
        "cavity",
    ]
    assert df.height == 5
    assert df["fruit_id"].is_unique().all()
    # A01 occurs in both seasons and must stay two different fruit
    assert df.filter(pl.col("file_stem") == "A01").height == 2


def test_list_volumes(tmp_path):
    for name in ["A01.nii", "B02.nii.gz", "notes.txt"]:
        (tmp_path / name).write_bytes(b"")
    vols = list_volumes(tmp_path, "2526")
    assert vols["fruit_id"].to_list() == ["2526_A01", "2526_B02"]
    assert vols["box"].to_list() == ["2526_A", "2526_B"]


def test_list_volumes_rejects_same_fruit_twice(tmp_path):
    for name in ["A01.nii", "A01.nii.gz"]:
        (tmp_path / name).write_bytes(b"")
    with pytest.raises(ValueError, match="more than once"):
        list_volumes(tmp_path, "2526")
