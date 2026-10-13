import numpy as np
import polars as pl
import pytest

torch = pytest.importorskip("torch")
nib = pytest.importorskip("nibabel")

import pearlatent.encode_run as encode_run  # noqa: E402
from pearlatent.encode_run import assemble, chunk_bounds, encode_in_chunks  # noqa: E402


class _TinyEncoder(torch.nn.Module):
    """Same attribute names as Hugo's Encoder: feature_layers, then mu_head."""

    def __init__(self):
        super().__init__()
        self.feature_layers = torch.nn.Sequential(
            torch.nn.Conv3d(1, 2, kernel_size=4, stride=4), torch.nn.ReLU()
        )
        self.mu_head = torch.nn.Conv3d(2, 1, kernel_size=1)


class _TinyVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = _TinyEncoder()

    def decode(self, mu):
        return torch.nn.functional.interpolate(mu, scale_factor=4)


@pytest.fixture
def volumes(tmp_path, monkeypatch):
    # Hugo's metrics need his clone; replace them by a stand-in for this test.
    monkeypatch.setattr(
        encode_run,
        "whole_volume_metrics",
        lambda x, x_hat: {"mae_volume": float((x - x_hat).abs().mean())},
    )
    rng = np.random.default_rng(0)
    rows = []
    for i in range(5):
        vol = np.zeros((128, 128, 128), dtype=np.uint16)
        vol[40:90, 40:90, 20:110] = rng.integers(500, 1000, size=(50, 50, 90))
        path = tmp_path / f"A{i:02d}.nii"
        nib.save(nib.Nifti1Image(vol, np.eye(4)), path)
        rows.append(
            {"fruit_id": f"2526_A{i:02d}", "path": str(path), "decode": i % 2 == 0}
        )
    return pl.DataFrame(rows)


def _run(vae, volumes, out_dir, max_chunks=None):
    out_dir.mkdir(exist_ok=True)
    return encode_in_chunks(
        vae, volumes, out_dir, "cpu", batch_size=2, chunk_size=2, max_chunks=max_chunks
    )


def test_chunk_bounds():
    assert chunk_bounds(5, 2) == [(0, 2), (2, 4), (4, 5)]


def test_stopped_and_resumed_equals_uninterrupted(volumes, tmp_path):
    torch.manual_seed(0)
    vae = _TinyVAE().eval()
    whole, resumed = tmp_path / "whole", tmp_path / "resumed"
    assert _run(vae, volumes, whole)
    # Stop after one chunk, then continue: only the missing chunks are computed.
    assert not _run(vae, volumes, resumed, max_chunks=1)
    assert len(list((resumed / "parts").glob("chunk_*.npz"))) == 1
    assert _run(vae, volumes, resumed)
    for out in (whole, resumed):
        assemble(volumes, out, chunk_size=2)
    for name in ("latents.npy", "masks.npy", "planes.npy", "slice_mae_body.npy"):
        np.testing.assert_array_equal(np.load(whole / name), np.load(resumed / name))
    recon = pl.read_parquet(resumed / "reconstruction.parquet")
    assert recon["fruit_id"].to_list() == ["2526_A00", "2526_A02", "2526_A04"]
    manifest = pl.read_parquet(resumed / "manifest.parquet")
    assert manifest["recon_row"].to_list() == [0, None, 1, None, 2]
    assert np.load(resumed / "planes.npy").shape == (3, 2, 3, 128, 128)


def test_resume_refuses_other_fruit(volumes, tmp_path):
    vae = _TinyVAE().eval()
    out = tmp_path / "run"
    _run(vae, volumes, out, max_chunks=1)
    with pytest.raises(RuntimeError, match="other fruit"):
        _run(vae, volumes.head(4), out)
