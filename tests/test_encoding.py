import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pearlatent.encoding import encode_volumes  # noqa: E402


class _TinyEncoder(torch.nn.Module):
    """Same attribute names as Hugo's Encoder: feature_layers, then mu_head."""

    def __init__(self):
        super().__init__()
        self.feature_layers = torch.nn.Sequential(
            torch.nn.Conv3d(1, 3, kernel_size=4, stride=4), torch.nn.ReLU()
        )
        self.mu_head = torch.nn.Conv3d(3, 1, kernel_size=3, padding=1)


def test_encode_volumes_shapes_and_mask():
    torch.manual_seed(0)
    enc = _TinyEncoder().eval()
    vols = np.random.default_rng(0).random((2, 16, 16, 16), dtype=np.float32)
    masks = np.zeros((2, 4, 4, 4), dtype=bool)
    masks[0, :2] = True
    masks[1] = True
    mu, pooled = encode_volumes(enc, vols, masks, "cpu")
    assert mu.shape == (2, 4, 4, 4)
    assert pooled.shape == (2, 2, 3)
    # The pooled mean equals the mean of the feature layer over the masked cells
    feats = enc.feature_layers(torch.from_numpy(vols).unsqueeze(1)).detach().numpy()
    expected = feats[0][:, masks[0]].mean(axis=1)
    np.testing.assert_allclose(pooled[0, 0], expected, rtol=1e-5)
