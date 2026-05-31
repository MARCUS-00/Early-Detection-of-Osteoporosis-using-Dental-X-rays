"""
Tests for UNet forward pass — synthetic tensors only, no dataset files or weights.
"""
import pytest
import torch

from osteo.segmentation.unet import UNet


def test_output_shape_default():
    """Default UNet: (1,3,100,100) input -> (1,1,100,100) output."""
    model = UNet()
    model.eval()
    x = torch.zeros(1, 3, 100, 100)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 1, 100, 100), (
        f"Expected (1,1,100,100), got {tuple(out.shape)}"
    )


def test_output_values_in_unit_interval():
    """All output values must lie in [0, 1] (sigmoid is applied in forward)."""
    model = UNet()
    model.eval()
    x = torch.randn(1, 3, 100, 100)
    with torch.no_grad():
        out = model(x)
    assert float(out.min()) >= 0.0, f"Min value {float(out.min())} < 0"
    assert float(out.max()) <= 1.0, f"Max value {float(out.max())} > 1"


def test_output_shape_batch():
    """Batch of 2 images -> batch of 2 masks."""
    model = UNet()
    model.eval()
    x = torch.zeros(2, 3, 100, 100)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 1, 100, 100)


def test_output_shape_custom_features():
    """Custom feature list [32, 64] -> output still (1,1,H,W)."""
    model = UNet(in_channels=3, out_channels=1, features=[32, 64])
    model.eval()
    x = torch.zeros(1, 3, 64, 64)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 1, 64, 64)


def test_output_shape_non_square():
    """Non-square spatial input: (1,3,128,96) -> (1,1,128,96)."""
    model = UNet()
    model.eval()
    x = torch.zeros(1, 3, 128, 96)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 1, 128, 96)
