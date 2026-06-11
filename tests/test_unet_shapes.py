'''Tests for UNet forward pass — synthetic tensors only, no files needed.'''
import pytest
import torch
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from osteo.segmentation.unet import UNet


def test_output_shape_default():
    model = UNet(); model.eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 3, 100, 100))
    assert out.shape == (1, 1, 100, 100)

def test_output_values_in_unit_interval():
    model = UNet(); model.eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, 100, 100))
    assert float(out.min()) >= 0.0
    assert float(out.max()) <= 1.0

def test_output_shape_batch():
    model = UNet(); model.eval()
    with torch.no_grad():
        out = model(torch.zeros(2, 3, 100, 100))
    assert out.shape == (2, 1, 100, 100)

def test_output_shape_custom_features():
    model = UNet(in_channels=3, out_channels=1, features=[32, 64]); model.eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 3, 64, 64))
    assert out.shape == (1, 1, 64, 64)

def test_output_shape_non_square():
    model = UNet(); model.eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 3, 128, 96))
    assert out.shape == (1, 1, 128, 96)
