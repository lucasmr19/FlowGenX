"""
tests/unitary/test_gaf_representation.py

Unit tests for GAFRepresentation integrated with Flow/PacketWindow.
"""

from __future__ import annotations
import numpy as np
import torch
import pytest

from src.representations.vision.gaf import GAFConfig, GAFRepresentation

# ---------------------------------------------------------------------
# Minimal synthetic Flow / PacketWindow for tests
# ---------------------------------------------------------------------

class DummyParsedPacket:
    def __init__(self, payload_len: int = 100, iat: float = 1.0):
        self.payload_len = payload_len
        self.iat = iat

class DummyFlow:
    def __init__(self, packets):
        self.packets = packets

def make_dummy_flow(n_packets: int = 32, start_payload: int = 100, payload_step: int = 5):
    packets = [DummyParsedPacket(payload_len=start_payload + i * payload_step, iat=1.0 + i)
               for i in range(n_packets)]
    return DummyFlow(packets)

# ---------------------------------------------------------------------
# 1) Encode and output shape tests
# ---------------------------------------------------------------------

def test_encode_produces_tensor_and_correct_shape():
    flow = make_dummy_flow()
    cfg = GAFConfig(image_size=32, field_name="payload_len")
    rep = GAFRepresentation(cfg)
    rep.fit([flow])

    out = rep.encode(flow)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (1, 32, 32)
    assert torch.isfinite(out).all()

def test_encode_respects_field_name_configuration():
    flow = make_dummy_flow()
    # encode iat instead of payload_len
    cfg = GAFConfig(image_size=32, field_name="iat")
    rep = GAFRepresentation(cfg)
    rep.fit([flow])

    out = rep.encode(flow)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (1, 32, 32)
    assert torch.isfinite(out).all()

# ---------------------------------------------------------------------
# 2) Binning
# ---------------------------------------------------------------------

def test_sum_binning_reduces_length_and_sums_blocks():
    flow = make_dummy_flow(n_packets=8, start_payload=1, payload_step=1)
    rep = GAFRepresentation(GAFConfig(image_size=2, field_name="payload_len", use_binning=True, bin_size=4))
    rep.fit([flow])

    ts = np.array([p.payload_len for p in flow.packets], dtype=np.float32)
    binned = rep._sum_binning(ts, bin_size=4)
    assert binned.shape == (2,)
    assert np.allclose(binned, np.array([10.0, 26.0], dtype=np.float32), atol=1e-6)

# ---------------------------------------------------------------------
# 3) Gamma and rescale
# ---------------------------------------------------------------------

def test_gamma_and_rescale_behavior():
    flow = make_dummy_flow()
    rep = GAFRepresentation(GAFConfig(image_size=32, field_name="payload_len", gamma=0.25, rescale_to_01=True))
    rep.fit([flow])

    img = rep.encode(flow)[0].numpy()
    assert img.min() >= 0.0 and img.max() <= 1.0
    # for gamma < 1, values are pushed upward on average
    rep_no_gamma = GAFRepresentation(GAFConfig(image_size=32, field_name="payload_len", gamma=None, rescale_to_01=True))
    rep_no_gamma.fit([flow])
    img_no_gamma = rep_no_gamma.encode(flow)[0].numpy()
    assert img.mean() >= img_no_gamma.mean() - 1e-6

# ---------------------------------------------------------------------
# 4) Variance sanity
# ---------------------------------------------------------------------

def test_output_not_constant_on_synthetic_data():
    flow = make_dummy_flow(n_packets=32, start_payload=100, payload_step=5)
    rep = GAFRepresentation(GAFConfig(image_size=32, field_name="payload_len", gamma=0.25, rescale_to_01=True))
    rep.fit([flow])
    img = rep.encode(flow)[0].numpy()
    assert img.std() > 0.01
    assert not np.allclose(img, 1.0)
    assert np.all(np.isfinite(img))

# ---------------------------------------------------------------------
# 5) Output shape property
# ---------------------------------------------------------------------

def test_output_shape_property_matches_configuration():
    rep1 = GAFRepresentation(GAFConfig(image_size=32, field_name="payload_len"))
    rep2 = GAFRepresentation(GAFConfig(image_size=32, field_name="iat"))
    rep3 = GAFRepresentation(GAFConfig(image_size=32, field_name="payload_len"))  # single feature
    assert rep1.output_shape == (1, 32, 32)
    assert rep2.output_shape == (1, 32, 32)
    assert rep3.output_shape == (1, 32, 32)

# ---------------------------------------------------------------------
# 6) Method validation
# ---------------------------------------------------------------------

def test_invalid_method_raises_value_error():
    with pytest.raises(ValueError):
        GAFRepresentation(GAFConfig(image_size=32, field_name="payload_len", method="invalid")).encode(make_dummy_flow())