import torch
from src.representations.vision import (
    GASFRepresentation, GASFConfig
)
from src.utils.make_synthetic_flow import make_synthetic_flow


def test_gasf_output_range():
    rep = GASFRepresentation(GASFConfig(image_size=16, n_steps=16))
    flows = [make_synthetic_flow()]
    rep.fit(flows)

    tensor = rep.encode(flows[0])

    assert torch.all(tensor <= 1.0)
    assert torch.all(tensor >= -1.0)