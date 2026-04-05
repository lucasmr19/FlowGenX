from src.representations.vision import (
    NprintRepresentation, NprintConfig
)
from src.utils.make_synthetic_flow import make_synthetic_flow


def test_nprint_binary_output():
    rep = NprintRepresentation(NprintConfig(max_packets=5))
    flows = [make_synthetic_flow()]
    rep.fit(flows)

    tensor = rep.encode(flows[0])
    unique = tensor.unique().tolist()

    assert all(v in (0.0, 1.0) for v in unique)