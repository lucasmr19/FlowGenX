import pytest
from src.representations.sequential.tokenizer import (
    FlatTokenizer, SequentialConfig
)
from src.utils.make_synthetic_flow import make_synthetic_flow


def test_encode_without_fit_raises():
    rep = FlatTokenizer(SequentialConfig(max_length=32))
    flow = make_synthetic_flow()

    with pytest.raises(RuntimeError):
        rep.encode(flow)


def test_padding_behavior():
    rep = FlatTokenizer(SequentialConfig(max_length=16))
    flows = [make_synthetic_flow()]
    rep.fit(flows)

    tensor = rep.encode(flows[0])
    assert tensor.shape[0] == 16