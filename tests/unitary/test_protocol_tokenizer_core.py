from src.representations.sequential.tokenizer import (
    ProtocolAwareTokenizer, ProtocolAwareConfig
)
from src.utils.make_synthetic_flow import make_synthetic_flow


def test_tcp_state_encoding():
    rep = ProtocolAwareTokenizer(
        ProtocolAwareConfig(max_length=64, encode_tcp_state=True)
    )

    flows = [make_synthetic_flow()]
    rep.fit(flows)
    tokens = rep.decode(rep.encode(flows[0]))

    assert any("tcp_flag" in t or "SYN" in t for t in tokens)