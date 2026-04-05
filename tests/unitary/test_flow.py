import pytest
from src.data_utils.preprocessing import Flow, ParsedPacket


def test_flow_initialization():
    flow = Flow(
        flow_id="f1",
        src_ip="1.1.1.1",
        dst_ip="2.2.2.2",
        sport=1234,
        dport=80,
        protocol=6,
        start_time=0.0,
    )

    assert flow.flow_id == "f1"
    assert flow.packets == []


def test_flow_packet_append():
    flow = Flow(
        flow_id="f1",
        src_ip="1.1.1.1",
        dst_ip="2.2.2.2",
        sport=1234,
        dport=80,
        protocol=6,
        start_time=0.0,
    )

    pkt = ParsedPacket(
        timestamp=0.1,
        iat=0.1,
        ip_src="1.1.1.1",
        ip_dst="2.2.2.2",
        ip_version=4,
        ip_ttl=64,
        ip_proto=6,
        ip_len=100,
        sport=1234,
        dport=80,
        tcp_flags=0x10,
        tcp_win=1024,
        payload_bytes=b"\x00",
        payload_len=1,
        direction=0,
        flow_id="f1",
    )

    flow.packets.append(pkt)
    flow.end_time = 0.1

    assert len(flow.packets) == 1
    assert flow.packets[0].ip_proto == 6
    assert flow.end_time == 0.1