""" 
Funciones para crear flujos sintéticos de prueba. No requieren Scapy ni archivos PCAP, 
lo que los hace ideales para testing rápido.
"""

import numpy as np
from typing import List
from ..preprocessing import Flow, ParsedPacket


def make_synthetic_flow(
    n_packets: int = 20,
    protocol:  int = 6,   # TCP por defecto
    seed:      int = 0,
) -> Flow:
    """Crea un flujo sintético coherente con ParsedPacket."""

    rng = np.random.default_rng(seed)

    src_ip = f"192.168.{rng.integers(0,256)}.{rng.integers(1,255)}"
    dst_ip = f"10.0.{rng.integers(0,256)}.{rng.integers(1,255)}"
    sport = int(rng.integers(1024, 65535))
    dport = 443

    flow = Flow(
        flow_id  = f"test_flow_{seed}",
        src_ip   = src_ip,
        dst_ip   = dst_ip,
        sport    = sport,
        dport    = dport,
        protocol = protocol,
        start_time = 1000.0,
    )

    ts = flow.start_time

    for _ in range(n_packets):
        # --- timing ---
        iat = float(rng.exponential(0.01))
        ts += iat

        # --- payload ---
        payload_len = int(rng.integers(20, 1400))
        payload = rng.integers(0, 256, size=payload_len, dtype=np.uint8).tobytes()

        # --- IP total length (IPv4: header 20 bytes) ---
        ip_total_len = payload_len + 20

        # --- crear paquete ---
        pkt = ParsedPacket(
            timestamp=ts,
            iat=iat,

            # -------------------------
            # IPv4 (representación real)
            # -------------------------
            ipv4_src=src_ip,
            ipv4_dst=dst_ip,
            ipv4_ttl=64,
            ipv4_proto=protocol,
            ipv4_tl=ip_total_len,

            # -------------------------
            # Transporte
            # -------------------------
            sport=sport,
            dport=dport,

            payload_bytes=payload,
            payload_len=payload_len,
        )

        # -------------------------
        # Coherencia por protocolo
        # -------------------------
        if protocol == 6:  # TCP
            pkt.tcp_sprt = sport
            pkt.tcp_dprt = dport
            pkt.tcp_syn = 1 if rng.random() < 0.1 else 0
            pkt.tcp_ackf = 1 if rng.random() < 0.7 else 0

        elif protocol == 17:  # UDP
            pkt.udp_sport = sport
            pkt.udp_dport = dport
            pkt.udp_len = payload_len + 8

        # ICMP u otros podrían añadirse aquí

        flow.packets.append(pkt)

    flow.end_time = ts
    return flow


def make_dataset(n_flows: int = 50) -> List[Flow]:
    return [make_synthetic_flow(n_packets=20, seed=i) for i in range(n_flows)]