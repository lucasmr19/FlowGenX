"""
src/reconstruction/heuristics.py
=================================
Utilidades puras reutilizables entre reconstructores.

Todas las funciones son PURAS (sin estado) y operan sobre tipos primitivos
o listas de ParsedPacket. No importan clases concretas de representación.

Categorías:
  - Segmentación de bytes en paquetes
  - Asignación de puertos y protocolos plausibles
  - Inferencia de flags TCP
  - Generación de timestamps sintéticos
  - Estimación de payload a partir de longitud
  - Constantes y helpers para representación nprint
"""

from __future__ import annotations

import random
import time
from typing import List, Optional, Tuple

from src.data_utils.preprocessing import ParsedPacket


_COMMON_TCP_PORTS = [80, 443, 8080, 22, 25, 110, 143, 3306, 5432]
_COMMON_UDP_PORTS = [53, 67, 68, 123, 161, 5060]

_TYPICAL_PACKET_SIZES = [40, 52, 64, 128, 256, 512, 1024, 1460, 1500]

TCP_FLAG_FIN = 0x01
TCP_FLAG_SYN = 0x02
TCP_FLAG_RST = 0x04
TCP_FLAG_PSH = 0x08
TCP_FLAG_ACK = 0x10
TCP_FLAG_URG = 0x20

# ---------------------------------------------------------------------------
# Campos nprint canónicos
# Orden fijo: cada índice de columna en la representación imagen → nombre de campo.
# ---------------------------------------------------------------------------
_NPRINT_FIELDS: List[str] = [
    "ip_ver", "ip_ihl", "ip_tos", "ip_len", "ip_id",
    "ip_flg", "ip_off", "ip_ttl", "ip_pro", "ip_sum",
    "ip_src", "ip_dst",
    "tcp_sport", "tcp_dport", "tcp_seq", "tcp_ack", "tcp_doff", "tcp_res",
    "tcp_fin", "tcp_syn", "tcp_rst", "tcp_psh", "tcp_ack_flag", "tcp_urg",
    "tcp_win", "tcp_sum", "tcp_urp",
    "udp_sport", "udp_dport", "udp_len", "udp_sum",
    "payload",
]


def segment_bytes_into_packets(
    raw_bytes: bytes,
    max_payload: int = 1460,
    min_payload: int = 1,
    seed: Optional[int] = None,
) -> List[bytes]:
    rng = random.Random(seed)
    segments: List[bytes] = []
    offset = 0
    while offset < len(raw_bytes):
        size = rng.randint(min_payload, max_payload)
        chunk = raw_bytes[offset: offset + size]
        if chunk:
            segments.append(chunk)
        offset += size
    return segments


def tokens_to_bytes(tokens: List[int], vocab_size: int = 256) -> bytes:
    if vocab_size <= 256:
        return bytes([min(max(int(t), 0), 255) for t in tokens])
    return bytes([int(round(t * 255 / (vocab_size - 1))) for t in tokens])


def bytes_to_tokens(raw: bytes, vocab_size: int = 256) -> List[int]:
    if vocab_size <= 256:
        return [int(b) for b in raw]
    return [int(round(b * (vocab_size - 1) / 255)) for b in raw]


def infer_protocol_from_port(dport: int) -> int:
    udp_ports = {53, 67, 68, 123, 161, 162, 5060, 5061}
    return 17 if dport in udp_ports else 6


def assign_synthetic_ports(
    proto: int = 6,
    seed: Optional[int] = None,
) -> Tuple[int, int]:
    rng = random.Random(seed)
    if proto == 17:
        dport = rng.choice(_COMMON_UDP_PORTS)
    else:
        dport = rng.choice(_COMMON_TCP_PORTS)
    sport = rng.randint(49152, 65535)
    return sport, dport


def assign_synthetic_ips(
    seed: Optional[int] = None,
) -> Tuple[str, str]:
    rng = random.Random(seed)
    src = f"10.{rng.randint(0,254)}.{rng.randint(0,254)}.{rng.randint(1,254)}"
    dst = f"10.{rng.randint(0,254)}.{rng.randint(0,254)}.{rng.randint(1,254)}"
    return src, dst


def infer_tcp_flags(
    packet_index: int,
    total_packets: int,
    has_data: bool = True,
) -> int:
    if total_packets == 1:
        return TCP_FLAG_PSH | TCP_FLAG_ACK

    if packet_index == 0:
        return TCP_FLAG_SYN
    if packet_index == 1:
        return TCP_FLAG_SYN | TCP_FLAG_ACK
    if packet_index == 2:
        return TCP_FLAG_ACK
    if packet_index == total_packets - 1:
        return TCP_FLAG_FIN | TCP_FLAG_ACK
    return (TCP_FLAG_PSH | TCP_FLAG_ACK) if has_data else TCP_FLAG_ACK


def recompose_tcp_flags_from_fields(fields: dict) -> int:
    """
    Recompone el byte de flags TCP a partir de un dict con claves individuales.
    Claves esperadas (presencia = flag activo):
        tcp_fin, tcp_syn, tcp_rst, tcp_psh, tcp_ack, tcp_urg
    """
    mapping = {
        "tcp_fin": TCP_FLAG_FIN,
        "tcp_syn": TCP_FLAG_SYN,
        "tcp_rst": TCP_FLAG_RST,
        "tcp_psh": TCP_FLAG_PSH,
        "tcp_ack": TCP_FLAG_ACK,
        "tcp_urg": TCP_FLAG_URG,
    }
    result = 0
    for key, mask in mapping.items():
        if fields.get(key, 0):
            result |= mask
    return result


def generate_timestamps(
    n_packets: int,
    base_time: Optional[float] = None,
    mean_gap: float = 0.001,
    jitter: float = 0.0005,
    seed: Optional[int] = None,
) -> List[float]:
    rng = random.Random(seed)
    t = base_time or time.time()
    timestamps = []
    for _ in range(n_packets):
        timestamps.append(t)
        gap = mean_gap + rng.uniform(-jitter, jitter)
        t += max(gap, 1e-6)
    return timestamps


def generate_synthetic_payload(
    length: int,
    proto: int = 6,
    seed: Optional[int] = None,
) -> bytes:
    rng = random.Random(seed)
    if length <= 0:
        return b""
    return bytes([rng.randint(0, 255) for _ in range(length)])


def estimate_packet_length(payload: bytes, proto: int = 6) -> int:
    header = 20
    if proto == 6:
        header += 20
    elif proto == 17:
        header += 8
    return header + len(payload)


def quantize_series_to_bytes(series, n_bins: int = 256) -> bytes:
    """
    Cuantiza una serie temporal normalizada (valores en [-1, 1]) a bytes (0-255).
    """
    normed = (series.clamp(-1, 1) + 1.0) / 2.0   # → [0, 1]
    quantized = (normed * (n_bins - 1)).round().int().clamp(0, 255)
    return bytes(quantized.tolist())
