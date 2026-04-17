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
from dataclasses import dataclass, field

from src.data_utils.preprocessing import ParsedPacket
from src.reconstruction.base import ReconstructionMeta, MOD32


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

def _compose_tcp_flags(*, fin: int = 0, syn: int = 0, rst: int = 0, psh: int = 0, ack: int = 0, urg: int = 0) -> int:
    """
    Usa el helper existente para mantener el mismo encoding de flags que el resto del proyecto.
    """
    return recompose_tcp_flags_from_fields(
        {
            "tcp_fin": int(fin),
            "tcp_syn": int(syn),
            "tcp_rst": int(rst),
            "tcp_psh": int(psh),
            "tcp_ack": int(ack),
            "tcp_urg": int(urg),
        }
    )

def _safe_int(value, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def _clamp_port(value, default: int) -> int:
    try:
        v = int(value)
    except Exception:
        return default
    return v if 0 <= v <= 65535 else default


def _is_valid_ip_str(value) -> bool:
    return isinstance(value, str) and value not in ("", "0.0.0.0", "None")


def _infer_payload_len_from_packet(pkt: ParsedPacket) -> int:
    """
    Estimación conservadora del payload a partir de longitudes disponibles.
    Si ya existe payload real, se respeta.
    """
    if getattr(pkt, "payload", None) is not None and len(pkt.payload) > 0:
        return len(pkt.payload)

    ip_len = _safe_int(getattr(pkt, "ip_len", -1), -1)
    if ip_len < 0:
        return 0

    proto = _safe_int(getattr(pkt, "ip_proto", 6), 6)
    ip_header_len = 20
    if proto == 6:
        l4_header_len = 20
    elif proto == 17:
        l4_header_len = 8
    elif proto == 1:
        l4_header_len = 8
    else:
        l4_header_len = 0

    return max(0, ip_len - ip_header_len - l4_header_len)


def _canonical_flow_key(pkt: ParsedPacket) -> tuple:
    """
    Clave canónica para agrupar ambos sentidos de un mismo flujo.
    Para TCP/UDP agrupa por endpoints sin importar dirección.
    """
    proto = _safe_int(getattr(pkt, "ip_proto", 6), 6)
    src_ip = getattr(pkt, "ip_src", "0.0.0.0") or "0.0.0.0"
    dst_ip = getattr(pkt, "ip_dst", "0.0.0.0") or "0.0.0.0"
    sport = _clamp_port(getattr(pkt, "sport", 0), 0)
    dport = _clamp_port(getattr(pkt, "dport", 0), 0)

    a = (src_ip, sport)
    b = (dst_ip, dport)
    if proto in (6, 17):
        x, y = sorted([a, b], key=lambda t: (t[0], t[1]))
        return (proto, x, y)

    return (proto, a, b)

HTTP_REQUEST_STUB  = b"GET / HTTP/1.1\r\nHost: example.com\r\n\r\n"
HTTP_RESPONSE_STUB = b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n"
DNS_QUERY_STUB     = b"\x00\x01\x01\x00\x00\x01\x00\x00\x00\x00\x00\x00"

def generate_protocol_aware_payload(dport: int, position: int, rng: random.Random) -> bytes:
    """Genera payloads mínimamente realistas según el puerto destino."""
    if dport == 80 and position > 2:
        return HTTP_REQUEST_STUB if position % 2 == 0 else HTTP_RESPONSE_STUB
    if dport == 53:
        return DNS_QUERY_STUB
    # Fallback: payload aleatorio con longitud típica para el puerto
    typical_size = {443: 512, 22: 48, 25: 64}.get(dport, rng.randint(8, 256))
    return bytes(rng.randint(0, 255) for _ in range(typical_size))

# ---------------------------------------------------------------------------
# FlowState
# ---------------------------------------------------------------------------

@dataclass
class FlowState:
    """
    Estado mínimo de un flujo bidireccional TCP.

    Mantiene:
      - endpoints canónicos
      - dirección iniciadora
      - ISNs por sentido
      - next_seq por sentido
      - fase de handshake
      - cierre lógico
    """
    flow_key: tuple
    initiator_ip: str
    initiator_port: int
    responder_ip: str
    responder_port: int
    protocol: int = 6
    rng_seed: Optional[int] = None

    iss_initiator: int = field(init=False)
    iss_responder: int = field(init=False)
    next_seq_initiator: int = field(init=False)
    next_seq_responder: int = field(init=False)

    handshake_stage: int = field(default=0, init=False)
    established: bool = field(default=False, init=False)
    closed: bool = field(default=False, init=False)
    packet_count: int = field(default=0, init=False)

    _rng: random.Random = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.rng_seed)
        self.iss_initiator = self._rng.getrandbits(32)
        self.iss_responder = self._rng.getrandbits(32)
        self.next_seq_initiator = self.iss_initiator
        self.next_seq_responder = self.iss_responder

    @classmethod
    def from_packet(
        cls,
        pkt: ParsedPacket,
        *,
        flow_key: tuple,
        fallback_src: str,
        fallback_dst: str,
        fallback_sport: int,
        fallback_dport: int,
        rng_seed: Optional[int] = None,
    ) -> "FlowState":
        src_ip = pkt.ip_src if _is_valid_ip_str(getattr(pkt, "ip_src", None)) else fallback_src
        dst_ip = pkt.ip_dst if _is_valid_ip_str(getattr(pkt, "ip_dst", None)) else fallback_dst
        sport = _clamp_port(getattr(pkt, "sport", None), fallback_sport)
        dport = _clamp_port(getattr(pkt, "dport", None), fallback_dport)

        return cls(
            flow_key=flow_key,
            initiator_ip=src_ip,
            initiator_port=sport,
            responder_ip=dst_ip,
            responder_port=dport,
            protocol=6,
            rng_seed=rng_seed,
        )

    def direction_of(self, pkt: ParsedPacket) -> Optional[int]:
        """
        0 = iniciador -> responder
        1 = responder -> iniciador
        """
        src_ip = getattr(pkt, "ip_src", None)
        dst_ip = getattr(pkt, "ip_dst", None)
        sport = _clamp_port(getattr(pkt, "sport", 0), 0)
        dport = _clamp_port(getattr(pkt, "dport", 0), 0)

        if src_ip == self.initiator_ip and dst_ip == self.responder_ip and sport == self.initiator_port and dport == self.responder_port:
            return 0
        if src_ip == self.responder_ip and dst_ip == self.initiator_ip and sport == self.responder_port and dport == self.initiator_port:
            return 1

        # Fallback laxo: si coincide IP pero no puertos, o viceversa.
        if src_ip == self.initiator_ip and dst_ip == self.responder_ip:
            return 0
        if src_ip == self.responder_ip and dst_ip == self.initiator_ip:
            return 1

        return None

    def _apply_directional_identity(self, pkt: ParsedPacket, direction: int) -> None:
        if direction == 0:
            pkt.ip_src = self.initiator_ip
            pkt.ip_dst = self.responder_ip
            pkt.sport = self.initiator_port
            pkt.dport = self.responder_port
        else:
            pkt.ip_src = self.responder_ip
            pkt.ip_dst = self.initiator_ip
            pkt.sport = self.responder_port
            pkt.dport = self.initiator_port

    def _ensure_payload(self, pkt: ParsedPacket, payload_len: int) -> None:
        if payload_len <= 0:
            if getattr(pkt, "payload", None) is None:
                pkt.payload = b""
            return

        if getattr(pkt, "payload", None) is None or len(pkt.payload) == 0:
            pkt.payload = bytes(self._rng.getrandbits(8) for _ in range(payload_len))

    def _refresh_lengths(self, pkt: ParsedPacket) -> None:
        payload_len = len(pkt.payload) if getattr(pkt, "payload", None) is not None else 0
        proto = _safe_int(getattr(pkt, "ip_proto", 6), 6)

        ip_header_len = 20
        if proto == 6:
            l4_len = 20
        elif proto == 17:
            l4_len = 8
        else:
            l4_len = 8

        min_total = ip_header_len + l4_len + payload_len
        if _safe_int(getattr(pkt, "ip_len", -1), -1) < min_total:
            pkt.ip_len = min_total

        if proto == 17:
            expected_udp = 8 + payload_len
            if _safe_int(getattr(pkt, "udp_len", -1), -1) < expected_udp:
                pkt.udp_len = expected_udp

    def repair_tcp_packet(
        self,
        pkt: ParsedPacket,
        *,
        position: int,
        total: int,
        timestamp: float,
        meta: ReconstructionMeta,
        force_close_last: bool = True,
    ) -> ParsedPacket:
        """
        Aplica una FSM TCP muy conservadora:
          0: SYN
          1: SYN/ACK
          2: ACK
          3+: data / ACK / cierre opcional
        """
        self.packet_count += 1
        pkt.ip_proto = 6
        pkt.timestamp = timestamp

        direction = self.direction_of(pkt)

        # Handshake: forzamos la estructura correcta aunque la entrada venga torcida.
        if position == 0:
            direction = 0
            self._apply_directional_identity(pkt, direction)
            pkt.tcp_seq = self.next_seq_initiator
            pkt.tcp_ack = 0
            pkt.tcp_flags = _compose_tcp_flags(syn=1, ack=0)
            pkt.tcp_window = 65535
            pkt.payload = b""
            self.next_seq_initiator = (self.next_seq_initiator + 1) % MOD32
            self.handshake_stage = 1
            meta.repair_notes.append("TCP handshake: pkt[0] forzado a SYN")
            self._refresh_lengths(pkt)
            return pkt

        if position == 1 and not self.closed:
            direction = 1
            self._apply_directional_identity(pkt, direction)
            pkt.tcp_seq = self.next_seq_responder
            pkt.tcp_ack = self.next_seq_initiator
            pkt.tcp_flags = _compose_tcp_flags(syn=1, ack=1)
            pkt.tcp_window = 65535
            pkt.payload = b""
            self.next_seq_responder = (self.next_seq_responder + 1) % MOD32
            self.handshake_stage = 2
            meta.repair_notes.append("TCP handshake: pkt[1] forzado a SYN/ACK")
            self._refresh_lengths(pkt)
            return pkt

        if position == 2 and not self.closed:
            direction = 0
            self._apply_directional_identity(pkt, direction)
            pkt.tcp_seq = self.next_seq_initiator
            pkt.tcp_ack = self.next_seq_responder
            pkt.tcp_flags = _compose_tcp_flags(ack=1)
            pkt.tcp_window = 65535
            self.established = True
            self.handshake_stage = 3
            meta.repair_notes.append("TCP handshake: pkt[2] forzado a ACK")
            self._refresh_lengths(pkt)
            return pkt

        # Estado ya establecido.
        self.established = True

        if direction is None:
            # Fallback: alternancia por posición si la dirección observada no es usable.
            direction = 0 if position % 2 == 0 else 1
            meta.repair_notes.append(f"TCP direction ambigua en pkt[{position}] -> fallback alternado")

        self._apply_directional_identity(pkt, direction)

        payload_len = len(pkt.payload) if getattr(pkt, "payload", None) is not None else 0
        if payload_len == 0:
            payload_len = _infer_payload_len_from_packet(pkt)

        # Si no hay payload, al menos dejamos ACK. Si hay payload, PSH+ACK.
        seq_val = self.next_seq_initiator if direction == 0 else self.next_seq_responder
        ack_val = self.next_seq_responder if direction == 0 else self.next_seq_initiator

        pkt.tcp_seq = seq_val
        pkt.tcp_ack = ack_val
        pkt.tcp_window = 65535

        if position == total - 1 and total >= 4 and force_close_last and not self.closed:
            # Cierre sintético del flujo.
            pkt.tcp_flags = _compose_tcp_flags(fin=1, ack=1)
            self.closed = True
            advance = payload_len + 1
            meta.repair_notes.append(f"TCP flow cierre sintético en pkt[{position}]")
        else:
            if payload_len > 0:
                pkt.tcp_flags = _compose_tcp_flags(psh=1, ack=1)
                self._ensure_payload(pkt, payload_len)
                advance = payload_len
            else:
                pkt.tcp_flags = _compose_tcp_flags(ack=1)
                advance = 0

        if direction == 0:
            self.next_seq_initiator = (seq_val + advance) % MOD32
        else:
            self.next_seq_responder = (seq_val + advance) % MOD32

        self._refresh_lengths(pkt)
        return pkt

