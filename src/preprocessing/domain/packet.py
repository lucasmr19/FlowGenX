"""

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# Estructuras de datos
# ---------------------------------------------------------------------------

@dataclass
class ParsedPacket:
    """
    Representación normalizada de un paquete de red.

    Todos los campos son opcionales porque no todos los protocolos
    los exponen. Los campos ausentes se representan como None o -1.
    """
    # --- timestamps ---
    timestamp: float = 0.0
    iat: float = 0.0

    # -------------------------------------------------
    # Ethernet
    # -------------------------------------------------
    eth_dhost: Optional[bytes] = None
    eth_shost: Optional[bytes] = None
    eth_ethertype: int = -1

    # -------------------------------------------------
    # IPv4
    # -------------------------------------------------
    ipv4_ver: int = 4
    ipv4_hl: int = -1
    ipv4_tos: int = -1
    ipv4_tl: int = -1
    ipv4_id: int = -1

    ipv4_rbit: int = 0
    ipv4_dfbit: int = 0
    ipv4_mfbit: int = 0
    ipv4_foff: int = 0

    ipv4_ttl: int = -1
    ipv4_proto: int = -1
    ipv4_cksum: int = -1

    ipv4_src: Optional[str] = None
    ipv4_dst: Optional[str] = None

    ipv4_opt: Optional[bytes] = None

    sport: int = 0
    dport: int = 0

    # -------------------------------------------------
    # IPv6
    # -------------------------------------------------
    ipv6_ver: int = 6
    ipv6_tc: int = -1
    ipv6_fl: int = -1
    ipv6_len: int = -1
    ipv6_nh: int = -1
    ipv6_hl: int = -1

    ipv6_src: Optional[str] = None
    ipv6_dst: Optional[str] = None

    # -------------------------------------------------
    # TCP
    # -------------------------------------------------
    tcp_sprt: int = -1
    tcp_dprt: int = -1

    tcp_seq: int = -1
    tcp_ackn: int = -1

    tcp_doff: int = -1
    tcp_res: int = 0

    tcp_ns: int = 0
    tcp_cwr: int = 0
    tcp_ece: int = 0
    tcp_urg: int = 0
    tcp_ackf: int = 0
    tcp_psh: int = 0
    tcp_rst: int = 0
    tcp_syn: int = 0
    tcp_fin: int = 0

    tcp_wsize: int = -1
    tcp_cksum: int = -1
    tcp_urp: int = -1

    tcp_opt: Optional[bytes] = None

    # -------------------------------------------------
    # UDP
    # -------------------------------------------------
    udp_sport: int = -1
    udp_dport: int = -1
    udp_len: int = -1
    udp_cksum: int = -1

    # -------------------------------------------------
    # ICMP
    # -------------------------------------------------
    icmp_type: int = -1
    icmp_code: int = -1
    icmp_cksum: int = -1
    icmp_roh: int = -1

    # -------------------------------------------------
    # Payload
    # -------------------------------------------------
    payload_len: int = 0
    payload_bytes: bytes = b""

    # -------------------------------------------------
    # Meta
    # -------------------------------------------------
    direction: int = 0
    flow_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = self.__dict__.copy()
        d.pop("payload_bytes")
        return d

    # ----- propiedades derivadas -----

    @property
    def ip_version(self) -> int:
        if self.ipv4_src is not None:
            return 4
        if self.ipv6_src is not None:
            return 6
        return -1
    
    # --- Compatibility aliases for reconstruction / evaluation ---

    @property
    def ip_src(self) -> Optional[str]:
        return self.ipv4_src or self.ipv6_src

    @ip_src.setter
    def ip_src(self, value: Optional[str]) -> None:
        self.ipv4_src = value
        if value is not None:
            self.ipv6_src = None


    @property
    def ip_dst(self) -> Optional[str]:
        return self.ipv4_dst or self.ipv6_dst

    @ip_dst.setter
    def ip_dst(self, value: Optional[str]) -> None:
        self.ipv4_dst = value
        if value is not None:
            self.ipv6_dst = None


    @property
    def ip_proto(self) -> int:
        if self.ipv4_proto != -1:
            return self.ipv4_proto
        if self.ipv6_nh != -1:
            return self.ipv6_nh
        return -1

    @ip_proto.setter
    def ip_proto(self, value: int) -> None:
        self.ipv4_proto = value
        self.ipv6_nh = value


    @property
    def ip_len(self) -> int:
        if self.ipv4_tl != -1:
            return self.ipv4_tl
        if self.ipv6_len != -1:
            return self.ipv6_len
        return 0

    @ip_len.setter
    def ip_len(self, value: int) -> None:
        self.ipv4_tl = value
        self.ipv6_len = value


    @property
    def ip_ttl(self) -> int:
        if self.ipv4_ttl != -1:
            return self.ipv4_ttl
        if self.ipv6_hl != -1:
            return self.ipv6_hl
        return -1

    @ip_ttl.setter
    def ip_ttl(self, value: int) -> None:
        self.ipv4_ttl = value
        self.ipv6_hl = value


    @property
    def tcp_ack(self) -> int:
        return self.tcp_ackn

    @tcp_ack.setter
    def tcp_ack(self, value: int) -> None:
        self.tcp_ackn = value


    @property
    def tcp_window(self) -> int:
        return self.tcp_wsize

    @tcp_window.setter
    def tcp_window(self, value: int) -> None:
        self.tcp_wsize = value


    @property
    def payload(self) -> bytes:
        return self.payload_bytes

    @payload.setter
    def payload(self, value: bytes) -> None:
        self.payload_bytes = value or b""
        self.payload_len = len(self.payload_bytes)


    @property
    def tcp_flags(self) -> int:
        flags = 0
        flags |= (self.tcp_fin  << 0)
        flags |= (self.tcp_syn  << 1)
        flags |= (self.tcp_rst  << 2)
        flags |= (self.tcp_psh  << 3)
        flags |= (self.tcp_ackf << 4)
        flags |= (self.tcp_urg  << 5)
        return flags

    @tcp_flags.setter
    def tcp_flags(self, value: int) -> None:
        self.tcp_fin  = (value >> 0) & 1
        self.tcp_syn  = (value >> 1) & 1
        self.tcp_rst  = (value >> 2) & 1
        self.tcp_psh  = (value >> 3) & 1
        self.tcp_ackf = (value >> 4) & 1
        self.tcp_urg  = (value >> 5) & 1