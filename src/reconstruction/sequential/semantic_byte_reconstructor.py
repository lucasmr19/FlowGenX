"""
src/reconstruction/sequential/semantic_byte_reconstructor.py
============================================================

Reconstrucción para SemanticByteTokenizer. Convierte tokens → bytes directamente
y luego intenta parsear cabeceras IP/TCP/UDP desde los propios bytes. Si el parse falla,
produce ParsedPacket con payload crudo. Mayor fidelidad que ProtocolAwareTokenizer pero no garantiza exactitud de cabeceras.
"""

from __future__ import annotations

from typing import List, Optional

import torch

from src.reconstruction.base import FlowReconstructor, ReconstructionMeta
from src.data_utils.preprocessing import ParsedPacket
from src.reconstruction.heuristics import (
    assign_synthetic_ips,
    assign_synthetic_ports,
    estimate_packet_length,
    generate_timestamps,
    infer_protocol_from_port,
    infer_tcp_flags,
    segment_bytes_into_packets,
    tokens_to_bytes,
)


# ---------------------------------------------------------------------------
# SemanticByteReconstructor
# ---------------------------------------------------------------------------

class SemanticByteReconstructor(FlowReconstructor):
    """
    Reconstrucción desde SemanticByteTokenizer.

    Los tokens representan bytes reales (vocabulario ≈ 256).
    La reconstrucción:
      1. Convierte tokens → bytes directamente (lookup 1:1 o proyección lineal).
      2. Intenta parsear cabeceras IP/TCP/UDP desde los propios bytes
         (si los primeros 40 bytes tienen estructura válida).
      3. Aplica heurísticas de fallback donde el parse falla.

    Mayor fidelidad que FlatTokenizer pero no garantiza exactitud de cabeceras.
    """

    # Umbral mínimo de bytes para intentar parsear cabecera IP
    _MIN_HEADER_BYTES = 20

    def __init__(
        self,
        vocab_size: int = 256,
        try_parse_headers: bool = True,
        max_payload_bytes: int = 1460,
        seed: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.try_parse_headers = try_parse_headers
        self.max_payload_bytes = max_payload_bytes
        self.seed = seed

    # ------------------------------------------------------------------
    # decode
    # ------------------------------------------------------------------

    def decode(self, samples: torch.Tensor) -> List[List[ParsedPacket]]:
        """
        samples : (B, L) — batch de secuencias de bytes como tokens enteros.

        Intenta parsear cabeceras IP/TCP/UDP desde los propios bytes.
        Si el parse falla, produce ParsedPacket con payload crudo.
        """
        B, L = samples.shape
        result = []

        for b in range(B):
            token_seq = samples[b].int().tolist()
            raw_bytes = tokens_to_bytes(token_seq, vocab_size=self.vocab_size)
            pkts = self._parse_or_segment(raw_bytes)
            result.append(pkts)

        return result

    def _parse_or_segment(self, raw_bytes: bytes) -> List[ParsedPacket]:
        """
        Intenta parsear estructuras IP desde los bytes crudos.
        Si no es posible, cae a segmentación simple.
        """
        pkts = []
        offset = 0

        while offset < len(raw_bytes):
            remaining = raw_bytes[offset:]

            if self.try_parse_headers and len(remaining) >= self._MIN_HEADER_BYTES:
                pkt, consumed = self._try_parse_ip_packet(remaining)
                if pkt is not None:
                    pkts.append(pkt)
                    offset += consumed
                    continue

            chunk_size = min(self.max_payload_bytes, len(remaining))
            pkts.append(ParsedPacket(payload_bytes=remaining[:chunk_size]))
            offset += chunk_size

        return pkts if pkts else [ParsedPacket(payload_bytes=raw_bytes[:self.max_payload_bytes])]

    def _try_parse_ip_packet(
        self, data: bytes
    ) -> tuple[Optional[ParsedPacket], int]:
        """
        Intenta parsear un paquete IP desde `data`.

        Returns (ParsedPacket, bytes_consumed) o (None, 0).
        """
        try:
            ver_ihl = data[0]
            version = (ver_ihl >> 4) & 0xF
            ihl = (ver_ihl & 0xF) * 4

            if version != 4 or ihl < 20 or ihl > len(data):
                return None, 0

            ip_len = int.from_bytes(data[2:4], "big")
            if ip_len < ihl or ip_len > len(data):
                ip_len = min(len(data), 1500)

            ip_proto = data[9]
            src_ip = ".".join(str(b) for b in data[12:16])
            dst_ip = ".".join(str(b) for b in data[16:20])

            pkt = ParsedPacket()
            pkt.ip_src = src_ip
            pkt.ip_dst = dst_ip
            pkt.ip_proto = ip_proto
            pkt.ip_len = ip_len
            pkt.ip_ttl = data[8]

            transport_data = data[ihl:ip_len]

            if ip_proto == 6 and len(transport_data) >= 20:  # TCP
                pkt.sport = int.from_bytes(transport_data[0:2], "big")
                pkt.dport = int.from_bytes(transport_data[2:4], "big")
                pkt.tcp_seq = int.from_bytes(transport_data[4:8], "big")
                pkt.tcp_ack = int.from_bytes(transport_data[8:12], "big")
                tcp_doff = ((transport_data[12] >> 4) & 0xF) * 4
                pkt.tcp_flags = transport_data[13]
                pkt.tcp_window = int.from_bytes(transport_data[14:16], "big")
                payload_start = ihl + max(tcp_doff, 20)
                pkt.payload = data[payload_start:ip_len]

            elif ip_proto == 17 and len(transport_data) >= 8:  # UDP
                pkt.sport = int.from_bytes(transport_data[0:2], "big")
                pkt.dport = int.from_bytes(transport_data[2:4], "big")
                pkt.udp_len = int.from_bytes(transport_data[4:6], "big")
                pkt.payload = transport_data[8:]

            else:
                pkt.payload = transport_data

            return pkt, ip_len

        except (IndexError, ValueError):
            return None, 0

    # ------------------------------------------------------------------
    # heuristics
    # ------------------------------------------------------------------

    def heuristics(
        self,
        packets: List[ParsedPacket],
        *,
        meta: ReconstructionMeta,
    ) -> List[ParsedPacket]:
        """
        Para SemanticByte: rellena campos que el parse no pudo recuperar.
        No sobreescribe campos ya parseados exitosamente.
        """
        if not packets:
            return packets

        src_ip, dst_ip = assign_synthetic_ips(seed=self.seed)
        sport, dport = assign_synthetic_ports(proto=6, seed=self.seed)
        n = len(packets)
        timestamps = generate_timestamps(n, base_time=self.base_timestamp, seed=self.seed)

        for i, pkt in enumerate(packets):
            if pkt.ip_src == "0.0.0.0" or pkt.ip_src is None:
                pkt.ip_src = src_ip
            if pkt.ip_dst == "0.0.0.0" or pkt.ip_dst is None:
                pkt.ip_dst = dst_ip
            if pkt.sport == 0:
                pkt.sport = sport
            if pkt.dport == 0:
                pkt.dport = dport

            pkt.timestamp = timestamps[i]

            if pkt.ip_len < 0:
                pkt.ip_len = estimate_packet_length(pkt.payload, pkt.ip_proto)

            if pkt.ip_proto == 6 and pkt.tcp_flags == 0:
                pkt.tcp_flags = infer_tcp_flags(
                    packet_index=i,
                    total_packets=n,
                    has_data=len(pkt.payload) > 0,
                )

        return packets
