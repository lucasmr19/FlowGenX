"""
src/reconstruction/sequential/semantic_byte_reconstructor.py
============================================================

Reconstrucción desde SemanticByteTokenizer.

Perfil: PARTIAL | agresividad=0.50 (is_moderate) | needs_flow_state=True
─────────────────────────────────────────────────────────────────────────
SemanticByteTokenizer codifica bytes reales (vocab≈256). El modelo genera
una secuencia de bytes que puede (o no) tener estructura IP válida.

Estrategia:
  1. decode()         → tokens → bytes → intento de parseo IP/TCP/UDP.
                        Si los primeros bytes forman una cabecera IP válida,
                        se extraen campos directamente.
  2. heuristics()     → completa puertos, flags, longitudes y coherencia
                        temporal donde el parse no pudo recuperarlos.
  3. _repair_intra    → rellena campos faltantes (is_moderate).
  4. _synthesize      → NO activo (needs_payload_synthesis=True pero is_moderate).
  5. _repair_inter    → FlowState: FSM TCP bidireccional con handshake real
                        (needs_flow_state=True). Genera seq/ack coherentes.

Diferencia respecto a Flat/ProtocolAware:
  - La heurística es moderada: completa lo que falta, no inventa todo.
  - FlowState garantiza un flujo TCP semánticamente válido.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch

from src.reconstruction.base import (
    FlowReconstructor,
    InvertibilityLevel,
    ReconstructionMeta,
    ReconstructionProfile,
)
from src.preprocessing import ParsedPacket
from src.reconstruction.heuristics import (
    assign_synthetic_ips,
    assign_synthetic_ports,
    estimate_packet_length,
    generate_timestamps,
    infer_tcp_flags,
    segment_bytes_into_packets,
    tokens_to_bytes,
    _is_valid_ip_str,
    _clamp_port,
)


class SemanticByteReconstructor(FlowReconstructor):
    """
    Reconstrucción desde SemanticByteTokenizer.

    El vocabulario de tokens ≈ 256 (un token por valor de byte).
    La secuencia de tokens es directamente una secuencia de bytes.
    """

    # ── Perfil de reconstrucción ──────────────────────────────────────────
    @property
    def profile(self) -> ReconstructionProfile:
        return ReconstructionProfile(
            invertibility=InvertibilityLevel.PARTIAL,
            needs_flow_state=True,        # FSM TCP bidireccional
            needs_payload_synthesis=True, # activo solo con is_lossy (no aplica aquí)
            repair_aggressiveness=0.50,   # is_moderate
        )

    _MIN_HEADER_BYTES = 20   # mínimo para intentar parsear IP

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
        (B, L) de tokens-byte → List[List[ParsedPacket]].

        Intenta parsear cabeceras IP/TCP/UDP directamente de los bytes.
        Si el parse falla, segmenta los bytes como payload crudo.
        """
        B, L = samples.shape
        result = []

        for b in range(B):
            token_seq  = samples[b].int().tolist()
            raw_bytes  = tokens_to_bytes(token_seq, vocab_size=self.vocab_size)
            pkts       = self._parse_or_segment(raw_bytes)
            result.append(pkts)

        return result

    def _parse_or_segment(self, raw_bytes: bytes) -> List[ParsedPacket]:
        pkts: List[ParsedPacket] = []
        offset = 0

        while offset < len(raw_bytes):
            remaining = raw_bytes[offset:]
            if self.try_parse_headers and len(remaining) >= self._MIN_HEADER_BYTES:
                pkt, consumed = self._try_parse_ip_packet(remaining)
                if pkt is not None:
                    pkts.append(pkt)
                    offset += consumed
                    continue

            # Fallback: segmentar como payload crudo con tamaño conservador
            fallback_max = max(8, min(40, len(remaining) // 3 + 1))
            chunks = segment_bytes_into_packets(
                remaining, max_payload=fallback_max, min_payload=4, seed=self.seed
            )
            pkts.extend(ParsedPacket(payload_bytes=c) for c in chunks)
            break

        return pkts or [ParsedPacket(payload_bytes=raw_bytes[: self.max_payload_bytes])]

    def _try_parse_ip_packet(
        self, data: bytes
    ) -> Tuple[Optional[ParsedPacket], int]:
        """
        Intenta parsear un paquete IPv4 desde los primeros bytes de `data`.
        Returns (ParsedPacket, bytes_consumidos) o (None, 0).
        """
        try:
            ver_ihl = data[0]
            version = (ver_ihl >> 4) & 0xF
            ihl     = (ver_ihl & 0xF) * 4

            if version != 4 or ihl < 20 or ihl > len(data):
                return None, 0

            ip_len = int.from_bytes(data[2:4], "big")
            if ip_len < ihl or ip_len > len(data):
                ip_len = min(len(data), 1500)

            ip_proto = data[9]
            src_ip   = ".".join(str(b) for b in data[12:16])
            dst_ip   = ".".join(str(b) for b in data[16:20])

            pkt         = ParsedPacket()
            pkt.ip_src  = src_ip
            pkt.ip_dst  = dst_ip
            pkt.ip_proto = ip_proto
            pkt.ip_len  = ip_len
            pkt.ip_ttl  = data[8]

            transport = data[ihl:ip_len]

            if ip_proto == 6 and len(transport) >= 20:
                pkt.sport      = int.from_bytes(transport[0:2], "big")
                pkt.dport      = int.from_bytes(transport[2:4], "big")
                pkt.tcp_seq    = int.from_bytes(transport[4:8], "big")
                pkt.tcp_ack    = int.from_bytes(transport[8:12], "big")
                tcp_doff       = ((transport[12] >> 4) & 0xF) * 4
                pkt.tcp_flags  = transport[13]
                pkt.tcp_window = int.from_bytes(transport[14:16], "big")
                pkt.payload    = data[ihl + max(tcp_doff, 20): ip_len]

            elif ip_proto == 17 and len(transport) >= 8:
                pkt.sport   = int.from_bytes(transport[0:2], "big")
                pkt.dport   = int.from_bytes(transport[2:4], "big")
                pkt.udp_len = int.from_bytes(transport[4:6], "big")
                pkt.payload = transport[8:]

            else:
                pkt.payload = transport

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
        Completa los campos que el parse no recuperó.

        Política moderada:
          - IPs: rellenar si "0.0.0.0" o None.
          - Puertos: rellenar si 0 (el modelo puede haber generado puertos
            válidos en los bytes; los respetamos si son no-cero).
          - Protocolo: inferir si ausente.
          - Flags TCP: inferir por posición si son 0 (el parse puede haber
            extraído flags reales; si son 0 probablemente no los generó).
          - Timestamps: siempre asignar (la FSM los ajustará después).
          - ip_len / udp_len: corregir solo si son claramente incorrectos.
        """
        if not packets:
            return packets

        # Inferir endpoints del flujo desde los paquetes parseados
        src_ip = next(
            (p.ip_src for p in packets if _is_valid_ip_str(p.ip_src)), None
        )
        dst_ip = next(
            (p.ip_dst for p in packets if _is_valid_ip_str(p.ip_dst)), None
        )
        if not src_ip or not dst_ip:
            src_ip, dst_ip = assign_synthetic_ips(seed=self.seed)

        sport = next((p.sport for p in packets if _clamp_port(p.sport, 0) > 0), None)
        dport = next((p.dport for p in packets if _clamp_port(p.dport, 0) > 0), None)
        if not sport or not dport:
            s_fb, d_fb = assign_synthetic_ports(proto=6, seed=self.seed)
            sport = sport or s_fb
            dport = dport or d_fb

        n          = len(packets)
        timestamps = generate_timestamps(n, base_time=self.base_timestamp, seed=self.seed)

        for i, pkt in enumerate(packets):
            # IPs
            if not _is_valid_ip_str(getattr(pkt, "ip_src", None)):
                pkt.ip_src = src_ip
            if not _is_valid_ip_str(getattr(pkt, "ip_dst", None)):
                pkt.ip_dst = dst_ip

            # Puertos
            if _clamp_port(getattr(pkt, "sport", 0), 0) == 0:
                pkt.sport = sport
            if _clamp_port(getattr(pkt, "dport", 0), 0) == 0:
                pkt.dport = dport

            # Protocolo
            if pkt.ip_proto not in (1, 6, 17):
                pkt.ip_proto = 6

            # TTL
            if not (1 <= pkt.ip_ttl <= 255):
                pkt.ip_ttl = 64

            # ip_len
            payload_len = len(pkt.payload) if pkt.payload else 0
            if pkt.ip_len < 20:
                pkt.ip_len = estimate_packet_length(pkt.payload or b"", pkt.ip_proto)

            # TCP flags: inferir si el parse no los extrajo (flags==0 es ACK
            # en TCP pero también puede significar "no parseado")
            if pkt.ip_proto == 6:
                if pkt.tcp_flags == 0:
                    pkt.tcp_flags = infer_tcp_flags(i, n, has_data=bool(pkt.payload))
                if not (0 < pkt.tcp_window <= 65535):
                    pkt.tcp_window = 65535

            # UDP
            elif pkt.ip_proto == 17:
                if pkt.udp_len < 8:
                    pkt.udp_len = 8 + payload_len

            # Timestamp base (la FSM lo refinará)
            pkt.timestamp = timestamps[i]

        return packets