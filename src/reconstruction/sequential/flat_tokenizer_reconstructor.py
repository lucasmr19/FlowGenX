from __future__ import annotations

import logging
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
# FlatTokenizerReconstructor
# ---------------------------------------------------------------------------

class FlatTokenizerReconstructor(FlowReconstructor):
    """
    Reconstrucción desde FlatTokenizer.

    El tokenizador flat mapea bytes a tokens en un vocabulario arbitrario.
    La reconstrucción:
      1. Proyecta tokens → bytes via lookup lineal.
      2. Segmenta los bytes en paquetes de longitud variable (TCP-like).
      3. Añade cabeceras sintéticas con heurísticas de puerto/protocolo.

    No recupera cabeceras reales. Genera tráfico estadísticamente plausible.
    """

    def __init__(
        self,
        vocab_size: int = 256,
        max_payload_bytes: int = 1460,
        min_payload_bytes: int = 1,
        seed: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.max_payload_bytes = max_payload_bytes
        self.min_payload_bytes = min_payload_bytes
        self.seed = seed

    # ------------------------------------------------------------------
    # decode
    # ------------------------------------------------------------------

    def decode(self, samples: torch.Tensor) -> List[List[ParsedPacket]]:
        """
        samples : (B, L) — batch de secuencias de tokens enteros.

        Convierte cada fila en una lista de ParsedPacket en bruto:
          - payload = bytes decodificados del chunk de tokens
          - resto de campos = centinelas (se fijan en heuristics)
        """
        B, L = samples.shape
        result = []

        for b in range(B):
            token_seq = samples[b].int().tolist()
            raw_bytes = tokens_to_bytes(token_seq, vocab_size=self.vocab_size)

            chunks = segment_bytes_into_packets(
                raw_bytes,
                max_payload=self.max_payload_bytes,
                min_payload=self.min_payload_bytes,
                seed=self.seed,
            )

            pkts = [ParsedPacket(payload_bytes=chunk) for chunk in chunks]
            result.append(pkts)

        return result

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
        Para FlatTokenizer asignamos heurísticas completas de cabecera:
          - IPs sintéticas RFC 1918
          - Puerto destino típico (TCP) → derivamos protocolo
          - Puerto origen efímero
          - Flags TCP según posición en flujo
          - Timestamps uniformes con jitter
        """
        if not packets:
            return packets

        src_ip, dst_ip = assign_synthetic_ips(seed=self.seed)
        sport, dport = assign_synthetic_ports(proto=6, seed=self.seed)
        proto = infer_protocol_from_port(dport)
        n = len(packets)
        timestamps = generate_timestamps(n, base_time=self.base_timestamp, seed=self.seed)

        for i, pkt in enumerate(packets):
            pkt.ip_src = src_ip
            pkt.ip_dst = dst_ip
            pkt.sport = sport
            pkt.dport = dport
            pkt.ip_proto = proto
            pkt.ip_ttl = 64
            pkt.timestamp = timestamps[i]
            pkt.ip_len = estimate_packet_length(pkt.payload, proto)

            if proto == 6:
                pkt.tcp_flags = infer_tcp_flags(
                    packet_index=i,
                    total_packets=n,
                    has_data=len(pkt.payload) > 0,
                )
                pkt.tcp_window = 65535
            elif proto == 17:
                pkt.udp_len = 8 + len(pkt.payload)

        return packets