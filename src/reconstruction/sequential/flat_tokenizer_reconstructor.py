"""
src/reconstruction/sequential/flat_tokenizer_reconstructor.py
=============================================================

Reconstrucción desde FlatTokenizer.

Perfil: PARTIAL | agresividad=0.15 (is_structured)
────────────────────────────────────────────────────
El FlatTokenizer mapea bytes → tokens en un vocabulario arbitrario. No
codifica campos de red; toda la semántica L3/L4 la genera heuristics().

Por eso:
  - heuristics()          asigna IPs, puertos, proto, flags, timestamps.
  - _repair_intra_*       solo valida rangos (no sintetiza de nuevo).
  - _repair_inter_*       solo timestamps monótonos (no toca TCP seq/ack).
  - needs_flow_state=False: FlowState sería un parche sobre un parche.
"""

from __future__ import annotations

from typing import List, Optional

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
    infer_protocol_from_port,
    infer_tcp_flags,
    segment_bytes_into_packets,
    tokens_to_bytes,
)


class FlatTokenizerReconstructor(FlowReconstructor):
    """
    Reconstrucción desde FlatTokenizer.

    Pipeline:
        tokens → bytes (lookup lineal)
        → segmentación en chunks de longitud variable
        → heuristics(): cabeceras sintéticas completas
        → reparación mínima (solo rangos + timestamps)
    """

    # ── Perfil de reconstrucción ──────────────────────────────────────────
    @property
    def profile(self) -> ReconstructionProfile:
        return ReconstructionProfile(
            invertibility=InvertibilityLevel.PARTIAL,
            needs_flow_state=False,
            needs_payload_synthesis=False,
            repair_aggressiveness=0.15,   # is_structured: confiar en heuristics
        )

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
        samples : (B, L) — secuencias de tokens enteros.

        Convierte cada fila en bytes brutos y segmenta en paquetes.
        Los campos de cabecera son centinelas; se fijan en heuristics().
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
        Asigna cabeceras sintéticas completas a todos los paquetes.

        Dado que FlatTokenizer no codifica campos de red, heuristics() es
        la única fuente de semántica L3/L4 para esta representación.

        La reparación posterior (_repair_intra_ranges_only) solo validará
        rangos; no sobreescribirá lo que aquí se establece.
        """
        if not packets:
            return packets

        src_ip, dst_ip = assign_synthetic_ips(seed=self.seed)
        sport, dport   = assign_synthetic_ports(proto=6, seed=self.seed)
        proto          = infer_protocol_from_port(dport)
        n              = len(packets)
        timestamps     = generate_timestamps(
            n, base_time=self.base_timestamp, seed=self.seed
        )

        for i, pkt in enumerate(packets):
            pkt.ip_src   = src_ip
            pkt.ip_dst   = dst_ip
            pkt.sport    = sport
            pkt.dport    = dport
            pkt.ip_proto = proto
            pkt.ip_ttl   = 64
            pkt.timestamp = timestamps[i]
            pkt.ip_len   = estimate_packet_length(pkt.payload or b"", proto)

            if proto == 6:
                pkt.tcp_flags  = infer_tcp_flags(i, n, has_data=bool(pkt.payload))
                pkt.tcp_window = 65535
            elif proto == 17:
                pkt.udp_len = 8 + len(pkt.payload or b"")

        return packets