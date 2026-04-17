"""
src/reconstruction/vision/gaf_reconstructor.py
===============================================

Reconstrucción desde GAFRepresentation (GASF / GADF).

Perfil: LOSSY | agresividad=0.90 | needs_flow_state=True | needs_payload_synthesis=True
────────────────────────────────────────────────────────────────────────────────────────
La inversión GAF NO es biyectiva. Solo se usa la diagonal de la imagen
(GASF(i,i) = cos²(φ_i)) para recuperar una serie temporal aproximada,
que se cuantiza a bytes y se segmenta en paquetes.

Toda la semántica de red es heurística:
  - decode()      → diagonal GASF → serie → bytes → paquetes crudos.
  - heuristics()  → asigna IPs, puertos, proto, flags (la heurística ES el método).
  - _repair_intra → síntesis completa (is_lossy).
  - _synthesize   → genera payloads donde faltan.
  - _repair_inter → FlowState: TCP FSM bidireccional con handshake completo.
"""

from __future__ import annotations

from typing import List, Optional

import torch

from src.reconstruction.base import (
    ChunkReconstructor,
    InvertibilityLevel,
    ReconstructionMeta,
    ReconstructionProfile,
)
from src.data_utils.preprocessing import ParsedPacket
from src.reconstruction.heuristics import (
    assign_synthetic_ips,
    assign_synthetic_ports,
    estimate_packet_length,
    generate_timestamps,
    infer_protocol_from_port,
    infer_tcp_flags,
    quantize_series_to_bytes,
    segment_bytes_into_packets,
)


class GAFReconstructor(ChunkReconstructor):
    """
    Reconstrucción desde GAFRepresentation (GASF/GADF).

    Pipeline:
        imagen (H, W) ∈ [-1, 1]
        → diagonal GASF → serie temporal ∈ [-1, 1]   [inversión aproximada]
        → cuantización → bytes
        → segmentación en paquetes
        → heurísticas completas (IPs, puertos, flags)
        → FlowState TCP FSM

    La inversión no es exacta. El tráfico generado es estadísticamente
    plausible pero no fiel al original.
    """

    # ── Perfil de reconstrucción ──────────────────────────────────────────
    @property
    def profile(self) -> ReconstructionProfile:
        return ReconstructionProfile(
            invertibility=InvertibilityLevel.LOSSY,
            needs_flow_state=True,        # FSM TCP bidireccional
            needs_payload_synthesis=True, # sintetizar payloads donde falten
            repair_aggressiveness=0.90,   # is_lossy: heurística es el método
        )

    def __init__(
        self,
        max_payload_bytes: int = 1460,
        min_payload_bytes: int = 1,
        n_bins: int = 256,
        seed: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.max_payload_bytes = max_payload_bytes
        self.min_payload_bytes = min_payload_bytes
        self.n_bins = n_bins
        self.seed = seed

    # ------------------------------------------------------------------
    # decode
    # ------------------------------------------------------------------

    def decode(self, samples: torch.Tensor) -> List[List[ParsedPacket]]:
        """
        samples : (B, C, H, W) o (B, H, W) — imágenes GASF en [-1, 1].

        Por muestra:
          1. Extraer diagonal → serie temporal
          2. Cuantizar → bytes
          3. Segmentar → ParsedPackets crudos (sin cabeceras)
        """
        if samples.dim() == 3:
            samples = samples.unsqueeze(1)

        result = []
        for b in range(samples.shape[0]):
            img    = samples[b]
            series = self.inverse_gasf_diagonal(img)
            raw    = quantize_series_to_bytes(series, n_bins=self.n_bins)
            chunks = segment_bytes_into_packets(
                raw,
                max_payload=self.max_payload_bytes,
                min_payload=self.min_payload_bytes,
                seed=self.seed,
            )
            result.append([ParsedPacket(payload_bytes=c) for c in chunks])

        return result

    # ------------------------------------------------------------------
    # heuristics  (la heurística ES el método para LOSSY)
    # ------------------------------------------------------------------

    def heuristics(
        self,
        packets: List[ParsedPacket],
        *,
        meta: ReconstructionMeta,
    ) -> List[ParsedPacket]:
        """
        Asigna cabeceras completas a los paquetes GASF.

        Para representaciones LOSSY la heurística no es un parche:
        ES la forma de producir tráfico funcional a partir de datos
        que no tienen semántica de red.

        Nota: el FlowState posterior (_repair_inter_with_flow_state)
        sobreescribirá las flags TCP para garantizar el handshake correcto.
        Aquí solo establecemos los valores de campo que la FSM no toca
        (IPs, puertos, TTL, ip_len).
        """
        if not packets:
            return packets

        src_ip, dst_ip = assign_synthetic_ips(seed=self.seed)
        sport, dport   = assign_synthetic_ports(proto=6, seed=self.seed)
        proto          = infer_protocol_from_port(dport)
        n              = len(packets)
        timestamps     = generate_timestamps(n, base_time=self.base_timestamp, seed=self.seed)

        for i, pkt in enumerate(packets):
            pkt.ip_src    = src_ip
            pkt.ip_dst    = dst_ip
            pkt.sport     = sport
            pkt.dport     = dport
            pkt.ip_proto  = proto
            pkt.ip_ttl    = 64
            pkt.timestamp = timestamps[i]
            pkt.ip_len    = estimate_packet_length(pkt.payload or b"", proto)

            if proto == 6:
                # Flags preliminares; FlowState las sobreescribirá con la FSM
                pkt.tcp_flags  = infer_tcp_flags(i, n, has_data=bool(pkt.payload))
                pkt.tcp_window = 65535
            elif proto == 17:
                pkt.udp_len = 8 + len(pkt.payload or b"")

        return packets

    # ------------------------------------------------------------------
    # Inversión GASF (diagonal)
    # ------------------------------------------------------------------

    @staticmethod
    def inverse_gasf_diagonal(image: torch.Tensor) -> torch.Tensor:
        """
        Inversión aproximada de GASF usando la diagonal principal.

            GASF(i,i) = cos(2·φ_i)  →  φ_i = arccos(GASF(i,i)) / 2
            x̂_i = cos(φ_i)

        Parameters
        ----------
        image : (H, W) o (C, H, W) en [-1, 1].

        Returns
        -------
        Serie temporal (N,) en [-1, 1].
        """
        img  = image[0] if image.dim() == 3 else image
        diag = torch.clamp(img.diagonal(), -1.0, 1.0)
        phi  = torch.acos(diag) / 2.0
        x    = torch.cos(phi)
        return 2.0 * x - 1.0   # → [-1, 1]