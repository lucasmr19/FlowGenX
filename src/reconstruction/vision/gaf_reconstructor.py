"""
src/reconstruction/vision.py
=============================
Reconstructores para representaciones visuales.

Clases
------
GAFReconstructor
    Imagen GASF → serie temporal (vía diagonal) → bytes → paquetes sintéticos.
    La inversión es aproximada (no biyectiva). Se usa solo la diagonal principal.

NprintImageReconstructor
    Imagen RGB tipo NetDiffusion → decolorización → matriz nprint →
    campos de red → paquetes.
    Totalmente heurístico. Requiere thresholds y mapeos color → valor.

Ambas clases mantienen el contrato de BaseReconstructor:
    decode()               → List[List[ParsedPacket]]
    heuristics(…, *, meta) → List[ParsedPacket]
"""

from __future__ import annotations

from typing import List, Optional

import torch

from src.reconstruction.base import ChunkReconstructor, ReconstructionMeta
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

# ---------------------------------------------------------------------------
# GAFReconstructor
# ---------------------------------------------------------------------------

class GAFReconstructor(ChunkReconstructor):
    """
    Reconstrucción desde GAFRepresentation (GASF/GADF).

    Pipeline:
        imagen (H, W) ∈ [-1, 1]
        → diagonal GASF → serie temporal ∈ [-1, 1]   [inversión aproximada]
        → cuantización → bytes
        → segmentación en paquetes
        → heurísticas de cabecera

    Nota:
        La inversión GAF NO es exacta. El resto de la imagen (correlaciones
        cruzadas) se descarta por ser matemáticamente no invertible.
        Genera tráfico estadísticamente plausible, no fiel al original.
    """

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
        samples : (B, C, H, W) o (B, H, W) — batch de imágenes GASF en [-1, 1].

        Pipeline por muestra:
          1. Extraer diagonal → serie temporal
          2. Cuantizar → bytes
          3. Segmentar bytes → lista de ParsedPacket en bruto
        """
        if samples.dim() == 3:
            samples = samples.unsqueeze(1)

        B = samples.shape[0]
        result = []

        for b in range(B):
            img = samples[b]  # (C, H, W)

            series = self.inverse_gasf_diagonal(img)   # (N,) ∈ [-1, 1]
            raw_bytes = quantize_series_to_bytes(series, n_bins=self.n_bins)

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
        Tras la inversión GAF, los bytes no tienen semántica de cabecera.
        Aplicamos las mismas heurísticas que FlatTokenizer:
          - IPs y puertos sintéticos
          - Flags TCP por posición
          - Timestamps uniformes
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
                pkt.tcp_flags = infer_tcp_flags(i, n, has_data=len(pkt.payload) > 0)
                pkt.tcp_window = 65535
            elif proto == 17:
                pkt.udp_len = 8 + len(pkt.payload)

        return packets

    # ------------------------------------------------------------------
    # Inversión GAF (método estático)
    # ------------------------------------------------------------------

    @staticmethod
    def inverse_gasf_diagonal(image: torch.Tensor) -> torch.Tensor:
        """
        Inversión aproximada de GASF usando solo la diagonal.

            GASF(i,i) = cos(2·φ_i)  →  φ_i = arccos(GASF(i,i)) / 2

        La serie temporal normalizada se recupera como:
            x̂_i = cos(φ_i)

        Parameters
        ----------
        image : torch.Tensor de shape (H, W) o (C, H, W), valores en [-1, 1].

        Returns
        -------
        torch.Tensor de shape (N,) con la serie temporal en [-1, 1].
        """
        if image.dim() == 3:
            img = image[0]   # primer canal
        else:
            img = image

        diag = torch.clamp(img.diagonal(), -1.0, 1.0)  # GASF(i,i)
        phi = torch.acos(diag) / 2.0                   # φ_i
        series = torch.cos(phi)                         # x̂_i ∈ [0, 1]
        series = 2.0 * series - 1.0                     # → [-1, 1]
        return series
