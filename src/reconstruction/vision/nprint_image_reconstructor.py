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

from typing import Dict, List, Optional

import torch

from src.reconstruction.base import ChunkReconstructor,PacketWindowReconstructor, ReconstructionMeta
from src.data_utils.preprocessing import ParsedPacket
from src.reconstruction.heuristics import (
    _NPRINT_FIELDS,
    assign_synthetic_ips,
    assign_synthetic_ports,
    estimate_packet_length,
    generate_timestamps,
    infer_protocol_from_port,
    infer_tcp_flags,
    quantize_series_to_bytes,
    recompose_tcp_flags_from_fields,
    segment_bytes_into_packets,
)

# ---------------------------------------------------------------------------
# NprintImageReconstructor
# ---------------------------------------------------------------------------

class NprintImageReconstructor(PacketWindowReconstructor):
    """
    Reconstrucción desde NprintImageRepresentation.

    Inspirado en NetDiffusion. La imagen RGB codifica la matriz nprint:
      - Cada fila de la imagen → un paquete
      - Cada columna → un campo de la cabecera nprint
      - Valores de color → valor de campo (cuantizado)

    Pipeline:
        imagen RGB (C, H, W) → thresholding/cuantización por canal
        → matriz (H, F) de valores de campo
        → por fila: decode_nprint_row() → nprint_fields_to_packet() → ParsedPacket
        → heurísticas de corrección estructural
    """

    _N_FIELDS = len(_NPRINT_FIELDS)

    def __init__(
        self,
        max_packets_per_flow: int = 64,
        pixel_scale: float = 255.0,
        proto_threshold: float = 128.0,
        flag_threshold: float = 128.0,
        patch_size: int = 8,
        seed: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.max_packets_per_flow = max_packets_per_flow
        self.pixel_scale = pixel_scale
        self.proto_threshold = proto_threshold
        self.flag_threshold = flag_threshold
        self.patch_size = patch_size
        self.seed = seed

    # ------------------------------------------------------------------
    # decode
    # ------------------------------------------------------------------

    def decode(self, samples: torch.Tensor) -> List[List[ParsedPacket]]:
        """
        samples : (B, C, H, W) — batch de imágenes nprint en [0, 1] o [0, 255].

        Cada fila de la imagen (dimensión H) → un paquete.
        """
        if samples.dim() == 3:
            samples = samples.unsqueeze(1)

        B, C, H, W = samples.shape

        if samples.max() <= 1.0:
            samples = samples * self.pixel_scale

        result = []

        for b in range(B):
            img = samples[b]   # (C, H, W)
            gray = img[0]      # (H, W) — primer canal

            field_matrix = self._image_to_field_matrix(gray)   # (n_pkts, n_fields)

            pkts = []
            n_pkts = min(field_matrix.shape[0], self.max_packets_per_flow)
            for row_idx in range(n_pkts):
                row = field_matrix[row_idx]
                fields = self.decode_nprint_row(row)
                pkt = self.nprint_fields_to_packet(fields)
                pkts.append(pkt)

            result.append(pkts)

        return result

    def _image_to_field_matrix(self, gray: torch.Tensor) -> torch.Tensor:
        """
        Convierte una imagen 2D (H, W) en una matriz (n_pkts, n_fields)
        agregando columnas en bloques para que coincidan con _N_FIELDS.
        """
        H, W = gray.shape
        n_fields = self._N_FIELDS

        if W >= n_fields:
            blocks = torch.chunk(gray, n_fields, dim=1)
            field_matrix = torch.stack([b.mean(dim=1) for b in blocks], dim=1)
        else:
            field_matrix = torch.nn.functional.interpolate(
                gray.unsqueeze(0).unsqueeze(0),
                size=(H, n_fields),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).squeeze(0)

        field_matrix = field_matrix.clamp(0, 255).round()
        return field_matrix

    # ------------------------------------------------------------------
    # Helpers estáticos de decodificación nprint
    # ------------------------------------------------------------------

    @staticmethod
    def decode_nprint_row(
        row: torch.Tensor,
        field_widths: Optional[Dict[str, int]] = None,
    ) -> dict:
        """
        Decodifica una fila de la representación nprint en un diccionario de campos.

        Parameters
        ----------
        row          : tensor 1D con los valores de los campos en orden _NPRINT_FIELDS.
        field_widths : ancho en bits de cada campo (si None usa defaults razonables).

        Returns
        -------
        dict con nombres de campo → valor int.
        """
        defaults = {
            "ip_ver": 4,       "ip_ihl": 5,        "ip_tos": 0,
            "ip_len": 40,      "ip_id": 0,          "ip_flg": 2,
            "ip_off": 0,       "ip_ttl": 64,        "ip_pro": 6,
            "ip_sum": 0,       "ip_src": 0,         "ip_dst": 0,
            "tcp_sport": 0,    "tcp_dport": 80,     "tcp_seq": 0,
            "tcp_ack": 0,      "tcp_doff": 5,       "tcp_res": 0,
            "tcp_fin": 0,      "tcp_syn": 0,        "tcp_rst": 0,
            "tcp_psh": 0,      "tcp_ack_flag": 0,   "tcp_urg": 0,
            "tcp_win": 65535,  "tcp_sum": 0,        "tcp_urp": 0,
            "udp_sport": 0,    "udp_dport": 53,     "udp_len": 8,
            "udp_sum": 0,      "payload": 0,
        }
        result = dict(defaults)
        values = row.int().tolist()
        for i, fname in enumerate(_NPRINT_FIELDS):
            if i < len(values):
                result[fname] = int(values[i])
        return result

    @staticmethod
    def nprint_fields_to_packet(fields: dict) -> ParsedPacket:
        """
        Construye un ParsedPacket a partir de un diccionario de campos nprint.
        Recompone flags TCP y maneja proto TCP/UDP.
        """
        proto = fields.get("ip_pro", 6)

        tcp_flags = recompose_tcp_flags_from_fields({
            "tcp_fin": fields.get("tcp_fin", 0),
            "tcp_syn": fields.get("tcp_syn", 0),
            "tcp_rst": fields.get("tcp_rst", 0),
            "tcp_psh": fields.get("tcp_psh", 0),
            "tcp_ack": fields.get("tcp_ack_flag", 0),
            "tcp_urg": fields.get("tcp_urg", 0),
        })

        def int_to_ip(val: int) -> str:
            val = max(0, min(int(val), 0xFFFFFFFF))
            return (
                f"{(val >> 24) & 0xFF}.{(val >> 16) & 0xFF}"
                f".{(val >> 8) & 0xFF}.{val & 0xFF}"
            )

        pkt = ParsedPacket()
        pkt.ip_src = int_to_ip(fields.get("ip_src", 0x0A000001))
        pkt.ip_dst = int_to_ip(fields.get("ip_dst", 0x0A000002))
        pkt.ip_proto = proto
        pkt.ip_len = fields.get("ip_len", -1)
        pkt.ip_ttl = fields.get("ip_ttl", 64)
        pkt.sport = fields.get("tcp_sport" if proto == 6 else "udp_sport", 0)
        pkt.dport = fields.get("tcp_dport" if proto == 6 else "udp_dport", 80)
        pkt.tcp_seq = fields.get("tcp_seq", 0)
        pkt.tcp_ack = fields.get("tcp_ack", 0)
        pkt.tcp_flags = tcp_flags
        pkt.tcp_window = fields.get("tcp_win", 65535)
        pkt.udp_len = fields.get("udp_len", -1)
        pkt.payload = b""
        return pkt

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
        Para NprintImage los campos ya están parcialmente decodificados.
        Heurísticas aplicadas:
          1. Corregir IPs: si ip_src == ip_dst, asignar IPs distintas.
          2. Inferir protocolo desde dport si ip_proto es incoherente.
          3. Sincronizar flags TCP (SYN al inicio, FIN al final).
          4. Añadir timestamps.
          5. Calcular ip_len si es negativo.
        """
        if not packets:
            return packets

        n = len(packets)
        timestamps = generate_timestamps(n, base_time=self.base_timestamp, seed=self.seed)

        default_src, default_dst = assign_synthetic_ips(seed=self.seed)
        default_sport, default_dport = assign_synthetic_ports(proto=6, seed=self.seed)

        for i, pkt in enumerate(packets):
            pkt.timestamp = timestamps[i]

            if pkt.ip_src == "0.0.0.0" or pkt.ip_src is None or pkt.ip_src == pkt.ip_dst:
                pkt.ip_src = default_src
                pkt.ip_dst = default_dst

            if pkt.sport == 0:
                pkt.sport = default_sport
            if pkt.dport == 0:
                pkt.dport = default_dport

            inferred_proto = infer_protocol_from_port(pkt.dport)
            if pkt.ip_proto not in (6, 17, 1):
                pkt.ip_proto = inferred_proto
                meta.repair_notes.append(f"pkt[{i}] ip_proto incoherente → {inferred_proto}")

            if pkt.ip_proto == 6 and pkt.tcp_flags == 0:
                pkt.tcp_flags = infer_tcp_flags(i, n, has_data=True)

            if pkt.ip_len < 0:
                pkt.ip_len = estimate_packet_length(pkt.payload, pkt.ip_proto)

        return packets
