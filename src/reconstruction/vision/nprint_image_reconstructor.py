"""
src/reconstruction/vision/nprint_image_reconstructor.py
=============================
Reconstructores para representaciones visuales.

NprintImageReconstructor
    Imagen RGB tipo NetDiffusion → decolorización → matriz nprint →
    campos de red → paquetes.
    Totalmente heurístico. Requiere thresholds y mapeos color → valor.

Mantiene el contrato de BaseReconstructor:
    decode()               → List[List[ParsedPacket]]
    heuristics(…, *, meta) → List[ParsedPacket]
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import copy

from src.reconstruction.base import (PacketWindowReconstructor, ReconstructionMeta)
from src.data_utils.preprocessing import ParsedPacket
from src.reconstruction.heuristics import (
    _NPRINT_FIELDS,
    assign_synthetic_ips,
    assign_synthetic_ports,
    estimate_packet_length,
    _compose_tcp_flags,
    infer_protocol_from_port,
    FlowState,
    _safe_int,
    _is_valid_ip_str, 
    _clamp_port,
     _canonical_flow_key, 
     _infer_payload_len_from_packet
)

# ---------------------------------------------------------------------------
# NprintImageReconstructor
# ---------------------------------------------------------------------------

class NprintImageReconstructor(PacketWindowReconstructor):
    """
    Reconstrucción desde una imagen nPrint visual.

    Estrategia:
      1) decode(): imagen -> matriz de campos -> ParsedPacket parcial
      2) heuristics(): limpieza local de campos
      3) _repair_inter_packet(): semántica de flujo TCP
      4) _repair_container(): orden final y coherencia del contenedor
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
        samples:
            (B, C, H, W) o (B, H, W)

        Cada fila de la imagen se interpreta como candidato a paquete.
        """
        if samples.dim() == 2:
            samples = samples.unsqueeze(0).unsqueeze(0)
        elif samples.dim() == 3:
            # (B, H, W) -> (B, 1, H, W)
            samples = samples.unsqueeze(1)

        if samples.max() <= 1.0:
            samples = samples * self.pixel_scale

        result: List[List[ParsedPacket]] = []

        for b in range(samples.shape[0]):
            img = samples[b]                   # (C, H, W)
            gray = img.float().mean(dim=0)     # (H, W) robusto frente a RGB

            field_matrix = self._image_to_field_matrix(gray)

            pkts: List[ParsedPacket] = []
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
        Convierte una imagen 2D (H, W) en una matriz (n_pkts, n_fields).

        - Si W >= n_fields: agregación por bloques de columnas.
        - Si W < n_fields: interpolación bilineal.
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
    # Decodificación de una fila nPrint
    # ------------------------------------------------------------------

    @staticmethod
    def decode_nprint_row(
        row: torch.Tensor,
        field_widths: Optional[Dict[str, int]] = None,
    ) -> dict:
        """
        Convierte una fila de la representación nPrint en un diccionario de campos.

        Si un valor no es recuperable, se deja en -1 para que la reparación
        posterior lo trate como desconocido.
        """
        defaults = {
            "ip_ver": 4,
            "ip_ihl": 5,
            "ip_tos": 0,
            "ip_len": -1,
            "ip_id": 0,
            "ip_flg": 2,
            "ip_off": 0,
            "ip_ttl": 64,
            "ip_pro": 6,
            "ip_sum": 0,
            "ip_src": 0,
            "ip_dst": 0,
            "tcp_sport": 0,
            "tcp_dport": 80,
            "tcp_seq": 0,
            "tcp_ack": 0,
            "tcp_doff": 5,
            "tcp_res": 0,
            "tcp_fin": 0,
            "tcp_syn": 0,
            "tcp_rst": 0,
            "tcp_psh": 0,
            "tcp_ack_flag": 0,
            "tcp_urg": 0,
            "tcp_win": 65535,
            "tcp_sum": 0,
            "tcp_urp": 0,
            "udp_sport": 0,
            "udp_dport": 53,
            "udp_len": -1,
            "udp_sum": 0,
            "payload": 0,
        }

        result = dict(defaults)
        values = row.round().int().tolist()

        for i, fname in enumerate(_NPRINT_FIELDS):
            if i < len(values):
                value = int(values[i])
                result[fname] = value

        return result

    @staticmethod
    def nprint_fields_to_packet(fields: dict) -> ParsedPacket:
        """
        Construye un ParsedPacket a partir de un diccionario de campos nPrint.

        Importante:
            aquí no se pretende producir un paquete final; se crea un candidato.
            La consistencia semántica fuerte se impondrá en heuristics() y
            _repair_inter_packet().
        """
        proto = _safe_int(fields.get("ip_pro", 6), 6)

        def int_to_ip(val) -> str:
            if isinstance(val, str) and _is_valid_ip_str(val):
                return val
            v = _safe_int(val, 0)
            v = max(0, min(v, 0xFFFFFFFF))
            return f"{(v >> 24) & 0xFF}.{(v >> 16) & 0xFF}.{(v >> 8) & 0xFF}.{v & 0xFF}"

        pkt = ParsedPacket()

        pkt.ip_src = int_to_ip(fields.get("ip_src", 0x0A000001))
        pkt.ip_dst = int_to_ip(fields.get("ip_dst", 0x0A000002))
        pkt.ip_proto = proto
        pkt.ip_len = _safe_int(fields.get("ip_len", -1), -1)
        pkt.ip_ttl = _safe_int(fields.get("ip_ttl", -1), -1)

        pkt.sport = _safe_int(fields.get("tcp_sport" if proto == 6 else "udp_sport", -1), -1)
        pkt.dport = _safe_int(fields.get("tcp_dport" if proto == 6 else "udp_dport", -1), -1)

        pkt.tcp_seq = _safe_int(fields.get("tcp_seq", 0), 0)
        pkt.tcp_ack = _safe_int(fields.get("tcp_ack", 0), 0)
        pkt.tcp_window = _safe_int(fields.get("tcp_win", -1), -1)
        pkt.tcp_flags = _compose_tcp_flags(
            fin=_safe_int(fields.get("tcp_fin", 0), 0),
            syn=_safe_int(fields.get("tcp_syn", 0), 0),
            rst=_safe_int(fields.get("tcp_rst", 0), 0),
            psh=_safe_int(fields.get("tcp_psh", 0), 0),
            ack=_safe_int(fields.get("tcp_ack_flag", 0), 0),
            urg=_safe_int(fields.get("tcp_urg", 0), 0),
        )

        pkt.udp_len = _safe_int(fields.get("udp_len", -1), -1)

        # Payload sintético mínimo si la longitud IP sugiere que hay datos.
        payload_len = 0
        if pkt.ip_len > 0:
            if proto == 6:
                payload_len = max(0, pkt.ip_len - 20 - 20)
            elif proto == 17:
                payload_len = max(0, pkt.ip_len - 20 - 8)

        pkt.payload = bytes(payload_len) if payload_len > 0 else b""
        pkt.timestamp = -1.0  # la fase inter-packet lo corrige
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
        Normalización local:
          - IPs válidas y distintas
          - protocolo coherente
          - puertos dentro de rango
          - TTL válido
          - longitudes mínimas
        La semántica TCP fuerte se deja para _repair_inter_packet().
        """
        if not packets:
            return packets

        default_src, default_dst = assign_synthetic_ips(seed=self.seed)
        default_sport, default_dport = assign_synthetic_ports(proto=6, seed=self.seed)

        for i, pkt in enumerate(packets):
            # IPs
            if not _is_valid_ip_str(getattr(pkt, "ip_src", None)):
                pkt.ip_src = default_src
                meta.repair_notes.append(f"pkt[{i}] ip_src vacío -> {default_src}")

            if not _is_valid_ip_str(getattr(pkt, "ip_dst", None)) or pkt.ip_dst == pkt.ip_src:
                pkt.ip_dst = default_dst if default_dst != pkt.ip_src else "10.0.0.2"
                meta.repair_notes.append(f"pkt[{i}] ip_dst inválido/igual a src -> {pkt.ip_dst}")

            # Protocolo
            proto = _safe_int(getattr(pkt, "ip_proto", -1), -1)
            if proto not in (1, 6, 17):
                inferred = infer_protocol_from_port(_clamp_port(getattr(pkt, "dport", 0), 0))
                pkt.ip_proto = inferred
                meta.repair_notes.append(f"pkt[{i}] ip_proto {proto} -> {inferred}")

            # Puertos
            if pkt.ip_proto == 6:
                pkt.sport = _clamp_port(getattr(pkt, "sport", None), default_sport)
                pkt.dport = _clamp_port(getattr(pkt, "dport", None), default_dport)
                if _safe_int(getattr(pkt, "tcp_window", -1), -1) <= 0 or _safe_int(getattr(pkt, "tcp_window", -1), -1) > 65535:
                    pkt.tcp_window = 65535
            elif pkt.ip_proto == 17:
                pkt.sport = _clamp_port(getattr(pkt, "sport", None), default_sport)
                pkt.dport = _clamp_port(getattr(pkt, "dport", None), default_dport)
                if _safe_int(getattr(pkt, "udp_len", -1), -1) < 8:
                    pkt.udp_len = 8 + len(pkt.payload or b"")

            # TTL
            ttl = _safe_int(getattr(pkt, "ip_ttl", -1), -1)
            if not (1 <= ttl <= 255):
                pkt.ip_ttl = 64

            # Longitud IP
            if _safe_int(getattr(pkt, "ip_len", -1), -1) < 0:
                pkt.ip_len = estimate_packet_length(pkt.payload, pkt.ip_proto)

            # Timestamp aún no definitiva.
            if not isinstance(getattr(pkt, "timestamp", None), (int, float)):
                pkt.timestamp = -1.0

        return packets

    # ------------------------------------------------------------------
    # Inter-packet repair (TCP stateful)
    # ------------------------------------------------------------------

    def _repair_inter_packet(
        self,
        packets: List[ParsedPacket],
        *,
        meta: ReconstructionMeta,
    ) -> List[ParsedPacket]:
        """
        Reparación inter-paquete con estado de flujo.

        Para TCP:
          - agrupa por flujo canónico
          - fuerza handshake SYN -> SYN/ACK -> ACK
          - mantiene seq/ack coherentes
          - genera timestamps monótonos por flujo
          - opcionalmente cierra el flujo al final

        Para UDP/otros:
          - preserva el orden y fija timestamps monótonos
          - asegura longitudes mínimas
        """
        if not packets:
            return packets

        # Agrupamos por flujo canónico.
        groups: Dict[tuple, List[Tuple[int, ParsedPacket]]] = {}
        for idx, pkt in enumerate(packets):
            key = _canonical_flow_key(pkt)
            groups.setdefault(key, []).append((idx, pkt))

        repaired_with_keys: List[Tuple[float, int, ParsedPacket]] = []

        # Desplazamiento entre flujos para evitar colisiones temporales.
        flow_offset = 0.0

        # Ordenamos grupos por aparición temprana.
        sorted_groups = sorted(
            groups.items(),
            key=lambda kv: min(original_idx for original_idx, _ in kv[1]),
        )

        for group_idx, (flow_key, items) in enumerate(sorted_groups):
            # Orden interno por timestamp si ya existe; si no, por índice.
            items.sort(
                key=lambda x: (
                    _safe_int(getattr(x[1], "timestamp", -1), -1)
                    if isinstance(getattr(x[1], "timestamp", None), (int, float))
                    else -1,
                    x[0],
                )
            )

            first_idx, first_pkt = items[0]
            proto = _safe_int(getattr(first_pkt, "ip_proto", 6), 6)

            group_start = self.base_timestamp + flow_offset
            group_gap = self.inter_packet_gap

            if proto == 6:
                state = FlowState.from_packet(
                    first_pkt,
                    flow_key=flow_key,
                    fallback_src=assign_synthetic_ips(seed=self.seed)[0],
                    fallback_dst=assign_synthetic_ips(seed=self.seed)[1],
                    fallback_sport=assign_synthetic_ports(proto=6, seed=self.seed)[0],
                    fallback_dport=assign_synthetic_ports(proto=6, seed=self.seed)[1],
                    rng_seed=self.seed,
                )

                for pos, (original_idx, pkt) in enumerate(items):
                    pkt = copy.deepcopy(pkt)
                    pkt = state.repair_tcp_packet(
                        pkt,
                        position=pos,
                        total=len(items),
                        timestamp=group_start + pos * group_gap,
                        meta=meta,
                        force_close_last=True,
                    )
                    repaired_with_keys.append((pkt.timestamp, original_idx, pkt))

            elif proto == 17:
                # Reparación UDP simple: timestamps monótonos y longitud coherente.
                for pos, (original_idx, pkt) in enumerate(items):
                    pkt = copy.deepcopy(pkt)
                    pkt.timestamp = group_start + pos * group_gap
                    pkt.ip_proto = 17
                    pkt.sport = _clamp_port(getattr(pkt, "sport", 0), 0)
                    pkt.dport = _clamp_port(getattr(pkt, "dport", 0), 0)

                    payload_len = len(pkt.payload) if getattr(pkt, "payload", None) is not None else 0
                    if payload_len == 0:
                        payload_len = _infer_payload_len_from_packet(pkt)
                        if payload_len > 0:
                            pkt.payload = bytes(self.seed or 0 for _ in range(payload_len))

                    pkt.udp_len = max(8, 8 + payload_len)
                    if _safe_int(getattr(pkt, "ip_len", -1), -1) < 20 + pkt.udp_len:
                        pkt.ip_len = 20 + pkt.udp_len

                    repaired_with_keys.append((pkt.timestamp, original_idx, pkt))

            else:
                # ICMP / otros: monotonicidad y limpieza ligera.
                for pos, (original_idx, pkt) in enumerate(items):
                    pkt = copy.deepcopy(pkt)
                    pkt.timestamp = group_start + pos * group_gap
                    if _safe_int(getattr(pkt, "ip_len", -1), -1) < 0:
                        pkt.ip_len = estimate_packet_length(pkt.payload, pkt.ip_proto)
                    repaired_with_keys.append((pkt.timestamp, original_idx, pkt))

            # Reservamos tiempo para el siguiente flujo.
            flow_offset += max(1.0, (len(items) + 1) * self.inter_packet_gap * 2.0)

        # Orden final global por timestamp.
        repaired_with_keys.sort(key=lambda t: (t[0], t[1]))
        repaired = [pkt for _, _, pkt in repaired_with_keys]

        return repaired

    # ------------------------------------------------------------------
    # Container repair
    # ------------------------------------------------------------------

    def _repair_container(self, container, *, meta: ReconstructionMeta):
        """
        Ordena el contenedor y deja start/end time consistentes.
        """
        if getattr(container, "packets", None):
            container.packets.sort(key=lambda p: (_safe_int(getattr(p, "timestamp", 0.0), 0.0), getattr(p, "ip_src", ""), getattr(p, "ip_dst", "")))
            container.start_time = float(container.packets[0].timestamp)
            container.end_time = float(container.packets[-1].timestamp)
        return container