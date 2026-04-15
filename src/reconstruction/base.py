"""
src/reconstruction/base.py
==========================

Contrato base del módulo de reconstrucción.

Idea general
------------
model_output -> project() -> decode() -> heuristics() -> repair -> build_container()

Responsabilidades:
  - Reutilizar ParsedPacket y Flow del pipeline de preprocessing.
  - Definir un contrato común para reconstructores.
  - Implementar reparación estructural compartida en 3 capas:
      1) _repair_intra_packet   -> campos fuera de rango
      2) _repair_inter_packet   -> timestamps monótonos, secuencias básicas
      3) _repair_container     -> coherencia del contenedor final
  - Facilitar trazabilidad con ReconstructionMeta (warnings, notas de reparación, etc.)
  - Ser agnóstico a la representación de entrada (tokens, latentes, etc.) y al tipo de modelo.
  - Permitir extensibilidad para heurísticas y reparaciones específicas por modelo o representación.
  - Mantener una API simple y clara para integración con el pipeline de evaluación.
  - Documentar claramente el contrato y las responsabilidades de cada método.
"""

from __future__ import annotations

import copy
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Generic, List, Optional, TypeVar, Union

import torch

from src.data_utils.preprocessing import ParsedPacket, Flow, PacketWindow, TrafficChunk

logger = logging.getLogger(__name__)

ContainerT = TypeVar("ContainerT")


# ---------------------------------------------------------------------------
# Contenedor de flujo sintético
# ---------------------------------------------------------------------------

@dataclass
class SyntheticFlow(Flow):
    """
    Contenedor de un flujo de tráfico reconstruido / sintético.

    Agrupa los paquetes que pertenecen a la misma 5-tupla de red y expone
    los metadatos mínimos necesarios para serialización y evaluación.
    """
    flow_id: str = ""
    src_ip: str = "0.0.0.0"
    dst_ip: str = "0.0.0.0"
    sport: int = 0
    dport: int = 0
    protocol: int = 6           # 6=TCP, 17=UDP, 1=ICMP
    packets: List[ParsedPacket] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0
    meta: Optional[ReconstructionMeta] = None

@dataclass
class SyntheticPacketWindow(PacketWindow):
    window_id: int
    packets: List[ParsedPacket] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0
    meta: Optional[ReconstructionMeta] = None

@dataclass
class SyntheticTrafficChunk(TrafficChunk):
    chunk_id: int
    packets: List[ParsedPacket] = field(default_factory=list)
    start_time: float = 0.0
    duration: float = 0.0
    meta: Optional[ReconstructionMeta] = None

SyntheticSample = Union[SyntheticFlow, SyntheticPacketWindow, SyntheticTrafficChunk]

# ---------------------------------------------------------------------------
# Metadatos de reconstrucción
# ---------------------------------------------------------------------------

@dataclass
class ReconstructionMeta:
    """
    Metadatos auxiliares de reconstrucción.
    """
    representation_name: str = ""
    model_name: str = ""
    label: int = -1
    warnings: List[str] = field(default_factory=list)
    repair_notes: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Contrato abstracto
# ---------------------------------------------------------------------------

class BaseReconstructor(ABC, Generic[ContainerT]):
    """
    Contrato base para reconstrucción de tráfico.

    Pipeline:

        samples -> decode() -> heuristics() -> repairs -> build_container()

    Subclases deben implementar:
      - decode(samples)       → List[List[ParsedPacket]]
      - heuristics(packets, *, meta)  → List[ParsedPacket]
      - _build_container(packets, *, meta) → ContainerT
    """

    def __init__(
        self,
        *,
        inter_packet_gap: float = 0.001,
        base_timestamp: Optional[float] = None,
        representation_name: str = "",
        model_name: str = "",
        verbose: bool = False,
    ) -> None:
        self.inter_packet_gap = float(inter_packet_gap)
        self.base_timestamp = float(base_timestamp) if base_timestamp is not None else time.time()
        self.representation_name = representation_name
        self.model_name = model_name
        self.verbose = verbose

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def reconstruct(
        self,
        samples: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        *,
        already_projected: bool = True,
    ) -> List[ContainerT]:
        """
        Orquesta la reconstrucción completa.
        """
        del already_projected

        decoded = self.decode(samples)

        outputs: List[ContainerT] = []
        for idx, raw_packets in enumerate(decoded):
            label = int(labels[idx].item()) if labels is not None else -1

            meta = ReconstructionMeta(
                representation_name=self.representation_name,
                model_name=self.model_name,
                label=label,
            )

            packets = self.heuristics(raw_packets, meta=meta)
            packets = [self._repair_intra_packet(pkt, meta=meta) for pkt in packets]
            packets = self._repair_inter_packet(packets, meta=meta)

            container = self._build_container(packets, meta=meta)
            container = self._repair_container(container, meta=meta)

            outputs.append(container)

            if self.verbose:
                logger.debug(
                    "[Reconstructor] sample=%d packets=%d label=%d",
                    idx,
                    len(packets),
                    label,
                )

        return outputs

    # ------------------------------------------------------------------
    # Métodos abstractos
    # ------------------------------------------------------------------

    @abstractmethod
    def decode(self, samples: torch.Tensor) -> List[List[ParsedPacket]]:
        raise NotImplementedError

    @abstractmethod
    def heuristics(
        self,
        packets: List[ParsedPacket],
        *,
        meta: ReconstructionMeta,
    ) -> List[ParsedPacket]:
        raise NotImplementedError

    @abstractmethod
    def _build_container(
        self,
        packets: List[ParsedPacket],
        *,
        meta: ReconstructionMeta,
    ) -> ContainerT:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Reparación estructural compartida
    # ------------------------------------------------------------------

    def _clone_packet(self, pkt: ParsedPacket) -> ParsedPacket:
        """Crea una copia defensiva para evitar mutaciones colaterales."""
        return copy.deepcopy(pkt)

    def _repair_intra_packet(
        self,
        pkt: ParsedPacket,
        *,
        meta: ReconstructionMeta,
    ) -> ParsedPacket:
        """
        Capa 1: reparación intra-paquete.
        Siempre usa los property setters de ParsedPacket para evitar
        inconsistencias entre ipv4_*/ipv6_* y sus aliases públicos.
        """
        pkt = self._clone_packet(pkt)
        notes = meta.repair_notes

        # --- IPs ---
        if not pkt.ip_src:
            pkt.ip_src = "10.0.0.1"
            notes.append("ip_src vacío -> 10.0.0.1")
        if not pkt.ip_dst:
            pkt.ip_dst = "10.0.0.2"
            notes.append("ip_dst vacío -> 10.0.0.2")

        # --- Protocolo ---
        proto = pkt.ip_proto
        if proto not in (1, 6, 17):
            notes.append(f"ip_proto {proto} -> 6")
            pkt.ip_proto = 6
            proto = 6

        # --- Puertos ---
        if not (0 <= pkt.sport <= 65535):
            notes.append(f"sport {pkt.sport} -> 0")
            pkt.sport = 0
        if not (0 <= pkt.dport <= 65535):
            notes.append(f"dport {pkt.dport} -> 0")
            pkt.dport = 0

        # --- TTL ---
        ttl = pkt.ip_ttl
        if not (1 <= ttl <= 255):
            notes.append(f"ip_ttl {ttl} -> 64")
            pkt.ip_ttl = 64

        # --- Longitud IP ---
        payload_len = len(pkt.payload) if pkt.payload is not None else 0
        min_len = 20
        if proto == 6:
            min_len += 20
        elif proto == 17:
            min_len += 8
        elif proto == 1:
            min_len += 8

        if pkt.ip_len < min_len:
            expected_len = max(min_len + payload_len, min_len)
            notes.append(f"ip_len {pkt.ip_len} -> {expected_len}")
            pkt.ip_len = expected_len

        # --- Campos TCP ---
        if proto == 6:
            if not (0 <= pkt.tcp_seq < 2**32):
                notes.append(f"tcp_seq {pkt.tcp_seq} -> 0")
                pkt.tcp_seq = 0
            if not (0 <= pkt.tcp_ack < 2**32):
                notes.append(f"tcp_ack {pkt.tcp_ack} -> 0")
                pkt.tcp_ack = 0
            if not (0 <= pkt.tcp_flags <= 255):
                notes.append(f"tcp_flags {pkt.tcp_flags} -> 0")
                pkt.tcp_flags = 0
            if not (0 < pkt.tcp_window <= 65535):
                notes.append(f"tcp_window {pkt.tcp_window} -> 65535")
                pkt.tcp_window = 65535

        # --- Campos UDP ---
        if proto == 17:
            expected_udp_len = 8 + payload_len
            if pkt.udp_len < 8:
                notes.append(f"udp_len {pkt.udp_len} -> {expected_udp_len}")
                pkt.udp_len = expected_udp_len

        # --- Timestamp ---
        if not isinstance(pkt.timestamp, (int, float)) or pkt.timestamp < 0:
            notes.append(f"timestamp inválido -> {self.base_timestamp}")
            pkt.timestamp = self.base_timestamp

        return pkt

    def _repair_inter_packet(
        self,
        packets: List[ParsedPacket],
        *,
        meta: ReconstructionMeta,
    ) -> List[ParsedPacket]:
        """
        Capa 2: reparación inter-paquete.
        Garantiza timestamps monótonos y secuencias TCP plausibles.
        """
        if not packets:
            return packets

        repaired: List[ParsedPacket] = []
        t = self.base_timestamp
        next_seq = 1000

        for pkt in packets:
            pkt = self._clone_packet(pkt)

            if pkt.timestamp <= t:
                t = t + self.inter_packet_gap
                pkt.timestamp = t
                meta.repair_notes.append(f"timestamp ajustado a monotónico: {t:.6f}")
            else:
                t = pkt.timestamp

            if pkt.ip_proto == 6:
                if pkt.tcp_seq <= 0:
                    pkt.tcp_seq = next_seq
                payload_len = len(pkt.payload) if pkt.payload is not None else 0
                next_seq = (pkt.tcp_seq + max(payload_len, 1)) % (2**32)

            repaired.append(pkt)

        return repaired

    def _repair_container(
        self,
        container: ContainerT,
        *,
        meta: ReconstructionMeta,
    ) -> ContainerT:
        """
        Capa 3: reparación a nivel de contenedor.
        Implementación por defecto: no-op. Las subclases pueden sobreescribir.
        """
        return container

    def _default_meta(self, label: int = -1) -> ReconstructionMeta:
        return ReconstructionMeta(
            representation_name=self.representation_name,
            model_name=self.model_name,
            label=label,
        )


# ---------------------------------------------------------------------------
# Implementación concreta por defecto de _build_container → SyntheticFlow
# ---------------------------------------------------------------------------

class FlowReconstructor(BaseReconstructor[SyntheticFlow]):
    """
    Mixin que implementa _build_container → SyntheticFlow.

    La mayoría de reconstructores concretos heredan de esta clase en lugar
    de BaseReconstructor directamente, salvo que necesiten un contenedor
    de tipo distinto.
    """

    def _build_container(
        self,
        packets: List[ParsedPacket],
        *,
        meta: ReconstructionMeta,
    ) -> SyntheticFlow:
        first = packets[0] if packets else None
        return SyntheticFlow(
            flow_id=first.flow_id if first and first.flow_id else "",
            src_ip=first.ip_src or "0.0.0.0" if first else "0.0.0.0",
            dst_ip=first.ip_dst or "0.0.0.0" if first else "0.0.0.0",
            sport=first.sport if first else 0,
            dport=first.dport if first else 0,
            protocol=first.ip_proto if first else 6,
            packets=packets,
            label=meta.label,
            meta=meta,
        )

class PacketWindowReconstructor(BaseReconstructor[SyntheticPacketWindow]):
    def _build_container(self, packets: List[ParsedPacket], *, meta: ReconstructionMeta) -> SyntheticPacketWindow:
        first = packets[0] if packets else None
        last = packets[-1] if packets else None
        return SyntheticPacketWindow(
            window_id=0,
            packets=packets,
            start_time=first.timestamp if first else 0.0,
            end_time=last.timestamp if last else 0.0,
            label=meta.label,
            meta=meta,
        )

class ChunkReconstructor(BaseReconstructor[SyntheticTrafficChunk]):
    def _build_container(self, packets: List[ParsedPacket], *, meta: ReconstructionMeta) -> SyntheticTrafficChunk:
        first = packets[0] if packets else None
        last = packets[-1] if packets else None
        start = first.timestamp if first else 0.0
        end = last.timestamp if last else start
        return SyntheticTrafficChunk(
            chunk_id=0,
            packets=packets,
            start_time=start,
            duration=max(0.0, end - start),
            label=meta.label,
            meta=meta,
        )
