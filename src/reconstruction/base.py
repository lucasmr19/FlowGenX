"""
src/reconstruction/base.py
==========================

Contrato base del módulo de reconstrucción.

Arquitectura
------------
El orquestador usa el *perfil de reconstrucción* de cada representación para
decidir qué capas de reparación aplica y con qué agresividad:

    Representación            Invertibility  FlowState  PayloadSynth  Agresividad
    ─────────────────────────────────────────────────────────────────────────────
    FlatTokenizer             PARTIAL        False       False         0.15
    ProtocolAware             PARTIAL        False       False         0.20
    SemanticByte              PARTIAL        True        True          0.50
    GAF / NprintImage         LOSSY          True        True          0.90

Pipeline por muestra:

    decode() → heuristics() → _repair_intra_*() → [payload_synthesis]
             → _repair_inter_*() → _build_container() → _repair_container()

Capas de reparación según agresividad:

    is_structured  (aggressiveness < 0.35)
        intra  → _repair_intra_ranges_only()   solo valida rangos, no sintetiza
        inter  → _repair_timestamps_only()     solo timestamps monótonos
        TCP    → confiamos en decoder/heuristics; NO tocamos seq/ack/flags

    is_moderate  (0.35 ≤ aggressiveness < 0.70)
        intra  → _repair_intra_packet()        rellena campos faltantes
        inter  → _repair_inter_packet()        timestamps + seq naive
        TCP    → _repair_inter_with_flow_state() si needs_flow_state

    is_lossy  (aggressiveness ≥ 0.70)
        intra  → _repair_intra_packet()        rellena + sintetiza
        inter  → _repair_inter_with_flow_state() siempre
        payload→ _synthesize_missing_payloads() si needs_payload_synthesis
"""

from __future__ import annotations

import copy
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Generic, List, Optional, TypeVar, Union

import torch

from src.data_utils.preprocessing import ParsedPacket, Flow, PacketWindow, TrafficChunk
from src.utils.logger_config import LOGGER

ContainerT = TypeVar("ContainerT")
MOD32 = 2**32


# ---------------------------------------------------------------------------
# Perfil de reconstrucción
# ---------------------------------------------------------------------------

class InvertibilityLevel(Enum):
    """
    Grado de invertibilidad de la representación.

    EXACT    : los tokens codifican campos de red biunívocamente.
    PARTIAL  : se recuperan campos L3/L4 pero no todo el payload.
    LOSSY    : representación visual; la heurística ES el método.
    """
    EXACT   = "exact"
    PARTIAL = "partial"
    LOSSY   = "lossy"


@dataclass
class ReconstructionProfile:
    """
    Descriptor del comportamiento de reconstrucción.

    El orquestador lee este perfil para elegir qué capas de reparación
    aplica y con qué intensidad, evitando "inventar" tráfico cuando la
    representación ya lleva semántica suficiente.

    Parámetros
    ----------
    invertibility        : nivel de invertibilidad.
    needs_flow_state     : si True, inter-packet usa la FSM TCP completa
                           (FlowState bidireccional con handshake).
    needs_payload_synthesis : si True y la agresividad es alta, se generan
                           payloads sintéticos donde falten.
    repair_aggressiveness : [0.0, 1.0].
        < 0.35  → is_structured: confiar en decoder, solo rangos.
        [0.35, 0.70) → is_moderate: rellena + seq naive.
        ≥ 0.70  → is_lossy: síntesis completa, FlowState siempre.
    """
    invertibility: InvertibilityLevel
    needs_flow_state: bool = False
    needs_payload_synthesis: bool = False
    repair_aggressiveness: float = 0.5

    _LOW: float = 0.35
    _HIGH: float = 0.70

    @property
    def is_structured(self) -> bool:
        """Representación secuencial estructurada: confiar en el decoder."""
        return self.repair_aggressiveness < self._LOW

    @property
    def is_moderate(self) -> bool:
        return self._LOW <= self.repair_aggressiveness < self._HIGH

    @property
    def is_lossy(self) -> bool:
        """Representación visual o muy comprimida: heurística como método."""
        return self.repair_aggressiveness >= self._HIGH

    def __str__(self) -> str:
        tier = (
            "structured" if self.is_structured
            else ("moderate" if self.is_moderate else "lossy")
        )
        return (
            f"Profile({self.invertibility.value}, tier={tier}, "
            f"fs={self.needs_flow_state}, ps={self.needs_payload_synthesis}, "
            f"agg={self.repair_aggressiveness:.2f})"
        )


# ---------------------------------------------------------------------------
# Contenedores sintéticos
# ---------------------------------------------------------------------------

@dataclass
class SyntheticFlow(Flow):
    flow_id: str = ""
    src_ip: str = "0.0.0.0"
    dst_ip: str = "0.0.0.0"
    sport: int = 0
    dport: int = 0
    protocol: int = 6
    packets: List[ParsedPacket] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0
    meta: Optional["ReconstructionMeta"] = None


@dataclass
class SyntheticPacketWindow(PacketWindow):
    window_id: int = 0
    packets: List[ParsedPacket] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0
    meta: Optional["ReconstructionMeta"] = None


@dataclass
class SyntheticTrafficChunk(TrafficChunk):
    chunk_id: int = 0
    packets: List[ParsedPacket] = field(default_factory=list)
    start_time: float = 0.0
    duration: float = 0.0
    meta: Optional["ReconstructionMeta"] = None


SyntheticSample = Union[SyntheticFlow, SyntheticPacketWindow, SyntheticTrafficChunk]


# ---------------------------------------------------------------------------
# Metadatos de reconstrucción
# ---------------------------------------------------------------------------

@dataclass
class ReconstructionMeta:
    representation_name: str = ""
    model_name: str = ""
    label: int = -1
    warnings: List[str] = field(default_factory=list)
    repair_notes: List[str] = field(default_factory=list)
    profile: Optional[ReconstructionProfile] = None


# ---------------------------------------------------------------------------
# Contrato abstracto
# ---------------------------------------------------------------------------

class BaseReconstructor(ABC, Generic[ContainerT]):
    """
    Contrato base para reconstructores de tráfico.

    Subclases deben implementar:
        decode(samples)                    → List[List[ParsedPacket]]
        heuristics(packets, *, meta)       → List[ParsedPacket]
        _build_container(packets, *, meta) → ContainerT

    Subclases deben sobreescribir:
        profile                            → ReconstructionProfile
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
    # Perfil de reconstrucción — sobreescribir en cada subclase
    # ------------------------------------------------------------------

    @property
    def profile(self) -> ReconstructionProfile:
        """
        Perfil de reconstrucción de esta representación.

        Valor por defecto conservador (PARTIAL, sin FlowState).
        Cada reconstructor concreto debe sobreescribirlo.
        """
        return ReconstructionProfile(
            invertibility=InvertibilityLevel.PARTIAL,
            needs_flow_state=False,
            needs_payload_synthesis=False,
            repair_aggressiveness=0.5,
        )

    # ------------------------------------------------------------------
    # API pública: orquestador driven por perfil
    # ------------------------------------------------------------------

    def reconstruct(
        self,
        samples: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        *,
        already_projected: bool = True,
    ) -> List[ContainerT]:
        """
        Orquesta la reconstrucción adaptando las capas de reparación al perfil.

        Pipeline (varía según profile.repair_aggressiveness):

            is_structured  → intra: solo rangos    | inter: solo timestamps
            is_moderate    → intra: campos falt.   | inter: naive / FlowState
            is_lossy       → intra: síntesis compl.| inter: FlowState + payload
        """
        del already_projected
        prof = self.profile
        decoded = self.decode(samples)

        outputs: List[ContainerT] = []

        for idx, raw_packets in enumerate(decoded):
            label = int(labels[idx].item()) if labels is not None else -1

            meta = ReconstructionMeta(
                representation_name=self.representation_name,
                model_name=self.model_name,
                label=label,
                profile=prof,
            )

            # ── heurísticas de dominio ────────────────────────────────────
            packets = self.heuristics(raw_packets, meta=meta)

            # ── capa 1: intra-packet ──────────────────────────────────────
            if prof.is_structured:
                # Solo rangos: no sintetizar lo que el decoder no generó
                packets = [self._repair_intra_ranges_only(pkt, meta=meta) for pkt in packets]
            else:
                # Rellena campos faltantes + validación completa
                packets = [self._repair_intra_packet(pkt, meta=meta) for pkt in packets]

            # ── síntesis de payload (solo LOSSY con needs_payload_synthesis) ─
            if prof.needs_payload_synthesis and prof.is_lossy:
                packets = self._synthesize_missing_payloads(packets, meta=meta)

            # ── capa 2: inter-packet ──────────────────────────────────────
            if prof.needs_flow_state:
                # FSM TCP bidireccional completa
                packets = self._repair_inter_with_flow_state(packets, meta=meta)
            elif prof.is_structured:
                # Solo timestamps monótonos; NO tocar TCP seq/ack/flags
                packets = self._repair_timestamps_only(packets, meta=meta)
            else:
                # Inter naive (timestamps + seq básico sin FSM)
                packets = self._repair_inter_packet(packets, meta=meta)

            # ── capa 3: contenedor ────────────────────────────────────────
            container = self._build_container(packets, meta=meta)
            container = self._repair_container(container, meta=meta)
            outputs.append(container)

            if self.verbose:
                LOGGER.debug(
                    "[Reconstructor] sample=%d  pkts=%d  label=%d  %s",
                    idx, len(packets), label, prof,
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
    # Capa 1a: intra-packet MÍNIMO (is_structured)
    # ------------------------------------------------------------------

    def _clone_packet(self, pkt: ParsedPacket) -> ParsedPacket:
        return copy.deepcopy(pkt)

    def _repair_intra_ranges_only(
        self,
        pkt: ParsedPacket,
        *,
        meta: ReconstructionMeta,
    ) -> ParsedPacket:
        """
        Validación de rangos sin síntesis.

        Para representaciones estructuradas (Flat, ProtocolAware):
        heuristics() ya estableció todos los campos. Aquí solo garantizamos
        que ningún valor sea absurdo, sin inventar campos vacíos.

        Regla: campo en 0/None → lo dejamos. Campo fuera de rango → clampeamos.
        """
        pkt = self._clone_packet(pkt)
        notes = meta.repair_notes

        # Puertos: solo clampear overflow
        if pkt.sport < 0 or pkt.sport > 65535:
            notes.append(f"sport {pkt.sport} overflow -> 0")
            pkt.sport = 0
        if pkt.dport < 0 or pkt.dport > 65535:
            notes.append(f"dport {pkt.dport} overflow -> 0")
            pkt.dport = 0

        # TTL
        if not (1 <= pkt.ip_ttl <= 255):
            notes.append(f"ip_ttl {pkt.ip_ttl} -> 64")
            pkt.ip_ttl = 64

        # ip_len: solo corregir valores negativos
        if pkt.ip_len < 0:
            proto = pkt.ip_proto if pkt.ip_proto in (1, 6, 17) else 6
            min_len = 40 if proto == 6 else 28
            notes.append(f"ip_len {pkt.ip_len} negativo -> {min_len}")
            pkt.ip_len = min_len

        # TCP: clampear rangos, NO modificar seq/ack/flags si son 0
        if pkt.ip_proto == 6:
            if pkt.tcp_seq < 0:
                pkt.tcp_seq = 0
            pkt.tcp_seq = pkt.tcp_seq % MOD32
            if pkt.tcp_ack < 0:
                pkt.tcp_ack = 0
            pkt.tcp_ack = pkt.tcp_ack % MOD32
            if not (0 <= pkt.tcp_flags <= 255):
                notes.append(f"tcp_flags {pkt.tcp_flags} -> 0")
                pkt.tcp_flags = 0
            if pkt.tcp_window > 65535:
                notes.append(f"tcp_window {pkt.tcp_window} -> 65535")
                pkt.tcp_window = 65535

        # Timestamp
        if not isinstance(pkt.timestamp, (int, float)) or pkt.timestamp < 0:
            notes.append(f"timestamp inválido -> {self.base_timestamp}")
            pkt.timestamp = self.base_timestamp

        return pkt

    # ------------------------------------------------------------------
    # Capa 1b: intra-packet COMPLETO (is_moderate / is_lossy)
    # ------------------------------------------------------------------

    def _repair_intra_packet(
        self,
        pkt: ParsedPacket,
        *,
        meta: ReconstructionMeta,
    ) -> ParsedPacket:
        """
        Intra-packet completo: rellena campos faltantes + valida rangos.
        Para representaciones PARTIAL (SemanticByte) y LOSSY (visual).
        """
        pkt = self._clone_packet(pkt)
        notes = meta.repair_notes

        # IPs
        if not pkt.ip_src:
            pkt.ip_src = "10.0.0.1"
            notes.append("ip_src vacío -> 10.0.0.1")
        if not pkt.ip_dst:
            pkt.ip_dst = "10.0.0.2"
            notes.append("ip_dst vacío -> 10.0.0.2")

        # Protocolo
        proto = pkt.ip_proto
        if proto not in (1, 6, 17):
            notes.append(f"ip_proto {proto} -> 6")
            pkt.ip_proto = 6
            proto = 6

        # Puertos
        if not (0 <= pkt.sport <= 65535):
            notes.append(f"sport {pkt.sport} -> 0")
            pkt.sport = 0
        if not (0 <= pkt.dport <= 65535):
            notes.append(f"dport {pkt.dport} -> 0")
            pkt.dport = 0

        # TTL
        if not (1 <= pkt.ip_ttl <= 255):
            notes.append(f"ip_ttl {pkt.ip_ttl} -> 64")
            pkt.ip_ttl = 64

        # ip_len
        payload_len = len(pkt.payload) if pkt.payload is not None else 0
        min_len = 20 + (20 if proto == 6 else 8 if proto in (17, 1) else 0)
        if pkt.ip_len < min_len:
            expected = min_len + payload_len
            notes.append(f"ip_len {pkt.ip_len} -> {expected}")
            pkt.ip_len = expected

        # TCP
        if proto == 6:
            if not (0 <= pkt.tcp_seq < MOD32):
                notes.append(f"tcp_seq {pkt.tcp_seq} -> 0")
                pkt.tcp_seq = 0
            if not (0 <= pkt.tcp_ack < MOD32):
                notes.append(f"tcp_ack {pkt.tcp_ack} -> 0")
                pkt.tcp_ack = 0
            if not (0 <= pkt.tcp_flags <= 255):
                notes.append(f"tcp_flags {pkt.tcp_flags} -> 0")
                pkt.tcp_flags = 0
            if not (0 < pkt.tcp_window <= 65535):
                notes.append(f"tcp_window {pkt.tcp_window} -> 65535")
                pkt.tcp_window = 65535

        # UDP
        if proto == 17:
            exp_udp = 8 + payload_len
            if pkt.udp_len < 8:
                notes.append(f"udp_len {pkt.udp_len} -> {exp_udp}")
                pkt.udp_len = exp_udp

        # Timestamp
        if not isinstance(pkt.timestamp, (int, float)) or pkt.timestamp < 0:
            notes.append(f"timestamp inválido -> {self.base_timestamp}")
            pkt.timestamp = self.base_timestamp

        return pkt

    # ------------------------------------------------------------------
    # Síntesis de payload
    # ------------------------------------------------------------------

    def _synthesize_missing_payloads(
        self,
        packets: List[ParsedPacket],
        *,
        meta: ReconstructionMeta,
    ) -> List[ParsedPacket]:
        """
        Genera payloads sintéticos cuando ip_len > header_size y el payload
        está vacío o es None.

        Solo se usa con representaciones LOSSY + needs_payload_synthesis=True.
        Genera payloads mínimamente realistas según el puerto destino.
        """
        from src.reconstruction.heuristics import (
            generate_protocol_aware_payload,
            _infer_payload_len_from_packet,
        )
        rng = random.Random(getattr(self, "seed", None))

        for i, pkt in enumerate(packets):
            if getattr(pkt, "payload", None) is not None and len(pkt.payload) > 0:
                continue
            payload_len = _infer_payload_len_from_packet(pkt)
            if payload_len <= 0:
                continue
            dport = int(getattr(pkt, "dport", 0) or 0)
            raw = generate_protocol_aware_payload(dport=dport, position=i, rng=rng)
            # Ajustar longitud exacta
            if len(raw) >= payload_len:
                pkt.payload = raw[:payload_len]
            else:
                pkt.payload = raw + bytes(rng.randint(0, 255) for _ in range(payload_len - len(raw)))
            meta.repair_notes.append(f"pkt[{i}] payload sintetizado: {len(pkt.payload)}B (dport={dport})")

        return packets

    # ------------------------------------------------------------------
    # Capa 2a: inter-packet MÍNIMO (is_structured)
    # ------------------------------------------------------------------

    def _repair_timestamps_only(
        self,
        packets: List[ParsedPacket],
        *,
        meta: ReconstructionMeta,
    ) -> List[ParsedPacket]:
        """
        Solo garantiza timestamps monótonamente crecientes.

        Para is_structured: NO toca TCP seq/ack/flags. El decoder y las
        heuristics ya los establecieron; confiar en ellos.
        """
        if not packets:
            return packets

        repaired: List[ParsedPacket] = []
        t = self.base_timestamp

        for pkt in packets:
            pkt = self._clone_packet(pkt)
            ts = pkt.timestamp
            if not isinstance(ts, (int, float)) or ts <= t:
                t += self.inter_packet_gap
                pkt.timestamp = t
                meta.repair_notes.append(f"ts monotónico: {t:.6f}")
            else:
                t = ts
            repaired.append(pkt)

        return repaired

    # ------------------------------------------------------------------
    # Capa 2b: inter-packet NAIVE (is_moderate sin FlowState)
    # ------------------------------------------------------------------

    def _repair_inter_packet(
        self,
        packets: List[ParsedPacket],
        *,
        meta: ReconstructionMeta,
    ) -> List[ParsedPacket]:
        """
        Inter-packet naive: timestamps monótonos + seq TCP básico (unidireccional).
        Sin FSM completa. Para is_moderate sin needs_flow_state.
        """
        if not packets:
            return packets

        repaired: List[ParsedPacket] = []
        t = self.base_timestamp
        next_seq = 1000

        for pkt in packets:
            pkt = self._clone_packet(pkt)
            ts = pkt.timestamp
            if not isinstance(ts, (int, float)) or ts <= t:
                t += self.inter_packet_gap
                pkt.timestamp = t
                meta.repair_notes.append(f"ts ajustado: {t:.6f}")
            else:
                t = ts

            if pkt.ip_proto == 6:
                if pkt.tcp_seq <= 0:
                    pkt.tcp_seq = next_seq
                payload_len = len(pkt.payload) if pkt.payload is not None else 0
                next_seq = (pkt.tcp_seq + max(payload_len, 1)) % MOD32

            repaired.append(pkt)

        return repaired

    # ------------------------------------------------------------------
    # Capa 2c: inter-packet con FSM TCP (needs_flow_state)
    # ------------------------------------------------------------------

    def _repair_inter_with_flow_state(
        self,
        packets: List[ParsedPacket],
        *,
        meta: ReconstructionMeta,
    ) -> List[ParsedPacket]:
        """
        Inter-packet con FSM TCP bidireccional completa (FlowState).

        Reconstruye un flujo TCP semánticamente válido:
          pkt[0] → SYN
          pkt[1] → SYN-ACK
          pkt[2] → ACK (establecido)
          pkt[3..n-2] → PSH-ACK con datos
          pkt[n-1] → FIN-ACK (cierre)

        Para flujos no-TCP: cae a _repair_timestamps_only.

        Usado por: SemanticByte (moderate), GAF, NprintImage (lossy).
        """
        from src.reconstruction.heuristics import (
            FlowState,
            _canonical_flow_key,
            _is_valid_ip_str,
            _clamp_port,
            assign_synthetic_ips,
            assign_synthetic_ports,
        )

        if not packets:
            return packets

        # ── Protocolo del flujo ──────────────────────────────────────────
        # Determinar por mayoría; si no es TCP, fallback a timestamps.
        from collections import Counter
        proto_votes = Counter(
            p.ip_proto for p in packets if getattr(p, "ip_proto", None) in (1, 6, 17)
        )
        proto = proto_votes.most_common(1)[0][0] if proto_votes else 6

        if proto != 6:
            meta.repair_notes.append(f"FlowState: proto={proto} (no-TCP) → solo timestamps")
            return self._repair_timestamps_only(packets, meta=meta)

        # ── Endpoints del flujo ──────────────────────────────────────────
        first = packets[0]
        seed = getattr(self, "seed", None)

        src_ip = getattr(first, "ip_src", None)
        dst_ip = getattr(first, "ip_dst", None)
        if not _is_valid_ip_str(src_ip) or not _is_valid_ip_str(dst_ip):
            src_ip, dst_ip = assign_synthetic_ips(seed=seed)
            meta.repair_notes.append("IPs inválidas → sintéticas para FlowState")

        sport = _clamp_port(getattr(first, "sport", 0), 0)
        dport = _clamp_port(getattr(first, "dport", 0), 0)
        if sport == 0 or dport == 0:
            s_fb, d_fb = assign_synthetic_ports(proto=6, seed=seed)
            sport = sport or s_fb
            dport = dport or d_fb
            meta.repair_notes.append(f"puertos incompletos → sport={sport} dport={dport}")

        # ── Construir FlowState ──────────────────────────────────────────
        flow_key = _canonical_flow_key(first)
        state = FlowState(
            flow_key=flow_key,
            initiator_ip=str(src_ip),
            initiator_port=int(sport),
            responder_ip=str(dst_ip),
            responder_port=int(dport),
            protocol=6,
            rng_seed=seed,
        )

        # ── Aplicar FSM paquete a paquete ────────────────────────────────
        repaired: List[ParsedPacket] = []
        t = self.base_timestamp
        n = len(packets)

        for i, pkt in enumerate(packets):
            pkt = self._clone_packet(pkt)

            # Timestamp (independiente de la FSM)
            ts = pkt.timestamp
            if not isinstance(ts, (int, float)) or ts <= t:
                t += self.inter_packet_gap
                pkt.timestamp = t
            else:
                t = ts

            pkt.ip_proto = 6
            pkt = state.repair_tcp_packet(
                pkt,
                position=i,
                total=n,
                timestamp=t,
                meta=meta,
                force_close_last=True,
            )
            repaired.append(pkt)

        meta.repair_notes.append(
            f"FlowState TCP aplicado: {n} pkts | "
            f"{src_ip}:{sport} → {dst_ip}:{dport}"
        )
        return repaired

    # ------------------------------------------------------------------
    # Capa 3: reparación de contenedor (no-op por defecto)
    # ------------------------------------------------------------------

    def _repair_container(
        self,
        container: ContainerT,
        *,
        meta: ReconstructionMeta,
    ) -> ContainerT:
        """Capa 3: coherencia a nivel de contenedor. No-op por defecto."""
        return container

    def _default_meta(self, label: int = -1) -> ReconstructionMeta:
        return ReconstructionMeta(
            representation_name=self.representation_name,
            model_name=self.model_name,
            label=label,
            profile=self.profile,
        )


# ---------------------------------------------------------------------------
# Mixins de contenedor
# ---------------------------------------------------------------------------

class FlowReconstructor(BaseReconstructor[SyntheticFlow]):
    """Mixin → _build_container produce SyntheticFlow."""

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
    """Mixin → _build_container produce SyntheticPacketWindow."""

    def _build_container(
        self,
        packets: List[ParsedPacket],
        *,
        meta: ReconstructionMeta,
    ) -> SyntheticPacketWindow:
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
    """Mixin → _build_container produce SyntheticTrafficChunk."""

    def _build_container(
        self,
        packets: List[ParsedPacket],
        *,
        meta: ReconstructionMeta,
    ) -> SyntheticTrafficChunk:
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