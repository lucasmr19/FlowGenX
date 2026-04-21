"""
src/reconstruction/sequential/protocol_aware_reconstructor.py
=============================================================

Reconstrucción desde ProtocolAwareTokenizer.

Perfil: PARTIAL | agresividad=0.20 (is_structured)
───────────────────────────────────────────────────
ProtocolAwareTokenizer codifica campos L3/L4 como tokens nombrados:
  <FWD>/<BWD> <L3> ip_version:4 ip_proto:6 size_bin31 <L4> sport:80 ...

El decoder extrae directamente campos de red. heuristics() solo rellena
los campos que el modelo no generó (missing, no sobreescribe los presentes).

Por eso:
  - decode()              parsea tokens → campos de red.
  - heuristics()          rellena huecos; preserva lo decodificado.
  - _repair_intra_*       solo rangos (no síntesis).
  - _repair_inter_*       solo timestamps (no toca TCP seq/ack/flags).
  - needs_flow_state=False: el decoder ya generó la estructura del flujo.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional

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
)
from src.utils.logger_config import LOGGER


class ProtocolAwareReconstructor(FlowReconstructor):
    """
    Reconstrucción desde ProtocolAwareTokenizer.

    Estructura esperada por paquete:
        <FWD>/<BWD>
        <L3> ip_version:X ip_proto:Y ip_len:Z ip_ttl:...
        <L4> sport:S dport:D tcp_state:... tcp_win:...
        <PAY> byte:xx byte:yy ...
        <SEP>

    El decoder extrae campos de red de los tokens. heuristics() solo
    rellena los que el modelo no generó. La reparación es mínima.
    """

    # ── Perfil de reconstrucción ──────────────────────────────────────────
    @property
    def profile(self) -> ReconstructionProfile:
        return ReconstructionProfile(
            invertibility=InvertibilityLevel.PARTIAL,
            needs_flow_state=False,
            needs_payload_synthesis=False,
            repair_aggressiveness=0.20,   # is_structured
        )

    # ── Tokens especiales ─────────────────────────────────────────────────
    PAD_TOKEN  = "<PAD>"
    UNK_TOKEN  = "<UNK>"
    BOS_TOKEN  = "<BOS>"
    EOS_TOKEN  = "<EOS>"
    SEP_TOKEN  = "<SEP>"
    DIR_FWD    = "<FWD>"
    DIR_BWD    = "<BWD>"
    L3_SEP     = "<L3>"
    L4_SEP     = "<L4>"
    PAYLOAD_SEP = "<PAY>"
    # SemanticByte adicionales (tolerados si aparecen en este decoder también)
    SEM_TOKEN  = "<SEM>"
    BYTES_SEP  = "<BYTES>"

    TCP_STATES: Dict[str, int] = {
        "SYN":     0x002,
        "SYN-ACK": 0x012,
        "ACK":     0x010,
        "PSH-ACK": 0x018,
        "FIN":     0x001,
        "FIN-ACK": 0x011,
        "RST":     0x004,
        "RST-ACK": 0x014,
        "OTHER":   0x000,
    }

    def __init__(
        self,
        vocab: Any = None,
        id_to_token: Optional[Dict[int, str]] = None,
        vocab_size: int = 10_000,
        max_payload_bytes: int = 1460,
        seed: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.vocab = vocab
        self.id_to_token = id_to_token or {}
        self.vocab_size = int(getattr(vocab, "vocab_size", vocab_size))
        self.max_payload_bytes = max_payload_bytes
        self.seed = seed

    # ------------------------------------------------------------------
    # decode
    # ------------------------------------------------------------------

    def decode(self, samples: torch.Tensor) -> List[List[ParsedPacket]]:
        """
        (B, L) de IDs → List[List[ParsedPacket]].

        Si hay vocabulario disponible: IDs → tokens string → parseo jerárquico.
        Si no: fallback conservador a bytes crudos.
        """
        if samples.dim() != 2:
            raise ValueError(
                f"ProtocolAwareReconstructor espera (B, L), recibido {tuple(samples.shape)}"
            )

        result: List[List[ParsedPacket]] = []

        for b in range(samples.shape[0]):
            ids    = samples[b].detach().cpu().tolist()
            tokens = self._decode_ids(ids)

            if tokens:
                pkts = self._tokens_to_packets(tokens)
            else:
                raw_bytes = tokens_to_bytes(ids, vocab_size=self.vocab_size)
                chunks    = segment_bytes_into_packets(
                    raw_bytes,
                    max_payload=self.max_payload_bytes,
                    min_payload=1,
                    seed=self.seed,
                )
                pkts = [ParsedPacket(payload_bytes=c) for c in chunks]

            result.append(pkts)

        return result

    def _decode_ids(self, ids: List[int]) -> List[str]:
        if self.vocab is not None and hasattr(self.vocab, "decode_sequence"):
            try:
                return list(self.vocab.decode_sequence(ids))
            except Exception:
                pass
        if self.id_to_token:
            return [self.id_to_token.get(int(i), self.UNK_TOKEN) for i in ids]
        return []

    def _tokens_to_packets(self, tokens: List[str]) -> List[ParsedPacket]:
        packets: List[ParsedPacket] = []
        current: List[str] = []

        for tok in tokens:
            if tok in (self.PAD_TOKEN, self.EOS_TOKEN):
                break
            if tok in (self.BOS_TOKEN, self.SEM_TOKEN):
                continue
            if tok == self.SEP_TOKEN:
                if current:
                    packets.append(self._parse_packet_tokens(current))
                    current = []
                continue
            current.append(tok)

        if current:
            packets.append(self._parse_packet_tokens(current))

        return packets or [ParsedPacket()]

    def _parse_packet_tokens(self, tokens: List[str]) -> ParsedPacket:
        pkt = ParsedPacket()
        payload = bytearray()
        in_payload = False

        for tok in tokens:
            if tok in (self.DIR_FWD, self.DIR_BWD):
                pkt.direction = 0 if tok == self.DIR_FWD else 1
            elif tok in (self.L3_SEP, self.L4_SEP, self.BYTES_SEP):
                in_payload = False
            elif tok == self.PAYLOAD_SEP:
                in_payload = True
            elif tok == self.UNK_TOKEN:
                pass
            elif tok.startswith("ip_version:"):
                self._set_ip_version(pkt, self._safe_int(tok.split(":", 1)[1], -1))
            elif tok.startswith("ip_proto:"):
                pkt.ip_proto = self._safe_int(tok.split(":", 1)[1], -1)
            elif tok.startswith("ip_len:"):
                pkt.ip_len = self._safe_int(tok.split(":", 1)[1], 0)
            elif tok.startswith("size_bin"):
                pkt.ip_len = self._size_bin_to_len(tok)
            elif tok.startswith("ip_ttl:"):
                pkt.ip_ttl = self._ttl_from_bucket(tok.split(":", 1)[1])
            elif tok.startswith("sport:"):
                pkt.sport = self._safe_int(tok.split(":", 1)[1], 0)
            elif tok.startswith("dport:"):
                pkt.dport = self._safe_int(tok.split(":", 1)[1], 0)
            elif tok.startswith("tcp_state:"):
                pkt.tcp_flags = self._tcp_flags_from_state(tok.split(":", 1)[1])
            elif tok.startswith("tcp_flags:"):
                pkt.tcp_flags = self._safe_int(tok.split(":", 1)[1], 0)
            elif tok.startswith("tcp_window:"):
                pkt.tcp_window = self._win_from_bucket(tok.split(":", 1)[1])
            elif tok.startswith("udp_len:"):
                pkt.udp_len = self._safe_int(tok.split(":", 1)[1], -1)
            elif tok.startswith("byte:"):
                payload.append(self._safe_hex_byte(tok.split(":", 1)[1]))
            elif tok.startswith("pos:"):
                pass  # posición del byte; ignorar en reconstrucción

        if payload:
            pkt.payload = bytes(payload)

        # Inferir protocolo si no se decodificó
        if pkt.ip_proto == -1:
            if pkt.tcp_flags != 0 or pkt.tcp_window > 0:
                pkt.ip_proto = 6
            elif pkt.udp_len > 0:
                pkt.ip_proto = 17

        # ip_len mínimo si no se decodificó
        if pkt.ip_len <= 0:
            pkt.ip_len = estimate_packet_length(
                pkt.payload or b"",
                pkt.ip_proto if pkt.ip_proto in (1, 6, 17) else 6,
            )

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
        Rellena campos faltantes preservando lo que el decoder extrajo.

        Política: solo escribir si el campo está vacío / inválido.
        NO sobreescribir lo que el modelo generó; la reparación posterior
        solo validará rangos.
        """
        if not packets:
            return packets

        proto     = self._infer_flow_protocol(packets)
        src_ip, dst_ip = self._infer_flow_ips(packets)
        sport, dport   = self._infer_flow_ports(packets, proto)
        timestamps = generate_timestamps(
            len(packets), base_time=self.base_timestamp, seed=self.seed
        )

        for i, pkt in enumerate(packets):
            # IPs: solo rellenar si vacías
            if not pkt.ip_src:
                pkt.ip_src = src_ip
            if not pkt.ip_dst:
                pkt.ip_dst = dst_ip

            # Puertos: solo rellenar si ausentes
            if pkt.sport <= 0:
                pkt.sport = sport
            if pkt.dport <= 0:
                pkt.dport = dport

            # Protocolo: solo rellenar si inválido
            if pkt.ip_proto not in (1, 6, 17):
                pkt.ip_proto = proto

            # TTL: solo rellenar si ausente
            if pkt.ip_ttl <= 0:
                pkt.ip_ttl = 64

            # ip_len: solo si ausente
            if pkt.ip_len <= 0:
                pkt.ip_len = estimate_packet_length(pkt.payload or b"", pkt.ip_proto)

            # TCP: solo flags si ausentes (el decoder puede haberlos seteado)
            if pkt.ip_proto == 6:
                if pkt.tcp_flags == 0:
                    pkt.tcp_flags = infer_tcp_flags(i, len(packets), has_data=bool(pkt.payload))
                if pkt.tcp_window <= 0:
                    pkt.tcp_window = 65535

            # UDP: solo si ausente
            elif pkt.ip_proto == 17:
                if pkt.udp_len <= 0:
                    pkt.udp_len = 8 + len(pkt.payload or b"")

            # Timestamp siempre (lo gestiona la capa inter; aquí ponemos base)
            pkt.timestamp = timestamps[i]

        return packets

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _infer_flow_protocol(self, packets: List[ParsedPacket]) -> int:
        candidates = [p.ip_proto for p in packets if p.ip_proto in (1, 6, 17)]
        if candidates:
            return Counter(candidates).most_common(1)[0][0]
        for pkt in packets:
            if pkt.tcp_flags != 0 or pkt.tcp_window > 0:
                return 6
            if pkt.udp_len > 0:
                return 17
        return 6

    def _infer_flow_ips(self, packets: List[ParsedPacket]):
        src = next((p.ip_src for p in packets if p.ip_src), None)
        dst = next((p.ip_dst for p in packets if p.ip_dst), None)
        if src and dst:
            return src, dst
        return assign_synthetic_ips(seed=self.seed)

    def _infer_flow_ports(self, packets: List[ParsedPacket], proto: int):
        sport = next((p.sport for p in packets if p.sport > 0), None)
        dport = next((p.dport for p in packets if p.dport > 0), None)
        if sport and dport:
            return int(sport), int(dport)
        return assign_synthetic_ports(proto=proto, seed=self.seed)

    @staticmethod
    def _set_ip_version(pkt: ParsedPacket, version: int) -> None:
        if version == 4:
            pkt.ipv4_ver = 4
        elif version == 6:
            pkt.ipv6_ver = 6

    @staticmethod
    def _safe_int(value: str, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _safe_hex_byte(value: str) -> int:
        try:
            return int(value, 16) & 0xFF
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _ttl_from_bucket(bucket: str) -> int:
        return {"low": 32, "mid64": 64, "mid128": 128, "high": 255}.get(
            bucket.strip().lower(), 64
        )

    @staticmethod
    def _win_from_bucket(bucket: str) -> int:
        return {"zero": 0, "small": 512, "medium": 8192, "large": 32768, "max": 65535}.get(
            bucket.strip().lower(), 65535
        )

    @classmethod
    def _tcp_flags_from_state(cls, state: str) -> int:
        return cls.TCP_STATES.get(state.strip().upper(), 0)

    @staticmethod
    def _size_bin_to_len(token: str) -> int:
        try:
            idx = int(token.replace("size_bin", ""))
        except ValueError:
            return 0
        return max(20, min(1500, 64 + idx * 64))