"""
src/reconstruction/sequential/semantic_byte_reconstructor.py
============================================================

Reconstrucción para ProtocolAwareTokenizer. Convierte tokens → paquetes sintéticos con semántica L3/L4.
Estructura esperada por paquete:
  <FWD>/<BWD>
  <L3> ip_version:X ip_proto:Y ip_len:Z ip_ttl:...
  <L4> sport:S dport:D tcp_state:... tcp_win:...
  <PAY> byte:xx byte:yy ...
  <SEP>

Pipeline para ambos:
    decode()      → tokens → bytes brutos por flujo
    heuristics()  → segmentación + puertos + timestamps + flags TCP

"""

from __future__ import annotations

from typing import List, Optional, Any, Dict
from collections import Counter

import torch

from src.reconstruction.base import FlowReconstructor, ReconstructionMeta
from src.data_utils.preprocessing import ParsedPacket
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

    Convierte una secuencia de IDs de tokens en:
      1. tokens string (si hay vocabulario disponible),
      2. paquetes sintéticos con semántica L3/L4,
      3. flujo final vía FlowReconstructor.

    Estructura esperada por paquete:
      <FWD>/<BWD>
      <L3> ip_version:X ip_proto:Y ip_len:Z ip_ttl:...
      <L4> sport:S dport:D tcp_state:... tcp_win:...
      <PAY> byte:xx byte:yy ...
      <SEP>
    """

    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    BOS_TOKEN = "<BOS>"
    EOS_TOKEN = "<EOS>"
    SEP_TOKEN = "<SEP>"
    DIR_FWD = "<FWD>"
    DIR_BWD = "<BWD>"
    L3_SEP = "<L3>"
    L4_SEP = "<L4>"
    PAYLOAD_SEP = "<PAY>"

    TCP_STATES = {
        "SYN": 0x002,
        "SYN-ACK": 0x012,
        "ACK": 0x010,
        "PSH-ACK": 0x018,
        "FIN": 0x001,
        "FIN-ACK": 0x011,
        "RST": 0x004,
        "RST-ACK": 0x014,
        "OTHER": 0x000,
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
        Decodifica una batch de secuencias de IDs a listas de ParsedPacket.

        Si se proporciona vocab/id_to_token:
          IDs -> tokens string -> parseo jerárquico.

        Si no hay vocabulario disponible:
          fallback conservador a bytes crudos.
        """
        if samples.dim() != 2:
            raise ValueError(
                f"ProtocolAwareReconstructor espera (B, L), recibido {tuple(samples.shape)}"
            )

        result: List[List[ParsedPacket]] = []

        for b in range(samples.shape[0]):
            ids = samples[b].detach().cpu().tolist()
            tokens = self._decode_ids(ids)

            if tokens:
                pkts = self._tokens_to_packets(tokens)
            else:
                # Fallback: tratar la secuencia como bytes si no hay decoder de tokens.
                raw_bytes = tokens_to_bytes(ids, vocab_size=self.vocab_size)
                chunks = segment_bytes_into_packets(
                    raw_bytes,
                    max_payload=self.max_payload_bytes,
                    min_payload=1,
                    seed=self.seed,
                )
                pkts = [ParsedPacket(payload_bytes=chunk) for chunk in chunks]

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
            if tok == self.BOS_TOKEN:
                continue
            if tok == self.SEP_TOKEN:
                if current:
                    packets.append(self._parse_packet_tokens(current))
                    current = []
                continue
            current.append(tok)

        if current:
            packets.append(self._parse_packet_tokens(current))

        return packets if packets else [ParsedPacket()]

    def _parse_packet_tokens(self, tokens: List[str]) -> ParsedPacket:
        pkt = ParsedPacket()
        payload = bytearray()
        in_payload = False

        for tok in tokens:
            if tok in (self.DIR_FWD, self.DIR_BWD):
                pkt.direction = 0 if tok == self.DIR_FWD else 1
                continue

            if tok == self.L3_SEP:
                in_payload = False
                continue

            if tok == self.L4_SEP:
                in_payload = False
                continue

            if tok == self.PAYLOAD_SEP:
                in_payload = True
                continue

            if tok == self.UNK_TOKEN:
                continue

            if tok.startswith("ip_version:"):
                self._set_ip_version(pkt, self._safe_int(tok.split(":", 1)[1], default=-1))
                continue

            if tok.startswith("ip_proto:"):
                pkt.ip_proto = self._safe_int(tok.split(":", 1)[1], default=-1)
                continue

            if tok.startswith("ip_len:"):
                pkt.ip_len = self._safe_int(tok.split(":", 1)[1], default=0)
                continue

            if tok.startswith("size_bin"):
                pkt.ip_len = self._size_bin_to_len(tok)
                continue

            if tok.startswith("ip_ttl:"):
                pkt.ip_ttl = self._ttl_from_bucket(tok.split(":", 1)[1])
                continue

            if tok.startswith("sport:"):
                pkt.sport = self._safe_int(tok.split(":", 1)[1], default=0)
                continue

            if tok.startswith("dport:"):
                pkt.dport = self._safe_int(tok.split(":", 1)[1], default=0)
                continue

            if tok.startswith("tcp_state:"):
                state = tok.split(":", 1)[1]
                pkt.tcp_flags = self._tcp_flags_from_state(state)
                continue

            if tok.startswith("tcp_flags:"):
                pkt.tcp_flags = self._safe_int(tok.split(":", 1)[1], default=0)
                continue

            if tok.startswith("tcp_window:"):
                pkt.tcp_window = self._win_from_bucket(tok.split(":", 1)[1])
                continue

            if tok.startswith("udp_len:"):
                pkt.udp_len = self._safe_int(tok.split(":", 1)[1], default=-1)
                continue

            if tok.startswith("payload_len:"):
                # Se usa como señal auxiliar; la longitud real la impone el payload reconstruido.
                continue

            if tok.startswith("byte:"):
                payload.append(self._safe_hex_byte(tok.split(":", 1)[1]))
                continue

            if tok.startswith("pos:"):
                continue

        if payload:
            pkt.payload = bytes(payload)

        if pkt.ip_proto == -1:
            if pkt.tcp_flags != 0 or pkt.tcp_window > 0 or pkt.tcp_seq != -1 or pkt.tcp_ack != -1:
                pkt.ip_proto = 6
            elif pkt.udp_len > 0:
                pkt.ip_proto = 17

        if pkt.ip_len <= 0:
            pkt.ip_len = estimate_packet_length(pkt.payload, pkt.ip_proto if pkt.ip_proto in (1, 6, 17) else 6)

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
        Rellena campos faltantes preservando lo ya decodificado.
        """
        if not packets:
            return packets

        proto = self._infer_flow_protocol(packets)
        src_ip, dst_ip = self._infer_flow_ips(packets)
        sport, dport = self._infer_flow_ports(packets, proto)
        timestamps = generate_timestamps(
            len(packets),
            base_time=self.base_timestamp,
            seed=self.seed,
        )

        for i, pkt in enumerate(packets):
            if not pkt.ip_src:
                pkt.ip_src = src_ip
            if not pkt.ip_dst:
                pkt.ip_dst = dst_ip

            if pkt.sport <= 0:
                pkt.sport = sport
            if pkt.dport <= 0:
                pkt.dport = dport

            if pkt.ip_proto not in (1, 6, 17):
                pkt.ip_proto = proto

            if pkt.ip_ttl <= 0:
                pkt.ip_ttl = 64

            if pkt.ip_len <= 0:
                pkt.ip_len = estimate_packet_length(pkt.payload, pkt.ip_proto)

            if pkt.ip_proto == 6:
                if pkt.tcp_flags == 0:
                    pkt.tcp_flags = infer_tcp_flags(
                        packet_index=i,
                        total_packets=len(packets),
                        has_data=len(pkt.payload) > 0,
                    )
                if pkt.tcp_window <= 0:
                    pkt.tcp_window = 65535

            elif pkt.ip_proto == 17:
                if pkt.udp_len <= 0:
                    pkt.udp_len = 8 + len(pkt.payload)

            pkt.timestamp = timestamps[i]

        return packets

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _infer_flow_protocol(self, packets: List[ParsedPacket]) -> int:
        proto_candidates = [p.ip_proto for p in packets if p.ip_proto in (1, 6, 17)]
        if proto_candidates:
            return Counter(proto_candidates).most_common(1)[0][0]

        # Inferencia por campos L4 si falta ip_proto.
        for pkt in packets:
            if pkt.tcp_flags != 0 or pkt.tcp_seq != -1 or pkt.tcp_ack != -1:
                return 6
            if pkt.udp_len > 0:
                return 17

        return 6

    def _infer_flow_ips(self, packets: List[ParsedPacket]) -> tuple[str, str]:
        src = next((p.ip_src for p in packets if p.ip_src), None)
        dst = next((p.ip_dst for p in packets if p.ip_dst), None)

        if src and dst:
            return src, dst

        return assign_synthetic_ips(seed=self.seed)

    def _infer_flow_ports(self, packets: List[ParsedPacket], proto: int) -> tuple[int, int]:
        sport = next((p.sport for p in packets if p.sport > 0), None)
        dport = next((p.dport for p in packets if p.dport > 0), None)

        if sport and dport:
            return int(sport), int(dport)

        if proto == 6:
            return assign_synthetic_ports(proto=6, seed=self.seed)
        if proto == 17:
            return assign_synthetic_ports(proto=17, seed=self.seed)

        return assign_synthetic_ports(proto=6, seed=self.seed)

    def _set_ip_version(self, pkt: ParsedPacket, version: int) -> None:
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
        bucket = bucket.strip().lower()
        if bucket == "low":
            return 32
        if bucket == "mid64":
            return 64
        if bucket == "mid128":
            return 128
        if bucket == "high":
            return 255
        return 64

    @staticmethod
    def _win_from_bucket(bucket: str) -> int:
        bucket = bucket.strip().lower()
        if bucket == "zero":
            return 0
        if bucket == "small":
            return 512
        if bucket == "medium":
            return 8192
        if bucket == "large":
            return 32768
        if bucket == "max":
            return 65535
        return 65535

    @classmethod
    def _tcp_flags_from_state(cls, state: str) -> int:
        return cls.TCP_STATES.get(state.strip().upper(), 0)

    @staticmethod
    def _size_bin_to_len(token: str) -> int:
        """
        size_binN → longitud aproximada.
        Solo es una heurística para reconstrucción plausible.
        """
        try:
            idx = int(token.replace("size_bin", ""))
        except ValueError:
            return 0

        # Aproximación monotónica en rangos típicos de paquetes
        return max(20, min(1500, 64 + idx * 64))