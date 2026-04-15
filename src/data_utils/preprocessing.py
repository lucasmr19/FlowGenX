"""
data_utils/preprocessing.py
=====================
Pipeline de preprocesado: PCAP → paquetes normalizados → flujos.

Flujo de datos:
  PCAP file
    └─► PCAPReader              (lectura con Scapy, filtrado, muestreo)
          └─► PacketParser      (extracción de campos por capa)
                └─► TrafficAggregator (usar algún tipo de agregador de tráfico)
                      └─► FlowNormalizer   (normalización stateful)
                            └─► FeatureNormalizer   (min-max / z-score por campo)

API de la pipeline para evitar data leakage:

    # ----- datos de entrenamiento -----
    pipeline = PCAPPipeline(normalize=True)
    flows_train = pipeline.fit_process("train.pcap")   # fit + transform

    # ----- datos de validación / test -----
    flows_val  = pipeline.process("val.pcap")          # solo transform
    flows_test = pipeline.process("test.pcap")

Todas las clases son stateless o tienen fit() separado para respetar
la separación train / val / test sin data leakage.
"""

from __future__ import annotations
from abc import ABC, abstractmethod

import copy
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, Union, Iterator

import numpy as np
from ..utils.logger_config import LOGGER

from scapy.all import (
    IP, IPv6, TCP, UDP, ICMP, Ether, Raw,
    rdpcap, PcapReader, Packet,
)


# ---------------------------------------------------------------------------
# Estructuras de datos
# ---------------------------------------------------------------------------

@dataclass
class ParsedPacket:
    """
    Representación normalizada de un paquete de red.

    Todos los campos son opcionales porque no todos los protocolos
    los exponen. Los campos ausentes se representan como None o -1.
    """
    # --- timestamps ---
    timestamp: float = 0.0
    iat: float = 0.0

    # -------------------------------------------------
    # Ethernet
    # -------------------------------------------------
    eth_dhost: Optional[bytes] = None
    eth_shost: Optional[bytes] = None
    eth_ethertype: int = -1

    # -------------------------------------------------
    # IPv4
    # -------------------------------------------------
    ipv4_ver: int = 4
    ipv4_hl: int = -1
    ipv4_tos: int = -1
    ipv4_tl: int = -1
    ipv4_id: int = -1

    ipv4_rbit: int = 0
    ipv4_dfbit: int = 0
    ipv4_mfbit: int = 0
    ipv4_foff: int = 0

    ipv4_ttl: int = -1
    ipv4_proto: int = -1
    ipv4_cksum: int = -1

    ipv4_src: Optional[str] = None
    ipv4_dst: Optional[str] = None

    ipv4_opt: Optional[bytes] = None

    sport: int = 0
    dport: int = 0

    # -------------------------------------------------
    # IPv6
    # -------------------------------------------------
    ipv6_ver: int = 6
    ipv6_tc: int = -1
    ipv6_fl: int = -1
    ipv6_len: int = -1
    ipv6_nh: int = -1
    ipv6_hl: int = -1

    ipv6_src: Optional[str] = None
    ipv6_dst: Optional[str] = None

    # -------------------------------------------------
    # TCP
    # -------------------------------------------------
    tcp_sprt: int = -1
    tcp_dprt: int = -1

    tcp_seq: int = -1
    tcp_ackn: int = -1

    tcp_doff: int = -1
    tcp_res: int = 0

    tcp_ns: int = 0
    tcp_cwr: int = 0
    tcp_ece: int = 0
    tcp_urg: int = 0
    tcp_ackf: int = 0
    tcp_psh: int = 0
    tcp_rst: int = 0
    tcp_syn: int = 0
    tcp_fin: int = 0

    tcp_wsize: int = -1
    tcp_cksum: int = -1
    tcp_urp: int = -1

    tcp_opt: Optional[bytes] = None

    # -------------------------------------------------
    # UDP
    # -------------------------------------------------
    udp_sport: int = -1
    udp_dport: int = -1
    udp_len: int = -1
    udp_cksum: int = -1

    # -------------------------------------------------
    # ICMP
    # -------------------------------------------------
    icmp_type: int = -1
    icmp_code: int = -1
    icmp_cksum: int = -1
    icmp_roh: int = -1

    # -------------------------------------------------
    # Payload
    # -------------------------------------------------
    payload_len: int = 0
    payload_bytes: bytes = b""

    # -------------------------------------------------
    # Meta
    # -------------------------------------------------
    direction: int = 0
    flow_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = self.__dict__.copy()
        d.pop("payload_bytes")
        return d

    # ----- propiedades derivadas -----

    @property
    def ip_version(self) -> int:
        if self.ipv4_src is not None:
            return 4
        if self.ipv6_src is not None:
            return 6
        return -1
    
    # --- Compatibility aliases for reconstruction / evaluation ---

    @property
    def ip_src(self) -> Optional[str]:
        return self.ipv4_src or self.ipv6_src

    @ip_src.setter
    def ip_src(self, value: Optional[str]) -> None:
        self.ipv4_src = value
        if value is not None:
            self.ipv6_src = None


    @property
    def ip_dst(self) -> Optional[str]:
        return self.ipv4_dst or self.ipv6_dst

    @ip_dst.setter
    def ip_dst(self, value: Optional[str]) -> None:
        self.ipv4_dst = value
        if value is not None:
            self.ipv6_dst = None


    @property
    def ip_proto(self) -> int:
        if self.ipv4_proto != -1:
            return self.ipv4_proto
        if self.ipv6_nh != -1:
            return self.ipv6_nh
        return -1

    @ip_proto.setter
    def ip_proto(self, value: int) -> None:
        self.ipv4_proto = value
        self.ipv6_nh = value


    @property
    def ip_len(self) -> int:
        if self.ipv4_tl != -1:
            return self.ipv4_tl
        if self.ipv6_len != -1:
            return self.ipv6_len
        return 0

    @ip_len.setter
    def ip_len(self, value: int) -> None:
        self.ipv4_tl = value
        self.ipv6_len = value


    @property
    def ip_ttl(self) -> int:
        if self.ipv4_ttl != -1:
            return self.ipv4_ttl
        if self.ipv6_hl != -1:
            return self.ipv6_hl
        return -1

    @ip_ttl.setter
    def ip_ttl(self, value: int) -> None:
        self.ipv4_ttl = value
        self.ipv6_hl = value


    @property
    def tcp_ack(self) -> int:
        return self.tcp_ackn

    @tcp_ack.setter
    def tcp_ack(self, value: int) -> None:
        self.tcp_ackn = value


    @property
    def tcp_window(self) -> int:
        return self.tcp_wsize

    @tcp_window.setter
    def tcp_window(self, value: int) -> None:
        self.tcp_wsize = value


    @property
    def payload(self) -> bytes:
        return self.payload_bytes

    @payload.setter
    def payload(self, value: bytes) -> None:
        self.payload_bytes = value or b""
        self.payload_len = len(self.payload_bytes)


    @property
    def tcp_flags(self) -> int:
        flags = 0
        flags |= (self.tcp_fin  << 0)
        flags |= (self.tcp_syn  << 1)
        flags |= (self.tcp_rst  << 2)
        flags |= (self.tcp_psh  << 3)
        flags |= (self.tcp_ackf << 4)
        flags |= (self.tcp_urg  << 5)
        return flags

    @tcp_flags.setter
    def tcp_flags(self, value: int) -> None:
        self.tcp_fin  = (value >> 0) & 1
        self.tcp_syn  = (value >> 1) & 1
        self.tcp_rst  = (value >> 2) & 1
        self.tcp_psh  = (value >> 3) & 1
        self.tcp_ackf = (value >> 4) & 1
        self.tcp_urg  = (value >> 5) & 1


@dataclass(kw_only=True)
class TrafficSample:
    label: Optional[int] = None
    class_name: Optional[str] = None   # "Benign", "Malware", "Skype", etc.
    source: Optional[str] = None       # ruta del pcap origen

@dataclass
class Flow(TrafficSample):
    """
    Agrupa el tráfico de red en flujos bidireccionales: colección ordenada
    de ParsedPacket identificados por la 5-tupla canónica.
    """
    flow_id:    str
    src_ip:     str
    dst_ip:     str
    sport:      int
    dport:      int
    protocol:   int

    packets:    List[ParsedPacket] = field(default_factory=list)
    start_time: float = 0.0
    end_time:   float = 0.0

    stats: Dict[str, float] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def num_packets(self) -> int:
        return len(self.packets)

    @property
    def num_bytes(self) -> int:
        return sum(
            p.ipv4_tl if p.ipv4_tl > 0 else p.ipv6_len
            for p in self.packets
        )

    def __len__(self) -> int:
        return len(self.packets)


@dataclass
class PacketWindow(TrafficSample):
    """
    Agrupa el tráfico de red en ventanas de paquetes consecutivos.
    """
    window_id:  int
    packets:    List[ParsedPacket]
    start_time: float
    end_time:   float

    def __len__(self) -> int:
        return len(self.packets)

@dataclass
class TrafficChunk(TrafficSample):
    """
    Temporal chunk of packets with optional overlap.

    Each chunk contains packets in:
        [start_time, start_time + duration)

    Compatible with Flow / PacketWindow representations.
    """
    chunk_id:    int
    packets:     List[ParsedPacket]
    start_time:  float
    duration:    float

    stats: Dict[str, float] = field(default_factory=dict)

    @property
    def end_time(self) -> float:
        return self.start_time + self.duration

    @property
    def num_packets(self) -> int:
        return len(self.packets)

    @property
    def num_bytes(self) -> int:
        return sum(
            p.ipv4_tl if p.ipv4_tl > 0 else p.ipv6_len
            for p in self.packets
        )

    def __len__(self) -> int:
        return len(self.packets)

    def compute_stats(self) -> None:
        pkts = self.packets

        if not pkts:
            self.stats = {}
            return

        sizes = np.array(
            [p.ipv4_tl if p.ipv4_tl > 0 else p.ipv6_len for p in pkts],
            dtype=np.float32
        )

        iats = np.array([p.iat for p in pkts[1:]], dtype=np.float32)

        self.stats = {
            "num_packets": float(len(pkts)),
            "num_bytes":   float(sizes.sum()),
            "duration":    self.duration,
            "mean_pkt_size": float(sizes.mean()),
            "std_pkt_size":  float(sizes.std()) if len(sizes) > 1 else 0.0,
            "mean_iat":      float(iats.mean()) if len(iats) > 0 else 0.0,
            "std_iat":       float(iats.std())  if len(iats) > 1 else 0.0,
            "fwd_packets":   float(sum(1 for p in pkts if p.direction == 0)),
            "bwd_packets":   float(sum(1 for p in pkts if p.direction == 1)),
        }


# ---------------------------------------------------------------------------
# 1. Lector de PCAPs
# ---------------------------------------------------------------------------

class PCAPReader:
    """
    Lee archivos PCAP y devuelve paquetes Scapy crudos de forma iterativa.

    Parameters
    ----------
    max_packets : int, optional
        Límite de paquetes a leer. None = sin límite.
    protocols   : list of str, optional
        Filtro de protocolos: ["TCP", "UDP", "ICMP"]. None = todos.
    streaming   : bool
        Si True usa PcapReader (streaming), si False rdpcap (en memoria).
    """

    PROTOCOL_MAP = {"TCP": 6, "UDP": 17, "ICMP": 1}

    def __init__(
        self,
        max_packets: Optional[int] = None,
        protocols:   Optional[List[str]] = None,
        streaming:   bool = True,
    ) -> None:
        self.max_packets = max_packets
        self.protocols   = protocols
        self.streaming   = streaming
        self._proto_nums = (
            {self.PROTOCOL_MAP[p] for p in protocols if p in self.PROTOCOL_MAP}
            if protocols else None
        )

    def read(self, pcap_path: Union[str, Path]) -> Iterator[Packet]:
        pcap_path = Path(pcap_path)
        if not pcap_path.exists():
            raise FileNotFoundError(f"PCAP no encontrado: {pcap_path}")

        LOGGER.info("Leyendo PCAP: %s (streaming=%s)", pcap_path, self.streaming)
        count = 0

        reader = PcapReader(str(pcap_path)) if self.streaming else iter(rdpcap(str(pcap_path)))

        for pkt in reader:
            if self.max_packets and count >= self.max_packets:
                break
            if self._passes_filter(pkt):
                yield pkt
                count += 1

        LOGGER.info("Leídos %d paquetes de %s", count, pcap_path.name)

    def read_all(self, pcap_path: Union[str, Path]) -> List[Packet]:
        return list(self.read(pcap_path))

    def _passes_filter(self, pkt: Packet) -> bool:
        if self._proto_nums is None:
            return True
        if IP in pkt:
            return pkt[IP].proto in self._proto_nums
        return False


# ---------------------------------------------------------------------------
# 2. Parser de paquetes
# ---------------------------------------------------------------------------

class PacketParser:
    """
    Convierte paquetes Scapy crudos en ParsedPacket normalizados.
    """

    def __init__(
        self,
        max_payload_bytes: int = 20,
        include_payload:   bool = True,
    ) -> None:
        self.max_payload_bytes = max_payload_bytes
        self.include_payload   = include_payload

    def parse(
        self,
        pkt: Packet,
        prev_timestamp: Optional[float] = None,
    ) -> Optional[ParsedPacket]:
        if IP not in pkt and IPv6 not in pkt:
            return None

        parsed = ParsedPacket()
        parsed.timestamp = float(pkt.time)

        if prev_timestamp is not None:
            parsed.iat = max(0.0, parsed.timestamp - prev_timestamp)

        # Ethernet
        if Ether in pkt:
            eth = pkt[Ether]
            parsed.eth_dhost    = bytes.fromhex(eth.dst.replace(":", ""))
            parsed.eth_shost    = bytes.fromhex(eth.src.replace(":", ""))
            parsed.eth_ethertype = eth.type

        # IPv4
        if IP in pkt:
            ip = pkt[IP]
            parsed.ipv4_ver   = ip.version
            parsed.ipv4_hl    = ip.ihl
            parsed.ipv4_tos   = ip.tos
            parsed.ipv4_tl    = ip.len
            parsed.ipv4_id    = ip.id
            parsed.ipv4_dfbit = int(ip.flags.DF)
            parsed.ipv4_mfbit = int(ip.flags.MF)
            parsed.ipv4_rbit  = 0
            parsed.ipv4_foff  = ip.frag
            parsed.ipv4_ttl   = ip.ttl
            parsed.ipv4_proto = ip.proto
            parsed.ipv4_cksum = ip.chksum
            parsed.ipv4_src   = ip.src
            parsed.ipv4_dst   = ip.dst
            if ip.options:
                parsed.ipv4_opt = (
                    bytes(ip.options)[:40]
                    if isinstance(ip.options, bytes) else None
                )

        # IPv6
        elif IPv6 in pkt:
            ip6 = pkt[IPv6]
            parsed.ipv6_ver = 6
            parsed.ipv6_tc  = ip6.tc
            parsed.ipv6_fl  = ip6.fl
            parsed.ipv6_len = ip6.plen
            parsed.ipv6_nh  = ip6.nh
            parsed.ipv6_hl  = ip6.hlim
            parsed.ipv6_src = ip6.src
            parsed.ipv6_dst = ip6.dst

        # TCP
        if TCP in pkt:
            tcp = pkt[TCP]
            parsed.tcp_sprt  = tcp.sport
            parsed.tcp_dprt  = tcp.dport
            parsed.sport     = tcp.sport
            parsed.dport     = tcp.dport
            parsed.tcp_seq   = tcp.seq
            parsed.tcp_ackn  = tcp.ack
            parsed.tcp_doff  = tcp.dataofs
            parsed.tcp_wsize = tcp.window
            parsed.tcp_cksum = tcp.chksum
            parsed.tcp_urp   = tcp.urgptr

            flags = int(tcp.flags)
            parsed.tcp_ns  = (flags >> 8) & 1
            parsed.tcp_cwr = (flags >> 7) & 1
            parsed.tcp_ece = (flags >> 6) & 1
            parsed.tcp_urg = (flags >> 5) & 1
            parsed.tcp_ackf= (flags >> 4) & 1
            parsed.tcp_psh = (flags >> 3) & 1
            parsed.tcp_rst = (flags >> 2) & 1
            parsed.tcp_syn = (flags >> 1) & 1
            parsed.tcp_fin = flags & 1

            if tcp.dataofs and tcp.dataofs > 5:
                tcp_raw    = bytes(tcp)
                header_len = tcp.dataofs * 4
                options    = tcp_raw[20:header_len]
                parsed.tcp_opt = options[:40]

        # UDP
        elif UDP in pkt:
            udp = pkt[UDP]
            parsed.udp_sport = udp.sport
            parsed.udp_dport = udp.dport
            parsed.udp_len   = udp.len
            parsed.udp_cksum = udp.chksum
            parsed.sport     = udp.sport
            parsed.dport     = udp.dport

        # ICMP
        elif ICMP in pkt:
            icmp = pkt[ICMP]
            parsed.icmp_type  = icmp.type
            parsed.icmp_code  = icmp.code
            parsed.icmp_cksum = icmp.chksum

        # Payload
        if self.include_payload and Raw in pkt:
            raw = bytes(pkt[Raw].load)
            parsed.payload_bytes = raw[:self.max_payload_bytes]
            parsed.payload_len   = len(raw)

        return parsed

    def parse_sequence(self, packets: List[Packet]) -> List[ParsedPacket]:
        result   = []
        prev_ts  = None
        for pkt in packets:
            p = self.parse(pkt, prev_timestamp=prev_ts)
            if p is not None:
                prev_ts = p.timestamp
                result.append(p)
        return result


# ---------------------------------------------------------------------------
# 3. Agregadores de tráfico
# ---------------------------------------------------------------------------

class TrafficAggregatorBase(ABC):

    @abstractmethod
    def aggregate(
        self,
        packets: List[ParsedPacket],
    ) -> List[TrafficSample]:
        pass


class FlowAggregator(TrafficAggregatorBase):
    """
    Agrupa ParsedPackets en flujos bidireccionales usando la 5-tupla canónica.

    Parameters
    ----------
    timeout_s      : segundos de inactividad para cerrar un flujo
    max_pkts_flow  : máximo de paquetes por flujo
    min_pkts_flow  : descarta flujos con menos paquetes
    """

    def __init__(
        self,
        timeout_s:     float = 120.0,
        max_pkts_flow: int   = 100,
        min_pkts_flow: int   = 2,
    ) -> None:
        self.timeout_s     = timeout_s
        self.max_pkts_flow = max_pkts_flow
        self.min_pkts_flow = min_pkts_flow

    def aggregate(self, packets: List[ParsedPacket]) -> List[Flow]:
        flows: Dict[str, Flow] = {}

        for pkt in sorted(packets, key=lambda p: p.timestamp):
            fid         = self._flow_id(pkt)
            pkt.flow_id = fid

            if fid in flows:
                last_ts = flows[fid].end_time
                if pkt.timestamp - last_ts > self.timeout_s:
                    new_fid         = fid + f"_{int(last_ts)}"
                    flows[new_fid]  = flows.pop(fid)

            if fid not in flows:
                flows[fid] = Flow(
                    flow_id    = fid,
                    src_ip     = pkt.ip_src or "",
                    dst_ip     = pkt.ip_dst or "",
                    sport      = pkt.sport,
                    dport      = pkt.dport,
                    protocol   = pkt.ip_proto,
                    start_time = pkt.timestamp,
                )

            flow          = flows[fid]
            pkt.direction = self._direction(pkt, flow)

            if flow.packets:
                pkt.iat = max(0.0, pkt.timestamp - flow.packets[-1].timestamp)

            flow.packets.append(pkt)
            flow.end_time = pkt.timestamp

        valid_flows = [
            f for f in flows.values()
            if len(f) >= self.min_pkts_flow
        ]
        for f in valid_flows:
            self._compute_stats(f)

        valid_flows.sort(key=lambda f: f.start_time)
        LOGGER.info(
            "Flujos extraídos: %d (de %d paquetes)",
            len(valid_flows), len(packets),
        )
        return valid_flows

    @staticmethod
    def _flow_id(pkt: ParsedPacket) -> str:
        fwd = (pkt.ip_src, pkt.ip_dst, pkt.sport, pkt.dport, pkt.ip_proto)
        bwd = (pkt.ip_dst, pkt.ip_src, pkt.dport, pkt.sport, pkt.ip_proto)
        canonical = min(fwd, bwd)
        key = "_".join(str(x) for x in canonical)
        return hashlib.md5(key.encode()).hexdigest()[:16]

    @staticmethod
    def _direction(pkt: ParsedPacket, flow: Flow) -> int:
        return 0 if pkt.ip_src == flow.src_ip else 1

    @staticmethod
    def _compute_stats(flow: Flow) -> None:
        pkts  = flow.packets
        sizes = np.array([p.ip_len for p in pkts],     dtype=np.float32)
        iats  = np.array([p.iat    for p in pkts[1:]], dtype=np.float32)

        flow.stats = {
            "num_packets":   float(len(pkts)),
            "num_bytes":     float(sizes.sum()),
            "duration":      flow.duration,
            "mean_pkt_size": float(sizes.mean()),
            "std_pkt_size":  float(sizes.std())  if len(sizes) > 1 else 0.0,
            "mean_iat":      float(iats.mean())  if len(iats)  > 0 else 0.0,
            "std_iat":       float(iats.std())   if len(iats)  > 1 else 0.0,
            "fwd_packets":   float(sum(1 for p in pkts if p.direction == 0)),
            "bwd_packets":   float(sum(1 for p in pkts if p.direction == 1)),
        }


class PacketWindowAggregator(TrafficAggregatorBase):
    """
    Agrupa paquetes en ventanas secuenciales de tamaño fijo.
    """

    def __init__(
        self,
        window_size: int = 1024,
        stride: Optional[int] = None,
    ) -> None:
        self.window_size = window_size
        self.stride      = stride or window_size

    def aggregate(self, packets: List[ParsedPacket]) -> List[PacketWindow]:
        packets = sorted(packets, key=lambda p: p.timestamp)
        windows = []
        idx     = 0

        for i in range(0, len(packets), self.stride):
            chunk = packets[i : i + self.window_size]
            if len(chunk) < 2:
                continue
            windows.append(PacketWindow(
                window_id  = idx,
                packets    = chunk,
                start_time = chunk[0].timestamp,
                end_time   = chunk[-1].timestamp,
            ))
            idx += 1

        return windows


class TrafficChunkAggregator(TrafficAggregatorBase):
    """
    Temporal chunking with stride (overlapping windows).

    Parameters
    ----------
    chunk_duration : float
        Duration of each chunk (seconds)

    stride : float
        Step between consecutive chunks (seconds)

        - stride < duration → overlap
        - stride = duration → no overlap
        - stride > duration → gaps
    """

    def __init__(
        self,
        chunk_duration: float = 1.0,
        stride: float = None,
        drop_empty: bool = True,
    ):
        self.chunk_duration = chunk_duration
        self.stride = stride if stride is not None else chunk_duration
        self.drop_empty = drop_empty

    def aggregate(self, packets: List[ParsedPacket]) -> List[TrafficChunk]:
        if not packets:
            return []

        # Orden temporal (crítico)
        packets = sorted(packets, key=lambda p: p.timestamp)

        t_start = packets[0].timestamp
        t_end   = packets[-1].timestamp

        chunks: List[TrafficChunk] = []
        chunk_id = 0

        current_start = t_start

        # Sliding window temporal
        while current_start <= t_end:
            current_end = current_start + self.chunk_duration

            # Selección eficiente de paquetes dentro del intervalo
            chunk_packets = [
                p for p in packets
                if current_start <= p.timestamp < current_end
            ]

            if chunk_packets or not self.drop_empty:
                chunk = TrafficChunk(
                    chunk_id=chunk_id,
                    packets=chunk_packets,
                    start_time=current_start,
                    duration=self.chunk_duration
                )
                chunk.compute_stats()
                chunks.append(chunk)
                chunk_id += 1

            current_start += self.stride

        return chunks


# ---------------------------------------------------------------------------
# 4. Normalizador de características
# ---------------------------------------------------------------------------

class FeatureNormalizer:
    """
    Ajusta y aplica normalización (min-max o z-score) sobre arrays numéricos.

    Se ajusta SOLO sobre datos de entrenamiento para evitar data leakage.

    Parameters
    ----------
    method : "minmax" | "zscore"
    clip   : si True, hace clip a [0, 1] tras min-max
    eps    : evita división por cero
    """

    def __init__(
        self,
        method: str   = "minmax",
        clip:   bool  = True,
        eps:    float = 1e-8,
    ) -> None:
        assert method in ("minmax", "zscore"), f"Método desconocido: {method}"
        self.method  = method
        self.clip    = clip
        self.eps     = eps
        self._fitted = False
        self._params: Dict[str, np.ndarray] = {}

    def fit(self, data: np.ndarray) -> "FeatureNormalizer":
        """data shape: (N, F)"""
        if self.method == "minmax":
            self._params["min"] = data.min(axis=0)
            self._params["max"] = data.max(axis=0)
        else:
            self._params["mean"] = data.mean(axis=0)
            self._params["std"]  = data.std(axis=0)
        self._fitted = True
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Llama a fit() antes de transform().")
        data = data.astype(np.float32)
        if self.method == "minmax":
            rng = self._params["max"] - self._params["min"] + self.eps
            out = (data - self._params["min"]) / rng
            if self.clip:
                out = np.clip(out, 0.0, 1.0)
        else:
            out = (data - self._params["mean"]) / (self._params["std"] + self.eps)
        return out

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        return self.fit(data).transform(data)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Llama a fit() antes de inverse_transform().")
        data = data.astype(np.float32)
        if self.method == "minmax":
            rng = self._params["max"] - self._params["min"] + self.eps
            return data * rng + self._params["min"]
        else:
            return data * (self._params["std"] + self.eps) + self._params["mean"]

    def get_params(self) -> Dict[str, np.ndarray]:
        return dict(self._params)

    def set_params(self, params: Dict[str, np.ndarray]) -> None:
        self._params = params
        self._fitted = bool(params)


# ---------------------------------------------------------------------------
# 5. FlowNormalizer
# ---------------------------------------------------------------------------

# Campos continuos de ParsedPacket susceptibles de normalización.
# No se incluyen campos categóricos (puertos, flags, IPs) ni timestamps absolutos.
NUMERIC_PACKET_FIELDS: Tuple[str, ...] = (
    "iat",          # inter-arrival time (segundos)
    "ipv4_tl",      # total length IPv4  (bytes)
    "ipv6_len",     # payload length IPv6 (bytes)
    "tcp_wsize",    # ventana TCP
    "tcp_seq",      # número de secuencia TCP  ← útil para DDPM/Transformer
    "tcp_ackn",     # número de ACK TCP
    "udp_len",      # longitud UDP
    "payload_len",  # bytes de payload
    "ipv4_ttl",     # TTL / hop-limit
)


class FlowNormalizer:
    """
    Normaliza los campos numéricos continuos de los paquetes dentro de
    una lista de TrafficSample (Flow o PacketWindow).

    Envuelve FeatureNormalizer y se encarga de:
    - Extraer los campos numéricos de cada ParsedPacket como matriz (N, F).
    - Ajustar el normalizador con datos de entrenamiento (fit).
    - Escribir los valores normalizados de vuelta en los objetos ParsedPacket
      (transform, en una copia profunda para no mutar los originales).

    API
    ---
    normalizer = FlowNormalizer(method="minmax")
    train_samples = normalizer.fit_transform(train_samples)   # ajusta y normaliza
    test_samples  = normalizer.transform(test_samples)        # solo normaliza

    Parameters
    ----------
    method : "minmax" | "zscore"
        Método de normalización aplicado por FeatureNormalizer.
    fields : tuple of str, optional
        Campos a normalizar. Por defecto usa NUMERIC_PACKET_FIELDS.
    copy   : bool
        Si True (por defecto) opera sobre copias profundas para no mutar
        los objetos originales. Útil al reutilizar el mismo split de datos
        con distintos configuraciones de normalización.
    """

    def __init__(
        self,
        method: str = "minmax",
        fields: Optional[Tuple[str, ...]] = None,
        copy:   bool = True,
    ) -> None:
        self.fields     = fields or NUMERIC_PACKET_FIELDS
        self.copy       = copy
        self._norm      = FeatureNormalizer(method=method)
        self._fitted    = False

    # ------------------------------------------------------------------
    # Acceso al FeatureNormalizer interno (útil para inspección / tests)
    # ------------------------------------------------------------------

    @property
    def feature_normalizer(self) -> FeatureNormalizer:
        return self._norm

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    # ------------------------------------------------------------------
    # Extracción / reinserción de la matriz de características
    # ------------------------------------------------------------------

    def _collect_packets(
        self, samples: List[TrafficSample]
    ) -> List[ParsedPacket]:
        """Devuelve todos los ParsedPacket de la lista de muestras, en orden."""
        pkts: List[ParsedPacket] = []
        for s in samples:
            pkts.extend(s.packets)
        return pkts

    def _to_matrix(self, packets: List[ParsedPacket]) -> np.ndarray:
        """
        (N_packets, len(fields))  float32.
        Campos ausentes o negativos se mapean a 0.0 antes de normalizar.
        """
        rows = []
        for pkt in packets:
            row = []
            for f in self.fields:
                val = getattr(pkt, f, 0)
                # Valores centinela (-1) se tratan como 0 para no contaminar stats
                if val is None or (isinstance(val, (int, float)) and val < 0):
                    val = 0.0
                row.append(float(val))
            rows.append(row)
        if not rows:
            return np.zeros((0, len(self.fields)), dtype=np.float32)
        return np.array(rows, dtype=np.float32)

    def _write_back(
        self,
        packets: List[ParsedPacket],
        matrix:  np.ndarray,
    ) -> None:
        """Escribe los valores normalizados de vuelta en los ParsedPacket."""
        for i, pkt in enumerate(packets):
            for j, f in enumerate(self.fields):
                setattr(pkt, f, float(matrix[i, j]))

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def fit(self, samples: List[TrafficSample]) -> "FlowNormalizer":
        """
        Calcula estadísticas sobre los campos numéricos de TODAS las muestras.
        Solo debe llamarse con datos de entrenamiento.
        """
        pkts   = self._collect_packets(samples)
        matrix = self._to_matrix(pkts)

        if matrix.shape[0] == 0:
            LOGGER.warning("FlowNormalizer.fit(): sin paquetes, normalización desactivada.")
            return self

        self._norm.fit(matrix)
        self._fitted = True

        LOGGER.info(
            "FlowNormalizer ajustado sobre %d paquetes (%d campos).",
            matrix.shape[0], matrix.shape[1],
        )
        return self

    def transform(
        self, samples: List[TrafficSample]
    ) -> List[TrafficSample]:
        """
        Aplica la normalización ajustada.
        Devuelve copias de las muestras si self.copy=True.

        Parameters
        ----------
        samples : lista de Flow o PacketWindow

        Returns
        -------
        Lista de muestras con campos numéricos normalizados.
        """
        if not self._fitted:
            raise RuntimeError(
                "FlowNormalizer no ajustado. Llama a fit() o fit_transform() "
                "con los datos de entrenamiento primero."
            )

        out_samples = copy.deepcopy(samples) if self.copy else samples
        pkts        = self._collect_packets(out_samples)

        if not pkts:
            return out_samples

        matrix     = self._to_matrix(pkts)
        normalized = self._norm.transform(matrix)
        self._write_back(pkts, normalized)

        return out_samples

    def fit_transform(
        self, samples: List[TrafficSample]
    ) -> List[TrafficSample]:
        """Equivalente a fit(samples).transform(samples) en un solo paso."""
        return self.fit(samples).transform(samples)

    def inverse_transform(
        self, samples: List[TrafficSample]
    ) -> List[TrafficSample]:
        """
        Desnormaliza las muestras (útil para reconstruir flujos sintéticos
        antes de pasar por decode() en representaciones visuales).
        """
        if not self._fitted:
            raise RuntimeError("FlowNormalizer no ajustado.")

        out_samples = copy.deepcopy(samples) if self.copy else samples
        pkts        = self._collect_packets(out_samples)
        matrix      = self._to_matrix(pkts)
        restored    = self._norm.inverse_transform(matrix)
        self._write_back(pkts, restored)
        return out_samples

    # ------------------------------------------------------------------
    # Persistencia
    # ------------------------------------------------------------------

    def get_state(self) -> Dict:
        return {
            "fields":        self.fields,
            "norm_params":   self._norm.get_params(),
            "norm_method":   self._norm.method,
            "norm_clip":     self._norm.clip,
            "fitted":        self._fitted,
        }

    @classmethod
    def from_state(cls, state: Dict) -> "FlowNormalizer":
        obj = cls(method=state["norm_method"], fields=state["fields"])
        obj._norm.clip = state["norm_clip"]
        obj._norm.set_params(state["norm_params"])
        obj._fitted = state["fitted"]
        return obj


# ---------------------------------------------------------------------------
# 6. Pipeline completo: PCAP → Flujos (con normalización opcional)
# ---------------------------------------------------------------------------

class PCAPPipeline:
    """
    Orquesta el pipeline completo: PCAP → flujos normalizados.

    Encadena: PCAPReader → PacketParser → Aggregator → FlowNormalizer (opc.)

    API sin data leakage
    --------------------
    El normalizer se ajusta SOLO con datos de entrenamiento:

        # Entrenamiento
        pipeline       = PCAPPipeline(normalize=True)
        flows_train    = pipeline.fit_process("train.pcap")

        # Validación / test (reutiliza el normalizer ya ajustado)
        flows_val      = pipeline.process("val.pcap")
        flows_test     = pipeline.process("test.pcap")

    Si normalize=False (por defecto) el comportamiento es idéntico a la
    versión anterior: devuelve flujos con valores crudos.

    Parameters
    ----------
    aggregator        : clase o instancia de TrafficAggregatorBase.
    max_packets       : límite de paquetes a leer del PCAP.
    protocols         : filtro de protocolos (["TCP", "UDP", ...]).
    max_payload_bytes : bytes de payload a extraer por paquete.
    streaming         : si True usa PcapReader en modo streaming.
    normalize         : si True activa FlowNormalizer.
    norm_method       : "minmax" | "zscore" (solo si normalize=True).
    norm_fields       : campos a normalizar (None = NUMERIC_PACKET_FIELDS).
    norm_copy         : si True, transform() devuelve copias profundas.
    **aggregator_kwargs : argumentos extra pasados al aggregator.
    """

    def __init__(
        self,
        aggregator:         Union[Type[TrafficAggregatorBase], TrafficAggregatorBase] = FlowAggregator,
        max_packets:        Optional[int]              = None,
        protocols:          Optional[List[str]]        = None,
        max_payload_bytes:  int                        = 20,
        streaming:          bool                       = True,
        normalize:          bool                       = False,
        norm_method:        str                        = "minmax",
        norm_fields:        Optional[Tuple[str, ...]]  = None,
        norm_copy:          bool                       = True,
        **aggregator_kwargs,
    ) -> None:
        self.reader = PCAPReader(
            max_packets=max_packets,
            protocols=protocols,
            streaming=streaming,
        )
        self.parser = PacketParser(
            max_payload_bytes=max_payload_bytes,
            include_payload=True,
        )
        self.aggregator = (
            aggregator(**aggregator_kwargs)
            if isinstance(aggregator, type)
            else aggregator
        )

        self.normalize = normalize
        self.normalizer: Optional[FlowNormalizer] = (
            FlowNormalizer(
                method=norm_method,
                fields=norm_fields,
                copy=norm_copy,
            )
            if normalize else None
        )

    # ------------------------------------------------------------------
    # Núcleo interno: PCAP → samples sin normalizar
    # ------------------------------------------------------------------

    def _raw_process(self, pcap_path: Union[str, Path]) -> List[TrafficSample]:
        raw_pkts    = list(self.reader.read(pcap_path))
        parsed_pkts = self.parser.parse_sequence(raw_pkts)
        return self.aggregator.aggregate(parsed_pkts)

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def fit_process(
        self, pcap_path: Union[str, Path]
    ) -> List[TrafficSample]:
        """
        Procesa el PCAP, ajusta el normalizer sobre los datos resultantes
        y devuelve las muestras normalizadas.

        Uso exclusivo con datos de ENTRENAMIENTO para evitar data leakage.

        Returns
        -------
        List[TrafficSample] normalizados (si normalize=True) o crudos.
        """
        samples = self._raw_process(pcap_path)

        if self.normalizer is not None:
            LOGGER.info("fit_process: ajustando FlowNormalizer sobre %d muestras.", len(samples))
            samples = self.normalizer.fit_transform(samples)

        return samples

    def process(
        self, pcap_path: Union[str, Path]
    ) -> List[TrafficSample]:
        """
        Procesa el PCAP y aplica el normalizer ya ajustado (si existe).

        Uso para datos de VALIDACIÓN y TEST.

        Raises
        ------
        RuntimeError si normalize=True pero fit_process() no se ha llamado.

        Returns
        -------
        List[TrafficSample] normalizados (si normalize=True) o crudos.
        """
        samples = self._raw_process(pcap_path)

        if self.normalizer is not None:
            if not self.normalizer.is_fitted:
                raise RuntimeError(
                    "El normalizer no está ajustado. "
                    "Llama a fit_process() con datos de entrenamiento antes de process()."
                )
            LOGGER.info("process: normalizando %d muestras.", len(samples))
            samples = self.normalizer.transform(samples)

        return samples

    def process_directory(
        self,
        pcap_dir:     Union[str, Path],
        glob_pattern: str = "*.pcap",
        fit:          bool = False,
    ) -> List[TrafficSample]:
        """
        Procesa todos los PCAPs de un directorio.

        Parameters
        ----------
        fit : si True llama a fit_process() en el primer PCAP encontrado
              y process() en el resto. Útil cuando todo el directorio es train.
        """
        pcap_dir   = Path(pcap_dir)
        all_samples: List[TrafficSample] = []
        pcap_files  = sorted(pcap_dir.glob(glob_pattern))

        for i, pcap_file in enumerate(pcap_files):
            try:
                if fit and i == 0:
                    samples = self.fit_process(pcap_file)
                else:
                    samples = self.process(pcap_file)
                all_samples.extend(samples)
            except Exception as exc:
                LOGGER.error("Error procesando %s: %s", pcap_file, exc)

        LOGGER.info("Total muestras procesadas: %d", len(all_samples))
        return all_samples

def build_pipeline_from_representation(
    representation,
    **pipeline_kwargs,
) -> PCAPPipeline:
    aggregator_cls = representation.get_default_aggregator()

    return PCAPPipeline(
        aggregator=aggregator_cls,
        **pipeline_kwargs,
    )