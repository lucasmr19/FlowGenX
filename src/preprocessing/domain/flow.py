from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from .sample_base import TrafficSample
from .packet import ParsedPacket

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