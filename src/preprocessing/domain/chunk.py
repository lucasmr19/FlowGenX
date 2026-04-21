from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List
import numpy as np

from .sample_base import TrafficSample
from .packet import ParsedPacket

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