from __future__ import annotations

from typing import Optional, List

from .aggregation_base import TrafficAggregatorBase
from ..domain import PacketWindow, ParsedPacket

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