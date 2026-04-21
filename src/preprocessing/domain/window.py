from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .sample_base import TrafficSample
from .packet import ParsedPacket

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