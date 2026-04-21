from __future__ import annotations
from abc import ABC, abstractmethod

from typing import List

from ..domain.sample_base import TrafficSample
from ..domain.packet import ParsedPacket

class TrafficAggregatorBase(ABC):

    @abstractmethod
    def aggregate(
        self,
        packets: List[ParsedPacket],
    ) -> List[TrafficSample]:
        pass