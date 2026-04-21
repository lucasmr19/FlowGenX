from __future__ import annotations

from typing import List

from .aggregation_base import TrafficAggregatorBase
from ..domain.chunk import TrafficChunk
from ..domain.packet import ParsedPacket

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
        timestamps = [p.timestamp for p in packets]

        j = 0  # puntero izquierdo (inicio ventana)
        n = len(packets)

        current_start = t_start

        while current_start <= t_end:
            current_end = current_start + self.chunk_duration

            # Avanzar j hasta el primer paquete dentro de la ventana
            while j < n and timestamps[j] < current_start:
                j += 1

            # k parte desde j
            k = j
            while k < n and packets[k].timestamp < current_end:
                k += 1

            chunk_packets = packets[j:k]

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