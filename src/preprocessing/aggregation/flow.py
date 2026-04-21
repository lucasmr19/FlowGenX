from __future__ import annotations

import hashlib
from typing import Dict, List
import numpy as np

from ...utils.logger_config import LOGGER
from .aggregation_base import TrafficAggregatorBase
from ..domain.flow import Flow
from ..domain.packet import ParsedPacket

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
        min_pkts_flow: int   = 1,
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
                    # cerrar flujo actual
                    old_flow = flows[fid]

                    # crear nuevo ID único
                    fid = fid + f"_{int(pkt.timestamp)}"

                    flows[fid] = Flow(
                        flow_id=fid,
                        src_ip=pkt.ip_src or "",
                        dst_ip=pkt.ip_dst or "",
                        sport=pkt.sport,
                        dport=pkt.dport,
                        protocol=pkt.ip_proto,
                        start_time=pkt.timestamp,
                    )

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