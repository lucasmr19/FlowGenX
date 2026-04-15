# src/evaluation/traffic_structural.py

from __future__ import annotations

from typing import List, Union
import numpy as np

from src.evaluation.base import BaseEvaluator, EvaluationReport, EvaluationResult
from src.data_utils.preprocessing import ParsedPacket


FlowLike = Union[List[ParsedPacket], object]  # object = Flow


class TrafficStructuralEvaluator(BaseEvaluator):
    """
    Evaluador estructural a nivel de tráfico real (post-reconstruction).

    Entrada:
        - List[Flow] o List[List[ParsedPacket]]

    Métricas:
        - packet_validity_rate
        - flow_coherence_score
        - tcp_handshake_validity_rate
        - protocol_distribution_distance
        - packet_size_realism
    """

    def __init__(self) -> None:
        super().__init__(
            name="TrafficStructuralEvaluator",
            category="traffic"
        )

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        real: List[FlowLike],
        synthetic: List[FlowLike],
        **kwargs,
    ) -> EvaluationReport:

        report = EvaluationReport(
            evaluator_name=self.name,
            category=self.category
        )

        real_packets = self._flatten(real)
        synth_packets = self._flatten(synthetic)

        # ------------------------------------------------------------------
        # 1. Packet validity
        # ------------------------------------------------------------------
        report.results.append(self._packet_validity(synth_packets))

        # ------------------------------------------------------------------
        # 2. Flow coherence
        # ------------------------------------------------------------------
        report.results.append(self._flow_coherence(synthetic))

        # ------------------------------------------------------------------
        # 3. TCP handshake validity
        # ------------------------------------------------------------------
        report.results.append(self._tcp_handshake_rate(synthetic))

        # ------------------------------------------------------------------
        # 4. Protocol distribution
        # ------------------------------------------------------------------
        report.results.append(
            self._protocol_distribution(real_packets, synth_packets)
        )

        # ------------------------------------------------------------------
        # 5. Packet size realism
        # ------------------------------------------------------------------
        report.results.append(
            self._packet_size_realism(real_packets, synth_packets)
        )

        return report

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _flatten(self, flows: List[FlowLike]) -> List[ParsedPacket]:
        packets = []
        for f in flows:
            if hasattr(f, "packets"):
                packets.extend(f.packets)
            else:
                packets.extend(f)
        return packets

    # ------------------------------------------------------------------
    # Métricas
    # ------------------------------------------------------------------

    def _packet_validity(self, packets: List[ParsedPacket]) -> EvaluationResult:
        valid = 0

        for pkt in packets:
            if (
                pkt.ip_src
                and pkt.ip_dst
                and 0 <= pkt.sport <= 65535
                and 0 <= pkt.dport <= 65535
                and pkt.ip_proto in (1, 6, 17)
            ):
                valid += 1

        rate = valid / max(len(packets), 1)

        return EvaluationResult(
            metric_name="packet_validity_rate",
            value=float(rate),
            metadata={"n_packets": len(packets)},
        )

    # ------------------------------------------------------------------

    def _flow_coherence(self, flows: List[FlowLike]) -> EvaluationResult:
        scores = []

        for f in flows:
            packets = f.packets if hasattr(f, "packets") else f
            if len(packets) < 2:
                continue

            timestamps = [pkt.timestamp for pkt in packets]
            monotonic = np.all(np.diff(timestamps) > 0)

            scores.append(1.0 if monotonic else 0.0)

        return EvaluationResult(
            metric_name="flow_coherence_score",
            value=float(np.mean(scores) if scores else 0.0),
        )

    # ------------------------------------------------------------------

    def _tcp_handshake_rate(self, flows: List[FlowLike]) -> EvaluationResult:
        valid = 0
        total = 0

        for f in flows:
            packets = f.packets if hasattr(f, "packets") else f

            syn = any(getattr(p, "tcp_syn", 0) for p in packets)
            ack = any(getattr(p, "tcp_ackf", 0) for p in packets)

            if syn:
                total += 1
                if ack:
                    valid += 1

        rate = valid / max(total, 1)

        return EvaluationResult(
            metric_name="tcp_handshake_validity_rate",
            value=float(rate),
            metadata={"n_tcp_flows": total},
        )

    # ------------------------------------------------------------------

    def _protocol_distribution(
        self,
        real_packets: List[ParsedPacket],
        synth_packets: List[ParsedPacket],
    ) -> EvaluationResult:

        def get_dist(packets):
            counts = {1: 0, 6: 0, 17: 0}
            for p in packets:
                if p.ip_proto in counts:
                    counts[p.ip_proto] += 1

            total = sum(counts.values()) + 1e-8
            return np.array([counts[k] / total for k in (1, 6, 17)])

        p_real = get_dist(real_packets)
        p_synth = get_dist(synth_packets)

        kl = np.sum(p_real * np.log((p_real + 1e-8) / (p_synth + 1e-8)))

        return EvaluationResult(
            metric_name="protocol_kl_divergence",
            value=float(kl),
        )

    # ------------------------------------------------------------------

    def _packet_size_realism(
        self,
        real_packets: List[ParsedPacket],
        synth_packets: List[ParsedPacket],
    ) -> EvaluationResult:

        real_sizes = [p.ip_len for p in real_packets if p.ip_len > 0]
        synth_sizes = [p.ip_len for p in synth_packets if p.ip_len > 0]

        if not real_sizes or not synth_sizes:
            return EvaluationResult(
                metric_name="packet_size_mse",
                value=0.0,
            )

        mean_real = np.mean(real_sizes)
        mean_synth = np.mean(synth_sizes)

        mse = (mean_real - mean_synth) ** 2

        return EvaluationResult(
            metric_name="packet_size_mse",
            value=float(mse),
            metadata={
                "mean_real": float(mean_real),
                "mean_synth": float(mean_synth),
            },
        )