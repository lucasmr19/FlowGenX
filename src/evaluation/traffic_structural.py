from __future__ import annotations

from typing import List, Union
import numpy as np
from scipy.stats import wasserstein_distance

from src.evaluation.base import BaseEvaluator, EvaluationReport, EvaluationResult
from src.data_utils.preprocessing import ParsedPacket


FlowLike = Union[List[ParsedPacket], object]


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
        - inter_arrival_time_emd
        - bytes_per_flow_emd
        - flow_duration_emd
        - flow_packet_count_kl
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

        # ------------------------------------------------------------------
        # 6. Temporal realism
        # ------------------------------------------------------------------
        report.results.append(
            self._inter_arrival_time_distribution(real, synthetic)
        )

        # ------------------------------------------------------------------
        # 7. Bytes per flow
        # ------------------------------------------------------------------
        report.results.append(
            self._bytes_per_flow_distribution(real, synthetic)
        )

        # ------------------------------------------------------------------
        # 8. Flow duration
        # ------------------------------------------------------------------
        report.results.append(
            self._flow_duration_distribution(real, synthetic)
        )

        # ------------------------------------------------------------------
        # 9. Packet count distribution
        # ------------------------------------------------------------------
        report.results.append(
            self._flow_packet_count_distribution(real, synthetic)
        )

        return report

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_packets(self, flow: FlowLike) -> List[ParsedPacket]:
        if flow is None:
            return []
        packets = getattr(flow, "packets", flow)
        try:
            return list(packets)
        except TypeError:
            return []

    def _flatten(self, flows: List[FlowLike]) -> List[ParsedPacket]:
        packets = []
        for f in flows:
            packets.extend(self._get_packets(f))
        return packets

    def _flow_stats(self, flows: List[FlowLike]):
        """
        Devuelve estadísticas por flujo:
          - packet_counts
          - durations
          - total_bytes
          - iats
        """
        packet_counts = []
        durations = []
        total_bytes = []
        iats = []

        for f in flows:
            packets = self._get_packets(f)
            if not packets:
                continue

            packet_counts.append(len(packets))

            # Bytes totales por flujo
            flow_bytes = 0.0
            timestamps = []

            for pkt in packets:
                ip_len = getattr(pkt, "ip_len", None)
                if ip_len is not None:
                    try:
                        ip_len = float(ip_len)
                        if np.isfinite(ip_len) and ip_len > 0:
                            flow_bytes += ip_len
                    except (TypeError, ValueError):
                        pass

                ts = getattr(pkt, "timestamp", None)
                if ts is not None:
                    try:
                        ts = float(ts)
                        if np.isfinite(ts):
                            timestamps.append(ts)
                    except (TypeError, ValueError):
                        pass

            total_bytes.append(flow_bytes)

            if len(timestamps) >= 2:
                ts_sorted = np.sort(np.asarray(timestamps, dtype=float))
                flow_duration = float(ts_sorted[-1] - ts_sorted[0])
                if np.isfinite(flow_duration) and flow_duration >= 0:
                    durations.append(flow_duration)

                diffs = np.diff(ts_sorted)
                diffs = diffs[np.isfinite(diffs) & (diffs >= 0)]
                if diffs.size:
                    iats.extend(diffs.tolist())

        return {
            "packet_counts": np.asarray(packet_counts, dtype=float),
            "durations": np.asarray(durations, dtype=float),
            "total_bytes": np.asarray(total_bytes, dtype=float),
            "iats": np.asarray(iats, dtype=float),
        }

    # ------------------------------------------------------------------
    # Métricas
    # ------------------------------------------------------------------

    def _packet_validity(self, packets: List[ParsedPacket]) -> EvaluationResult:
        valid = 0

        for pkt in packets:
            if (
                getattr(pkt, "ip_src", None)
                and getattr(pkt, "ip_dst", None)
                and 0 <= getattr(pkt, "sport", -1) <= 65535
                and 0 <= getattr(pkt, "dport", -1) <= 65535
                and getattr(pkt, "ip_proto", None) in (1, 6, 17)
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
            packets = self._get_packets(f)
            if len(packets) < 2:
                continue

            timestamps = []
            for pkt in packets:
                ts = getattr(pkt, "timestamp", None)
                if ts is None:
                    continue
                try:
                    ts = float(ts)
                    if np.isfinite(ts):
                        timestamps.append(ts)
                except (TypeError, ValueError):
                    pass

            if len(timestamps) < 2:
                continue

            timestamps = np.asarray(timestamps, dtype=float)
            monotonic = np.all(np.diff(timestamps) >= 0)

            scores.append(1.0 if monotonic else 0.0)

        return EvaluationResult(
            metric_name="flow_coherence_score",
            value=float(np.mean(scores) if scores else 0.0),
            metadata={
                "n_flows_evaluated": len(scores),
                "n_flows_skipped": len(flows) - len(scores),
            }
        )

    # ------------------------------------------------------------------

    def _tcp_handshake_rate(self, flows):
        valid = 0
        total = 0
        SYN = 0x02
        ACK = 0x10

        for f in flows:
            packets = self._get_packets(f)
            tcp_pkts = [p for p in packets if getattr(p, "ip_proto", 0) == 6]
            if not tcp_pkts:
                continue

            flags_list = [getattr(p, "tcp_flags", 0) for p in tcp_pkts]
            has_syn = any((int(f) & SYN) != 0 for f in flags_list)
            has_ack = any((int(f) & ACK) != 0 for f in flags_list)

            if has_syn:
                total += 1
                if has_ack:
                    valid += 1

        return EvaluationResult(
            metric_name="tcp_handshake_validity_rate",
            value=float(valid / total) if total > 0 else float("nan"),
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
                proto = getattr(p, "ip_proto", None)
                if proto in counts:
                    counts[proto] += 1

            total = sum(counts.values()) + 1e-8
            return np.array([counts[k] / total for k in (1, 6, 17)], dtype=float)

        p = get_dist(real_packets)
        q = get_dist(synth_packets)
        m = (p + q) / 2
        eps = 1e-8
        jsd = (
            0.5 * np.sum(p * np.log((p + eps) / (m + eps)))
            + 0.5 * np.sum(q * np.log((q + eps) / (m + eps)))
        )

        return EvaluationResult(
            metric_name="protocol_jsd",
            value=float(np.clip(jsd, 0, None)),
        )

    # ------------------------------------------------------------------

    def _packet_size_realism(
        self,
        real_packets: List[ParsedPacket],
        synth_packets: List[ParsedPacket],
    ) -> EvaluationResult:

        real_sizes = []
        synth_sizes = []

        for p in real_packets:
            v = getattr(p, "ip_len", None)
            if v is None:
                continue
            try:
                v = float(v)
                if np.isfinite(v) and v > 0:
                    real_sizes.append(v)
            except (TypeError, ValueError):
                pass

        for p in synth_packets:
            v = getattr(p, "ip_len", None)
            if v is None:
                continue
            try:
                v = float(v)
                if np.isfinite(v) and v > 0:
                    synth_sizes.append(v)
            except (TypeError, ValueError):
                pass

        if not real_sizes or not synth_sizes:
            return EvaluationResult(metric_name="packet_size_emd", value=float("nan"))

        emd = wasserstein_distance(real_sizes, synth_sizes)
        return EvaluationResult(
            metric_name="packet_size_emd",
            value=float(emd),
            metadata={
                "mean_real": float(np.mean(real_sizes)),
                "mean_synth": float(np.mean(synth_sizes)),
                "p50_real": float(np.median(real_sizes)),
                "p50_synth": float(np.median(synth_sizes)),
            },
        )

    # ------------------------------------------------------------------
    # Nuevas métricas
    # ------------------------------------------------------------------

    def _inter_arrival_time_distribution(self, real_flows, synth_flows):
        """EMD entre distribuciones de IAT. Captura patrones temporales."""
        real_stats = self._flow_stats(real_flows)
        synth_stats = self._flow_stats(synth_flows)

        real_iats = real_stats["iats"]
        synth_iats = synth_stats["iats"]

        if real_iats.size == 0 or synth_iats.size == 0:
            return EvaluationResult(
                metric_name="inter_arrival_time_emd",
                value=float("nan"),
                metadata={
                    "n_real_iats": int(real_iats.size),
                    "n_synth_iats": int(synth_iats.size),
                },
            )

        emd = wasserstein_distance(real_iats, synth_iats)

        return EvaluationResult(
            metric_name="inter_arrival_time_emd",
            value=float(emd),
            metadata={
                "n_real_iats": int(real_iats.size),
                "n_synth_iats": int(synth_iats.size),
                "mean_real_iat": float(np.mean(real_iats)),
                "mean_synth_iat": float(np.mean(synth_iats)),
                "median_real_iat": float(np.median(real_iats)),
                "median_synth_iat": float(np.median(synth_iats)),
            },
        )

    def _bytes_per_flow_distribution(self, real_flows, synth_flows):
        """Wasserstein distance en bytes totales por flujo."""
        real_stats = self._flow_stats(real_flows)
        synth_stats = self._flow_stats(synth_flows)

        real_bytes = real_stats["total_bytes"]
        synth_bytes = synth_stats["total_bytes"]

        if real_bytes.size == 0 or synth_bytes.size == 0:
            return EvaluationResult(
                metric_name="bytes_per_flow_emd",
                value=float("nan"),
                metadata={
                    "n_real_flows": int(real_bytes.size),
                    "n_synth_flows": int(synth_bytes.size),
                },
            )

        emd = wasserstein_distance(real_bytes, synth_bytes)

        return EvaluationResult(
            metric_name="bytes_per_flow_emd",
            value=float(emd),
            metadata={
                "n_real_flows": int(real_bytes.size),
                "n_synth_flows": int(synth_bytes.size),
                "mean_real_bytes": float(np.mean(real_bytes)),
                "mean_synth_bytes": float(np.mean(synth_bytes)),
                "median_real_bytes": float(np.median(real_bytes)),
                "median_synth_bytes": float(np.median(synth_bytes)),
            },
        )

    def _flow_duration_distribution(self, real_flows, synth_flows):
        """Wasserstein distance en duración de flujos."""
        real_stats = self._flow_stats(real_flows)
        synth_stats = self._flow_stats(synth_flows)

        real_durations = real_stats["durations"]
        synth_durations = synth_stats["durations"]

        if real_durations.size == 0 or synth_durations.size == 0:
            return EvaluationResult(
                metric_name="flow_duration_emd",
                value=float("nan"),
                metadata={
                    "n_real_flows_with_duration": int(real_durations.size),
                    "n_synth_flows_with_duration": int(synth_durations.size),
                },
            )

        emd = wasserstein_distance(real_durations, synth_durations)

        return EvaluationResult(
            metric_name="flow_duration_emd",
            value=float(emd),
            metadata={
                "n_real_flows_with_duration": int(real_durations.size),
                "n_synth_flows_with_duration": int(synth_durations.size),
                "mean_real_duration": float(np.mean(real_durations)),
                "mean_synth_duration": float(np.mean(synth_durations)),
                "median_real_duration": float(np.median(real_durations)),
                "median_synth_duration": float(np.median(synth_durations)),
            },
        )

    def _flow_packet_count_distribution(self, real_flows, synth_flows):
        """KL divergence en número de paquetes por flujo."""
        real_stats = self._flow_stats(real_flows)
        synth_stats = self._flow_stats(synth_flows)

        real_counts = real_stats["packet_counts"].astype(int)
        synth_counts = synth_stats["packet_counts"].astype(int)

        if real_counts.size == 0 or synth_counts.size == 0:
            return EvaluationResult(
                metric_name="flow_packet_count_kl",
                value=float("nan"),
                metadata={
                    "n_real_flows": int(real_counts.size),
                    "n_synth_flows": int(synth_counts.size),
                },
            )

        max_count = int(max(real_counts.max(), synth_counts.max()))
        if max_count < 1:
            return EvaluationResult(
                metric_name="flow_packet_count_kl",
                value=float("nan"),
                metadata={
                    "n_real_flows": int(real_counts.size),
                    "n_synth_flows": int(synth_counts.size),
                },
            )

        # Distribución discreta sobre {1, ..., max_count}
        eps = 1e-12
        p = np.bincount(real_counts, minlength=max_count + 1)[1:].astype(float) + eps
        q = np.bincount(synth_counts, minlength=max_count + 1)[1:].astype(float) + eps

        p = p / p.sum()
        q = q / q.sum()

        kl = float(np.sum(p * np.log(p / q)))

        return EvaluationResult(
            metric_name="flow_packet_count_kl",
            value=kl,
            metadata={
                "n_real_flows": int(real_counts.size),
                "n_synth_flows": int(synth_counts.size),
                "mean_real_packets": float(np.mean(real_counts)),
                "mean_synth_packets": float(np.mean(synth_counts)),
                "median_real_packets": float(np.median(real_counts)),
                "median_synth_packets": float(np.median(synth_counts)),
            },
        )