from __future__ import annotations

from typing import Dict, Tuple, List, Optional
import copy
import numpy as np

from ...utils.logger_config import LOGGER
from ..domain.packet import ParsedPacket
from ..domain.sample_base import TrafficSample
from .feature import FeatureNormalizer

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