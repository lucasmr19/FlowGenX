from __future__ import annotations

from pathlib import Path
from typing import  List, Optional, Tuple, Type, Union

from ...utils.logger_config import LOGGER

from ..domain.sample_base import TrafficSample
from ..io.pcap_reader import PCAPReader
from ..parsing.packet_parser import PacketParser
from ..aggregation.aggregation_base import TrafficAggregatorBase
from ..aggregation.flow import FlowAggregator
from ..normalization.flow import FlowNormalizer


class PCAPPipeline:
    """
    Orquesta el pipeline completo: PCAP → flujos normalizados.

    Encadena: PCAPReader → PacketParser → Aggregator → FlowNormalizer (opc.)

    API sin data leakage
    --------------------
    El normalizer se ajusta SOLO con datos de entrenamiento:

        # Entrenamiento
        pipeline       = PCAPPipeline(normalize=True)
        flows_train    = pipeline.fit_process("train.pcap")

        # Validación / test (reutiliza el normalizer ya ajustado)
        flows_val      = pipeline.process("val.pcap")
        flows_test     = pipeline.process("test.pcap")

    Si normalize=False (por defecto) el comportamiento es idéntico a la
    versión anterior: devuelve flujos con valores crudos.

    Parameters
    ----------
    aggregator        : clase o instancia de TrafficAggregatorBase.
    max_packets       : límite de paquetes a leer del PCAP.
    protocols         : filtro de protocolos (["TCP", "UDP", ...]).
    max_payload_bytes : bytes de payload a extraer por paquete.
    streaming         : si True usa PcapReader en modo streaming.
    normalize         : si True activa FlowNormalizer.
    norm_method       : "minmax" | "zscore" (solo si normalize=True).
    norm_fields       : campos a normalizar (None = NUMERIC_PACKET_FIELDS).
    norm_copy         : si True, transform() devuelve copias profundas.
    **aggregator_kwargs : argumentos extra pasados al aggregator.
    """

    def __init__(
        self,
        aggregator:         Union[Type[TrafficAggregatorBase], TrafficAggregatorBase] = FlowAggregator,
        max_packets:        Optional[int]              = None,
        protocols:          Optional[List[str]]        = None,
        max_payload_bytes:  int                        = 20,
        streaming:          bool                       = True,
        normalize:          bool                       = False,
        norm_method:        str                        = "minmax",
        norm_fields:        Optional[Tuple[str, ...]]  = None,
        norm_copy:          bool                       = True,
        **aggregator_kwargs,
    ) -> None:
        self.reader = PCAPReader(
            max_packets=max_packets,
            protocols=protocols,
            streaming=streaming,
        )
        self.parser = PacketParser(
            max_payload_bytes=max_payload_bytes,
            include_payload=True,
        )
        self.aggregator = (
            aggregator(**aggregator_kwargs)
            if isinstance(aggregator, type)
            else aggregator
        )

        self.normalize = normalize
        self.normalizer: Optional[FlowNormalizer] = (
            FlowNormalizer(
                method=norm_method,
                fields=norm_fields,
                copy=norm_copy,
            )
            if normalize else None
        )

    # ------------------------------------------------------------------
    # Núcleo interno: PCAP → samples sin normalizar
    # ------------------------------------------------------------------

    def _raw_process(self, pcap_path: Union[str, Path]) -> List[TrafficSample]:
        raw_pkts    = list(self.reader.read(pcap_path))
        parsed_pkts = self.parser.parse_sequence(raw_pkts)
        return self.aggregator.aggregate(parsed_pkts)

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def fit_process(
        self, pcap_path: Union[str, Path]
    ) -> List[TrafficSample]:
        """
        Procesa el PCAP, ajusta el normalizer sobre los datos resultantes
        y devuelve las muestras normalizadas.

        Uso exclusivo con datos de ENTRENAMIENTO para evitar data leakage.

        Returns
        -------
        List[TrafficSample] normalizados (si normalize=True) o crudos.
        """
        samples = self._raw_process(pcap_path)

        if self.normalizer is not None:
            LOGGER.info("fit_process: ajustando FlowNormalizer sobre %d muestras.", len(samples))
            samples = self.normalizer.fit_transform(samples)

        return samples

    def process(
        self, pcap_path: Union[str, Path]
    ) -> List[TrafficSample]:
        """
        Procesa el PCAP y aplica el normalizer ya ajustado (si existe).

        Uso para datos de VALIDACIÓN y TEST.

        Raises
        ------
        RuntimeError si normalize=True pero fit_process() no se ha llamado.

        Returns
        -------
        List[TrafficSample] normalizados (si normalize=True) o crudos.
        """
        samples = self._raw_process(pcap_path)

        if self.normalizer is not None:
            if not self.normalizer.is_fitted:
                raise RuntimeError(
                    "El normalizer no está ajustado. "
                    "Llama a fit_process() con datos de entrenamiento antes de process()."
                )
            LOGGER.info("process: normalizando %d muestras.", len(samples))
            samples = self.normalizer.transform(samples)

        return samples

    def process_directory(
        self,
        pcap_dir:     Union[str, Path],
        glob_pattern: str = "*.pcap",
        fit:          bool = False,
    ) -> List[TrafficSample]:
        """
        Procesa todos los PCAPs de un directorio.

        Parameters
        ----------
        fit : si True llama a fit_process() en el primer PCAP encontrado
              y process() en el resto. Útil cuando todo el directorio es train.
        """
        pcap_dir   = Path(pcap_dir)
        all_samples: List[TrafficSample] = []
        pcap_files  = sorted(pcap_dir.glob(glob_pattern))

        for i, pcap_file in enumerate(pcap_files):
            try:
                if fit and i == 0:
                    samples = self.fit_process(pcap_file)
                else:
                    samples = self.process(pcap_file)
                all_samples.extend(samples)
            except Exception as exc:
                LOGGER.error("Error procesando %s: %s", pcap_file, exc)

        LOGGER.info("Total muestras procesadas: %d", len(all_samples))
        return all_samples