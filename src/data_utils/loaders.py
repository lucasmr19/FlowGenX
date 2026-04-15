"""
data_utils/loaders.py
===============
Datasets y DataLoaders de PyTorch para el framework de tráfico de red.

Jerarquía:
  TrafficDataset          ← Dataset genérico sobre List[TrafficSample]
  RepresentationDataset   ← Aplica una TrafficRepresentation en __getitem__
  TrafficDataModule       ← Gestiona splits train/val/test reproducibles

Separación clave: el DataModule conoce los datos crudos (flujos);
las representaciones se pasan como argumento y se aplican bajo demanda.
Esto evita materializar todas las representaciones a la vez.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Type, TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from .preprocessing import PCAPPipeline, TrafficSample, TrafficAggregatorBase, FlowAggregator, PacketWindowAggregator
from ..utils.logger_config import LOGGER

if TYPE_CHECKING:
    from ..representations.base import TrafficRepresentation
    
# ---------------------------------------------------------------------------
# Dataset base — trabaja sobre flujos ya parseados
# ---------------------------------------------------------------------------

class TrafficDataset(Dataset):
    """
    Dataset PyTorch sobre una lista de objetos TrafficSample.

    Opcionalmente aplica un transform (callable) que convierte
    un TrafficSample en cualquier estructura (tensor, dict, etc.).

    Parameters
    ----------
    samples     : lista de TrafficSample producida por PCAPPipeline.
    transform : callable opcional TrafficSample → Any (p.ej. una TrafficRepresentation)
    label_fn  : callable opcional TrafficSample → int para clasificación downstream
    """

    def __init__(
        self,
        samples:     List[TrafficSample],
        transform: Optional[Callable[[TrafficSample], Any]] = None,
        label_fn:  Optional[Callable[[TrafficSample], int]]  = None,
    ) -> None:
        self.samples     = samples
        self.transform = transform
        self.label_fn  = label_fn

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Union[Any, Tuple[Any, int]]:
        sample = self.samples[idx]
        item = self.transform(sample) if self.transform else sample
        if self.label_fn is not None:
            return item, self.label_fn(sample)
        return item

    def get_sample(self, idx: int) -> TrafficSample:
        """Acceso directo al TrafficSample subyacente (sin transform)."""
        return self.samples[idx]


# ---------------------------------------------------------------------------
# Dataset con representación incorporada
# ---------------------------------------------------------------------------

class RepresentationDataset(Dataset):
    """
    Dataset que aplica una TrafficRepresentation en __getitem__.

    La representación debe haber sido ajustada (fit()) antes de pasarla.
    El resultado de encode() se devuelve como tensor.

    Parameters
    ----------
    samples        : lista de TrafficSample (Flow | PacketWindow)
    representation : TrafficRepresentation ya ajustada
    label_fn       : callable opcional TrafficSample → int
    cache          : si True, precalcula y cachea todos los tensores en RAM
    """

    def __init__(
        self,
        samples:          List[TrafficSample],
        representation: "TrafficRepresentation",
        label_fn:       Optional[Callable[[TrafficSample], int]] = None,
        cache:          bool = False,
    ) -> None:
        self.samples        = samples
        self.representation = representation
        self.label_fn       = label_fn
        self._cache: Optional[List[Tensor]] = None

        if cache:
            LOGGER.info("Precalculando representaciones en caché (%d muestras)...", len(samples))
            self._cache = [representation.encode(s) for s in samples]
            LOGGER.info("Caché completada.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Union[Tensor, Tuple[Tensor, int]]:
        tensor = (
            self._cache[idx]
            if self._cache is not None
            else self.representation.encode(self.samples[idx])
        )
        if self.label_fn is not None:
            return tensor, self.label_fn(self.samples[idx])
        return tensor

    def get_sample(self, idx: int) -> TrafficSample:
        return self.samples[idx]


# ---------------------------------------------------------------------------
# DataModule — gestión de splits y loaders
# ---------------------------------------------------------------------------

class TrafficDataModule:
    """
    Gestiona la partición y carga de datos para un experimento.

    Características:
    - Split estratificado o aleatorio en train/val/test.
    - Semilla fija para reproducibilidad.
    - Ajusta la representación SOLO sobre train (sin data leakage).
    - Devuelve DataLoaders con collate_fn personalizable.

    Parameters
    ----------
    samples          : todos los flujos disponibles (antes de particionar)
    representation : instancia de TrafficRepresentation (aún sin ajustar)
    train_ratio    : fracción para entrenamiento (default 0.7)
    val_ratio      : fracción para validación   (default 0.15)
    test_ratio     : fracción para test          (default 0.15)
    seed           : semilla aleatoria
    label_fn       : función TrafficSample → int (clasificación downstream)
    batch_size     : tamaño de batch para los DataLoaders
    num_workers    : workers del DataLoader
    cache_datasets : si True precalcula los encodings en RAM
    """

    def __init__(
        self,
        samples:          List[TrafficSample],
        representation: "TrafficRepresentation",
        train_ratio:    float = 0.70,
        val_ratio:      float = 0.15,
        test_ratio:     float = 0.15,
        seed:           int   = 42,
        label_fn:       Optional[Callable[[TrafficSample], int]] = None,
        batch_size:     int   = 32,
        num_workers:    int   = 0,
        cache_datasets: bool  = False,
    ) -> None:
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Los ratios de split deben sumar 1.0"

        self.samples          = samples
        self.representation = representation
        self.train_ratio    = train_ratio
        self.val_ratio      = val_ratio
        self.test_ratio     = test_ratio
        self.seed           = seed
        self.label_fn       = label_fn
        self.batch_size     = batch_size
        self.num_workers    = num_workers
        self.cache_datasets = cache_datasets

        self._train_samples: Optional[List[TrafficSample]] = None
        self._val_samples:   Optional[List[TrafficSample]] = None
        self._test_samples:  Optional[List[TrafficSample]] = None
        self._is_setup     = False

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """
        Particiona los datos y ajusta la representación sobre train.
        Debe llamarse una sola vez antes de obtener los DataLoaders.
        """
        if self._is_setup:
            LOGGER.warning("DataModule ya configurado. Ignorando setup() duplicado.")
            return

        # 1. Shuffle con semilla fija
        indices = list(range(len(self.samples)))
        rng = random.Random(self.seed)
        rng.shuffle(indices)

        n = len(indices)
        n_train = int(n * self.train_ratio)
        n_val   = int(n * self.val_ratio)

        train_idx = indices[:n_train]
        val_idx   = indices[n_train: n_train + n_val]
        test_idx  = indices[n_train + n_val:]

        self._train_samples = [self.samples[i] for i in train_idx]
        self._val_samples   = [self.samples[i] for i in val_idx]
        self._test_samples  = [self.samples[i] for i in test_idx]

        LOGGER.info(
            "Split: train=%d / val=%d / test=%d",
            len(self._train_samples), len(self._val_samples), len(self._test_samples),
        )

        # 2. Ajustar representación SOLO sobre train
        LOGGER.info("Ajustando representación '%s' sobre datos de train...",
                    self.representation.config.name)
        self.representation.fit(self._train_samples)
        LOGGER.info("Representación ajustada.")

        self._is_setup = True

    # ------------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------------

    def train_dataloader(self) -> DataLoader:
        self._check_setup()
        ds = self._make_dataset(self._train_samples)
        return DataLoader(
            ds,
            batch_size  = self.batch_size,
            shuffle     = True,
            num_workers = self.num_workers,
            collate_fn  = self._collate_fn,
            pin_memory  = torch.cuda.is_available(),
        )

    def val_dataloader(self) -> DataLoader:
        self._check_setup()
        ds = self._make_dataset(self._val_samples)
        return DataLoader(
            ds,
            batch_size  = self.batch_size,
            shuffle     = False,
            num_workers = self.num_workers,
            collate_fn  = self._collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        self._check_setup()
        ds = self._make_dataset(self._test_samples)
        return DataLoader(
            ds,
            batch_size  = self.batch_size,
            shuffle     = False,
            num_workers = self.num_workers,
            collate_fn  = self._collate_fn,
        )

    # ------------------------------------------------------------------
    # Acceso a flujos crudos (para evaluación downstream)
    # ------------------------------------------------------------------

    @property
    def train_samples(self) -> List[TrafficSample]:
        self._check_setup()
        return self._train_samples

    @property
    def val_samples(self) -> List[TrafficSample]:
        self._check_setup()
        return self._val_samples

    @property
    def test_samples(self) -> List[TrafficSample]:
        self._check_setup()
        return self._test_samples

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_dataset(self, samples: List[TrafficSample]) -> RepresentationDataset:
        return RepresentationDataset(
            samples        = samples,
            representation = self.representation,
            label_fn       = self.label_fn,
            cache          = self.cache_datasets,
        )

    @staticmethod
    def _collate_fn(batch: List[Any]) -> Any:
        """
        Collate por defecto: si los items son tensores los apila,
        si son tuplas (tensor, label) los desempaqueta.
        """
        if isinstance(batch[0], tuple):
            tensors, labels = zip(*batch)
            return torch.stack(list(tensors)), torch.tensor(list(labels))
        return torch.stack(batch)

    def _check_setup(self) -> None:
        if not self._is_setup:
            raise RuntimeError("Llama a setup() antes de obtener DataLoaders.")

    # ------------------------------------------------------------------
    # Estadísticas del dataset
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """Devuelve un resumen estadístico de los splits."""
        self._check_setup()
        return {
            "total_samples":  len(self.samples),
            "train_samples":  len(self._train_samples),
            "val_samples":    len(self._val_samples),
            "test_samples":   len(self._test_samples),
            "representation": str(self.representation),
            "batch_size":   self.batch_size,
            "seed":         self.seed,
        }

def build_datamodule_from_pcap(
    pcap_path:      Union[str, Path],
    representation: TrafficRepresentation,
    *,
    aggregator: Union[Type[TrafficAggregatorBase], TrafficAggregatorBase] = FlowAggregator,
    aggregator_kwargs: Optional[Dict[str, Any]] = None,
    max_payload_bytes: int = 20,
    streaming:        bool = True,
    max_packets:    Optional[int] = None,
    protocols:      Optional[List[str]] = None,
    train_ratio:    float = 0.70,
    val_ratio:      float = 0.15,
    test_ratio:     float = 0.15,
    batch_size:     int   = 32,
    num_workers:    int   = 0,
    seed:           int   = 42,
    label_fn:       Optional[Callable[[TrafficSample], int]] = None,
) -> TrafficDataModule:
    """
    Función conveniente que orquesta todo el pipeline:
      PCAP → TrafficSamples → TrafficDataModule configurado y listo.

    Example
    -------
    >>> from nf_framework.representations.sequential import SequentialRepresentation
    >>> rep = SequentialRepresentation(SequentialConfig())
    >>> dm  = build_datamodule_from_pcap("traffic.pcap", rep, batch_size=64)
    >>> dm.setup()
    >>> for batch in dm.train_dataloader():
    ...     print(batch.shape)
    """
    pipeline = PCAPPipeline(
        aggregator=aggregator,
        max_packets=max_packets,
        protocols=protocols,
        max_payload_bytes=max_payload_bytes,
        streaming=streaming,
        **(aggregator_kwargs or {}),
    )
    samples = pipeline.process(pcap_path)

    dm = TrafficDataModule(
        samples        = samples,
        representation = representation,
        train_ratio    = train_ratio,
        val_ratio      = val_ratio,
        test_ratio     = test_ratio,
        batch_size     = batch_size,
        num_workers    = num_workers,
        seed           = seed,
        label_fn       = label_fn,
    )
    return dm


# ---------------------------------------------------------------------------
# Factory helper (MULTI-PCAP + clases por directorio)
# ---------------------------------------------------------------------------

def build_datamodule_from_dir(
    root_path:      Union[str, Path],
    representation: "TrafficRepresentation",
    *,
    aggregator: Union[Type[TrafficAggregatorBase], TrafficAggregatorBase] = FlowAggregator,
    aggregator_kwargs: Optional[Dict[str, Any]] = None,
    max_payload_bytes: int = 20,
    streaming:        bool = True,
    max_packets:      Optional[int] = None,
    protocols:        Optional[List[str]] = None,
    train_ratio:      float = 0.70,
    val_ratio:        float = 0.15,
    test_ratio:       float = 0.15,
    batch_size:       int   = 32,
    num_workers:      int   = 0,
    seed:             int   = 42,
    label_fn:         Optional[Callable[[TrafficSample], int]] = None,
) -> TrafficDataModule:
    """
    Construye un TrafficDataModule a partir de un directorio de PCAPs.

    Estructura esperada:
        root_path/
            class_1/
                a.pcap
                b.pcap
            class_2/
                c.pcap

    Cada subdirectorio se interpreta como una clase.

    Si root_path contiene directamente .pcap, se asigna clase única.

    Añade automáticamente a cada sample:
        - sample.label
        - sample.class_name
        - sample.source
    """

    root_path = Path(root_path)

    pipeline = PCAPPipeline(
        aggregator=aggregator,
        max_packets=max_packets,
        protocols=protocols,
        max_payload_bytes=max_payload_bytes,
        streaming=streaming,
        **(aggregator_kwargs or {}),
    )

    all_samples: List[TrafficSample] = []
    class_to_label: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Caso 1: directorio con subcarpetas (multi-clase)
    # ------------------------------------------------------------------
    subdirs = [d for d in root_path.iterdir() if d.is_dir()]

    if subdirs:
        LOGGER.info("Detectadas %d clases (subdirectorios).", len(subdirs))

        for class_idx, class_dir in enumerate(sorted(subdirs)):
            class_name = class_dir.name
            class_to_label[class_name] = class_idx

            pcap_files = list(class_dir.glob("*.pcap"))
            LOGGER.info("Clase '%s': %d pcaps", class_name, len(pcap_files))

            for pcap_file in pcap_files:
                samples = pipeline.process(pcap_file)

                for s in samples:
                    # metadata (requiere que exista en TrafficSampleBase)
                    setattr(s, "label", class_idx)
                    setattr(s, "class_name", class_name)
                    setattr(s, "source", str(pcap_file))

                all_samples.extend(samples)

    # ------------------------------------------------------------------
    # Caso 2: directorio plano con .pcap
    # ------------------------------------------------------------------
    else:
        pcap_files = list(root_path.glob("*.pcap"))

        if not pcap_files:
            raise ValueError(f"No se encontraron archivos .pcap en {root_path}")

        LOGGER.info("Modo single-class (%d pcaps).", len(pcap_files))

        for pcap_file in pcap_files:
            samples = pipeline.process(pcap_file)

            for s in samples:
                setattr(s, "label", 0)
                setattr(s, "class_name", "default")
                setattr(s, "source", str(pcap_file))

            all_samples.extend(samples)

    if not all_samples:
        raise RuntimeError("No se generaron muestras desde los PCAPs.")

    LOGGER.info("Total samples generados: %d", len(all_samples))

    # ------------------------------------------------------------------
    # Label function por defecto
    # ------------------------------------------------------------------
    if label_fn is None:
        label_fn = lambda s: getattr(s, "label", 0)

    # ------------------------------------------------------------------
    # Construcción del DataModule
    # ------------------------------------------------------------------
    dm = TrafficDataModule(
        samples        = all_samples,
        representation = representation,
        train_ratio    = train_ratio,
        val_ratio      = val_ratio,
        test_ratio     = test_ratio,
        batch_size     = batch_size,
        num_workers    = num_workers,
        seed           = seed,
        label_fn       = label_fn,
    )

    return dm