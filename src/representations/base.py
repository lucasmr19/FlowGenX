"""
representations/base.py
=======================
Clase base abstracta para todas las representaciones de tráfico de red.

Define el contrato que debe cumplir cualquier representación:
  - encode: convierte paquetes/flujos en tensores para el modelo generativo
  - decode: operación inversa (cuando la representación es invertible)
  - fit:    ajusta parámetros estadísticos sobre el conjunto de entrenamiento

Todas las representaciones concretas heredan de TrafficRepresentation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor

from ..utils.logger_config import LOGGER

from ..data_utils.preprocessing import PCAPPipeline


# ---------------------------------------------------------------------------
# Tipos y enumeraciones auxiliares
# ---------------------------------------------------------------------------

class RepresentationType(Enum):
    """Paradigma representacional (secuencial vs visual)."""
    SEQUENTIAL = auto()
    VISUAL     = auto()


class Invertibility(Enum):
    """Indica si decode() puede reconstruir tráfico funcional."""
    INVERTIBLE     = auto()   # reconstrucción exacta o casi exacta
    APPROXIMATE    = auto()   # reconstrucción parcial con pérdidas
    NON_INVERTIBLE = auto()   # representación de solo-codificación (p.ej. GASF)


# ---------------------------------------------------------------------------
# Configuración base
# ---------------------------------------------------------------------------

@dataclass
class RepresentationConfig:
    """
    Parámetros comunes a todas las representaciones.

    Cada subclase puede extender esta dataclass añadiendo campos propios.
    Se recomienda serializar/deserializar con OmegaConf o YAML estándar.
    """
    representation_type: str = "base" # nombre del tipo de representación (secuencial, visual, etc.)
    name: str = "base_representation"  # clave del registry

    # Dimensiones de salida del encoder
    output_shape: Optional[Tuple[int, ...]] = None

    # Longitud máxima de secuencia (representaciones secuenciales)
    max_length: int = 512

    # Normalización: "minmax", "zscore", None
    normalization: Optional[str] = "minmax"

    # Semilla para reproducibilidad
    seed: int = 42

    # Campos de protocolo a incluir (None = todos los disponibles)
    include_fields: Optional[List[str]] = None

    # Metadatos adicionales libres
    extra: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Clase base abstracta
# ---------------------------------------------------------------------------

class TrafficRepresentation(ABC):
    """
    Contrato base para toda representación de tráfico en el framework.

    Ejemplo de uso
    ----------
    >>> cfg  = MyRepresentationConfig(max_length=256)
    >>> rep  = MyRepresentation(cfg)
    >>> rep.fit(train_packets)          # ajuste de vocabulario / estadísticas
    >>> tensor = rep.encode(packets)    # (B, ...) listo para el modelo
    >>> packets_hat = rep.decode(tensor)  # solo si es invertible

    Ciclo de vida
    -------------
    1. Instanciar con una config.
    2. Llamar a fit() sobre datos de entrenamiento (una sola vez).
    3. Usar encode() / decode() de forma stateless.
    4. Persistir con save() / cargar con load().
    """

    def __init__(self, config: RepresentationConfig) -> None:
        self.config = config
        self._is_fitted: bool = False
        self._dtype: torch.dtype = torch.float32

    # ------------------------------------------------------------------
    # Propiedades abstractas — deben definir las subclases
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def representation_type(self) -> RepresentationType:
        """SEQUENTIAL o VISUAL."""

    @property
    @abstractmethod
    def invertibility(self) -> Invertibility:
        """Indica si decode() ofrece reconstrucción funcional."""

    @property
    @abstractmethod
    def output_shape(self) -> Tuple[int, ...]:
        """
        Forma del tensor de salida de encode() para una sola muestra.
        Ej: (512,) para secuencial, (64, 64) para visual 2D.
        """

    # ------------------------------------------------------------------
    # Métodos abstractos — deben implementar las subclases
    # ------------------------------------------------------------------

    @abstractmethod
    def fit(self, samples: List[Any]) -> "TrafficRepresentation":
        """
        Ajusta los parámetros internos de la representación.

        Parameters
        ----------
        samples : list
            Lista de paquetes, flujos o cualquier estructura de entrada
            aceptada por esta representación concreta.

        Returns
        -------
        self : permite encadenamiento fluent (rep.fit(data).encode(x))
        """

    @abstractmethod
    def encode(self, sample: Any) -> Tensor:
        """
        Convierte una muestra de tráfico en un tensor PyTorch.

        Parameters
        ----------
        sample : Any
            Paquete individual, flujo, lista de paquetes, etc.
            El tipo exacto lo define cada subclase.

        Returns
        -------
        Tensor de forma self.output_shape.
        """

    @abstractmethod
    def decode(self, tensor: Tensor) -> Any:
        """
        Operación inversa de encode().

        Para representaciones NON_INVERTIBLE puede lanzar
        NotImplementedError o devolver una aproximación.

        Parameters
        ----------
        tensor : Tensor
            Tensor de forma self.output_shape.

        Returns
        -------
        Estructura de tráfico reconstruida (tipo según la subclase).
        """
    
    @abstractmethod
    def get_default_aggregator(self):
        """
        Tipo de agregador de tráfico utilizado para la representación.

        Returns
        -------
        Clase del agregador (p.ej. FlowAggregator, PacketWindowAggregator, TrafficChunkAggregator).
        """

    # ------------------------------------------------------------------
    # Métodos con implementación por defecto (sobreescribibles)
    # ------------------------------------------------------------------
    
    def project(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Proyecta x al espacio válido de la representación.
        Por defecto: identidad.
        """
        return x

    def encode_batch(self, samples: List[Any]) -> Tensor:
        """
        Codifica una lista de muestras en un tensor batched (B, *output_shape).

        La implementación por defecto aplica encode() en bucle y apila.
        Las subclases pueden sobreescribir para vectorizar.
        """
        self._check_fitted()
        tensors = [self.encode(s) for s in samples]
        return torch.stack(tensors, dim=0)

    def decode_batch(self, tensor: Tensor) -> List[Any]:
        """
        Decodifica un tensor batched en lista de muestras.
        """
        self._check_fitted()
        return [self.decode(tensor[i]) for i in range(tensor.shape[0])]

    # ------------------------------------------------------------------
    # Persistencia
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        """
        Guarda el estado interno de la representación (vocabulario,
        estadísticas de normalización, etc.) en disco.

        Usa torch.save con un diccionario de estado.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "config":     self.config,
            "is_fitted":  self._is_fitted,
            "state_dict": self._get_state_dict(),
            "class":      self.__class__.__name__,
        }
        torch.save(state, path)
        LOGGER.info("Representación '%s' guardada en %s", self.config.name, path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "TrafficRepresentation":
        """
        Carga una representación previamente guardada.

        Nota: la subclase concreta debe llamar a su propio load() si
        añade campos adicionales en _get_state_dict().
        """
        path = Path(path)
        state = torch.load(path, weights_only=False)
        instance = cls(state["config"])
        instance._is_fitted = state["is_fitted"]
        instance._set_state_dict(state["state_dict"])
        LOGGER.info("Representación '%s' cargada desde %s",
                    instance.config.name, path)
        return instance

    def _get_state_dict(self) -> Dict[str, Any]:
        """
        Devuelve el estado serializable de la representación.
        Las subclases deben extender este método.
        """
        return {}

    def _set_state_dict(self, state: Dict[str, Any]) -> None:
        """
        Restaura el estado desde un dict. Las subclases deben extender.
        """
        pass
    
    def build_preprocessing_pipeline(self, **kwargs) -> PCAPPipeline:
        return PCAPPipeline(
            aggregator=self.get_default_aggregator(),
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Utilidades internas
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                f"La representación '{self.config.name}' no ha sido ajustada. "
                "Llama a fit() antes de encode() / decode()."
            )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name={self.config.name!r}, "
            f"type={self.representation_type.name}, "
            f"invertible={self.invertibility.name}, "
            f"fitted={self._is_fitted})"
        )

class TrafficEncoder(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError