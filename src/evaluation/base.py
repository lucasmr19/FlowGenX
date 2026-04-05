"""
src/evaluation/base.py
======================
Clases base del módulo de evaluación.

Define las abstracciones sobre las que se construyen todos los evaluadores:
EvaluationResult → unidad mínima de resultado
EvaluationReport → colección de resultados de un evaluador
BaseEvaluator    → contrato abstracto para todos los evaluadores
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch


# ---------------------------------------------------------------------------
# Estructuras de datos de resultados
# ---------------------------------------------------------------------------


@dataclass
class EvaluationResult:
    """
    Unidad mínima de resultado de evaluación.

    Attributes
    ----------
    metric_name : str
        Nombre canónico de la métrica (ej. "mean_emd", "js_divergence").
    value : float
        Valor escalar de la métrica.
    metadata : dict
        Información adicional opcional (ej. valores por feature, parámetros).
    """

    metric_name: str
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"EvaluationResult({self.metric_name}={self.value:.6f})"


@dataclass
class EvaluationReport:
    """
    Colección de resultados producidos por un único evaluador.

    Attributes
    ----------
    evaluator_name : str
        Nombre del evaluador que generó el reporte.
    results : list[EvaluationResult]
        Lista ordenada de resultados individuales.
    """

    evaluator_name: str
    results: List[EvaluationResult] = field(default_factory=list)

    def summary(self) -> Dict[str, float]:
        """Devuelve un dict plano {nombre_métrica: valor}."""
        return {r.metric_name: r.value for r in self.results}

    def get(self, metric_name: str) -> Optional[EvaluationResult]:
        """Recupera un resultado por nombre, o None si no existe."""
        for r in self.results:
            if r.metric_name == metric_name:
                return r
        return None

    def __repr__(self) -> str:
        lines = [f"EvaluationReport [{self.evaluator_name}]"]
        for r in self.results:
            lines.append(f"  {r.metric_name}: {r.value:.6f}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Contrato abstracto
# ---------------------------------------------------------------------------


class BaseEvaluator(ABC):
    """
    Contrato abstracto para todos los evaluadores del framework.

    Todos los evaluadores reciben tensores reales y sintéticos de igual shape
    (N, ...) y devuelven un EvaluationReport.

    Parameters
    ----------
    name : str
        Nombre identificador del evaluador (usado en reports y logs).
    """

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def evaluate(
        self,
        real: torch.Tensor,
        synthetic: torch.Tensor,
        **kwargs: Any,
    ) -> EvaluationReport:
        """
        Evalúa la calidad del tráfico sintético respecto al real.

        Parameters
        ----------
        real : torch.Tensor
            Batch de muestras reales, shape (N, ...).
        synthetic : torch.Tensor
            Batch de muestras sintéticas, shape (M, ...).
        **kwargs :
            Argumentos adicionales específicos de cada evaluador
            (ej. etiquetas para TSTR).

        Returns
        -------
        EvaluationReport
            Reporte con todas las métricas calculadas.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"