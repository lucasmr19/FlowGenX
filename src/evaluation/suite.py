"""
src/evaluation/suite.py
========================
Orquestador del módulo de evaluación: EvaluationSuite.

Agrupa múltiples evaluadores y los ejecuta sobre el mismo par
(real, synthetic), produciendo un reporte unificado y un resumen
tabular de todas las métricas.

Uso típico
----------
    from src.evaluation import EvaluationSuite, StatisticalEvaluator
    from src.evaluation import StructuralEvaluator, TSTREvaluator

    suite = EvaluationSuite([
        StatisticalEvaluator(),
        StructuralEvaluator(representation_type="nprint"),
        TSTREvaluator(),
    ])

    result = suite.run(
        real=real_tensor,
        synthetic=synth_tensor,
        real_labels=y_real,
        synthetic_labels=y_synth,
    )

    print(result)
    df = result.to_dataframe()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import pandas as pd
from .base import BaseEvaluator, EvaluationReport, EvaluationResult


# ---------------------------------------------------------------------------
# SuiteResult: reporte unificado de todos los evaluadores
# ---------------------------------------------------------------------------


@dataclass
class SuiteResult:
    """
    Resultado agregado de todos los evaluadores de la suite.

    Attributes
    ----------
    representation_name : str
        Nombre de la representación evaluada (para logging y tablas).
    model_name : str
        Nombre del modelo generativo (para identificar la combinación RxM).
    reports : list[EvaluationReport]
        Reportes individuales de cada evaluador.
    """

    representation_name: str
    model_name: str
    reports: List[EvaluationReport] = field(default_factory=list)

    def summary(self) -> Dict[str, float]:
        """Dict plano {métrica: valor} de todos los evaluadores."""
        result: Dict[str, float] = {}
        for report in self.reports:
            for metric_name, value in report.summary().items():
                # Prefijar con nombre del evaluador para evitar colisiones
                key = f"{report.evaluator_name}.{metric_name}"
                result[key] = value
        return result

    def get_metric(self, metric_name: str) -> Optional[float]:
        """
        Busca una métrica por nombre en todos los reportes.
        Devuelve el primer match o None.
        """
        for report in self.reports:
            result = report.get(metric_name)
            if result is not None:
                return result.value
        return None

    def to_dataframe(self):
        """
        Convierte el resumen a un DataFrame de pandas (filas = métricas).

        Returns
        -------
        pd.DataFrame con columnas [representation, model, metric, value].
        """
        rows = []
        for metric_name, value in self.summary().items():
            evaluator, metric = metric_name.split(".", 1)
            rows.append({
                "representation": self.representation_name,
                "model": self.model_name,
                "evaluator": evaluator,
                "metric": metric,
                "value": value,
            })
        return pd.DataFrame(rows)

    def __repr__(self) -> str:
        lines = [
            f"SuiteResult [{self.representation_name} x {self.model_name}]",
            "=" * 60,
        ]
        for report in self.reports:
            lines.append(str(report))
        lines.append("=" * 60)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# EvaluationSuite
# ---------------------------------------------------------------------------


class EvaluationSuite:
    """
    Orquesta múltiples evaluadores sobre un par (real, synthetic).

    Parameters
    ----------
    evaluators : list[BaseEvaluator]
        Lista de evaluadores a ejecutar. El orden determina el orden
        de los reportes en SuiteResult.
    representation_name : str
        Nombre de la representación (para logging). Por defecto "unknown".
    model_name : str
        Nombre del modelo generativo (para logging). Por defecto "unknown".
    verbose : bool
        Si True, imprime el resumen tras cada evaluación.
    """

    def __init__(
        self,
        evaluators: List[BaseEvaluator],
        representation_name: str = "unknown",
        model_name: str = "unknown",
        verbose: bool = True,
    ) -> None:
        if not evaluators:
            raise ValueError("EvaluationSuite requiere al menos un evaluador.")
        self.evaluators = evaluators
        self.representation_name = representation_name
        self.model_name = model_name
        self.verbose = verbose

    def run(
        self,
        real: torch.Tensor,
        synthetic: torch.Tensor,
        **kwargs: Any,
    ) -> SuiteResult:
        """
        Ejecuta todos los evaluadores y devuelve un SuiteResult.

        Parameters
        ----------
        real : torch.Tensor, shape (N, ...)
            Datos reales de referencia.
        synthetic : torch.Tensor, shape (M, ...)
            Datos sintéticos a evaluar.
        **kwargs :
            Argumentos adicionales pasados a todos los evaluadores.
            Ejemplo: real_labels=..., synthetic_labels=...

        Returns
        -------
        SuiteResult con los reportes de todos los evaluadores.
        """
        suite_result = SuiteResult(
            representation_name=self.representation_name,
            model_name=self.model_name,
        )

        for evaluator in self.evaluators:
            if self.verbose:
                print(f"  → [{evaluator.name}] evaluando...")

            report = evaluator.evaluate(real, synthetic, **kwargs)
            suite_result.reports.append(report)

            if self.verbose:
                for metric, value in report.summary().items():
                    print(f"       {metric}: {value:.6f}")

        if self.verbose:
            print()

        return suite_result

    def add_evaluator(self, evaluator: BaseEvaluator) -> "EvaluationSuite":
        """Añade un evaluador a la suite (interfaz fluida)."""
        self.evaluators.append(evaluator)
        return self

    def __repr__(self) -> str:
        names = [e.name for e in self.evaluators]
        return (
            f"EvaluationSuite("
            f"representation='{self.representation_name}', "
            f"model='{self.model_name}', "
            f"evaluators={names})"
        )