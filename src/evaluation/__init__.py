"""
src/evaluation/__init__.py
===========================
Módulo de evaluación del framework de comparativa de tráfico de red.

Exporta las clases principales para uso externo:
    - EvaluationResult, EvaluationReport, BaseEvaluator  (abstracciones base)
    - StatisticalEvaluator                                (EMD, JS, Pearson)
    - StructuralEvaluator                                 (validez por representación)
    - TSTREvaluator                                       (TSTR + baseline TRTR)
    - EvaluationSuite, SuiteResult                        (orquestador)

Ejemplo rápido
--------------
    from src.evaluation import EvaluationSuite, StatisticalEvaluator
    from src.evaluation import StructuralEvaluator, TSTREvaluator

    suite = EvaluationSuite(
        evaluators=[
            StatisticalEvaluator(),
            StructuralEvaluator(representation_type="nprint"),
            TSTREvaluator(),
        ],
        representation_name="nprint",
        model_name="DDPM",
    )
    result = suite.run(real_tensor, synth_tensor, real_labels=y, synthetic_labels=y_synth)
    print(result)
"""

from .base import BaseEvaluator, EvaluationReport, EvaluationResult
from .downstream import TSTREvaluator
from .statistical import StatisticalEvaluator
from .structural import StructuralEvaluator
from .suite import EvaluationSuite, SuiteResult

__all__ = [
    # Base
    "BaseEvaluator",
    "EvaluationReport",
    "EvaluationResult",
    # Evaluadores
    "StatisticalEvaluator",
    "StructuralEvaluator",
    "TSTREvaluator",
    # Orquestador
    "EvaluationSuite",
    "SuiteResult",
]