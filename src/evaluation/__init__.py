"""
src/evaluation/__init__.py
===========================
Módulo de evaluación del framework de comparativa de tráfico de red.

Exporta las clases principales para uso externo:
    - EvaluationResult, EvaluationReport, BaseEvaluator  (abstracciones base)
    - StatisticalEvaluator                                (EMD, JS, Pearson)       [PRIMARY]
    - URSEvaluator                                        (FID, MMD)               [PRIMARY]
    - ConditionalEvaluator                                (fidelidad condicional)  [CRITICAL]
    - StructuralEvaluator                                 (validez representación) [SECONDARY]
    - TrafficStructuralEvaluator                          (nivel tráfico/paquete)  [SECONDARY]
    - TSTREvaluator, TRTREvaluator, ...                   (tareas downstream)      [TERTIARY]
    - EvaluationSuite, SuiteResult                        (orquestador)

Prioridad de métricas
---------------------
  PRIMARY   : statistical (JS, EMD), urs (FID, MMD)  → calidad del modelo generativo
  CRITICAL  : conditional (accuracy, f1)              → fidelidad condicional (modelos cond.)
  SECONDARY : structural, traffic_structural           → validez estructural
  TERTIARY  : tstr, trtr, anomaly                     → utilidad downstream (no calidad generativa)
"""

from .base import BaseEvaluator, EvaluationReport, EvaluationResult
from .conditional import ConditionalEvaluator
from .tasks import (
    DownstreamTask,
    SupervisedClassificationTask,
    TSTRTask,
    ClassificationProbeTask,
    AnomalyDetectionTask,
    TaskRunnerEvaluator,
    TSTREvaluator,
    TRTREvaluator,
    TSTRTRTRComparisonEvaluator,
)
from .urs import URSEvaluator
from .statistical import StatisticalEvaluator
from .structural import StructuralEvaluator
from .suite import EvaluationSuite, SuiteResult
from .traffic_structural import TrafficStructuralEvaluator

__all__ = [
    # Base
    "BaseEvaluator",
    "EvaluationReport",
    "EvaluationResult",
    # PRIMARY: calidad generativa
    "StatisticalEvaluator",
    "URSEvaluator",
    # CRITICAL: fidelidad condicional
    "ConditionalEvaluator",
    # SECONDARY: validez estructural
    "StructuralEvaluator",
    "TrafficStructuralEvaluator",
    # TERTIARY: utilidad downstream
    "DownstreamTask",
    "SupervisedClassificationTask",
    "TSTRTask",
    "ClassificationProbeTask",
    "AnomalyDetectionTask",
    "TaskRunnerEvaluator",
    "TSTREvaluator",
    "TRTREvaluator",
    "TSTRTRTRComparisonEvaluator",
    # Orquestador
    "EvaluationSuite",
    "SuiteResult",
]