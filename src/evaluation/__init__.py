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
"""

from .base import BaseEvaluator, EvaluationReport, EvaluationResult
from .tasks import (    DownstreamTask,
                        SupervisedClassificationTask,
                        TSTRTask,
                        ClassificationProbeTask,
                        AnomalyDetectionTask,
                        TaskRunnerEvaluator,
                        TSTREvaluator,
                        TRTREvaluator,
                        TSTRTRTRComparisonEvaluator) 
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
    # Tareas
    "DownstreamTask",
    "SupervisedClassificationTask",
    "TSTRTask",
    "ClassificationProbeTask",
    "AnomalyDetectionTask",
    # Evaluadores
    "TaskRunnerEvaluator",
    "TSTREvaluator",
    "TRTREvaluator",
    "TSTRTRTRComparisonEvaluator",
    "StatisticalEvaluator",
    "StructuralEvaluator",
    "TSTREvaluator",
    "URSEvaluator",
    "TrafficStructuralEvaluator",
    # Orquestador
    "EvaluationSuite",
    "SuiteResult",
]