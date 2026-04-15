"""
src/evaluation/tasks/downstream.py
==================================

Evaluación basada en tareas downstream.

Estructura:
- Tasks: definen el protocolo de entrenamiento/test
- Evaluators: adaptan una Task a la interfaz BaseEvaluator
- Comparator opcional: ejecuta TSTR y TRTR y calcula gaps

Incluye:
- TSTRTask
- TRTRTask
- ClassificationProbeTask
- AnomalyDetectionTask
- TaskRunnerEvaluator
- TSTREvaluator
- TRTREvaluator
- TSTRTRTRComparisonEvaluator
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from ..base import BaseEvaluator, EvaluationReport, EvaluationResult
from .task_base import (
    DownstreamTask,
    SupervisedClassificationTask,
    _flatten,
    _to_numpy,
    _prepare_features,
    _fit_and_score_classifier,
)


# ---------------------------------------------------------------------------
# Concrete tasks
# ---------------------------------------------------------------------------

class TSTRTask(SupervisedClassificationTask):
    """
    Train on Synthetic, Test on Real.
    """

    name = "TSTRTask"

    def run(
        self,
        synthetic: torch.Tensor | np.ndarray,
        synthetic_labels: np.ndarray,
        real: torch.Tensor | np.ndarray,
        real_labels: np.ndarray,
    ) -> Dict[str, float]:
        self.fit(synthetic, synthetic_labels)
        return self.test(real, real_labels)


class TRTRTask(SupervisedClassificationTask):
    """
    Train on Real, Test on Real.
    Baseline de referencia.
    """

    name = "TRTRTask"

    def run(
        self,
        real_train: torch.Tensor | np.ndarray,
        real_train_labels: np.ndarray,
        real_test: torch.Tensor | np.ndarray,
        real_test_labels: np.ndarray,
    ) -> Dict[str, float]:
        self.fit(real_train, real_train_labels)
        return self.test(real_test, real_test_labels)


class ClassificationProbeTask(SupervisedClassificationTask):
    """
    Probe sobre un espacio de features fijo.

    Si se pasa un encoder, se usa como extractor de embeddings congelado.
    Si no, se aplana la entrada.
    """

    name = "ClassificationProbeTask"

    def __init__(
        self,
        encoder: Optional[torch.nn.Module] = None,
        max_features: int = 256,
        classifier: str = "mlp",
        device: str = "cpu",
    ) -> None:
        super().__init__(max_features=max_features, classifier=classifier)
        self.encoder = encoder.eval() if encoder is not None else None
        self.device = device

    @torch.no_grad()
    def _encode(self, X: torch.Tensor | np.ndarray) -> np.ndarray:
        if self.encoder is None:
            return _flatten(X)

        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)

        Z = self.encoder(X.to(self.device))
        return _to_numpy(Z).reshape(Z.shape[0], -1)

    def fit(self, X: torch.Tensor | np.ndarray, y: np.ndarray) -> None:
        self._X_train = self._encode(X)
        self._y_train = np.asarray(y).astype(int)

    def test(self, X: torch.Tensor | np.ndarray, y: np.ndarray) -> Dict[str, float]:
        if self._X_train is None or self._y_train is None:
            raise RuntimeError("Primero llama a fit().")

        X_test = self._encode(X)
        X_train_prep, X_test_prep = self._prepare(self._X_train, X_test)
        acc, f1 = _fit_and_score_classifier(
            X_train_prep,
            self._y_train,
            X_test_prep,
            np.asarray(y),
        )
        return {
            "probe_accuracy": acc,
            "probe_f1_macro": f1,
        }


class AnomalyDetectionTask(DownstreamTask):
    """
    Tarea de detección de anomalías.

    Convención:
    - y = 0 -> normal
    - y = 1 -> anomalía

    Entrenamiento:
    - por defecto entrena con las muestras normales del conjunto de entrada
    """

    name = "AnomalyDetectionTask"

    def __init__(
        self,
        max_features: int = 256,
        contamination: float = 0.1,
    ) -> None:
        self.max_features = max_features
        self.contamination = contamination
        self._model: Optional[IsolationForest] = None
        self._scaler: Optional[StandardScaler] = None
        self._pca: Optional[PCA] = None

    def fit(self, X: torch.Tensor | np.ndarray, y: np.ndarray) -> None:
        X = _flatten(X)
        y = np.asarray(y).astype(int)

        X_train = X[y == 0]
        if len(X_train) == 0:
            raise ValueError(
                "No hay muestras normales (label 0) para entrenar el detector de anomalías."
            )

        self._scaler = StandardScaler()
        X_train = self._scaler.fit_transform(X_train)

        if X_train.shape[1] > self.max_features:
            n_components = min(self.max_features, max(1, X_train.shape[0] - 1))
            if n_components < X_train.shape[1]:
                self._pca = PCA(n_components=n_components, random_state=42)
                X_train = self._pca.fit_transform(X_train)

        self._model = IsolationForest(
            contamination=self.contamination,
            random_state=42,
            n_estimators=200,
        )
        self._model.fit(X_train)

    def test(self, X: torch.Tensor | np.ndarray, y: np.ndarray) -> Dict[str, float]:
        if self._model is None or self._scaler is None:
            raise RuntimeError("Primero llama a fit().")

        X = _flatten(X)
        y = np.asarray(y).astype(int)

        X_test = self._scaler.transform(X)
        if self._pca is not None:
            X_test = self._pca.transform(X_test)

        scores = -self._model.decision_function(X_test)
        pred = (scores >= np.median(scores)).astype(int)

        out: Dict[str, float] = {
            "anomaly_f1_macro": float(
                f1_score(y, pred, average="macro", zero_division=0)
            ),
        }

        if len(np.unique(y)) > 1:
            out["anomaly_roc_auc"] = float(roc_auc_score(y, scores))
            out["anomaly_pr_auc"] = float(average_precision_score(y, scores))
        else:
            out["anomaly_roc_auc"] = float("nan")
            out["anomaly_pr_auc"] = float("nan")

        return out


# ---------------------------------------------------------------------------
# Generic runner / adapter
# ---------------------------------------------------------------------------

class TaskRunnerEvaluator(BaseEvaluator):
    """
    Adaptador genérico que convierte una Task en un BaseEvaluator.
    """

    def __init__(
        self,
        task: DownstreamTask,
        name: Optional[str] = None,
        category: str = "utility",
    ) -> None:
        super().__init__(name=name or task.name)
        self.task = task
        self.category = category

    def evaluate(
        self,
        real: torch.Tensor,
        synthetic: torch.Tensor,
        real_labels: Optional[np.ndarray] = None,
        synthetic_labels: Optional[np.ndarray] = None,
        real_train: Optional[torch.Tensor] = None,
        real_train_labels: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> EvaluationReport:
        report = EvaluationReport(
            evaluator_name=self.name,
            results=[],
        )

        if isinstance(self.task, TSTRTask):
            if real_labels is None or synthetic_labels is None:
                report.results.append(
                    EvaluationResult(
                        metric_name="tstr_accuracy",
                        value=float("nan"),
                        metadata={
                            "error": "Se requieren real_labels y synthetic_labels."
                        },
                    )
                )
                return report

            metrics = self.task.run(
                synthetic=synthetic,
                synthetic_labels=synthetic_labels,
                real=real,
                real_labels=real_labels,
            )
            report.results.extend(
                [
                    EvaluationResult("tstr_accuracy", metrics["accuracy"]),
                    EvaluationResult("tstr_f1_macro", metrics["f1_macro"]),
                ]
            )
            return report

        if isinstance(self.task, TRTRTask):
            if real_train is None or real_train_labels is None or real_labels is None:
                report.results.append(
                    EvaluationResult(
                        metric_name="trtr_accuracy",
                        value=float("nan"),
                        metadata={
                            "error": "Se requieren real_train, real_train_labels y real_labels."
                        },
                    )
                )
                return report

            metrics = self.task.run(
                real_train=real_train,
                real_train_labels=real_train_labels,
                real_test=real,
                real_test_labels=real_labels,
            )
            report.results.extend(
                [
                    EvaluationResult("trtr_accuracy", metrics["accuracy"]),
                    EvaluationResult("trtr_f1_macro", metrics["f1_macro"]),
                ]
            )
            return report

        if isinstance(self.task, ClassificationProbeTask):
            if real_labels is None:
                report.results.append(
                    EvaluationResult(
                        metric_name="probe_accuracy",
                        value=float("nan"),
                        metadata={"error": "Se requiere real_labels."},
                    )
                )
                return report

            if synthetic_labels is None:
                report.results.append(
                    EvaluationResult(
                        metric_name="probe_accuracy",
                        value=float("nan"),
                        metadata={
                            "error": "Se requieren synthetic_labels para entrenar el probe."
                        },
                    )
                )
                return report

            self.task.fit(synthetic, synthetic_labels)
            metrics = self.task.test(real, real_labels)
            report.results.extend(
                [
                    EvaluationResult("probe_accuracy", metrics["probe_accuracy"]),
                    EvaluationResult("probe_f1_macro", metrics["probe_f1_macro"]),
                ]
            )
            return report

        if isinstance(self.task, AnomalyDetectionTask):
            if real_labels is None:
                report.results.append(
                    EvaluationResult(
                        metric_name="anomaly_f1_macro",
                        value=float("nan"),
                        metadata={"error": "Se requiere real_labels."},
                    )
                )
                return report

            if synthetic_labels is None:
                report.results.append(
                    EvaluationResult(
                        metric_name="anomaly_f1_macro",
                        value=float("nan"),
                        metadata={
                            "error": "Se requieren synthetic_labels para entrenar la tarea."
                        },
                    )
                )
                return report

            self.task.fit(synthetic, synthetic_labels)
            metrics = self.task.test(real, real_labels)
            for k, v in metrics.items():
                report.results.append(EvaluationResult(metric_name=k, value=v))
            return report

        raise NotImplementedError(
            f"Tarea no soportada: {type(self.task).__name__}"
        )


# ---------------------------------------------------------------------------
# Individual evaluators
# ---------------------------------------------------------------------------

class TSTREvaluator(TaskRunnerEvaluator):
    """
    Evaluador TSTR puro:
    Train on Synthetic, Test on Real.
    """

    def __init__(self, max_features: int = 256) -> None:
        super().__init__(
            task=TSTRTask(max_features=max_features),
            name="TSTREvaluator",
            category="utility",
        )


class TRTREvaluator(TaskRunnerEvaluator):
    """
    Evaluador TRTR puro:
    Train on Real, Test on Real.
    """

    def __init__(self, max_features: int = 256) -> None:
        super().__init__(
            task=TRTRTask(max_features=max_features),
            name="TRTREvaluator",
            category="utility",
        )


# ---------------------------------------------------------------------------
# Optional comparator
# ---------------------------------------------------------------------------

class TSTRTRTRComparisonEvaluator(BaseEvaluator):
    """
    Ejecuta TSTR y TRTR y calcula el gap entre ambos.

    Este evaluador no sustituye a TSTREvaluator ni a TRTREvaluator.
    Solo agrega la comparación:
    - accuracy_gap = trtr_accuracy - tstr_accuracy
    - f1_gap = trtr_f1_macro - tstr_f1_macro
    """

    def __init__(self, max_features: int = 256) -> None:
        super().__init__(name="TSTRTRTRComparisonEvaluator")
        self.tstr = TSTREvaluator(max_features=max_features)
        self.trtr = TRTREvaluator(max_features=max_features)

    def evaluate(
        self,
        real: torch.Tensor,
        synthetic: torch.Tensor,
        real_labels: Optional[np.ndarray] = None,
        synthetic_labels: Optional[np.ndarray] = None,
        real_train: Optional[torch.Tensor] = None,
        real_train_labels: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> EvaluationReport:
        report = EvaluationReport(
            evaluator_name=self.name,
            results=[],
        )

        tstr_report = self.tstr.evaluate(
            real=real,
            synthetic=synthetic,
            real_labels=real_labels,
            synthetic_labels=synthetic_labels,
            **kwargs,
        )
        report.results.extend(tstr_report.results)

        if real_train is None or real_train_labels is None:
            return report

        trtr_report = self.trtr.evaluate(
            real=real,
            synthetic=synthetic,
            real_labels=real_labels,
            synthetic_labels=synthetic_labels,
            real_train=real_train,
            real_train_labels=real_train_labels,
            **kwargs,
        )
        report.results.extend(trtr_report.results)

        tstr_acc = tstr_report.get("tstr_accuracy").value if tstr_report.get("tstr_accuracy") else float("nan")
        tstr_f1 = tstr_report.get("tstr_f1_macro").value if tstr_report.get("tstr_f1_macro") else float("nan")
        trtr_acc = trtr_report.get("trtr_accuracy").value if trtr_report.get("trtr_accuracy") else float("nan")
        trtr_f1 = trtr_report.get("trtr_f1_macro").value if trtr_report.get("trtr_f1_macro") else float("nan")

        if np.isfinite(tstr_acc) and np.isfinite(trtr_acc):
            report.results.append(
                EvaluationResult(
                    metric_name="accuracy_gap",
                    value=float(trtr_acc - tstr_acc),
                    metadata={
                        "description": "trtr_accuracy - tstr_accuracy",
                    },
                )
            )

        if np.isfinite(tstr_f1) and np.isfinite(trtr_f1):
            report.results.append(
                EvaluationResult(
                    metric_name="f1_gap",
                    value=float(trtr_f1 - tstr_f1),
                )
            )

        return report