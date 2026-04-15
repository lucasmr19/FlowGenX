"""
src/evaluation/tasks/task_base.py
============================

Contratos base para tareas downstream.

   - Definen el protocolo de entrenamiento y test.
   - No dependen de BaseEvaluator.
   - Ejemplos: TSTRTask, TRTRTask, ClassificationProbeTask, AnomalyDetectionTask.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    """Convierte tensores o arrays a np.ndarray 2D/1D según corresponda."""
    if isinstance(x, np.ndarray):
        return x
    arr = x.detach().cpu().float().numpy()
    return arr


def _flatten(x: torch.Tensor | np.ndarray) -> np.ndarray:
    """Aplana (N, ...) -> (N, F)."""
    arr = _to_numpy(x)
    return arr.reshape(arr.shape[0], -1)


def _prepare_features(
    X_train: np.ndarray,
    X_test: np.ndarray,
    max_features: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estandariza y, si procede, reduce dimensionalidad con PCA.
    """
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    n_features = X_train.shape[1]
    if n_features > max_features:
        n_components = min(max_features, max(1, X_train.shape[0] - 1))
        if n_components < n_features:
            pca = PCA(n_components=n_components, random_state=42)
            X_train = pca.fit_transform(X_train)
            X_test = pca.transform(X_test)

    return X_train, X_test


def _mlp_classifier() -> MLPClassifier:
    """Clasificador ligero por defecto para tasks supervisadas."""
    return MLPClassifier(
        hidden_layer_sizes=(128, 64),
        max_iter=300,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
    )


def _fit_and_score_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[float, float]:
    """
    Entrena un clasificador simple y devuelve accuracy y macro-F1.
    """
    y_train = np.asarray(y_train).astype(int)
    y_test = np.asarray(y_test).astype(int)

    if len(np.unique(y_train)) < 2:
        return float("nan"), float("nan")

    clf = _mlp_classifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = float(accuracy_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    return acc, f1


# ---------------------------------------------------------------------------
# Task base
# ---------------------------------------------------------------------------

class DownstreamTask(ABC):
    """
    Contrato base de una tarea downstream.

    La tarea decide:
    - qué datos usa para entrenar
    - qué datos usa para test
    - qué métricas devuelve
    """

    name: str = "DownstreamTask"

    @abstractmethod
    def fit(self, X: torch.Tensor | np.ndarray, y: np.ndarray) -> None:
        raise NotImplementedError

    @abstractmethod
    def test(self, X: torch.Tensor | np.ndarray, y: np.ndarray) -> Dict[str, float]:
        raise NotImplementedError


class SupervisedClassificationTask(DownstreamTask):
    """
    Base para tareas supervisadas de clasificación.
    """

    def __init__(self, max_features: int = 256, classifier: str = "mlp") -> None:
        self.max_features = max_features
        self.classifier = classifier
        self._X_train: Optional[np.ndarray] = None
        self._y_train: Optional[np.ndarray] = None

    def _prepare(self, X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return _prepare_features(X_train, X_test, self.max_features)

    def fit(self, X: torch.Tensor | np.ndarray, y: np.ndarray) -> None:
        self._X_train = _flatten(X)
        self._y_train = np.asarray(y).astype(int)

    def test(self, X: torch.Tensor | np.ndarray, y: np.ndarray) -> Dict[str, float]:
        if self._X_train is None or self._y_train is None:
            raise RuntimeError("Primero llama a fit().")

        X_test = _flatten(X)
        X_train_prep, X_test_prep = self._prepare(self._X_train, X_test)
        acc, f1 = _fit_and_score_classifier(X_train_prep, self._y_train, X_test_prep, np.asarray(y))
        return {
            "accuracy": acc,
            "f1_macro": f1,
        }