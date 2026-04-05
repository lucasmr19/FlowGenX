"""
src/evaluation/downstream.py
=============================
Evaluación basada en tarea: protocolo TSTR + baseline TRTR.

Protocolo TSTR (Train on Synthetic, Test on Real)
-------------------------------------------------
Entrena un clasificador simple con datos sintéticos etiquetados y
evalúa sobre datos reales. La diferencia de rendimiento respecto
al baseline TRTR (Train on Real, Test on Real) cuantifica la
utilidad práctica del tráfico generado.

Interpretación de resultados
-----------------------------
- tstr_accuracy / tstr_f1       : Rendimiento con datos sintéticos.
- trtr_accuracy / trtr_f1       : Baseline con datos reales (si se proporciona).
- accuracy_gap = trtr - tstr    : Cuánto peor es el sintético vs el real.
  → Gap ≈ 0  : Los datos sintéticos son tan útiles como los reales.
  → Gap > 0.1: El modelo generativo no preserva la semántica de clase.

Clasificador interno
---------------------
Se usa un MLPClassifier de sklearn (ligero, sin GPU). Para datasets
grandes o representaciones de alta dimensión se aplica PCA previo.
El objetivo no es maximizar el accuracy en sí, sino usarlo como
señal comparativa entre configuraciones representación--modelo.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from .base import BaseEvaluator, EvaluationReport, EvaluationResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _flatten(tensor: torch.Tensor) -> np.ndarray:
    """Aplana tensor (N, ...) a array NumPy (N, F)."""
    arr = tensor.detach().cpu().float().numpy()
    return arr.reshape(arr.shape[0], -1)


def _prepare_features(
    X_train: np.ndarray,
    X_test: np.ndarray,
    max_features: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normaliza y opcionalmente reduce dimensionalidad con PCA.

    Parameters
    ----------
    X_train, X_test : np.ndarray shape (N, F)
    max_features : int
        Si F > max_features, aplica PCA para reducir a max_features componentes.

    Returns
    -------
    X_train_prep, X_test_prep normalizados (y reducidos si aplica).
    """
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    n_features = X_train.shape[1]
    if n_features > max_features:
        n_components = min(max_features, X_train.shape[0] - 1)
        pca = PCA(n_components=n_components, random_state=42)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

    return X_train, X_test


def _train_and_eval(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[float, float]:
    """
    Entrena un MLP simple y devuelve (accuracy, macro-f1) sobre test.
    """
    clf = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        max_iter=300,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    return acc, f1


# ---------------------------------------------------------------------------
# Evaluador TSTR
# ---------------------------------------------------------------------------


class TSTREvaluator(BaseEvaluator):
    """
    Evaluación basada en tarea mediante el protocolo TSTR.

    Parameters
    ----------
    max_features : int
        Dimensionalidad máxima antes de aplicar PCA. Por defecto 256.
    include_trtr : bool
        Si True, también calcula el baseline TRTR usando `real_train_X`
        y `real_train_y` proporcionados en kwargs de `evaluate()`.
    """

    def __init__(
        self,
        max_features: int = 256,
        include_trtr: bool = True,
    ) -> None:
        super().__init__("TSTREvaluator")
        self.max_features = max_features
        self.include_trtr = include_trtr

    def evaluate(
        self,
        real: torch.Tensor,
        synthetic: torch.Tensor,
        real_labels: Optional[np.ndarray] = None,
        synthetic_labels: Optional[np.ndarray] = None,
        real_train: Optional[torch.Tensor] = None,
        real_train_labels: Optional[np.ndarray] = None,
        **kwargs,
    ) -> EvaluationReport:
        """
        Ejecuta el protocolo TSTR y opcionalmente el baseline TRTR.

        Parameters
        ----------
        real : torch.Tensor, shape (N_test, ...)
            Datos reales usados como conjunto de TEST.
        synthetic : torch.Tensor, shape (N_train, ...)
            Datos sintéticos usados como conjunto de TRAIN.
        real_labels : np.ndarray, shape (N_test,)
            Etiquetas de clase para los datos reales de test.
        synthetic_labels : np.ndarray, shape (N_train,)
            Etiquetas de clase para los datos sintéticos de train.
        real_train : torch.Tensor, optional
            Datos reales de entrenamiento para el baseline TRTR.
            Solo necesario si include_trtr=True.
        real_train_labels : np.ndarray, optional
            Etiquetas para real_train.

        Returns
        -------
        EvaluationReport con métricas:
            - tstr_accuracy, tstr_f1
            - trtr_accuracy, trtr_f1  (si include_trtr=True y se pasan datos)
            - accuracy_gap, f1_gap    (si ambos disponibles)
        """
        report = EvaluationReport(evaluator_name=self.name)

        if real_labels is None or synthetic_labels is None:
            report.results.append(EvaluationResult(
                metric_name="tstr_accuracy",
                value=float("nan"),
                metadata={"error": "Se requieren real_labels y synthetic_labels para TSTR."},
            ))
            return report

        # Preparar features
        X_real = _flatten(real)
        X_synth = _flatten(synthetic)

        X_synth_prep, X_real_prep = _prepare_features(
            X_synth, X_real, self.max_features
        )

        # --- TSTR ---
        tstr_acc, tstr_f1 = _train_and_eval(
            X_synth_prep, synthetic_labels,
            X_real_prep, real_labels,
        )
        report.results.append(EvaluationResult(
            metric_name="tstr_accuracy",
            value=tstr_acc,
            metadata={
                "n_train_synth": len(X_synth),
                "n_test_real": len(X_real),
                "n_classes": len(np.unique(real_labels)),
            },
        ))
        report.results.append(EvaluationResult(
            metric_name="tstr_f1_macro",
            value=tstr_f1,
        ))

        # --- Baseline TRTR (opcional) ---
        if (
            self.include_trtr
            and real_train is not None
            and real_train_labels is not None
        ):
            X_real_train = _flatten(real_train)
            X_rt_prep, X_real_test_prep = _prepare_features(
                X_real_train, X_real, self.max_features
            )
            trtr_acc, trtr_f1 = _train_and_eval(
                X_rt_prep, real_train_labels,
                X_real_test_prep, real_labels,
            )
            report.results.append(EvaluationResult(
                metric_name="trtr_accuracy",
                value=trtr_acc,
                metadata={"n_train_real": len(X_real_train)},
            ))
            report.results.append(EvaluationResult(
                metric_name="trtr_f1_macro",
                value=trtr_f1,
            ))

            # Gap TRTR - TSTR (cuanto mayor, peor el sintético)
            report.results.append(EvaluationResult(
                metric_name="accuracy_gap",
                value=trtr_acc - tstr_acc,
                metadata={
                    "description": "trtr_accuracy - tstr_accuracy. "
                                   "Gap ≈ 0 → datos sintéticos tan útiles como reales.",
                },
            ))
            report.results.append(EvaluationResult(
                metric_name="f1_gap",
                value=trtr_f1 - tstr_f1,
            ))

        return report