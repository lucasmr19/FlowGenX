"""
src/evaluation/conditional.py
==============================
Evaluador de generación condicional para modelos generativos condicionados.

Mide si las muestras sintéticas generadas bajo una etiqueta de clase
son correctamente identificadas como pertenecientes a esa clase por un
clasificador entrenado sobre datos reales.

Protocolo
---------
1. Entrena un clasificador supervisado sobre (real_data, real_labels).
2. Predice la clase de cada muestra sintética: preds = clf.predict(synth).
3. Compara las predicciones con las etiquetas condicionantes originales.

Métricas
--------
- conditional_accuracy  : precisión de clase media sobre las muestras sintéticas.
- conditional_f1_macro  : macro-F1 sobre las muestras sintéticas.

Interpretación
--------------
- Valor alto → el modelo genera muestras que "parecen" de la clase correcta
  a ojos de un clasificador real.
- Valor bajo → el modelo ignora o confunde la condición.

Nota: esta métrica es complementaria a las estadísticas de distribución
(JS, EMD) y al URS. No reemplaza ninguna de ellas.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from .base import BaseEvaluator, EvaluationReport, EvaluationResult

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ConditionalEvaluator
# ---------------------------------------------------------------------------


class ConditionalEvaluator(BaseEvaluator):
    """
    Evaluador de fidelidad condicional.

    Evalúa si las muestras sintéticas generadas con una etiqueta de clase
    son reconocidas como pertenecientes a esa clase por un clasificador
    entrenado sobre datos reales.

    Parameters
    ----------
    max_features : int
        Número máximo de features tras PCA. Por defecto 256.
    classifier : str
        Tipo de clasificador interno. Actualmente sólo ``"mlp"`` está soportado.

    Raises
    ------
    ValueError
        Si ``real_labels`` o ``synthetic_labels`` no se proporcionan en
        ``evaluate()``.
    """

    def __init__(
        self,
        max_features: int = 256,
        classifier: str = "mlp",
    ) -> None:
        super().__init__(name="ConditionalEvaluator", category="conditional")
        self.max_features = max_features
        self.classifier = classifier

    # ------------------------------------------------------------------
    # BaseEvaluator contract
    # ------------------------------------------------------------------

    def evaluate(
        self,
        real: torch.Tensor,
        synthetic: torch.Tensor,
        real_labels: Optional[np.ndarray] = None,
        synthetic_labels: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> EvaluationReport:
        """
        Evalúa la fidelidad condicional del modelo generativo.

        Parameters
        ----------
        real : torch.Tensor, shape (N, ...)
            Muestras reales de referencia.
        synthetic : torch.Tensor, shape (M, ...)
            Muestras sintéticas generadas condicionalmente.
        real_labels : np.ndarray, shape (N,)
            Etiquetas reales para entrenar el clasificador.
        synthetic_labels : np.ndarray, shape (M,)
            Etiquetas condicionantes usadas al generar las muestras sintéticas.

        Returns
        -------
        EvaluationReport
            Con métricas ``conditional_accuracy`` y ``conditional_f1_macro``.
        """
        report = EvaluationReport(
            evaluator_name=self.name,
            results=[],
            category=self.category,
        )

        # --- Guard: labels required ------------------------------------------
        if real_labels is None:
            LOGGER.warning(
                "[ConditionalEvaluator] real_labels no disponibles. "
                "No se puede evaluar la fidelidad condicional."
            )
            report.results.append(
                EvaluationResult(
                    metric_name="conditional_accuracy",
                    value=float("nan"),
                    metadata={"error": "real_labels requeridos"},
                )
            )
            report.results.append(
                EvaluationResult(
                    metric_name="conditional_f1_macro",
                    value=float("nan"),
                    metadata={"error": "real_labels requeridos"},
                )
            )
            return report

        if synthetic_labels is None:
            LOGGER.warning(
                "[ConditionalEvaluator] synthetic_labels no disponibles. "
                "Sin etiquetas condicionantes no se puede calcular la fidelidad condicional."
            )
            report.results.append(
                EvaluationResult(
                    metric_name="conditional_accuracy",
                    value=float("nan"),
                    metadata={"error": "synthetic_labels requeridos"},
                )
            )
            report.results.append(
                EvaluationResult(
                    metric_name="conditional_f1_macro",
                    value=float("nan"),
                    metadata={"error": "synthetic_labels requeridos"},
                )
            )
            return report

        # --- Check class diversity -------------------------------------------
        unique_real = np.unique(np.asarray(real_labels))
        if len(unique_real) < 2:
            LOGGER.warning(
                "[ConditionalEvaluator] Sólo hay %d clase(s) en real_labels. "
                "Se requieren al menos 2 para entrenar el clasificador.",
                len(unique_real),
            )
            report.results.append(
                EvaluationResult(
                    metric_name="conditional_accuracy",
                    value=float("nan"),
                    metadata={"error": "Se requieren >= 2 clases en real_labels"},
                )
            )
            report.results.append(
                EvaluationResult(
                    metric_name="conditional_f1_macro",
                    value=float("nan"),
                    metadata={"error": "Se requieren >= 2 clases en real_labels"},
                )
            )
            return report

        # --- Feature preparation --------------------------------------------
        try:
            real_np  = self._to_features(real)
            synth_np = self._to_features(synthetic)
            real_y   = np.asarray(real_labels).astype(int)
            synth_y  = np.asarray(synthetic_labels).astype(int)

            real_np, synth_np = self._preprocess(real_np, synth_np)

            # --- Train classifier on real data ------------------------------
            clf = self._build_classifier()
            clf.fit(real_np, real_y)

            # --- Predict on synthetic, compare with conditioning labels -----
            preds = clf.predict(synth_np)
            acc = float(accuracy_score(synth_y, preds))
            f1  = float(f1_score(synth_y, preds, average="macro", zero_division=0))

            LOGGER.info(
                "[ConditionalEvaluator] conditional_accuracy=%.4f  "
                "conditional_f1_macro=%.4f",
                acc, f1,
            )

            report.results.append(
                EvaluationResult(
                    metric_name="conditional_accuracy",
                    value=acc,
                    metadata={
                        "n_real":  len(real_y),
                        "n_synth": len(synth_y),
                        "classes": unique_real.tolist(),
                    },
                )
            )
            report.results.append(
                EvaluationResult(
                    metric_name="conditional_f1_macro",
                    value=f1,
                    metadata={"average": "macro"},
                )
            )

        except Exception as exc:
            LOGGER.warning(
                "[ConditionalEvaluator] Error durante la evaluación: %s", exc,
                exc_info=True,
            )
            report.results.append(
                EvaluationResult(
                    metric_name="conditional_accuracy",
                    value=float("nan"),
                    metadata={"error": str(exc)},
                )
            )
            report.results.append(
                EvaluationResult(
                    metric_name="conditional_f1_macro",
                    value=float("nan"),
                    metadata={"error": str(exc)},
                )
            )

        return report

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_features(x: torch.Tensor | np.ndarray) -> np.ndarray:
        """Convierte tensor o array a ndarray 2D (N, F)."""
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().float().numpy()
        return np.asarray(x).reshape(x.shape[0], -1)

    def _preprocess(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Estandariza y reduce dimensionalidad si es necesario."""
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        n_features = X_train.shape[1]
        if n_features > self.max_features:
            n_components = min(self.max_features, max(1, X_train.shape[0] - 1))
            if n_components < n_features:
                pca = PCA(n_components=n_components, random_state=42)
                X_train = pca.fit_transform(X_train)
                X_test  = pca.transform(X_test)

        return X_train, X_test

    def _build_classifier(self) -> MLPClassifier:
        """Construye el clasificador interno."""
        return MLPClassifier(
            hidden_layer_sizes=(128, 64),
            max_iter=300,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
        )