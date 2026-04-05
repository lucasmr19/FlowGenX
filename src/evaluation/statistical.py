"""
src/evaluation/statistical.py
==============================
Evaluador de fidelidad estadística.

Métricas implementadas
----------------------
1. Mean Earth Mover Distance (EMD / Wasserstein-1) por feature.
2. Mean Jensen-Shannon Divergence por feature.
3. Distancia entre matrices de correlación de Pearson.

Todas las métricas operan feature a feature sobre tensores aplanados a (N, F),
lo que las hace agnósticas a la representación (sequential, GASF, nprint).

Interpretación
--------------
- EMD y JS miden cuánto difieren las distribuciones marginales por feature.
  → Valores cercanos a 0 indican alta fidelidad estadística.
- La distancia Frobenius entre matrices de correlación captura si el modelo
  preserva la estructura de dependencia entre features.
  → Valores cercanos a 0 indican que las correlaciones están bien reproducidas.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance

from .base import BaseEvaluator, EvaluationReport, EvaluationResult


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------


def _to_numpy_2d(tensor: torch.Tensor) -> np.ndarray:
    """Aplana tensor (N, ...) a array NumPy (N, F)."""
    arr = tensor.detach().cpu().float().numpy()
    return arr.reshape(arr.shape[0], -1)


def _js_divergence_1d(
    a: np.ndarray,
    b: np.ndarray,
    bins: int = 64,
) -> float:
    """
    Jensen-Shannon divergence entre dos distribuciones 1D via histograma.

    Usa la raíz cuadrada de la divergencia JS (métrica real en [0, 1]).
    """
    lo = min(float(a.min()), float(b.min()))
    hi = max(float(a.max()), float(b.max()))
    if lo == hi:
        return 0.0
    edges = np.linspace(lo, hi, bins + 1)
    p, _ = np.histogram(a, bins=edges)
    q, _ = np.histogram(b, bins=edges)
    # jensenshannon de scipy espera distribuciones de probabilidad
    p = p.astype(float) + 1e-10
    q = q.astype(float) + 1e-10
    p /= p.sum()
    q /= q.sum()
    # jensenshannon devuelve la raíz cuadrada de la divergencia JS
    return float(jensenshannon(p, q))


def _correlation_matrix(arr: np.ndarray) -> np.ndarray:
    """
    Calcula la matriz de correlación de Pearson de (N, F) → (F, F).
    Reemplaza NaN por 0 (features constantes).
    """
    corr = np.corrcoef(arr.T)
    return np.nan_to_num(corr, nan=0.0)


# ---------------------------------------------------------------------------
# Evaluador principal
# ---------------------------------------------------------------------------


class StatisticalEvaluator(BaseEvaluator):
    """
    Fidelidad estadística entre tráfico real y sintético.

    Parameters
    ----------
    n_features_cap : int
        Número máximo de features a considerar (evita OOM en nprint/GASF).
        Por defecto 512.
    bins : int
        Número de bins para histogramas en el cálculo de JS.
        Por defecto 64.
    compute_correlation : bool
        Si es True, calcula también la distancia entre matrices de correlación.
        Puede ser costoso para n_features_cap elevado.
    """

    def __init__(
        self,
        n_features_cap: int = 512,
        bins: int = 64,
        compute_correlation: bool = True,
    ) -> None:
        super().__init__("StatisticalEvaluator")
        self.n_features_cap = n_features_cap
        self.bins = bins
        self.compute_correlation = compute_correlation

    # ------------------------------------------------------------------
    # Interfaz pública
    # ------------------------------------------------------------------

    def evaluate(
        self,
        real: torch.Tensor,
        synthetic: torch.Tensor,
        **kwargs,
    ) -> EvaluationReport:
        """
        Calcula fidelidad estadística entre tensores real y sintético.

        Parameters
        ----------
        real : torch.Tensor
            Muestras reales, shape (N, ...).
        synthetic : torch.Tensor
            Muestras sintéticas, shape (M, ...).

        Returns
        -------
        EvaluationReport con métricas:
            - mean_emd
            - mean_js_divergence
            - pearson_corr_matrix_distance  (si compute_correlation=True)
        """
        report = EvaluationReport(evaluator_name=self.name)

        real_np = _to_numpy_2d(real)
        synth_np = _to_numpy_2d(synthetic)

        # Aplicar cap de features
        n_features = min(real_np.shape[1], synth_np.shape[1], self.n_features_cap)
        real_np = real_np[:, :n_features]
        synth_np = synth_np[:, :n_features]

        # 1. EMD por feature
        report.results.append(self._compute_emd(real_np, synth_np, n_features))

        # 2. JS Divergence por feature
        report.results.append(self._compute_js(real_np, synth_np, n_features))

        # 3. Distancia entre matrices de correlación de Pearson
        if self.compute_correlation and n_features > 1:
            corr_result = self._compute_correlation_distance(real_np, synth_np, n_features)
            if corr_result is not None:
                report.results.append(corr_result)

        return report

    # ------------------------------------------------------------------
    # Cálculo de métricas individuales
    # ------------------------------------------------------------------

    def _compute_emd(
        self,
        real: np.ndarray,
        synth: np.ndarray,
        n_features: int,
    ) -> EvaluationResult:
        emd_per_feature: List[float] = []
        for i in range(n_features):
            emd_per_feature.append(
                wasserstein_distance(real[:, i], synth[:, i])
            )
        mean_emd = float(np.mean(emd_per_feature))
        return EvaluationResult(
            metric_name="mean_emd",
            value=mean_emd,
            metadata={
                "n_features_evaluated": n_features,
                "min_emd": float(np.min(emd_per_feature)),
                "max_emd": float(np.max(emd_per_feature)),
                "std_emd": float(np.std(emd_per_feature)),
                # Primeras 16 features para inspección
                "per_feature_sample": emd_per_feature[:16],
            },
        )

    def _compute_js(
        self,
        real: np.ndarray,
        synth: np.ndarray,
        n_features: int,
    ) -> EvaluationResult:
        js_per_feature: List[float] = []
        for i in range(n_features):
            js_per_feature.append(
                _js_divergence_1d(real[:, i], synth[:, i], self.bins)
            )
        mean_js = float(np.mean(js_per_feature))
        return EvaluationResult(
            metric_name="mean_js_divergence",
            value=mean_js,
            metadata={
                "n_features_evaluated": n_features,
                "min_js": float(np.min(js_per_feature)),
                "max_js": float(np.max(js_per_feature)),
                "std_js": float(np.std(js_per_feature)),
                "per_feature_sample": js_per_feature[:16],
            },
        )

    def _compute_correlation_distance(
        self,
        real: np.ndarray,
        synth: np.ndarray,
        n_features: int,
    ) -> Optional[EvaluationResult]:
        """
        Distancia de Frobenius entre matrices de correlación de Pearson.

        Necesita al menos 3 muestras para calcular correlaciones fiables.
        """
        min_samples = min(real.shape[0], synth.shape[0])
        if min_samples < 3:
            return None

        real_corr = _correlation_matrix(real)
        synth_corr = _correlation_matrix(synth)
        frobenius_dist = float(np.linalg.norm(real_corr - synth_corr, "fro"))

        # Normalizar por número de features para comparabilidad entre shapes
        normalized_dist = frobenius_dist / n_features

        return EvaluationResult(
            metric_name="pearson_corr_matrix_distance",
            value=normalized_dist,
            metadata={
                "frobenius_raw": frobenius_dist,
                "n_features": n_features,
            },
        )