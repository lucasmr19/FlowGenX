"""
src/evaluation/structural.py
=============================
Evaluador de validez estructural del tráfico sintético.

Métricas implementadas
----------------------
1. valid_sample_rate      : Fracción de muestras que superan todas las
                            restricciones de la representación.
2. field_range_violation  : Fracción de valores fuera del rango esperado.
3. binary_field_purity    : (solo nprint) Fracción de valores binarios
                            que son estrictamente 0 o 1 (post-umbral).

Representaciones soportadas
----------------------------
- "sequential"  Tokens enteros ≥ 0 y < vocab_size.
- "gasf"        Valores flotantes en [-1.0, 1.0].
- "nprint"      Campos binarios en {0, 1} (tolerancia configurable).

El evaluador es intencionalmente estricto: detecta violaciones que
indican que el modelo generativo no ha aprendido las restricciones
estructurales de la representación.

Nota sobre invertibilidad
--------------------------
La tasa de reconstrucción funcional (conversión sintético → .pcap válido)
requiere acceso a la representación concreta y se delega a cada
RepresentationBase mediante `decode_batch`. Este evaluador cubre la
capa anterior: ¿los tensores generados cumplen las restricciones de
la representación?
"""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np
import torch

from .base import BaseEvaluator, EvaluationReport, EvaluationResult

RepresentationType = Literal["sequential", "gasf", "nprint"]


# ---------------------------------------------------------------------------
# Evaluador
# ---------------------------------------------------------------------------


class StructuralEvaluator(BaseEvaluator):
    """
    Validez estructural del tráfico sintético según el tipo de representación.

    Parameters
    ----------
    representation_type : "sequential" | "gasf" | "nprint"
        Tipo de representación. Determina qué restricciones se verifican.
    vocab_size : int, optional
        Solo para representation_type="sequential".
        Número de tokens válidos. Los tokens deben estar en [0, vocab_size).
    binary_threshold : float
        Solo para representation_type="nprint".
        Un valor se considera binario si está en [0, t] U [1-t, 1] con t=threshold.
        Por defecto 0.1 (permite cierto error de redondeo post-generación).
    gasf_tolerance : float
        Solo para representation_type="gasf".
        Tolerancia alrededor de [-1, 1]. Por defecto 0.05.
    """

    def __init__(
        self,
        representation_type: RepresentationType,
        vocab_size: Optional[int] = None,
        binary_threshold: float = 0.1,
        gasf_tolerance: float = 0.05,
    ) -> None:
        super().__init__("StructuralEvaluator")
        self.representation_type = representation_type
        self.vocab_size = vocab_size
        self.binary_threshold = binary_threshold
        self.gasf_tolerance = gasf_tolerance

        if representation_type == "sequential" and vocab_size is None:
            raise ValueError(
                "vocab_size es obligatorio para representation_type='sequential'."
            )

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
        Evalúa la validez estructural de las muestras sintéticas.

        El tensor `real` se usa solo como referencia para calcular
        la tasa de violaciones en datos reales (baseline esperado ≈ 0.0).

        Returns
        -------
        EvaluationReport con métricas según el tipo de representación.
        """
        report = EvaluationReport(evaluator_name=self.name)

        synth_np = synthetic.detach().cpu().float().numpy()
        real_np = real.detach().cpu().float().numpy()

        if self.representation_type == "sequential":
            report.results.extend(self._evaluate_sequential(real_np, synth_np))
        elif self.representation_type == "gasf":
            report.results.extend(self._evaluate_gasf(real_np, synth_np))
        elif self.representation_type == "nprint":
            report.results.extend(self._evaluate_nprint(real_np, synth_np))
        else:
            raise ValueError(
                f"Tipo de representación desconocido: '{self.representation_type}'. "
                "Usa 'sequential', 'gasf' o 'nprint'."
            )

        return report

    # ------------------------------------------------------------------
    # Validación por tipo de representación
    # ------------------------------------------------------------------

    def _evaluate_sequential(
        self,
        real: np.ndarray,
        synth: np.ndarray,
    ):
        """
        Sequential: tokens deben ser enteros en [0, vocab_size).

        Restricciones verificadas
        -------------------------
        - Todos los valores ≥ 0.
        - Todos los valores < vocab_size.
        - Todos los valores son enteros (parte decimal = 0).
        """
        results = []

        # Fracción de tokens fuera de rango [0, vocab_size)
        out_of_range_synth = np.mean(
            (synth < 0) | (synth >= self.vocab_size)
        )
        out_of_range_real = np.mean(
            (real < 0) | (real >= self.vocab_size)
        )
        results.append(EvaluationResult(
            metric_name="token_out_of_range_rate",
            value=float(out_of_range_synth),
            metadata={
                "vocab_size": self.vocab_size,
                "real_baseline": float(out_of_range_real),
                "description": "Fracción de tokens fuera de [0, vocab_size). "
                               "Baseline real debería ser 0.0.",
            },
        ))

        # Fracción de valores no enteros
        non_integer_synth = np.mean(np.abs(synth - np.round(synth)) > 1e-3)
        results.append(EvaluationResult(
            metric_name="non_integer_token_rate",
            value=float(non_integer_synth),
            metadata={
                "description": "Fracción de tokens con parte decimal ≠ 0. "
                               "Los modelos autoregresivos deberían producir 0.0.",
            },
        ))

        # Tasa de muestras completamente válidas
        valid_mask = np.all(
            (synth >= 0) & (synth < self.vocab_size)
            & (np.abs(synth - np.round(synth)) < 1e-3),
            axis=tuple(range(1, synth.ndim)),
        )
        results.append(EvaluationResult(
            metric_name="valid_sample_rate",
            value=float(np.mean(valid_mask)),
            metadata={"n_valid": int(np.sum(valid_mask)), "n_total": len(synth)},
        ))

        return results

    def _evaluate_gasf(
        self,
        real: np.ndarray,
        synth: np.ndarray,
    ):
        """
        GASF: valores deben estar en [-1, 1].

        Las imágenes GASF son cosenos de sumas de ángulos, por lo que
        están acotadas en [-1, 1] por construcción. Un modelo que viola
        esto ha generado imágenes fuera del dominio válido.
        """
        results = []
        lo = -1.0 - self.gasf_tolerance
        hi = 1.0 + self.gasf_tolerance

        # Fracción de pixels fuera de rango
        out_synth = np.mean((synth < lo) | (synth > hi))
        out_real = np.mean((real < lo) | (real > hi))
        results.append(EvaluationResult(
            metric_name="pixel_out_of_range_rate",
            value=float(out_synth),
            metadata={
                "expected_range": f"[{lo:.2f}, {hi:.2f}]",
                "real_baseline": float(out_real),
                "synth_min": float(synth.min()),
                "synth_max": float(synth.max()),
                "real_min": float(real.min()),
                "real_max": float(real.max()),
            },
        ))

        # Tasa de imágenes completamente válidas
        per_sample_valid = np.all(
            (synth >= lo) & (synth <= hi),
            axis=tuple(range(1, synth.ndim)),
        )
        results.append(EvaluationResult(
            metric_name="valid_sample_rate",
            value=float(np.mean(per_sample_valid)),
            metadata={"n_valid": int(np.sum(per_sample_valid)), "n_total": len(synth)},
        ))

        return results

    def _evaluate_nprint(
        self,
        real: np.ndarray,
        synth: np.ndarray,
    ):
        """
        nprint: campos deben ser binarios ∈ {0, 1}.

        El formato nPrint codifica los bytes de la cabecera como bits,
        por lo que cada campo debe ser 0 o 1. Los modelos de difusión
        generan valores continuos que deben ser umbralados; esta métrica
        cuantifica cuántos valores NO son binarios antes de umbralizar.

        Shape esperado: (N, n_packets, n_features) o (N, n_features).
        """
        results = []
        t = self.binary_threshold

        # Un valor es "binario" si está en [0, t] ∪ [1-t, 1]
        is_binary = (
            ((synth >= 0.0) & (synth <= t))
            | ((synth >= (1.0 - t)) & (synth <= 1.0))
        )
        non_binary_rate_synth = float(np.mean(~is_binary))

        is_binary_real = (
            ((real >= 0.0) & (real <= t))
            | ((real >= (1.0 - t)) & (real <= 1.0))
        )
        non_binary_rate_real = float(np.mean(~is_binary_real))

        results.append(EvaluationResult(
            metric_name="non_binary_field_rate",
            value=non_binary_rate_synth,
            metadata={
                "binary_threshold": t,
                "real_baseline": non_binary_rate_real,
                "description": "Fracción de campos que no están en {0,1} "
                               "(antes de umbralizar). Real debería ser ≈ 0.0.",
            },
        ))

        # Tasa de paquetes completamente válidos (todos los campos binarios)
        # Shape (N, n_packets, n_features) → valid si todos los campos del paquete son binarios
        per_sample_valid = np.all(
            is_binary,
            axis=tuple(range(1, synth.ndim)),
        )
        results.append(EvaluationResult(
            metric_name="valid_sample_rate",
            value=float(np.mean(per_sample_valid)),
            metadata={"n_valid": int(np.sum(per_sample_valid)), "n_total": len(synth)},
        ))

        # Tasa de paquetes reconstruibles tras umbralización (≥ 50% de campos binarios exactos)
        binarized = (synth >= 0.5).astype(float)
        exact_match_rate_per_field = float(
            np.mean(np.abs(synth - binarized) < t)
        )
        results.append(EvaluationResult(
            metric_name="binarization_confidence",
            value=exact_match_rate_per_field,
            metadata={
                "description": "Fracción de campos que caen cerca del umbral 0.5 "
                               "tras binarización. Valores altos indican que el "
                               "modelo ha aprendido bien la naturaleza binaria.",
            },
        ))

        return results