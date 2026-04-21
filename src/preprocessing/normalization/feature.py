from __future__ import annotations

from typing import Dict
import numpy as np

class FeatureNormalizer:
    """
    Ajusta y aplica normalización (min-max o z-score) sobre arrays numéricos.

    Se ajusta SOLO sobre datos de entrenamiento para evitar data leakage.

    Parameters
    ----------
    method : "minmax" | "zscore"
    clip   : si True, hace clip a [0, 1] tras min-max
    eps    : evita división por cero
    """

    def __init__(
        self,
        method: str   = "minmax",
        clip:   bool  = True,
        eps:    float = 1e-8,
    ) -> None:
        assert method in ("minmax", "zscore"), f"Método desconocido: {method}"
        self.method  = method
        self.clip    = clip
        self.eps     = eps
        self._fitted = False
        self._params: Dict[str, np.ndarray] = {}

    def fit(self, data: np.ndarray) -> "FeatureNormalizer":
        """data shape: (N, F)"""
        if self.method == "minmax":
            self._params["min"] = data.min(axis=0)
            self._params["max"] = data.max(axis=0)
        else:
            self._params["mean"] = data.mean(axis=0)
            self._params["std"]  = data.std(axis=0)
        self._fitted = True
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Llama a fit() antes de transform().")
        data = data.astype(np.float32)
        if self.method == "minmax":
            rng = self._params["max"] - self._params["min"] + self.eps
            out = (data - self._params["min"]) / rng
            if self.clip:
                out = np.clip(out, 0.0, 1.0)
        else:
            out = (data - self._params["mean"]) / (self._params["std"] + self.eps)
        return out

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        return self.fit(data).transform(data)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Llama a fit() antes de inverse_transform().")
        data = data.astype(np.float32)
        if self.method == "minmax":
            rng = self._params["max"] - self._params["min"] + self.eps
            return data * rng + self._params["min"]
        else:
            return data * (self._params["std"] + self.eps) + self._params["mean"]

    def get_params(self) -> Dict[str, np.ndarray]:
        return dict(self._params)

    def set_params(self, params: Dict[str, np.ndarray]) -> None:
        self._params = params
        self._fitted = bool(params)