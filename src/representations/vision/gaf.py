"""
representations/vision/gaf.py
==============================

Stateless Gramian Angular Field (GAF) representation of 1D sequences.

This implementation follows the NetDiffus pipeline:

https://github.com/Nirhoshan/NetDiffus/blob/main/gasf_conversion.py

- No dataset-level fitting or global normalization
- Per-sample scaling
- Optional sum-based binning
- Supports GASF / GADF
- Gamma correction and resizing
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional, Tuple
import numpy as np
import torch
from torch import Tensor
from skimage.transform import resize

from ...preprocessing import TrafficChunk
from ..base import TrafficRepresentation, RepresentationConfig, RepresentationType, Invertibility


@dataclass
class GAFConfig(RepresentationConfig):
    representation_type: str = "gaf"
    name: str = "gaf"
    image_size: int = 128             # final square image
    method: str = "summation"         # "summation" (GASF) or "difference" (GADF)
    gamma: Optional[float] = 0.25     # gamma correction exponent
    use_binning: bool = True
    bin_size: int = 4                 # block size for sum-binning
    field_name: str = "payload_len"   # ParsedPacket field to encode
    rescale_to_01: bool = True


class GAFRepresentation(TrafficRepresentation):
    """
    GAF representation that works directly over TrafficChunk objects.
    """

    def __init__(self, config: Optional[GAFConfig] = None) -> None:
        config = config or GAFConfig()
        super().__init__(config)
        self.cfg = config
        self._is_fitted = True  # stateless

    @property
    def representation_type(self) -> RepresentationType:
        return RepresentationType.VISUAL

    @property
    def invertibility(self) -> Invertibility:
        return Invertibility.NON_INVERTIBLE

    @property
    def output_shape(self) -> Tuple[int, ...]:
        s = self.cfg.image_size
        return (1, s, s)

    def fit(self, samples: list[Any]) -> "GAFRepresentation":
        # Stateless
        self._is_fitted = True
        return self

    def encode(self, window: TrafficChunk) -> Tensor:
      """
      Convert a TrafficChunk into a GAF image.
      Works robustly for real PCAP, avoiding constant matrices.
      """
      # 1. Extraer serie numérica desde los ParsedPacket
      ts = np.array([getattr(pkt, self.cfg.field_name) for pkt in window.packets], dtype=np.float32)

      # 2. Binning opcional
      if self.cfg.use_binning and len(ts) > self.cfg.bin_size:
          ts = self._sum_binning(ts, self.cfg.bin_size)

      # 3. Interpolación al tamaño de imagen
      s = self.cfg.image_size
      if len(ts) != s:
          ts = self._interpolate(ts, s)

      # 4. Escala robusta [-1, 1] usando min-max por flujo
      ts_min, ts_max = ts.min(), ts.max()
      if ts_max - ts_min > 1e-6:
          ts = (ts - ts_min) / (ts_max - ts_min)  # [0,1]
          ts = ts * 2.0 - 1.0                     # [-1,1]
      else:
          ts = np.zeros_like(ts)  # evita matriz constante de 1s o -1s

      # 5. Computar GAF
      gaf = self._compute_gaf(ts)

      # 6. Rescale a [0,1] si corresponde
      if self.cfg.rescale_to_01:
          gaf = gaf * 0.5 + 0.5

      # 7. Aplicar gamma solo si está definido
      if self.cfg.gamma is not None:
          gaf = np.power(np.clip(gaf, 0.0, 1.0), self.cfg.gamma)

      # 8. Ajuste final de tamaño por si acaso
      if gaf.shape[0] != s:
          gaf = resize(gaf, (s, s), anti_aliasing=True)

      return torch.tensor(gaf[np.newaxis, ...], dtype=torch.float32)

    def decode(self, tensor: Tensor) -> NotImplementedError:
        raise NotImplementedError("GAF is non-invertible.")
    
    def get_default_aggregator(self):
        from ...preprocessing import TrafficChunkAggregator
        return TrafficChunkAggregator

    def project(self, x, **kwargs):
        if self.cfg.rescale_to_01:
            x = x.clamp(0.0, 1.0)
        else:
            x = x.clamp(-1.0, 1.0)

        # simetría sí es correcta
        x = 0.5 * (x + x.transpose(-1, -2))

        return x

    # -----------------------
    # Helpers
    # -----------------------
    def _compute_gaf(self, ts: np.ndarray) -> np.ndarray:
        ts = np.clip(ts, -1.0, 1.0)
        phi = np.arccos(ts)
        cos_phi, sin_phi = np.cos(phi), np.sin(phi)

        if self.cfg.method in ("summation", "s"):
            return np.outer(cos_phi, cos_phi) - np.outer(sin_phi, sin_phi)
        if self.cfg.method in ("difference", "d"):
            return np.outer(sin_phi, cos_phi) - np.outer(cos_phi, sin_phi)
        raise ValueError(f"Invalid method '{self.cfg.method}'.")

    def _sum_binning(self, ts: np.ndarray, bin_size: int) -> np.ndarray:
        n_bins = len(ts) // bin_size
        if n_bins == 0:
            return np.array([ts.sum()], dtype=np.float32)
        trimmed = ts[: n_bins * bin_size]
        return trimmed.reshape(n_bins, bin_size).sum(axis=1)

    def _interpolate(self, ts: np.ndarray, size: int) -> np.ndarray:
        if len(ts) < 2:
            return np.repeat(ts, size)
        x_old = np.linspace(0, 1, len(ts), dtype=np.float32)
        x_new = np.linspace(0, 1, size, dtype=np.float32)
        return np.interp(x_new, x_old, ts)