from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from ..base import GenerativeModelConfig


# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

@dataclass
class DiffusionConfig(GenerativeModelConfig):
    """Hiperparámetros del modelo de difusión."""
    name: str = "traffic_ddpm"

    # Dimensiones de entrada
    in_channels:  int = 1           # 1 para nprint, 2 para GASF combinado
    image_height: int = 20          # H
    image_width:  int = 193         # W

    # UNet
    base_ch:           int           = 64
    channel_mults:     Tuple[int, ...] = (1, 2, 4)
    n_res_per_level:   int           = 2
    attention_levels:  Tuple[int, ...] = (1, 2)
    n_heads:           int           = 4
    dropout:           float         = 0.1

    # Condicionamiento por clase (0 = desactivado)
    num_classes: int = 0

    # Scheduler de ruido
    timesteps:   int   = 1000
    beta_start:  float = 1e-4
    beta_end:    float = 0.02
    beta_schedule: str = "cosine"   # "linear" | "cosine"

    # Muestreo DDIM acelerado
    ddim_steps:  int  = 50          # pasos de inferencia (< timesteps)
    ddim_eta:    float = 0.0        # 0.0 = determinístico, 1.0 = DDPM estándar

    # Clip en muestreo
    clip_denoised: bool = True