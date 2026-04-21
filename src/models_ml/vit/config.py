"""
generative_models/vit/config.py
================================
Configuración del Vision Transformer generativo (MAE-style).

Sigue la misma convención que TransformerConfig para facilitar
la integración con el resto del pipeline (trainers, loggers, serialización).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Tuple

from ..base import GenerativeModelConfig


@dataclass
class ViTConfig(GenerativeModelConfig):
    """
    Hiperparámetros del Vision Transformer autoencoder (MAE-style).

    Organización
    ------------
    1. Metadatos del modelo
    2. Representación de imagen
    3. Encoder ViT
    4. Decoder MAE
    5. Condicionalidad
    6. Entrenamiento y regularización
    7. Generación / muestreo
    """

    # ------------------------------------------------------------------
    # 1. Metadatos
    # ------------------------------------------------------------------
    model_type: str = "vit"
    name:       str = "traffic_vit"

    # ------------------------------------------------------------------
    # 2. Representación de imagen
    # ------------------------------------------------------------------

    image_size:  int = 224          # Lado del cuadrado de entrada (píxeles)
    patch_size:  int = 16           # Lado de cada parche (píxeles); image_size % patch_size == 0
    in_channels: int = 3            # Canales de entrada (1 = grayscale, 3 = RGB)

    # Normalización pixel: media y std por canal (formato lista para serialización JSON)
    pixel_mean: Tuple[float, ...] = field(default_factory=lambda: (0.5, 0.5, 0.5))
    pixel_std:  Tuple[float, ...] = field(default_factory=lambda: (0.5, 0.5, 0.5))

    # ------------------------------------------------------------------
    # 3. Encoder ViT
    # ------------------------------------------------------------------

    d_model:  int   = 768    # Dimensión del embedding (base: 768, large: 1024)
    n_heads:  int   = 12     # Cabezas de atención (d_model // n_heads debe ser entero)
    n_layers: int   = 12     # Número de bloques transformer en el encoder
    d_ff:     int   = 3072   # Dimensión interna del MLP (típicamente 4×d_model)
    dropout:  float = 0.1

    # Codificación posicional del encoder
    # "sinusoidal_2d" : sin/cos factorizado por fila/columna (no aprendida)
    # "learnable"     : embeddings aprendidos por posición (más común en ViT)
    pos_encoding: Literal["sinusoidal_2d", "learnable"] = "learnable"

    # ------------------------------------------------------------------
    # 4. Decoder MAE
    # ------------------------------------------------------------------
    # El decoder es un transformer más ligero que el encoder.
    # Recibe las representaciones latentes del encoder +
    # tokens [MASK] aprendibles en las posiciones enmascaradas.

    decoder_d_model:  int = 512   # Dimensión del decoder (< d_model)
    decoder_n_heads:  int = 16
    decoder_n_layers: int = 8
    decoder_d_ff:     int = 2048

    # Estrategia de enmascaramiento en entrenamiento MAE
    # "random"  : uniforme por parche (He et al., 2022)
    # "block"   : bloques contiguos (más difícil, mejor para datos estructurados)
    # "grid"    : cada N parches (más informativo para tráfico periódico)
    mask_strategy: Literal["random", "block", "grid"] = "random"
    mask_ratio:    float = 0.75   # Fracción de parches enmascarados en entrenamiento

    # Pérdida de reconstrucción
    # "mse"     : MSE pixel-level normalizado
    # "mae"     : L1 pixel-level
    # "smooth_l1": compromiso entre MSE y MAE
    recon_loss: Literal["mse", "mae", "smooth_l1"] = "mse"

    # Si True, normalizar parche target a media 0 / std 1 antes de calcular la pérdida
    # (sigue la práctica de MAE original; mejora las representaciones aprendidas)
    normalize_target: bool = True

    # ------------------------------------------------------------------
    # 5. Condicionalidad
    # ------------------------------------------------------------------

    num_classes: int   = 2     # 0 = modelo no condicional
    cond_dim:    int   = 64    # Dimensión del embedding de clase

    # Modalidad de inyección del condicionamiento:
    # "prefix"   : prepend un token de clase antes de los parches
    # "add"      : suma el embedding de clase a todos los tokens del encoder
    cond_mode: Literal["prefix", "add"] = "prefix"

    # ------------------------------------------------------------------
    # 6. Entrenamiento y regularización
    # ------------------------------------------------------------------

    # Peso del término de pérdida de reconstrucción vs pérdida auxiliar
    recon_weight: float = 1.0

    # Drop-path (stochastic depth) por capa del encoder
    # 0.0 = desactivado; típico: 0.1–0.2 para modelos grandes
    drop_path_rate: float = 0.1

    # EMA del encoder para features estables (usado en generate/infer)
    ema_decay: float = 0.999   # 0.0 = desactivado

    # ------------------------------------------------------------------
    # 7. Generación / muestreo iterativo
    # ------------------------------------------------------------------

    # Pasos de refinamiento iterativo en generate()
    # Sigue el esquema MAGE (Li et al., 2023): empezar con todos los
    # parches enmascarados y ir revelando los de mayor confianza.
    generation_steps: int   = 8
    temperature:      float = 1.0   # Temperatura de muestreo de parches
    top_k:            int   = 0     # 0 = desactivado
    top_p:            float = 1.0   # 1.0 = desactivado

    # ------------------------------------------------------------------
    # Propiedades derivadas (no configurables por el usuario)
    # ------------------------------------------------------------------

    @property
    def num_patches(self) -> int:
        """Número total de parches por imagen."""
        grid = self.image_size // self.patch_size
        return grid * grid

    @property
    def patch_dim(self) -> int:
        """Dimensión aplanada de un parche: C × P × P."""
        return self.in_channels * self.patch_size * self.patch_size

    @property
    def grid_size(self) -> int:
        """Número de parches por lado de la cuadrícula."""
        return self.image_size // self.patch_size

    def __post_init__(self) -> None:
        super().__post_init__() if hasattr(super(), "__post_init__") else None
        assert self.image_size % self.patch_size == 0, (
            f"image_size ({self.image_size}) debe ser divisible por "
            f"patch_size ({self.patch_size})."
        )
        assert self.d_model % self.n_heads == 0, (
            f"d_model ({self.d_model}) debe ser divisible por n_heads ({self.n_heads})."
        )
        assert self.decoder_d_model % self.decoder_n_heads == 0, (
            f"decoder_d_model ({self.decoder_d_model}) debe ser divisible "
            f"por decoder_n_heads ({self.decoder_n_heads})."
        )
        assert 0.0 < self.mask_ratio < 1.0, "mask_ratio debe estar en (0, 1)."