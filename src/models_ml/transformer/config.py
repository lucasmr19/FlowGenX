from __future__ import annotations

from dataclasses import dataclass
import torch

from ..base import GenerativeModelConfig

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

@dataclass
class TransformerConfig(GenerativeModelConfig):
    """Hiperparámetros del Transformer autoregresivo."""
    model_type: str = "transformer"
    name: str = "traffic_transformer"

    # Vocabulario (debe coincidir con el del tokenizador usado)
    vocab_size:  int = 10_000
    pad_token_id: int = 0
    
    # Condicionalidad (solo para modelos condicionales)
    num_classes: int = 0   # 0 = desactivado
    cond_dim: int = 32

    # Arquitectura
    d_model:     int = 256     # dimensión del embedding
    n_heads:     int = 8       # cabezas de atención
    n_layers:    int = 6       # capas del decoder
    d_ff:        int = 1024    # dimensión feed-forward (típicamente 4×d_model)
    max_seq_len: int = 512     # longitud máxima de secuencia
    dropout:     float = 0.1

    # Generación
    temperature: float = 1.2
    top_k:       int   = 0    # 0 = desactivado
    top_p:       float = 1.0   # 1.0 = desactivado (nucleus sampling)
