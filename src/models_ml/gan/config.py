from __future__ import annotations

from dataclasses import dataclass

from ..base import GenerativeModelConfig


# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

@dataclass
class GANConfig(GenerativeModelConfig):
    """Hiperparámetros de la GAN secuencial."""
    name: str = "traffic_gan"

    # Vocabulario (debe coincidir con el tokenizador)
    vocab_size:   int = 10_000
    pad_token_id: int = 0
    seq_len:      int = 128       # longitud de secuencia fija

    # Espacio latente
    latent_dim:   int = 128

    # Generador (LSTM)
    gen_hidden:   int = 512
    gen_layers:   int = 2
    gen_dropout:  float = 0.1

    # Discriminador (Transformer encoder)
    disc_d_model: int = 256
    disc_n_heads: int = 8
    disc_n_layers: int = 4
    disc_d_ff:    int = 512
    disc_dropout: float = 0.1

    # Entrenamiento WGAN-GP
    n_critic:         int   = 5      # pasos del discriminador por paso del generador
    lambda_gp:        float = 10.0   # peso del gradient penalty
    clip_value:       float = 0.01   # solo si use_gradient_penalty=False
    use_gradient_penalty: bool = True