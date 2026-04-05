"""
generative_models/__init__.py
==============================
REGISTRY unificado de modelos generativos.

Permite instanciar cualquier modelo por nombre desde configs YAML:
  model = get_model("ddpm", config)
"""

from .base import (
    GenerativeModel,
    GenerativeModelConfig,
    ModelType,
    InputDomain,
)
# Transformer
from .transformer.model import TrafficTransformer
from .transformer.config import TransformerConfig

# Diffusion
from .diffusion.ddpm import TrafficDDPM, DiffusionConfig
from .diffusion.unet import UNet2D

# GAN
from .gan.model import TrafficGAN
from .gan.config import GANConfig

REGISTRY = {
    "transformer": TrafficTransformer,
    "ddpm":        TrafficDDPM,
    "gan":         TrafficGAN,
}

CONFIG_REGISTRY = {
    "transformer": TransformerConfig,
    "ddpm":        DiffusionConfig,
    "gan":         GANConfig,
}


def get_model(name: str, config=None) -> GenerativeModel:
    """
    Instancia un modelo por nombre.

    Parameters
    ----------
    name   : "transformer" | "ddpm" | "gan"
    config : instancia de la Config correspondiente (None = defaults)
    """
    if name not in REGISTRY:
        raise ValueError(
            f"Modelo desconocido: '{name}'. "
            f"Disponibles: {list(REGISTRY.keys())}"
        )
    return REGISTRY[name](config)


def get_config(name: str, **kwargs) -> GenerativeModelConfig:
    """Instancia la config de un modelo con overrides opcionales."""
    if name not in CONFIG_REGISTRY:
        raise ValueError(f"Config desconocida: '{name}'")
    return CONFIG_REGISTRY[name](**kwargs)


__all__ = [
    "GenerativeModel", "GenerativeModelConfig", "ModelType", "InputDomain",
    "TrafficTransformer", "TransformerConfig",
    "TrafficDDPM", "DiffusionConfig", "UNet2D",
    "TrafficGAN", "GANConfig",
    "REGISTRY", "CONFIG_REGISTRY",
    "get_model", "get_config",
]