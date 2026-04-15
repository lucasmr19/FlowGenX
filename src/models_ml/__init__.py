"""
models_ml/__init__.py
==============================
REGISTRY unificado de modelos generativos.

Permite instanciar cualquier modelo por nombre desde configs YAML.
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

REGISTRY_MODELS = {
    "transformer": TrafficTransformer,
    "ddpm":        TrafficDDPM,
    "gan":         TrafficGAN,
}

CONFIG_REGISTRY_MODELS = {
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
    if name not in REGISTRY_MODELS:
        raise ValueError(
            f"Modelo desconocido: '{name}'. "
            f"Disponibles: {list(REGISTRY_MODELS.keys())}"
        )
    return REGISTRY_MODELS[name](config)


def get_model_config(name: str, **kwargs) -> GenerativeModelConfig:
    """Instancia la config de un modelo con overrides opcionales."""
    if name not in CONFIG_REGISTRY_MODELS:
        raise ValueError(f"Config de modelo desconocida: '{name}'")
    return CONFIG_REGISTRY_MODELS[name](**kwargs)


__all__ = [
    "GenerativeModel", "GenerativeModelConfig", "ModelType", "InputDomain",
    "TrafficTransformer", "TransformerConfig",
    "TrafficDDPM", "DiffusionConfig", "UNet2D",
    "TrafficGAN", "GANConfig",
    "REGISTRY_MODELS", "CONFIG_REGISTRY_MODELS",
    "get_model", "get_model_config",
]