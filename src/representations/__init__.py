"""
representations/
Exporta todas las representaciones disponibles para uso directo.
"""
from .base import (
    TrafficRepresentation,
    RepresentationConfig,
    RepresentationType,
    Invertibility,
)
from .sequential import (
    FlatTokenizer,
    ProtocolAwareTokenizer,
    SemanticByteTokenizer,
    SequentialConfig,
    ProtocolAwareConfig,
    SemanticByteConfig,
    SequenceTrafficEncoder
)
from .vision import (
    GAFRepresentation,
    GAFConfig,
    NprintRepresentation,
    NprintConfig,
    NprintImageConfig,
    NprintImageRepresentation,
    VisionTrafficEncoder
)

REGISTRY_REPRESENTATIONS: dict = {
    "flat_tokenizer":           FlatTokenizer,
    "protocol_aware": ProtocolAwareTokenizer,
    "semantic_byte": SemanticByteTokenizer,
    "gaf":                      GAFRepresentation,
    "nprint":                   NprintRepresentation,
    "nprint_image":             NprintImageRepresentation,
}

CONFIG_REGISTRY_REPRESENTATIONS: dict = {
    "flat_tokenizer": SequentialConfig,
    "protocol_aware": ProtocolAwareConfig,
    "semantic_byte": SemanticByteConfig,
    "gaf": GAFConfig,
    "nprint": NprintConfig,
    "nprint_image": NprintImageConfig,
}

def get_representation(name: str, config=None) -> TrafficRepresentation:
    """
    Instancia una representación por nombre.

    Parameters
    ----------
    name   : "transformer" | "ddpm" | "gan"
    config : instancia de la Config correspondiente (None = defaults)
    """
    if name not in REGISTRY_REPRESENTATIONS:
        raise ValueError(
            f"Representación desconocida: '{name}'. "
            f"Disponibles: {list(REGISTRY_REPRESENTATIONS.keys())}"
        )
    return REGISTRY_REPRESENTATIONS[name](config)


def get_representation_config(name: str, **kwargs) -> RepresentationConfig:
    """Instancia la config de una representación con overrides opcionales."""
    if name not in CONFIG_REGISTRY_REPRESENTATIONS:
        raise ValueError(f"Config de representación desconocida: '{name}'")
    return CONFIG_REGISTRY_REPRESENTATIONS[name](**kwargs)

__all__ = [
    "TrafficRepresentation", "RepresentationConfig", "RepresentationType", "Invertibility",
    "FlatTokenizer", "ProtocolAwareTokenizer", "SequentialConfig", "ProtocolAwareConfig",
    "GAFRepresentation", "GAFConfig", "NprintRepresentation", "NprintConfig",
    "NprintImageRepresentation", "NprintImageConfig",
    "REGISTRY_REPRESENTATIONS", "CONFIG_REGISTRY_REPRESENTATIONS", 
    "SequenceTrafficEncoder", "VisionTrafficEncoder",
]