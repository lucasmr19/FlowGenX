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
    SequentialConfig,
    ProtocolAwareConfig,
)
from .vision import (
    GAFRepresentation,
    GAFConfig,
    NprintRepresentation,
    NprintConfig,
    NprintImageConfig,
    NprintImageRepresentation
)

REGISTRY: dict = {
    "flat_tokenizer":           FlatTokenizer,
    "protocol_aware_tokenizer": ProtocolAwareTokenizer,
    "gaf":                      GAFRepresentation,
    "nprint":                   NprintRepresentation,
    "nprint_image":             NprintImageRepresentation,
}

def get_representation(name: str, config=None):
    """Factory: instancia una representación por nombre."""
    if name not in REGISTRY:
        raise ValueError(f"Representación desconocida: '{name}'. "
                         f"Disponibles: {list(REGISTRY.keys())}")
    cls = REGISTRY[name]
    return cls(config)